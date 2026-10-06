"""Train the cone-colour head of a pretrained EfficientNet V2 S."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from cone_classifier import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    ConeColorClassifier,
    build_model,
    crop_from_box,
    preprocess_crop,
)


class ConeCropDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load labelled manifest rows, caching the most recently used image."""

    def __init__(self, rows: list[dict[str, str]], data_root: Path) -> None:
        self.rows = rows
        self.data_root = data_root
        self._cached_path: Path | None = None
        self._cached_image: Image.Image | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = self.data_root / row["image_path"]
        if self._cached_path != image_path or self._cached_image is None:
            with Image.open(image_path) as source:
                self._cached_image = source.convert("RGB")
            self._cached_path = image_path

        box = tuple(float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
        crop = crop_from_box(self._cached_image, box)
        tensor = preprocess_crop(crop)
        return tensor, CLASS_NAMES.index(row["target_label"])


class ImageGroupedShuffleSampler(Sampler[int]):
    """Shuffle image order while keeping each image's boxes adjacent.

    This allows the dataset's one-image cache to avoid decoding the same source
    image once for every cone annotation.
    """

    def __init__(self, rows: list[dict[str, str]], seed: int) -> None:
        self.seed = seed
        self.epoch = 0
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[row["image_path"]].append(index)
        self.groups = dict(groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.groups.values())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        image_paths = list(self.groups)
        rng.shuffle(image_paths)
        for image_path in image_paths:
            indices = self.groups[image_path][:]
            rng.shuffle(indices)
            yield from indices


def _read_split(
    manifest_path: Path,
    split: str,
    max_per_class: int | None,
    seed: int,
) -> list[dict[str, str]]:
    by_class: dict[str, list[dict[str, str]]] = {name: [] for name in CLASS_NAMES}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "split",
            "target_label",
            "image_path",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
        for row in reader:
            if row["split"] != split or not row["target_label"]:
                continue
            if row["target_label"] not in by_class:
                raise ValueError(f"unsupported target label {row['target_label']!r}")
            by_class[row["target_label"]].append(
                {
                    key: row[key]
                    for key in (
                        "target_label",
                        "image_path",
                        "x_min",
                        "y_min",
                        "x_max",
                        "y_max",
                    )
                }
            )

    rng = random.Random(seed + (1 if split == "val" else 0))
    selected: list[dict[str, str]] = []
    for class_name in CLASS_NAMES:
        rows = by_class[class_name]
        if not rows:
            raise ValueError(f"split {split!r} has no examples for class {class_name!r}")
        rng.shuffle(rows)
        if max_per_class is not None:
            if max_per_class <= 0:
                raise ValueError("per-class limits must be positive")
            rows = rows[:max_per_class]
        selected.extend(rows)
    return selected


def _metrics(
    confusion: torch.Tensor,
) -> dict[str, Any]:
    matrix = confusion.to(dtype=torch.float64)
    total = float(matrix.sum())
    accuracy = float(matrix.diag().sum() / max(total, 1.0))
    recalls: dict[str, float] = {}
    f1_scores: list[float] = []
    for index, name in enumerate(CLASS_NAMES):
        true_positive = float(matrix[index, index])
        support = float(matrix[index].sum())
        predicted = float(matrix[:, index].sum())
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[name] = recall
        f1_scores.append(f1)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class_recall": recalls,
        "confusion_matrix": matrix.to(torch.int64).tolist(),
    }


def _format_duration(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


class EpochProgress:
    """Show live batch metrics on a terminal and sparse updates in logs."""

    def __init__(
        self, phase: str, epoch: int, epochs: int, batch_count: int, crop_count: int
    ) -> None:
        self.phase = phase
        self.batch_count = batch_count
        self.started = time.monotonic()
        self.last_update = self.started
        self.last_percent = -10
        self.interactive = sys.stderr.isatty()
        print(
            f"Epoch {epoch}/{epochs} | {phase} | {crop_count:,} crops in {batch_count:,} batches",
            file=sys.stderr,
            flush=True,
        )

    def update(
        self, batch_number: int, example_count: int, loss_total: float, correct_count: int
    ) -> None:
        now = time.monotonic()
        percent = round(100 * batch_number / self.batch_count)
        complete = batch_number == self.batch_count
        if self.interactive:
            if not complete and now - self.last_update < 0.25:
                return
        elif not complete and percent < self.last_percent + 10:
            return

        elapsed = now - self.started
        eta = elapsed / batch_number * (self.batch_count - batch_number)
        filled = round(12 * batch_number / self.batch_count)
        bar = "#" * filled + "-" * (12 - filled)
        timing = f"time {_format_duration(elapsed)}" if complete else f"ETA {_format_duration(eta)}"
        line = (
            f"  [{bar}] {percent:3d}%  {batch_number:,}/{self.batch_count:,}"
            f"  loss {loss_total / example_count:.4f}"
            f"  acc {correct_count / example_count:.1%}  {timing}"
        )
        if self.interactive:
            print(f"\r\033[2K{line}", end="\n" if complete else "", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)
        self.last_update = now
        self.last_percent = percent


def _run_epoch(
    model: ConeColorClassifier,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    epochs: int,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    if training:
        model.network.features.eval()  # Preserve frozen backbone batch-norm statistics.
    loss_total = 0.0
    example_count = 0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    progress = EpochProgress(
        "train" if training else "val", epoch, epochs, len(loader), len(loader.dataset)
    )

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_number, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            loss_total += float(loss.detach()) * batch_size
            example_count += batch_size
            predicted = logits.detach().argmax(dim=1).to("cpu")
            labels_cpu = labels.detach().to("cpu")
            bins = torch.bincount(
                labels_cpu * len(CLASS_NAMES) + predicted,
                minlength=len(CLASS_NAMES) ** 2,
            )
            confusion += bins.reshape(len(CLASS_NAMES), len(CLASS_NAMES))
            progress.update(batch_number, example_count, loss_total, int(confusion.diag().sum()))

    if example_count == 0:
        raise ValueError("data loader produced no examples")
    return loss_total / example_count, _metrics(confusion)


def _save_history(path: Path, history: list[dict[str, Any]]) -> None:
    fields = (
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_per_class_recall",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in history:
            writer.writerow(
                {
                    **item,
                    "val_per_class_recall": json.dumps(item["val_per_class_recall"]),
                }
            )


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is unavailable")
    return device


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_rows = _read_split(args.manifest, "train", args.max_train_per_class, args.seed)
    val_rows = _read_split(args.manifest, "val", args.max_val_per_class, args.seed)
    train_counts = Counter(row["target_label"] for row in train_rows)
    class_weights = torch.tensor(
        [len(train_rows) / (len(CLASS_NAMES) * train_counts[name]) for name in CLASS_NAMES],
        dtype=torch.float32,
    )

    device = _choose_device(args.device)
    train_dataset = ConeCropDataset(train_rows, args.data_root)
    val_dataset = ConeCropDataset(val_rows, args.data_root)
    train_sampler = ImageGroupedShuffleSampler(train_rows, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(pretrained=True).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    output_layer = model.network.classifier[-1]
    for parameter in output_layer.parameters():
        parameter.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        output_layer.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "max_train_per_class": args.max_train_per_class,
        "max_val_per_class": args.max_val_per_class,
        "train_counts": dict(train_counts),
        "val_counts": dict(Counter(row["target_label"] for row in val_rows)),
        "class_names": CLASS_NAMES,
        "model_architecture": "efficientnet_v2_s",
        "input_size": INPUT_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "pretrained_backbone": True,
        "trainable_parameters": "EfficientNet V2 S final classifier layer only",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch - 1)
        train_loss, train_metrics = _run_epoch(
            model, train_loader, device, criterion, optimizer, epoch=epoch, epochs=args.epochs
        )
        val_loss, val_metrics = _run_epoch(
            model, val_loader, device, criterion, None, epoch=epoch, epochs=args.epochs
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_per_class_recall": val_metrics["per_class_recall"],
        }
        history.append(record)
        _save_history(args.output_dir / "history.csv", history)
        print(json.dumps(record), flush=True)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": {
                    key: value.detach().to("cpu")
                    for key, value in model.state_dict().items()
                },
                "class_names": CLASS_NAMES,
                "input_size": INPUT_SIZE,
                "imagenet_mean": IMAGENET_MEAN,
                "imagenet_std": IMAGENET_STD,
                "epoch": epoch,
                "validation_metrics": val_metrics,
                "config": config,
            }
            temp_checkpoint = args.output_dir / "best_model.pt.tmp"
            torch.save(checkpoint, temp_checkpoint)
            temp_checkpoint.replace(args.output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    result = {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "checkpoint": str(args.output_dir / "best_model.pt"),
        "history": str(args.output_dir / "history.csv"),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/fsoco_bounding_boxes_train")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/efficientnet_v2_s")
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        help="optional balanced training subset limit per class",
    )
    parser.add_argument(
        "--max-val-per-class",
        type=int,
        help="optional validation subset limit per class",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must all be positive")
    if args.num_workers < 0:
        parser.error("num-workers cannot be negative")
    try:
        train(args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
