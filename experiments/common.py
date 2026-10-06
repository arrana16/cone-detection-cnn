"""Shared data, training, and evaluation code for the 32x32 experiments."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from cone_classifier import IMAGENET_MEAN, IMAGENET_STD, crop_from_box


CLASS_NAMES = ("blue", "yellow", "unknown")
INPUT_SIZE = 32
MANIFEST_DEFAULT = Path("data/manifests/amz32.csv")
DATA_ROOT_DEFAULT = Path("data/fsoco_bounding_boxes_train")


def preprocess_crop(crop: Image.Image) -> torch.Tensor:
    """Resize a square RGB crop to normalized CHW float data."""

    rgb = crop.convert("RGB")
    side = max(rgb.size)
    padded = Image.new(
        "RGB",
        (side, side),
        tuple(round(channel * 255) for channel in IMAGENET_MEAN),
    )
    padded.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
    resized = padded.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class ConeCropDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load manifest rows and cache one decoded source image at a time."""

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
        return preprocess_crop(crop), CLASS_NAMES.index(row["target_label"])


class ImageGroupedShuffleSampler(Sampler[int]):
    """Shuffle image order while keeping each image's cone rows adjacent."""

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


def read_split(
    manifest_path: Path,
    split: str,
    max_per_class: int | None,
    seed: int,
) -> list[dict[str, str]]:
    """Read one manifest split, optionally limiting each class reproducibly."""

    by_class: dict[str, list[dict[str, str]]] = {name: [] for name in CLASS_NAMES}
    required = {
        "split",
        "target_label",
        "image_path",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    }
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
        for row in reader:
            if row["split"] != split or not row["target_label"]:
                continue
            if row["target_label"] not in by_class:
                raise ValueError(f"unsupported target label {row['target_label']!r}")
            by_class[row["target_label"]].append(dict(row))

    rng = random.Random(seed + (1 if split == "val" else 2 if split == "test" else 0))
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


def choose_device(name: str) -> torch.device:
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


def _metrics(confusion: torch.Tensor) -> dict[str, Any]:
    values: dict[str, dict[str, float | int]] = {}
    f1s: list[float] = []
    total = max(int(confusion.sum()), 1)
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1s.append(f1)
    return {
        "accuracy": float(confusion.diag().sum()) / total,
        "macro_f1": sum(f1s) / len(f1s),
        "per_class": values,
        "confusion_matrix": confusion.to(torch.int64).tolist(),
    }


def _make_loader(
    rows: list[dict[str, str]],
    data_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    *,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    return DataLoader(
        ConeCropDataset(rows, data_root),
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    phase: str,
    epoch: int,
    epochs: int,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    if training and hasattr(model, "set_frozen_backbone_eval"):
        model.set_frozen_backbone_eval()
    loss_total = 0.0
    count = 0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    started = time.monotonic()
    last_percent = -10
    last_update = started
    interactive = sys.stderr.isatty()
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
            count += batch_size
            predicted = logits.detach().argmax(dim=1).cpu()
            labels_cpu = labels.detach().cpu()
            bins = torch.bincount(
                labels_cpu * len(CLASS_NAMES) + predicted,
                minlength=len(CLASS_NAMES) ** 2,
            )
            confusion += bins.reshape(len(CLASS_NAMES), len(CLASS_NAMES))
            percent = int(100 * batch_number / max(len(loader), 1))
            now = time.monotonic()
            should_update = (
                (interactive and now - last_update >= 0.5)
                or (not interactive and percent >= last_percent + 10)
                or batch_number == len(loader)
            )
            if should_update:
                elapsed = time.monotonic() - started
                eta = elapsed / batch_number * (len(loader) - batch_number)
                avg_loss = loss_total / count
                accuracy = float(confusion.diag().sum()) / count
                line = (
                    f"Epoch {epoch}/{epochs} | {phase} | {percent:3d}% "
                    f"{batch_number:,}/{len(loader):,} batches | "
                    f"loss {avg_loss:.4f} | acc {accuracy:.1%} | "
                    f"ETA {eta / 60:.1f} min"
                )
                if interactive:
                    sys.stderr.write("\r\033[2K" + line)
                    if batch_number == len(loader):
                        sys.stderr.write("\n")
                    sys.stderr.flush()
                else:
                    print(line, file=sys.stderr, flush=True)
                last_percent = percent
                last_update = now
    if not count:
        raise ValueError("data loader produced no examples")
    return loss_total / count, _metrics(confusion)


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    fields = (
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_per_class",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({**row, "val_per_class": json.dumps(row["val_per_class"])})


def train_model(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    model_name: str,
    model_settings: dict[str, Any],
) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"run directory is not empty: {args.output_dir}")
    source_checkpoint = None
    start_epoch = 0
    best_f1 = -1.0
    if args.checkpoint is not None:
        source_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if source_checkpoint.get("model_architecture") != model_name:
            raise ValueError("checkpoint architecture does not match the selected model")
        if tuple(source_checkpoint.get("class_names", ())) != CLASS_NAMES:
            raise ValueError("checkpoint class names do not match this experiment")
        if source_checkpoint.get("input_size") != INPUT_SIZE:
            raise ValueError("checkpoint input size does not match this experiment")
        source_config = source_checkpoint.get("config", {})
        for key in ("manifest", "data_root"):
            if Path(source_config.get(key, "")).resolve() != getattr(args, key).resolve():
                raise ValueError(f"checkpoint {key} does not match the requested data")
        for key in ("seed", "max_train_per_class", "max_val_per_class"):
            if source_config.get(key) != getattr(args, key):
                raise ValueError(f"checkpoint {key} does not match the requested data")
        start_epoch = source_checkpoint.get("epoch", 0)
        if not isinstance(start_epoch, int) or start_epoch < 1:
            raise ValueError("checkpoint has no valid epoch")
        best_f1 = source_checkpoint.get("validation_metrics", {}).get("macro_f1")
        if not isinstance(best_f1, (int, float)):
            raise ValueError("checkpoint has no validation macro F1")
        model.load_state_dict(source_checkpoint["model_state_dict"])
    train_rows = read_split(args.manifest, "train", args.max_train_per_class, args.seed)
    val_rows = read_split(args.manifest, "val", args.max_val_per_class, args.seed)
    train_counts = Counter(row["target_label"] for row in train_rows)
    class_weights = torch.tensor(
        [len(train_rows) / (len(CLASS_NAMES) * train_counts[name]) for name in CLASS_NAMES],
        dtype=torch.float32,
    )

    device = choose_device(args.device)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    sampler = ImageGroupedShuffleSampler(train_rows, args.seed)
    train_loader = _make_loader(
        train_rows, args.data_root, device, args.batch_size, args.num_workers, sampler=sampler
    )
    val_loader = _make_loader(
        val_rows, args.data_root, device, args.batch_size, args.num_workers
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_architecture": model_name,
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "start_epoch": start_epoch,
        "end_epoch": start_epoch + args.epochs,
        "source_checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "optimizer_state_restored": False,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "num_workers": args.num_workers,
        "max_train_per_class": args.max_train_per_class,
        "max_val_per_class": args.max_val_per_class,
        "train_counts": dict(train_counts),
        "val_counts": dict(Counter(row["target_label"] for row in val_rows)),
        "class_names": CLASS_NAMES,
        "input_size": INPUT_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "crop_context_fraction": 0.15,
        **model_settings,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    best_epoch = start_epoch
    if source_checkpoint is not None:
        initial_checkpoint = {**source_checkpoint, "config": config}
        torch.save(initial_checkpoint, args.output_dir / "best_model.pt")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch + 1, end_epoch + 1):
        sampler.set_epoch(epoch - 1)
        train_loss, train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            phase="train",
            epoch=epoch,
            epochs=end_epoch,
        )
        val_loss, val_metrics = _run_epoch(
            model,
            val_loader,
            device,
            criterion,
            None,
            phase="val",
            epoch=epoch,
            epochs=end_epoch,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_per_class": val_metrics["per_class"],
        }
        history.append(record)
        _write_history(args.output_dir / "history.csv", history)
        print(json.dumps(record), flush=True)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            checkpoint = {
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "model_architecture": model_name,
                "class_names": CLASS_NAMES,
                "input_size": INPUT_SIZE,
                "epoch": epoch,
                "validation_metrics": val_metrics,
                "config": config,
            }
            temp = args.output_dir / "best_model.pt.tmp"
            torch.save(checkpoint, temp)
            temp.replace(args.output_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    result = {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "checkpoint": str(args.output_dir / "best_model.pt"),
        "history": str(args.output_dir / "history.csv"),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def evaluate_checkpoint(
    args: argparse.Namespace,
    *,
    build_model: Callable[[], nn.Module],
    model_name: str,
) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_architecture") != model_name:
        raise ValueError(
            f"checkpoint architecture {checkpoint.get('model_architecture')!r} "
            f"does not match {model_name!r}"
        )
    if tuple(checkpoint.get("class_names", ())) != CLASS_NAMES:
        raise ValueError("checkpoint class names do not match this experiment")
    config = checkpoint.get("config", {})
    manifest = args.manifest or Path(config.get("manifest", MANIFEST_DEFAULT))
    data_root = args.data_root or Path(config.get("data_root", DATA_ROOT_DEFAULT))
    rows = read_split(manifest, args.split, args.max_per_class, args.seed)
    device = choose_device(args.device)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    loader = _make_loader(rows, data_root, device, args.batch_size, args.num_workers)

    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    predictions: list[dict[str, Any]] = []
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=device.type == "cuda"))
            probabilities = logits.softmax(dim=1).cpu()
            predicted = probabilities.argmax(dim=1)
            labels_cpu = labels.cpu()
            bins = torch.bincount(
                labels_cpu * len(CLASS_NAMES) + predicted,
                minlength=len(CLASS_NAMES) ** 2,
            )
            confusion += bins.reshape(len(CLASS_NAMES), len(CLASS_NAMES))
            for local_index in range(len(labels_cpu)):
                row = rows[offset + local_index]
                item: dict[str, Any] = {
                    "image_path": row["image_path"],
                    "object_id": row.get("object_id", ""),
                    "original_class": row.get("original_class", ""),
                    "target_label": CLASS_NAMES[int(labels_cpu[local_index])],
                    "predicted_label": CLASS_NAMES[int(predicted[local_index])],
                }
                for class_index, class_name in enumerate(CLASS_NAMES):
                    item[f"probability_{class_name}"] = float(
                        probabilities[local_index, class_index]
                    )
                predictions.append(item)
            offset += len(labels_cpu)

    metrics = _metrics(confusion)
    result = {
        "model_architecture": model_name,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "example_count": len(rows),
        **metrics,
    }
    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"metrics_{args.split}.json"
    predictions_path = output_dir / f"predictions_{args.split}.csv"
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fields = list(predictions[0]) if predictions else []
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    result["metrics_file"] = str(metrics_path)
    result["predictions_file"] = str(predictions_path)
    print(json.dumps(result, indent=2), flush=True)
    return result


def add_train_arguments(parser: argparse.ArgumentParser, *, learning_rate: float) -> None:
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="epochs to train, or additional epochs when using --checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="start from a saved best_model.pt in a new output directory",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=learning_rate)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-train-per-class", type=int)
    parser.add_argument("--max-val-per-class", type=int)


def add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-per-class", type=int)


def run_cli(
    argv: list[str] | None,
    *,
    model_name: str,
    build_model: Callable[[bool], nn.Module],
    learning_rate: float,
    model_settings: dict[str, Any],
    pretrained_for_training: bool = True,
) -> int:
    parser = argparse.ArgumentParser(description=f"Train and evaluate {model_name}.")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="train and select a checkpoint on validation")
    add_train_arguments(train_parser, learning_rate=learning_rate)
    eval_parser = commands.add_parser("eval", help="evaluate a checkpoint on a chosen split")
    add_eval_arguments(eval_parser)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers cannot be negative")
    if args.command == "train":
        if args.epochs < 1 or args.patience < 1:
            parser.error("epochs and patience must be positive")
        if args.max_train_per_class is not None and args.max_train_per_class < 1:
            parser.error("max-train-per-class must be positive")
        if args.max_val_per_class is not None and args.max_val_per_class < 1:
            parser.error("max-val-per-class must be positive")
        try:
            train_model(
                args,
                model=build_model(pretrained_for_training),
                model_name=model_name,
                model_settings=model_settings,
            )
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        if args.max_per_class is not None and args.max_per_class < 1:
            parser.error("max-per-class must be positive")
        try:
            evaluate_checkpoint(
                args,
                build_model=lambda: build_model(False),
                model_name=model_name,
            )
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    return 0
