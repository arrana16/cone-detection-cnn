"""Create a deterministic per-cone CSV manifest from FSOCO annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


CLASS_TO_LABEL = {
    "blue_cone": "blue",
    "yellow_cone": "yellow",
    "orange_cone": "other",
    "large_orange_cone": "other",
    "unknown_cone": "",
}
LABEL_SCHEMES = {
    "legacy": CLASS_TO_LABEL,
    "amz32": {
        "blue_cone": "blue",
        "yellow_cone": "yellow",
        "orange_cone": "unknown",
        "large_orange_cone": "unknown",
        "unknown_cone": "unknown",
    },
}
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
TINY_SIDE_PIXELS = 8
FIELDNAMES = (
    "image_path",
    "annotation_path",
    "contributor",
    "image_id",
    "object_id",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "image_width",
    "image_height",
    "box_width",
    "box_height",
    "original_class",
    "target_label",
    "split",
    "image_tags",
    "object_tags",
    "issue_flag",
    "tiny_flag",
)


def _tag_names(tags: Any) -> list[str]:
    if not isinstance(tags, list):
        return []
    names = []
    for tag in tags:
        if isinstance(tag, dict) and isinstance(tag.get("name"), str):
            names.append(tag["name"])
    return names


def _is_issue_tag(name: str) -> bool:
    return name == "issue" or name.startswith("issue:")


def _read_dataset(
    root: Path, class_to_label: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    annotation_files = sorted(root.glob("*/ann/*.json"))
    if not annotation_files:
        raise ValueError(f"no annotation files found under {root}/<contributor>/ann")

    rows: list[dict[str, Any]] = []
    image_counts: Counter[str] = Counter()
    for annotation_path in annotation_files:
        contributor = annotation_path.parent.parent.name
        image_id = annotation_path.name[:-5]  # remove the final '.json'
        image_path = annotation_path.parent.parent / "img" / image_id
        if not image_path.is_file():
            raise FileNotFoundError(f"image for annotation is missing: {image_path}")

        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read annotation {annotation_path}: {exc}") from exc

        with Image.open(image_path) as image:
            image_width, image_height = image.size
        annotation_size = annotation.get("size") or {}
        expected_size = (annotation_size.get("width"), annotation_size.get("height"))
        if expected_size != (image_width, image_height):
            raise ValueError(
                f"image/annotation dimensions differ for {image_path}: "
                f"image={(image_width, image_height)}, annotation={expected_size}"
            )

        image_tags = _tag_names(annotation.get("tags"))
        objects = annotation.get("objects")
        if not isinstance(objects, list):
            raise ValueError(f"annotation has no object list: {annotation_path}")
        image_counts[contributor] += 1

        for object_index, item in enumerate(objects):
            if not isinstance(item, dict):
                raise ValueError(f"invalid object #{object_index} in {annotation_path}")
            class_name = item.get("classTitle")
            if class_name not in class_to_label:
                raise ValueError(
                    f"unsupported class {class_name!r} in {annotation_path}"
                )
            if item.get("geometryType") != "rectangle":
                raise ValueError(
                    f"unsupported geometry {item.get('geometryType')!r} "
                    f"in {annotation_path}"
                )
            points = (item.get("points") or {}).get("exterior")
            if not isinstance(points, list) or len(points) != 2:
                raise ValueError(f"expected two rectangle corners in {annotation_path}")
            try:
                x_values = [float(point[0]) for point in points]
                y_values = [float(point[1]) for point in points]
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(f"invalid box coordinates in {annotation_path}") from exc
            if not all(math.isfinite(value) for value in x_values + y_values):
                raise ValueError(f"non-finite box coordinates in {annotation_path}")
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            if not (
                0 <= x_min < x_max <= image_width
                and 0 <= y_min < y_max <= image_height
            ):
                raise ValueError(
                    f"box outside image bounds in {annotation_path}: "
                    f"{(x_min, y_min, x_max, y_max)} vs {(image_width, image_height)}"
                )

            box_width, box_height = x_max - x_min, y_max - y_min
            object_tags = _tag_names(item.get("tags"))
            rows.append(
                {
                    "image_path": image_path.relative_to(root).as_posix(),
                    "annotation_path": annotation_path.relative_to(root).as_posix(),
                    "contributor": contributor,
                    "image_id": image_id,
                    "object_id": item.get("id", object_index),
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "image_width": image_width,
                    "image_height": image_height,
                    "box_width": box_width,
                    "box_height": box_height,
                    "original_class": class_name,
                    "target_label": class_to_label[class_name],
                    "image_tags": image_tags,
                    "object_tags": object_tags,
                    "issue_flag": any(
                        _is_issue_tag(tag) for tag in image_tags + object_tags
                    ),
                    "tiny_flag": min(box_width, box_height) < TINY_SIDE_PIXELS,
                }
            )

    return rows, dict(image_counts)


def _assign_group_splits(
    image_counts: dict[str, int], seed: int
) -> dict[str, str]:
    """Greedily assign whole contributors to approximate 80/10/10 image ratios."""

    if len(image_counts) < len(SPLIT_FRACTIONS):
        raise ValueError("at least three contributor folders are needed for the split")
    rng = random.Random(seed)
    tie_break = {name: rng.random() for name in image_counts}
    groups = sorted(
        image_counts,
        key=lambda name: (-image_counts[name], tie_break[name], name),
    )
    total_images = sum(image_counts.values())
    targets = {
        split: total_images * fraction
        for split, fraction in SPLIT_FRACTIONS.items()
    }
    current = {split: 0 for split in SPLIT_FRACTIONS}
    group_counts = {split: 0 for split in SPLIT_FRACTIONS}
    assignments: dict[str, str] = {}

    for index, group in enumerate(groups):
        remaining_groups = len(groups) - index
        empty_splits = [split for split, count in group_counts.items() if count == 0]
        candidates = empty_splits if len(empty_splits) >= remaining_groups else list(SPLIT_FRACTIONS)
        size = image_counts[group]

        def score(split: str) -> tuple[float, float, str]:
            projected = dict(current)
            projected[split] += size
            normalized_error = sum(
                ((projected[name] - targets[name]) / targets[name]) ** 2
                for name in SPLIT_FRACTIONS
            )
            return normalized_error, -targets[split] + current[split], split

        chosen = min(candidates, key=score)
        assignments[group] = chosen
        current[chosen] += size
        group_counts[chosen] += 1
    return assignments


def _json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def prepare_manifest(
    root: Path,
    output_path: Path,
    seed: int = 42,
    label_scheme: str = "legacy",
) -> dict[str, Any]:
    """Validate annotations and write the manifest plus a split summary JSON."""

    if label_scheme not in LABEL_SCHEMES:
        raise ValueError(
            f"unsupported label scheme {label_scheme!r}; "
            f"choose from {', '.join(sorted(LABEL_SCHEMES))}"
        )
    rows, image_counts = _read_dataset(root, LABEL_SCHEMES[label_scheme])
    assignments = _assign_group_splits(image_counts, seed)
    for row in rows:
        row["split"] = assignments[row["contributor"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["image_tags"] = _json_list(row["image_tags"])
            encoded["object_tags"] = _json_list(row["object_tags"])
            encoded["issue_flag"] = str(row["issue_flag"]).lower()
            encoded["tiny_flag"] = str(row["tiny_flag"]).lower()
            writer.writerow(encoded)
    temp_path.replace(output_path)

    split_images = Counter()
    split_contributors: dict[str, list[str]] = {name: [] for name in SPLIT_FRACTIONS}
    for contributor, split in assignments.items():
        split_images[split] += image_counts[contributor]
        split_contributors[split].append(contributor)
    target_counts = Counter()
    source_counts = Counter()
    for row in rows:
        source_counts[row["original_class"]] += 1
        if row["target_label"]:
            target_counts[row["target_label"]] += 1
    total_images = sum(image_counts.values())
    summary = {
        "label_scheme": label_scheme,
        "seed": seed,
        "target_split_fractions": SPLIT_FRACTIONS,
        "image_count": total_images,
        "contributor_count": len(image_counts),
        "split_image_counts": dict(split_images),
        "split_image_fractions": {
            split: split_images[split] / total_images for split in SPLIT_FRACTIONS
        },
        "split_contributors": {
            split: sorted(names) for split, names in split_contributors.items()
        },
        "row_count": len(rows),
        "source_class_counts": dict(sorted(source_counts.items())),
        "training_label_counts": dict(sorted(target_counts.items())),
        "issue_tagged_row_count": sum(row["issue_flag"] for row in rows),
        "tiny_box_row_count": sum(row["tiny_flag"] for row in rows),
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_temp_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary_temp_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary_temp_path.replace(summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/fsoco_bounding_boxes_train"),
        help="FSOCO root containing contributor/ann and contributor/img folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifest.csv"),
        help="CSV output path (a matching *_summary.json is also written)",
    )
    parser.add_argument("--seed", type=int, default=42, help="split seed (default: 42)")
    parser.add_argument(
        "--label-scheme",
        choices=tuple(LABEL_SCHEMES),
        default="legacy",
        help="class mapping to write (default: legacy)",
    )
    args = parser.parse_args(argv)
    try:
        summary = prepare_manifest(
            args.data_root, args.output, args.seed, label_scheme=args.label_scheme
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
