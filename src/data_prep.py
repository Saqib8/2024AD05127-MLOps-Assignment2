"""Turn the raw Kaggle download into a clean 224x224 dataset with a fixed split.

The Kaggle cats and dogs dataset ships in a couple of different layouts
depending on which mirror you grab. Some have train/cats and train/dogs
folders, others have every file dumped flat as cat.0.jpg / dog.0.jpg. The
functions below read the label out of the path so either layout works.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def label_from_path(path) -> str | None:
    """Work out whether a file is a cat or a dog from its path.

    Checks the file name first, then walks up the parent folders. Returns None
    when neither word shows up, which lets the caller skip junk files.
    """
    path = Path(path)
    parts = [path.stem] + [p.name for p in path.parents]
    for part in parts:
        lowered = part.lower()
        has_cat = "cat" in lowered
        has_dog = "dog" in lowered
        # ignore anything ambiguous such as a folder literally named
        # "cats_and_dogs", otherwise the label would depend on ordering
        if has_cat and not has_dog:
            return "cat"
        if has_dog and not has_cat:
            return "dog"
    return None


def list_images(root) -> list[Path]:
    """Every readable image under root, sorted so runs are reproducible."""
    root = Path(root)
    found = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(found)


def preprocess_image(image: Image.Image, size: int = config.IMAGE_SIZE) -> Image.Image:
    """Force an image to size x size RGB.

    Converting to RGB matters because a handful of files in this dataset are
    greyscale or have an alpha channel, and the model expects three channels.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer")
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize((size, size), Image.BILINEAR)


def stratified_split(
    items: list[tuple[Path, str]],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    seed: int = config.RANDOM_SEED,
) -> dict[str, list[tuple[Path, str]]]:
    """Split per class so train, val and test keep the same cat/dog balance.

    Anything left over after the train and val slices goes to test, so the
    three ratios always add up to the full dataset with no dropped rows.
    """
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1:
        raise ValueError("ratios must sit between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must leave room for a test set")

    by_label: dict[str, list[tuple[Path, str]]] = {}
    for path, label in items:
        by_label.setdefault(label, []).append((path, label))

    rng = random.Random(seed)
    splits: dict[str, list[tuple[Path, str]]] = {"train": [], "val": [], "test": []}

    for label in sorted(by_label):
        group = sorted(by_label[label])
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])

    for split in splits.values():
        rng.shuffle(split)
    return splits


def _write_manifest(rows: list[tuple[str, str, str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filepath", "label", "split"])
        writer.writerows(rows)


def build_dataset(
    raw_dir: Path,
    processed_dir: Path,
    size: int = config.IMAGE_SIZE,
    limit_per_class: int | None = None,
) -> dict[str, int]:
    """Read raw images, resize them and lay them out as processed/<split>/<label>."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)

    labelled: list[tuple[Path, str]] = []
    skipped = 0
    for path in list_images(raw_dir):
        label = label_from_path(path)
        if label is None:
            skipped += 1
            continue
        labelled.append((path, label))

    if not labelled:
        raise RuntimeError(f"no labelled images found under {raw_dir}")

    if limit_per_class is not None:
        capped: list[tuple[Path, str]] = []
        counts: Counter[str] = Counter()
        for path, label in labelled:
            if counts[label] < limit_per_class:
                capped.append((path, label))
                counts[label] += 1
        labelled = capped

    splits = stratified_split(labelled)

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    for split_name in splits:
        for label in config.CLASS_NAMES:
            (processed_dir / split_name / label).mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, str]] = []
    written: Counter[str] = Counter()
    corrupt = 0

    for split_name, entries in splits.items():
        for index, (source, label) in enumerate(entries):
            target = processed_dir / split_name / label / f"{label}_{index:05d}.jpg"
            try:
                with Image.open(source) as image:
                    image.load()
                    preprocess_image(image, size).save(target, format="JPEG", quality=92)
            except (UnidentifiedImageError, OSError):
                # the raw Kaggle archive contains a few truncated files
                corrupt += 1
                continue
            rows.append((str(target.relative_to(processed_dir)), label, split_name))
            written[split_name] += 1

    _write_manifest(rows, processed_dir / "manifest.csv")

    summary = dict(written)
    summary["skipped_unlabelled"] = skipped
    summary["skipped_corrupt"] = corrupt
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the cats vs dogs dataset")
    parser.add_argument("--raw-dir", default=str(config.RAW_DIR))
    parser.add_argument("--processed-dir", default=str(config.PROCESSED_DIR))
    parser.add_argument("--size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="cap images per class, handy for a quick smoke test",
    )
    args = parser.parse_args()

    summary = build_dataset(
        Path(args.raw_dir), Path(args.processed_dir), args.size, args.limit_per_class
    )
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
