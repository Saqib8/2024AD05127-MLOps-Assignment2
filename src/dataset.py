"""Torch datasets and loaders built on top of the processed folder layout."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src import config


def build_transforms(train: bool, size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """Augment the training split, leave val and test alone.

    Horizontal flips and small rotations are safe here because a mirrored cat
    is still a cat. No vertical flip, upside down pets are not something the
    model will ever be asked about.
    """
    normalise = transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalise,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalise,
        ]
    )


def build_dataloaders(
    processed_dir: Path = config.PROCESSED_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader], list[str]]:
    """One loader per split. Returns the loaders and the class name ordering."""
    processed_dir = Path(processed_dir)
    loaders: dict[str, DataLoader] = {}
    class_names: list[str] = []

    for split in ("train", "val", "test"):
        split_dir = processed_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"{split_dir} is missing, run python -m src.data_prep first"
            )
        dataset = ImageFolder(split_dir, transform=build_transforms(split == "train"))
        if not class_names:
            class_names = dataset.classes
        elif dataset.classes != class_names:
            raise RuntimeError("class ordering differs between splits")

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders, class_names
