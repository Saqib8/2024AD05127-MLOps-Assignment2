"""Baseline CNN for cats vs dogs.

Four conv blocks with batch norm, then global average pooling into a small
classifier head. Global pooling instead of a flatten keeps the parameter
count low, which is what you want on a 4 GB laptop GPU.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src import config


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(config.CHANNELS, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: nn.Module, class_names: list[str], destination: Path) -> Path:
    """Save weights together with the metadata the API needs to load them back."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": config.IMAGE_SIZE,
            "architecture": "SimpleCNN",
        },
        destination,
    )
    return destination


def load_model(path: Path, device: str = "cpu") -> tuple[nn.Module, list[str]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", config.CLASS_NAMES)
    model = SimpleCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names
