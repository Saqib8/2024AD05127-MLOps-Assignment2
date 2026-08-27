"""Inference helpers shared by the API, the smoke test and the unit tests.

Keeping this separate from api/main.py means the prediction logic can be
tested without spinning up a web server.
"""
from __future__ import annotations

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src import config
from src.model import load_model


def build_inference_transform(size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """Same preprocessing the validation split used during training.

    If this ever drifts from src.dataset.build_transforms(train=False) the
    served model will quietly get worse, so the two are deliberately identical.
    """
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ]
    )


def bytes_to_image(payload: bytes) -> Image.Image:
    """Decode uploaded bytes into an RGB PIL image.

    Raises ValueError on anything that is not a readable image so the API can
    turn it into a 400 rather than a 500.
    """
    if not payload:
        raise ValueError("empty upload")
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except Exception as error:
        raise ValueError("file is not a readable image") from error
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


class Predictor:
    """Loads the checkpoint once and answers prediction calls."""

    def __init__(self, model_path: Path | str | None = None, device: str = "cpu") -> None:
        self.model_path = Path(model_path or config.MODEL_DIR / config.MODEL_FILENAME)
        if not self.model_path.exists():
            raise FileNotFoundError(
                "no model at {}, train one first with python -m src.train".format(
                    self.model_path
                )
            )
        self.device = device
        self.model, self.class_names = load_model(self.model_path, device)
        self.transform = build_inference_transform()

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> dict:
        """Return the predicted label plus a probability for every class."""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).tolist()

        best = int(max(range(len(probabilities)), key=probabilities.__getitem__))
        return {
            "label": self.class_names[best],
            "confidence": round(probabilities[best], 4),
            "probabilities": {
                name: round(value, 4)
                for name, value in zip(self.class_names, probabilities)
            },
        }

    def predict_bytes(self, payload: bytes) -> dict:
        return self.predict_image(bytes_to_image(payload))
