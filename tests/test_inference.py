"""Unit tests for the model utilities and the inference path.

These build a small untrained model on the fly, so they run in CI without
needing the dataset or a trained checkpoint.
"""
import io

import pytest
import torch
from PIL import Image

from src import config
from src.model import SimpleCNN, count_parameters, load_model, save_model
from src.predict import Predictor, build_inference_transform, bytes_to_image


@pytest.fixture
def checkpoint(tmp_path):
    """A saved SimpleCNN with random weights."""
    model = SimpleCNN(num_classes=2)
    path = tmp_path / "test_model.pt"
    save_model(model, config.CLASS_NAMES, path)
    return path


def image_bytes(size=(80, 60), mode="RGB", fmt="JPEG") -> bytes:
    buffer = io.BytesIO()
    colour = (200, 150, 100) if mode == "RGB" else 128
    Image.new(mode, size, colour).save(buffer, format=fmt)
    return buffer.getvalue()


class TestModel:
    def test_forward_returns_one_logit_per_class(self):
        model = SimpleCNN(num_classes=2).eval()
        batch = torch.randn(4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        with torch.no_grad():
            output = model(batch)
        assert output.shape == (4, 2)

    def test_softmax_over_the_output_sums_to_one(self):
        model = SimpleCNN(num_classes=2).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
        probabilities = torch.softmax(output, dim=1)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_model_has_trainable_parameters(self):
        assert count_parameters(SimpleCNN()) > 0

    def test_save_then_load_gives_identical_weights(self, checkpoint):
        loaded, class_names = load_model(checkpoint)
        assert class_names == config.CLASS_NAMES

        original = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key, value in original["state_dict"].items():
            assert torch.equal(value, loaded.state_dict()[key])

    def test_loaded_model_is_in_eval_mode(self, checkpoint):
        loaded, _ = load_model(checkpoint)
        assert loaded.training is False


class TestByteDecoding:
    def test_decodes_a_valid_jpeg(self):
        image = bytes_to_image(image_bytes())
        assert image.mode == "RGB"
        assert image.size == (80, 60)

    def test_converts_greyscale_upload_to_rgb(self):
        assert bytes_to_image(image_bytes(mode="L", fmt="PNG")).mode == "RGB"

    def test_rejects_empty_payload(self):
        with pytest.raises(ValueError):
            bytes_to_image(b"")

    def test_rejects_a_non_image_payload(self):
        with pytest.raises(ValueError):
            bytes_to_image(b"this is clearly not a picture")


class TestInferenceTransform:
    def test_produces_the_tensor_shape_the_model_expects(self):
        tensor = build_inference_transform()(Image.new("RGB", (640, 128)))
        assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    def test_normalisation_moves_values_off_the_zero_one_range(self):
        tensor = build_inference_transform()(Image.new("RGB", (50, 50), (0, 0, 0)))
        # a pure black image normalises to a negative value, not 0
        assert tensor.min() < 0


class TestPredictor:
    def test_predict_image_returns_a_well_formed_result(self, checkpoint):
        predictor = Predictor(checkpoint)
        result = predictor.predict_image(Image.new("RGB", (300, 200), (10, 200, 90)))

        assert result["label"] in config.CLASS_NAMES
        assert 0.0 <= result["confidence"] <= 1.0
        assert set(result["probabilities"]) == set(config.CLASS_NAMES)
        assert result["probabilities"][result["label"]] == result["confidence"]
        assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)

    def test_predict_bytes_matches_predict_image(self, checkpoint):
        predictor = Predictor(checkpoint)
        payload = image_bytes(size=(120, 120))
        assert predictor.predict_bytes(payload) == predictor.predict_image(
            bytes_to_image(payload)
        )

    def test_prediction_is_deterministic(self, checkpoint):
        predictor = Predictor(checkpoint)
        image = Image.new("RGB", (224, 224), (33, 66, 99))
        assert predictor.predict_image(image) == predictor.predict_image(image)

    def test_missing_checkpoint_raises_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Predictor(tmp_path / "does_not_exist.pt")
