"""Tests for the FastAPI service.

A throwaway checkpoint is written before the app is imported so the startup
hook finds a model to load.
"""
import io
import os

import pytest
from PIL import Image

from src import config
from src.model import SimpleCNN, save_model


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    model_path = tmp_path_factory.mktemp("model") / "api_test_model.pt"
    save_model(SimpleCNN(num_classes=2), config.CLASS_NAMES, model_path)
    os.environ["MODEL_PATH"] = str(model_path)

    from fastapi.testclient import TestClient

    import api.main as api_main

    # reimport safety: the module caches MODEL_PATH at import time
    api_main.MODEL_PATH = model_path

    with TestClient(api_main.app) as test_client:
        yield test_client


def upload(size=(200, 200), fmt="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, (140, 90, 60)).save(buffer, format=fmt)
    buffer.seek(0)
    return {"file": ("sample.jpg", buffer, "image/jpeg")}


class TestHealth:
    def test_health_reports_ok_when_the_model_is_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["classes"] == config.CLASS_NAMES

    def test_root_lists_the_endpoints(self, client):
        body = client.get("/").json()
        assert "/predict" in body["endpoints"]
        assert "/health" in body["endpoints"]


class TestPredict:
    def test_returns_a_label_and_probabilities(self, client):
        response = client.post("/predict", files=upload())
        assert response.status_code == 200

        body = response.json()
        assert body["label"] in config.CLASS_NAMES
        assert set(body["probabilities"]) == set(config.CLASS_NAMES)
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["filename"] == "sample.jpg"

    def test_rejects_a_file_that_is_not_an_image(self, client):
        files = {"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
        assert client.post("/predict", files=files).status_code == 400

    def test_requires_a_file(self, client):
        assert client.post("/predict").status_code == 422

    def test_response_carries_a_request_id(self, client):
        response = client.post("/predict", files=upload())
        assert response.headers.get("X-Request-ID")


class TestMonitoring:
    def test_metrics_endpoint_exposes_prometheus_text(self, client):
        client.post("/predict", files=upload())
        body = client.get("/metrics").text
        assert "api_requests_total" in body
        assert "api_request_latency_seconds" in body
        assert "predictions_total" in body

    def test_stats_counts_requests_and_predictions(self, client):
        before = client.get("/stats").json()
        client.post("/predict", files=upload())
        after = client.get("/stats").json()

        assert after["predictions_total"] == before["predictions_total"] + 1
        assert after["requests_total"] > before["requests_total"]
        assert after["average_latency_ms"] >= 0
