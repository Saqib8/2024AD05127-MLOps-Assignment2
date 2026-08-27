"""FastAPI service that serves the cats vs dogs classifier.

Endpoints:
    GET  /health   liveness and readiness, reports whether the model loaded
    POST /predict  multipart image upload, returns label and probabilities
    GET  /metrics  Prometheus exposition format
    GET  /stats    the same counters as plain JSON, easier to read in a demo
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter as PromCounter
from prometheus_client import Gauge, Histogram, generate_latest

from src import config
from src.predict import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("catdog.api")

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(config.MODEL_DIR / config.MODEL_FILENAME))
)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))

REQUEST_COUNT = PromCounter(
    "api_requests_total", "Requests handled", ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "End to end request latency",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTION_COUNT = PromCounter(
    "predictions_total", "Predictions returned", ["label"]
)
MODEL_READY = Gauge("model_loaded", "1 when the model is loaded and serving")

# Lightweight in process counters so /stats works even without Prometheus.
_stats_lock = Lock()
_stats = {
    "requests_total": 0,
    "predictions_total": 0,
    "errors_total": 0,
    "latency_sum_seconds": 0.0,
    "labels": Counter(),
    "started_at": time.time(),
}

state: dict = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of on every request."""
    try:
        state["predictor"] = Predictor(MODEL_PATH)
        MODEL_READY.set(1)
        logger.info("model loaded from %s", MODEL_PATH)
    except Exception as error:
        # Do not crash the container. /health reports the problem instead,
        # which is what lets Kubernetes mark the pod unready rather than
        # restart looping forever.
        state["error"] = str(error)
        MODEL_READY.set(0)
        logger.error("could not load model: %s", error)
    yield
    logger.info("shutting down")


app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Baseline CNN served for the pet adoption platform use case",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_and_time(request: Request, call_next):
    """Log every request and record latency.

    Only metadata is logged. The uploaded image itself is never written to the
    log, we record its size instead.
    """
    request_id = str(uuid.uuid4())[:8]
    started = time.perf_counter()

    response = await call_next(request)

    elapsed = time.perf_counter() - started
    endpoint = request.url.path

    REQUEST_COUNT.labels(endpoint, request.method, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(endpoint).observe(elapsed)

    with _stats_lock:
        _stats["requests_total"] += 1
        _stats["latency_sum_seconds"] += elapsed
        if response.status_code >= 400:
            _stats["errors_total"] += 1

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "id=%s method=%s path=%s status=%s latency_ms=%.1f",
        request_id,
        request.method,
        endpoint,
        response.status_code,
        elapsed * 1000,
    )
    return response


@app.get("/")
def root() -> dict:
    return {
        "service": "cats-vs-dogs-classifier",
        "version": app.version,
        "endpoints": ["/health", "/predict", "/metrics", "/stats", "/docs"],
    }


@app.get("/health")
def health() -> JSONResponse:
    """Readiness probe. Returns 503 while the model is not usable."""
    ready = state["predictor"] is not None
    body = {
        "status": "ok" if ready else "unavailable",
        "model_loaded": ready,
        "model_path": str(MODEL_PATH),
        "classes": state["predictor"].class_names if ready else [],
        "uptime_seconds": round(time.time() - _stats["started_at"], 1),
    }
    if not ready:
        body["detail"] = state["error"]
    return JSONResponse(body, status_code=200 if ready else 503)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    predictor = state["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="model is not loaded")

    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="file larger than {} bytes".format(MAX_UPLOAD_BYTES),
        )

    try:
        result = predictor.predict_bytes(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    PREDICTION_COUNT.labels(result["label"]).inc()
    with _stats_lock:
        _stats["predictions_total"] += 1
        _stats["labels"][result["label"]] += 1

    logger.info(
        "prediction filename=%s bytes=%d label=%s confidence=%.4f",
        file.filename,
        len(payload),
        result["label"],
        result["confidence"],
    )

    result["filename"] = file.filename
    return result


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/stats")
def stats() -> dict:
    with _stats_lock:
        total = _stats["requests_total"]
        average_latency = (
            _stats["latency_sum_seconds"] / total if total else 0.0
        )
        return {
            "requests_total": total,
            "predictions_total": _stats["predictions_total"],
            "errors_total": _stats["errors_total"],
            "average_latency_ms": round(average_latency * 1000, 2),
            "predictions_by_label": dict(_stats["labels"]),
            "uptime_seconds": round(time.time() - _stats["started_at"], 1),
        }
