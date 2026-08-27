"""Post deploy smoke test.

Waits for the health endpoint to come up, then sends the labelled sample
images through /predict and checks both the response shape and the answer.
Exits non zero on any failure so the CD pipeline stops before anyone thinks
the deployment worked.

    python scripts/smoke_test.py --base-url http://localhost:30080
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image

VALID_LABELS = {"cat", "dog"}


def log(message: str) -> None:
    print("[smoke] {}".format(message), flush=True)


def fail(message: str) -> None:
    print("[smoke] FAILED: {}".format(message), file=sys.stderr, flush=True)
    sys.exit(1)


def wait_for_health(base_url: str, timeout: int, interval: float = 3.0) -> dict:
    """Poll /health until it answers 200 or we run out of patience."""
    deadline = time.time() + timeout
    attempt = 0
    last_error = "no attempt made"

    while time.time() < deadline:
        attempt += 1
        try:
            response = requests.get("{}/health".format(base_url), timeout=10)
            if response.status_code == 200:
                body = response.json()
                if body.get("model_loaded") is True:
                    log("health ok after {} attempt(s): {}".format(attempt, body))
                    return body
                last_error = "service is up but the model did not load: {}".format(body)
            else:
                last_error = "status {} from /health".format(response.status_code)
        except requests.RequestException as error:
            last_error = str(error)

        log("attempt {} not ready yet ({})".format(attempt, last_error))
        time.sleep(interval)

    fail("health check never passed within {}s. Last error: {}".format(timeout, last_error))


def expected_label(name: str) -> str | None:
    """Read the true label out of a sample filename, if it has one."""
    lowered = name.lower()
    if "cat" in lowered and "dog" not in lowered:
        return "cat"
    if "dog" in lowered and "cat" not in lowered:
        return "dog"
    return None


def pick_images(explicit: str | None) -> list:
    """Images to send, as (filename, bytes, expected label or None).

    Prefers the labelled samples committed under samples/, because those let
    us check the answer and not just the response shape. A model with its
    class order flipped would still return a valid looking payload, so shape
    alone is not enough. Falls back to a generated image so the test stays
    runnable anywhere.
    """
    if explicit:
        path = Path(explicit)
        if not path.exists():
            fail("image {} does not exist".format(path))
        log("using image {}".format(path))
        return [(path.name, path.read_bytes(), expected_label(path.name))]

    samples = sorted(Path("samples").glob("*.jpg")) if Path("samples").exists() else []
    if samples:
        log("using {} bundled sample(s)".format(len(samples)))
        return [(p.name, p.read_bytes(), expected_label(p.name)) for p in samples]

    log("no sample image available, generating one")
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), (128, 110, 90)).save(buffer, format="JPEG")
    return [("generated.jpg", buffer.getvalue(), None)]


def check_prediction(base_url: str, filename: str, payload: bytes, expected=None) -> dict:
    files = {"file": (filename, payload, "image/jpeg")}
    started = time.perf_counter()
    try:
        response = requests.post("{}/predict".format(base_url), files=files, timeout=30)
    except requests.RequestException as error:
        fail("prediction request raised: {}".format(error))

    elapsed_ms = (time.perf_counter() - started) * 1000

    if response.status_code != 200:
        fail("prediction returned {} with body {}".format(response.status_code, response.text[:400]))

    body = response.json()
    log("prediction took {:.0f} ms: {}".format(elapsed_ms, json.dumps(body)))

    if body.get("label") not in VALID_LABELS:
        fail("unexpected label {}".format(body.get("label")))

    confidence = body.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        fail("confidence {} is not a probability".format(confidence))

    probabilities = body.get("probabilities") or {}
    if set(probabilities) != VALID_LABELS:
        fail("probabilities should cover both classes, got {}".format(probabilities))

    total = sum(probabilities.values())
    if abs(total - 1.0) > 0.01:
        fail("probabilities sum to {} instead of 1".format(total))

    if expected is not None and body["label"] != expected:
        # these samples are held out images the trained model classifies
        # correctly, so a miss here means the wrong weights shipped or the
        # class order got swapped somewhere between training and serving
        fail(
            "{} should be classified as {} but came back {} at {:.4f}".format(
                filename, expected, body["label"], confidence
            )
        )
    if expected is not None:
        log("  label matches the expected {}".format(expected))

    return body


def check_metrics(base_url: str) -> None:
    """Not fatal on its own, but worth reporting if monitoring is broken."""
    try:
        response = requests.get("{}/metrics".format(base_url), timeout=10)
    except requests.RequestException as error:
        fail("metrics endpoint unreachable: {}".format(error))

    if response.status_code != 200:
        fail("metrics returned {}".format(response.status_code))
    if "api_requests_total" not in response.text:
        fail("metrics output is missing api_requests_total")
    log("metrics endpoint is exposing counters")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post deploy smoke test")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=int, default=120, help="seconds to wait for health")
    parser.add_argument("--image", default=None, help="optional image to send to /predict")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    log("target {}".format(base_url))

    wait_for_health(base_url, args.timeout)
    for filename, payload, expected in pick_images(args.image):
        check_prediction(base_url, filename, payload, expected)
    check_metrics(base_url)

    log("all checks passed")


if __name__ == "__main__":
    main()
