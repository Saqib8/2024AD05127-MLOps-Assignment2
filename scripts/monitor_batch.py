"""Post deployment model performance tracking (M5.2).

Sends a batch of held out test images to the running service, compares the
answers against the true labels and writes a small report. This is the
"is the deployed model still any good" check, separate from the offline test
metrics that training produced.

    python scripts/monitor_batch.py --base-url http://localhost:8000 --limit 100
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import requests

from src import config


def gather_samples(test_dir: Path, limit: int, seed: int = 13) -> list:
    """Pick a balanced batch of images with their true label from the folder name."""
    samples = []
    for label in config.CLASS_NAMES:
        folder = test_dir / label
        if not folder.exists():
            continue
        images = sorted(folder.glob("*.jpg"))
        random.Random(seed).shuffle(images)
        samples.extend((path, label) for path in images[: limit // 2])

    random.Random(seed).shuffle(samples)
    return samples


def send(base_url: str, path: Path) -> tuple:
    files = {"file": (path.name, path.read_bytes(), "image/jpeg")}
    started = time.perf_counter()
    response = requests.post("{}/predict".format(base_url), files=files, timeout=30)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the deployed model on a batch")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--test-dir", default=str(config.PROCESSED_DIR / "test"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--output", default=str(config.REPORT_DIR / "post_deployment_report.json")
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="exit non zero if live accuracy drops below this",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    samples = gather_samples(Path(args.test_dir), args.limit)

    if not samples:
        print(
            "no test images under {}, run python -m src.data_prep first".format(
                args.test_dir
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    print("sending {} images to {}".format(len(samples), base_url))

    correct = 0
    latencies = []
    confusion = Counter()
    failures = []

    for index, (path, truth) in enumerate(samples, start=1):
        try:
            body, elapsed_ms = send(base_url, path)
        except requests.RequestException as error:
            failures.append({"file": path.name, "error": str(error)})
            continue

        predicted = body["label"]
        latencies.append(elapsed_ms)
        confusion[(truth, predicted)] += 1
        if predicted == truth:
            correct += 1

        if index % 20 == 0:
            print("  {}/{} done".format(index, len(samples)))

    answered = len(latencies)
    if answered == 0:
        print("every request failed, is the service running?", file=sys.stderr)
        sys.exit(1)

    accuracy = correct / answered
    report = {
        "base_url": base_url,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requests_sent": len(samples),
        "requests_answered": answered,
        "requests_failed": len(failures),
        "live_accuracy": round(accuracy, 4),
        "correct": correct,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1),
            "median": round(statistics.median(latencies), 1),
            "p95": round(sorted(latencies)[int(0.95 * (answered - 1))], 1),
            "max": round(max(latencies), 1),
        },
        "confusion": {
            "{} predicted as {}".format(truth, predicted): count
            for (truth, predicted), count in sorted(confusion.items())
        },
        "failures": failures[:10],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + json.dumps(report, indent=2))
    print("\nreport written to {}".format(output))

    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        print(
            "live accuracy {:.4f} is below the {:.4f} threshold".format(
                accuracy, args.min_accuracy
            ),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
