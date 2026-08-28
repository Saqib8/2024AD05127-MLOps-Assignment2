"""Capture the UI screenshots that go into the report.

Drives a headless browser over the running services and the public GitHub
pages. Everything it captures has to actually be up, so bring the stack up
first:

    docker compose up -d
    mlflow ui --backend-store-uri file:./mlruns --port 5000

    python scripts/capture_screenshots.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"

BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

# name, url, window size, how long to let the page settle
SHOTS = [
    (
        # the bare experiment url lands on the GenAI traces tab in mlflow 3,
        # /runs is the classic training runs table
        "mlflow_experiment_runs",
        "http://localhost:5000/#/experiments/942246411713194689/runs",
        (1600, 760),
        2000,
    ),
    (
        "mlflow_run_detail",
        "http://localhost:5000/#/experiments/942246411713194689"
        "/runs/9dcca5b4e81645fb9068d53e4789f52d",
        (1600, 900),
        2500,
    ),
    (
        "swagger_ui",
        "http://localhost:8000/docs",
        (1500, 730),
        6000,
    ),
    (
        "grafana_dashboard",
        "http://localhost:3000/d/catdog-api/cats-vs-dogs-inference-service"
        "?orgId=1&from=now-30m&to=now&kiosk",
        (1600, 910),
        8000,
    ),
    (
        "prometheus_targets",
        "http://localhost:9090/targets",
        (1500, 380),
        6000,
    ),
    (
        "github_actions_ci",
        "https://github.com/Saqib8/2024AD05127-MLOps-Assignment2/actions/runs/33065652256",
        (1600, 1000),
        8000,
    ),
    (
        "github_actions_cd",
        "https://github.com/Saqib8/2024AD05127-MLOps-Assignment2/actions/runs/33065889252",
        (1600, 1000),
        8000,
    ),
]


def find_browser() -> Path:
    for path in BROWSERS:
        if path.exists():
            return path
    raise SystemExit("no Chrome or Edge found")


def reachable(url: str) -> bool:
    if not url.startswith("http://localhost"):
        return True
    root = "/".join(url.split("/")[:3])
    try:
        urlopen(root, timeout=4)
        return True
    except URLError:
        return False
    except Exception:
        # a 4xx still means something answered on that port
        return True


def capture(browser: Path, name: str, url: str, size, budget: int) -> bool:
    destination = OUT / "{}.png".format(name)
    command = [
        str(browser),
        # the old headless mode is the one that still honours --screenshot
        "--headless=old",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        "--window-size={},{}".format(*size),
        "--virtual-time-budget={}".format(budget),
        "--screenshot={}".format(destination),
        url,
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        # single page apps that poll on a timer never let virtual time run
        # out, so the browser sits there forever. mlflow does this. halving
        # the budget usually gets under the polling interval.
        return False

    if destination.exists() and destination.stat().st_size > 20000:
        print("  ok      {:<24} {:>7} KB".format(name, destination.stat().st_size // 1024))
        return True

    destination.unlink(missing_ok=True)
    return False


def capture_with_retries(browser: Path, name: str, url: str, size, budget: int) -> bool:
    for attempt_budget in (budget, budget // 2, budget // 4):
        if capture(browser, name, url, size, attempt_budget):
            return True
        print("  retry   {:<24} budget {} did not settle".format(name, attempt_budget))
    print("  FAILED  {:<24} {}".format(name, url))
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture report screenshots")
    parser.add_argument("--only", default=None, help="capture just this one by name")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    browser = find_browser()
    print("using", browser.name)
    print()

    captured = 0
    for name, url, size, budget in SHOTS:
        if args.only and args.only != name:
            continue
        if not reachable(url):
            print("  skipped {:<24} service is not up".format(name))
            continue
        if capture_with_retries(browser, name, url, size, budget):
            captured += 1
        time.sleep(1)

    print()
    print("captured {} screenshots into {}".format(captured, OUT))
    if captured == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
