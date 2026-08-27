"""Build the submission zip.

Packages everything the brief asks for and nothing it does not. The dataset and
the DVC cache are deliberately left out, which is the whole point of tracking
them with DVC: the pointer files and the remote config travel instead, and
`dvc pull` restores the images.

    python scripts/make_submission.py
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLL = "2024AD05127"
NAME = "{}_MLOps_Assignment2".format(ROLL)

# whole directories that go in
INCLUDE_DIRS = [
    "src",
    "api",
    "tests",
    "scripts",
    "notebooks",
    "k8s",
    "monitoring",
    "samples",
    "reports",
    "models",
    "mlruns",
    ".github",
]

# individual files that go in
INCLUDE_FILES = [
    "README.md",
    "REPORT.md",
    "Dockerfile",
    ".dockerignore",
    "docker-compose.yml",
    "requirements.txt",
    "requirements-api.txt",
    "pytest.ini",
    "conftest.py",
    "Makefile",
    ".gitignore",
    ".gitattributes",
    ".dvcignore",
    "dvc.yaml",
    "dvc.lock",
    ".dvc/config",
    "data/raw.dvc",
    "data/.gitignore",
]

# anything matching these is dropped wherever it appears
EXCLUDE_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".git",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def wanted(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return True


def collect() -> list:
    found = []

    for name in INCLUDE_DIRS:
        directory = ROOT / name
        if not directory.exists():
            print("  skipping missing directory:", name)
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and wanted(path):
                found.append(path)

    for name in INCLUDE_FILES:
        path = ROOT / name
        if path.is_file():
            found.append(path)
        else:
            print("  skipping missing file:", name)

    return found


def repo_note() -> str:
    return (
        "MLOps (S1-25_AIMLCZG523) Assignment 2\n"
        "Name: Saqib\n"
        "Roll number: {}\n\n"
        "GitHub repository:\n"
        "  https://github.com/Saqib8/2024AD05127-MLOps-Assignment2\n\n"
        "Container image (GitHub Container Registry):\n"
        "  ghcr.io/saqib8/2024ad05127-mlops-assignment2\n\n"
        "CI and CD runs are under the Actions tab of the repository above.\n\n"
        "The dataset is not in this zip. It is versioned with DVC, so the\n"
        "pointer files travel instead of the images:\n"
        "  data/raw.dvc          points at the 25000 raw images\n"
        "  dvc.lock              records the processed 224x224 split\n"
        "  .dvc/config           names the remote\n\n"
        "Restore it with:\n"
        "  kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset \\\n"
        "      -p data/raw --unzip\n"
        "  python -m src.data_prep\n\n"
        "Start here: REPORT.md maps every requirement in the brief to the file\n"
        "that satisfies it. README.md is the usage guide.\n".format(ROLL)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the submission zip")
    parser.add_argument("--output", default=str(ROOT / "{}.zip".format(NAME)))
    args = parser.parse_args()

    destination = Path(args.output)
    files = collect()

    print("packaging {} files".format(len(files)))

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, "{}/{}".format(NAME, path.relative_to(ROOT).as_posix()))
        archive.writestr("{}/GITHUB_REPO.txt".format(NAME), repo_note())

    size_mb = destination.stat().st_size / (1024 * 1024)
    print("wrote {} ({:.1f} MB)".format(destination, size_mb))


if __name__ == "__main__":
    main()
