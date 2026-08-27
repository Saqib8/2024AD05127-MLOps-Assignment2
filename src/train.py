"""Train the baseline CNN and log everything to MLflow."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# MLflow 3 refuses a bare local file store unless this is set. Doing it here
# means the training script works without extra shell setup.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn  # noqa: E402

from src import config  # noqa: E402
from src.dataset import build_dataloaders  # noqa: E402
from src.model import SimpleCNN, count_parameters, save_model  # noqa: E402


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, device, optimiser=None):
    """One pass over a loader. Passing an optimiser switches on training."""
    training = optimiser is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    seen = 0

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            if training:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            seen += images.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    probabilities = []
    truths = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())
        truths.append(targets.numpy())
    return np.concatenate(probabilities), np.concatenate(truths)


def score(probabilities: np.ndarray, truths: np.ndarray) -> dict:
    """Standard binary classification metrics, dog treated as the positive class."""
    predictions = probabilities.argmax(1)
    return {
        "accuracy": float(accuracy_score(truths, predictions)),
        "precision": float(precision_score(truths, predictions, zero_division=0)),
        "recall": float(recall_score(truths, predictions, zero_division=0)),
        "f1": float(f1_score(truths, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truths, probabilities[:, 1])),
    }


def plot_curves(history: dict, destination: Path) -> Path:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="validation")
    axes[0].set_title("Loss per epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="validation")
    axes[1].set_title("Accuracy per epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def plot_confusion(matrix: np.ndarray, class_names: list, destination: Path) -> Path:
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)))
    axis.set_xticklabels(class_names)
    axis.set_yticks(range(len(class_names)))
    axis.set_yticklabels(class_names)
    axis.set_xlabel("predicted")
    axis.set_ylabel("actual")
    axis.set_title("Confusion matrix (test split)")

    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            colour = "white" if matrix[i, j] > threshold else "black"
            axis.text(j, i, int(matrix[i, j]), ha="center", va="center", color=colour)

    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    device = pick_device(args.device)
    loaders, class_names = build_dataloaders(
        Path(args.processed_dir), args.batch_size, args.num_workers
    )

    model = SimpleCNN(num_classes=len(class_names), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    model_path = Path(args.model_dir) / config.MODEL_FILENAME
    started = time.time()

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "architecture": "SimpleCNN",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "image_size": config.IMAGE_SIZE,
                "optimiser": "AdamW",
                "augmentation": "randomresizedcrop+flip+rotate+colorjitter",
                "device": str(device),
                "trainable_parameters": count_parameters(model),
                "train_images": len(loaders["train"].dataset),
                "val_images": len(loaders["val"].dataset),
                "test_images": len(loaders["test"].dataset),
            }
        )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(
                model, loaders["train"], criterion, device, optimiser
            )
            val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": optimiser.param_groups[0]["lr"],
                },
                step=epoch,
            )

            print(
                "epoch {:02d}/{}  train loss {:.4f} acc {:.4f}  "
                "val loss {:.4f} acc {:.4f}".format(
                    epoch, args.epochs, train_loss, train_acc, val_loss, val_acc
                )
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, class_names, model_path)
                print("  saved new best checkpoint to {}".format(model_path))

        # reload the best checkpoint before scoring the held out test split
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

        probabilities, truths = collect_predictions(model, loaders["test"], device)
        metrics = score(probabilities, truths)
        metrics["training_seconds"] = round(time.time() - started, 1)
        mlflow.log_metrics({"test_" + k: v for k, v in metrics.items()})

        report_dir = Path(args.report_dir)
        curves = plot_curves(history, report_dir / "training_curves.png")
        matrix = confusion_matrix(truths, probabilities.argmax(1))
        confusion_png = plot_confusion(
            matrix, class_names, report_dir / "confusion_matrix.png"
        )

        metrics_path = report_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        history_path = report_dir / "history.json"
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(curves), artifact_path="plots")
        mlflow.log_artifact(str(confusion_png), artifact_path="plots")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(model_path), artifact_path="model")

        print("\ntest metrics")
        for key in sorted(metrics):
            print("  {}: {}".format(key, metrics[key]))
        print("\nmlflow run id: {}".format(run.info.run_id))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cats vs dogs baseline CNN")
    parser.add_argument("--processed-dir", default=str(config.PROCESSED_DIR))
    parser.add_argument("--model-dir", default=str(config.MODEL_DIR))
    parser.add_argument("--report-dir", default=str(config.REPORT_DIR))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
