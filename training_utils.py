from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras


def merge_histories(
    transfer_history: keras.callbacks.History,
    fine_tune_history: keras.callbacks.History | None,
) -> dict[str, list[float]]:
    merged = {key: list(values) for key, values in transfer_history.history.items()}
    if fine_tune_history is not None:
        for key, values in fine_tune_history.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def save_history_csv(history: dict[str, list[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(history)
    epoch_count = max((len(history[key]) for key in keys), default=0)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *keys])
        for epoch_index in range(epoch_count):
            writer.writerow(
                [epoch_index + 1]
                + [
                    history[key][epoch_index] if epoch_index < len(history[key]) else ""
                    for key in keys
                ]
            )


def save_training_plot(
    history: dict[str, list[float]],
    transfer_epoch_count: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("loss", [])) + 1)

    figure = plt.figure(figsize=(9, 6))
    plt.plot(epochs, history.get("accuracy", []), label="Training accuracy")
    plt.plot(epochs, history.get("val_accuracy", []), label="Validation accuracy")
    if len(history.get("loss", [])) > transfer_epoch_count:
        plt.axvline(transfer_epoch_count + 0.5, linestyle="--", label="Fine-tuning starts")
    plt.title("Plant Disease Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)

    loss_path = output_path.with_name("training_loss.png")
    figure = plt.figure(figsize=(9, 6))
    plt.plot(epochs, history.get("loss", []), label="Training loss")
    plt.plot(epochs, history.get("val_loss", []), label="Validation loss")
    if len(history.get("loss", [])) > transfer_epoch_count:
        plt.axvline(transfer_epoch_count + 0.5, linestyle="--", label="Fine-tuning starts")
    plt.title("Plant Disease Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(loss_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_evaluation_outputs(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    probabilities: np.ndarray,
    image_paths: Sequence[str],
    class_names: Sequence[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    class_ids = list(range(len(class_names)))

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=class_ids,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(true_labels, predicted_labels, labels=class_ids)
    figure_size = max(10, min(30, len(class_names) * 0.65))
    figure = plt.figure(figsize=(figure_size, figure_size))
    axis = figure.add_subplot(111)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[str(index) for index in class_ids],
    )
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title("Test-set Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_path", "true_class_id", "true_class_name",
            "predicted_class_id", "predicted_class_name",
            "predicted_confidence", "correct",
        ])
        for index, image_path in enumerate(image_paths):
            true_id = int(true_labels[index])
            predicted_id = int(predicted_labels[index])
            writer.writerow([
                image_path,
                true_id,
                class_names[true_id],
                predicted_id,
                class_names[predicted_id],
                f"{float(probabilities[index, predicted_id]):.6f}",
                true_id == predicted_id,
            ])
