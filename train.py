from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from config import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    DATASET_DIR,
    FINAL_MODEL_PATH,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LAYERS,
    MODEL_DIR,
    MODEL_INFO_PATH,
    OUTPUT_DIR,
    RANDOM_SEED,
    SPLIT_MANIFEST_PATH,
    TRANSFER_EPOCHS,
    TRANSFER_STAGE_MODEL_PATH,
)
from dataset_utils import (
    build_dataset,
    collect_samples,
    find_dataset_root,
    print_dataset_summary,
    save_split_manifest,
    split_samples,
)
from model_utils import build_transfer_learning_model, enable_fine_tuning
from training_utils import (
    merge_histories,
    save_evaluation_outputs,
    save_history_csv,
    save_training_plot,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 for plant-disease classification."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--transfer-epochs", type=int, default=TRANSFER_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    parser.add_argument("--fine-tune-layers", type=int, default=FINE_TUNE_LAYERS)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_hardware() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"TensorFlow GPU devices: {gpus}" if gpus else "No TensorFlow GPU detected. Training will use the CPU.")


def build_callbacks(checkpoint_path: Path, csv_path: Path) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(csv_path, append=False),
    ]


def main() -> None:
    args = parse_arguments()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.transfer_epochs <= 0:
        raise ValueError("--transfer-epochs must be greater than zero.")
    if args.fine_tune_epochs < 0:
        raise ValueError("--fine-tune-epochs cannot be negative.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_random_seeds(RANDOM_SEED)
    configure_hardware()

    dataset_root = find_dataset_root(args.dataset)
    print(f"\nUsing dataset root:\n{dataset_root}")
    image_paths, labels, class_names = collect_samples(dataset_root)
    splits = split_samples(image_paths, labels)
    print_dataset_summary(splits, class_names)

    save_split_manifest(splits, class_names, SPLIT_MANIFEST_PATH)
    CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_paths, train_labels = splits["train"]
    validation_paths, validation_labels = splits["validation"]
    test_paths, test_labels = splits["test"]

    train_dataset = build_dataset(train_paths, train_labels, args.batch_size, True)
    validation_dataset = build_dataset(
        validation_paths, validation_labels, args.batch_size, False
    )
    test_dataset = build_dataset(test_paths, test_labels, args.batch_size, False)

    classes = np.arange(len(class_names))
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=np.asarray(train_labels)
    )
    class_weight = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }

    pretrained_weights = "imagenet" if args.weights == "imagenet" else None
    try:
        model, _ = build_transfer_learning_model(
            len(class_names), weights=pretrained_weights
        )
    except Exception as error:
        if pretrained_weights == "imagenet":
            raise RuntimeError(
                "Could not load MobileNetV2 ImageNet weights. The first run "
                "normally needs internet access. Retry online, or use "
                "python train.py --weights none (not recommended for true "
                f"transfer learning). Original error: {error}"
            ) from error
        raise

    model.summary()
    print("\nStage 1: training classifier head")
    transfer_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.transfer_epochs,
        class_weight=class_weight,
        callbacks=build_callbacks(
            TRANSFER_STAGE_MODEL_PATH,
            OUTPUT_DIR / "transfer_training_log.csv",
        ),
    )

    model = keras.models.load_model(TRANSFER_STAGE_MODEL_PATH)
    _, transfer_validation_accuracy = model.evaluate(validation_dataset, verbose=1)
    print(f"Transfer-stage validation accuracy: {transfer_validation_accuracy * 100:.2f}%")

    fine_tune_history = None
    selected_stage = "transfer"

    if args.fine_tune_epochs > 0:
        base_model = next(
            layer for layer in model.layers
            if isinstance(layer, keras.Model) and "mobilenetv2" in layer.name.lower()
        )
        enable_fine_tuning(model, base_model, args.fine_tune_layers)
        candidate_path = MODEL_DIR / "fine_tuned_candidate.keras"

        print("\nStage 2: fine-tuning MobileNetV2")
        fine_tune_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=args.fine_tune_epochs,
            class_weight=class_weight,
            callbacks=build_callbacks(
                candidate_path,
                OUTPUT_DIR / "fine_tune_training_log.csv",
            ),
        )

        candidate_model = keras.models.load_model(candidate_path)
        _, candidate_accuracy = candidate_model.evaluate(validation_dataset, verbose=1)
        print(f"Fine-tuned validation accuracy: {candidate_accuracy * 100:.2f}%")

        if candidate_accuracy >= transfer_validation_accuracy:
            model = candidate_model
            selected_stage = "fine_tuned"
        else:
            print("Fine-tuning did not improve validation accuracy; keeping stage 1.")
            model = keras.models.load_model(TRANSFER_STAGE_MODEL_PATH)

        if candidate_path.exists():
            candidate_path.unlink()

    model.save(FINAL_MODEL_PATH)
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    probabilities = model.predict(test_dataset, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    save_evaluation_outputs(
        test_labels,
        predictions,
        probabilities,
        test_paths,
        class_names,
        OUTPUT_DIR,
    )

    merged_history = merge_histories(transfer_history, fine_tune_history)
    save_history_csv(merged_history, OUTPUT_DIR / "combined_history.csv")
    save_training_plot(
        merged_history,
        len(transfer_history.history.get("loss", [])),
        OUTPUT_DIR / "training_accuracy.png",
    )

    model_info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "number_of_classes": len(class_names),
        "total_images": len(image_paths),
        "training_images": len(train_paths),
        "validation_images": len(validation_paths),
        "test_images": len(test_paths),
        "selected_stage": selected_stage,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "weights": args.weights,
        "class_names": class_names,
    }
    MODEL_INFO_PATH.write_text(
        json.dumps(model_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nTraining complete")
    print("=" * 72)
    print(f"Selected model stage: {selected_stage}")
    print(f"Test loss:            {test_loss:.4f}")
    print(f"Test accuracy:        {test_accuracy * 100:.2f}%")
    print(f"Saved model:          {FINAL_MODEL_PATH}")
    print(f"Results folder:       {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
