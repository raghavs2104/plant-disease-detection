from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from config import CLASS_NAMES_PATH, FINAL_MODEL_PATH, IMAGE_HEIGHT, IMAGE_WIDTH
from dataset_utils import humanize_class_name


def load_model_and_classes(
    model_path: Path = FINAL_MODEL_PATH,
    class_names_path: Path = CLASS_NAMES_PATH,
) -> tuple[keras.Model, list[str]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}\nRun train.py first.")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class-name file not found: {class_names_path}\nRun train.py first.")

    model = keras.models.load_model(model_path)
    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    if len(class_names) != model.output_shape[-1]:
        raise ValueError("Saved class names do not match the model output size.")
    return model, class_names


def load_image_for_prediction(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as error:
        raise ValueError(f"Could not read image: {image_path}") from error

    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def predict_top_k(
    model: keras.Model,
    class_names: list[str],
    image_path: Path,
    top_k: int = 3,
) -> list[tuple[int, str, str, float]]:
    probabilities = model.predict(load_image_for_prediction(image_path), verbose=0)[0]
    top_k = max(1, min(top_k, len(class_names)))
    indices = np.argsort(probabilities)[-top_k:][::-1]
    return [
        (
            int(index),
            class_names[int(index)],
            humanize_class_name(class_names[int(index)]),
            float(probabilities[int(index)]),
        )
        for index in indices
    ]
