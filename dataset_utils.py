from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANDOM_SEED,
    SUPPORTED_IMAGE_EXTENSIONS,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)


def _directory_contains_images(directory: Path) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        for path in directory.rglob("*")
    )


def _class_directories(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        [
            child
            for child in directory.iterdir()
            if child.is_dir() and _directory_contains_images(child)
        ],
        key=lambda path: path.name.casefold(),
    )


def find_dataset_root(dataset_dir: Path) -> Path:
    """Find the folder whose immediate child folders are disease classes."""
    dataset_dir = dataset_dir.resolve()
    candidates = [
        dataset_dir / "PlantVillage" / "color",
        dataset_dir / "plantvillage" / "color",
        dataset_dir / "PlantVillage",
        dataset_dir / "plantvillage",
        dataset_dir / "color",
        dataset_dir / "Color",
        dataset_dir / "train",
        dataset_dir / "Train",
        dataset_dir / "training",
        dataset_dir / "Training",
        dataset_dir,
    ]

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)

    for candidate in unique_candidates:
        if len(_class_directories(candidate)) >= 2:
            return candidate

    if dataset_dir.exists():
        for candidate in sorted(dataset_dir.rglob("*")):
            try:
                depth = len(candidate.relative_to(dataset_dir).parts)
            except ValueError:
                continue
            if candidate.is_dir() and depth <= 4:
                if len(_class_directories(candidate)) >= 2:
                    return candidate

    checked = "\n".join(f"  - {path}" for path in unique_candidates)
    raise FileNotFoundError(
        "Could not locate plant-disease class folders.\n\n"
        "Each class must be a separate folder containing images, for example:\n"
        "dataset/PlantVillage/color/Tomato___Late_blight/*.jpg\n"
        "dataset/PlantVillage/color/Tomato___healthy/*.jpg\n\n"
        f"Checked common locations:\n{checked}"
    )


def collect_samples(dataset_root: Path) -> tuple[list[str], list[int], list[str]]:
    class_dirs = _class_directories(dataset_root)
    if len(class_dirs) < 2:
        raise ValueError(f"At least two class folders are required inside: {dataset_root}")

    class_names = [directory.name for directory in class_dirs]
    image_paths: list[str] = []
    labels: list[int] = []

    for class_id, class_directory in enumerate(class_dirs):
        class_image_count = 0
        for image_path in sorted(class_directory.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                image_paths.append(str(image_path.resolve()))
                labels.append(class_id)
                class_image_count += 1
        if class_image_count < 7:
            raise ValueError(
                f"Class '{class_directory.name}' contains only {class_image_count} images. "
                "At least 7 are required for train, validation and test splits."
            )

    return image_paths, labels, class_names


def split_samples(
    image_paths: Sequence[str], labels: Sequence[int]
) -> dict[str, tuple[list[str], list[int]]]:
    if not np.isclose(TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION, 1.0):
        raise ValueError("Train, validation and test fractions must sum to 1.")

    train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
        list(image_paths),
        list(labels),
        train_size=TRAIN_FRACTION,
        random_state=RANDOM_SEED,
        stratify=labels,
        shuffle=True,
    )

    validation_relative = VALIDATION_FRACTION / (VALIDATION_FRACTION + TEST_FRACTION)
    validation_paths, test_paths, validation_labels, test_labels = train_test_split(
        remaining_paths,
        remaining_labels,
        train_size=validation_relative,
        random_state=RANDOM_SEED,
        stratify=remaining_labels,
        shuffle=True,
    )

    return {
        "train": (train_paths, train_labels),
        "validation": (validation_paths, validation_labels),
        "test": (test_paths, test_labels),
    }


def save_split_manifest(
    splits: dict[str, tuple[list[str], list[int]]],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "image_path", "class_id", "class_name"])
        for split_name, (paths, labels) in splits.items():
            for image_path, class_id in zip(paths, labels):
                writer.writerow([split_name, image_path, class_id, class_names[class_id]])


def load_split_manifest(
    manifest_path: Path,
) -> tuple[dict[str, tuple[list[str], list[int]]], list[str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}\nRun train.py first.")

    split_paths = {"train": [], "validation": [], "test": []}
    split_labels = {"train": [], "validation": [], "test": []}
    class_mapping: dict[int, str] = {}

    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            split_name = row["split"]
            if split_name not in split_paths:
                continue
            image_path = row["image_path"]
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Manifest image is missing: {image_path}")
            class_id = int(row["class_id"])
            split_paths[split_name].append(image_path)
            split_labels[split_name].append(class_id)
            class_mapping[class_id] = row["class_name"]

    class_names = [class_mapping[index] for index in sorted(class_mapping)]
    splits = {
        name: (split_paths[name], split_labels[name]) for name in split_paths
    }
    return splits, class_names


def _load_image(image_path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH), antialias=True)
    return tf.cast(image, tf.float32), tf.cast(label, tf.int32)


def build_dataset(
    image_paths: Iterable[str],
    labels: Iterable[int],
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    paths_list = list(image_paths)
    labels_list = list(labels)
    if not paths_list:
        raise ValueError("Cannot build a dataset from an empty image list.")

    dataset = tf.data.Dataset.from_tensor_slices((paths_list, labels_list))
    if training:
        dataset = dataset.shuffle(
            min(len(paths_list), 10_000),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(
        _load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def print_dataset_summary(
    splits: dict[str, tuple[list[str], list[int]]], class_names: Sequence[str]
) -> None:
    print("\nDataset summary")
    print("=" * 72)
    print(f"Number of classes: {len(class_names)}")
    for split_name, (paths, _) in splits.items():
        print(f"{split_name.title():12s}: {len(paths):6d} images")

    counts = Counter(splits["train"][1])
    print("\nTraining images per class")
    print("-" * 72)
    for class_id, class_name in enumerate(class_names):
        print(f"{class_id:02d} | {class_name[:50]:50s} | {counts[class_id]:5d}")


def humanize_class_name(class_name: str) -> str:
    readable = class_name.replace("___", " - ").replace("__", " ").replace("_", " ")
    return " ".join(readable.split())
