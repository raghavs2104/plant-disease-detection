from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def build_transfer_learning_model(
    number_of_classes: int,
    weights: str | None = "imagenet",
) -> tuple[keras.Model, keras.Model]:
    if number_of_classes < 2:
        raise ValueError("At least two classes are required.")

    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.12),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.10, 0.10),
            layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    inputs = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        name="leaf_image",
    )
    x = augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu", name="classifier_dense")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(
        number_of_classes,
        activation="softmax",
        name="disease_probabilities",
    )(x)

    model = keras.Model(inputs, outputs, name="plant_disease_mobilenet_v2")
    compile_model(model, learning_rate=1e-3)
    return model, base_model


def enable_fine_tuning(
    model: keras.Model,
    base_model: keras.Model,
    trainable_last_layers: int,
) -> None:
    if trainable_last_layers <= 0:
        raise ValueError("trainable_last_layers must be greater than zero.")

    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - trainable_last_layers)

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    for layer in base_model.layers[freeze_until:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

    compile_model(model, learning_rate=1e-5)
