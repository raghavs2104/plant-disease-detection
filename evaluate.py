from __future__ import annotations

import argparse
import numpy as np

from config import BATCH_SIZE, OUTPUT_DIR, SPLIT_MANIFEST_PATH
from dataset_utils import build_dataset, load_split_manifest
from inference_utils import load_model_and_classes
from training_utils import save_evaluation_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved test split.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    splits, manifest_classes = load_split_manifest(SPLIT_MANIFEST_PATH)
    model, saved_classes = load_model_and_classes()
    if manifest_classes != saved_classes:
        raise ValueError("Manifest classes do not match the saved model classes.")

    test_paths, test_labels = splits["test"]
    dataset = build_dataset(test_paths, test_labels, args.batch_size, False)
    test_loss, test_accuracy = model.evaluate(dataset, verbose=1)
    probabilities = model.predict(dataset, verbose=1)
    predictions = np.argmax(probabilities, axis=1)

    save_evaluation_outputs(
        test_labels,
        predictions,
        probabilities,
        test_paths,
        saved_classes,
        OUTPUT_DIR,
    )
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
