from __future__ import annotations

import argparse
from pathlib import Path
from tkinter import Tk, filedialog

from inference_utils import load_model_and_classes, predict_top_k


def choose_image() -> Path | None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Select a plant-leaf image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(selected) if selected else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict disease from one leaf image.")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    image_path = args.image or choose_image()
    if image_path is None:
        print("No image selected.")
        return

    model, class_names = load_model_and_classes()
    predictions = predict_top_k(model, class_names, image_path, args.top_k)

    print(f"\nImage: {image_path}")
    print("Top predictions")
    print("=" * 72)
    for rank, (_, raw_name, readable_name, confidence) in enumerate(predictions, 1):
        print(
            f"{rank}. {readable_name} ({confidence * 100:.2f}%) "
            f"[folder: {raw_name}]"
        )


if __name__ == "__main__":
    main()
