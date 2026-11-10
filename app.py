from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from config import LOW_CONFIDENCE_THRESHOLD
from inference_utils import load_model_and_classes, predict_top_k


class PlantDiseaseApplication:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Plant Disease Detection using Transfer Learning")
        self.root.geometry("900x720")
        self.root.minsize(760, 650)
        self.selected_image_path: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        try:
            self.model, self.class_names = load_model_and_classes()
        except Exception as error:
            messagebox.showerror("Model error", str(error))
            self.root.destroy()
            return

        self._build_interface()

    def _build_interface(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        container = ttk.Frame(self.root, padding=22)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            container,
            text="Plant Disease Detection",
            font=("Segoe UI", 23, "bold"),
        ).pack(pady=(0, 4))
        ttk.Label(
            container,
            text="MobileNetV2 Transfer Learning",
            font=("Segoe UI", 12),
        ).pack(pady=(0, 10))
        ttk.Label(
            container,
            text=(
                "Select a clear photograph containing one leaf. Predictions "
                "outside the model's training classes are unreliable."
            ),
            wraplength=820,
            justify=tk.CENTER,
        ).pack(pady=(0, 14))

        self.image_label = ttk.Label(container, text="No image selected", anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=8)

        button_frame = ttk.Frame(container)
        button_frame.pack(pady=10)
        ttk.Button(
            button_frame,
            text="Select Leaf Image",
            command=self.select_image,
        ).pack(side=tk.LEFT, padx=8)
        ttk.Button(
            button_frame,
            text="Predict Disease",
            command=self.predict,
        ).pack(side=tk.LEFT, padx=8)

        self.result_label = ttk.Label(
            container,
            text="Prediction will appear here.",
            font=("Segoe UI", 12),
            wraplength=820,
            justify=tk.LEFT,
        )
        self.result_label.pack(fill=tk.X, pady=(12, 8))

        ttk.Separator(container).pack(fill=tk.X, pady=8)
        ttk.Label(
            container,
            text=(
                "Educational classifier only. It is not a substitute for "
                "laboratory diagnosis or advice from an agricultural specialist."
            ),
            wraplength=820,
            justify=tk.CENTER,
            font=("Segoe UI", 9),
        ).pack()

    def select_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select a plant-leaf image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return

        image_path = Path(selected)
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((570, 390), Image.Resampling.LANCZOS)
        except Exception as error:
            messagebox.showerror("Image error", f"Could not open image.\n\n{error}")
            return

        self.selected_image_path = image_path
        self.preview_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.preview_photo, text="")
        self.result_label.configure(text="Click Predict Disease.")

    def predict(self) -> None:
        if self.selected_image_path is None:
            messagebox.showwarning("No image", "Select a leaf image first.")
            return

        try:
            predictions = predict_top_k(
                self.model,
                self.class_names,
                self.selected_image_path,
                top_k=3,
            )
        except Exception as error:
            messagebox.showerror("Prediction error", str(error))
            return

        lines: list[str] = []
        if predictions[0][3] < LOW_CONFIDENCE_THRESHOLD:
            lines.extend([
                "LOW-CONFIDENCE RESULT: the image may not match the training dataset.",
                "",
            ])

        lines.append("Top predictions:")
        for rank, (_, _, readable_name, confidence) in enumerate(predictions, 1):
            lines.append(f"{rank}. {readable_name} — {confidence * 100:.2f}%")

        self.result_label.configure(text="\n".join(lines))


def main() -> None:
    root = tk.Tk()
    PlantDiseaseApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
