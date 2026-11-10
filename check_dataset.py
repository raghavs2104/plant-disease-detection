from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from config import DATASET_DIR
from dataset_utils import collect_samples, find_dataset_root, humanize_class_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the dataset structure.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    args = parser.parse_args()

    root = find_dataset_root(args.dataset)
    paths, labels, class_names = collect_samples(root)
    counts = Counter(labels)

    print(f"\nDataset root: {root}")
    print(f"Classes:      {len(class_names)}")
    print(f"Images:       {len(paths)}")
    print("\nImages per class")
    print("=" * 78)
    for class_id, class_name in enumerate(class_names):
        print(
            f"{class_id:02d} | {humanize_class_name(class_name)[:54]:54s} | "
            f"{counts[class_id]:5d}"
        )


if __name__ == "__main__":
    main()
