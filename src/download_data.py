"""Data Downloader for SignalGuard.

This module automates downloading datasets from Kaggle Hub.
It fetches a vehicle + number plate dataset and optionally a vehicle-color dataset.
The downloaded files are stored under `data/raw/`.
"""

import os
import sys
from pathlib import Path
import kagglehub
import shutil

def _download_dataset(dataset_ref: str, target_dir: Path) -> int:
    """Download a KaggleHub dataset.

    Args:
        dataset_ref: KaggleHub dataset reference, e.g. "andrewmvd/car-plate-detection".
        target_dir: Directory where the dataset should be extracted.
    Returns:
        Number of image files copied to the target directory.
    """
    print(f"Downloading dataset: {dataset_ref}")
    # kagglehub returns the path to the downloaded dataset archive (or folder)
    try:
        dataset_path = kagglehub.dataset_download(dataset_ref)
    except Exception as e:
        print(f"Failed to download {dataset_ref}: {e}")
        return 0

    src_path = Path(dataset_path)
    if not src_path.exists():
        print(f"Dataset path does not exist: {src_path}")
        return 0

    # If it's a zip or tar, KaggleHub extracts it; we assume a folder with images and annotations.
    copied = 0
    for root, _, files in os.walk(src_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_file = Path(root) / f
                dst_file = target_dir / src_file.name
                shutil.copy2(src_file, dst_file)
                copied += 1
    print(f"Copied {copied} images to {target_dir}")
    return copied

def download_datasets():
    """Download and prepare the required datasets.

    - Vehicle + number plate dataset (primary).
    - Small vehicle-color dataset (optional, if available).
    """
    print("SignalGuard Data Downloader")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    # Primary dataset
    total_images += _download_dataset("andrewmvd/car-plate-detection", raw_dir)

    # Attempt a secondary dataset (vehicle color). If not found, continue silently.
    secondary_ref = "paultimothymooney/vehicle-color-dataset"
    total_images += _download_dataset(secondary_ref, raw_dir)

    print(f"Total images downloaded: {total_images}")

if __name__ == "__main__":
    download_datasets()

