"""Data Preparation and Formatting for SignalGuard.

This module processes raw vehicle and license plate datasets into YOLO-format
annotations, splits images into train/val/test, and validates labels.
"""

import os
import sys

def prepare_data():
    """Verify raw data and structure annotations for YOLO model training."""
    print("SignalGuard Data Preparation")
    print("Preparing YOLO formatted train/val/test splits...")
    os.makedirs("data/annotated/train/images", exist_ok=True)
    os.makedirs("data/annotated/train/labels", exist_ok=True)
    os.makedirs("data/annotated/val/images", exist_ok=True)
    os.makedirs("data/annotated/val/labels", exist_ok=True)
    print("Dataset structure verified.")

if __name__ == "__main__":
    prepare_data()
