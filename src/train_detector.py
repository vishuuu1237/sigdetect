"""CPU-friendly YOLOv8 detector training.

This script trains a YOLOv8n model on the prepared dataset using CPU.
It uses a small number of epochs (1) for quick verification –
for full training, run the Colab notebook.
"""

import os
import sys
import shutil
from ultralytics import YOLO

def main():
    # Ensure the pretrained weights are available
    pretrained = "yolov8n.pt"
    if not os.path.exists(pretrained):
        print("Downloading pretrained YOLOv8n weights...")
        YOLO(pretrained)  # This triggers download to cache
    # Load model
    model = YOLO(pretrained)
    # Train with CPU-friendly settings (1 epoch for quick test)
    results = model.train(
        data="data/dataset.yaml",
        epochs=1,            # quick verification; increase for real training
        imgsz=640,
        batch=4,
        device="cpu",
        patience=20,
        augment=True,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015,
        hsv_v=0.4,
        degrees=10,
    )
    # Save the best checkpoint to models/detector_best.pt
    best_path = getattr(results, "best", "runs/detect/train/weights/best.pt")
    os.makedirs("models", exist_ok=True)
    dest = "models/detector_best.pt"
    try:
        shutil.copy2(best_path, dest)
        print(f"Best model saved to {dest}")
    except Exception as e:
        print(f"Failed to copy best model: {e}")
    print("CPU training is slow. Use colab_train.ipynb for full training.")

if __name__ == "__main__":
    main()
