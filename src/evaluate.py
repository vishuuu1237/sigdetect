"""Evaluation script for YOLOv8 detector.

Loads the trained model (fallback to pretrained yolov8n.pt) and runs validation
on the dataset defined in `data/dataset.yaml`. The resulting metrics are saved
to `output/metrics.json`.
"""

import os
import json
from ultralytics import YOLO

def main():
    # Determine model path
    model_path = "models/detector_best.pt"
    if not os.path.exists(model_path):
        print(f"{model_path} not found, using pretrained yolov8n.pt instead.")
        model_path = "yolov8n.pt"
        if not os.path.exists(model_path):
            # trigger download of pretrained weights
            YOLO(model_path)
    # Load model
    model = YOLO(model_path)
    # Run validation (no training)
    results = model.val(data="data/dataset.yaml", batch=4, imgsz=640, device="cpu")
    # Extract metrics
    metrics = {}
    if hasattr(results, "metrics"):
        metrics = results.metrics
    else:
        # ultralytics may return dict directly
        metrics = results
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_path}")
    # Print some key metrics for quick view
    if isinstance(metrics, dict):
        print("mAP50:", metrics.get("map50", "N/A"))
        print("Precision:", metrics.get("precision", "N/A"))
        print("Recall:", metrics.get("recall", "N/A"))

if __name__ == "__main__":
    main()
