"""Output handling utilities for SignalGuard.

Provides `save_violation` which crops the offending vehicle and plate, saves the
cropped image and an annotated full‑frame image, and logs a CSV entry.
"""

import os
import cv2
import csv
import datetime

OUTPUT_ROOT = os.path.join("output", "violations")
CSV_PATH = os.path.join("output", "violations.csv")

def _ensure_dirs():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            # Updated columns: timestamp, track_id, plate_text, vehicle_type, color, occupants, violation_reason, image_path
            writer.writerow(["timestamp", "track_id", "plate_text", "vehicle_type", "color", "occupants", "violation_reason", "image_path"])

def save_violation(frame, bbox, plate, vehicle_type, color, occupants, reason, track_id):
    """Save a violation example and log it to output/violations.csv."""
    _ensure_dirs()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{track_id}"
    folder_path = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    x1, y1, x2, y2 = map(int, bbox)
    vehicle_crop = frame[y1:y2, x1:x2]
    vehicle_path = os.path.join(folder_path, "vehicle.jpg")
    if vehicle_crop.size > 0:
        cv2.imwrite(vehicle_path, vehicle_crop)

    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
    label = f"ID:{track_id} PLATE:{plate} VIOL:{reason}"
    cv2.putText(annotated, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    annotated_path = os.path.join(folder_path, "annotated.jpg")
    cv2.imwrite(annotated_path, annotated)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            track_id,
            plate if plate else "N/A",
            vehicle_type,
            color,
            occupants,
            reason,
            vehicle_path,
        ])
    print(f"Violation saved to {folder_path}")

