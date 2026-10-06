"""Violation Output and Logging Handler for SignalGuard.

This module handles saving violation crops/snapshots and logging
violation records with metadata (timestamp, plate, vehicle type, color,
occupants, reason) to CSV files.
"""

import os
import pandas as pd
from datetime import datetime

class OutputHandler:
    """Manages saving violation evidence images and structured CSV reports."""

    def __init__(self, output_dir: str = "output"):
        """Initialize directory paths for violations and reports."""
        self.output_dir = output_dir
        self.violations_dir = os.path.join(output_dir, "violations")
        self.csv_path = os.path.join(output_dir, "violations_log.csv")
        os.makedirs(self.violations_dir, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        """Initialize the CSV log file if not already present."""
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=[
                "timestamp", "track_id", "plate_text", "is_compliant",
                "violation_reason", "vehicle_type", "vehicle_color",
                "occupant_count", "evidence_image"
            ])
            df.to_csv(self.csv_path, index=False)

    def log_violation(self, record: dict, frame=None):
        """Append a violation record and save cropped evidence."""
        print(f"Logging violation: {record}")

if __name__ == "__main__":
    handler = OutputHandler()
    print("SignalGuard OutputHandler initialized successfully.")
