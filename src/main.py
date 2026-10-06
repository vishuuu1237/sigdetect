"""Main Entrypoint for SignalGuard.

This module provides CLI and orchestration commands to run the
SignalGuard pipeline on images or video sources.
"""

import sys
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SignalGuard: Traffic Signal Compliance & Violation System"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/sample_traffic.mp4",
        help="Path to input image or video file"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=3,
        help="Frame skipping interval for CPU optimization"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch interactive Streamlit dashboard"
    )
    return parser.parse_args()

def main():
    """Execute SignalGuard operations."""
    args = parse_args()
    print("========================================")
    print("🚦 SignalGuard Traffic Compliance System")
    print("========================================")
    print(f"Source: {args.source}")
    print(f"Frame skip: {args.frame_skip}")
    print("System initialized.")

if __name__ == "__main__":
    main()
