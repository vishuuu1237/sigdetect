"""Main entry point for SignalGuard pipeline.

Provides a CLI to run the end‑to‑end inference pipeline on a single image or a video source.
"""

import argparse
import os
import cv2
from pipeline import SignalGuardPipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="SignalGuard: Traffic Signal Compliance & Violation System"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/sample_traffic.mp4",
        help="Path to input image or video file",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=3,
        help="Frame skipping interval for CPU optimization",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU inference if available",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("========================================")
    print("🚦 SignalGuard Traffic Compliance System")
    print("========================================")
    print(f"Source: {args.source}")
    print(f"Frame skip: {args.frame_skip}")
    print("System initialized.")

    pipeline = SignalGuardPipeline(frame_skip=args.frame_skip, use_gpu=args.use_gpu)

    if not os.path.isfile(args.source):
        print(f"Error: source file {args.source} does not exist.")
        return

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    ext = os.path.splitext(args.source)[1].lower()
    if ext in image_exts:
        frame = cv2.imread(args.source)
        if frame is None:
            print("Failed to read the image.")
            return
        annotated = pipeline.process_frame(frame)
        os.makedirs("output", exist_ok=True)
        out_path = os.path.join("output", "annotated_image.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"Annotated image saved to {out_path}")
    else:
        pipeline.process_video(args.source)

    print("PHASE 6 COMPLETE")

if __name__ == "__main__":
    main()
