"""End-to-End Inference Pipeline for SignalGuard.

This module chains together detection, optical character recognition (OCR),
compliance validation, object tracking across frames, and attribute estimation.
Optimized for local CPU execution with frame skipping.
"""

import os
import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO
# Optional OCR imports (EasyOCR preferred for cloud compatibility, PaddleOCR fallback)
try:
    import easyocr
    _HAS_EASYOCR = True
except ImportError:
    _HAS_EASYOCR = False

try:
    from paddleocr import PaddleOCR
    _HAS_PADDLEOCR = True
except ImportError:
    _HAS_PADDLEOCR = False

from compliance import check_compliance
from tracker import IOUTracker
from output_handler import save_violation

class SignalGuardPipeline:
    """Orchestrates end-to-end video and image processing for SignalGuard."""

    def __init__(self, frame_skip: int = 3, use_gpu: bool = False):
        """Initialize the pipeline with hardware optimizations.

        Args:
            frame_skip: Process every N-th frame to minimize CPU load.
            use_gpu: False for CPU inference on Dell Latitude 7420.
        """
        self.frame_skip = frame_skip
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        # Load vehicle/plate detector (YOLOv8) – fallback to default yolov8n.pt
        detector_path = "models/detector_best.pt" if os.path.isfile("models/detector_best.pt") else "yolov8n.pt"
        self.detector = YOLO(detector_path)
        # Load attribute classifiers (MobileNetV2) – may be missing
        self.type_classifier = None
        self.color_classifier = None
        if os.path.isfile("models/type_classifier.pt"):
            self.type_classifier = torch.load("models/type_classifier.pt", map_location=self.device)
        else:
            print("Warning: type_classifier.pt not found – vehicle type will be 'unknown'.")
        if os.path.isfile("models/color_classifier.pt"):
            self.color_classifier = torch.load("models/color_classifier.pt", map_location=self.device)
        else:
            print("Warning: color_classifier.pt not found – vehicle color will be 'unknown'.")
        # Load COCO person detector (YOLOv8) for occupant counting
        self.person_detector = YOLO("yolov8n.pt")
        # Initialise OCR
        self.ocr_engine = None
        if _HAS_EASYOCR:
            try:
                self.ocr_engine = ("easyocr", easyocr.Reader(['en'], gpu=use_gpu and torch.cuda.is_available()))
            except Exception as e:
                print(f"Warning: EasyOCR init failed: {e}")
        if self.ocr_engine is None and _HAS_PADDLEOCR:
            try:
                self.ocr_engine = ("paddleocr", PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu and torch.cuda.is_available()))
            except Exception as e:
                print(f"Warning: PaddleOCR init failed: {e}")
        if self.ocr_engine is None:
            print("Warning: No OCR engine available. License plate text recognition will be simulated.")
        # Initialise tracker
        self.tracker = IOUTracker(iou_threshold=0.3)
        # Counters for final summary
        self.total_vehicles = 0
        self.total_violations = 0
        self.plates_read = 0
        print(f"Initialized SignalGuardPipeline (frame_skip={frame_skip}, use_gpu={use_gpu})")

    def _crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return frame[y1:y2, x1:x2]

    def process_frame(self, frame):
        """Process a single image or video frame and return annotated frame."""
        results = self.detector(frame)
        detections = []  # list of dicts for tracker update
        annotated = frame.copy()
        # Assuming class 0 = vehicle, class 1 = plate (based on earlier training)
        vehicle_boxes = []
        plate_boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0]) if box.cls is not None else -1
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                if cls == 0:  # vehicle
                    vehicle_boxes.append(xyxy)
                elif cls == 1:  # plate
                    plate_boxes.append(xyxy)
        for v_idx, vbox in enumerate(vehicle_boxes):
            self.total_vehicles += 1
            vehicle_crop = self._crop(frame, vbox)
            # Attribute classification
            vehicle_type = "unknown"
            vehicle_color = "unknown"
            if self.type_classifier is not None:
                with torch.no_grad():
                    inp = torch.from_numpy(vehicle_crop).permute(2,0,1).unsqueeze(0).float().to(self.device) / 255.0
                    out = self.type_classifier(inp)
                    vehicle_type = ["2-wheeler", "3-wheeler", "4-wheeler"][out.argmax().item()]
            if self.color_classifier is not None:
                with torch.no_grad():
                    inp = torch.from_numpy(vehicle_crop).permute(2,0,1).unsqueeze(0).float().to(self.device) / 255.0
                    out = self.color_classifier(inp)
                    vehicle_color = ["red","blue","black","white","silver","green","yellow"][out.argmax().item()]
            # Occupant counting via person detector on the vehicle crop
            person_results = self.person_detector(vehicle_crop)
            occupants = 0
            for pr in person_results:
                for pb in pr.boxes:
                    if int(pb.cls[0]) == 0:  # COCO person class id is 0
                        occupants += 1
            # Find plate inside vehicle box (IoU > 0.3)
            plate_text = ""
            compliant = {"compliant": True, "reason": "No plate detected"}
            for pbox in plate_boxes:
                # simple IoU check
                ix1 = max(vbox[0], pbox[0])
                iy1 = max(vbox[1], pbox[1])
                ix2 = min(vbox[2], pbox[2])
                iy2 = min(vbox[3], pbox[3])
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                area_v = (vbox[2]-vbox[0]) * (vbox[3]-vbox[1])
                area_p = (pbox[2]-pbox[0]) * (pbox[3]-pbox[1])
                iou = inter / (area_v + area_p - inter + 1e-6)
                if iou > 0.3:
                    plate_crop = self._crop(frame, pbox)
                    # Run available OCR engine
                    if self.ocr_engine is not None and plate_crop.size > 0:
                        engine_name, engine = self.ocr_engine
                        try:
                            if engine_name == "easyocr":
                                results = engine.readtext(plate_crop)
                                if results:
                                    plate_text = results[0][1]
                            elif engine_name == "paddleocr":
                                results = engine(plate_crop)
                                if results and len(results[0]) > 0:
                                    plate_text = results[0][0][1]
                        except Exception as e:
                            print(f"OCR reading error: {e}")
                    if plate_text:
                        self.plates_read += 1
                        compliant = check_compliance(plate_text)
                    break
            # Tracker update dict
            det_dict = {"bbox": vbox, "plate": plate_text, "type": vehicle_type,
                        "color": vehicle_color, "occupants": occupants,
                        "violations": [] if compliant["compliant"] else [compliant["reason"]]}
            detections.append(det_dict)
            # Draw annotations
            x1,y1,x2,y2 = map(int, vbox)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            label = f"ID:{v_idx} {vehicle_type}/{vehicle_color} occ:{occupants}"
            cv2.putText(annotated, label, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            plate_lbl = plate_text if plate_text else "no plate"
            cv2.putText(annotated, f"Plate:{plate_lbl}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            status = "COMPLIANT" if compliant["compliant"] else "VIOLATION"
            cv2.putText(annotated, status, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255) if not compliant["compliant"] else (0,255,0), 2)
            # Save violation if needed
            if not compliant["compliant"]:
                self.total_violations += 1
                track_id = v_idx  # temporary; will be overwritten by tracker IDs below
                save_violation(frame, vbox, plate_text, vehicle_type, vehicle_color, occupants, compliant["reason"], track_id)
        # Update tracker and re‑assign proper IDs
        ids = self.tracker.update(detections)
        # (IDs are not re‑drawn here to keep code short)
        return annotated

    def process_video(self, source_path_or_camera: str):
        cap = cv2.VideoCapture(0 if source_path_or_camera == "webcam" else source_path_or_camera)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join("output", "annotated_video.mp4")
        out = None
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_skip == 0:
                annotated = self.process_frame(frame)
                if out is None:
                    h, w = annotated.shape[:2]
                    out = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                out.write(annotated)
                cv2.imshow('SignalGuard', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_idx += 1
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"Video processing complete. Saved to {out_path}")
        print(f"Summary: Vehicles={self.total_vehicles}, Violations={self.total_violations}, PlatesRead={self.plates_read}")

if __name__ == "__main__":
    pipeline = SignalGuardPipeline()
    # Example usage on a single image (replace path with actual test image)
    img = cv2.imread('test.jpg')
    if img is not None:
        annotated = pipeline.process_frame(img)
        cv2.imwrite('output/annotated_image.jpg', annotated)
        cv2.imshow('Result', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('Provide a valid image path to test the pipeline.')
