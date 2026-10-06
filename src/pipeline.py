"""End-to-End Inference Pipeline for SignalGuard.

This module chains together detection, optical character recognition (OCR),
compliance validation, object tracking across frames, and attribute estimation.
Optimized for local CPU execution and cloud deployment.
"""

import os
import sys
import re
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
            use_gpu: False for CPU inference on Dell Latitude 7420 / Cloud.
        """
        self.frame_skip = frame_skip
        self.use_gpu = use_gpu
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Load vehicle/plate detector (custom model if present, otherwise standard YOLOv8)
        detector_path = "models/detector_best.pt" if os.path.isfile("models/detector_best.pt") else "yolov8n.pt"
        print(f"Loading detector from: {detector_path}")
        self.detector = YOLO(detector_path)
        
        # Load attribute classifiers (MobileNetV2) – may be missing
        self.type_classifier = None
        self.color_classifier = None
        if os.path.isfile("models/type_classifier.pt"):
            try:
                loaded = torch.load("models/type_classifier.pt", map_location=self.device)
                if hasattr(loaded, 'eval'):
                    loaded.eval()
                    self.type_classifier = loaded
                else:
                    self.type_classifier = None
            except Exception as e:
                print(f"Warning: Failed to load type_classifier.pt: {e}")
        if os.path.isfile("models/color_classifier.pt"):
            try:
                loaded = torch.load("models/color_classifier.pt", map_location=self.device)
                if hasattr(loaded, 'eval'):
                    loaded.eval()
                    self.color_classifier = loaded
                else:
                    self.color_classifier = None
            except Exception as e:
                print(f"Warning: Failed to load color_classifier.pt: {e}")
                
        # Person detector for occupant counting
        self.person_detector = YOLO("yolov8n.pt")
        
        # Initialise OCR Engine
        self.ocr_engine = None
        if _HAS_EASYOCR:
            try:
                print("Initializing EasyOCR reader (en)...")
                self.ocr_engine = ("easyocr", easyocr.Reader(['en'], gpu=use_gpu and torch.cuda.is_available()))
            except Exception as e:
                print(f"Warning: EasyOCR init failed: {e}")
        if self.ocr_engine is None and _HAS_PADDLEOCR:
            try:
                print("Initializing PaddleOCR reader (en)...")
                self.ocr_engine = ("paddleocr", PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu and torch.cuda.is_available()))
            except Exception as e:
                print(f"Warning: PaddleOCR init failed: {e}")
        if self.ocr_engine is None:
            print("Warning: No OCR engine available. License plate text recognition will be simulated.")
            
        # Initialise tracker
        self.tracker = IOUTracker(iou_threshold=0.3)
        
        # Performance Counters
        self.total_vehicles = 0
        self.total_violations = 0
        self.plates_read = 0
        print(f"SignalGuardPipeline ready (device={self.device})")

    def _crop(self, frame: np.ndarray, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]

    def _estimate_color(self, crop: np.ndarray) -> str:
        """Estimate dominant vehicle color using HSV histogram."""
        if crop.size == 0:
            return "unknown"
        # Use center 60% of crop to avoid background/road noise
        h, w = crop.shape[:2]
        ch1, ch2 = int(h * 0.2), int(h * 0.8)
        cw1, cw2 = int(w * 0.2), int(w * 0.8)
        center_crop = crop[ch1:ch2, cw1:cw2] if (ch2 > ch1 and cw2 > cw1) else crop
        
        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        h_val = float(np.median(hsv[:, :, 0]))
        s_val = float(np.mean(hsv[:, :, 1]))
        v_val = float(np.mean(hsv[:, :, 2]))

        if v_val < 45:
            return "black"
        if s_val < 35 and v_val > 175:
            return "white"
        if s_val < 45:
            return "silver/grey"
        if (h_val < 10 or h_val > 165):
            return "red"
        if 10 <= h_val < 25:
            return "orange"
        if 25 <= h_val < 35:
            return "yellow"
        if 35 <= h_val < 85:
            return "green"
        if 85 <= h_val < 135:
            return "blue"
        return "dark"

    def _estimate_type(self, yolo_cls_name: str, crop: np.ndarray) -> str:
        """Estimate vehicle category (2-wheeler, 3-wheeler, 4-wheeler)."""
        name = yolo_cls_name.lower()
        if name in ["motorcycle", "motorbike", "bicycle", "bike", "scooter"]:
            return "2-wheeler"
        if name in ["auto", "rickshaw", "autorickshaw", "tuk-tuk"]:
            return "3-wheeler"
        if name in ["car", "truck", "bus", "van", "suv", "jeep"]:
            return "4-wheeler"
        
        # If unknown label, use aspect ratio of crop
        if crop.size > 0:
            h, w = crop.shape[:2]
            aspect = w / float(h + 1e-5)
            if aspect < 0.85:
                return "2-wheeler"
            elif aspect > 1.2:
                return "4-wheeler"
        return "vehicle"

    def _extract_plate_text_and_box(self, vehicle_crop: np.ndarray, vehicle_box):
        """Run OCR on the vehicle crop to find text and localize the plate box."""
        if self.ocr_engine is None or vehicle_crop.size == 0:
            return "", None

        engine_name, engine = self.ocr_engine
        vh, vw = vehicle_crop.shape[:2]
        vx1, vy1 = vehicle_box[0], vehicle_box[1]

        best_text = ""
        best_box = None

        try:
            if engine_name == "easyocr":
                # Run EasyOCR on vehicle crop
                ocr_results = engine.readtext(vehicle_crop)
                for bbox, text, prob in ocr_results:
                    clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
                    # Look for plausible plate formats (e.g. state code + numbers, 4-12 chars)
                    if len(clean) >= 4 and any(c.isdigit() for c in clean) and any(c.isalpha() for c in clean):
                        best_text = clean
                        # Map polygon to bounding box in absolute image coordinates
                        pts = np.array(bbox)
                        px1 = int(vx1 + pts[:, 0].min())
                        py1 = int(vy1 + pts[:, 1].min())
                        px2 = int(vx1 + pts[:, 0].max())
                        py2 = int(vy1 + pts[:, 1].max())
                        best_box = [px1, py1, px2, py2]
                        break
                    elif len(clean) >= 4 and not best_text:
                        best_text = clean

            elif engine_name == "paddleocr":
                ocr_results = engine(vehicle_crop)
                if ocr_results and len(ocr_results[0]) > 0:
                    for line in ocr_results[0]:
                        text = line[1][0]
                        clean = re.sub(r'[^A-Za-z0-9]', '', text).upper()
                        if len(clean) >= 4:
                            best_text = clean
                            pts = np.array(line[0])
                            px1 = int(vx1 + pts[:, 0].min())
                            py1 = int(vy1 + pts[:, 1].min())
                            px2 = int(vx1 + pts[:, 0].max())
                            py2 = int(vy1 + pts[:, 1].max())
                            best_box = [px1, py1, px2, py2]
                            break
        except Exception as e:
            print(f"OCR inference error: {e}")

        return best_text, best_box

    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.25):
        """Process a single image or video frame and return annotated frame and detections."""
        if frame is None or frame.size == 0:
            return frame, []

        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Run YOLO detector
        results = self.detector(frame, conf=conf_threshold, verbose=False)
        
        vehicle_detections = []
        explicit_plate_boxes = []

        # Vehicle & Plate keyword matcher
        vehicle_keywords = {"vehicle", "car", "motorcycle", "motorbike", "bus", "truck", "auto", "rickshaw", "bicycle", "van"}
        plate_keywords = {"number_plate", "plate", "license_plate", "license", "licence_plate", "numberplate"}

        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                cls_name = names.get(cls_id, "").lower()
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                xyxy = box.xyxy[0].cpu().numpy().tolist()

                # Check if detected class is a vehicle
                if cls_name in vehicle_keywords or any(vk in cls_name for vk in vehicle_keywords) or cls_id in [1, 2, 3, 5, 7]:
                    vehicle_detections.append((xyxy, cls_name, conf))
                # Check if detected class is a license plate
                elif cls_name in plate_keywords or any(pk in cls_name for pk in plate_keywords):
                    explicit_plate_boxes.append(xyxy)

        detections = []
        
        for v_idx, (vbox, orig_cls_name, conf) in enumerate(vehicle_detections):
            self.total_vehicles += 1
            vehicle_crop = self._crop(frame, vbox)
            
            # 1. Attribute estimation: Type & Color
            vehicle_type = self._estimate_type(orig_cls_name, vehicle_crop)
            vehicle_color = self._estimate_color(vehicle_crop)

            if self.type_classifier is not None and vehicle_crop.size > 0:
                try:
                    with torch.no_grad():
                        resized = cv2.resize(vehicle_crop, (224, 224))
                        inp = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
                        out = self.type_classifier(inp)
                        vehicle_type = ["2-wheeler", "3-wheeler", "4-wheeler"][out.argmax().item()]
                except Exception:
                    pass

            if self.color_classifier is not None and vehicle_crop.size > 0:
                try:
                    with torch.no_grad():
                        resized = cv2.resize(vehicle_crop, (224, 224))
                        inp = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
                        out = self.color_classifier(inp)
                        vehicle_color = ["red", "blue", "black", "white", "silver", "green", "yellow"][out.argmax().item()]
                except Exception:
                    pass

            # 2. Occupant counting via person detector
            occupants = 0
            if vehicle_crop.size > 0:
                try:
                    person_results = self.person_detector(vehicle_crop, conf=0.3, classes=[0], verbose=False)
                    for pr in person_results:
                        occupants += len(pr.boxes)
                except Exception:
                    pass

            # 3. Plate extraction & OCR
            plate_text = ""
            plate_box = None

            # Check if any explicit plate box belongs to this vehicle
            for pbox in explicit_plate_boxes:
                ix1, iy1 = max(vbox[0], pbox[0]), max(vbox[1], pbox[1])
                ix2, iy2 = min(vbox[2], pbox[2]), min(vbox[3], pbox[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_p = (pbox[2] - pbox[0]) * (pbox[3] - pbox[1])
                if area_p > 0 and (inter / area_p) > 0.4:
                    plate_box = pbox
                    p_crop = self._crop(frame, pbox)
                    if p_crop.size > 0 and self.ocr_engine is not None:
                        engine_name, engine = self.ocr_engine
                        try:
                            if engine_name == "easyocr":
                                ores = engine.readtext(p_crop)
                                if ores:
                                    plate_text = re.sub(r'[^A-Za-z0-9]', '', ores[0][1]).upper()
                            elif engine_name == "paddleocr":
                                ores = engine(p_crop)
                                if ores and len(ores[0]) > 0:
                                    plate_text = re.sub(r'[^A-Za-z0-9]', '', ores[0][0][1]).upper()
                        except Exception:
                            pass
                    break

            # If no explicit plate detected, scan the vehicle crop with OCR
            if not plate_text and vehicle_crop.size > 0:
                extracted_text, detected_pbox = self._extract_plate_text_and_box(vehicle_crop, vbox)
                if extracted_text:
                    plate_text = extracted_text
                    if detected_pbox:
                        plate_box = detected_pbox

            # 4. Compliance check
            compliant = check_compliance(plate_text)
            if plate_text:
                self.plates_read += 1

            det_dict = {
                "bbox": vbox,
                "plate": plate_text,
                "type": vehicle_type,
                "color": vehicle_color,
                "occupants": occupants,
                "compliant": compliant["compliant"],
                "violations": [] if compliant["compliant"] else [compliant["reason"]]
            }
            detections.append(det_dict)

            # 5. Draw Annotations
            vx1, vy1, vx2, vy2 = map(int, vbox)
            box_color = (0, 200, 0) if compliant["compliant"] else (0, 0, 220)  # Green vs Red
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), box_color, 2)

            # Highlight Number Plate if detected
            if plate_box is not None:
                px1, py1, px2, py2 = map(int, plate_box)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 255), 2)  # Yellow plate border
                if plate_text:
                    cv2.putText(annotated, plate_text, (px1, max(15, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw clean status badge
            status_text = "COMPLIANT" if compliant["compliant"] else f"VIOLATION: {compliant['reason']}"
            badge_line1 = f"{vehicle_type.upper()} ({vehicle_color}) | Occ: {occupants}"
            badge_line2 = f"Plate: {plate_text if plate_text else 'N/A'} | {status_text}"

            # Badge background
            bg_y1 = max(0, vy1 - 42)
            bg_y2 = max(20, vy1 - 2)
            cv2.rectangle(annotated, (vx1, bg_y1), (min(w, vx1 + max(len(badge_line1), len(badge_line2)) * 9), bg_y2), (20, 20, 20), -1)
            cv2.putText(annotated, badge_line1, (vx1 + 5, bg_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            status_color = (100, 255, 100) if compliant["compliant"] else (100, 100, 255)
            cv2.putText(annotated, badge_line2, (vx1 + 5, bg_y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

            # 6. Save violation if non-compliant
            if not compliant["compliant"]:
                self.total_violations += 1
                try:
                    save_violation(frame, vbox, plate_text, vehicle_type, vehicle_color, occupants, compliant["reason"], v_idx)
                except Exception as e:
                    print(f"Warning: Failed to log violation: {e}")

        # Update tracker
        self.tracker.update(detections)

        return annotated, detections

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
                annotated, _ = self.process_frame(frame)
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
    img = cv2.imread('test.jpg')
    if img is not None:
        annotated, dets = pipeline.process_frame(img)
        os.makedirs('output', exist_ok=True)
        cv2.imwrite('output/annotated_image.jpg', annotated)
        print(f"Processed image. Found {len(dets)} detections.")
    else:
        print('Provide a valid image path to test the pipeline.')
