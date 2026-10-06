"""Simple IoU based tracker for SignalGuard.

Provides `IOUTracker` that maintains a mapping of track IDs to bounding boxes.
On each update it matches new detections to existing tracks based on IoU >
threshold and assigns IDs. Unmatched detections receive new IDs.
"""

import numpy as np


def _iou(boxA, boxB):
    # boxes are [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = inter / (areaA + areaB - inter + 1e-6)
    return iou


class IOUTracker:
    """Tracks objects across frames using IoU similarity.

    Attributes
    ----------
    iou_threshold: float
        Minimum IoU required to consider a detection the same object.
    next_id: int
        Counter for assigning new track IDs.
    tracks: dict[int, list]
        Mapping of track_id -> latest bbox.
    """

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.next_id = 0
        self.tracks = {}

    def update(self, detections):
        """Update tracker with a list of detection dicts.

        Parameters
        ----------
        detections: list[dict]
            Each dict must contain a ``bbox`` key with [x1, y1, x2, y2].

        Returns
        -------
        list[int]
            List of track IDs corresponding to the input detections order.
        """
        assigned_ids = []
        used_tracks = set()
        for det in detections:
            best_iou = 0.0
            best_id = None
            for tid, tr_bbox in self.tracks.items():
                if tid in used_tracks:
                    continue
                iou = _iou(det["bbox"], tr_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_id is not None and best_iou >= self.iou_threshold:
                assigned_ids.append(best_id)
                self.tracks[best_id] = det["bbox"]
                used_tracks.add(best_id)
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = det["bbox"]
                assigned_ids.append(new_id)
                used_tracks.add(new_id)
        return assigned_ids

# Backward compatibility
VehicleTracker = IOUTracker

