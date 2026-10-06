"""Tests for src/tracker.py – IoU-based object tracker."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tracker import IOUTracker, _iou


# ── IoU helper ────────────────────────────────────────────────────────────────

def test_iou_identical_boxes():
    box = [0, 0, 100, 100]
    assert abs(_iou(box, box) - 1.0) < 1e-5


def test_iou_no_overlap():
    a = [0, 0, 50, 50]
    b = [100, 100, 200, 200]
    assert abs(_iou(a, b)) < 1e-5


def test_iou_partial_overlap():
    a = [0, 0, 100, 100]
    b = [50, 50, 150, 150]
    # intersection = 50*50 = 2500, union = 10000 + 10000 - 2500 = 17500
    expected = 2500 / 17500
    assert abs(_iou(a, b) - expected) < 1e-4


# ── Tracker basics ────────────────────────────────────────────────────────────

def test_tracker_assigns_new_ids():
    tracker = IOUTracker(iou_threshold=0.3)
    dets = [
        {"bbox": [0, 0, 50, 50]},
        {"bbox": [200, 200, 300, 300]},
    ]
    ids = tracker.update(dets)
    assert len(ids) == 2
    assert ids[0] != ids[1]


def test_tracker_maintains_ids_across_frames():
    tracker = IOUTracker(iou_threshold=0.3)
    # Frame 1
    dets1 = [{"bbox": [0, 0, 100, 100]}]
    ids1 = tracker.update(dets1)
    # Frame 2 – same box, slight shift
    dets2 = [{"bbox": [5, 5, 105, 105]}]
    ids2 = tracker.update(dets2)
    assert ids1[0] == ids2[0], "Same object should keep the same ID"


def test_tracker_new_id_when_no_overlap():
    tracker = IOUTracker(iou_threshold=0.3)
    ids1 = tracker.update([{"bbox": [0, 0, 50, 50]}])
    ids2 = tracker.update([{"bbox": [500, 500, 600, 600]}])
    assert ids1[0] != ids2[0], "Non-overlapping box should get a new ID"


def test_tracker_multiple_objects_stable():
    tracker = IOUTracker(iou_threshold=0.3)
    dets1 = [
        {"bbox": [0, 0, 50, 50]},
        {"bbox": [200, 200, 300, 300]},
    ]
    ids1 = tracker.update(dets1)

    # Slight shift on both
    dets2 = [
        {"bbox": [3, 3, 53, 53]},
        {"bbox": [202, 202, 302, 302]},
    ]
    ids2 = tracker.update(dets2)
    assert ids1 == ids2, "Both tracks should be preserved"


def test_tracker_handles_empty_detections():
    tracker = IOUTracker(iou_threshold=0.3)
    ids = tracker.update([])
    assert ids == []
