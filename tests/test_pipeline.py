"""Tests for src/pipeline.py – SignalGuardPipeline smoke test.

These tests verify that the pipeline can be instantiated and that
process_frame produces an annotated image and writes output artefacts.
They use a synthetic test image so no real data is required.
"""

import sys
import os
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="module")
def pipeline():
    """Instantiate the pipeline once for the whole module (model loading is slow)."""
    from pipeline import SignalGuardPipeline
    return SignalGuardPipeline(frame_skip=1, use_gpu=False)


@pytest.fixture
def sample_image():
    """Create a simple 640x480 BGR test image with a coloured rectangle."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (400, 350), (0, 128, 255), -1)  # orange box
    cv2.putText(img, "KA01AB1234", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


def test_pipeline_initialises(pipeline):
    """Pipeline should load without errors even when custom models are missing."""
    assert pipeline is not None
    assert pipeline.detector is not None


def test_process_frame_returns_image(pipeline, sample_image):
    """process_frame must return an ndarray of the same spatial dimensions."""
    annotated = pipeline.process_frame(sample_image)
    # process_frame may return (annotated, detections) tuple or just the image
    if isinstance(annotated, tuple):
        annotated = annotated[0]
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape[:2] == sample_image.shape[:2]


def test_output_directory_exists_after_run(pipeline, sample_image):
    """After running the pipeline, output/ directory tree should exist."""
    _ = pipeline.process_frame(sample_image)
    assert os.path.isdir("output"), "output/ directory should be created by the pipeline"
