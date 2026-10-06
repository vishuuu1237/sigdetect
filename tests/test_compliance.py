"""Tests for src/compliance.py – Indian plate validation."""

import sys
import os
import pytest

# Ensure project root is on sys.path so bare `from compliance import ...` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compliance import check_compliance, _normalize


# ── valid plates ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("plate", [
    "KA01AB1234",
    "MH12DE1234",
    "DL1CAB1234",
    "TN09Z1234",
    "AP05AB1234",
])
def test_valid_plates(plate):
    result = check_compliance(plate)
    assert result["compliant"] is True, f"Expected compliant for {plate!r}, got {result}"


# ── case / punctuation normalisation ─────────────────────────────────────────

@pytest.mark.parametrize("plate", [
    "ka01ab1234",      # lowercase
    "KA-01-AB-1234",   # hyphens
    "KA 01 AB 1234",   # spaces
    "ka-01-ab-1234",   # both
])
def test_normalised_plates_are_valid(plate):
    result = check_compliance(plate)
    assert result["compliant"] is True, f"Expected compliant for {plate!r}, got {result}"


# ── invalid plates ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("plate,reason_substr", [
    ("INVALID",          "Regex mismatch"),
    ("12AB1234",         "Regex mismatch"),
    ("KA01",             "Regex mismatch"),
    ("",                 "Regex mismatch"),
    ("KA01ABCD12345",    "Regex mismatch"),
])
def test_invalid_plates(plate, reason_substr):
    result = check_compliance(plate)
    assert result["compliant"] is False
    assert reason_substr.lower() in result["reason"].lower(), (
        f"Expected reason containing {reason_substr!r}, got {result['reason']!r}"
    )


# ── normalisation helper ─────────────────────────────────────────────────────

def test_normalize_strips_and_uppercases():
    assert _normalize("ka-01 ab-1234") == "KA01AB1234"
    assert _normalize("  MH 12 DE 1234  ") == "MH12DE1234"
