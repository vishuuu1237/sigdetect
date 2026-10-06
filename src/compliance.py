"""Compliance checking utilities for SignalGuard.

Provides `check_compliance` which validates Indian license plate numbers against
the official regex and performs a few heuristic checks for common violations.
"""

import re
from typing import Dict

# Regex for Indian vehicle registration plates
PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

def _normalize(text: str) -> str:
    """Normalize plate text: uppercase and strip spaces/hyphens."""
    return text.upper().replace(" ", "").replace("-", "")

def check_compliance(plate_text: str) -> Dict[str, object]:
    """Validate a license plate string against Indian standard format.

    Returns a dict with:
        - ``compliant`` (bool): True if the plate matches the official pattern.
        - ``reason``   (str) : Explanation when not compliant.
    """
    if not plate_text or not plate_text.strip():
        return {"compliant": False, "reason": "No plate detected / unreadable"}

    normalized = _normalize(plate_text)

    # Basic pattern check: e.g., MH12AB1234 or DL1CA1234 or HR26DQ5551
    if PLATE_REGEX.fullmatch(normalized):
        # Length check for standard Indian registration plates
        if not (8 <= len(normalized) <= 12):
            return {"compliant": False, "reason": "Non-standard character count"}
        if "?" in normalized or "~" in normalized:
            return {"compliant": False, "reason": "Plate text partially occluded or blurred"}
        return {"compliant": True, "reason": "Plate compliant (Standard HSRP format)"}
    else:
        # Check if partially valid or non-standard format
        if len(normalized) < 4:
            return {"compliant": False, "reason": "Incomplete plate number"}
        return {"compliant": False, "reason": "Invalid registration format (Non-HSRP / Fancy Font)"}

if __name__ == "__main__":
    # Simple test harness
    samples = [
        "KA01AB1234",
        "ka01ab1234",
        "KA-01-AB-1234",
        "KA01X1234",
        "KA01?B1234",
        "INVALID"
    ]
    for s in samples:
        print(f"{s!r} -> {check_compliance(s)}")
