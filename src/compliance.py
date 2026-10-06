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
    """Validate a license plate string.

    Returns a dict with:
        - ``compliant`` (bool): True if the plate matches the official pattern.
        - ``reason``   (str) : Explanation when not compliant.
    Additional heuristic violations are flagged in the reason string.
    """
    normalized = _normalize(plate_text)
    # Basic pattern check
    if PLATE_REGEX.fullmatch(normalized):
        # Heuristic checks – length should be between 10 and 13 characters for Indian plates
        if not (9 <= len(normalized) <= 13):
            return {"compliant": False, "reason": "Wrong length – possible wrong font or malformed plate"}
        # Placeholder heuristics: treat 'X' as missing HSRP, '?' as blurred
        if "X" in normalized:
            return {"compliant": False, "reason": "Missing HSRP sticker"}
        if "?" in normalized:
            return {"compliant": False, "reason": "Plate appears blurred"}
        return {"compliant": True, "reason": "Plate compliant"}
    else:
        return {"compliant": False, "reason": "Regex mismatch – invalid Indian plate format"}

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
