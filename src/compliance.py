"""License Plate Government Compliance Checker for SignalGuard.

This module validates recognized license plates against official government
standards (e.g., standard Indian HSRP format: State Code [2 chars] +
District Code [2 digits] + Series [1-3 letters] + Number [4 digits]).
"""

import re

def check_plate_compliance(plate_text: str) -> dict:
    """Validate license plate string format against government standards.

    Args:
        plate_text: Cleaned alphanumeric plate string.

    Returns:
        dict: Compliance verdict, formatted plate, and reason if non-compliant.
    """
    cleaned = re.sub(r'[^A-Z0-9]', '', plate_text.upper().strip())
    # Standard format: 2 letters (state), 2 digits (rto), optional 1-3 letters, 4 digits
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{0,3}[0-9]{4}$'
    is_valid = bool(re.match(pattern, cleaned))
    
    return {
        "raw_text": plate_text,
        "cleaned_text": cleaned,
        "is_compliant": is_valid,
        "reason": "Valid format" if is_valid else "Violates standard registration format"
    }

if __name__ == "__main__":
    sample_plate = "DL01AB1234"
    result = check_plate_compliance(sample_plate)
    print(f"Sample test: {sample_plate} -> {result}")
