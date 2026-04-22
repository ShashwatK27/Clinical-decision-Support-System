"""
dosage_validator.py — Safe dosage range validation for common drugs.

Extracts dosage information from prescription text and checks it against
curated safe single-dose and daily-dose limits.

Usage:
    from mapping.dosage_validator import extract_dosages, validate_dosages
    dosages = extract_dosages("ibuprofen 400mg twice daily and metformin 5000mg")
    warnings = validate_dosages(dosages)
"""

from __future__ import annotations
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Safe dose reference table
# All values in mg unless unit is specified explicitly.
# Sources: BNF / standard pharmacology references.
# ---------------------------------------------------------------------------
_DOSE_RANGES: dict[str, dict] = {
    # NSAIDs / Analgesics
    "ibuprofen":       {"max_single": 800,    "max_daily": 3200,  "unit": "mg",  "note": "Max 3200 mg/day; 400 mg/dose OTC"},
    "naproxen":        {"max_single": 1000,   "max_daily": 1500,  "unit": "mg",  "note": "Max 1500 mg/day"},
    "diclofenac":      {"max_single": 150,    "max_daily": 150,   "unit": "mg",  "note": "Max 150 mg/day"},
    "aspirin":         {"max_single": 1000,   "max_daily": 4000,  "unit": "mg",  "note": "75–100 mg for antiplatelet; up to 4 g/day for pain"},
    "paracetamol":     {"max_single": 1000,   "max_daily": 4000,  "unit": "mg",  "note": "Max 4 g/day; 2 g/day in liver disease"},
    "acetaminophen":   {"max_single": 1000,   "max_daily": 4000,  "unit": "mg",  "note": "Max 4 g/day"},

    # Diabetes
    "metformin":       {"max_single": 1000,   "max_daily": 3000,  "unit": "mg",  "note": "Max 3000 mg/day; titrate slowly"},
    "glipizide":       {"max_single": 20,     "max_daily": 40,    "unit": "mg",  "note": "Max 40 mg/day"},
    "sitagliptin":     {"max_single": 100,    "max_daily": 100,   "unit": "mg",  "note": "100 mg once daily max"},

    # Hypertension / Cardiovascular
    "lisinopril":      {"max_single": 40,     "max_daily": 40,    "unit": "mg",  "note": "Max 40 mg/day"},
    "losartan":        {"max_single": 100,    "max_daily": 100,   "unit": "mg",  "note": "Max 100 mg/day"},
    "amlodipine":      {"max_single": 10,     "max_daily": 10,    "unit": "mg",  "note": "Max 10 mg/day"},
    "atenolol":        {"max_single": 100,    "max_daily": 100,   "unit": "mg",  "note": "Max 100 mg/day"},
    "hydrochlorothiazide": {"max_single": 50, "max_daily": 50,    "unit": "mg",  "note": "Max 50 mg/day"},
    "metoprolol":      {"max_single": 200,    "max_daily": 400,   "unit": "mg",  "note": "Max 400 mg/day"},
    "furosemide":      {"max_single": 80,     "max_daily": 600,   "unit": "mg",  "note": "Max 600 mg/day in severe oedema"},
    "warfarin":        {"max_single": 10,     "max_daily": 10,    "unit": "mg",  "note": "Highly variable; INR-guided dosing"},

    # Statins
    "atorvastatin":    {"max_single": 80,     "max_daily": 80,    "unit": "mg",  "note": "Max 80 mg/day"},
    "simvastatin":     {"max_single": 40,     "max_daily": 40,    "unit": "mg",  "note": "Max 40 mg/day (20 mg with amlodipine)"},
    "rosuvastatin":    {"max_single": 40,     "max_daily": 40,    "unit": "mg",  "note": "Max 40 mg/day"},

    # Antibiotics
    "amoxicillin":     {"max_single": 1000,   "max_daily": 3000,  "unit": "mg",  "note": "Max 3 g/day standard; up to 6 g/day high-dose"},
    "ciprofloxacin":   {"max_single": 750,    "max_daily": 1500,  "unit": "mg",  "note": "Max 1500 mg/day oral"},
    "azithromycin":    {"max_single": 500,    "max_daily": 500,   "unit": "mg",  "note": "Max 500 mg/day"},
    "doxycycline":     {"max_single": 200,    "max_daily": 200,   "unit": "mg",  "note": "Max 200 mg/day"},

    # Neurology / Psychiatry
    "gabapentin":      {"max_single": 1200,   "max_daily": 3600,  "unit": "mg",  "note": "Max 3600 mg/day"},
    "pregabalin":      {"max_single": 300,    "max_daily": 600,   "unit": "mg",  "note": "Max 600 mg/day"},
    "sertraline":      {"max_single": 200,    "max_daily": 200,   "unit": "mg",  "note": "Max 200 mg/day"},
    "fluoxetine":      {"max_single": 60,     "max_daily": 60,    "unit": "mg",  "note": "Max 60 mg/day (80 mg for bulimia)"},
    "escitalopram":    {"max_single": 20,     "max_daily": 20,    "unit": "mg",  "note": "Max 20 mg/day (10 mg in elderly)"},
    "alprazolam":      {"max_single": 1,      "max_daily": 4,     "unit": "mg",  "note": "Max 4 mg/day; taper on discontinuation"},
    "zolpidem":        {"max_single": 10,     "max_daily": 10,    "unit": "mg",  "note": "Max 10 mg/day; short-term use only"},

    # GI / Thyroid
    "omeprazole":      {"max_single": 40,     "max_daily": 80,    "unit": "mg",  "note": "Max 80 mg/day (Zollinger-Ellison up to 120 mg)"},
    "esomeprazole":    {"max_single": 40,     "max_daily": 80,    "unit": "mg",  "note": "Max 80 mg/day"},
    "pantoprazole":    {"max_single": 40,     "max_daily": 80,    "unit": "mg",  "note": "Max 80 mg/day"},
    "levothyroxine":   {"max_single": 200,    "max_daily": 200,   "unit": "mcg", "note": "Highly individual; TSH-guided"},

    # Immunology / Rheumatology
    "prednisone":      {"max_single": 80,     "max_daily": 80,    "unit": "mg",  "note": "Highly variable; taper gradually"},
    "methotrexate":    {"max_single": 30,     "max_daily": 30,    "unit": "mg",  "note": "Weekly dosing; max 30 mg/week for RA"},
    "colchicine":      {"max_single": 1,      "max_daily": 2,     "unit": "mg",  "note": "Max 2 mg/day for gout prevention"},
    "allopurinol":     {"max_single": 300,    "max_daily": 800,   "unit": "mg",  "note": "Max 800 mg/day"},
}

# Unit conversion multipliers to mg
_UNIT_TO_MG = {
    "mg": 1.0,
    "g":  1000.0,
    "mcg": 0.001,
    "ug":  0.001,
    "units": None,  # can't convert
    "ml":    None,
}

# Regex: optional drug name prefix + dose + unit
_DOSE_PATTERN = re.compile(
    r"(?P<drug>[a-z][a-z\-]+)\s+(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|ug|g\b)",
    re.IGNORECASE,
)


@dataclass
class DosageWarning:
    drug: str
    dose_value: float
    dose_unit: str
    limit_value: float
    limit_type: str   # "single-dose" | "daily-dose"
    note: str
    severity: str     # "high" | "caution"


def extract_dosages(text: str) -> dict[str, tuple[float, str]]:
    """
    Extract drug → (dose_value, unit) pairs from free-text prescription.

    Only the first match per drug name is returned. The drug name from the
    regex is a raw token — callers should fuzzy-correct the key if needed.

    Returns:
        dict mapping lowercase drug token → (float dose, str unit)
    """
    result: dict[str, tuple[float, str]] = {}
    for m in _DOSE_PATTERN.finditer(text.lower()):
        drug = m.group("drug").strip()
        val  = float(m.group("value"))
        unit = m.group("unit").lower()
        if drug not in result:
            result[drug] = (val, unit)
    return result


def validate_dosages(
    dosages: dict[str, tuple[float, str]],
    corrected_drugs: list[str],
) -> list[DosageWarning]:
    """
    Validate extracted dosages against the safe-range table.

    Args:
        dosages:        Output of extract_dosages() — {raw_drug: (value, unit)}
        corrected_drugs: Fuzzy-corrected drug list (used to map raw → canonical)

    Returns:
        List of DosageWarning objects, empty if everything is within range.
    """
    warnings: list[DosageWarning] = []

    # Build a reverse map: partial token → canonical drug name
    # (handles "metformin" in text matching "metformin" in corrected list)
    token_to_canonical: dict[str, str] = {}
    for canonical in corrected_drugs:
        token_to_canonical[canonical.lower()] = canonical
        # Also allow the first 5 chars as prefix match
        if len(canonical) >= 5:
            token_to_canonical[canonical[:5].lower()] = canonical

    for raw_drug, (value, unit) in dosages.items():
        # Try to resolve raw token to a canonical drug name
        canonical = token_to_canonical.get(raw_drug)
        if canonical is None:
            # Try prefix matching
            for tok, can in token_to_canonical.items():
                if raw_drug.startswith(tok) or tok.startswith(raw_drug):
                    canonical = can
                    break

        if canonical is None:
            continue  # Can't validate unknown drug

        ref = _DOSE_RANGES.get(canonical.lower())
        if ref is None:
            continue  # No reference data — can't validate

        ref_unit = ref["unit"]

        # Convert dose to mg for comparison if units match
        multiplier = _UNIT_TO_MG.get(unit)
        ref_multiplier = _UNIT_TO_MG.get(ref_unit, 1.0)

        if multiplier is None or ref_multiplier is None:
            continue  # Incomparable units (e.g. ml vs mg)

        dose_mg = value * multiplier
        max_single_mg = ref["max_single"] * ref_multiplier
        max_daily_mg  = ref["max_daily"]  * ref_multiplier

        # Check single-dose limit first
        if dose_mg > max_daily_mg * 1.5:
            # More than 1.5× the daily max — HIGH severity
            warnings.append(DosageWarning(
                drug=canonical,
                dose_value=value,
                dose_unit=unit,
                limit_value=ref["max_daily"],
                limit_type="daily-dose",
                note=ref["note"],
                severity="high",
            ))
        elif dose_mg > max_single_mg:
            # Exceeds single-dose max — CAUTION
            warnings.append(DosageWarning(
                drug=canonical,
                dose_value=value,
                dose_unit=unit,
                limit_value=ref["max_single"],
                limit_type="single-dose",
                note=ref["note"],
                severity="caution",
            ))

    return warnings
