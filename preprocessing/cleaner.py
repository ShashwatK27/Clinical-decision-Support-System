
import re

# Common instruction words to remove
STOPWORDS = {
    "patient", "should", "take", "one", "two", "three", "four", "five",
    "for", "pain", "tablet", "tablets", "capsule", "capsules", "tabs",
    "and", "with", "the", "under", "over", "signature", "twice", "day"
}


MEDICAL_UNITS = {
    # Weights/Volumes
    "mg", "ml", "g", "mcg", "kg", "units", "iu", "percentage",
    # Forms
    "tablet", "tablets", "tab", "capsule", "capsules", "cap", 
    "syrup", "suspension", "ointment", "cream", "gel", "injection",
    "solution", "spray", "puffs", "inhaler", "drops",
    # Quantities
    "box", "pack", "strip", "bottle"
}

def clean_medications(meds):
    cleaned = []
    for med in meds:
        med = med.lower()

        # 1. Targeted Regex for numbers followed by units (e.g., "500mg", "10 ml")
        # This handles the space or lack of space between the number and unit
        unit_pattern = r'\b\d+\s*(' + '|'.join(MEDICAL_UNITS) + r')\b'
        med = re.sub(unit_pattern, '', med)

        # 2. Standalone unit names (e.g., "take 2 tablets")
        form_pattern = r'\b(' + '|'.join(MEDICAL_UNITS) + r')\b'
        med = re.sub(form_pattern, '', med)

        # 3. Existing number and punctuation cleaning
        med = re.sub(r'\b\d+\b', '', med)
        med = re.sub(r'[^\w\s]', '', med)

        # 4. Final filter
        words = med.split()
        words = [w for w in words if w not in STOPWORDS and len(w) > 2]
        
        cleaned.append(" ".join(words).strip())
    return cleaned


def build_embedding_text(drugs, cleaned_meds):
    """
    Combine drugs + cleaned medication info into strong embedding input
    """
    return " ".join(drugs) + " " + " ".join(cleaned_meds)

