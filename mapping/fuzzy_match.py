import json
import os
from difflib import get_close_matches
from utils.logger_config import get_logger

logger = get_logger("fuzzy_match")

def load_drugs():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base_dir, "data", "lexicons", "drugs.json")
    with open(path, 'r') as f:
        drugs_list = json.load(f)
        return set(drugs_list)

KNOWN_DRUGS = load_drugs()
BLACKLIST = {"pain", "for", "back", "daily", "start", "day", "take", "and"}

def correct_drug_list(drugs):
    """Map to known drugs with fuzzy fallback for typos."""
    corrected = []
    logger.debug(f"Correcting drug list: {drugs}")
    
    for drug in drugs:
        if drug in BLACKLIST:
            logger.debug(f"Skipping blacklisted drug: {drug}")
            continue
        if drug in KNOWN_DRUGS:
            corrected.append(drug)
            logger.debug(f"Exact match found: {drug}")
        else:
            # Fuzzy match with 85% threshold
            matches = get_close_matches(drug, KNOWN_DRUGS, n=1, cutoff=0.85)
            if matches:
                corrected.append(matches[0])
                logger.info(f"Fuzzy match: '{drug}' -> '{matches[0]}'")
            else:
                logger.warning(f"No match found for drug: {drug}")
    
    logger.info(f"Corrected drugs: {corrected}")
    return corrected
