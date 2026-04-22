import json
import os
from difflib import get_close_matches

from utils.logger_config import get_logger

logger = get_logger("fuzzy_match")

TYPO_ALIASES = {
    "iboprofen": "ibuprofen",
    "ibuprophen": "ibuprofen",
    "metaformin": "metformin",
    "ciproflaxacin": "ciprofloxacin",
    "ciprofolxacin": "ciprofloxacin",
}


def load_drugs():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base_dir, "data", "lexicons", "drugs.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            drugs_list = json.load(f)
            return set(drugs_list)
    except FileNotFoundError:
        logger.warning(
            f"Drug lexicon not found at '{path}'. "
            "Fuzzy matching will be disabled. Run scripts/build_lexicon.py to generate it."
        )
        return set()
    except Exception as e:
        logger.error(f"Failed to load drug lexicon: {e}")
        return set()


KNOWN_DRUGS = load_drugs()

BLACKLIST = {
    "pain", "for", "back", "daily", "start", "day", "take", "takes", "and",
    "before", "after", "with", "food", "meals", "directed", "needed",
    "morning", "evening", "night", "once", "twice", "times",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "a", "an", "of", "tablet", "tablets", "tab", "tabs", "capsule", "capsules",
    "patient", "female", "male", "woman", "man", "adult", "elderly",
    "presenting", "prescribed", "initiated", "diagnosed",
    "anxiety", "depression", "fever", "nausea", "vomiting", "infection",
    "inflammation", "reflux", "diabetes", "hypertension", "cholesterol",
    "asthma", "allergy", "allergies", "insomnia", "neuropathy", "seizure",
    "seizures", "disorder", "disease", "condition", "symptom", "symptoms",
    "the", "this", "that", "due", "from", "has", "have", "was", "were",
    "are", "all", "also", "may", "can", "not", "per", "use", "used",
    "add", "new", "both", "well", "good", "better",
    # Known false-positive fuzzy matches — common words that accidentally resemble drug names
    "metals", "aftera", "acid", "base", "salt", "mineral", "compound", "meal",
}



def correct_drug_list(drugs, cutoff=0.88):
    """Map to known drugs with fuzzy fallback for typos."""
    corrected = []
    logger.debug(f"Correcting drug list: {drugs}")

    if not KNOWN_DRUGS:
        logger.warning("KNOWN_DRUGS is empty - returning input drugs uncorrected.")
        return [d for d in drugs if d not in BLACKLIST]

    for drug in drugs:
        drug = drug.lower().strip()

        if drug in BLACKLIST:
            logger.debug(f"Skipping blacklisted drug: {drug}")
            continue

        alias_match = TYPO_ALIASES.get(drug)
        if alias_match:
            corrected.append(alias_match)
            logger.info(f"Alias match: '{drug}' -> '{alias_match}'")
            continue

        if drug in KNOWN_DRUGS:
            corrected.append(drug)
            logger.debug(f"Exact match found: {drug}")
            continue

        matches = get_close_matches(drug, KNOWN_DRUGS, n=1, cutoff=cutoff)
        if matches:
            best_match = TYPO_ALIASES.get(matches[0], matches[0])
            corrected.append(best_match)
            logger.info(f"Fuzzy match: '{drug}' -> '{best_match}' (cutoff={cutoff})")
        else:
            logger.warning(f"No match found for drug: {drug}")

    deduped = list(dict.fromkeys(corrected))
    logger.info(f"Corrected drugs: {deduped}")
    return deduped
