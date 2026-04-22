import json
from pathlib import Path
from utils.logger_config import get_logger

logger = get_logger("condition_mapper")

# Load drug-to-condition knowledge from disk, falling back to a small default set.
# This allows the mapper to use large KB files and still provide fallback labels.

def load_drug_kb():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_kb_path = base_dir / "data" / "medical_dataset" / "drug_kb.json"
    labeled_kb_path = base_dir / "data" / "lexicons" / "labeled_drugs.json"

    knowledge_base = {}

    if dataset_kb_path.exists():
        try:
            with open(dataset_kb_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for drug, info in raw.items():
                conditions = info.get("conditions", [])
                if conditions:
                    knowledge_base[drug] = conditions
        except Exception as e:
            logger.warning(f"Unable to load dataset drug KB: {e}")

    if labeled_kb_path.exists():
        try:
            with open(labeled_kb_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for drug, labels in raw.items():
                if drug not in knowledge_base and labels:
                    knowledge_base[drug] = labels
        except Exception as e:
            logger.warning(f"Unable to load labeled drug KB: {e}")

    # --- Clean seed KB -------------------------------------------------
    # These are curated clinical condition labels for the most common drugs.
    # They are ALWAYS merged in because the FDA/labeled KBs only contain
    # pharmacological class labels (e.g. "hmg-coa reductase inhibitor")
    # which get noise-filtered at predict time. Without this seed, common
    # drugs like atorvastatin and lisinopril would produce no predictions.
    CLEAN_SEED = {
        "ibuprofen":           ["pain", "inflammation"],
        "naproxen":            ["pain", "inflammation"],
        "diclofenac":          ["pain", "inflammation"],
        "paracetamol":         ["fever", "pain"],
        "acetaminophen":       ["fever", "pain"],
        "aspirin":             ["pain", "heart disease prevention"],
        "metformin":           ["diabetes"],
        "glipizide":           ["diabetes"],
        "sitagliptin":         ["diabetes"],
        "insulin glargine":    ["diabetes"],
        "lisinopril":          ["hypertension"],
        "losartan":            ["hypertension"],
        "amlodipine":          ["hypertension", "angina"],
        "atenolol":            ["hypertension"],
        "hydrochlorothiazide": ["hypertension"],
        "metoprolol":          ["hypertension", "heart failure"],
        "carvedilol":          ["heart failure", "hypertension"],
        "furosemide":          ["edema", "heart failure"],
        "atorvastatin":        ["high cholesterol"],
        "simvastatin":         ["high cholesterol"],
        "rosuvastatin":        ["high cholesterol"],
        "warfarin":            ["blood clot prevention"],
        "clopidogrel":         ["heart disease prevention"],
        "amoxicillin":         ["bacterial infection"],
        "ciprofloxacin":       ["bacterial infection"],
        "ciproflaxacin":       ["bacterial infection"],
        "azithromycin":        ["bacterial infection"],
        "doxycycline":         ["bacterial infection"],
        "gabapentin":          ["neuropathy", "seizures"],
        "pregabalin":          ["neuropathy"],
        "levothyroxine":       ["thyroid disorder"],
        "esomeprazole":        ["acid reflux", "gerd"],
        "omeprazole":          ["acid reflux", "gerd"],
        "pantoprazole":        ["acid reflux", "gerd"],
        "famotidine":          ["acid reflux", "ulcer prevention"],
        "prednisone":          ["inflammation", "autoimmune disease"],
        "methotrexate":        ["autoimmune disease", "rheumatoid arthritis"],
        "hydroxychloroquine":  ["autoimmune disease", "rheumatoid arthritis"],
        "loratadine":          ["allergic rhinitis", "allergies"],
        "cetirizine":          ["allergic rhinitis", "allergies"],
        "fexofenadine":        ["allergic rhinitis", "allergies"],
        "diphenhydramine":     ["allergies", "insomnia"],
        "montelukast":         ["asthma", "allergic rhinitis"],
        "salbutamol":          ["asthma"],
        "budesonide":          ["asthma"],
        "sertraline":          ["depression", "anxiety"],
        "fluoxetine":          ["depression"],
        "escitalopram":        ["depression", "anxiety"],
        "alprazolam":          ["anxiety"],
        "zolpidem":            ["insomnia"],
        "allopurinol":         ["gout"],
        "colchicine":          ["gout"],
    }

    # Merge: clean seed wins for drugs it knows about; FDA KB fills the rest
    for drug, conditions in CLEAN_SEED.items():
        knowledge_base[drug] = conditions   # always overwrite with clean label

    logger.info(f"Loaded drug KB with {len(knowledge_base)} entries "
                f"({len(CLEAN_SEED)} clean-seed + {len(knowledge_base) - len(CLEAN_SEED)} FDA entries)")
    return knowledge_base



class ConditionMapper:
    """
    Handles the transformation of drug names into clinical conditions.
    Using direct medical condition mapping (not pharmacological classes).
    """

    def __init__(self):
        # Load KB in __init__ rather than at module level so that:
        # 1. File I/O only happens when the class is actually instantiated.
        # 2. Tests can easily patch load_drug_kb() without monkeypatching a global.
        self.knowledge_base = load_drug_kb()

    def predict(self, user_drugs, vector_results=None):
        """
        Hybrid prediction logic:
        1. Rule-Based: Uses exact matches from the FDA knowledge base.
        2. Vector-Based: Incorporates labels from semantically similar historical cases.
        3. Noise Filtering: Removes common medical 'false positives' like allergens.

        Each result dict includes a 'source' field:
            'rule-based' — confidence came only from the KB
            'vector'     — confidence came only from vector similarity
            'both'       — condition was reinforced by both sources
        """
        logger.info(f"Predicting conditions for drugs: {user_drugs}")
        condition_scores = {}
        # Track which conditions come from which source
        rule_conditions: set = set()
        vector_conditions: set = set()

        # Pharmacological class labels that leak in from the FDA drug KB.
        # These are drug-mechanism labels, not clinical conditions a clinician
        # would act on. Filter them out so only clean condition terms reach the UI.
        excluded_keywords = [
            # Existing allergen noise
            "allergenic extract", "food allergen", "plant allergen",
            # Pharmacological class / mechanism labels
            "inhibitor", "agonist", "antagonist", "blocker", "receptor",
            "enzyme", "cytochrome", "transporter", "substrate", "inducer",
            "reuptake", "channel", "modulator", "pump", "synthase",
            "kinase", "hormone replacement", "pharmacological class",
        ]

        def _is_noise(label: str) -> bool:
            """Return True if label looks like a pharmacological class, not a condition."""
            low = label.lower()
            return any(kw in low for kw in excluded_keywords)

        # 1. Rule-Based Scoring (Primary Signal)
        # Weight: 2.0 (High confidence because it's a direct FDA link)
        for drug in user_drugs:
            if drug in self.knowledge_base:
                for condition in self.knowledge_base[drug]:
                    if _is_noise(condition):
                        continue

                    condition_scores[condition] = condition_scores.get(condition, 0) + 2.0
                    rule_conditions.add(condition)
                logger.debug(f"Found {len(self.knowledge_base.get(drug, []))} conditions for drug '{drug}'")
            else:
                logger.debug(f"No condition mapping for drug '{drug}'. Relying on vector-based context only.")

        # 2. Vector-Based Scoring (Contextual Signal)
        # Weight: Variable (0.6 - 0.9 depending on similarity score)
        if vector_results:
            logger.info(f"Processing {len(vector_results)} vector search results")
            for meta, score in vector_results:
                # We assume metadata contains 'conditions' from the indexed dataset
                if isinstance(meta, dict) and 'conditions' in meta:
                    for cond in meta['conditions']:
                        if _is_noise(cond):
                            continue

                        # Increment score by the similarity (e.g., +0.85)
                        condition_scores[cond] = condition_scores.get(cond, 0) + score
                        vector_conditions.add(cond)
                    logger.debug(f"Added vector conditions: {meta['conditions']}")

        # 3. Final Filtering & Ranking
        if not condition_scores:
            logger.warning(f"No conditions found for drugs: {user_drugs}")
            return []

        # Find the highest score to use for adaptive filtering
        max_score = max(condition_scores.values())
        results = []

        for cond, score in condition_scores.items():
            # Skip drug names that were used as fallback conditions (lower confidence)
            if score == 1.0 and cond in user_drugs:
                continue

            # Adaptive Threshold: Only keep conditions that are at least 40%
            # as likely as the most confident result (lowered from 50% for better coverage)
            if score >= (max_score * 0.4):
                # Determine source explicitly (avoids fragile float == comparisons in the UI)
                in_rule = cond in rule_conditions
                in_vector = cond in vector_conditions
                if in_rule and in_vector:
                    source = "both"
                elif in_rule:
                    source = "rule-based"
                else:
                    source = "vector"

                results.append({
                    "condition_label": cond,
                    "confidence": round(score, 2),
                    "source": source,
                })

        # Sort results: Highest confidence first
        results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        logger.info(f"Predicted {len(results)} conditions: {[r['condition_label'] for r in results]}")
        return results


if __name__ == "__main__":
    # Quick debug test
    mapper = ConditionMapper()
    test_drugs = ["ibuprofen", "esomeprazole"]
    print(f"Testing mapper with: {test_drugs}")
    print(json.dumps(mapper.predict(test_drugs), indent=2))