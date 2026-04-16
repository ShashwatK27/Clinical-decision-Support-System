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

    if not knowledge_base:
        knowledge_base = {
            "ibuprofen": ["pain", "inflammation"],
            "paracetamol": ["fever", "pain"],
            "acetaminophen": ["fever", "pain"],
            "metformin": ["diabetes"],
            "ciprofloxacin": ["bacterial infection"],
            "ciproflaxacin": ["bacterial infection"],
            "amoxicillin": ["bacterial infection"],
            "atorvastatin": ["high cholesterol"],
            "gabapentin": ["neuropathy", "seizures"],
            "lisinopril": ["hypertension"],
            "levothyroxine": ["thyroid disorder"],
            "esomeprazole": ["acid reflux", "gerd"],
            "aspirin": ["pain", "heart disease prevention"],
            "prednisone": ["inflammation", "autoimmune disease"],
            "loratadine": ["allergic rhinitis", "urticaria", "allergies"],
            "cetirizine": ["allergic rhinitis", "urticaria", "allergies"],
            "fexofenadine": ["allergic rhinitis", "urticaria", "allergies"],
            "diphenhydramine": ["allergies", "insomnia", "urticaria"],
            "omeprazole": ["acid reflux", "gerd", "ulcer prevention"],
            "famotidine": ["acid reflux", "ulcer prevention"],
            "simvastatin": ["high cholesterol"],
            "losartan": ["hypertension"],
            "amlodipine": ["hypertension", "angina"],
            "hydrochlorothiazide": ["hypertension"],
        }

    logger.info(f"Loaded drug KB with {len(knowledge_base)} entries")
    return knowledge_base

DRUG_TO_CONDITION = load_drug_kb()

class ConditionMapper:
    """
    Handles the transformation of drug names into clinical conditions.
    Using direct medical condition mapping (not pharmacological classes).
    """

    def __init__(self):
        self.knowledge_base = DRUG_TO_CONDITION

    def predict(self, user_drugs, vector_results=None):
        """
        Hybrid prediction logic:
        1. Rule-Based: Uses exact matches from the FDA knowledge base.
        2. Vector-Based: Incorporates labels from semantically similar historical cases.
        3. Noise Filtering: Removes common medical 'false positives' like allergens.
        """
        logger.info(f"Predicting conditions for drugs: {user_drugs}")
        condition_scores = {}
        
        # Define terms that are almost always false positives in a general prescription context
        excluded_keywords = ["allergenic extract", "food allergen", "plant allergen"]

        # 1. Rule-Based Scoring (Primary Signal)
        # Weight: 2.0 (High confidence because it's a direct FDA link)
        for drug in user_drugs:
            if drug in self.knowledge_base:
                for condition in self.knowledge_base[drug]:
                    # Filter out noise like 'Standardized Grass Pollen Allergenic Extract'
                    if any(noise in condition.lower() for noise in excluded_keywords):
                        continue

                    condition_scores[condition] = condition_scores.get(condition, 0) + 2.0
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
                        if any(noise in cond.lower() for noise in excluded_keywords):
                            continue
                        
                        # Increment score by the similarity (e.g., +0.85)
                        condition_scores[cond] = condition_scores.get(cond, 0) + score
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
                results.append({
                    "condition_label": cond,
                    "confidence": round(score, 2)
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