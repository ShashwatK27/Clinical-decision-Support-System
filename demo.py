"""
demo.py — Quick end-to-end demo of the CDSS pipeline.

Loads the pre-built vector store and runs three sample prescriptions
through the complete Parse → Correct → Embed → Search → Predict flow.

Usage:
    python demo.py
"""

import sys
from pathlib import Path

from preprocessing.parser import parse_prescription
from preprocessing.cleaner import clean_medications, build_embedding_text
from mapping.fuzzy_match import correct_drug_list
from mapping.condition_mapper import ConditionMapper
from embeddings.embedding import get_embedding
from vector_db.store import VectorStore

_PROJECT_ROOT = Path(__file__).resolve().parent

DEMO_PRESCRIPTIONS = [
    "iboprofen 200mg and metaformin 500mg",
    "paracetamol 650mg for fever",
    "52y female with acid reflux. Prescribed omeprazole 20mg once daily and escitalopram 10mg.",
]

RECOMMENDATIONS = {
    "pain":           "Consider pain management strategies",
    "inflammation":   "Monitor for inflammatory signs",
    "diabetes":       "Monitor blood glucose levels",
    "fever":          "Monitor vital signs; consider antipyretics",
    "acid reflux":    "Verify PPI dosing; check for long-term safety",
    "gerd":           "Lifestyle modifications + pharmacotherapy",
    "depression":     "Follow up on mental health status",
    "anxiety":        "Monitor patient psychological status",
    "hypertension":   "Encourage lifestyle modification; monitor BP",
    "high cholesterol": "Dietary counselling + statin monitoring",
}


def run_demo():
    mapper = ConditionMapper()
    store = VectorStore.load(str(_PROJECT_ROOT / "vector_store"))

    if store is None:
        print("ERROR: Run 'python pipeline/build_db.py' first to build the vector store.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  CLINICAL DECISION SUPPORT SYSTEM - DEMO")
    print(f"  Vector store: {len(store.vectors)} cases loaded")
    print("=" * 70)

    for i, prescription in enumerate(DEMO_PRESCRIPTIONS, 1):
        print(f"\n{'─' * 70}")
        print(f"  PRESCRIPTION #{i}: '{prescription}'")
        print("─" * 70)

        # Step 1: Parse
        parsed = parse_prescription(prescription)
        corrected = correct_drug_list(parsed["drugs"])
        print(f"  [1] Drugs detected:   {corrected}")

        if not corrected:
            print("  No drugs found — skipping.\n")
            continue

        # Step 2: Embed + Search
        cleaned_meds = clean_medications(parsed["medications"])
        query_vector = get_embedding(build_embedding_text(corrected, cleaned_meds))
        results = store.search(query_vector, top_k=3, threshold=0.60)
        print(f"  [2] Similar cases:    {len(results)} found")

        # Step 3: Predict
        predictions = mapper.predict(corrected, vector_results=results)
        print(f"  [3] Conditions:")
        if predictions:
            for pred in predictions[:5]:
                print(f"        * {pred['condition_label'].upper()} "
                      f"(score: {pred['confidence']}, source: {pred['source']})")
        else:
            print("        No strong matches.")

        # Step 4: Recommendations
        print(f"  [4] Recommendations:")
        conditions = [p["condition_label"] for p in predictions[:3]]
        for cond in conditions:
            rec = RECOMMENDATIONS.get(cond, "Monitor patient status regularly")
            print(f"        * {cond}: {rec}")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_demo()
