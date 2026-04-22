"""
CDSS CLI — Interactive prescription analysis tool.

This script loads the pre-built vector store and runs an interactive
query loop. Run pipeline/build_db.py first to build the store.

Usage:
    python main.py
"""

import sys
from pathlib import Path
from preprocessing.parser import parse_prescription
from preprocessing.cleaner import clean_medications, build_embedding_text
from embeddings.embedding import get_embedding
from vector_db.store import VectorStore
from mapping.fuzzy_match import correct_drug_list
from mapping.condition_mapper import ConditionMapper

_PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    mapper = ConditionMapper()

    # Load pre-built store (run pipeline/build_db.py if this fails)
    store = VectorStore.load(str(_PROJECT_ROOT / "vector_store"))
    if store is None:
        print("ERROR: vector_store/ not found. Run: python pipeline/build_db.py")
        sys.exit(1)

    print(f"CDSS Online. Loaded {len(store.vectors)} cases. Type 'exit' to quit.\n")

    while True:
        print("=" * 50)
        user_input = input("Enter prescription (or 'exit'): ").strip()

        if user_input.lower() == "exit":
            print("Exiting CDSS.")
            break

        if not user_input:
            continue

        # Parse and correct
        parsed = parse_prescription(user_input)
        confirmed_drugs = correct_drug_list(parsed["drugs"])

        if not confirmed_drugs:
            print("  No drugs detected. Try a more specific prescription.\n")
            continue

        # Embed and search
        cleaned_meds = clean_medications(parsed["medications"])
        query_vector = get_embedding(build_embedding_text(confirmed_drugs, cleaned_meds))
        search_results = store.search(query_vector, top_k=3, threshold=0.60)

        # Predict
        predictions = mapper.predict(confirmed_drugs, vector_results=search_results)

        print("\nPREDICTED CONDITIONS:")
        if predictions:
            for pred in predictions[:5]:
                print(f"  * {pred['condition_label'].upper()} "
                      f"(score: {pred['confidence']}, source: {pred['source']})")
        else:
            print("  No strong matches found.")

        print("\nSIMILAR CASES:")
        if search_results:
            for meta, score in search_results:
                drugs_str = ", ".join(meta.get("drugs", []))
                print(f"  [{score:.2f}] {drugs_str}")
        else:
            print("  No similar cases found.")
        print()


if __name__ == "__main__":
    main()
