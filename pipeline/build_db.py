
import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_from_disk
from preprocessing.parser import parse_prescription
from preprocessing.cleaner import clean_medications, build_embedding_text
from embeddings.embedding import get_embedding
from vector_db.store import VectorStore
from mapping.fuzzy_match import correct_drug_list
from mapping.condition_mapper import ConditionMapper
from collections import Counter

# -------------------------------
# LOAD DATASET
# -------------------------------
dataset = load_from_disk("data/medical_dataset")

store = VectorStore()
mapper = ConditionMapper()

print("** Building vector database...\n")


# -------------------------------
# BUILD VECTOR DATABASE (Updated)
# -------------------------------
for sample in dataset["train"].select(range(500)):
    raw_text = sample["ground_truth"]
    parsed = parse_prescription(raw_text)
    valid_drugs = correct_drug_list(parsed["drugs"])
    
    if not valid_drugs:
        continue

    # NEW: Pre-calculate the conditions for this case using the mapper
    # This "hydrates" the metadata so the search function can find it later
    case_conditions = []
    for drug in valid_drugs:
        if drug in mapper.knowledge_base:
            case_conditions.extend(mapper.knowledge_base[drug])

    cleaned_meds = clean_medications(parsed["medications"])
    embedding_input = build_embedding_text(valid_drugs, cleaned_meds)
    vector = get_embedding(embedding_input)
    
    # Updated Metadata
    metadata = {
        "drugs": valid_drugs,
        "conditions": list(set(case_conditions)), # CRITICAL: Store the labels here!
        "original_text": raw_text
    }
    store.add(vector, metadata)

print("** Vector DB ready!\n")

# Save the vector store to disk for use in the web app
store_path = store.save("vector_store.pkl")
print(f"** Vector store saved to: {store_path}")
print(f"   - {len(store.vectors)} vectors")
print(f"   - {len(store.metadata)} metadata entries\n")


# -------------------------------
# INTERACTIVE CDSS
# -------------------------------
while True:
    print("\n-----------------------------------")
    user_input = input("Enter prescription (or 'exit'): ")

    if user_input.lower() == "exit":
        print("👋 Exiting CDSS...")
        break

    # 🔹 Parse user input
    result = parse_prescription(user_input)

    # 🔥 Fuzzy correction
    result["drugs"] = correct_drug_list(result["drugs"])

    if not result["drugs"]:
        print("⚠️ No drugs detected!")
        continue

    # 🔹 Clean + prepare text
    cleaned_meds = clean_medications(result["medications"])
    query_text = build_embedding_text(result["drugs"], cleaned_meds)

    # 🔹 Embedding
    query_vector = get_embedding(query_text)

    # 🔹 Search similar cases
    results = store.search(query_vector)

    # -------------------------------
    # CONDITION PREDICTION
    predictions = mapper.predict(result["drugs"], vector_results=results)

    print("\n🧠 Predicted conditions:")
    if not predictions:
        print("- Low confidence. No strong matches found.")
    else:
        for item in predictions[:5]:
            print(f"- {item['condition_label']} (confidence: {item['confidence']})")

    # -------------------------------
    # DISPLAY RESULTS
    # -------------------------------
    print("\n🔍 Similar Cases:\n")

    for meta, score in results:
        if score < 0.65:
            continue

        if isinstance(meta, dict):
            drugs_found = ", ".join(meta.get("drugs", []))
            print(f"Case Drugs: {drugs_found}")
            print(f"Original text: {meta.get('original_text', '')}")
        else:
            print(f"Case metadata: {meta}")

        print(f"Similarity: {score:.3f}\n")
