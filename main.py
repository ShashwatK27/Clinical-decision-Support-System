import os
import numpy as np
from datasets import load_from_disk
from preprocessing.parser import parse_prescription
from preprocessing.cleaner import clean_medications, build_embedding_text
from embeddings.embedding import get_embedding
from vector_db.store import VectorStore
from mapping.fuzzy_match import correct_drug_list
from mapping.condition_mapper import ConditionMapper

# 1. INITIALIZATION
store = VectorStore()
mapper = ConditionMapper()
try:
    dataset = load_from_disk("data/medical_dataset")
except:
    print("❌ Dataset not found. Please check your data/medical_dataset path.")
    exit()

# 2. BUILDING THE KNOWLEDGE BASE (Indexing)
print("⚡ Building Clinical Knowledge Base...")
# We index the first 500 samples for the demo
for sample in dataset["train"].select(range(500)):
    raw_text = sample["ground_truth"]
    
    # Process
    parsed = parse_prescription(raw_text)
    valid_drugs = correct_drug_list(parsed["drugs"])
    
    if not valid_drugs:
        continue

    cleaned_meds = clean_medications(parsed["medications"])
    embedding_input = build_embedding_text(valid_drugs, cleaned_meds)
    
    # Create Vector
    vector = get_embedding(embedding_input)
    
    # Store with Metadata
    metadata = {
        "drugs": valid_drugs,
        "original_text": raw_text
    }
    store.add(vector, metadata)

print(f"✅ Successfully indexed {len(store.vectors)} clinical cases.\n")

# 3. INTERACTIVE CDSS LOOP
print("🩺 CDSS Engine Online. Ready for prescription input.")
while True:
    print("\n" + "="*50)
    user_input = input("Enter Prescription Text (or 'exit'): ")

    if user_input.lower() == 'exit':
        print("Stopping CDSS...")
        break

    # Step A: Parsing & Lexicon Verification
    result = parse_prescription(user_input)
    confirmed_drugs = correct_drug_list(result["drugs"])

    # Step B: Semantic Search
    # We use the cleaned version for embedding to handle typos via math
    cleaned_meds = clean_medications(result["medications"])
    query_text = build_embedding_text(confirmed_drugs, cleaned_meds)
    query_vector = get_embedding(query_text)
    
    # Search for top 3 similar historical cases
    search_results = store.search(query_vector, top_k=3, threshold=0.60)

    # Step C: Hybrid Condition Prediction
    # Combines Rule-based (FDA) + Similarity-based (Vector)
    predictions = mapper.predict(confirmed_drugs, vector_results=search_results)

    # 4. DISPLAY RESULTS
    print("\n🧠 PREDICTED CLINICAL CONDITIONS:")
    if not predictions:
        print("   - Low confidence. No strong matches found.")
    else:
        for pred in predictions[:5]:
            print(f"   • {pred['condition_label'].upper()} (Score: {pred['confidence']})")

    print("\n🔍 SIMILAR HISTORICAL CASES:")
    if not search_results:
        print("   - No similar cases found in database.")
    else:
        for meta, score in search_results:
            # Show a clean summary of what the system found
            drugs_found = ", ".join(meta['drugs'])
            print(f"   [{score:.2f}] Drugs: {drugs_found}")