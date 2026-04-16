"""
Interactive demo of the fixed CDSS system
Shows the complete workflow with realistic prescription examples
"""

from preprocessing.parser import parse_prescription
from preprocessing.cleaner import clean_medications, build_embedding_text
from mapping.fuzzy_match import correct_drug_list
from mapping.condition_mapper import ConditionMapper
from embeddings.embedding import get_embedding
from vector_db.store import VectorStore
import numpy as np

print("\n" + "=" * 70)
print("🩺 CLINICAL DECISION SUPPORT SYSTEM - DEMO")
print("=" * 70)

# Initialize
store = VectorStore()
mapper = ConditionMapper()

print("\n⚡ Building clinical knowledge base...")
# Pre-populate with a few example cases
example_cases = [
    (["ibuprofen"], ["pain", "inflammation"]),
    (["metformin"], ["diabetes"]),
    (["paracetamol"], ["fever"]),
]

for drugs, conditions in example_cases:
    text = " ".join(drugs + conditions)
    vector = get_embedding(text)
    metadata = {"drugs": drugs, "conditions": conditions, "original_text": text}
    store.add(vector, metadata)

print(f"✅ Knowledge base ready with {len(store.vectors)} cases\n")

# Demo prescriptions
demo_prescriptions = [
    "iboprofen 200mg and metaformin 500mg",
    "paracetamol 650 mg",
    "Patient should take ibuprofen daily for pain",
]

for i, prescription in enumerate(demo_prescriptions, 1):
    print("\n" + "-" * 70)
    print(f"📋 PRESCRIPTION #{i}")
    print("-" * 70)
    print(f"Input: '{prescription}'")
    
    # Step 1: Parse
    result = parse_prescription(prescription)
    print(f"\n1️⃣  PARSING:")
    print(f"   Raw drugs extracted: {result['drugs']}")
    print(f"   Medications: {result['medications']}")
    
    # Step 2: Fuzzy correction
    corrected_drugs = correct_drug_list(result["drugs"])
    print(f"\n2️⃣  FUZZY MATCHING & CORRECTION:")
    print(f"   → Corrected drugs: {corrected_drugs}")
    
    # Step 3: Embedding & search
    cleaned_meds = clean_medications(result["medications"])
    query_text = build_embedding_text(corrected_drugs, cleaned_meds)
    query_vector = get_embedding(query_text)
    search_results = store.search(query_vector, top_k=3, threshold=0.60)
    
    print(f"\n3️⃣  SEMANTIC SEARCH:")
    print(f"   Similar cases found: {len(search_results)}")
    for meta, score in search_results[:2]:
        print(f"      • {meta['drugs']} (similarity: {score:.2f})")
    
    # Step 4: Condition prediction
    predictions = mapper.predict(corrected_drugs, vector_results=search_results)
    
    print(f"\n4️⃣  CONDITION PREDICTION:")
    if predictions:
        for pred in predictions[:5]:
            print(f"   • {pred['condition_label'].upper()} (confidence: {pred['confidence']})")
    else:
        print("   - Low confidence. No strong matches found.")
    
    print(f"\n5️⃣  CLINICAL DECISION SUPPORT:")
    if predictions:
        conditions = [p["condition_label"] for p in predictions]
        recommendations = {
            "pain": "Consider pain management strategies",
            "inflammation": "Monitor for inflammatory signs",
            "diabetes": "Monitor blood glucose levels",
            "fever": "Monitor vital signs; consider antipyretics",
        }
        
        print("   Recommended actions:")
        for cond in conditions[:3]:
            rec = recommendations.get(cond, "Monitor patient status")
            print(f"      ✓ {cond.upper()}: {rec}")
    
    print()

print("\n" + "=" * 70)
print("✅ DEMO COMPLETE")
print("=" * 70)
print("\nKey improvements demonstrated:")
print("  ✅ Typo tolerance: 'iboprofen' → 'ibuprofen'")
print("  ✅ Clinical conditions: 'pain', 'diabetes' (not pharmacological classes)")
print("  ✅ Multi-drug reasoning: combines predictions from multiple drugs")
print("  ✅ No blocking: even with typos, inference proceeds")
print("  ✅ Confidence scoring: only high-confidence predictions shown")
print("\n")
