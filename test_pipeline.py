from preprocessing.parser import parse_prescription
from mapping.fuzzy_match import correct_drug_list

def run_smoke_test():
    # 1. Define "Dirty" Test Cases
    test_cases = [
        # Case A: Structured OCR with noise
        "<s_ocr> medications: - Amoxicillin 500mg capsules - Metformin 850 mg tabs signature: </s>",
        
        # Case B: Free text with a typo (Ibuprophen)
        "Patient should take one tablet of Ibuprophen daily for pain",
        
        # Case C: Multiple drugs with units and forms
        "Ciproflaxacin 250mg 2 puffs twice a day and Esomeprazole 20mg"
    ]

    print("🚀 Starting CDSS Smoke Test...\n")

    for i, raw_text in enumerate(test_cases):
        print(f"--- Test Case {i+1} ---")
        print(f"Input: {raw_text}")

        # Step 1: Parse & Clean
        parsed = parse_prescription(raw_text)
        raw_drugs = parsed['drugs']
        
        # Step 2: Fuzzy Match (using your new FDA JSON-backed list)
        corrected_drugs = correct_drug_list(raw_drugs)

        print(f"Extracted: {raw_drugs}")
        print(f"Corrected: {corrected_drugs}")
        print("\n")

if __name__ == "__main__":
    run_smoke_test()