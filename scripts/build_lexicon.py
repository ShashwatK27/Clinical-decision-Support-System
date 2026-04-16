import json

def extract_drug_lexicon(raw_json_path, output_json_path):
    with open(raw_json_path, 'r') as f:
        data = json.load(f)
    
    drug_set = set()
    
    for product in data.get('results', []):
        # 1. Get Generic Name (e.g., "Esomeprazole")
        if 'generic_name' in product:
            drug_set.add(product['generic_name'].lower())
            
        # 2. Get Brand Name (e.g., "basic care esomeprazole magnesium")
        if 'brand_name' in product:
            drug_set.add(product['brand_name'].lower())
            
        # 3. Get Active Ingredients
        for ingredient in product.get('active_ingredients', []):
            if 'name' in ingredient:
                drug_set.add(ingredient['name'].lower())

    # Save as a clean list for fuzzy_match.py
    with open(output_json_path, 'w') as f:
        json.dump(sorted(list(drug_set)), f, indent=4)
    
    print(f"✅ Lexicon built! Extracted {len(drug_set)} unique drug tokens.")

# Run it
extract_drug_lexicon('data/lexicons/raw_drug.json', 'data/lexicons/drugs.json')