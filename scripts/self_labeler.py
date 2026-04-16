import json

def build_labeled_knowledge(raw_fda_path, output_path):
    with open(raw_fda_path, 'r') as f:
        data = json.load(f)
    
    knowledge_base = {}

    for product in data.get('results', []):
        # 1. Get the names
        names = []
        if 'generic_name' in product: names.append(product['generic_name'].lower())
        if 'brand_name' in product: names.append(product['brand_name'].lower())
        
        # 2. Extract Labels from Pharmacological Class (EPC)
        # This tells us exactly what the drug treats (e.g., "Thiazide Diuretic")
        openfda = product.get('openfda', {})
        labels = openfda.get('pharm_class_epc', []) + openfda.get('pharm_class_moa', [])
        
        # Clean the labels (remove the "[EPC]" or "[MoA]" tags)
        clean_labels = [l.split('[')[0].strip().lower() for l in labels]

        # 3. Map every name variation to these labels
        for name in names:
            if clean_labels:
                if name not in knowledge_base:
                    knowledge_base[name] = set()
                knowledge_base[name].update(clean_labels)

    # Convert sets to lists for JSON serialization
    final_kb = {k: list(v) for k, v in knowledge_base.items()}

    with open(output_path, 'w') as f:
        json.dump(final_kb, f, indent=4)
    
    print(f"✅ Self-labeling complete! Labeled {len(final_kb)} drugs with clinical classes.")

# Run the labeler
build_labeled_knowledge('data/lexicons/raw_drug.json', 'data/lexicons/labeled_drugs.json')