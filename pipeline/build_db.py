
import sys
import json
from pathlib import Path

# Add parent directory to path so imports work regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from datasets import load_from_disk
from preprocessing.parser import parse_prescription
from preprocessing.cleaner import clean_medications, build_embedding_text
from embeddings.embedding import get_embedding
from vector_db.store import VectorStore
from mapping.fuzzy_match import correct_drug_list
from mapping.condition_mapper import ConditionMapper

store = VectorStore()
mapper = ConditionMapper()

print("** Building vector database...\n")
indexed = 0
skipped = 0

# ---------------------------------------------------------------
# SOURCE 1: Real medical dataset (first 500 samples)
# ---------------------------------------------------------------
print("[1/2] Indexing real dataset...")
try:
    dataset = load_from_disk(str(_PROJECT_ROOT / "data" / "medical_dataset"))
    sample_count = min(500, len(dataset["train"]))
    for sample in dataset["train"].select(range(sample_count)):
        raw_text = sample["ground_truth"]
        parsed = parse_prescription(raw_text)
        valid_drugs = correct_drug_list(parsed["drugs"])

        if not valid_drugs:
            skipped += 1
            continue

        case_conditions = []
        for drug in valid_drugs:
            if drug in mapper.knowledge_base:
                case_conditions.extend(mapper.knowledge_base[drug])

        cleaned_meds = clean_medications(parsed["medications"])
        embedding_input = build_embedding_text(valid_drugs, cleaned_meds)
        vector = get_embedding(embedding_input)

        store.add(vector, {
            "drugs": valid_drugs,
            "conditions": list(set(case_conditions)),
            "original_text": raw_text,
            "source": "real",
        })
        indexed += 1

    print(f"   Real dataset: {indexed} cases indexed, {skipped} skipped.\n")
except Exception as e:
    print(f"   WARNING: Could not load real dataset ({e}). Skipping.\n")

# ---------------------------------------------------------------
# SOURCE 2: Synthetic cases
# ---------------------------------------------------------------
synthetic_path = _PROJECT_ROOT / "data" / "synthetic_cases.json"
if not synthetic_path.exists():
    print("[2/2] Synthetic data not found — generating now...")
    # Auto-generate if the file doesn't exist
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
    from generate_synthetic_data import generate_dataset
    cases = generate_dataset(2000)
    synthetic_path.parent.mkdir(parents=True, exist_ok=True)
    with open(synthetic_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"   Generated {len(cases)} synthetic cases.\n")

print("[2/2] Indexing synthetic cases...")
with open(synthetic_path, "r", encoding="utf-8") as f:
    synthetic_cases = json.load(f)

syn_indexed = 0
for case in synthetic_cases:
    drugs = case.get("drugs", [])
    conditions = case.get("conditions", [])
    raw_text = case.get("ground_truth", "")

    if not drugs:
        continue

    embedding_input = " ".join(drugs) + " " + " ".join(conditions)
    vector = get_embedding(embedding_input)

    store.add(vector, {
        "drugs": drugs,
        "conditions": conditions,
        "original_text": raw_text,
        "source": "synthetic",
    })
    syn_indexed += 1

print(f"   Synthetic: {syn_indexed} cases indexed.\n")

# ---------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------
store_path = store.save(str(_PROJECT_ROOT / "vector_store"))
print(f"** Vector store saved to: {store_path}")
print(f"   - {len(store.vectors)} total vectors")
print(f"   - Real: {indexed}  |  Synthetic: {syn_indexed}")
print("\nDone! You can now run: streamlit run streamlit_app.py\n")
