"""
build_lexicon.py — Extract a clean drug name lexicon from FDA raw JSON.

Reads the raw FDA drug product JSON (data/lexicons/raw_drug.json) and
writes a deduplicated, sorted list of lowercase drug names to
data/lexicons/drugs.json, which is used by mapping/fuzzy_match.py.

Usage:
    python scripts/build_lexicon.py
"""

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_drug_lexicon(raw_json_path: Path, output_json_path: Path) -> None:
    with open(raw_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    drug_set = set()

    for product in data.get("results", []):
        # Generic name (e.g. "esomeprazole")
        if "generic_name" in product:
            drug_set.add(product["generic_name"].lower().strip())

        # Brand name (e.g. "nexium")
        if "brand_name" in product:
            drug_set.add(product["brand_name"].lower().strip())

        # Active ingredients (handles combination products)
        for ingredient in product.get("active_ingredients", []):
            if "name" in ingredient:
                drug_set.add(ingredient["name"].lower().strip())

    # Remove empty strings that can slip in from malformed entries
    drug_set.discard("")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(sorted(drug_set), f, indent=2, ensure_ascii=False)

    print(f"Lexicon saved: {len(drug_set)} unique drug tokens -> {output_json_path}")


if __name__ == "__main__":
    raw_path = _PROJECT_ROOT / "data" / "lexicons" / "raw_drug.json"
    out_path = _PROJECT_ROOT / "data" / "lexicons" / "drugs.json"

    if not raw_path.exists():
        print(f"ERROR: raw drug JSON not found at {raw_path}")
        raise SystemExit(1)

    extract_drug_lexicon(raw_path, out_path)
