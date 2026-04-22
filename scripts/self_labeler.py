"""
self_labeler.py — Build a drug → clinical-condition knowledge base from FDA data.

Reads the raw FDA drug product JSON and writes a mapping of drug names to
their pharmacological class labels. Output is used as a fallback KB by
mapping/condition_mapper.py.

NOTE: This produces pharmacological class labels (e.g. "thiazide diuretic"),
not plain clinical conditions. The condition_mapper filters out mechanism
labels (inhibitor, agonist, etc.) automatically at prediction time.

Usage:
    python scripts/self_labeler.py
"""

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_labeled_knowledge(raw_fda_path: Path, output_path: Path) -> None:
    with open(raw_fda_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    knowledge_base: dict[str, set] = {}

    for product in data.get("results", []):
        names = []
        if "generic_name" in product:
            names.append(product["generic_name"].lower().strip())
        if "brand_name" in product:
            names.append(product["brand_name"].lower().strip())

        openfda = product.get("openfda", {})
        # pharm_class_epc = Established Pharmacological Class (e.g. "Proton Pump Inhibitor [EPC]")
        # pharm_class_moa = Mechanism of Action
        raw_labels = openfda.get("pharm_class_epc", []) + openfda.get("pharm_class_moa", [])

        # Strip the "[EPC]" / "[MoA]" bracket tags
        clean_labels = [label.split("[")[0].strip().lower() for label in raw_labels]
        clean_labels = [l for l in clean_labels if l]  # drop empties

        for name in names:
            if name and clean_labels:
                if name not in knowledge_base:
                    knowledge_base[name] = set()
                knowledge_base[name].update(clean_labels)

    # Convert sets to sorted lists for stable JSON output
    final_kb = {k: sorted(v) for k, v in sorted(knowledge_base.items())}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_kb, f, indent=2, ensure_ascii=False)

    print(f"Self-labeling complete: {len(final_kb)} drugs labeled -> {output_path}")


if __name__ == "__main__":
    raw_path = _PROJECT_ROOT / "data" / "lexicons" / "raw_drug.json"
    out_path = _PROJECT_ROOT / "data" / "lexicons" / "labeled_drugs.json"

    if not raw_path.exists():
        print(f"ERROR: raw drug JSON not found at {raw_path}")
        raise SystemExit(1)

    build_labeled_knowledge(raw_path, out_path)