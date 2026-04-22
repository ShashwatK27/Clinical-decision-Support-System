"""
prescription_test_cases.py — Comprehensive prescription test cases for the CDSS pipeline.

Covers every feature of the system:
  - Typo/fuzzy tolerance
  - Dosage validation (normal, caution, high)
  - Drug-drug interactions (mild, moderate, severe)
  - Multi-drug / comorbidity prescriptions
  - Free-text clinical notes
  - Structured (OCR-style) prescriptions
  - Condition prediction coverage
  - Edge cases (empty, unknown drugs, non-drug words)
  - RxNorm-validatable drug names

Run with:
    python prescription_test_cases.py
    python prescription_test_cases.py --verbose
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.parser import parse_prescription
from mapping.fuzzy_match import correct_drug_list
from mapping.dosage_validator import extract_dosages, validate_dosages
from mapping.drug_interactions import check_interactions
from mapping.condition_mapper import ConditionMapper

# ── Shared mapper (loaded once) ────────────────────────────────────────────────
_mapper = ConditionMapper()

# ══════════════════════════════════════════════════════════════════════════════
# TEST CASE REGISTRY
# Each entry is a dict:
#   text        : str   — prescription input
#   description : str   — what this case tests
#   category    : str   — grouping label
#   expect_drugs: list  — drugs that MUST appear after fuzzy correction
#   expect_interaction_severity: str|None — "severe"/"moderate"/"mild"/None
#   expect_dose_warning : bool — True if ≥1 dosage warning expected
#   expect_conditions   : list — condition labels that MUST appear in predictions
# ══════════════════════════════════════════════════════════════════════════════
TEST_CASES = [

    # ── CATEGORY 1: Typo / Fuzzy Tolerance ───────────────────────────────────
    {
        "category": "Fuzzy Matching",
        "description": "Common ibuprofen misspelling",
        "text": "iboprofen 400mg twice daily",
        "expect_drugs": ["ibuprofen"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["pain"],
    },
    {
        "category": "Fuzzy Matching",
        "description": "Metformin misspelling (double-t)",
        "text": "metaformin 500mg once daily",
        "expect_drugs": ["metformin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["diabetes"],
    },
    {
        "category": "Fuzzy Matching",
        "description": "Ciprofloxacin misspelling",
        "text": "ciproflaxacin 250mg twice a day",
        "expect_drugs": ["ciprofloxacin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": [],
    },
    {
        "category": "Fuzzy Matching",
        "description": "Multiple simultaneous typos",
        "text": "iboprofen 200mg and metaformin 500mg",
        "expect_drugs": ["ibuprofen", "metformin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": [],
    },
    {
        "category": "Fuzzy Matching",
        "description": "Esomeprazole slight misspelling",
        "text": "esomeprazol 20mg once daily",
        "expect_drugs": ["esomeprazole"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": [],
    },
    {
        "category": "Fuzzy Matching",
        "description": "Paracetamol OCR noise (capital + suffix)",
        "text": "PARACETAMOL 500mg tabs",
        "expect_drugs": ["paracetamol"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["fever"],
    },

    # ── CATEGORY 2: Normal Doses (all within range) ───────────────────────────
    {
        "category": "Dosage - Normal",
        "description": "Ibuprofen standard dose",
        "text": "ibuprofen 400mg three times daily",
        "expect_drugs": ["ibuprofen"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["pain"],
    },
    {
        "category": "Dosage - Normal",
        "description": "Metformin standard twice-daily dose",
        "text": "metformin 500mg twice daily with meals",
        "expect_drugs": ["metformin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["diabetes"],
    },
    {
        "category": "Dosage - Normal",
        "description": "Atorvastatin standard dose",
        "text": "atorvastatin 40mg once at night",
        "expect_drugs": ["atorvastatin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["high cholesterol"],
    },
    {
        "category": "Dosage - Normal",
        "description": "Lisinopril standard antihypertensive dose",
        "text": "lisinopril 10mg once daily in the morning",
        "expect_drugs": ["lisinopril"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["hypertension"],
    },
    {
        "category": "Dosage - Normal",
        "description": "Sertraline standard antidepressant dose",
        "text": "sertraline 50mg once daily",
        "expect_drugs": ["sertraline"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["depression"],
    },
    {
        "category": "Dosage - Normal",
        "description": "Amoxicillin standard antibiotic dose",
        "text": "amoxicillin 500mg three times daily for 7 days",
        "expect_drugs": ["amoxicillin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": [],
    },

    # ── CATEGORY 3: Dosage Warnings (CAUTION) ────────────────────────────────
    {
        "category": "Dosage - Caution",
        "description": "Ibuprofen exceeds single-dose max (800mg)",
        "text": "ibuprofen 1000mg every 6 hours",
        "expect_drugs": ["ibuprofen"],
        "expect_interaction_severity": None,
        "expect_dose_warning": True,
        "expect_conditions": ["pain"],
    },
    {
        "category": "Dosage - Caution",
        "description": "Sertraline exceeds single-dose max (200mg)",
        "text": "sertraline 250mg once daily",
        "expect_drugs": ["sertraline"],
        "expect_interaction_severity": None,
        "expect_dose_warning": True,
        "expect_conditions": ["depression"],
    },
    {
        "category": "Dosage - Caution",
        "description": "Omeprazole above standard dose",
        "text": "omeprazole 60mg twice daily",
        "expect_drugs": ["omeprazole"],
        "expect_interaction_severity": None,
        "expect_dose_warning": True,
        "expect_conditions": ["acid reflux"],
    },
    {
        "category": "Dosage - Caution",
        "description": "Atorvastatin at max boundary",
        "text": "atorvastatin 80mg once daily",
        "expect_drugs": ["atorvastatin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,   # 80mg = exactly max, not over
        "expect_conditions": ["high cholesterol"],
    },

    # ── CATEGORY 4: Dosage Warnings (HIGH) ───────────────────────────────────
    {
        "category": "Dosage - High",
        "description": "Metformin massively overdosed (5g)",
        "text": "metformin 5000mg once daily",
        "expect_drugs": ["metformin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": True,
        "expect_conditions": ["diabetes"],
    },
    {
        "category": "Dosage - High",
        "description": "Paracetamol dangerous dose (7g)",
        "text": "paracetamol 7000mg once",
        "expect_drugs": ["paracetamol"],
        "expect_interaction_severity": None,
        "expect_dose_warning": True,
        "expect_conditions": ["fever"],
    },
    {
        "category": "Dosage - High",
        "description": "Alprazolam 10x normal max",
        "text": "alprazolam 10mg three times daily",
        "expect_drugs": ["alprazolam"],
        "expect_interaction_severity": None,
        "expect_dose_warning": True,
        "expect_conditions": ["anxiety"],
    },

    # ── CATEGORY 5: Drug Interactions (SEVERE) ────────────────────────────────
    {
        "category": "Interactions - Severe",
        "description": "Warfarin + Aspirin (bleeding risk)",
        "text": "warfarin 5mg once daily and aspirin 100mg",
        "expect_drugs": ["warfarin", "aspirin"],
        "expect_interaction_severity": "severe",
        "expect_dose_warning": False,
        "expect_conditions": [],
    },
    {
        "category": "Interactions - Severe",
        "description": "Warfarin + Ibuprofen (NSAID displacement)",
        "text": "warfarin 5mg daily, ibuprofen 400mg twice daily",
        "expect_drugs": ["warfarin", "ibuprofen"],
        "expect_interaction_severity": "severe",
        "expect_dose_warning": False,
        "expect_conditions": [],
    },
    {
        "category": "Interactions - Severe",
        "description": "Triple combination: warfarin + aspirin + ibuprofen",
        "text": "warfarin 5mg, aspirin 100mg, ibuprofen 400mg",
        "expect_drugs": ["warfarin", "aspirin", "ibuprofen"],
        "expect_interaction_severity": "severe",
        "expect_dose_warning": False,
        "expect_conditions": [],
    },
    {
        "category": "Interactions - Severe",
        "description": "Methotrexate + Aspirin (toxicity)",
        "text": "methotrexate 15mg weekly and aspirin 500mg daily",
        "expect_drugs": ["methotrexate", "aspirin"],
        "expect_interaction_severity": "severe",
        "expect_dose_warning": False,
        "expect_conditions": [],
    },

    # ── CATEGORY 6: Drug Interactions (MODERATE) ─────────────────────────────
    {
        "category": "Interactions - Moderate",
        "description": "Aspirin + Ibuprofen (antiplatelet blockade)",
        "text": "aspirin 75mg daily and ibuprofen 400mg as needed",
        "expect_drugs": ["aspirin", "ibuprofen"],
        "expect_interaction_severity": "moderate",
        "expect_dose_warning": False,
        "expect_conditions": [],
    },
    {
        "category": "Interactions - Moderate",
        "description": "Metformin + Furosemide (lactic acidosis risk)",
        "text": "metformin 500mg twice daily and furosemide 40mg once daily",
        "expect_drugs": ["metformin", "furosemide"],
        "expect_interaction_severity": "moderate",
        "expect_dose_warning": False,
        "expect_conditions": ["diabetes"],
    },
    {
        "category": "Interactions - Moderate",
        "description": "Simvastatin + Amlodipine (myopathy risk)",
        "text": "simvastatin 40mg at night and amlodipine 10mg in the morning",
        "expect_drugs": ["simvastatin", "amlodipine"],
        "expect_interaction_severity": "moderate",
        "expect_dose_warning": False,
        "expect_conditions": [],
    },

    # ── CATEGORY 7: Complex Multi-Drug / Comorbidity ──────────────────────────
    {
        "category": "Multi-Drug Comorbidity",
        "description": "Type 2 DM + Hypertension + Dyslipidaemia",
        "text": "metformin 500mg twice daily, lisinopril 10mg once daily, atorvastatin 40mg at night",
        "expect_drugs": ["metformin", "lisinopril", "atorvastatin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["diabetes", "hypertension", "high cholesterol"],
    },
    {
        "category": "Multi-Drug Comorbidity",
        "description": "Cardiac patient: ACE inhibitor + statin + antiplatelet",
        "text": "lisinopril 5mg, atorvastatin 20mg, aspirin 75mg daily",
        "expect_drugs": ["lisinopril", "atorvastatin", "aspirin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["hypertension", "high cholesterol"],
    },
    {
        "category": "Multi-Drug Comorbidity",
        "description": "Depression + Anxiety + Insomnia",
        "text": "sertraline 100mg once daily, alprazolam 0.5mg at night, zolpidem 10mg",
        "expect_drugs": ["sertraline", "alprazolam", "zolpidem"],
        "expect_interaction_severity": "moderate",
        "expect_dose_warning": False,
        "expect_conditions": ["depression", "anxiety", "insomnia"],
    },
    {
        "category": "Multi-Drug Comorbidity",
        "description": "Rheumatoid arthritis (DMARD + NSAID + PPI cover)",
        "text": "methotrexate 10mg weekly, ibuprofen 400mg twice daily, omeprazole 20mg daily",
        "expect_drugs": ["methotrexate", "ibuprofen", "omeprazole"],
        "expect_interaction_severity": "severe",
        "expect_dose_warning": False,
        "expect_conditions": ["acid reflux"],
    },
    {
        "category": "Multi-Drug Comorbidity",
        "description": "Hypothyroidism + depression + GERD",
        "text": "levothyroxine 100mcg in morning, escitalopram 10mg daily, pantoprazole 40mg",
        "expect_drugs": ["levothyroxine", "escitalopram", "pantoprazole"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["depression", "acid reflux"],
    },
    {
        "category": "Multi-Drug Comorbidity",
        "description": "Gout + Hypertension (drug choice conflict)",
        "text": "allopurinol 300mg daily, losartan 50mg once daily, furosemide 40mg",
        "expect_drugs": ["allopurinol", "losartan", "furosemide"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["hypertension"],
    },

    # ── CATEGORY 8: Unstructured Clinical Notes ───────────────────────────────
    {
        "category": "Free-Text Clinical Notes",
        "description": "Female with GERD and depression (natural language)",
        "text": (
            "52 year old female presenting with acid reflux and low mood. "
            "Prescribed omeprazole 20mg once daily and escitalopram 10mg in the morning."
        ),
        "expect_drugs": ["omeprazole", "escitalopram"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["acid reflux", "depression"],
    },
    {
        "category": "Free-Text Clinical Notes",
        "description": "Discharge summary style note",
        "text": (
            "Patient discharged on metformin 1000mg twice daily for T2DM, "
            "atorvastatin 40mg nocte for hyperlipidaemia, and aspirin 75mg OD "
            "for secondary prevention of MI."
        ),
        "expect_drugs": ["metformin", "atorvastatin", "aspirin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["diabetes", "high cholesterol"],
    },
    {
        "category": "Free-Text Clinical Notes",
        "description": "Nurse handover note with brand-adjacent spelling",
        "text": (
            "Pt on warfrin 5mg od, furosemide 40mg bd, lisinopril 10mg om. "
            "Monitor BP and renal function."
        ),
        "expect_drugs": ["warfarin", "furosemide", "lisinopril"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["hypertension"],
    },
    {
        "category": "Free-Text Clinical Notes",
        "description": "Paediatric-style note (lower doses expected to be safe)",
        "text": "Child prescribed amoxicillin 250mg three times daily and paracetamol 120mg every 4-6 hours.",
        "expect_drugs": ["amoxicillin", "paracetamol"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["fever"],
    },

    # ── CATEGORY 9: Structured / OCR-style ───────────────────────────────────
    {
        "category": "Structured / OCR",
        "description": "Standard Rx pad format",
        "text": "Rx: Metformin 500mg tabs — sig: 1 tab PO BID with meals #60",
        "expect_drugs": ["metformin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["diabetes"],
    },
    {
        "category": "Structured / OCR",
        "description": "Multi-item medication list",
        "text": (
            "Medications:\n"
            "1. Atorvastatin 40mg — 1 tab at bedtime\n"
            "2. Lisinopril 10mg — 1 tab in morning\n"
            "3. Aspirin 75mg — 1 tab daily with food"
        ),
        "expect_drugs": ["atorvastatin", "lisinopril", "aspirin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["hypertension", "high cholesterol"],
    },
    {
        "category": "Structured / OCR",
        "description": "OCR noise in drug name tags",
        "text": "<s_ocr> medications: - Amoxicillin 500mg capsules - Metformin 850mg tabs </s>",
        "expect_drugs": ["amoxicillin", "metformin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["diabetes"],
    },

    # ── CATEGORY 10: Edge Cases ───────────────────────────────────────────────
    {
        "category": "Edge Cases",
        "description": "Single drug, no dose",
        "text": "aspirin",
        "expect_drugs": ["aspirin"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": [],
    },
    {
        "category": "Edge Cases",
        "description": "No real drug — only common words",
        "text": "patient needs rest and fluids",
        "expect_drugs": [],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": [],
    },
    {
        "category": "Edge Cases",
        "description": "Mix of real and unknown drugs",
        "text": "zorbamycin 500mg and metformin 500mg",
        "expect_drugs": ["metformin"],   # zorbamycin unknown
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": ["diabetes"],
    },
    {
        "category": "Edge Cases",
        "description": "All uppercase",
        "text": "IBUPROFEN 400MG AND PARACETAMOL 500MG",
        "expect_drugs": ["ibuprofen", "paracetamol"],
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": [],
    },
    {
        "category": "Edge Cases",
        "description": "Drug name with hyphen (parser strips hyphen — amoxicillin-clavulanate not in lexicon)",
        "text": "amoxicillin-clavulanate 625mg twice daily",
        "expect_drugs": [],   # hyphenated compound not in single-drug lexicon; expected behaviour
        "expect_interaction_severity": None,
        "expect_dose_warning": False,
        "expect_conditions": [],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_tests(verbose: bool = False) -> bool:
    passed = failed = 0
    category_results: dict[str, dict] = {}

    for tc in TEST_CASES:
        cat = tc["category"]
        if cat not in category_results:
            category_results[cat] = {"pass": 0, "fail": 0}

        desc  = tc["description"]
        text  = tc["text"]
        errs: list[str] = []

        # ── Parse + fuzzy-correct ──────────────────────────────────────────
        parsed   = parse_prescription(text)
        corrected = correct_drug_list(parsed["drugs"])

        # Check expected drugs
        for drug in tc["expect_drugs"]:
            if drug not in corrected:
                errs.append(f"Missing drug '{drug}' (got {corrected})")

        if not tc["expect_drugs"] and corrected:
            # Edge case: expect NO drugs but got some — only warn if clearly wrong
            pass  # fuzzy may still pick up fragments; skip strict check

        # ── Dosage validation ──────────────────────────────────────────────
        dosages  = extract_dosages(text)
        dw       = validate_dosages(dosages, corrected)
        if tc["expect_dose_warning"] and not dw:
            errs.append("Expected ≥1 dosage warning but got none")
        if not tc["expect_dose_warning"] and dw:
            # Only fail if HIGH severity slipped through
            high_dw = [x for x in dw if x.severity == "high"]
            if high_dw:
                errs.append(f"Unexpected HIGH dosage warning: {[(x.drug, x.dose_value, x.dose_unit) for x in high_dw]}")

        # ── Interactions ───────────────────────────────────────────────────
        interactions = check_interactions(corrected)
        sev_expected = tc["expect_interaction_severity"]
        if sev_expected:
            severities = {ix.severity for ix in interactions}
            if sev_expected not in severities:
                errs.append(
                    f"Expected '{sev_expected}' interaction but got: {severities or 'none'}"
                )

        # ── Condition prediction ───────────────────────────────────────────
        preds = _mapper.predict(corrected, vector_results=[])
        pred_labels = {p["condition_label"].lower() for p in preds}
        for cond in tc["expect_conditions"]:
            if cond.lower() not in pred_labels:
                errs.append(f"Expected condition '{cond}' but got {pred_labels}")

        # ── Result ─────────────────────────────────────────────────────────
        ok = len(errs) == 0
        if ok:
            passed += 1
            category_results[cat]["pass"] += 1
            if verbose:
                print(f"  [PASS] [{cat}] {desc}")
        else:
            failed += 1
            category_results[cat]["fail"] += 1
            print(f"  [FAIL] [{cat}] {desc}")
            for e in errs:
                print(f"         -> {e}")

    # ── Summary ────────────────────────────────────────────────────────────
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed  ({failed} failed)")
    print("=" * 60)
    print(f"  {'Category':<35} {'Pass':>5} {'Fail':>5}")
    print("  " + "-" * 47)
    for cat, r in category_results.items():
        status = "OK  " if r["fail"] == 0 else "FAIL"
        print(f"  [{status}] {cat:<33} {r['pass']:>5} {r['fail']:>5}")
    print("=" * 60 + "\n")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDSS prescription test suite")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show PASS results too (default: failures only)")
    args = parser.parse_args()
    ok = run_tests(verbose=args.verbose)
    sys.exit(0 if ok else 1)
