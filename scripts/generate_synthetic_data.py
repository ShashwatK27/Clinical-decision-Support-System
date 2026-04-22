"""
Synthetic Clinical Data Generator for CDSS.

Generates diverse, realistic prescription cases covering:
- Single and multi-drug prescriptions
- Varied dosages, frequencies, and routes
- Realistic patient demographics and clinical context
- Multiple prescription text formats (structured + free-text)

Run:
    python scripts/generate_synthetic_data.py
"""

import json
import random
from pathlib import Path

random.seed(42)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# Drug catalogue: drug → (dose_options, conditions)
# ------------------------------------------------------------------
DRUG_CATALOGUE = {
    "ibuprofen":            (["200mg", "400mg", "600mg", "800mg"],          ["pain", "inflammation"]),
    "paracetamol":          (["500mg", "650mg", "1000mg"],                   ["fever", "pain"]),
    "acetaminophen":        (["500mg", "650mg", "1000mg"],                   ["fever", "pain"]),
    "aspirin":              (["75mg", "100mg", "325mg"],                     ["pain", "heart disease prevention"]),
    "naproxen":             (["250mg", "500mg"],                             ["pain", "inflammation"]),
    "diclofenac":           (["50mg", "75mg", "100mg"],                      ["pain", "inflammation"]),
    "metformin":            (["500mg", "850mg", "1000mg"],                   ["diabetes"]),
    "glipizide":            (["5mg", "10mg"],                               ["diabetes"]),
    "insulin glargine":     (["10 units", "20 units", "30 units"],           ["diabetes"]),
    "sitagliptin":          (["50mg", "100mg"],                             ["diabetes"]),
    "lisinopril":           (["5mg", "10mg", "20mg", "40mg"],               ["hypertension"]),
    "losartan":             (["25mg", "50mg", "100mg"],                      ["hypertension"]),
    "amlodipine":           (["2.5mg", "5mg", "10mg"],                      ["hypertension", "angina"]),
    "hydrochlorothiazide":  (["12.5mg", "25mg"],                            ["hypertension"]),
    "atenolol":             (["25mg", "50mg", "100mg"],                      ["hypertension"]),
    "atorvastatin":         (["10mg", "20mg", "40mg", "80mg"],              ["high cholesterol"]),
    "simvastatin":          (["10mg", "20mg", "40mg"],                      ["high cholesterol"]),
    "rosuvastatin":         (["5mg", "10mg", "20mg"],                       ["high cholesterol"]),
    "amoxicillin":          (["250mg", "500mg", "875mg"],                   ["bacterial infection"]),
    "ciprofloxacin":        (["250mg", "500mg", "750mg"],                   ["bacterial infection"]),
    "azithromycin":         (["250mg", "500mg"],                            ["bacterial infection"]),
    "doxycycline":          (["100mg", "200mg"],                            ["bacterial infection"]),
    "gabapentin":           (["100mg", "300mg", "600mg"],                   ["neuropathy", "seizures"]),
    "pregabalin":           (["75mg", "150mg", "300mg"],                    ["neuropathy"]),
    "levothyroxine":        (["25mcg", "50mcg", "75mcg", "100mcg"],         ["thyroid disorder"]),
    "esomeprazole":         (["20mg", "40mg"],                              ["acid reflux", "gerd"]),
    "omeprazole":           (["20mg", "40mg"],                              ["acid reflux", "gerd"]),
    "pantoprazole":         (["20mg", "40mg"],                              ["acid reflux", "gerd"]),
    "prednisone":           (["5mg", "10mg", "20mg", "40mg"],              ["inflammation", "autoimmune disease"]),
    "loratadine":           (["10mg"],                                      ["allergic rhinitis", "allergies"]),
    "cetirizine":           (["5mg", "10mg"],                               ["allergic rhinitis", "allergies"]),
    "fexofenadine":         (["120mg", "180mg"],                            ["allergic rhinitis", "allergies"]),
    "diphenhydramine":      (["25mg", "50mg"],                              ["allergies", "insomnia"]),
    "sertraline":           (["25mg", "50mg", "100mg"],                     ["depression", "anxiety"]),
    "fluoxetine":           (["10mg", "20mg", "40mg"],                      ["depression"]),
    "escitalopram":         (["5mg", "10mg", "20mg"],                       ["depression", "anxiety"]),
    "alprazolam":           (["0.25mg", "0.5mg", "1mg"],                    ["anxiety"]),
    "zolpidem":             (["5mg", "10mg"],                               ["insomnia"]),
    "metoprolol":           (["25mg", "50mg", "100mg"],                     ["hypertension", "heart failure"]),
    "carvedilol":           (["3.125mg", "6.25mg", "12.5mg"],               ["heart failure", "hypertension"]),
    "furosemide":           (["20mg", "40mg", "80mg"],                      ["edema", "heart failure"]),
    "warfarin":             (["1mg", "2mg", "5mg"],                         ["blood clot prevention"]),
    "clopidogrel":          (["75mg"],                                      ["heart disease prevention"]),
    "allopurinol":          (["100mg", "300mg"],                            ["gout"]),
    "colchicine":           (["0.5mg", "1mg"],                              ["gout"]),
    "montelukast":          (["5mg", "10mg"],                               ["asthma", "allergic rhinitis"]),
    "salbutamol":           (["100mcg", "200mcg"],                          ["asthma"]),
    "budesonide":           (["200mcg", "400mcg"],                          ["asthma"]),
    "methotrexate":         (["2.5mg", "7.5mg", "15mg"],                    ["autoimmune disease", "rheumatoid arthritis"]),
    "hydroxychloroquine":   (["200mg", "400mg"],                            ["autoimmune disease", "rheumatoid arthritis"]),
}

FREQUENCIES = [
    "once daily", "twice daily", "three times daily",
    "every 4 hours", "every 6 hours", "every 8 hours",
    "at bedtime", "in the morning", "with meals",
    "as needed", "PRN", "bid", "tid",
]

ROUTES = ["oral", "by mouth", "PO", ""]  # "" = not specified (most common)

PATIENT_AGES = list(range(18, 85))
PATIENT_GENDERS = ["male", "female"]

CLINICAL_NOTES_TEMPLATES = [
    "Patient is a {age}-year-old {gender} presenting with {condition}. Prescribed {drug} {dose} {freq}.",
    "{drug} {dose} {freq} for management of {condition}.",
    "Rx: {drug} {dose} — {freq} — indication: {condition}.",
    "Medications: {drug} {dose} {freq}. Diagnosis: {condition}.",
    "Patient should take {drug} {dose} {freq} for {condition}.",
    "{drug} {dose} {freq} {route}. Monitor for response.",
    "Start {drug} {dose} {freq} for {condition}. Follow up in 2 weeks.",
    "Continue {drug} {dose} {freq} for ongoing {condition}.",
    "Initiated {drug} {dose} {freq} due to {condition}. Patient counselled.",
    "{age}y {gender}, diagnosed with {condition}. Prescribed {drug} {dose} {freq}.",
]

MULTI_DRUG_TEMPLATES = [
    "Medications:\n- {med1}\n- {med2}",
    "Medications:\n- {med1}\n- {med2}\n- {med3}",
    "Patient prescribed {med1} and {med2} for comorbid conditions.",
    "Rx: {med1}; {med2}. Monitor BP and glucose.",
    "{age}-year-old {gender} on {med1} and {med2} for {cond1} and {cond2}.",
    "Combined therapy: {med1} + {med2}. Review in 4 weeks.",
    "Add {med2} to existing {med1} regimen for better {cond2} control.",
    "Medications:\n- {med1}\n- {med2}\n- {med3}\nAll to be taken as directed.",
]


def _med_string(drug, dose, freq, route=""):
    parts = [drug, dose, freq]
    if route:
        parts.append(route)
    return " ".join(p for p in parts if p)


def _pick_drug():
    drug = random.choice(list(DRUG_CATALOGUE.keys()))
    doses, conditions = DRUG_CATALOGUE[drug]
    dose = random.choice(doses)
    freq = random.choice(FREQUENCIES)
    route = random.choice(ROUTES)
    condition = random.choice(conditions)
    return drug, dose, freq, route, condition


def generate_single_drug_case():
    drug, dose, freq, route, condition = _pick_drug()
    age = random.choice(PATIENT_AGES)
    gender = random.choice(PATIENT_GENDERS)
    template = random.choice(CLINICAL_NOTES_TEMPLATES)
    text = template.format(
        drug=drug, dose=dose, freq=freq, route=route,
        condition=condition, age=age, gender=gender,
    )
    _, conditions = DRUG_CATALOGUE[drug]
    return {
        "ground_truth": text,
        "drugs": [drug],
        "conditions": list(conditions),
    }


def generate_multi_drug_case(n_drugs=2):
    picks = [_pick_drug() for _ in range(n_drugs)]
    age = random.choice(PATIENT_AGES)
    gender = random.choice(PATIENT_GENDERS)

    med_strings = [_med_string(d, dose, freq, route) for d, dose, freq, route, _ in picks]
    cond1 = picks[0][4]
    cond2 = picks[1][4] if len(picks) > 1 else cond1

    template = random.choice(MULTI_DRUG_TEMPLATES)
    try:
        if n_drugs >= 3:
            text = template.format(
                med1=med_strings[0], med2=med_strings[1], med3=med_strings[2],
                cond1=cond1, cond2=cond2, age=age, gender=gender,
            )
        else:
            text = template.format(
                med1=med_strings[0], med2=med_strings[1],
                cond1=cond1, cond2=cond2, age=age, gender=gender,
            )
    except (KeyError, IndexError):
        # Template mismatch — fall back to simple format
        text = " and ".join(med_strings) + f" for {cond1}."

    drugs = [d for d, *_ in picks]
    conditions = list({c for d, *_ in picks for c in DRUG_CATALOGUE[d][1]})
    return {
        "ground_truth": text,
        "drugs": drugs,
        "conditions": conditions,
    }


def generate_dataset(n_total: int = 2000) -> list:
    cases = []
    single = int(n_total * 0.40)
    double = int(n_total * 0.40)
    triple = n_total - single - double

    for _ in range(single):
        cases.append(generate_single_drug_case())
    for _ in range(double):
        cases.append(generate_multi_drug_case(2))
    for _ in range(triple):
        cases.append(generate_multi_drug_case(3))

    random.shuffle(cases)
    return cases


if __name__ == "__main__":
    out_path = _PROJECT_ROOT / "data" / "synthetic_cases.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic clinical cases...")
    cases = generate_dataset(2000)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(cases)} cases to {out_path}")

    # Quick coverage summary
    from collections import Counter
    drug_counts = Counter(d for c in cases for d in c["drugs"])
    print(f"\nTop 10 drugs in synthetic data:")
    for drug, count in drug_counts.most_common(10):
        print(f"  {drug}: {count} cases")
    print(f"\nUnique drug combinations: {len(set(frozenset(c['drugs']) for c in cases))}")
