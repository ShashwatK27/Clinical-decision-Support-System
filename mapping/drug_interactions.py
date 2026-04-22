"""
drug_interactions.py — Local Drug-Drug Interaction (DDI) knowledge base.

Contains a curated set of clinically significant interactions covering
the 50 most commonly prescribed drug classes. Each interaction includes:
    - drugs: frozenset of two interacting drug names (or their class aliases)
    - severity: "severe" | "moderate" | "mild"
    - effect: short plain-English description of the interaction effect
    - recommendation: what the clinician should do

Usage:
    from mapping.drug_interactions import check_interactions
    warnings = check_interactions(["warfarin", "aspirin"])
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Interaction:
    drugs: frozenset
    severity: str          # "severe" | "moderate" | "mild"
    effect: str
    recommendation: str


# ---------------------------------------------------------------------------
# Curated DDI database
# Each entry covers ONE pair. Drug names are lowercase.
# ---------------------------------------------------------------------------
_RAW_INTERACTIONS: list[dict] = [
    # ── Severe ─────────────────────────────────────────────────────────────
    {
        "drugs": ["warfarin", "aspirin"],
        "severity": "severe",
        "effect": "Significantly increased bleeding risk (anticoagulant + antiplatelet).",
        "recommendation": "Avoid combination unless benefits outweigh risks. Monitor INR closely.",
    },
    {
        "drugs": ["warfarin", "ibuprofen"],
        "severity": "severe",
        "effect": "NSAIDs displace warfarin from protein binding, raising bleeding risk.",
        "recommendation": "Use paracetamol instead. Monitor INR if NSAID is unavoidable.",
    },
    {
        "drugs": ["warfarin", "naproxen"],
        "severity": "severe",
        "effect": "NSAIDs displace warfarin from protein binding, raising bleeding risk.",
        "recommendation": "Use paracetamol instead. Monitor INR if NSAID is unavoidable.",
    },
    {
        "drugs": ["warfarin", "diclofenac"],
        "severity": "severe",
        "effect": "NSAIDs displace warfarin from protein binding, raising bleeding risk.",
        "recommendation": "Use paracetamol instead. Monitor INR if NSAID is unavoidable.",
    },
    {
        "drugs": ["warfarin", "ciprofloxacin"],
        "severity": "severe",
        "effect": "Ciprofloxacin inhibits CYP1A2, increasing warfarin levels and INR.",
        "recommendation": "Monitor INR daily during co-therapy. Adjust warfarin dose.",
    },
    {
        "drugs": ["warfarin", "azithromycin"],
        "severity": "severe",
        "effect": "Azithromycin inhibits warfarin metabolism, raising bleeding risk.",
        "recommendation": "Monitor INR closely during and 1 week after antibiotic course.",
    },
    {
        "drugs": ["warfarin", "metronidazole"],
        "severity": "severe",
        "effect": "Metronidazole potentiates warfarin, markedly increasing INR.",
        "recommendation": "Reduce warfarin dose by ~50% and monitor INR every 2 days.",
    },
    {
        "drugs": ["sertraline", "tramadol"],
        "severity": "severe",
        "effect": "Risk of serotonin syndrome (agitation, hyperthermia, tachycardia).",
        "recommendation": "Avoid combination. Use alternative analgesic.",
    },
    {
        "drugs": ["fluoxetine", "tramadol"],
        "severity": "severe",
        "effect": "Risk of serotonin syndrome and lowered seizure threshold.",
        "recommendation": "Avoid. Use non-serotonergic analgesic.",
    },
    {
        "drugs": ["escitalopram", "tramadol"],
        "severity": "severe",
        "effect": "Risk of serotonin syndrome.",
        "recommendation": "Avoid. Use alternative analgesic.",
    },
    {
        "drugs": ["metformin", "ibuprofen"],
        "severity": "moderate",
        "effect": "NSAIDs can reduce renal blood flow, increasing risk of metformin-induced lactic acidosis.",
        "recommendation": "Use paracetamol for pain; monitor renal function if NSAID is essential.",
    },
    {
        "drugs": ["metformin", "naproxen"],
        "severity": "moderate",
        "effect": "NSAIDs reduce renal clearance of metformin, risking lactic acidosis.",
        "recommendation": "Monitor renal function; consider paracetamol instead.",
    },
    {
        "drugs": ["lisinopril", "losartan"],
        "severity": "severe",
        "effect": "Dual RAAS blockade causes hypotension, hyperkalaemia, and acute kidney injury.",
        "recommendation": "Contraindicated in most patients. Do not co-prescribe.",
    },
    {
        "drugs": ["lisinopril", "hydrochlorothiazide"],
        "severity": "moderate",
        "effect": "Additive hypotensive effect; risk of first-dose hypotension.",
        "recommendation": "Monitor BP closely, especially at initiation. Start at low dose.",
    },
    {
        "drugs": ["amlodipine", "simvastatin"],
        "severity": "moderate",
        "effect": "Amlodipine inhibits CYP3A4, raising simvastatin levels and myopathy risk.",
        "recommendation": "Limit simvastatin to 20 mg/day. Consider rosuvastatin instead.",
    },
    {
        "drugs": ["amlodipine", "atorvastatin"],
        "severity": "mild",
        "effect": "Slight increase in atorvastatin exposure via CYP3A4 inhibition.",
        "recommendation": "Generally safe. Monitor for muscle pain at high atorvastatin doses.",
    },
    # ── Moderate ───────────────────────────────────────────────────────────
    {
        "drugs": ["aspirin", "ibuprofen"],
        "severity": "moderate",
        "effect": "Ibuprofen may block aspirin's antiplatelet effect if taken within 2 hours.",
        "recommendation": "Take aspirin at least 2 hours before ibuprofen. Consider naproxen.",
    },
    {
        "drugs": ["metformin", "prednisone"],
        "severity": "moderate",
        "effect": "Corticosteroids increase blood glucose, counteracting metformin.",
        "recommendation": "Monitor blood glucose carefully; adjust metformin dose as needed.",
    },
    {
        "drugs": ["metformin", "furosemide"],
        "severity": "moderate",
        "effect": "Furosemide may raise metformin levels by competing for renal secretion.",
        "recommendation": "Monitor renal function and metformin levels during co-therapy.",
    },
    {
        "drugs": ["lisinopril", "potassium"],
        "severity": "moderate",
        "effect": "ACE inhibitors raise potassium; adding K+ supplements risks hyperkalaemia.",
        "recommendation": "Monitor serum potassium regularly.",
    },
    {
        "drugs": ["alprazolam", "zolpidem"],
        "severity": "moderate",
        "effect": "Additive CNS depression; risk of excessive sedation and respiratory suppression.",
        "recommendation": "Avoid combination. If both needed, use lowest effective doses.",
    },
    {
        "drugs": ["alprazolam", "diphenhydramine"],
        "severity": "moderate",
        "effect": "Additive sedation, anticholinergic effects.",
        "recommendation": "Use with caution; avoid in elderly patients.",
    },
    {
        "drugs": ["gabapentin", "zolpidem"],
        "severity": "moderate",
        "effect": "Additive CNS depression; increased sedation and fall risk.",
        "recommendation": "Use with caution. Counsel patient on fall risk.",
    },
    {
        "drugs": ["gabapentin", "diphenhydramine"],
        "severity": "moderate",
        "effect": "Additive CNS depression.",
        "recommendation": "Avoid in elderly. Monitor for excessive sedation.",
    },
    {
        "drugs": ["prednisone", "ibuprofen"],
        "severity": "moderate",
        "effect": "Increased GI bleeding risk (dual GI irritants).",
        "recommendation": "Add a PPI (e.g. omeprazole). Monitor for GI symptoms.",
    },
    {
        "drugs": ["atorvastatin", "ciprofloxacin"],
        "severity": "moderate",
        "effect": "Ciprofloxacin inhibits CYP3A4, raising atorvastatin levels and myopathy risk.",
        "recommendation": "Monitor for muscle pain (myopathy/rhabdomyolysis) during co-therapy.",
    },
    {
        "drugs": ["sertraline", "aspirin"],
        "severity": "moderate",
        "effect": "SSRIs + NSAIDs/antiplatelet: increased GI and systemic bleeding risk.",
        "recommendation": "Consider adding PPI prophylaxis. Monitor for signs of bleeding.",
    },
    {
        "drugs": ["fluoxetine", "aspirin"],
        "severity": "moderate",
        "effect": "SSRIs + NSAIDs/antiplatelet: increased GI and systemic bleeding risk.",
        "recommendation": "Consider adding PPI prophylaxis. Monitor for signs of bleeding.",
    },
    {
        "drugs": ["escitalopram", "aspirin"],
        "severity": "moderate",
        "effect": "SSRIs + NSAIDs/antiplatelet: increased GI and systemic bleeding risk.",
        "recommendation": "Consider adding PPI prophylaxis. Monitor for signs of bleeding.",
    },
    {
        "drugs": ["levothyroxine", "omeprazole"],
        "severity": "mild",
        "effect": "PPIs reduce gastric acid, impairing levothyroxine absorption.",
        "recommendation": "Take levothyroxine 30–60 min before omeprazole. Monitor TSH.",
    },
    {
        "drugs": ["levothyroxine", "esomeprazole"],
        "severity": "mild",
        "effect": "PPIs reduce gastric acid, impairing levothyroxine absorption.",
        "recommendation": "Take levothyroxine 30–60 min before PPI. Monitor TSH.",
    },
    {
        "drugs": ["levothyroxine", "pantoprazole"],
        "severity": "mild",
        "effect": "PPIs reduce gastric acid, impairing levothyroxine absorption.",
        "recommendation": "Take levothyroxine 30–60 min before PPI. Monitor TSH.",
    },
    {
        "drugs": ["methotrexate", "ibuprofen"],
        "severity": "severe",
        "effect": "NSAIDs reduce methotrexate renal clearance, causing toxic drug levels.",
        "recommendation": "Avoid combination. If NSAID essential, halve methotrexate dose and monitor.",
    },
    {
        "drugs": ["methotrexate", "naproxen"],
        "severity": "severe",
        "effect": "NSAIDs reduce methotrexate renal clearance, causing toxic drug levels.",
        "recommendation": "Avoid combination.",
    },
    {
        "drugs": ["methotrexate", "aspirin"],
        "severity": "severe",
        "effect": "Aspirin reduces renal clearance of methotrexate, risking haematological and GI toxicity.",
        "recommendation": "Avoid concurrent use. If antiplatelet needed, discuss risk-benefit with specialist.",
    },
    {
        "drugs": ["hydroxychloroquine", "azithromycin"],
        "severity": "severe",
        "effect": "Both prolong QT interval; increased risk of fatal arrhythmia.",
        "recommendation": "Contraindicated. Use alternative antibiotic.",
    },
    {
        "drugs": ["furosemide", "atenolol"],
        "severity": "mild",
        "effect": "Furosemide-induced hypokalaemia can potentiate atenolol bradycardia.",
        "recommendation": "Monitor electrolytes and heart rate.",
    },
    {
        "drugs": ["clopidogrel", "omeprazole"],
        "severity": "moderate",
        "effect": "Omeprazole inhibits CYP2C19, reducing clopidogrel's antiplatelet effect.",
        "recommendation": "Use pantoprazole instead of omeprazole.",
    },
    {
        "drugs": ["clopidogrel", "esomeprazole"],
        "severity": "moderate",
        "effect": "Esomeprazole inhibits CYP2C19, reducing clopidogrel activation.",
        "recommendation": "Use pantoprazole instead.",
    },
    {
        "drugs": ["allopurinol", "azathioprine"],
        "severity": "severe",
        "effect": "Allopurinol inhibits xanthine oxidase, causing azathioprine toxicity.",
        "recommendation": "Reduce azathioprine dose by 75% or avoid combination.",
    },
    {
        "drugs": ["colchicine", "azithromycin"],
        "severity": "moderate",
        "effect": "Azithromycin inhibits P-glycoprotein, raising colchicine toxicity risk.",
        "recommendation": "Reduce colchicine dose. Monitor for GI/muscle toxicity.",
    },
    {
        "drugs": ["metoprolol", "diphenhydramine"],
        "severity": "mild",
        "effect": "Diphenhydramine inhibits CYP2D6, raising metoprolol levels and bradycardia risk.",
        "recommendation": "Monitor heart rate. Use loratadine as non-sedating alternative.",
    },
    {
        "drugs": ["carvedilol", "diphenhydramine"],
        "severity": "mild",
        "effect": "CYP2D6 inhibition raises carvedilol levels; additive bradycardia risk.",
        "recommendation": "Monitor heart rate. Prefer non-sedating antihistamine.",
    },
    {
        "drugs": ["pregabalin", "zolpidem"],
        "severity": "moderate",
        "effect": "Additive CNS depression; increased sedation, respiratory suppression.",
        "recommendation": "Use with caution; counsel on fall risk.",
    },
    {
        "drugs": ["pregabalin", "alprazolam"],
        "severity": "moderate",
        "effect": "Additive CNS depression.",
        "recommendation": "Avoid if possible; use lowest effective doses.",
    },
]

# Build the indexed lookup: frozenset → Interaction
_INTERACTIONS: list[Interaction] = [
    Interaction(
        drugs=frozenset(row["drugs"]),
        severity=row["severity"],
        effect=row["effect"],
        recommendation=row["recommendation"],
    )
    for row in _RAW_INTERACTIONS
]

# Severity ordering for sorting
_SEVERITY_ORDER = {"severe": 0, "moderate": 1, "mild": 2}


def check_interactions(drug_list: list[str]) -> list[Interaction]:
    """
    Given a list of drug names, return all known pairwise interactions
    sorted from most to least severe.

    Args:
        drug_list: List of lowercase drug names (already fuzzy-corrected).

    Returns:
        List of Interaction objects (may be empty if no interactions found).
    """
    drugs_lower = {d.lower().strip() for d in drug_list}
    found: list[Interaction] = []

    for interaction in _INTERACTIONS:
        if interaction.drugs.issubset(drugs_lower):
            found.append(interaction)

    # Sort: severe first, then moderate, then mild
    found.sort(key=lambda x: _SEVERITY_ORDER.get(x.severity, 99))
    return found
