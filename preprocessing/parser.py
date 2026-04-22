import re


def clean_text(text):
    text = text.replace("<s_ocr>", "").replace("</s>", "")
    return text.lower().strip()


def extract_medication_block(text):
    """
    Extract only the medications section from a structured prescription.
    """
    match = re.search(r"medications:(.*?)(signature:|$)", text, re.DOTALL)
    if match:
        return match.group(1)
    return ""


def extract_medications(text):
    """
    Extract each medication line from a structured medications block.
    """
    meds_block = extract_medication_block(text)
    meds = re.split(r"- ", meds_block)
    meds = [m.strip() for m in meds if m.strip()]
    return meds


STOPWORDS = {
    "a", "an", "of",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "take", "takes", "taking", "taken", "at", "every", "twice", "daily",
    "once", "bedtime", "hours", "mg", "mcg", "g", "kg", "iu",
    "po", "bid", "qd", "qhs", "q4h", "q6h", "q8h", "q12h",
    "tsp", "tbsp", "prn", "tid", "qid", "stat", "ac", "pc",
    "tablet", "tablets", "tab", "capsule", "capsules", "cap",
    "tabs", "caplet", "caplets", "pill", "pills",
    "ml", "dr", "spray", "drop", "drops", "cream", "patch",
    "solution", "suspension", "dose", "doses", "puff", "puffs",
    "inhaler", "injection", "ointment", "gel", "syrup", "liquid",
    "oral", "topical", "intravenous", "subcutaneous", "intramuscular",
    "extended", "release", "immediate", "delayed", "sustained",
    "morning", "evening", "night", "noon", "afternoon",
    "before", "after", "with", "without", "food", "meals", "meal",
    "directed", "prescribed", "needed", "indicated",
    "patient", "female", "male", "woman", "man", "boy", "girl",
    "old", "year", "years", "age", "aged", "adult", "elderly",
    "presenting", "presents", "presented", "diagnosed", "diagnosis",
    "prescribed", "prescription", "start", "started", "continue",
    "initiated", "administered", "given", "indicated", "refer",
    "follow", "review", "monitor", "check", "assess", "counsel",
    "complaint", "history", "known", "case", "admitted", "discharged",
    "management", "treatment", "therapy", "regimen", "course",
    "pain", "fever", "anxiety", "depression", "nausea", "vomiting",
    "infection", "inflammation", "reflux", "diabetes", "hypertension",
    "cholesterol", "asthma", "allergy", "allergies", "insomnia",
    "neuropathy", "seizure", "seizures", "disorder", "disease",
    "condition", "symptom", "symptoms", "complaint", "complaints",
    "the", "and", "for", "due", "from", "this", "that", "which",
    "has", "have", "had", "was", "were", "are", "its", "their",
    "all", "also", "may", "can", "should", "will", "not", "both",
    "per", "use", "used", "using", "add", "added", "new", "start",
    "day", "days", "week", "weeks", "month", "months",
    "back", "well", "good", "better", "worse", "time", "times",
}


def extract_drug_names(meds):
    """
    Pull candidate drug-name tokens from a list of medication strings.

    Preserves hyphens so compound names like 'co-amoxiclav' are not destroyed.
    """
    drugs = []

    for med in meds:
        for word in med.split():
            word = re.sub(r"[^a-zA-Z\-]", "", word.lower()).strip("-")

            if len(word) < 3 or word in STOPWORDS:
                continue

            drugs.append(word)

    return list(set(drugs))


def parse_freetext(text):
    """
    Treat the full input as one medication line for unstructured text.
    """
    return extract_drug_names([text])


def parse_prescription(text):
    text = clean_text(text)
    meds = extract_medications(text)

    if meds:
        drugs = extract_drug_names(meds)
    else:
        meds = [text]
        drugs = parse_freetext(text)

    return {
        "medications": meds,
        "drugs": drugs,
    }
