import re


def clean_text(text):
    text = text.replace("<s_ocr>", "").replace("</s>", "")
    return text.lower().strip()


def extract_medication_block(text):
    """
    Extract only the medications section
    """
    match = re.search(r"medications:(.*?)(signature:|$)", text, re.DOTALL)
    if match:
        return match.group(1)
    return ""


def extract_medications(text):
    """
    Extract each medication line
    """
    meds_block = extract_medication_block(text)

    # Split using "-"
    meds = re.split(r"- ", meds_block)

    # Clean empty entries
    meds = [m.strip() for m in meds if m.strip()]

    return meds



STOPWORDS = {
    "take", "at", "every", "twice", "daily",
    "once", "bedtime", "hours", "mg", "mcg", "g",
    "po", "bid", "qd", "qhs", "q4h", "tsp", "tbsp",
    "tablet", "tablets", "tab", "capsule", "capsules",
    "ml", "dr", "spray", "drop", "drops", "cream", "patch",
    "tablet", "solution", "suspension", "dose", "doses", "prn"
}


def extract_drug_names(meds):
    drugs = []

    for med in meds:
        words = med.split()

        for word in words:
            # clean word
            word = re.sub(r'[^a-zA-Z]', '', word.lower())

            if len(word) < 3:
                continue

            if word in STOPWORDS:
                continue

            drugs.append(word)

    return list(set(drugs))  # remove duplicates







def parse_prescription(text):
    text = clean_text(text)

    # Try structured extraction first
    meds = extract_medications(text)

    # 🔥 If no meds found → fallback to raw text
    if not meds:
        meds = [text]

    drugs = extract_drug_names(meds)

    return {
        "medications": meds,
        "drugs": drugs
    }



