# Clinical Decision Support System (CDSS)

A hybrid clinical decision support system that combines:

- rule-based prescription parsing
- typo-tolerant drug normalization
- curated drug-to-condition knowledge
- semantic case retrieval with pretrained text embeddings
- dosage and drug-interaction safety checks

This project is not a trained disease-prediction model. It is a knowledge-based CDSS augmented with semantic similarity search.

## Overview

The system processes prescription text and helps a clinician by:

1. extracting likely drug names from unstructured text
2. correcting misspellings using a local drug lexicon and fuzzy matching
3. validating dosage ranges for supported drugs
4. checking curated drug-drug interaction rules
5. retrieving similar historical or synthetic cases using vector similarity
6. predicting likely clinical conditions from a hybrid of knowledge-base rules and retrieved cases
7. generating clinician-facing recommendations and a downloadable PDF report

## What This Project Is

- A rule-based and retrieval-augmented CDSS
- A semantic search system over indexed prescription cases
- A local decision-support tool for demos, experiments, and academic projects

## What This Project Is Not

- Not a trained classifier built in this repository
- Not a diagnostic engine
- Not a replacement for clinical judgement
- Not a complete or production-certified prescribing platform

## Architecture Summary

The project has two main workflows:

### 1. Offline Index-Building Workflow

`pipeline/build_db.py` creates the vector database used by the app:

1. load records from `data/medical_dataset`
2. parse prescription text to extract candidate drugs
3. fuzzy-correct extracted drug names
4. map known drugs to condition labels using the local knowledge base
5. clean medication text and build embedding input
6. generate normalized sentence embeddings with `all-MiniLM-L6-v2`
7. store vectors with metadata in `vector_store/`
8. optionally generate and index synthetic cases

The stored vector database is saved as:

- `vector_store/vectors.npy`
- `vector_store/metadata.json`

### 2. Online Prescription Analysis Workflow

The runtime flow used by `main.py` and `streamlit_app.py` is:

1. accept prescription text
2. parse likely drug tokens
3. fuzzy-correct drug names
4. extract and validate doses
5. check drug-drug interactions
6. clean medication text
7. generate an embedding for the query
8. search similar indexed cases by cosine similarity
9. combine rule-based condition mapping with retrieved-case evidence
10. filter and rank conditions
11. show recommendations, similar cases, and optional PDF output

## Where The "AI" Is

The ML-related part of the project is limited to pretrained embeddings:

- `embeddings/embedding.py` uses `sentence-transformers/all-MiniLM-L6-v2`
- the embedding model converts text into dense vectors
- those vectors are used for semantic similarity search

There is no model training code for condition prediction in this repository. Final condition outputs are produced by hybrid scoring logic in `mapping/condition_mapper.py`, not by a trained classifier.

## Core Components

### `preprocessing/parser.py`

- normalizes prescription text
- detects structured medication sections when present
- extracts candidate drug tokens from free text

### `mapping/fuzzy_match.py`

- loads a local drug lexicon from `data/lexicons/drugs.json`
- handles known typo aliases such as `iboprofen -> ibuprofen`
- applies fuzzy matching for misspelled drug names

### `mapping/condition_mapper.py`

- loads the drug-to-condition knowledge base
- merges curated clean seed mappings with dataset-derived mappings
- scores conditions using:
  - direct drug-to-condition matches
  - similar-case conditions from vector retrieval
- filters noisy pharmacological-class labels

### `mapping/dosage_validator.py`

- extracts values like `400mg`
- compares them against curated dose limits
- returns caution or high-severity warnings

### `mapping/drug_interactions.py`

- checks curated pairwise drug interaction rules
- labels findings as `severe`, `moderate`, or `mild`

### `embeddings/embedding.py`

- lazily loads the pretrained SentenceTransformer model
- returns normalized embeddings for search

### `vector_db/store.py`

- stores vectors and metadata
- performs cosine similarity search by dot product
- preserves metadata during retrieval
- saves data in safe NumPy + JSON format instead of pickle

### `utils/rxnorm_api.py`

- optionally validates drugs against the NLM RxNorm API
- enriches results with RxCUI and canonical names when available

### `utils/pdf_report.py`

- generates a clinician-facing PDF summary of one analysis

## Project Structure

```text
cdss/
├── main.py
├── demo.py
├── streamlit_app.py
├── README.md
├── requirements.txt
├── test_e2e.py
├── WEBAPP_SETUP.md
├── embeddings/
│   └── embedding.py
├── preprocessing/
│   ├── parser.py
│   └── cleaner.py
├── mapping/
│   ├── condition_mapper.py
│   ├── dosage_validator.py
│   ├── drug_interactions.py
│   └── fuzzy_match.py
├── pipeline/
│   └── build_db.py
├── vector_db/
│   └── store.py
├── utils/
│   ├── helpers.py
│   ├── logger_config.py
│   ├── pdf_report.py
│   └── rxnorm_api.py
├── scripts/
├── tests/
└── data/
```

## Installation

```bash
pip install -r requirements.txt
```

## Running The Project

### 1. Build the vector store

Run this first if `vector_store/` has not already been created:

```bash
python pipeline/build_db.py
```

### 2. Run the CLI app

```bash
python main.py
```

Example input:

```text
ibuprofen 200mg and metformin 500mg
```

### 3. Run the Streamlit web app

```bash
streamlit run streamlit_app.py
```

### 4. Run the demo

```bash
python demo.py
```

## Example Runtime Flow

Input:

```text
warfarin 5mg once daily and ibuprofen 400mg twice daily
```

Possible system behavior:

- extracts `warfarin` and `ibuprofen`
- validates detected doses
- flags a severe bleeding-risk interaction
- retrieves similar indexed cases
- predicts condition labels supported by the local knowledge base
- shows safety recommendations
- allows export of a PDF report

## Data Sources

The project uses a mix of local resources:

- local drug lexicon in `data/lexicons/drugs.json`
- local labeled mappings in `data/lexicons/labeled_drugs.json`
- dataset-derived drug knowledge in `data/medical_dataset/drug_kb.json`
- local train/validation/test dataset under `data/medical_dataset/`
- synthetic cases in `data/synthetic_cases.json`

## Outputs

Depending on the entry point, the system can produce:

- corrected drug names
- dosage warnings
- interaction warnings
- predicted clinical conditions
- similar historical cases
- clinical recommendations
- session history in the web app
- a generated PDF report

## Testing

Run unit and regression tests:

```bash
python -m unittest discover tests -v
```

Run the end-to-end test:

```bash
python test_e2e.py
```

## Limitations

- Predictions depend heavily on the local knowledge base
- Retrieval quality depends on the indexed cases and embedding quality
- Dosage validation covers only curated drugs and simple dose patterns
- Interaction checking is rule-based and not exhaustive
- RxNorm validation requires network access
- The system is decision-support only and should not be used as an autonomous clinical tool

## Recommended Description For Reports Or Viva

If you need one accurate line to describe the project, use:

> A hybrid clinical decision support system that combines rule-based reasoning, curated medical knowledge, and pretrained sentence embeddings for semantic case retrieval.

Or more simply:

> A rule-based CDSS with semantic similarity search.

## Future Improvements

- add a true supervised prediction model if ML classification is required
- expand the clinical knowledge base
- improve structured prescription parsing
- support stronger medical-domain embeddings
- add persistent audit logging and user authentication
- expose the workflow through a REST API

## Disclaimer

This project is for educational, research, and prototype use only. It is not a certified medical device and must not be used as a substitute for qualified clinical judgement.
