# 🩺 Clinical Decision Support System (CDSS)

A hybrid machine learning and knowledge-based system for prescribing clinical condition prediction using vector similarity search and rule-based drug mapping.

## 📋 Overview

The CDSS processes prescription text to:
1. **Extract drug names** from unstructured medical text
2. **Correct typos** using fuzzy matching (e.g., "iboprofen" → "ibuprofen")
3. **Predict clinical conditions** based on drug knowledge
4. **Search similar cases** using semantic embeddings
5. **Output actionable recommendations** for healthcare providers

## ✨ Key Features

- ✅ **Typo-tolerant matching**: Handles OCR errors and common misspellings
- ✅ **Clinical conditions**: Outputs real health conditions (not pharmacological classes)
- ✅ **Multi-drug reasoning**: Combines predictions from multiple drugs
- ✅ **Semantic search**: Finds similar historical cases using embeddings
- ✅ **Vector metadata preservation**: No data loss during vector store operations

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd cdss

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Interactive CDSS

```bash
python main.py
```

Then enter prescription text:
```
Enter Prescription Text (or 'exit'): ibuprofen 200mg and metformin 500mg

🧠 PREDICTED CLINICAL CONDITIONS:
   • PAIN (Score: 2.0)
   • INFLAMMATION (Score: 2.0)
   • DIABETES (Score: 2.0)
```

### 3. Run Demo

For a guided walkthrough with multiple examples:

```bash
python demo.py
```

### 4. Run Tests

Unit tests:
```bash
python -m unittest discover tests -v
```

End-to-end tests:
```bash
python test_e2e.py
```

## 📊 Project Structure

```
cdss/
├── main.py                          # Interactive CDSS entry point
├── demo.py                          # Guided demo with examples
├── test_e2e.py                      # End-to-end integration tests
├── test_pipeline.py                 # Smoke tests
├── requirements.txt                 # Python dependencies
│
├── embeddings/
│   ├── __init__.py
│   └── embedding.py                # Vector embedding using SentenceTransformers
│
├── preprocessing/
│   ├── __init__.py
│   ├── parser.py                   # Extract drugs from prescription text
│   └── cleaner.py                  # Clean medication descriptions
│
├── mapping/
│   ├── condition_mapper.py          # Drug → Clinical Condition mapping
│   └── fuzzy_match.py               # Typo correction via fuzzy matching
│
├── vector_db/
│   ├── __init__.py
│   └── store.py                     # In-memory vector store with metadata
│
├── tests/
│   ├── test_vector_store.py         # Unit tests for vector operations
│   └── test_condition_mapper.py      # Unit tests for condition prediction
│
└── data/
    ├── lexicons/
    │   ├── drugs.json               # Known FDA drug names
    │   └── labeled_drugs.json        # Drug-to-condition mappings
    └── medical_dataset/             # Training/validation data
        ├── train/
        ├── validation/
        └── test/
```

## 🔧 Key Components

### 1. **preprocessing/parser.py**
Extracts drug names from unstructured prescription text using regex patterns.

### 2. **mapping/fuzzy_match.py**
Maps user-input drugs to known FDA drug names. Includes:
- Exact matching (fast)
- Fuzzy matching with 85% similarity threshold (typo tolerance)

**Example:**
```python
from mapping.fuzzy_match import correct_drug_list
correct_drug_list(["iboprofen"])  # → ['ibuprofen']
```

### 3. **mapping/condition_mapper.py**
Maps drugs to actual clinical conditions (not pharmacological classes).

**Drug-to-Condition Mapping:**
```python
DRUG_TO_CONDITION = {
    "ibuprofen": ["pain", "inflammation"],
    "metformin": ["diabetes"],
    "paracetamol": ["fever", "pain"],
    # ... more drugs
}
```

**Prediction Logic:**
- Rule-based: Direct drug → condition lookup (weight: 2.0)
- Vector-based: Historical case similarity (weight: similarity score)
- Noise filtering: Removes spurious allergen keywords
- Confidence threshold: 40% of max score (adaptive filtering)

### 4. **vector_db/store.py**
In-memory vector store with metadata preservation.

```python
store = VectorStore()
store.add(embedding_vector, {"drugs": [...], "conditions": [...]})
results = store.search(query_vector, top_k=3, threshold=0.60)
```

### 5. **embeddings/embedding.py**
Uses SentenceTransformer ("all-MiniLM-L6-v2") for semantic embeddings.

## 📈 Test Results

### ✅ Unit Tests: 6/6 PASSED
- Vector store metadata operations
- Condition mapping logic
- Search threshold filtering

### ✅ Functional Tests: 15/15 PASSED
- Fuzzy matching (typo correction)
- Drug extraction from text
- Clinical condition prediction
- Multi-drug reasoning
- Confidence filtering

### ✅ Integration Tests: 3/3 PASSED
- End-to-end pipeline: parse → correct → predict
- Real prescription examples
- Metadata preservation throughout flow

Run tests:
```bash
python -m unittest discover tests -v
python test_e2e.py
```

## 🔨 Recent Fixes (v1.1)

This release fixed critical issues that were preventing proper system operation:

### **FIX 1: Fuzzy Matching for Typo Tolerance** ✅
- **Problem**: "iboprofen" was not matched to "ibuprofen"
- **Solution**: Added `difflib.get_close_matches()` with 85% similarity threshold
- **Impact**: System now handles OCR errors and common misspellings

### **FIX 2: Clinical Conditions Instead of Pharmacological Classes** ✅
- **Problem**: System output "CYCLOOXYGENASE INHIBITORS" instead of "pain"
- **Solution**: Replaced FDA classification mapping with clinical condition mapping
- **Impact**: Predictions are now clinically meaningful and actionable

### **FIX 3: Removed Blocking Warning Logic** ✅
- **Problem**: "⚠️ No FDA-recognized drugs" message blocked inference flow
- **Solution**: Removed blocking conditional; system proceeds with any valid input
- **Impact**: Predictions happen even when fuzzy matching is incomplete

### **FIX 4: Lowered Confidence Threshold** ✅
- **Problem**: Metformin and multi-drug predictions were filtered out
- **Solution**: Changed threshold from 50% → 40% of max score
- **Impact**: More conditions surface when relevant

### **FIX 5: Preserved Vector Metadata** ✅
- **Problem**: Metadata was lost during vector search (empty metadata trap)
- **Solution**: Ensured metadata dict passed through search pipeline
- **Impact**: All case information available during prediction

## 📝 Example Usage

### Interactive Mode
```bash
$ python main.py

Enter Prescription Text (or 'exit'): paracetamol 650mg

🧠 PREDICTED CLINICAL CONDITIONS:
   • FEVER (Score: 2.0)
   • PAIN (Score: 2.0)

🔍 SIMILAR HISTORICAL CASES:
   [0.73] Drugs: paracetamol
```

### Programmatic Usage
```python
from preprocessing.parser import parse_prescription
from mapping.fuzzy_match import correct_drug_list
from mapping.condition_mapper import ConditionMapper

mapper = ConditionMapper()
prescription = "iboprofen 200mg daily"

# Parse
parsed = parse_prescription(prescription)
print(parsed["drugs"])  # ['iboprofen']

# Correct
drugs = correct_drug_list(parsed["drugs"])
print(drugs)  # ['ibuprofen']

# Predict
predictions = mapper.predict(drugs)
for pred in predictions:
    print(f"{pred['condition_label']}: {pred['confidence']}")
# Output:
# pain: 2.0
# inflammation: 2.0
```

## 🔄 Data Flow

```
User Input (Prescription Text)
    ↓
[1] PARSE: Extract drug tokens
    ↓
[2] CORRECT: Fuzzy match to known drugs
    ↓
[3] CLEAN: Remove units, stopwords
    ↓
[4] EMBED: Generate semantic vector
    ↓
[5] SEARCH: Find similar cases in vector store
    ↓
[6] PREDICT: Combine rule-based + vector-based predictions
    ↓
[7] OUTPUT: Clinical conditions + confidence + recommendations
```

## 🛠️ Configuration

Core parameters in respective files:

**fuzzy_match.py**
```python
BLACKLIST = {"pain", "for", "back", "daily", "start", "day", "take", "and"}
cutoff = 0.85  # Fuzzy match similarity threshold
```

**condition_mapper.py**
```python
threshold = 0.4 * max_score  # Confidence filtering
excluded_keywords = ["allergenic extract", "food allergen", "plant allergen"]
```

**vector_db/store.py**
```python
top_k = 3
threshold = 0.60  # Cosine similarity minimum
```

## 🐛 Troubleshooting

### "Dataset not found" error
```bash
Error: Dataset not found. Please check your data/medical_dataset path.
```
**Solution**: Ensure HuggingFace datasets library can download, or provide local dataset.

### "No predictions" for valid drug
Ensure:
1. Drug is in `DRUG_TO_CONDITION` mapping
2. Confidence threshold is reasonable
3. Check `mapping/condition_mapper.py` for noise filters

### Model download issues
```bash
# Set HuggingFace token for faster downloads
export HF_TOKEN="your_token"
python main.py
```

## 📚 Dependencies

| Package | Purpose |
|---------|---------|
| `datasets` | Load medical dataset |
| `numpy` | Vector operations |
| `sentence-transformers` | Semantic embeddings (MiniLM) |
| `torch` | Deep learning backend |

## 📖 Citation & References

- **Embeddings**: [SentenceTransformers](https://www.sbert.net/)
- **Dataset**: [HuggingFace Datasets](https://huggingface.co/datasets)
- **Knowledge Base**: FDA drug classification mappings + RxNorm

## 📝 License

This project is for educational and research purposes.

## 🚀 Next Steps / Roadmap

**Phase 2 (Planned):**
- [ ] Add logging instead of print statements
- [ ] Implement error recovery
- [ ] Separate build_index.py from run_cdss.py
- [ ] Add persistent vector store (pickle/FAISS)

**Phase 3 (Future):**
- [ ] FastAPI REST endpoint
- [ ] Replace with BioBERT (medical domain)
- [ ] Implement confidence calibration
- [ ] Web UI (Streamlit/React)
- [ ] Docker containerization

## 👤 Contact & Support

For issues or questions, check TEST_REPORT.txt for complete validation results.

---

**Status**: ✅ Demo Ready | **Last Updated**: April 2026
