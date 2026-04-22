"""
End-to-end test suite for CDSS system — converted to proper unittest.
Tests the complete pipeline: parsing → fuzzy matching → embedding → prediction.

Run with:
    python -m pytest test_e2e.py -v
    # or
    python -m unittest test_e2e -v
"""

import unittest
import numpy as np

from preprocessing.parser import parse_prescription
from preprocessing.cleaner import clean_medications, build_embedding_text
from mapping.fuzzy_match import correct_drug_list
from mapping.condition_mapper import ConditionMapper
from embeddings.embedding import get_embedding
from vector_db.store import VectorStore


class TestFuzzyMatching(unittest.TestCase):
    """Issue #7 — typo correction via difflib fuzzy matching."""

    def test_typo_iboprofen(self):
        result = correct_drug_list(["iboprofen"])
        self.assertTrue(len(result) > 0, "Should fuzzy-match 'iboprofen' to 'ibuprofen'")
        self.assertEqual(result[0], "ibuprofen")

    def test_typo_metaformin(self):
        result = correct_drug_list(["metaformin"])
        self.assertTrue(len(result) > 0, "Should fuzzy-match 'metaformin' to 'metformin'")
        self.assertEqual(result[0], "metformin")

    def test_exact_match(self):
        result = correct_drug_list(["paracetamol"])
        self.assertIn("paracetamol", result)

    def test_blacklisted_words_excluded(self):
        result = correct_drug_list(["pain", "for", "daily"])
        self.assertEqual(result, [], "Blacklisted words should return empty list")


class TestDrugExtraction(unittest.TestCase):
    """Issue #8, #9 — parsing pipeline (structured and free-text)."""

    def test_freetext_ibuprofen(self):
        parsed = parse_prescription("Patient should take ibuprofen 200mg daily")
        corrected = correct_drug_list(parsed["drugs"])
        self.assertIn("ibuprofen", corrected)

    def test_freetext_multi_drug_with_typos(self):
        parsed = parse_prescription("iboprofen and metaformin 500mg")
        corrected = correct_drug_list(parsed["drugs"])
        self.assertIn("ibuprofen", corrected)
        self.assertIn("metformin", corrected)

    def test_freetext_paracetamol(self):
        parsed = parse_prescription("paracetamol 650 mg for fever")
        corrected = correct_drug_list(parsed["drugs"])
        self.assertIn("paracetamol", corrected)

    def test_hyphenated_drug_not_destroyed(self):
        """Issue #7 — hyphens must survive the token cleaning regex."""
        from preprocessing.parser import extract_drug_names
        result = extract_drug_names(["co-amoxiclav 625mg"])
        # The hyphenated token should be kept intact, not split or stripped
        self.assertIn("co-amoxiclav", result)


class TestConditionPrediction(unittest.TestCase):
    """Issue #10 — condition mapper returns correct clinical labels."""

    def setUp(self):
        self.mapper = ConditionMapper()

    def test_ibuprofen_predicts_pain_and_inflammation(self):
        predictions = self.mapper.predict(["ibuprofen"])
        labels = [p["condition_label"] for p in predictions]
        self.assertIn("pain", labels)
        self.assertIn("inflammation", labels)

    def test_metformin_predicts_diabetes(self):
        predictions = self.mapper.predict(["metformin"])
        labels = [p["condition_label"] for p in predictions]
        self.assertIn("diabetes", labels)

    def test_paracetamol_predicts_fever_and_pain(self):
        predictions = self.mapper.predict(["paracetamol"])
        labels = [p["condition_label"] for p in predictions]
        self.assertIn("fever", labels)
        self.assertIn("pain", labels)

    def test_multi_drug_prediction(self):
        predictions = self.mapper.predict(["ibuprofen", "metformin"])
        labels = [p["condition_label"] for p in predictions]
        self.assertIn("pain", labels)
        self.assertIn("diabetes", labels)

    def test_prediction_has_source_field(self):
        """Issue #6 — every result must have an explicit 'source' field."""
        predictions = self.mapper.predict(["ibuprofen"])
        for pred in predictions:
            self.assertIn("source", pred, "Each prediction must have a 'source' field")
            self.assertIn(pred["source"], {"rule-based", "vector", "both"})

    def test_unknown_drug_returns_empty(self):
        predictions = self.mapper.predict(["xyzunknowndrug99"])
        self.assertEqual(predictions, [])


class TestVectorStore(unittest.TestCase):
    """Issue #5 — threshold must be respected; metadata must be preserved."""

    def setUp(self):
        self.store = VectorStore()
        self.store.add(np.array([1.0, 0.0, 0.0]), {"drugs": ["ibuprofen"], "conditions": ["pain"]})
        self.store.add(np.array([0.9, 0.1, 0.0]), {"drugs": ["metformin"], "conditions": ["diabetes"]})

    def test_metadata_preserved(self):
        results = self.store.search(np.array([1.0, 0.0, 0.0]), top_k=2, threshold=0.5)
        self.assertEqual(len(results), 2)
        drugs_in_results = [meta["drugs"][0] for meta, _ in results]
        self.assertIn("ibuprofen", drugs_in_results)

    def test_threshold_respected_no_fallback(self):
        """Issue #5 — very high threshold should return empty, not junk results."""
        results = self.store.search(np.array([1.0, 0.0, 0.0]), top_k=2, threshold=0.999)
        # Only the perfect match [1.0, 0.0, 0.0] dot [1.0, 0.0, 0.0] = 1.0 passes
        self.assertEqual(len(results), 1)

    def test_zero_similarity_returns_empty(self):
        """Orthogonal vector should match nothing at a reasonable threshold."""
        results = self.store.search(np.array([0.0, 0.0, 1.0]), top_k=2, threshold=0.5)
        self.assertEqual(results, [])

    def test_scores_are_python_float(self):
        """Scores must be plain Python floats (not np.float32) for JSON serialisation."""
        results = self.store.search(np.array([1.0, 0.0, 0.0]), top_k=1, threshold=0.5)
        for _, score in results:
            self.assertIsInstance(score, float)


class TestFullPipeline(unittest.TestCase):
    """Issue #5/#8 — full Parse → Correct → Predict pipeline."""

    def setUp(self):
        self.mapper = ConditionMapper()

    def _run(self, prescription):
        parsed = parse_prescription(prescription)
        corrected = correct_drug_list(parsed["drugs"])
        predictions = self.mapper.predict(corrected)
        return corrected, predictions

    def test_typo_prescription_predicts_conditions(self):
        corrected, predictions = self._run("iboprofen 200mg and metaformin 500mg")
        self.assertIn("ibuprofen", corrected)
        self.assertIn("metformin", corrected)
        self.assertTrue(len(predictions) > 0, "Should predict at least one condition")

    def test_single_drug(self):
        _, predictions = self._run("paracetamol 650 mg")
        labels = [p["condition_label"] for p in predictions]
        self.assertIn("fever", labels)

    def test_natural_language_input(self):
        corrected, predictions = self._run("Patient takes ibuprofen daily")
        self.assertIn("ibuprofen", corrected)
        self.assertTrue(len(predictions) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
