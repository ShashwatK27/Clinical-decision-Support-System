import unittest
from unittest.mock import patch

from mapping.fuzzy_match import correct_drug_list
from preprocessing.parser import parse_prescription
from utils.helpers import sanitize_log_text


class FakeSplit(list):
    def select(self, indices):
        return [self[i] for i in indices]


class TestProjectRegressions(unittest.TestCase):
    def test_parse_prescription_uses_freetext_fallback(self):
        parsed = parse_prescription("Patient should take ibuprofen 200mg daily")
        self.assertEqual(parsed["medications"], ["patient should take ibuprofen 200mg daily"])
        self.assertIn("ibuprofen", parsed["drugs"])

    def test_sanitize_log_text_removes_raw_preview(self):
        summary = sanitize_log_text("metformin 500mg for John Doe")
        self.assertEqual(summary, "[28 chars, 5 words]")
        self.assertNotIn("metformin", summary)
        self.assertNotIn("John", summary)

    def test_fuzzy_cutoff_can_be_overridden(self):
        with patch("mapping.fuzzy_match.KNOWN_DRUGS", {"ibuprofen"}):
            self.assertEqual(correct_drug_list(["ibuprofan"], cutoff=0.95), [])
            self.assertEqual(correct_drug_list(["ibuprofan"], cutoff=0.80), ["ibuprofen"])

    def test_smoke_case_noise_words_are_filtered(self):
        parsed = parse_prescription(
            "<s_ocr> medications: - Amoxicillin 500mg capsules - Metformin 850 mg tabs signature: </s>"
        )
        self.assertEqual(sorted(parsed["drugs"]), ["amoxicillin", "metformin"])

    def test_common_typo_aliases_prefer_canonical_drugs(self):
        corrected = correct_drug_list(["Ibuprophen", "Ciproflaxacin", "metaformin"])
        self.assertEqual(corrected, ["ibuprofen", "ciprofloxacin", "metformin"])

    def test_main_build_store_keeps_conditions_for_hybrid_predictions(self):
        import main

        dataset = {
            "train": FakeSplit([
                {"ground_truth": "ibuprofen 200mg and metformin 500mg"},
            ])
        }
        mapper = type(
            "Mapper",
            (),
            {"knowledge_base": {"ibuprofen": ["pain"], "metformin": ["diabetes"]}},
        )()

        with patch("main.get_embedding", return_value=[1.0, 0.0]):
            store = main.build_store(dataset, mapper, limit=5)

        self.assertEqual(len(store.metadata), 1)
        self.assertEqual(store.metadata[0]["conditions"], ["diabetes", "pain"])


if __name__ == "__main__":
    unittest.main()
