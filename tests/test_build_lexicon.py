import unittest

from scripts.build_lexicon import looks_like_noise, sanitize_terms


class TestBuildLexicon(unittest.TestCase):
    def test_noise_filters_obvious_cosmetic_terms(self):
        self.assertTrue(looks_like_noise("broad spectrum sunscreen spf 50"))
        self.assertTrue(looks_like_noise("anti dandruff shampoo and conditioner"))
        self.assertTrue(looks_like_noise("vitamin c toner"))

    def test_noise_keeps_reasonable_drug_terms(self):
        self.assertFalse(looks_like_noise("ibuprofen"))
        self.assertFalse(looks_like_noise("hydrocortisone cream"))
        self.assertFalse(looks_like_noise("ciprofloxacin"))

    def test_sanitize_terms_drops_noise_and_normalizes(self):
        clean, removed = sanitize_terms([
            "Ibuprofen",
            "Hydrocortisone Cream",
            "Broad Spectrum Sunscreen SPF 50",
            "Anti Dandruff Shampoo And Conditioner",
        ])
        self.assertEqual(clean, ["hydrocortisone cream", "ibuprofen"])
        self.assertIn("broad spectrum sunscreen spf 50", removed)


if __name__ == "__main__":
    unittest.main()
