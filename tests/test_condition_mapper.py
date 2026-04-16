import unittest
from mapping.condition_mapper import ConditionMapper


class TestConditionMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = ConditionMapper()

    def test_predict_returns_rule_based_conditions(self):
        result = self.mapper.predict(["ibuprofen"])
        self.assertTrue(any(item["condition_label"] == "pain" for item in result))
        self.assertTrue(any(item["condition_label"] == "inflammation" for item in result))

    def test_predict_includes_vector_conditions_when_present(self):
        meta = {"conditions": ["proton pump inhibitor", "other condition"]}
        result = self.mapper.predict(["unknown-drug"], vector_results=[(meta, 0.8)])
        self.assertTrue(any(item["condition_label"] == "proton pump inhibitor" for item in result))
        self.assertTrue(any(item["condition_label"] == "other condition" for item in result))

    def test_predict_filters_allergen_noise_from_vector_results(self):
        meta = {"conditions": ["non-standardized plant allergenic extract"]}
        result = self.mapper.predict(["unknown-drug"], vector_results=[(meta, 0.9)])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
