import unittest
import numpy as np
from vector_db.store import VectorStore


class TestVectorStore(unittest.TestCase):
    def test_add_and_search_returns_metadata(self):
        store = VectorStore()
        store.add(np.array([1.0, 0.0]), {"id": "case1", "drugs": ["aspirin"]})
        store.add(np.array([0.0, 1.0]), {"id": "case2", "drugs": ["ibuprofen"]})

        results = store.search(np.array([1.0, 0.0]), top_k=1, threshold=0.0)

        self.assertEqual(len(results), 1)
        metadata, score = results[0]
        self.assertEqual(metadata["id"], "case1")
        self.assertAlmostEqual(score, 1.0)

    def test_search_threshold_filters_low_similarity(self):
        store = VectorStore()
        store.add(np.array([1.0, 0.0]), {"id": "case1"})

        results = store.search(np.array([0.0, 1.0]), top_k=1, threshold=0.5)
        self.assertEqual(results, [])

    def test_empty_store_search_returns_empty_list(self):
        store = VectorStore()
        results = store.search(np.array([1.0, 0.0]))
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
