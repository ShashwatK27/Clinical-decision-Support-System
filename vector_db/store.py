import numpy as np
import pickle
from pathlib import Path


class VectorStore:
    def __init__(self):
        self.vectors = []
        self.metadata = [] # Store dictionaries instead of just text

    def add(self, vector, meta_dict):
        self.vectors.append(vector)
        self.metadata.append(meta_dict)

    def search(self, query_vector, top_k=3, threshold=0.5):
        if not self.vectors:
            return []

        vectors = np.array(self.vectors)
        scores = np.dot(vectors, query_vector)

        # Get indices ordered by similarity
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for i in top_indices:
            if threshold is None or scores[i] >= threshold:
                results.append((self.metadata[i], scores[i]))

        # If no results meet the threshold, return the top_k weakest matches instead
        if not results:
            results = [(self.metadata[i], scores[i]) for i in top_indices]

        return results

    def save(self, filepath):
        """Persist the vector store to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectors': self.vectors,
                'metadata': self.metadata
            }, f)
        return str(path)

    @classmethod
    def load(cls, filepath):
        """Load a saved vector store from disk."""
        path = Path(filepath)
        if not path.exists():
            return None
        with open(path, 'rb') as f:
            data = pickle.load(f)
        store = cls()
        store.vectors = data['vectors']
        store.metadata = data['metadata']
        return store
