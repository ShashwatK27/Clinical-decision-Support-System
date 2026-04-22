"""
VectorStore — safe, pickle-free vector database.

Vectors are saved as a NumPy .npy file; metadata is saved as a plain JSON
file. Both live inside a single directory (default: vector_store/).

Performance note:
    The full matrix is pre-stacked once at add() / load() time so that
    search() never has to rebuild it from a list of individual arrays.

Why not pickle?
    pickle.load() executes arbitrary Python code contained in the file,
    which is a Remote Code Execution (RCE) risk if the file is ever
    tampered with or replaced by a malicious actor. NumPy + JSON are
    safe, transparent, and inspectable.
"""

import json
import numpy as np
from pathlib import Path


class VectorStore:
    def __init__(self):
        # _matrix: shape (N, D) pre-stacked array — None when empty
        self._matrix: np.ndarray | None = None
        self.metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def vectors(self) -> list:
        """Return vectors as a list (used for len() checks and iteration)."""
        if self._matrix is None:
            return []
        return list(self._matrix)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, vector: np.ndarray, meta_dict: dict):
        """Append a single vector and its associated metadata."""
        vec = np.array(vector, dtype=np.float32)
        if self._matrix is None:
            self._matrix = vec.reshape(1, -1)
        else:
            self._matrix = np.vstack([self._matrix, vec.reshape(1, -1)])
        self.metadata.append(meta_dict)

    def search(self, query_vector: np.ndarray, top_k: int = 3, threshold: float = 0.5):
        """
        Return up to top_k *diverse* entries whose cosine similarity to
        query_vector meets or exceeds the threshold.

        The matrix is pre-stacked at load/add time, so this is a single
        fast dot-product — no list-to-array conversion on every call.

        Deduplication: only the highest-scoring result per unique drug
        combination is returned, preventing the same prescription from
        appearing multiple times.
        """
        if self._matrix is None:
            return []

        q = np.array(query_vector, dtype=np.float32)
        scores = self._matrix.dot(q)           # shape (N,) — fast BLAS call

        sorted_indices = scores.argsort()[::-1]

        seen_drug_sets: set = set()
        results = []

        for i in sorted_indices:
            score = float(scores[i])
            if threshold is not None and score < threshold:
                break  # sorted descending — all remaining will also be below threshold

            meta = self.metadata[i]
            drug_key = frozenset(meta.get("drugs", []) if isinstance(meta, dict) else [])

            if drug_key in seen_drug_sets:
                continue

            seen_drug_sets.add(drug_key)
            results.append((meta, score))

            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Persistence (safe: numpy + JSON, NOT pickle)
    # ------------------------------------------------------------------

    def save(self, dirpath: str = "vector_store") -> str:
        """
        Persist vectors and metadata to disk.

        Creates a directory at dirpath/ containing:
            vectors.npy   — stacked NumPy float32 array of all embedding vectors
            metadata.json — list of metadata dicts

        Returns the absolute path to the directory.
        """
        path = Path(dirpath)
        path.mkdir(parents=True, exist_ok=True)

        matrix = self._matrix if self._matrix is not None else np.array([], dtype=np.float32)
        np.save(str(path / "vectors.npy"), matrix)

        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        return str(path.resolve())

    @classmethod
    def load(cls, dirpath: str = "vector_store"):
        """
        Load a vector store saved by save().

        Returns a VectorStore instance, or None if the directory does not exist.

        Note:
            Old .pkl files are NOT supported. Rebuild with:
                python pipeline/build_db.py
        """
        if str(dirpath).endswith(".pkl"):
            raise ValueError(
                "Legacy .pkl vector stores are no longer supported (RCE risk). "
                "Rebuild with: python pipeline/build_db.py"
            )

        path = Path(dirpath)
        if not path.exists() or not (path / "vectors.npy").exists():
            return None

        store = cls()

        raw = np.load(str(path / "vectors.npy"), allow_pickle=False)
        if raw.ndim == 2 and raw.shape[0] > 0:
            store._matrix = raw.astype(np.float32)   # keep it float32 for BLAS speed

        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                store.metadata = json.load(f)

        return store
