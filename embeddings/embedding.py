import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sentence-transformers is required for embeddings. "
                "Install project dependencies before running the CDSS pipeline."
            ) from exc
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embedding(text: str) -> np.ndarray:
    """
    Convert a single text into embedding vector
    """
    return _get_model().encode(text, normalize_embeddings=True)


def get_embeddings_batch(texts: list) -> np.ndarray:
    """
    Convert multiple texts into embeddings (FASTER)
    """
    return _get_model().encode(texts, normalize_embeddings=True)


