from sentence_transformers import SentenceTransformer
import numpy as np

# Load model ONCE (very important)
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> np.ndarray:
    """
    Convert a single text into embedding vector
    """
    return model.encode(text, normalize_embeddings=True)


def get_embeddings_batch(texts: list) -> np.ndarray:
    """
    Convert multiple texts into embeddings (FASTER)
    """
    return model.encode(texts, normalize_embeddings=True)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    """
    return float(np.dot(vec1, vec2))

