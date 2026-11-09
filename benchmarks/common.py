"""Shared helpers: load real embeddings if they exist, otherwise make synthetic ones."""
import numpy as np

from src.config import CORPUS_EMB_FILE, PROCESSED_DIR, QUERY_EMB_FILE


def synthetic(n, dim, n_queries, seed=0, n_clusters=50):
    """Clustered, L2-normalised vectors. Real text embeddings are clustered; uniform random
    high-dimensional vectors are an unrealistically hard (and meaningless) ANN test."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)

    def draw(m):
        x = centers[rng.integers(0, n_clusters, m)] + 0.6 * rng.standard_normal((m, dim)).astype(np.float32)
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

    return draw(n), draw(n_queries)


def load_vectors(n, dim, n_queries, use_real=True):
    corpus_path, query_path = PROCESSED_DIR / CORPUS_EMB_FILE, PROCESSED_DIR / QUERY_EMB_FILE
    if use_real and corpus_path.exists() and query_path.exists():
        corpus, queries = np.load(corpus_path), np.load(query_path)
        print(f"using real embeddings: corpus {corpus.shape}, queries {queries.shape}")
        return corpus, queries
    corpus, queries = synthetic(n, dim, n_queries)
    print(f"using SYNTHETIC clustered vectors: corpus {corpus.shape}, queries {queries.shape}")
    return corpus, queries
