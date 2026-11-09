"""TF-IDF + cosine baseline (scikit-learn vectorizer, our own scoring/top-k)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.retrieval.utils import topk_from_scores


class TfIdf:
    def __init__(self, stop_words="english"):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words, sublinear_tf=True, dtype=np.float32)
        self.matrix_t = None

    def fit(self, texts):
        self.matrix_t = self.vectorizer.fit_transform(texts).T.tocsr()  # rows are L2-normalised
        return self

    def search(self, queries, k: int = 100, chunk: int = 64):
        q = self.vectorizer.transform(queries).tocsr()
        all_ids, all_scores = [], []
        for start in range(0, q.shape[0], chunk):
            scores = (q[start:start + chunk] @ self.matrix_t).toarray()
            ids, vals = topk_from_scores(scores, k)
            all_ids.append(ids)
            all_scores.append(vals)
        return np.vstack(all_ids), np.vstack(all_scores)