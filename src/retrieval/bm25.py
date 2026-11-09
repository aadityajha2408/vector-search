"""BM25 implemented from scratch on top of scipy sparse matrices.

score(q, d) = sum over query terms t of  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
We precompute that per-(doc, term) weight once, so a query is just a sparse matrix product.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from src.retrieval.utils import topk_from_scores


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75, stop_words="english"):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", stop_words=stop_words)
        self.weights_t = None  # (n_terms, n_docs) CSR

    def fit(self, texts):
        tf = self.vectorizer.fit_transform(texts).tocsr().astype(np.float32)  # (n_docs, n_terms)
        n_docs, n_terms = tf.shape

        df = np.bincount(tf.indices, minlength=n_terms)  # docs containing each term
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = max(float(doc_len.mean()), 1e-9)

        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        denom = tf.data + self.k1 * (1.0 - self.b + self.b * doc_len[rows] / avg_len)
        data = idf[tf.indices] * tf.data * (self.k1 + 1.0) / denom

        weights = sp.csr_matrix((data.astype(np.float32), tf.indices, tf.indptr), shape=(n_docs, n_terms))
        self.weights_t = weights.T.tocsr()
        return self

    def search(self, queries, k: int = 100, chunk: int = 64):
        q = self.vectorizer.transform(queries).tocsr().astype(np.float32)
        q.data[:] = 1.0  # binary query-term presence
        all_ids, all_scores = [], []
        for start in range(0, q.shape[0], chunk):
            scores = (q[start:start + chunk] @ self.weights_t).toarray()
            ids, vals = topk_from_scores(scores, k)
            all_ids.append(ids)
            all_scores.append(vals)
        return np.vstack(all_ids), np.vstack(all_scores)