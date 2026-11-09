"""Embed the corpus and queries with a sentence-transformers model and save .npy files.

    python -m src.embeddings.encode --model sentence-transformers/all-MiniLM-L6-v2

Output rows follow the order of corpus.tsv / queries.tsv. Vectors are L2-normalised so that
inner product equals cosine similarity. Expect minutes (not seconds) for 50K passages on CPU.
"""
import argparse
import time

import numpy as np

from src.config import (CORPUS_EMB_FILE, CORPUS_FILE, PROCESSED_DIR, QUERIES_FILE, QUERY_EMB_FILE)
from src.data_io import read_corpus, read_queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None, help="cpu / cuda; default lets the library choose")
    args = p.parse_args()

    from sentence_transformers import SentenceTransformer  # imported late: heavy

    model = SentenceTransformer(args.model, device=args.device)
    print(f"model={args.model} dim={model.get_sentence_embedding_dimension()} "
          f"max_seq_length={model.max_seq_length} tokens (longer passages are truncated)")

    _, texts = read_corpus(PROCESSED_DIR / CORPUS_FILE)
    _, queries = read_queries(PROCESSED_DIR / QUERIES_FILE)

    def encode(items, label):
        t = time.perf_counter()
        emb = model.encode(items, batch_size=args.batch_size, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)
        print(f"{label}: {emb.shape} in {time.perf_counter() - t:.0f}s")
        return emb.astype(np.float32)

    np.save(PROCESSED_DIR / QUERY_EMB_FILE, encode(queries, "queries"))
    np.save(PROCESSED_DIR / CORPUS_EMB_FILE, encode(texts, "corpus"))


if __name__ == "__main__":
    main()