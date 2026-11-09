"""Sanity-check the processed subset: sizes, length distribution, duplicates, eyeball examples.

    python -m src.preprocessing.inspect_data
"""
import numpy as np

from src.config import CORPUS_FILE, PROCESSED_DIR, QRELS_FILE, QUERIES_FILE
from src.data_io import read_corpus, read_qrels, read_queries


def main():
    pids, texts = read_corpus(PROCESSED_DIR / CORPUS_FILE)
    qids, qtexts = read_queries(PROCESSED_DIR / QUERIES_FILE)
    qrels = read_qrels(PROCESSED_DIR / QRELS_FILE)

    print(f"passages: {len(texts)}   queries: {len(qids)}")
    rel_counts = [len(qrels[q]) for q in qids]
    print(f"avg relevant passages / query: {np.mean(rel_counts):.2f} (max {max(rel_counts)})")

    lengths = np.array([len(t.split()) for t in texts])
    print("passage length in words -> "
          f"min {lengths.min()}  median {int(np.median(lengths))}  "
          f"p95 {int(np.percentile(lengths, 95))}  max {lengths.max()}")

    print(f"empty passages: {(lengths == 0).sum()}")
    print(f"duplicate passage texts: {len(texts) - len(set(texts))}")
    print(f"pids sorted & unique: {bool(np.all(np.diff(pids) > 0))}")

    pid_to_row = {int(p): i for i, p in enumerate(pids)}
    print("\n--- 5 examples ---")
    for qid, qtext in list(zip(qids, qtexts))[:5]:
        pid = next(iter(qrels[qid]))
        print(f"Q: {qtext}\nA: {texts[pid_to_row[pid]][:200]}\n")


if __name__ == "__main__":
    main()
