# Mini Vector Search Engine

Hybrid semantic search over a 50K-passage MS MARCO subset, built around a from-scratch
C++ vector search core (exact top-k + HNSW) with Python bindings.

> Fill in the numbers below from YOUR runs. Do not leave placeholders in the final README.

## Results

| System | MRR@10 | nDCG@10 | Recall@100 | ms/query |
|---|---|---|---|---|
| TF-IDF | | | | |
| BM25 | | | | |
| Dense exact (MiniLM) | | | | |
| Hybrid RRF (BM25 + dense) | | | | |

ANN (HNSW, M=16, efConstruction=200, 1 thread): see `results/recall_qps.png`.

| Index | ef | Recall@10 | QPS |
|---|---|---|---|
| ours | | | |
| hnswlib | | | |

Evaluation set: 1,000 sampled dev queries, 50K-passage corpus. Numbers are **not comparable**
to the public MS MARCO leaderboard (corpus is ~170x smaller).

## Architecture

See `docs/architecture.md`.

## Quick start

```
python -m venv venv && .\venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
# download MS MARCO files into data/raw/ (see data/README.md)
python -m src.preprocessing.build_subset --num-queries 1000 --corpus-size 50000 --seed 42
python -m src.preprocessing.inspect_data
python -m src.embeddings.encode
python setup.py build_ext --inplace                        # C++ extension
python -m src.evaluation.evaluate --k 100
python -m benchmarks.bench_exact
python -m benchmarks.bench_ann --indexes ours hnswlib
python -m benchmarks.plot
pytest
```

## What failed / limitations

Write this honestly. Interviewers remember it.
