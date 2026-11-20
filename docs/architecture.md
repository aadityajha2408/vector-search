# Architecture

```
data/raw (MS MARCO)                                   data/processed
 collection.tsv, queries.dev.small.tsv,  --build_subset-->  corpus.tsv (sorted by pid), queries.tsv, qrels.tsv
 qrels.dev.small.tsv                                          |
                                                              +--> src.embeddings.encode --> corpus_emb.npy, query_emb.npy
                                                              |        (MiniLM, L2-normalised, row i == line i of corpus.tsv)
                                                              v
   +--------------------+     +--------------------------+     +---------------------------+
   | Sparse retrieval   |     | Dense retrieval          |     | ANN (HNSW)                |
   | BM25 (own impl.)   |     | DenseIndex               |     | ours (C++) / hnswlib /    |
   | TF-IDF             |     |  - numpy backend (BLAS)  |     | faiss  -- common API      |
   +---------+----------+     |  - C++ ExactIndex        |     +-------------+-------------+
             |                +------------+-------------+                   |
             +----------------+-----------+--------------------------------+
                              v
                    Reciprocal Rank Fusion  (src/retrieval/fusion.py)
                              v
                 Metrics: MRR@10, nDCG@10, Recall@K  (src/evaluation)
                 Benchmarks: latency, recall-vs-QPS  (benchmarks/)
```

## Key design decisions

- **Row index == line number == embedding row.** Search code returns row numbers; the evaluator maps rows to pids via the corpus file order. Never reorder corpus.tsv.
- **Vectors are L2-normalised**, so inner product equals cosine similarity and the C++ code needs only a dot product.
- **Exact C++ search** keeps a size-k min-heap per query (O(N log k)), tiles queries in blocks of 8 so each corpus vector is loaded once per block, and uses AVX2 intrinsics for the dot product.
- **HNSW**: multi-layer proximity graph; greedy descent on sparse upper layers, beam search (`ef`) on layer 0. Neighbour selection uses the diversity heuristic (ablation: `--no-heuristic`).
- **RRF** fuses ranks, not scores, because BM25 scores and cosine similarities are on different scales. Per-system weights are tuned on half the queries and reported on the other half.

## Known limitations / next optimisations

- Neighbour lists are `vector<vector<vector<uint32_t>>>`: easy to read, poor cache locality. A flat fixed-stride layout is the first thing to try.
- Index construction is single-threaded.
- HNSW stores full float32 vectors; product or scalar quantisation would cut memory.
- Exact search is not faster than a BLAS matrix multiply in every setting. Measure, do not assume.