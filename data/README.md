# Data

Nothing in this folder is committed. To recreate it:

1. Download `collectionandqueries.tar.gz` (MS MARCO passage ranking) and extract into `data/raw/`.
   Expected MD5 of the archive: 31644046b18952c1386cd4564ba2ae69.
   You need: `collection.tsv`, `queries.dev.small.tsv`, `qrels.dev.small.tsv`.
2. `python -m src.preprocessing.build_subset --num-queries 1000 --corpus-size 50000 --seed 42`
   writes `data/processed/{corpus,queries,qrels}.tsv`.
3. `python -m src.embeddings.encode` writes `data/processed/{corpus,query}_emb.npy`.