# Data

The data files are not committed (only this README is). To recreate them:

1. Download `collectionandqueries.tar.gz` (MS MARCO passage ranking, about 1 GB) and extract it into `data/raw/`:
   `https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz`
   (the older `msmarco.blob.core.windows.net` address no longer allows public access).
   Expected MD5 of the archive: 31644046b18952c1386cd4564ba2ae69.
   You need: `collection.tsv`, `queries.dev.small.tsv`, `qrels.dev.small.tsv`.
2. `python -m src.preprocessing.build_subset --num-queries 1000 --corpus-size 50000 --seed 42`
   writes `data/processed/{corpus,queries,qrels}.tsv`.
3. `python -m src.embeddings.encode` writes `data/processed/{corpus,query}_emb.npy`
   (about 1 hour on CPU).