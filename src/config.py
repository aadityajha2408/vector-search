"""Central place for filesystem paths so no script hardcodes them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# Files produced by src/preprocessing/build_subset.py
CORPUS_FILE = "corpus.tsv"      # pid <TAB> text, sorted by pid. Line i == embedding row i.
QUERIES_FILE = "queries.tsv"    # qid <TAB> text
QRELS_FILE = "qrels.tsv"        # qid <TAB> 0 <TAB> pid <TAB> 1

# Files produced by src/embeddings/encode.py
CORPUS_EMB_FILE = "corpus_emb.npy"
QUERY_EMB_FILE = "query_emb.npy"
