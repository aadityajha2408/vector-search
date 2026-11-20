"""Live demo: run the SAME query through BM25, dense search, hybrid fusion and our C++ HNSW.

Run from the repo root:
    streamlit run app/demo.py

First start takes ~10-20 s (loads data, builds BM25, loads the embedding model).
Ticking "Enable HNSW" builds our C++ HNSW index once (1-2 minutes) and then keeps it in memory.
"""
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import src...` works

import numpy as np
import streamlit as st

from src.config import CORPUS_EMB_FILE, CORPUS_FILE, PROCESSED_DIR, QRELS_FILE, QUERIES_FILE
from src.data_io import read_corpus, read_qrels, read_queries
from src.retrieval.bm25 import BM25
from src.retrieval.dense import DenseIndex, cpp_available
from src.retrieval.fusion import rrf_single

DEPTH = 100  # how deep each system searches before fusion / ranking diagnostics
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="Vector search demo", layout="wide")


@st.cache_resource(show_spinner="Loading corpus and building BM25...")
def load_data():
    pids, texts = read_corpus(PROCESSED_DIR / CORPUS_FILE)
    qids, qtexts = read_queries(PROCESSED_DIR / QUERIES_FILE)
    qrels = read_qrels(PROCESSED_DIR / QRELS_FILE)
    emb = np.load(PROCESSED_DIR / CORPUS_EMB_FILE)
    return {
        "pids": pids, "texts": texts, "qids": qids, "qtexts": qtexts, "qrels": qrels, "emb": emb,
        "bm25": BM25().fit(texts),
        "dense_numpy": DenseIndex("numpy").add(emb),
        "dense_cpp": DenseIndex("cpp").add(emb) if cpp_available() else None,
    }


@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    from sentence_transformers import SentenceTransformer  # heavy import, only when needed
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Building HNSW index (1-2 minutes, only once)...")
def build_hnsw(_emb, M, ef_construction):  # leading underscore: Streamlit will not try to hash the array
    from src.retrieval.ann import OurHNSW
    return OurHNSW(_emb.shape[1], M=M, ef_construction=ef_construction).build(_emb)


def encode(model, text):
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


def median_ms(fn, repeats=5):
    fn()  # warm-up run, not counted
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(1000 * (time.perf_counter() - t))
    return statistics.median(times)


def esc(text):
    """Escape Markdown/LaTeX characters: passages are full of '$', '*' and '_'."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!$<>~|])", r"\\\1", text)


def first_relevant_rank(rows, pids, relevant):
    for i, row in enumerate(rows):
        if int(pids[int(row)]) in relevant:
            return i + 1
    return None


def render_column(title, rows, notes, latency_ms, data, relevant, top_k):
    st.subheader(title)
    rank = first_relevant_rank(rows, data["pids"], relevant) if relevant else None
    parts = [f"{latency_ms:.1f} ms"]
    if relevant:
        parts.append(f"first relevant: #{rank}" if rank else f"relevant not in top {len(rows)}")
    st.caption(" · ".join(parts))
    for i in range(min(top_k, len(rows))):
        row = int(rows[i])
        pid = int(data["pids"][row])
        badge = "✅ " if pid in relevant else ""
        text = data["texts"][row]
        snippet = text if len(text) <= 260 else text[:260] + "..."
        st.markdown(f"**{i + 1}.** {badge}`pid {pid}` · {notes[i]}")
        st.markdown(esc(snippet))


def main():
    st.title("Mini vector search engine")
    st.caption("50K-passage MS MARCO subset · BM25 vs dense (MiniLM) vs hybrid RRF · C++ exact search and HNSW")

    if not (PROCESSED_DIR / CORPUS_EMB_FILE).exists():
        st.error("data/processed is missing the corpus or embeddings. Run build_subset and encode first (see README).")
        st.stop()

    data = load_data()
    model = load_model()
    dense = data["dense_cpp"] or data["dense_numpy"]
    dense_label = "Dense exact (C++)" if data["dense_cpp"] else "Dense exact (NumPy)"

    with st.sidebar:
        st.header("Controls")
        top_k = st.slider("Results shown per system", 3, 10, 5)
        weight = st.slider("Hybrid: dense weight (BM25 weight = 1)", 1.0, 8.0, 1.0, 0.5,
                           help="1.0 = equal-weight RRF. Higher trusts the dense ranking more.")
        hnsw_on = st.checkbox("Enable HNSW (our C++ index)", value=False, disabled=not cpp_available(),
                              help="Builds the index once (1-2 minutes), then it stays in memory.")
        ef = st.slider("HNSW ef (search effort)", 10, 400, 40, disabled=not hnsw_on)
        if not cpp_available():
            st.info("C++ extension not built, so only the NumPy backend is available.")

    sample = st.selectbox("Example query from the evaluation set (with known relevant passage)",
                          range(len(data["qtexts"])), format_func=lambda i: data["qtexts"][i])
    typed = st.text_input("...or type your own query (no relevance labels)")
    query = typed.strip() or data["qtexts"][sample]
    relevant = set() if typed.strip() else data["qrels"][data["qids"][sample]]
    st.markdown(f"**Query:** {esc(query)}")

    q_emb = encode(model, query)
    bm25_ids, bm25_scores = data["bm25"].search([query], DEPTH)
    dense_ids, dense_scores = dense.search(q_emb, DEPTH, num_threads=1)
    bm25_rank = {int(r): i + 1 for i, r in enumerate(bm25_ids[0])}
    dense_rank = {int(r): i + 1 for i, r in enumerate(dense_ids[0])}
    fused_ids = rrf_single([bm25_ids[0], dense_ids[0]], top_k=DEPTH, weights=[1.0, weight])

    t_bm25 = median_ms(lambda: data["bm25"].search([query], top_k))
    t_dense = median_ms(lambda: dense.search(q_emb, top_k, num_threads=1))

    columns = st.columns(4 if hnsw_on else 3)
    with columns[0]:
        render_column("BM25", bm25_ids[0], [f"score {s:.2f}" for s in bm25_scores[0]], t_bm25, data, relevant, top_k)
    with columns[1]:
        render_column(dense_label, dense_ids[0], [f"cosine {s:.3f}" for s in dense_scores[0]], t_dense, data, relevant, top_k)
    with columns[2]:
        notes = [f"BM25 #{bm25_rank.get(int(r), '>' + str(DEPTH))} · dense #{dense_rank.get(int(r), '>' + str(DEPTH))}"
                 for r in fused_ids]
        render_column("Hybrid RRF", fused_ids, notes, t_bm25 + t_dense, data, relevant, top_k)

    speed = [{"Step": "Query embedding (MiniLM, CPU)", "Median ms": round(median_ms(lambda: encode(model, query)), 2), "Note": ""},
             {"Step": "BM25 search", "Median ms": round(t_bm25, 2), "Note": "includes tokenising the query"},
             {"Step": "Dense exact, NumPy (BLAS)", "Median ms": round(median_ms(lambda: data["dense_numpy"].search(q_emb, top_k)), 2), "Note": "scans all 50,000 vectors"}]
    if data["dense_cpp"]:
        speed.append({"Step": "Dense exact, C++ (1 query = 1 thread)",
                      "Median ms": round(median_ms(lambda: data["dense_cpp"].search(q_emb, top_k, num_threads=1)), 2),
                      "Note": "scans all 50,000 vectors"})

    if hnsw_on:
        hnsw = build_hnsw(data["emb"], 16, 200)
        hnsw_ids, hnsw_scores = hnsw.search(q_emb, top_k, ef, 1)
        t_hnsw = median_ms(lambda: hnsw.search(q_emb, top_k, ef, 1))
        overlap = len(set(hnsw_ids[0].tolist()) & set(dense_ids[0][:top_k].tolist()))
        with columns[3]:
            render_column(f"HNSW ours (ef={ef})", hnsw_ids[0], [f"cosine {s:.3f}" for s in hnsw_scores[0]],
                          t_hnsw, data, relevant, top_k)
            st.caption(f"found {overlap} of the {top_k} exact nearest neighbours")
        speed.append({"Step": f"HNSW, C++ (ef={ef})", "Median ms": round(t_hnsw, 2),
                      "Note": f"approximate: {overlap}/{top_k} of exact top-{top_k}"})

    st.divider()
    st.subheader("Speed for this query")
    st.caption("Median of 5 runs after a warm-up. Single query, so multithreading does not help here.")
    st.dataframe(speed, hide_index=True, width="stretch")


main()