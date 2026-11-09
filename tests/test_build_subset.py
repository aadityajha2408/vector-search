"""End-to-end test of build_subset on a tiny fake MS MARCO, including reproducibility."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_fake_raw(raw: Path, n_passages=300, n_queries=40):
    raw.mkdir(parents=True)
    with open(raw / "collection.tsv", "w", encoding="utf-8", newline="\n") as f:
        for i in range(n_passages):
            f.write(f"{i}\tpassage number {i} about topic {i % 7}\n")
    with open(raw / "queries.dev.small.tsv", "w", encoding="utf-8", newline="\n") as f:
        for q in range(n_queries):
            f.write(f"{1000 + q}\t  query {q} topic {q % 7}\n")  # leading spaces on purpose
    with open(raw / "qrels.dev.small.tsv", "w", encoding="utf-8", newline="\n") as f:
        for q in range(n_queries):
            f.write(f"{1000 + q}\t0\t{(q * 7) % n_passages}\t1\n")


def run(raw, out, seed):
    cmd = [sys.executable, "-m", "src.preprocessing.build_subset", "--raw-dir", str(raw), "--out-dir", str(out),
           "--num-queries", "10", "--corpus-size", "60", "--collection-size", "300", "--seed", str(seed)]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)


def test_subset_is_correct_and_reproducible(tmp_path):
    raw = tmp_path / "raw"
    make_fake_raw(raw)
    run(raw, tmp_path / "a", seed=1)
    run(raw, tmp_path / "b", seed=1)
    run(raw, tmp_path / "c", seed=2)

    for name in ("corpus.tsv", "queries.tsv", "qrels.tsv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert (tmp_path / "a" / "queries.tsv").read_bytes() != (tmp_path / "c" / "queries.tsv").read_bytes()

    corpus_pids = [int(l.split("\t")[0]) for l in (tmp_path / "a" / "corpus.tsv").read_text().splitlines()]
    assert len(corpus_pids) == 60 and corpus_pids == sorted(corpus_pids)
    for line in (tmp_path / "a" / "qrels.tsv").read_text().splitlines():
        assert int(line.split("\t")[2]) in corpus_pids
    for line in (tmp_path / "a" / "queries.tsv").read_text().splitlines():
        text = line.split("\t")[1]
        assert text == text.strip()
    assert b"\r" not in (tmp_path / "a" / "corpus.tsv").read_bytes()