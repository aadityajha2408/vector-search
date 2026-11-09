"""Build a small, reproducible evaluation subset of MS MARCO passage ranking.

Why a subset? Embedding 8.8M passages on a laptop is not realistic. But if we just
took 50K random passages, most queries would lose their relevant passage and every
metric would be ~0. So we:
  1. sample queries,
  2. force-keep every passage judged relevant for them,
  3. add random "distractor" passages up to the target corpus size.

IMPORTANT INVARIANT: the output corpus is sorted by pid and never reordered.
Line i of corpus.tsv == row i of the embedding matrix == id returned by the C++ index.
Numbers from this subset are NOT comparable to the public MS MARCO leaderboard
(the corpus is ~170x smaller, so the task is easier).

Run from the repo root:
    python -m src.preprocessing.build_subset --num-queries 1000 --corpus-size 50000 --seed 42
"""
import argparse
import random
from pathlib import Path

from tqdm import tqdm

from src.config import CORPUS_FILE, PROCESSED_DIR, QRELS_FILE, QUERIES_FILE, RAW_DIR

MSMARCO_PASSAGE_COUNT = 8_841_823


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-queries", type=int, default=1000)
    p.add_argument("--corpus-size", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--collection-size", type=int, default=MSMARCO_PASSAGE_COUNT,
                   help="number of lines in collection.tsv (pids must be 0..N-1)")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    return p.parse_args()


def load_qrels(path: Path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, pid, rel = line.split()
            if int(rel) > 0:
                qrels.setdefault(int(qid), set()).add(int(pid))
    return qrels


def load_queries(path: Path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, text = line.rstrip("\n").partition("\t")
            queries[int(qid)] = text.strip()
    return queries


def clean(text: str) -> str:
    """Passages must stay on one line with no tabs, or the TSV format breaks."""
    return " ".join(text.split())


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    qrels = load_qrels(args.raw_dir / "qrels.dev.small.tsv")
    queries = load_queries(args.raw_dir / "queries.dev.small.tsv")

    # 1. Sample queries. Sorting first makes the sample independent of dict/file order.
    rng = random.Random(args.seed)
    eligible = sorted(q for q in qrels if q in queries)
    if args.num_queries > len(eligible):
        raise SystemExit(f"Only {len(eligible)} eligible queries, asked for {args.num_queries}")
    qids = sorted(rng.sample(eligible, args.num_queries))

    # 2. Passages that must be in the corpus.
    must_keep = set().union(*(qrels[q] for q in qids))
    n_distractors = args.corpus_size - len(must_keep)
    if n_distractors < 0:
        raise SystemExit(f"corpus-size {args.corpus_size} is smaller than the {len(must_keep)} relevant passages")

    # 3. Distractors: sample extra ids, drop any that collide with must_keep, take what we need.
    #    random.sample on a range() is memory-cheap (no 8.8M-element list).
    extra = rng.sample(range(args.collection_size), n_distractors + len(must_keep))
    distractors = [pid for pid in extra if pid not in must_keep][:n_distractors]
    keep = must_keep | set(distractors)

    # 4. One streaming pass over the big file.
    n_lines = 0
    n_written = 0
    out_corpus = args.out_dir / CORPUS_FILE
    with open(args.raw_dir / "collection.tsv", encoding="utf-8") as f_in, \
         open(out_corpus, "w", encoding="utf-8", newline="\n") as f_out:
        for line_no, line in enumerate(tqdm(f_in, total=args.collection_size, desc="collection")):
            pid_str, _, text = line.rstrip("\n").partition("\t")
            pid = int(pid_str)
            if pid != line_no:
                raise SystemExit(f"Assumption broken: line {line_no} has pid {pid} (pids must be consecutive)")
            n_lines += 1
            if pid in keep:
                f_out.write(f"{pid}\t{clean(text)}\n")
                n_written += 1
    if n_lines != args.collection_size:
        raise SystemExit(f"collection.tsv has {n_lines} lines, expected {args.collection_size}. "
                         f"Re-run with --collection-size {n_lines}")

    # 5. Queries and qrels for the sampled ids.
    with open(args.out_dir / QUERIES_FILE, "w", encoding="utf-8", newline="\n") as f:
        for qid in qids:
            f.write(f"{qid}\t{queries[qid]}\n")
    with open(args.out_dir / QRELS_FILE, "w", encoding="utf-8", newline="\n") as f:
        for qid in qids:
            for pid in sorted(qrels[qid]):
                f.write(f"{qid}\t0\t{pid}\t1\n")

    print(f"queries={len(qids)} relevant_passages={len(must_keep)} "
          f"distractors={len(distractors)} total_passages={n_written} seed={args.seed}")


if __name__ == "__main__":
    main()
