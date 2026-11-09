"""Plot recall vs QPS from results/ann_bench.csv  ->  results/recall_qps.png"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR


def main():
    with open(RESULTS_DIR / "ann_bench.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in sorted({r["index"] for r in rows}):
        pts = sorted((float(r["recall"]), float(r["qps"])) for r in rows if r["index"] == name)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=name)
    ax.set_xlabel("recall@k")
    ax.set_ylabel("queries / second (1 thread)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = RESULTS_DIR / "recall_qps.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
