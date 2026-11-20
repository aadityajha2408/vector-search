"""Compare HNSW variants: recall-vs-QPS plot plus QPS at fixed recall levels.

    python -m benchmarks.plot_ablation
Reads results/ann_main.csv, ann_noheuristic.csv, ann_M8.csv, ann_M32.csv (copies of ann_bench.csv
taken after each run). QPS at a target recall is linearly interpolated (in log QPS) between the two
measured ef settings that bracket it, which is a fair way to compare curves that do not share ef values.
"""
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR

RUNS = [  # (csv file, index name inside that file, legend label)
    ("ann_main.csv", "ours", "ours (M=16, heuristic)"),
    ("ann_main.csv", "hnswlib", "hnswlib (M=16)"),
    ("ann_noheuristic.csv", "ours", "ours, no heuristic (M=16)"),
    ("ann_M8.csv", "ours", "ours (M=8)"),
    ("ann_M32.csv", "ours", "ours (M=32)"),
]
TARGETS = [0.95, 0.98, 0.99]


def load(filename, index):
    with open(RESULTS_DIR / filename, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["index"] == index]
    build = float(rows[0]["build_s"])
    return sorted((float(r["recall"]), float(r["qps"])) for r in rows), build


def qps_at_recall(points, target):
    for (r1, q1), (r2, q2) in zip(points, points[1:]):
        if r1 <= target <= r2 and r2 > r1:
            frac = (target - r1) / (r2 - r1)
            return math.exp(math.log(q1) + frac * (math.log(q2) - math.log(q1)))
    return None


def main():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    print(f"{'variant':30s}{'build s':>9s}" + "".join(f"{'QPS@' + str(t):>12s}" for t in TARGETS))
    for filename, index, label in RUNS:
        points, build = load(filename, index)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=label)
        cells = []
        for t in TARGETS:
            q = qps_at_recall(points, t)
            cells.append(f"{q:12.0f}" if q else f"{'n/a':>12s}")
        print(f"{label:30s}{build:9.1f}" + "".join(cells))
    ax.set_xlabel("recall@10 vs exact search")
    ax.set_ylabel("queries / second (1 thread)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS_DIR / "ablation.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()