"""Readers for the processed subset files. All files are UTF-8, tab-separated."""
from pathlib import Path

import numpy as np


def read_corpus(path: Path):
    """Return (pids, texts). Row i of every embedding matrix corresponds to line i here."""
    pids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            pid, _, text = line.rstrip("\n").partition("\t")
            pids.append(int(pid))
            texts.append(text)
    return np.asarray(pids, dtype=np.int64), texts


def read_queries(path: Path):
    """Return (qids, texts) with surrounding whitespace stripped."""
    qids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, text = line.rstrip("\n").partition("\t")
            qids.append(int(qid))
            texts.append(text.strip())
    return qids, texts


def read_qrels(path: Path):
    """Return {qid: set(relevant pids)} from TREC-style 'qid 0 pid rel' lines."""
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, _, pid, rel = parts
            if int(rel) > 0:
                qrels.setdefault(int(qid), set()).add(int(pid))
    return qrels
