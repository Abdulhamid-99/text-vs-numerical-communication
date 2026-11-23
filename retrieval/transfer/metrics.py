"""Frozen transfer ranking, effect and interval calculations shared by both benchmarks."""

import hashlib
import math

import numpy as np
from benchmarks import require

BOOTSTRAP_SEED = 1357913
BOOTSTRAP_REPLICATES = 20000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rank_scores(scores, doc_ids):
    require(
        scores.ndim == 2
        and scores.shape[1] == len(doc_ids)
        and np.isfinite(scores).all()
        and len(set(doc_ids)) == len(doc_ids),
        "Score matrix",
    )
    tie = np.asarray(
        sorted(
            range(len(doc_ids)), key=lambda i: sha(("tie-819-" + doc_ids[i]).encode())
        )
    )
    return tie[np.argsort(-scores[:, tie], axis=1, kind="stable")[:, :10]]


def evaluate(top, qids, doc_ids, qrels):
    require(
        len(qids) == len(top) and set(qids) == set(qrels), "Query evaluation coverage"
    )
    result = []
    for qid, row in zip(qids, top):
        require(len(set(map(int, row))) == len(row), "Repeated retrieved document")
        relevant = set(qrels[qid])
        positions = [
            i + 1 for i, col in enumerate(row) if doc_ids[int(col)] in relevant
        ]
        first = min(positions) if positions else 0
        dcg = sum(1 / math.log2(p + 1) for p in positions)
        ideal = sum(1 / math.log2(p + 1) for p in range(1, min(10, len(relevant)) + 1))
        require(ideal > 0, "No positive qrels")
        result.append([float(first == 1), 1 / first if first else 0.0, dcg / ideal])
    return np.asarray(result)


def paired_interval(
    a, b, qids, components, seed=BOOTSTRAP_SEED, repetitions=BOOTSTRAP_REPLICATES
):
    require(
        len(a) == len(b) == len(qids) and len(set(qids)) == len(qids), "Metric coverage"
    )
    flat = [q for g in components for q in g]
    require(len(flat) == len(set(flat)) and set(flat) == set(qids), "Cluster coverage")
    lookup = {q: i for i, q in enumerate(qids)}
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    require(np.isfinite(delta).all(), "Nonfinite paired delta")
    sums = np.asarray([sum(delta[lookup[q]] for q in g) for g in components])
    sizes = np.asarray([len(g) for g in components])
    rng = np.random.default_rng(seed)
    values = []
    for first in range(0, repetitions, 500):
        picks = rng.integers(
            0, len(components), (min(500, repetitions - first), len(components))
        )
        values.extend((sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)).tolist())
    return {
        "delta_query_mean": float(delta.mean()),
        "ci95_cluster_percentile": np.percentile(values, [2.5, 97.5]).tolist(),
        "clusters": len(components),
        "queries": len(qids),
        "bootstrap_replicates": repetitions,
        "seed": seed,
        "inference": "descriptive transfer interval; assumes independent dependency components; no p-value or multiplicity claim",
    }
