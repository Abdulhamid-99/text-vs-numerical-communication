"""Replay the paper's selected 64-byte comparison from receiver-side numerical evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
HEADER = struct.Struct("<2sBBHH")
# Held-out queries, receiver documents, dependency components, historical BLAS threads.
DESIGNS = {"arguana": (989, 7376, 122, 6), "bright": (103, 57359, 100, 6),
           "scifact": (300, 5183, 247, 20)}
EXPECTED = {
    "arguana": (0.6405076505687227, 0.617059841923275, 63.365015166835185),
    "bright": (0.11298014407262376, 0.10174286268711669, 62.116504854368934),
    "scifact": (0.6699582801477678, 0.7292000972619217, 46.46666666666667),
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalise(matrix):
    require(matrix.dtype == np.float32 and np.isfinite(matrix).all(), "Invalid embeddings")
    length = np.sqrt(np.sum(matrix * matrix, axis=1, keepdims=True))
    require((length > 0).all(), "Zero-length embedding")
    return matrix / length


def rank(scores, tie_order):
    order = np.argsort(-scores[:, tie_order], axis=1, kind="stable")
    return tie_order[order]


def ranks_by_document(order):
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, order.shape[1]+1)[None, :], axis=1)
    return ranks


def rows_metrics(top, positive):
    discounts = 1 / np.log2(np.arange(2, 12))
    metrics = []
    for docs, targets in zip(top, positive, strict=True):
        targets = targets[targets >= 0]
        require(len(targets) > 0, "Missing gold judgment")
        hits = np.isin(docs, targets)
        metrics.append(np.sum(discounts[hits]) / np.sum(discounts[:min(10, len(targets))]))
    return np.array(metrics)


def inference(differences, groups, primary):
    count = int(groups.max()) + 1
    sizes = np.bincount(groups, minlength=count)
    require((sizes > 0).all(), "Empty component")
    sums = np.array([math.fsum(differences[groups == i]) for i in range(count)])
    bootstrap = np.random.default_rng(91372 if primary else 1357913)
    draws = bootstrap.integers(count, size=(20000, count))
    effects = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    result = {"components": count, "paired_difference": float(differences.mean()),
              "ci95": np.quantile(effects, [0.025, 0.975]).tolist()}
    if primary:
        random = np.random.default_rng(91371)
        signs = random.integers(0, 2, size=(100000, count), dtype=np.int8) * 2 - 1
        tail = np.count_nonzero(abs(signs @ sums) >= abs(math.fsum(sums)) - 1e-12)
        result["two_sided_p"] = float((tail + 1) / 100001)
    return result


def replay(data, dataset):
    q, n, components, threads = DESIGNS[dataset]
    numeric = data["numeric_packets"]
    require(numeric.dtype == np.uint8 and numeric.shape == (q, 56), "Missing numerical packets")
    header = np.frombuffer(HEADER.pack(b"SR", 1, 11, 384, 0), dtype=np.uint8)
    require((numeric[:, :8] == header).all(), "Invalid sign-packet header")
    sign_bits = np.unpackbits(numeric[:, 8:], axis=1, bitorder="little")
    sign_vectors = sign_bits.astype(np.float32) * 2 - 1
    headers = data["text_headers"]
    require(headers.shape == (q, 8) and headers.dtype == np.uint8, "Missing text frame headers")
    lengths = []
    for row in headers:
        magic, version, kind, length, reserved = HEADER.unpack(row.tobytes())
        require((magic, version, reserved) == (b"SR", 1, 0) and kind in (1, 2, 3), "Invalid text frame")
        require(kind != 3 or length % 2 == 0, "Invalid token payload length")
        lengths.append(8 + length)
    require(max(lengths) <= 64, "Text frame above ceiling")
    corpus = data["document_vectors"]
    text = data["text_vectors"]
    require(corpus.shape == (n, 384) and text.shape == (q, 384), "Missing receiver vectors")
    tie = data["tie_order"].astype(np.int64)
    require(np.array_equal(np.sort(tie), np.arange(n)), "Invalid tie order")
    groups = data["groups"].astype(int)
    require(groups.shape == (q,) and np.array_equal(np.unique(groups), np.arange(components)), "Invalid groups")
    gold = data["gold"]
    require(gold.ndim == 2 and gold.shape[0] == q and np.all((gold >= -1) & (gold < n)), "Gold index error")
    lexical_ranks = data["lexical_ranks"]
    require(lexical_ranks.shape == (q, n) and np.issubdtype(lexical_ranks.dtype, np.integer), "Incomplete lexical ranks")
    expected_ranks = np.arange(1, n + 1)
    for row in lexical_ranks:
        require(np.array_equal(np.sort(row), expected_ranks), "Lexical ranks do not form a permutation")
    # Recreate the full matrix product used by the study before applying any exclusions.
    with threadpool_limits(limits=threads, user_api="blas"):
        documents = normalise(corpus)
        numerical_scores = normalise(sign_vectors) @ documents.T
        dense_text = normalise(text) @ documents.T
    text_ranks = ranks_by_document(rank(dense_text, tie))
    score = 1.0 / (60.0 + text_ranks) + 1.0 / (60.0 + lexical_ranks)
    excluded = data["excluded"]
    require(excluded.ndim == 2 and excluded.shape[0] == q and np.all((excluded >= -1) & (excluded < n)), "Invalid exclusion table")
    for i, row in enumerate(excluded):
        excluded_ids = row[row >= 0]
        numerical_scores[i, excluded_ids] = -np.inf
        score[i, excluded_ids] = -np.inf
    num_top = rank(numerical_scores, tie)[:, :10]
    text_top = rank(score, tie)[:, :10]
    for key in ("numeric_top10", "text_top10"):
        require(data[key].shape == (q, 10), "Incorrect reference ranking shape")
    changed = {"numeric": int(np.any(num_top != data["numeric_top10"], axis=1).sum()),
               "text": int(np.any(text_top != data["text_top10"], axis=1).sum())}
    require(not any(changed.values()), f"Stored ranking mismatch: {changed}. Check the BLAS runtime.")
    a, b = rows_metrics(num_top, gold), rows_metrics(text_top, gold)
    expected_a, expected_b, expected_length = EXPECTED[dataset]
    np.testing.assert_allclose([a.mean(), b.mean(), np.mean(lengths)],
                               [expected_a, expected_b, expected_length], atol=1e-12, rtol=0)
    with threadpool_limits(limits=2, user_api="blas"):
        uncertainty = inference(a-b, groups, dataset == "arguana")
    if dataset == "arguana":
        require(round(uncertainty["two_sided_p"], 4) == 0.0302, "Primary sign-flip mismatch")
        np.testing.assert_array_equal(np.round(uncertainty["ci95"], 4), [0.0024, 0.0443])
    return {"dataset": dataset, "queries": q, "documents": n, "numeric_ndcg10": float(a.mean()),
            "text_ndcg10": float(b.mean()), "numeric_mean_bytes": 56.0, "text_mean_bytes": float(np.mean(lengths)),
            "percent_fewer_numeric_bytes": float(100*(1-56/np.mean(lengths))),
            "rows_with_changed_top10": changed, **uncertainty}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "arguana_receiver.npz")
    parser.add_argument("--dataset", choices=DESIGNS, default="arguana")
    args = parser.parse_args()
    if args.data.resolve() == HERE / "arguana_receiver.npz":
        expected = (HERE / "arguana_receiver.sha256").read_text().split()[0]
        require(digest(args.data.read_bytes()) == expected, "Packaged evidence checksum mismatch")
    with np.load(args.data, allow_pickle=False) as evidence:
        print(json.dumps(replay(evidence, args.dataset), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
