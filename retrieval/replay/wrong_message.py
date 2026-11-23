"""Deliver each frozen sender message to a recipient from another ArguAna component."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from check import gain, rank_hit, read_beir, require

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study import num_decode, num_key, retrieval


PERMUTATION_HASH = "a1058b023b54294b3eeb3a1a56c9116f34e400348ca0f6b4a31978e34b5e37c6"
SPLIT_HASHES = {
    "receiver_corpus": "1439713a7a0577307b3ccead69ac86a663548578a835d3bf3851d9c7b63551bd",
    "test": "10e2d12ad2e98547858aa086687539a157b3075615c9770d32d30d611c4cf2da",
}
ALL_CHANNELS = ("raw_r384_b1", "raw_r384_b4", "pca_r32_b4")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def donors(split):
    components = [split["query_component_ids"][str(i)] for i in split["test"]]
    groups = {}
    for row, component in enumerate(components):
        groups.setdefault(component, []).append(row)
    rng = np.random.default_rng(91827)
    keys = list(groups)
    rng.shuffle(keys)
    order = []
    for key in keys:
        members = np.array(groups[key], dtype=int)
        rng.shuffle(members)
        order.extend(members.tolist())
    maximum = max(map(len, groups.values()))
    perm = np.empty(len(order), dtype="<i4")
    perm[order] = np.roll(order, -maximum)
    require(hashlib.sha256(perm.tobytes()).hexdigest() == PERMUTATION_HASH,
            "Donor assignment differs from the frozen 989-row permutation")
    require(np.array_equal(np.sort(perm), np.arange(989)), "Donors are not a permutation")
    require(all(components[i] != components[j] for i, j in enumerate(perm)),
            "A sender shares its recipient's component")
    return perm


def unit_rows(values):
    require(np.isfinite(values).all(), "Nonfinite embedding")
    lengths = np.sqrt(np.sum(values * values, axis=1, keepdims=True))
    require(np.all(lengths > 0), "Zero-norm embedding")
    return values / lengths


def load_packets(run, key, permutation):
    folder = run / "test_packets" / key
    require(sorted(path.name for path in folder.glob("*.bin"))
            == [f"{i:05d}.bin" for i in range(len(permutation))], f"Incomplete packet bank: {key}")
    basis = mean = None
    if key.startswith("pca_"):
        with np.load(run / "basis_final.npz", allow_pickle=False) as archive:
            basis, mean = archive["basis"], archive["mean"]
    vectors = []
    sizes = []
    for j in permutation:
        wire = (folder / f"{j:05d}.bin").read_bytes()
        vector, config = num_decode(wire, basis, mean)
        require(num_key(config) == key, "Donor packet codec differs from the frozen channel")
        vectors.append(vector)
        sizes.append(len(wire))
    return np.stack(vectors), float(np.mean(sizes))


def evaluate(data, split_path, run, channels):
    queries, relevant = read_beir(data)
    split = read_json(split_path)
    for field, digest in SPLIT_HASHES.items():
        canonical = json.dumps(split[field], separators=(",", ":")).encode()
        require(hashlib.sha256(canonical).hexdigest() == digest, f"Changed split: {field}")
    perm = donors(split)
    corpus_index = np.asarray(split["receiver_corpus"])
    qids = [queries[i] for i in split["test"]]
    with (data / "corpus.jsonl").open(encoding="utf-8") as source:
        corpus_ids = [json.loads(row)["_id"] for row in source]
    sort_keys = [hashlib.sha256(("tie-819-" + corpus_ids[i]).encode()).hexdigest()
                 for i in corpus_index]
    order = np.argsort(sort_keys)
    corpus = np.load(run / "corpus_embeddings.npy", allow_pickle=False)
    sender = np.load(run / "test_embeddings.npy", allow_pickle=False)
    require(corpus.shape == (8674, 384) and sender.shape == (989, 384), "Unexpected array shape")
    receiver = unit_rows(corpus[corpus_index])
    correct = read_json(run / "test_results.json")
    cids = [corpus_ids[i] for i in corpus_index]
    correct_targets = {qid: [corpus_ids[relevant[qid]]] for qid in qids}
    metrics = {}
    for key in ("full_wrong_component", *channels):
        if key == "full_wrong_component":
            vectors, sizes = sender[perm], None
        else:
            vectors, sizes = load_packets(run, key, perm)
        scores = unit_rows(vectors) @ receiver.T
        require(np.isfinite(scores).all(), "Nonfinite cosine ranking")
        top = order[np.argsort(-scores[:, order], axis=1, kind="stable")[:, :10]]
        rankings = corpus_index[top].tolist()
        hits = [rank_hit(row, relevant[q]) for row, q in zip(rankings, qids, strict=True)]
        # Score against the donor as well: a broken decoder must not pass just because
        # the recipient control is zero. The correct-matched NDCG is an independent check.
        donor_hits = [rank_hit(row, relevant[qids[j]]) for row, j in zip(rankings, perm, strict=True)]
        donor_score = math.fsum(map(gain, donor_hits)) / len(donor_hits)
        if key in channels:
            with threadpool_limits(limits=6, user_api="blas"):
                matched = retrieval(vectors[np.argsort(perm)], corpus[corpus_index],
                                    qids, cids, correct_targets)
            expected = correct[key]["metrics"][2]
            require(math.isclose(float(matched[:, 2].mean()), expected, abs_tol=1e-10, rel_tol=0),
                    f"{key}: positive control in original matched order disagrees with the run")
        else:
            require(donor_score > 0, "The full-vector positive control has no hits")
        metrics[key] = {
            "hit_at_1": sum(h == 1 for h in hits) / len(hits),
            "mrr_at_10": sum((1 / h if h else 0) for h in hits) / len(hits),
            "ndcg_at_10": math.fsum(map(gain, hits)) / len(hits),
            "donor_aligned_ndcg_at_10": donor_score,
        }
        if sizes is not None:
            metrics[key]["mean_packet_bytes"] = sizes
    return {"queries": len(perm), "permutation_sha256": PERMUTATION_HASH, "conditions": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--work", type=Path, help="directory created by arguana.py prepare")
    sources.add_argument("--historical-run", type=Path, help="archived run_merged_cached_len_v2")
    parser.add_argument("--beir-dir", type=Path, help="source ArguAna directory for the historical run")
    parser.add_argument("--stage", choices=["primary", "heldout"], default="primary")
    args = parser.parse_args()
    if args.work:
        if args.beir_dir:
            parser.error("--beir-dir is only used with --historical-run")
        data, split, run = args.work / "data", args.work / "data_split.json", args.work / args.stage
    else:
        if args.beir_dir is None:
            parser.error("--historical-run requires --beir-dir")
        data, run = args.beir_dir, args.historical_run
        split = run / "data_split.json"
    channels = ALL_CHANNELS if args.stage == "heldout" else ALL_CHANNELS[:1]
    report = evaluate(data, split, run, channels)
    print(json.dumps(report, indent=2))
    for key, result in report["conditions"].items():
        require(result["ndcg_at_10"] == 0, f"{key}: differs from the published wrong-message control")


if __name__ == "__main__":
    main()
