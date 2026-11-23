"""Recompute the paper's 64-byte ArguAna comparison from query-level outcomes."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


OBSERVATIONS = Path(__file__).with_name("observations.csv")
TABLE_HASH = "c99c003ea5a6905404479cd2f4ccf4973ecee70b2204c2ac775a9c66c15af428"
COLUMNS = ("query_row", "component", "sign_bytes", "text_bytes", "sign_hit",
           "text_hit", "sign_top10", "text_top10")
SOURCE_HASHES = {
    "corpus.jsonl": "0627982cec4ee896df6dc98604ef6b499c83fd87a19eaaeb3f23b88bbef6cbce",
    "queries.jsonl": "f8b7c903f95d9a2d291de3915a8fac25085e9db679f19f59fea4145bfd1df618",
    "test.tsv": "0c47b481fa8b47fb9ca5e74bc8a017bdea14518c0e4a5d21345291180b3aca76",
}
EXPECTED = {
    "sign_ndcg10": 0.6405076505687227, "text_ndcg10": 0.617059841923275,
    "sign_bytes": 56.0, "text_bytes": 63.365015166835185,
    "paired_difference": 0.023447808645447446, "two_sided_p": 0.03023969760302397,
    "ci_low": 0.0024446393211464204, "ci_high": 0.04426329984407469,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_beir(directory):
    """Return query IDs and each positive document's zero-based corpus row."""
    paths = {name: directory / name for name in SOURCE_HASHES}
    if not paths["test.tsv"].is_file():
        paths["test.tsv"] = directory / "qrels" / "test.tsv"
    for name, path in paths.items():
        require(hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_HASHES[name],
                f"BEIR source differs from the paper: {name}")
    def ids(path):
        with path.open(encoding="utf-8") as source:
            return [json.loads(line)["_id"] for line in source]
    queries = ids(paths["queries.jsonl"])
    corpus = {key: i for i, key in enumerate(ids(paths["corpus.jsonl"]))}
    relevant = {}
    with paths["test.tsv"].open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            if int(row["score"]) > 0:
                query = row["query-id"]
                require(query not in relevant, "Expected one positive document per query")
                # Five source queries name a missing target and are excluded by the frozen split.
                relevant[query] = corpus.get(row["corpus-id"])
    return queries, relevant


def rank_hit(ranking, positive):
    require(positive is not None, "Held-out query has no positive target in the source corpus")
    return ranking.index(positive) + 1 if positive in ranking else 0


def gain(rank):
    return 1 / math.log2(rank + 1) if rank else 0.0


def observations(path, beir):
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        require(tuple(reader.fieldnames or ()) == COLUMNS, "Unexpected observation fields")
        rows = []
        for row in reader:
            parsed = {k: int(row[k]) for k in COLUMNS[:6]}
            require(0 <= parsed["query_row"] < 1406, "Query index outside the ArguAna release")
            for channel in ("sign", "text"):
                ranking = [int(x) for x in row[channel + "_top10"].split()]
                require(len(ranking) == len(set(ranking)) == 10, "Invalid top-ten ranking")
                require(all(0 <= x < 8674 for x in ranking), "Document index outside the release")
                hit = parsed[channel + "_hit"]
                require(0 <= hit <= 10, "Invalid positive hit rank")
                if beir is not None:
                    query = beir[0][parsed["query_row"]]
                    require(hit == rank_hit(ranking, beir[1][query]), "Rank and BEIR qrels disagree")
            require(parsed["sign_bytes"] == 56 and 0 < parsed["text_bytes"] <= 64,
                    "Packet outside the primary study's byte limits")
            rows.append(parsed)
    require(len(rows) == len({row["query_row"] for row in rows}) == 989,
            "Expected one row per held-out query")
    require({row["component"] for row in rows} == set(range(122)),
            "Expected exactly 122 components")
    return rows


def analyze(rows):
    sign = np.array([gain(row["sign_hit"]) for row in rows])
    text = np.array([gain(row["text_hit"]) for row in rows])
    differences = sign - text
    membership = np.array([row["component"] for row in rows])
    sums = np.array([math.fsum(differences[membership == i]) for i in range(122)])
    counts = np.bincount(membership)
    total = math.fsum(sums)
    flips = np.random.default_rng(91371).integers(0, 2, size=(100000, 122), dtype=np.int8) * 2 - 1
    tail = np.count_nonzero(np.abs(flips @ sums) >= abs(total) - 1e-12)
    samples = np.random.default_rng(91372).integers(0, 122, size=(20000, 122), dtype=np.int64)
    changes = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    low, high = np.quantile(changes, [.025, .975], method="linear")
    sign_bytes = np.mean([row["sign_bytes"] for row in rows])
    text_bytes = np.mean([row["text_bytes"] for row in rows])
    return {
        "queries": len(rows), "components": len(counts),
        "sign_ndcg10": float(sign.mean()), "text_ndcg10": float(text.mean()),
        "sign_bytes": float(sign_bytes), "text_bytes": float(text_bytes),
        "message_byte_saving_percent": float(100 * (1 - sign_bytes / text_bytes)),
        "paired_difference": total / len(rows),
        "ci_low": float(low), "ci_high": float(high),
        "two_sided_p": float((tail + 1) / 100001),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beir-dir", type=Path,
                        help="directory containing the original extracted BEIR ArguAna files")
    parser.add_argument("--observations", type=Path, default=OBSERVATIONS)
    args = parser.parse_args()
    if args.observations.resolve() == OBSERVATIONS.resolve():
        require(hashlib.sha256(OBSERVATIONS.read_bytes()).hexdigest() == TABLE_HASH,
                "Distributed observation table differs from the archived export")
    beir = read_beir(args.beir_dir) if args.beir_dir is not None else None
    results = analyze(observations(args.observations, beir))
    print(json.dumps(results, indent=2))
    for key, expected in EXPECTED.items():
        require(math.isclose(results[key], expected, rel_tol=0, abs_tol=1e-12),
                f"{key} differs from the archived result: {results[key]} versus {expected}")


if __name__ == "__main__":
    main()
