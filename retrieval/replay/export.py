"""Export query-level ArguAna rankings and packet costs from a completed run."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

from check import COLUMNS, rank_hit, read_beir, require

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import PROTOCOL, canonical_hash


FILES = {
    "data_split.json": "150db41365fc7096f418ff5258956073390bbba4f8091315f13e2811746ff3e8",
    "selection.json": "ffc9b28c0aa23e8c795577ac0e18606753ddc4e2b3c172693ab033a6673ef535",
    "test_ids.json": "23baf463e465ba19ae3c43aa3b5fd13a99f32b9aa168a8641949fa16bb8242dc",
    "test_packets/raw_r384_b1/top10.npy": "3094d961d5008d3279b01651c33d955d160d652893390464490a46dce9c2e9e8",
    "test_packets/text_lead_keywords_tokens_64/hybrid_top10.npy": "1d8eb635049e13ce052c78efd023a79439ee4bd5c00c322a1a76ee392242ce91",
}


def channel_files(path, name, count):
    ranks = np.load(path / name, allow_pickle=False)
    require(ranks.shape == (count, 10) and np.issubdtype(ranks.dtype, np.integer),
            "Unexpected ranking array")
    packets = sorted(path.glob("*.bin"))
    require([p.name for p in packets] == [f"{i:05d}.bin" for i in range(count)],
            "Incomplete packet inventory")
    return ranks, [p.stat().st_size for p in packets]


def historical_inputs(run):
    for name, digest in FILES.items():
        require(hashlib.sha256((run / name).read_bytes()).hexdigest() == digest,
                f"Historical input hash mismatch: {name}")
    split = json.loads((run / "data_split.json").read_text(encoding="utf-8"))
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))["64"]
    require(selection["numeric_key"] == "raw_r384_b1"
            and selection["text_key"] == "text_lead_keywords_tokens_64_hybrid",
            "The selected 64-byte channels differ")
    return split


def fresh_inputs(work):
    split = json.loads((work / "data_split.json").read_text(encoding="utf-8"))
    require(canonical_hash(split) == PROTOCOL["split_sha256_canonical"],
            "Split differs from the paper's frozen assignment")
    results = json.loads((work / "primary/test_results.json").read_text(encoding="utf-8"))
    require(results["raw_r384_b1"]["config"] == {"basis": "raw", "rank": 384, "bits": 1},
            "Unexpected numeric configuration")
    text = results["text_lead_keywords_tokens_64_hybrid"]
    require((text["method"], text["wire"], text["decoder"], text["budget"])
            == ("lead_keywords", "tokens", "hybrid", 64), "Unexpected text configuration")
    return split


def export(run, beir_dir, split, target):
    queries, relevant = read_beir(beir_dir)
    rows = split["test"]
    qids = [queries[row] for row in rows]
    require(qids == json.loads((run / "test_ids.json").read_text(encoding="utf-8")),
            "Held-out row order mismatch")
    receiver = np.asarray(split["receiver_corpus"])
    group_ids = sorted(split["test_components"])
    numeric, counts_numeric = channel_files(run / "test_packets/raw_r384_b1", "top10.npy", len(rows))
    text, counts_text = channel_files(run / "test_packets/text_lead_keywords_tokens_64",
                                      "hybrid_top10.npy", len(rows))
    for channel in (numeric, text):
        require(np.all((0 <= channel) & (channel < len(receiver))), "Invalid receiver index")
    table = io.StringIO(newline="")
    writer = csv.writer(table, lineterminator="\n")
    writer.writerow(COLUMNS)
    for i, (row, qid) in enumerate(zip(rows, qids, strict=True)):
        sign_rank = receiver[numeric[i]].tolist()
        text_rank = receiver[text[i]].tolist()
        writer.writerow([row, group_ids.index(split["query_component_ids"][str(row)]),
                         counts_numeric[i], counts_text[i],
                         rank_hit(sign_rank, relevant[qid]), rank_hit(text_rank, relevant[qid]),
                         " ".join(map(str, sign_rank)), " ".join(map(str, text_rank))])
    with target.open("x", encoding="utf-8", newline="") as output:
        output.write(table.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--historical-run", type=Path)
    source.add_argument("--work", type=Path, help="directory for a completed arguana.py primary run")
    parser.add_argument("--beir-dir", type=Path, help="original dataset for historical mode")
    parser.add_argument("--out", required=True, type=Path,
                        help="new CSV destination (will not overwrite an existing file)")
    args = parser.parse_args()
    if args.work:
        if args.beir_dir:
            parser.error("--work uses its own validated data directory")
        export(args.work / "primary", args.work / "data", fresh_inputs(args.work), args.out)
    else:
        if args.beir_dir is None:
            parser.error("--historical-run requires --beir-dir")
        export(args.historical_run, args.beir_dir, historical_inputs(args.historical_run), args.out)


if __name__ == "__main__":
    main()
