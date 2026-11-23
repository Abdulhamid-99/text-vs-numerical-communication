"""Re-run the two frozen 64-byte transfer channels at the original query and corpus sizes."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from benchmarks import acquire, read_data, require
from metrics import evaluate, paired_interval, rank_scores
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BUDGET = 64
SIGN = {"basis": "raw", "rank": 384, "bits": 1}
# These are the two channels chosen on ArguAna development, before the transfer.
TEXT_METHOD = "lead_keywords"
TEXT_WIRE = "tokens"


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temp, path)


def sender(study, encoder, queries, idf, packet_dir):
    vectors = study.encode(encoder, queries, "A_full_query_only")
    for i, (query, vector) in enumerate(zip(queries, vectors, strict=True)):
        numeric = study.num_packet(vector, SIGN, None, None)
        text = study.make_text(
            query, TEXT_METHOD, TEXT_WIRE, BUDGET, encoder.tokenizer, idf
        )
        require(len(numeric) == 56 and 8 <= len(text) <= BUDGET, "Invalid packet size")
        (packet_dir / f"{i:04d}.numeric").write_bytes(numeric)
        (packet_dir / f"{i:04d}.text").write_bytes(text)


def rank_channels(
    study, aliases, corpus, corpus_vectors, numeric_vectors, decoded_text, text_vectors
):
    """Receiver inputs are decoded packets, its own corpus and its corpus aliases."""
    corpus_norm = study.norm(corpus_vectors)
    numeric_top = rank_scores(study.norm(numeric_vectors) @ corpus_norm.T, aliases)
    dense = study.norm(text_vectors) @ corpus_norm.T
    vectorizer, index = study.bm25_index(corpus)
    sparse = (
        vectorizer.transform(decoded_text).astype(bool).astype(float) @ index.T
    ).toarray()
    text_top = rank_scores(study.rrf(dense, sparse, aliases), aliases)
    return {"numeric": numeric_top, "text": text_top}


def receiver(study, encoder, corpus, aliases, packet_dir, count):
    numerical = [(packet_dir / f"{i:04d}.numeric").read_bytes() for i in range(count)]
    text = [(packet_dir / f"{i:04d}.text").read_bytes() for i in range(count)]
    decoded_vectors = np.stack(
        [study.receive_numeric(packet, BUDGET, None, None)[0] for packet in numerical]
    )
    decoded_text = [
        study.receive_text(packet, BUDGET, encoder.tokenizer) for packet in text
    ]
    corpus_vectors = study.encode(encoder, corpus, "B_private_corpus")
    text_vectors = study.encode(encoder, decoded_text, "B_selected_text_hybrid")
    top = rank_channels(
        study,
        aliases,
        corpus,
        corpus_vectors,
        decoded_vectors,
        decoded_text,
        text_vectors,
    )
    return top, {"numeric": list(map(len, numerical)), "text": list(map(len, text))}


def summarize(data, top, costs, output):
    metrics = {}
    summaries = {}
    for name in ("numeric", "text"):
        require(top[name].shape == (len(data.query_ids), 10), "Incomplete rankings")
        require(len(costs[name]) == len(data.query_ids), "Incomplete packet costs")
        values = evaluate(top[name], data.query_ids, data.aliases, data.qrels)
        np.save(output / f"{name}_top10.npy", top[name])
        np.save(output / f"{name}_metrics.npy", values)
        metrics[name] = values[:, 2]
        summaries[name] = {
            "ndcg10": float(metrics[name].mean()),
            "mean_bytes": float(np.mean(costs[name])),
        }
    return {
        "queries": len(data.queries),
        "documents": len(data.corpus),
        "components": len(data.groups),
        "summaries": summaries,
        "numeric_minus_text": paired_interval(
            metrics["numeric"], metrics["text"], data.query_ids, data.groups
        ),
    }


def run(args):
    import study
    from model import verify_idf

    idf = json.loads(args.idf.read_text(encoding="utf-8"))
    verify_idf(idf)
    data = read_data(args.dataset, args.data)
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out / "status.json"
    write_json(status, {"status": "running"})
    try:
        study.OUT = args.out
        encoder = study.model_load()
        require(encoder.max_seq_length == 512, "Wrong encoder context length")
        packet_dir = args.out / "packets"
        packet_dir.mkdir()
        # These NumPy BLAS counts were recorded in the two final transfer audits.
        with threadpool_limits(
            limits=6 if args.dataset == "bright" else 20, user_api="blas"
        ):
            sender(study, encoder, data.queries, idf, packet_dir)
            top, costs = receiver(
                study, encoder, data.corpus, data.aliases, packet_dir, len(data.queries)
            )
        result = summarize(data, top, costs, args.out)
        write_json(
            args.out / "identifiers.json",
            {
                "query_ids": data.query_ids,
                "corpus_ids": data.doc_ids,
                "corpus_aliases": data.aliases,
            },
        )
        write_json(args.out / "costs.json", costs)
        write_json(args.out / "results.json", result)
        write_json(status, {"status": "completed", "queries": len(data.query_ids)})
        print(json.dumps(result, indent=2))
    except BaseException as error:
        write_json(status, {"status": "failed", "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run"):
        cmd = subs.add_parser(action)
        cmd.add_argument("dataset", choices=["bright", "scifact"])
        cmd.add_argument(
            "--data",
            type=Path,
            required=True,
            help="Directory for official benchmark inputs",
        )
        if action == "run":
            cmd.add_argument(
                "--idf", type=Path, required=True, help="Frozen ArguAna idf_final.json"
            )
            cmd.add_argument(
                "--out", type=Path, required=True, help="New output directory"
            )
    args = parser.parse_args()
    if args.action == "prepare":
        acquire(args.dataset, args.data, download=True)
        data = read_data(args.dataset, args.data)
        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "queries": len(data.queries),
                    "documents": len(data.corpus),
                    "components": len(data.groups),
                }
            )
        )
    else:
        run(args)


if __name__ == "__main__":
    main()
