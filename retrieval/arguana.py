"""Re-run the primary ArguAna experiment, or evaluate the full candidate grid."""

from argparse import ArgumentParser
from contextlib import contextmanager
import collections
import json
from pathlib import Path
import shutil
import tempfile
from transfer.download_source import download_file
from zipfile import ZipFile

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

import study
from model import PROTOCOL, canonical_hash, check_file, verify_idf
from split import build as build_split


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def new_directory(destination):
    """A failed stage never looks like a completed run."""
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=".arguana-"
    ) as staging:
        output = Path(staging) / "output"
        output.mkdir()
        yield output
        output.rename(destination)


def load_records(data):
    def rows(name):
        return [
            json.loads(line)
            for line in (data / name).read_text(encoding="utf-8").splitlines()
        ]

    relevance = collections.defaultdict(list)
    for line in (data / "test.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        query, document, grade = line.split("\t")
        if int(grade) > 0:
            relevance[query].append(document)
    return rows("corpus.jsonl"), rows("queries.jsonl"), relevance


def prepare_inputs(data, archive_path=None, mirror_dir=None):
    spec = PROTOCOL["dataset"]
    if archive_path:
        check_file(archive_path, spec["files"]["arguana.zip"])
        members = {
            "corpus.jsonl": "arguana/corpus.jsonl",
            "queries.jsonl": "arguana/queries.jsonl",
            "test.tsv": "arguana/qrels/test.tsv",
        }
        with ZipFile(archive_path) as archive:
            for name, member in members.items():
                (data / name).write_bytes(archive.read(member))
    else:
        from pyarrow import parquet

        for name, (url, expected) in spec["mirror"].items():
            path = data / name
            if mirror_dir:
                shutil.copyfile(mirror_dir / name, path)
            else:
                download_file(url, path)
            check_file(path, expected)
        for stem, keys in (
            ("corpus", ["_id", "title", "text"]),
            ("queries", ["_id", "text"]),
        ):
            rows = parquet.read_table(data / (stem + ".parquet")).to_pylist()
            with (data / (stem + ".jsonl")).open(
                "w", encoding="utf-8", newline="\n"
            ) as out:
                for row in rows:
                    restored = {key: row[key] for key in keys}
                    restored["metadata"] = {}
                    out.write(json.dumps(restored) + "\n")
    for name in ("corpus.jsonl", "queries.jsonl", "test.tsv"):
        check_file(data / name, spec["files"][name])


def prepare(work, archive_path=None, mirror_dir=None):
    with new_directory(work) as output:
        data = output / "data"
        data.mkdir()
        prepare_inputs(data, archive_path, mirror_dir)
        corpus, queries, relevance = load_records(data)
        split = build_split(corpus, queries, relevance)
        if canonical_hash(split) != PROTOCOL["split_sha256_canonical"]:
            raise ValueError("Recovered data split differs from the frozen assignment.")
        write_json(output / "data_split.json", split)
        counts = collections.Counter()
        for idx in split["fit_corpus"]:
            counts.update(set(study.words(study.text(corpus[idx]))))
        idf = {"N": len(split["fit_corpus"]), "df": counts}
        verify_idf(idf)
        write_json(output / "idf_final.json", idf)
        print(json.dumps(split["counts"]), flush=True)


def features(work):
    for name in ("corpus.jsonl", "queries.jsonl", "test.tsv"):
        check_file(work / "data" / name, PROTOCOL["dataset"]["files"][name])
    split = read_json(work / "data_split.json")
    if canonical_hash(split) != PROTOCOL["split_sha256_canonical"]:
        raise ValueError("Split changed after preparation.")
    idf = read_json(work / "idf_final.json")
    verify_idf(idf)
    return split, idf, load_records(work / "data")


def choose(rows, budgets):
    selected = {}
    for budget in budgets:
        numeric = [
            (k, v)
            for k, v in rows.items()
            if v["family"] == "numeric" and v["bytes_max"] <= budget
        ]
        text = [
            (k, v)
            for k, v in rows.items()
            if v["family"] == "text" and v["budget"] == budget
        ]

        def best(cells):
            return min(
                cells,
                key=lambda item: (
                    -item[1]["metrics"][2],
                    item[1]["bytes_mean"],
                    item[0],
                ),
            )

        nk, nv = best(numeric)
        tk, tv = best(text)
        selected[str(budget)] = dict(numeric_key=nk, numeric=nv, text_key=tk, text=tv)
    return selected


def save_packets(directory, packets):
    directory.mkdir()
    for i, packet in enumerate(packets):
        (directory / f"{i:05d}.bin").write_bytes(packet)


def validate_selection(selection, budgets):
    for budget in budgets:
        entry = selection[str(budget)]
        config, text = entry["numeric"]["config"], entry["text"]
        if (
            config not in study.NUM_CONFIGS
            or study.num_size(config) > budget
            or entry["numeric_key"] != study.num_key(config)
        ):
            raise ValueError(f"Invalid numerical selection at {budget} bytes")
        method, wire, decoder = text["method"], text["wire"], text["decoder"]
        expected = f"text_{method}_{wire}_{budget}_{decoder}"
        if (
            method not in study.TEXT_METHODS
            or wire not in study.TEXT_WIRES
            or decoder not in ("dense", "bm25", "hybrid")
            or text["budget"] != budget
            or entry["text_key"] != expected
        ):
            raise ValueError(f"Invalid text selection at {budget} bytes")


def run(work, stage, selected_file=None):
    split, idf, (corpus, queries, relevance) = features(work)
    partition = "dev" if stage == "development" else "test"
    idx = split[partition]
    qids = [queries[i]["_id"] for i in idx]
    query_text = [queries[i]["text"] for i in idx]
    corpus_text = [study.text(row) for row in corpus]
    receiver = split["receiver_corpus"]
    cids = [corpus[i]["_id"] for i in receiver]
    budgets = [64] if stage == "primary" else study.BUDGETS
    if stage == "development":
        configs = study.NUM_CONFIGS
        text_settings = [
            (m, w, b)
            for b in budgets
            for m in study.TEXT_METHODS
            for w in study.TEXT_WIRES
        ]
    else:
        selection = read_json(
            selected_file or Path(__file__).with_name("selection.json")
        )
        if (
            selected_file is None
            and canonical_hash(selection) != PROTOCOL["selection_sha256_canonical"]
        ):
            raise ValueError("Archived development selection changed")
        validate_selection(selection, budgets)
        configs = [selection[str(b)]["numeric"]["config"] for b in budgets]
        text_settings = [
            (selection[str(b)]["text"]["method"], selection[str(b)]["text"]["wire"], b)
            for b in budgets
        ]
    configs = {study.num_key(cfg): cfg for cfg in configs}
    with new_directory(work / stage) as output:
        study.OUT = output
        model = study.model_load()
        idf_size = len(
            json.dumps(idf, ensure_ascii=False, indent=2)
            .replace("\n", "\r\n")
            .encode("utf-8")
        )
        print("Encoding corpus and queries", flush=True)
        source_corpus = study.encode(model, corpus_text, "corpus")
        np.save(output / "corpus_embeddings.npy", source_corpus)
        ce = source_corpus[receiver]
        qe = study.encode(model, query_text, f"{partition}_sender")
        np.save(output / f"{partition}_embeddings.npy", qe)
        basis = mean = None
        if any(cfg["basis"] == "pca" for cfg in configs.values()):
            training = source_corpus[split["fit_corpus"]]
            mean = training.mean(axis=0).astype("<f4")
            centered = training.astype("float64") - mean
            _, vectors = eigh(centered.T @ centered, driver="evr")
            basis = vectors[:, ::-1].T.astype("<f4")
            np.savez(output / "basis_final.npz", basis=basis, mean=mean)
        bm25_vectorizer, bm25 = study.bm25_index([corpus_text[i] for i in receiver])
        banks = output / f"{partition}_packets"
        banks.mkdir()
        rows, metrics = {}, {}
        for key, config in configs.items():
            with threadpool_limits(limits=6, user_api="blas"):
                raw = [study.num_packet(x, config, basis, mean) for x in qe]
                if any(len(p) != study.num_size(config) for p in raw):
                    raise ValueError(
                        "Encoded numerical message violates its registered size"
                    )
                vectors = np.stack(
                    [
                        study.receive_numeric(p, study.num_size(config), basis, mean)[0]
                        for p in raw
                    ]
                )
                per_query, top = study.retrieval(
                    vectors, ce, qids, cids, relevance, True
                )
            directory = banks / key
            save_packets(directory, raw)
            np.save(directory / "top10.npy", top)
            rows[key] = dict(
                family="numeric",
                config=config,
                bytes_mean=float(np.mean(list(map(len, raw)))),
                bytes_max=max(map(len, raw)),
                setup_bytes=study.num_setup(config),
                metrics=per_query.mean(axis=0).tolist(),
            )
            metrics[key] = per_query
        for method, wire, budget in text_settings:
            key = f"text_{method}_{wire}_{budget}"
            print(f"Evaluating {key}", flush=True)
            if method == "semantic_window":
                raw = [
                    study.semantic_text(q, wire, budget, model, idf, v)
                    for q, v in zip(query_text, qe)
                ]
            else:
                raw = [
                    study.make_text(q, method, wire, budget, model.tokenizer, idf)
                    for q in query_text
                ]
            decoded = [study.receive_text(p, budget, model.tokenizer) for p in raw]
            embeddings = study.encode(model, decoded, f"{partition}_receiver_{key}")
            directory = banks / key
            save_packets(directory, raw)
            np.save(directory / "decoded_embeddings.npy", embeddings)
            # Historical development shards had 20 BLAS threads at 128/256/512; held-out rankings used six.
            threads = 20 if partition == "dev" and budget >= 128 else 6
            with threadpool_limits(limits=threads, user_api="blas"):
                dense = study.norm(embeddings) @ study.norm(ce).T
                sparse = (
                    bm25_vectorizer.transform(decoded).astype(bool).astype(float)
                    @ bm25.T
                ).toarray()
                for decoder, scores in (
                    ("dense", dense),
                    ("bm25", sparse),
                    ("hybrid", study.rrf(dense, sparse, cids)),
                ):
                    per_query, top = study.metrics_scores(scores, cids, relevance, qids)
                    fullkey = key + "_" + decoder
                    rows[fullkey] = dict(
                        family="text",
                        method=method,
                        wire=wire,
                        decoder=decoder,
                        budget=budget,
                        bytes_mean=float(np.mean(list(map(len, raw)))),
                        bytes_max=max(map(len, raw)),
                        # The reported dictionary size used the historical formatted JSON on Windows.
                        setup_bytes=idf_size
                        if method in ("keywords", "lead_keywords")
                        else 0,
                        metrics=per_query.mean(axis=0).tolist(),
                    )
                    np.save(directory / (decoder + "_top10.npy"), top)
                    metrics[fullkey] = per_query
        write_json(output / f"{partition}_ids.json", qids)
        write_json(output / f"{partition}_results.json", rows)
        np.savez(output / f"{partition}_metrics.npz", **metrics)
        if stage == "development":
            selection = choose(rows, budgets)
            write_json(output / "selection.json", selection)
        for budget in budgets:
            winners = selection[str(budget)]
            for key in (winners["numeric_key"], winners["text_key"]):
                print(
                    f"{budget} bytes: {key}: {rows[key]['metrics'][2]:.9f} NDCG@10; "
                    f"{rows[key]['bytes_mean']:.6f} mean bytes",
                    flush=True,
                )


def main():
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser(
        "prepare",
        help="Acquire pinned ArguAna inputs and recompute the frozen split and IDF.",
    )
    prep.add_argument("--work", required=True, type=Path)
    source = prep.add_mutually_exclusive_group()
    source.add_argument(
        "--mirror-dir", type=Path, help="Directory with the three pinned mirror files."
    )
    source.add_argument(
        "--archive",
        type=Path,
        help="Optional local copy of the exact public ArguAna ZIP.",
    )
    for stage in ("primary", "development", "heldout"):
        p = sub.add_parser(stage)
        p.add_argument("--work", required=True, type=Path)
        if stage == "heldout":
            p.add_argument(
                "--selection",
                type=Path,
                help="Use an independently recomputed development selection.",
            )
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.work, args.archive, args.mirror_dir)
    else:
        run(args.work.resolve(), args.stage, getattr(args, "selection", None))


if __name__ == "__main__":
    main()
