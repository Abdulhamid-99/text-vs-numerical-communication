"""Load the official no-fit transfer datasets and construct evaluation components."""

import csv
import hashlib
import io
import json
import tempfile
import unicodedata
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from download_source import download_file

BRIGHT_REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"
BRIGHT_URL = f"https://huggingface.co/datasets/xlangai/BRIGHT/resolve/{BRIGHT_REVISION}"
SCIFACT_URL = "https://huggingface.co/datasets/BeIR/scifact/resolve/b3b5335604bf5ee3c4447671af975ea25143d4f5"
SCIFACT_QRELS = "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
ARCHIVE_SHA = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
FILES = {
    "scifact": [
        (
            "scifact-corpus.parquet",
            f"{SCIFACT_URL}/corpus/corpus-00000-of-00001.parquet?download=true",
            "243324b35f03d82bd6d98a5f575966876e86cad7ce16e5333a35b1b793dc4f45",
        ),
        (
            "scifact-queries.parquet",
            f"{SCIFACT_URL}/queries/queries-00000-of-00001.parquet?download=true",
            "1c37956c5dc8b810b60302323c24d1a9e79e26411ba8f5ad9d0888642e2a9034",
        ),
        (
            "scifact-test.tsv",
            f"{SCIFACT_QRELS}/test.tsv?download=true",
            "0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c",
        ),
    ],
    "bright": [
        (
            "biology-examples.parquet",
            f"{BRIGHT_URL}/examples/biology-00000-of-00001.parquet",
            "6e105c4f09d9a70b8a20ed6a4d0e386823a5545151df41b3f0e64eb5c5987829",
        ),
        (
            "biology-documents.parquet",
            f"{BRIGHT_URL}/documents/biology-00000-of-00001.parquet",
            "8516d0c233f9c34e9eb6922b56e8a1698e5a6f6e504a9499fcd511cdd5741670",
        ),
    ],
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acquire(dataset, directory, download):
    """Accept an exact cached source or atomically install a verified download."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "scifact.zip"
    if dataset == "scifact" and archive.is_file():
        require(digest(archive) == ARCHIVE_SHA, "Source hash mismatch: scifact.zip")
        return
    for name, url, expected in FILES[dataset]:
        destination = directory / name
        if destination.is_file():
            require(digest(destination) == expected, f"Source hash mismatch: {name}")
            continue
        if not download:
            raise FileNotFoundError(
                f"Missing {name}. Use prepare to download this dataset first."
            )
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            download_file(url, temp_path)
            require(
                digest(temp_path) == expected,
                f"Downloaded source hash mismatch: {name}",
            )
            temp_path.replace(destination)
        finally:
            temp_path.unlink(missing_ok=True)


@dataclass
class Dataset:
    query_ids: list
    queries: list
    doc_ids: list
    corpus: list
    aliases: list
    qrels: dict
    groups: list


def connected_groups(keys_by_query):
    """Two queries are dependent if their keys overlap directly or transitively."""
    parents = {qid: qid for qid in keys_by_query}
    owners = {}

    def find(qid):
        while parents[qid] != qid:
            parents[qid] = parents[parents[qid]]
            qid = parents[qid]
        return qid

    for qid, keys in keys_by_query.items():
        for key in keys:
            owner = owners.setdefault(key, qid)
            parents[find(qid)] = find(owner)
    groups = defaultdict(list)
    for qid in keys_by_query:
        groups[find(qid)].append(qid)
    return sorted(sorted(group) for group in groups.values())


def read_data(dataset, directory):
    """Construct deterministic identities and labels after verifying original source files."""
    acquire(dataset, directory, download=False)
    if dataset == "bright":
        from pyarrow import parquet

        examples = parquet.read_table(
            directory / "biology-examples.parquet"
        ).to_pylist()
        documents = parquet.read_table(
            directory / "biology-documents.parquet"
        ).to_pylist()
        require(
            all(not (set(row["excluded_ids"]) - {"N/A"}) for row in examples),
            "Expected an empty Biology exclusion mask",
        )
        qids = [row["id"] for row in examples]
        queries = [row["query"] for row in examples]
        doc_ids = [doc["id"] for doc in documents]
        corpus = [doc["content"] for doc in documents]
        doc_lookup = dict(zip(doc_ids, corpus))
        qrels = {row["id"]: row["gold_ids"] for row in examples}
        keys = {
            row["id"]: [
                ("text", " ".join(doc_lookup[d].casefold().split()))
                for d in row["gold_ids"]
            ]
            + [("page", d.rsplit("/", 1)[-1]) for d in row["gold_ids_long"]]
            for row in examples
        }
        seed, width = 904133, 5
        expected = (103, 57359, 372, 100)
    else:
        if (directory / "scifact.zip").is_file():
            with zipfile.ZipFile(directory / "scifact.zip") as archive:
                documents = list(
                    map(json.loads, archive.read("scifact/corpus.jsonl").splitlines())
                )
                query_rows = list(
                    map(json.loads, archive.read("scifact/queries.jsonl").splitlines())
                )
                tsv = archive.read("scifact/qrels/test.tsv").decode("utf-8")
        else:
            from pyarrow import parquet

            documents = parquet.read_table(
                directory / "scifact-corpus.parquet"
            ).to_pylist()
            query_rows = parquet.read_table(
                directory / "scifact-queries.parquet"
            ).to_pylist()
            tsv = (directory / "scifact-test.tsv").read_text(encoding="utf-8")
        documents.sort(key=lambda row: row["_id"])
        all_queries = {row["_id"]: row["text"] for row in query_rows}
        qrels = defaultdict(list)
        for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
            require(int(row["score"]) > 0, "Unexpected nonpositive qrel")
            qrels[row["query-id"]].append(row["corpus-id"])
        doc_ids = [doc["_id"] for doc in documents]
        corpus = [
            (doc.get("title", "") + " " + doc["text"]).strip() for doc in documents
        ]
        qids = sorted(qrels)
        queries = [all_queries[q] for q in qids]
        keys = {
            q: [("doc", d) for d in qrels[q]]
            + [
                (
                    "query",
                    " ".join(
                        unicodedata.normalize("NFKC", all_queries[q]).casefold().split()
                    ),
                )
            ]
            for q in qids
        }
        seed, width = 904131, 4
        expected = (300, 5183, 339, 247)
    groups = connected_groups(keys)
    require(
        (len(qids), len(doc_ids), sum(map(len, qrels.values())), len(groups))
        == expected,
        "Unexpected benchmark inventory",
    )
    require(
        len(set(qids)) == len(qids) and len(set(doc_ids)) == len(doc_ids),
        "Duplicate identifiers",
    )
    perm = np.random.default_rng(seed).permutation(len(doc_ids))
    aliases = [f"doc_{int(i):0{width}d}" for i in perm]
    lookup = dict(zip(doc_ids, aliases))
    alias_qrels = {q: {lookup[d] for d in qrels[q]} for q in qids}
    return Dataset(qids, queries, doc_ids, corpus, aliases, alias_qrels, groups)
