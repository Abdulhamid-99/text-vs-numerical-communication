"""Build the replay inputs directly from actual packet files and encoder outputs."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import zlib

import numpy as np
from tokenizers import Tokenizer

from verify import DESIGNS, HEADER, digest, rank, ranks_by_document, replay, require


TOKENIZER_SHA = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load(path):
    return np.load(path, allow_pickle=False)


def decode_text(packet, tokenizer, vocabulary_size):
    require(8 <= len(packet) <= 64, "Text packet length outside budget")
    magic, version, kind, length, reserved = HEADER.unpack_from(packet)
    payload = packet[8:]
    require((magic, version, reserved, length) == (b"SR", 1, 0, len(payload)), "Malformed text header")
    if kind == 1:
        return payload.decode("utf-8")
    if kind == 2:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(payload, 131073)
        require(len(raw) <= 131072 and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail, "Bad zlib frame")
        return raw.decode("utf-8")
    require(kind == 3 and length % 2 == 0, "Bad token frame")
    ids = np.frombuffer(payload, dtype="<u2").tolist()
    require(all(i < vocabulary_size for i in ids), "Unknown token")
    return tokenizer.decode(ids, skip_special_tokens=False)


def lexical_ranks(docs, received, tie):
    """Compute full BM25 ranks. Only query terms need postings, but lengths use every token."""
    terms = [set(WORDS.findall(text.lower())) for text in received]
    needed = set.union(*terms)
    entries = defaultdict(list)
    lengths = []
    for i, document in enumerate(docs):
        tokens = WORDS.findall(document.lower())
        lengths.append(len(tokens))
        for term, count in Counter(x for x in tokens if x in needed).items():
            entries[term].append((i, count))
    lengths = np.array(lengths, dtype=np.float64)
    weights = {}
    for term, posting in entries.items():
        pos, count = np.asarray(posting).T
        idf = np.log(1 + (len(docs)-len(pos)+0.5) / (len(pos)+0.5))
        weights[term] = (pos, count * 2.2 / (count + 1.2 * (0.25 + 0.75*lengths[pos]/lengths.mean())) * idf)
    ranks = np.empty((len(received), len(docs)), dtype=np.uint16)
    require(len(docs) < 65536, "Corpus exceeds the replay rank format")
    for i, query in enumerate(terms):
        scores = np.zeros(len(docs))
        for term in sorted(query & weights.keys()):
            pos, weight = weights[term]
            scores[pos] += weight
        ranks[i] = ranks_by_document(rank(scores[None, :], tie))[0]
    return ranks


def padded(rows):
    output = np.full((len(rows), max(map(len, rows))), -1, dtype=np.int32)
    for i, values in enumerate(rows):
        output[i, :len(values)] = values
    return output


def packet_paths(root, pattern, count):
    return [(root / pattern.format(i=i)).read_bytes() for i in range(count)]


def collect(dataset, run, data, tokenizer, split_path=None):
    expected_n = DESIGNS[dataset][0]
    if dataset == "arguana":
        split = read(split_path or run/"data_split.json")
        indices = split["receiver_corpus"]
        all_docs = records(data/"corpus.jsonl")
        rows = [all_docs[i] for i in indices]
        ids = [d["_id"] for d in rows]
        qids = read(run/"test_ids.json")
        all_queries = records(data/"queries.jsonl")
        require(qids == [all_queries[i]["_id"] for i in split["test"]], "Held-out order differs from frozen split")
        labels = [split["query_component_ids"][str(i)] for i in split["test"]]
        components = {label: i for i, label in enumerate(sorted(set(labels)))}
        groups = [components[label] for label in labels]
        relevance = defaultdict(set)
        with (data/"test.tsv").open(encoding="utf-8", newline="") as stream:
            for q, d, value in list(csv.reader(stream, delimiter="\t"))[1:]:
                if int(value)>0:
                    relevance[q].add(d)
        excluded = [[] for _ in qids]
        aliases = ids
        base = run/"test_packets"
        numeric = packet_paths(base/"raw_r384_b1", "{i:05d}.bin", expected_n)
        text_dir = base/"text_lead_keywords_tokens_64"
        text = packet_paths(text_dir, "{i:05d}.bin", expected_n)
        document_vectors = load(run/"corpus_embeddings.npy")[indices]
        text_vectors = load(text_dir/"decoded_embeddings.npy")
        numeric_top = load(base/"raw_r384_b1/top10.npy")
        text_top = load(text_dir/"hybrid_top10.npy")
        sender = load(run/"test_embeddings.npy")
    else:
        bright = dataset == "bright"
        rows = records(data/"receiver/corpus.jsonl")
        ids = [d["id"] for d in rows]
        evaluator = data/"evaluator"
        qids = (read(evaluator/"query_ids.json") if bright else
                [q["id"] for q in records(data/"sender/test_queries.jsonl")])
        members = read(evaluator/("components.json" if bright else "test_dependency_groups.json"))
        by_qid = {q: i for i, group in enumerate(members) for q in group}
        require(len(by_qid) == sum(map(len, members)) and set(by_qid) == set(qids), "Component membership")
        groups = [by_qid[q] for q in qids]
        relevance = {q: {d for d, grade in values.items() if grade > 0}
                     for q, values in read(evaluator/("qrels.json" if bright else "test_qrels.json")).items()}
        excluded_map = read(evaluator/"exclusions.json") if bright else {}
        excluded = [excluded_map.get(q, []) for q in qids]
        alias_map = read(data/"receiver/aliases.json" if bright else run/"receiver_id_aliases.json")
        aliases = [alias_map[d] for d in ids]
        base = run/"packets"
        numeric = packet_paths(base, "numeric_primary/{i:03d}.bin" if bright else "{i:03d}_numeric.bin", expected_n)
        text = packet_paths(base, "textpacket__lead_keywords__tokens/{i:03d}.bin" if bright else "{i:03d}_text.bin", expected_n)
        document_vectors = load(run/"receiver_corpus_embeddings.npy")
        text_vectors = load(run/("embeddings_text__lead_keywords__tokens.npy" if bright else "receiver_selected_text_embeddings.npy"))
        numeric_top = load(run/("top10_numeric_primary.npy" if bright else "top10_selected_numeric.npy"))
        text_top = load(run/("top10_prior_selected_text.npy" if bright else "top10_selected_text.npy"))
        sender = load(run/("sender_embeddings.npy" if bright else "sender_query_embeddings.npy"))
    documents = [r["content"] for r in rows] if dataset == "bright" else [(r.get("title", "")+" "+r["text"]).strip() for r in rows]
    require(len(qids) == len(set(qids)) == expected_n, "Query inventory mismatch")
    require(len(ids) == len(set(ids)) == len(set(aliases)), "Duplicate receiver document or alias")
    by_doc = {d: i for i, d in enumerate(ids)}
    gold = padded([sorted(by_doc[d] for d in relevance[q]) for q in qids])
    excluded = padded([[by_doc[d] for d in row] for row in excluded])
    numerical_bytes = np.stack([np.frombuffer(p, np.uint8) for p in numeric])
    require(sender.shape == (expected_n, 384) and np.isfinite(sender).all(), "Missing sender embeddings")
    np.testing.assert_array_equal(numerical_bytes[:, 8:], np.packbits(sender >= 0, axis=1, bitorder="little"))
    vocabulary_size = tokenizer.get_vocab_size(with_added_tokens=True)
    received = [decode_text(p, tokenizer, vocabulary_size) for p in text]
    original_order = np.asarray(sorted(range(len(ids)), key=lambda i: digest(("tie-819-"+aliases[i]).encode())))
    return {"document_vectors": document_vectors, "text_vectors": text_vectors,
            "numeric_packets": numerical_bytes, "text_headers": np.stack([np.frombuffer(p[:8], np.uint8) for p in text]),
            "text_packet_sha256": np.stack([np.frombuffer(bytes.fromhex(digest(p)), np.uint8) for p in text]),
            "lexical_ranks": lexical_ranks(documents, received, original_order),
            "tie_order": original_order.astype(np.uint16), "groups": np.asarray(groups, dtype=np.uint16),
            "gold": gold, "excluded": excluded, "numeric_top10": numeric_top.astype(np.uint16),
            "text_top10": text_top.astype(np.uint16)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DESIGNS)
    parser.add_argument("--run", required=True, type=Path, help="Completed run containing packet files and embeddings")
    parser.add_argument("--data", required=True, type=Path, help="ArguAna data or prepared transfer inputs")
    parser.add_argument("--tokenizer", required=True, type=Path, help="Pinned BGE tokenizer.json")
    parser.add_argument("--split", type=Path, help="ArguAna data_split.json, if stored outside --run")
    parser.add_argument("--output", type=Path, help="Optionally write receiver-only numerical evidence, after verification")
    parser.add_argument("--compare", type=Path, help="Check all rebuilt array values and packet digests against a receiver package")
    args = parser.parse_args()
    require(digest(args.tokenizer.read_bytes()) == TOKENIZER_SHA, "Unexpected tokenizer revision")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.no_truncation()
    arrays = collect(args.dataset, args.run, args.data, tokenizer, args.split)
    print(json.dumps(replay(arrays, args.dataset), indent=2, allow_nan=False))
    if args.compare:
        with np.load(args.compare, allow_pickle=False) as evidence:
            require(set(arrays) == set(evidence.files), "Evidence array inventory changed")
            for name, values in arrays.items():
                np.testing.assert_array_equal(values, evidence[name], err_msg=name)
        print("All rebuilt arrays match the comparison file, including the text-packet SHA-256 digests.")
    if args.output:
        with args.output.open("xb") as stream:
            np.savez_compressed(stream, **arrays)


if __name__ == "__main__":
    main()

