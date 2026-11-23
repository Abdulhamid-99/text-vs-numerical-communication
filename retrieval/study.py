"""Sender codecs, receiver ranking and scoring from the retrieval experiment."""

import collections
import hashlib
import json
import re
import struct
import time
import zlib

import numpy as np

from model import model_load as model_load

OUT = None
BUDGETS = [32, 64, 128, 256, 512]
HEADER = struct.Struct("<2sBBHH")  # magic, version, kind, dimensions/length, reserved
TEXT_METHODS = [
    "prefix",
    "suffix",
    "headtail",
    "keywords",
    "lead_keywords",
    "central",
    "semantic_window",
]
TEXT_WIRES = ["utf8", "zlib", "tokens"]
NUM_CONFIGS = [
    {"basis": b, "rank": r, "bits": bits}
    for b in ["raw", "pca"]
    for r in ([384] if b == "raw" else [16, 32, 64, 96, 128, 192, 256, 384])
    for bits in ([1, 2, 4, 8] if b == "raw" else [2, 4, 8])
]


def sha(x):
    return hashlib.sha256(x).hexdigest()


def norm(x):
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite embedding")
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def encode(m, texts, role="unspecified"):
    started = time.perf_counter()
    result = m.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("<f4")
    if result.shape != (len(texts), 384) or not np.isfinite(result).all():
        raise ValueError("Invalid model output")
    with (OUT / "model_calls.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "role": role,
                    "n": len(texts),
                    "seconds": time.perf_counter() - started,
                    "text_sha256": [sha(s.encode()) for s in texts],
                    "embedding_sha256": [sha(row.tobytes()) for row in result],
                }
            )
            + "\n"
        )
    return result


def text(x):
    return (x.get("title", "") + " " + x["text"]).strip()


def words(s):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+", s.lower())


def sequence(s, method, idf):
    ws = s.split()
    if method == "prefix":
        return ws
    if method == "suffix":
        return list(reversed(ws))
    if method == "headtail":
        return [ws[i // 2] if i % 2 == 0 else ws[-1 - i // 2] for i in range(len(ws))]
    terms = words(s)
    freq = collections.Counter(terms)
    # Corpus-free at runtime: IDF dictionary learned on training docs and shared.
    rank = sorted(
        set(terms),
        key=lambda w: (
            -freq[w] * (np.log((idf["N"] + 1) / (idf["df"].get(w, 0) + 1)) + 1),
            terms.index(w),
        ),
    )
    if method == "keywords":
        return rank
    if method == "lead_keywords":
        return ws[:8] + [w for w in rank if w not in words(" ".join(ws[:8]))]
    # Extractive central sentence ordered by lexical coverage of the document.
    ss = re.split(r"(?<=[.!?])\s+", s)
    ordered = sorted(
        enumerate(ss),
        key=lambda it: (
            -sum(freq[w] for w in set(words(it[1]))) / max(1, len(words(it[1])) ** 0.5),
            it[0],
        ),
    )
    return [w for _, sentence in ordered for w in sentence.split()]


def tx_packet(content, wire, tok):
    if wire == "utf8":
        payload = content.encode()
        kind = 1
    elif wire == "zlib":
        payload = zlib.compress(content.encode(), 9)
        kind = 2
    else:
        ids = tok.encode(content, add_special_tokens=False)
        if any(i > 65535 for i in ids):
            raise ValueError("token uint16 overflow")
        payload = np.asarray(ids, dtype="<u2").tobytes()
        kind = 3
    return HEADER.pack(b"SR", 1, kind, len(payload), 0) + payload


def tx_decode(packet, tok):
    if len(packet) < 8:
        raise ValueError("Short packet")
    magic, v, k, n, res = HEADER.unpack(packet[:8])
    p = packet[8:]
    if magic != b"SR" or v != 1 or res or n != len(p):
        raise ValueError("Invalid text packet")
    if k == 1:
        return p.decode()
    if k == 2:
        d = zlib.decompressobj()
        out = d.decompress(p, 131073)
        if len(out) > 131072 or d.unconsumed_tail or d.unused_data or not d.eof:
            raise ValueError("Invalid compressed text")
        return out.decode()
    if k == 3:
        if len(p) % 2:
            raise ValueError("Invalid token payload")
        ids = np.frombuffer(p, dtype="<u2").tolist()
        vocab_size = len(tok)
        if any(i >= vocab_size for i in ids):
            raise ValueError("Invalid token ID")
        return tok.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    raise ValueError("Unknown text kind")


def make_text(s, method, wire, budget, tok, idf):
    full = [tx_packet(s, w, tok) for w in TEXT_WIRES]
    fitting = [p for p in full if len(p) <= budget]
    if fitting:
        return min(fitting, key=len)
    seq = sequence(s, method, idf)
    # Longest prefix in a sender-chosen ordering that fits actual packet bytes.
    # Scan all lengths because compressed lengths are not monotonic.
    best = tx_packet("", "utf8", tok)
    for n in range(1, len(seq) + 1):
        selected = seq[:n]
        if method == "suffix":
            selected = selected[::-1]
        elif method == "headtail":
            original = s.split()
            keep = set(range((n + 1) // 2)) | set(
                range(len(original) - n // 2, len(original))
            )
            selected = [w for i, w in enumerate(original) if i in keep]
        packet = tx_packet(" ".join(selected), wire, tok)
        if len(packet) <= budget:
            best = packet
        # Plain text/token sizes grow with prefix length. Tokens can merge but
        # never enough to fit after this conservative 100-byte excess bound.
        if wire != "zlib" and len(packet) > budget + 100:
            break
    return best


def semantic_text(s, wire, budget, m, idf, target):
    ws = s.split()
    starts = sorted(
        set(np.linspace(0, max(0, len(ws) - 1), min(24, len(ws)), dtype=int).tolist())
    )
    candidates = [
        make_text(" ".join(ws[start:]), "prefix", wire, budget, m.tokenizer, idf)
        for start in starts
    ]
    # Sender-only choice: preserve its own full query embedding, no corpus/qrels.
    decoded = [receive_text(p, budget, m.tokenizer) for p in candidates]
    em = encode(m, decoded, "sender_semantic_window_candidates")
    return candidates[int(np.argmax(em @ target))]


def num_key(cfg):
    return f"{cfg['basis']}_r{cfg['rank']}_b{cfg['bits']}"


def num_size(cfg):
    return 8 + (0 if cfg["bits"] == 1 else 4) + (cfg["rank"] * cfg["bits"] + 7) // 8


def num_setup(cfg):
    return 0 if cfg["basis"] == "raw" else 8 + 4 * (384 + 384 * cfg["rank"])


def num_packet(x, cfg, basis, mean):
    if (
        cfg not in NUM_CONFIGS
        or np.asarray(x).shape != (384,)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid input/config")
    if cfg["basis"] == "pca":
        validate_basis(basis, mean, cfg["rank"])
    r, b = cfg["rank"], cfg["bits"]
    y = x if cfg["basis"] == "raw" else (x - mean) @ basis[:r].T
    if b == 1:
        values = (y >= 0).astype("uint8")
        extra = b""
    else:
        scale = np.float32(max(float(np.max(np.abs(y))), 1e-12) / (2 ** (b - 1) - 1))
        values = (
            np.clip(np.rint(y / scale), -(2 ** (b - 1) - 1), 2 ** (b - 1) - 1)
            + (2 ** (b - 1) - 1)
        ).astype("uint8")
        extra = struct.pack("<f", scale)
    bitrows = ((values[:, None] >> np.arange(b)) & 1).astype("uint8")
    packed = np.packbits(bitrows.reshape(-1), bitorder="little").tobytes()
    return (
        HEADER.pack(b"SR", 1, 10 + b + (20 if cfg["basis"] == "pca" else 0), r, 0)
        + extra
        + packed
    )


def num_decode(packet, basis, mean):
    if len(packet) < 8:
        raise ValueError("Short packet")
    magic, v, k, r, res = HEADER.unpack(packet[:8])
    pca = k >= 30
    b = k - (30 if pca else 10)
    if magic != b"SR" or v != 1 or res or b not in [1, 2, 4, 8] or not 1 <= r <= 384:
        raise ValueError("Invalid numeric packet")
    cfg = {"basis": "pca" if pca else "raw", "rank": r, "bits": b}
    if len(packet) != num_size(cfg) or (not pca and r != 384):
        raise ValueError("Invalid numeric size")
    offset = 8 if b == 1 else 12
    values = (
        np.unpackbits(np.frombuffer(packet[offset:], dtype="uint8"), bitorder="little")[
            : r * b
        ].reshape(r, b)
        @ (2 ** np.arange(b))
    ).astype("float32")
    if b > 1 and np.any(values == 2**b - 1):
        raise ValueError("Unused quantization code")
    if b == 1:
        y = values * 2 - 1
    else:
        scale = struct.unpack("<f", packet[8:12])[0]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid scale")
        y = (values - (2 ** (b - 1) - 1)) * scale
    if pca:
        if b == 1:
            raise ValueError("One-bit PCA is not a registered codec")
        validate_basis(basis, mean, r)
        y = y @ basis[:r] + mean
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite reconstruction")
    return y, cfg


def validate_basis(basis, mean, r):
    if (
        np.asarray(basis).ndim != 2
        or basis.shape[1] != 384
        or basis.shape[0] < r
        or np.asarray(mean).shape != (384,)
    ):
        raise ValueError("Invalid basis shape")
    if not np.isfinite(basis[:r]).all() or not np.isfinite(mean).all():
        raise ValueError("Nonfinite basis")


def receive_text(packet, budget, tok):
    if type(budget) is not int or budget < 8 or len(packet) > budget:
        raise ValueError("Packet exceeds budget")
    return tx_decode(packet, tok)


def receive_numeric(packet, budget, basis, mean):
    if type(budget) is not int or budget < 8 or len(packet) > budget:
        raise ValueError("Packet exceeds budget")
    return num_decode(packet, basis, mean)


def metrics_scores(scores, cids, rel, qids):
    docs = set(cids)
    if (
        not qids
        or len(docs) != len(cids)
        or len(docs) < 10
        or len(set(qids)) != len(qids)
    ):
        raise ValueError("Incomplete or duplicated ranking identities")
    if scores.shape != (len(qids), len(cids)):
        raise ValueError("Score matrix does not cover every query and document")
    for q in qids:
        targets = rel.get(q)
        if (
            not targets
            or not docs.issuperset(targets)
            or len(set(targets)) != len(targets)
        ):
            raise ValueError(
                "Missing, duplicate or unknown positive relevance target: " + q
            )
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite retrieval score")
    order = np.asarray(
        sorted(range(len(cids)), key=lambda i: sha(("tie-819-" + cids[i]).encode()))
    )
    top = order[np.argsort(-scores[:, order], axis=1, kind="stable")[:, :10]]
    out = []
    for q, rr in zip(qids, top, strict=True):
        positions = [j + 1 for j, i in enumerate(rr) if cids[i] in rel[q]]
        rank = min(positions) if positions else 0
        dcg = sum(1 / np.log2(p + 1) for p in positions)
        ideal = sum(1 / np.log2(p + 1) for p in range(1, min(10, len(rel[q])) + 1))
        out.append([float(rank == 1), 1 / rank if rank else 0, dcg / ideal])
    return np.asarray(out), top


def retrieval(queries, corpus, qids, cids, rel, return_top=False):
    scores = norm(queries) @ norm(corpus).T
    out, top = metrics_scores(scores, cids, rel, qids)
    return (out, top) if return_top else out


def bm25_index(corpus):
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(tokenizer=words, token_pattern=None, lowercase=False)
    counts = vectorizer.fit_transform(corpus).astype("float64").tocsr()
    dl = np.asarray(counts.sum(axis=1)).ravel()
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(1 + (len(corpus) - df + 0.5) / (df + 0.5))
    coo = counts.tocoo()
    coo.data = (
        coo.data
        * 2.2
        / (coo.data + 1.2 * (0.25 + 0.75 * dl[coo.row] / dl.mean()))
        * idf[coo.col]
    )
    return vectorizer, coo.tocsr()


def rrf(a, b, cids):
    tie = np.asarray(
        sorted(range(len(cids)), key=lambda i: sha(("tie-819-" + cids[i]).encode()))
    )

    def ranks(s):
        order = tie[np.argsort(-s[:, tie], axis=1, kind="stable")]
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(s.shape[1])[None, :] + 1, axis=1)
        return ranks

    return 1 / (60 + ranks(a)) + 1 / (60 + ranks(b))
