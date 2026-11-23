"""Frozen, hash-validated evaluation manifests with explicit dev/test boundaries.

The entire manifest is an experimenter's private artifact: it contains targets
and counterfactual records. Only question and the assigned private passage may
enter agent observations. Hashes detect corruption, not malicious re-signing.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import string

from .base import Episode
from .multidoc_qa import _count
from .secret_passing import GENERATOR_VERSION, SecretPassingTask, format_record_passage, instance_hash

SCHEMA_VERSION = "budgetfair.task-manifest.v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _validate(document):
    if not isinstance(document, dict) or set(document) != {
        "schema_version", "task", "generator", "splits", "manifest_hash"
    }:
        raise ValueError("invalid manifest fields")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported manifest version")
    body = {k: v for k, v in document.items() if k != "manifest_hash"}
    if document["manifest_hash"] != _hash(body):
        raise ValueError("manifest hash mismatch")
    task = document["task"]
    if task not in ("secret_passing", "multidoc_qa") or not isinstance(document["generator"], dict):
        raise ValueError("unsupported manifest task/generator")
    if not isinstance(document["splits"], dict) or set(document["splits"]) != {"dev", "test"}:
        raise ValueError("manifest must contain explicit dev and test splits")
    if task == "secret_passing":
        params = document["generator"]
        try:
            SecretPassingTask(params["secret_mode"], params["n_digits"], params["n_letters"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid secret generator parameters") from exc
    identifiers, fingerprints, values = set(), set(), set()
    for split, rows in document["splits"].items():
        if not isinstance(rows, list):
            raise ValueError("manifest split must be a list")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"episode_id", "initial_state", "ground_truth", "metadata"}:
                raise ValueError("invalid manifest episode fields")
            ident, state, gold, meta = row["episode_id"], row["initial_state"], row["ground_truth"], row["metadata"]
            if not isinstance(ident, str) or not ident or ident in identifiers:
                raise ValueError("manifest episode identities must be unique across splits")
            if not isinstance(state, dict) or not isinstance(meta, dict) or meta.get("split") != split:
                raise ValueError("invalid state or split metadata")
            if not isinstance(gold, str) or not gold.strip() or not isinstance(state.get("question"), str) or not state["question"].strip():
                raise ValueError("empty question/target")
            digest = instance_hash(state, gold)
            if meta.get("instance_sha256") != digest or digest in fingerprints:
                raise ValueError("instance hash mismatch or repeated instance")
            identifiers.add(ident)
            fingerprints.add(digest)
            if task == "secret_passing":
                params = document["generator"]
                key = state.get("record_key")
                if params.get("version") == GENERATOR_VERSION and key is None:
                    raise ValueError("this generator version requires a public record key")
                if key is not None and (not isinstance(key, str) or not key.strip()
                        or f"item {key}?" not in state["question"]):
                    raise ValueError("record key must be present in the public question")
                mode = state.get("mode")
                if mode not in ("digit", "binding") or meta.get("mode") != mode or state.get("secret") != gold:
                    raise ValueError("invalid private record mode/target")
                if params["secret_mode"] != "both" and params["secret_mode"] != mode:
                    raise ValueError("episode mode differs from declared generator")
                try:
                    length = params["n_digits" if mode == "digit" else "n_letters"]
                except KeyError as exc:
                    raise ValueError("missing generator parameters") from exc
                alphabet = string.digits if mode == "digit" else string.ascii_uppercase
                if len(gold) != length or any(c not in alphabet for c in gold):
                    raise ValueError("target length/alphabet differs from generator")
                alternate = state.get("alternate_ground_truth")
                if not isinstance(alternate, str) or len(alternate) != length or any(c not in alphabet for c in alternate):
                    raise ValueError("invalid counterfactual target")
                for value, text_key in ((gold, "sender_text"), (alternate, "alternate_sender_text")):
                    folded = value.casefold()
                    if folded in values:
                        raise ValueError("targets/counterfactuals overlap within or across splits")
                    if folded in state["question"].casefold():
                        raise ValueError("target leaks into public question")
                    sender = state.get(text_key)
                    if not isinstance(sender, str) or value not in sender:
                        raise ValueError("private record does not contain its target")
                    values.add(folded)
                if not ident.endswith(digest):
                    raise ValueError("private-record ID is not content-bound")
            else:
                for key in ("passage_a", "passage_b", "title_a", "title_b"):
                    if not isinstance(state.get(key), str) or not state[key].strip():
                        raise ValueError("invalid QA manifest passages")
                if state["title_a"] == state["title_b"]:
                    raise ValueError("QA manifest requires two distinct articles")
    return document


def manifest_info(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return _validate(json.load(stream))


def load_task_manifest(path, *, split, task_name="secret_passing", expected_count=None):
    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    document = manifest_info(path)
    if document["task"] != task_name:
        raise ValueError("manifest task mismatch")
    rows = document["splits"][split]
    if expected_count is not None:
        _count(expected_count)
        if len(rows) != expected_count:
            raise ValueError(f"frozen {split} split contains {len(rows)}, requested {expected_count}; no implicit slicing")
    episodes = []
    for row in rows:
        episode = Episode(**row)
        episode.metadata["manifest_hash"] = document["manifest_hash"]
        episodes.append(episode)
    return episodes


def write_task_manifest(path, *, task_name, splits, generator):
    """Persist once; caller must prepare complete, disjoint dev/test episodes."""
    if not isinstance(splits, dict) or set(splits) != {"dev", "test"}:
        raise ValueError("provide exactly dev and test splits")
    rows = {}
    for split in ("dev", "test"):
        rows[split] = []
        for episode in splits.get(split, []):
            row = asdict(episode)
            row["metadata"].update(split=split, instance_sha256=instance_hash(row["initial_state"], row["ground_truth"]))
            row["metadata"].pop("manifest_hash", None)
            rows[split].append(row)
    body = {"schema_version": SCHEMA_VERSION, "task": task_name, "generator": generator, "splits": rows}
    document = dict(body, manifest_hash=_hash(body))
    _validate(document)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation makes a frozen split impossible to overwrite by retry.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return document


def create_secret_manifest(path, *, dev_count, test_count, secret_mode="binding",
                           n_digits=16, n_letters=16, minimum_entropy_bits=64):
    """Draw fresh high-entropy values using SystemRandom, then freeze both splits.

    Alternate records keep the same public entity/question but independently
    change its private value. They are oracle-only fields for controlled swaps.
    """
    _count(dev_count)
    _count(test_count)
    if not dev_count or not test_count:
        raise ValueError("both development and test splits must be nonempty")
    SecretPassingTask(secret_mode, n_digits, n_letters)  # parameter validation
    if isinstance(minimum_entropy_bits, bool) or not isinstance(minimum_entropy_bits, (int, float)) or not math.isfinite(minimum_entropy_bits) or minimum_entropy_bits < 0:
        raise ValueError("invalid minimum entropy")
    entropy = {"digit": n_digits * math.log2(10), "binding": n_letters * math.log2(26)}
    modes = ["digit", "binding"] if secret_mode == "both" else [secret_mode]
    if any(entropy[m] < minimum_entropy_bits for m in modes):
        raise ValueError("requested target space is below minimum_entropy_bits; increase its length")
    rng, used_values, used_entities = secrets.SystemRandom(), set(), set()

    def draw(alphabet, length, used, forbidden=""):
        if len(used) >= len(alphabet) ** length:
            raise ValueError("target space exhausted")
        for _ in range(10000):
            value = "".join(rng.choice(alphabet) for _ in range(length))
            if value not in used and value.casefold() not in forbidden.casefold():
                used.add(value)
                return value
        raise ValueError("unable to draw a disjoint private value")

    splits = {"dev": [], "test": []}
    for split, count in (("dev", dev_count), ("test", test_count)):
        for index in range(count):
            mode = modes[index % len(modes)]
            entity = draw(string.ascii_uppercase, 10, used_entities)
            question = f"What value is recorded for item {entity}? Copy only the value."
            alphabet, length = (string.digits, n_digits) if mode == "digit" else (string.ascii_uppercase, n_letters)
            value = draw(alphabet, length, used_values, question)
            alternate = draw(alphabet, length, used_values, question)
            def text(target):
                return format_record_passage(entity, target)
            state = {"question": question, "record_key": entity, "sender_text": text(value), "secret": value, "mode": mode,
                     "alternate_sender_text": text(alternate), "alternate_ground_truth": alternate}
            digest = instance_hash(state, value)
            ident = f"secret_{mode}_{split}_{index:06d}_{digest}"
            splits[split].append(Episode(ident, state, value, {"mode": mode, "split": split,
                "generator_version": GENERATOR_VERSION, "instance_sha256": digest}))
    generator = {"version": GENERATOR_VERSION, "randomness": "secrets.SystemRandom",
                 "secret_mode": secret_mode, "n_digits": n_digits, "n_letters": n_letters,
                 "entropy_bits_per_draw": {m: entropy[m] for m in modes},
                 "unique_primary_and_alternate_values": True}
    return write_task_manifest(path, task_name="secret_passing", splits=splits, generator=generator)


def validate_disjoint_manifests(*paths):
    """Reject reused instance identities/content and private targets across files."""
    ids, hashes, targets = set(), set(), set()
    for path in paths:
        doc = manifest_info(path)
        for rows in doc["splits"].values():
            for row in rows:
                ident, digest = row["episode_id"], row["metadata"]["instance_sha256"]
                if ident in ids or digest in hashes:
                    raise ValueError("manifest instances overlap")
                ids.add(ident)
                hashes.add(digest)
                if doc["task"] == "secret_passing":
                    for value in (row["ground_truth"], row["initial_state"]["alternate_ground_truth"]):
                        if value.casefold() in targets:
                            raise ValueError("manifest private targets overlap")
                        targets.add(value.casefold())
    return True
