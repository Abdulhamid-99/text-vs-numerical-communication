"""Seeded support-paragraph QA diagnostic; oracle support is not proof of communication."""
from __future__ import annotations
import hashlib
import json
import random
import re
import string
from .base import Episode, Task

SCORER_VERSION = "qa-em-punctuation-first-v2"
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")

def normalize_answer(s: str) -> str:
    if not isinstance(s, str):
        raise TypeError("answer normalization requires a string")
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return " ".join(_ARTICLE_RE.sub(" ", s).split())

def _count(n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("episode count must be a nonnegative integer")

def _strings(value, name):
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError(f"{name} must be a list of nonempty strings")
    return value

class MultiDocQA(Task):
    name = "multidoc_qa"
    scorer_version = SCORER_VERSION

    def __init__(self, dataset="hotpotqa_dev", question_type=None, *,
                 dataset_revision=None, manifest_path=None, split="dev"):
        if dataset != "hotpotqa_dev":
            raise ValueError(f"unsupported dataset: {dataset!r}")
        if question_type not in (None, "bridge", "comparison"):
            raise ValueError("question_type must be bridge, comparison, or None")
        if split not in ("dev", "test"):
            raise ValueError("split must be dev or test")
        self.dataset_id = dataset
        self.question_type = question_type
        self.dataset_revision = dataset_revision
        self.manifest_path, self.split = manifest_path, split
        self.manifest_hash = self.dataset_fingerprint = None

    def episodes(self, n, seed):
        _count(n)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if self.manifest_path is not None:
            from .manifest import load_task_manifest, manifest_info
            out = load_task_manifest(self.manifest_path, split=self.split,
                                    task_name=self.name, expected_count=n)
            self.manifest_hash = manifest_info(self.manifest_path)["manifest_hash"]
            return out
        if self.split == "test":
            raise ValueError("test evaluation requires a frozen task manifest")
        if n == 0:
            return []
        from datasets import load_dataset
        kwargs = {"split": "validation"}
        if self.dataset_revision is not None:
            kwargs["revision"] = self.dataset_revision
        ds = list(load_dataset("hotpotqa/hotpot_qa", "distractor", **kwargs))
        ids, candidates = set(), []
        for entry in ds:
            ident = entry.get("id")
            if not isinstance(ident, str) or not ident or ident in ids:
                raise ValueError("dataset IDs must be nonempty and unique")
            ids.add(ident)
            if self.question_type is not None and entry.get("type") != self.question_type:
                continue
            question, answer = entry.get("question"), entry.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{ident}: invalid question")
            if not isinstance(answer, str) or not normalize_answer(answer):
                raise ValueError(f"{ident}: empty normalized target")
            context, sf = entry.get("context"), entry.get("supporting_facts")
            if not isinstance(context, dict) or not isinstance(sf, dict):
                raise ValueError(f"{ident}: context/supporting_facts must be objects")
            titles = _strings(context.get("title"), "context.title")
            sentences = context.get("sentences")
            if not isinstance(sentences, list) or len(titles) != len(sentences):
                raise ValueError(f"{ident}: parallel context arrays differ in length")
            if len(set(titles)) != len(titles):
                raise ValueError(f"{ident}: duplicate context article")
            for paragraph in sentences:
                _strings(paragraph, "context.sentences entry")
            support = _strings(sf.get("title"), "supporting_facts.title")
            indexes = sf.get("sent_id")
            if not isinstance(indexes, list) or len(indexes) != len(support):
                raise ValueError(f"{ident}: malformed supporting-fact arrays")
            by_title = dict(zip(titles, sentences))
            for title, index in zip(support, indexes):
                if (isinstance(index, bool) or not isinstance(index, int) or index < 0
                        or title not in by_title or index >= len(by_title[title])):
                    raise ValueError(f"{ident}: supporting sentence absent/invalid")
            if len(set(support)) != 2:
                continue
            paragraphs = [(t, " ".join(s)) for t, s in zip(titles, sentences) if t in set(support)]
            state = {"question": question, "passage_a": paragraphs[0][1],
                     "passage_b": paragraphs[1][1], "title_a": paragraphs[0][0], "title_b": paragraphs[1][0]}
            digest = hashlib.sha256(json.dumps([ident, state, answer], sort_keys=True,
                                               ensure_ascii=False).encode()).hexdigest()
            candidates.append(Episode(ident, state, answer, {
                "type": entry.get("type", "unknown"), "level": entry.get("level", "unknown"),
                "instance_sha256": digest, "split": "dev",
                "preprocessing": "gold-support-two-articles-v2", "scorer_version": SCORER_VERSION}))
        candidates.sort(key=lambda e: e.episode_id)
        self.dataset_fingerprint = hashlib.sha256(json.dumps(
            [e.metadata["instance_sha256"] for e in candidates]).encode()).hexdigest()
        if len(candidates) < n:
            raise ValueError(f"requested {n} eligible episodes; only {len(candidates)} available")
        random.Random(seed).shuffle(candidates)
        selected = candidates[:n]
        self.manifest_hash = hashlib.sha256(json.dumps(
            [e.metadata["instance_sha256"] for e in selected]).encode()).hexdigest()
        for e in selected:
            e.metadata.update(sampling_seed=seed, dataset_fingerprint=self.dataset_fingerprint,
                              sample_manifest_hash=self.manifest_hash)
        return selected

    def score(self, episode, final_answer):
        if not isinstance(episode.ground_truth, str) or not normalize_answer(episode.ground_truth):
            raise ValueError("gold answer must have nonempty normalized content")
        if not isinstance(final_answer, str) or not final_answer.strip():
            return False, 0.0
        correct = normalize_answer(final_answer) == normalize_answer(episode.ground_truth)
        return correct, float(correct)
