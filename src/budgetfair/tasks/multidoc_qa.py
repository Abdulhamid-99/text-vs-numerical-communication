"""Multi-Document QA on HotpotQA (distractor split).

Each question has 2 supporting paragraphs from different Wikipedia articles.
We route paragraph 1 to `agent_a` and paragraph 2 to `agent_b`. Distractor
paragraphs are not used in the pilot.

Scoring is normalized exact-match (lowercase, strip articles + punctuation).
"""
from __future__ import annotations

import re
import string
from typing import Any

from .base import Episode, Task


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    """Standard SQuAD/HotpotQA EM normalization."""
    s = s.lower()
    s = _ARTICLE_RE.sub(" ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


class MultiDocQA(Task):
    name = "multidoc_qa"

    def __init__(self, dataset: str = "hotpotqa_dev", question_type: str | None = None) -> None:
        """`question_type` filters HotpotQA by its `type` field:
          - None: no filter (default).
          - "bridge": multi-hop reasoning questions (H3 predicts vector wins).
          - "comparison": symbolic-comparison questions (H3 predicts text wins).
        """
        self.dataset_id = dataset
        self.question_type = question_type

    def episodes(self, n: int, seed: int) -> list[Episode]:
        # Lazy import: unit tests for normalize_answer don't need the datasets package.
        from datasets import load_dataset

        if self.dataset_id != "hotpotqa_dev":
            raise NotImplementedError(f"Dataset {self.dataset_id!r} not supported.")

        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        sorted_entries = sorted(ds, key=lambda e: e["id"])
        if self.question_type is not None:
            sorted_entries = [e for e in sorted_entries if e.get("type") == self.question_type]

        out: list[Episode] = []
        for entry in sorted_entries:
            sf_titles = set(entry["supporting_facts"]["title"])
            paragraphs = [
                (title, " ".join(sentences))
                for title, sentences in zip(
                    entry["context"]["title"], entry["context"]["sentences"]
                )
                if title in sf_titles
            ]
            if len(paragraphs) < 2:
                continue
            out.append(
                Episode(
                    episode_id=entry["id"],
                    initial_state={
                        "question": entry["question"],
                        "passage_a": paragraphs[0][1],
                        "passage_b": paragraphs[1][1],
                        "title_a": paragraphs[0][0],
                        "title_b": paragraphs[1][0],
                    },
                    ground_truth=entry["answer"],
                    metadata={
                        "type": entry.get("type", "unknown"),
                        "level": entry.get("level", "unknown"),
                    },
                )
            )
            if len(out) >= n:
                break

        return out

    def score(self, episode: Episode, final_answer: Any) -> tuple[bool, float]:
        if final_answer is None:
            return False, 0.0
        gold = normalize_answer(str(episode.ground_truth))
        pred = normalize_answer(str(final_answer))
        em = pred == gold
        return em, (1.0 if em else 0.0)
