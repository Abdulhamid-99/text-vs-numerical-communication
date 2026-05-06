"""Multi-Document QA — two agents each hold one supporting passage, must answer
a question that requires cross-passage synthesis.

Default dataset: HotpotQA dev (distractor split). Each question has 2 supporting
passages from different Wikipedia articles; we route one to each agent.
"""
from __future__ import annotations

from .base import Episode, Task


class MultiDocQA(Task):
    name = "multidoc_qa"

    def __init__(self, dataset: str = "hotpotqa_dev") -> None:
        self.dataset = dataset

    def episodes(self, n: int, seed: int) -> list[Episode]:
        raise NotImplementedError

    def score(self, episode, final_answer):
        """Exact-match accuracy on the ground-truth answer."""
        raise NotImplementedError
