"""Task interface. A Task generates episodes and scores them."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Episode:
    episode_id: str
    initial_state: Any
    ground_truth: Any
    metadata: dict


@dataclass
class EpisodeResult:
    episode_id: str
    success: bool
    score: float       # task-specific (accuracy, steps-to-goal, etc.)
    final_answer: Any
    transcript: list   # full message log
    bits_used: int
    wall_time_ms: float


class Task(ABC):
    name: str

    @abstractmethod
    def episodes(self, n: int, seed: int) -> list[Episode]: ...

    @abstractmethod
    def score(self, episode: Episode, final_answer: Any) -> tuple[bool, float]: ...
