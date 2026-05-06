"""Base Agent interface. Concrete agents (Planner, Worker, Critic, KnowledgeHolder)
subclass this and override `system_prompt` and `act`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentObservation:
    """Whatever the task hands the agent each step (text, partial state, prior messages)."""
    payload: Any


@dataclass
class AgentAction:
    """Either a text message or a vector message, plus optional task-level decisions."""
    message: str | None = None
    vector: Any = None  # numpy.ndarray when vector channel is active
    task_action: Any = None  # e.g., navigation move, or final QA answer


class Agent(ABC):
    role: str
    system_prompt: str

    @abstractmethod
    def act(self, obs: AgentObservation) -> AgentAction:
        """Produce one action given the current observation."""
