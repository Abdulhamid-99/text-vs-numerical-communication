"""Agent abstractions used by the runner.

An Agent is initialized once per episode; the runner calls `act(obs)` per round.
The Agent does NOT see the channel, the budget, or the network — those are the
runner's concerns. The agent just decides what to say.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentObservation:
    """What the runner hands to an agent each round."""
    transcript: list[dict[str, str]] = field(default_factory=list)
    """Prior messages, ordered. Each item: {"sender": agent_id, "text": str}."""

    private_context: dict[str, Any] = field(default_factory=dict)
    """Private state visible only to this agent (e.g., a passage, a map)."""

    task_state: dict[str, Any] = field(default_factory=dict)
    """Public task state visible to all agents (question, current grid, round info)."""


@dataclass
class AgentAction:
    """What an agent returns. Either `message` (text), `vector` (vector channel
    payload), or `task_action` (final QA answer / navigation move) — or a combination.
    """
    message: str | None = None
    vector: Any = None  # numpy.ndarray when the vector channel is active
    task_action: Any = None
    is_final: bool = False


class Agent(ABC):
    role: str
    agent_id: str

    @abstractmethod
    def act(self, obs: AgentObservation) -> AgentAction:
        """Produce one action given the current observation."""
