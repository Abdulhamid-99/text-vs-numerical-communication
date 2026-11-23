"""Cooperative Navigation — Map agent sees a grid; Actor agent moves on it blindly.
Communication is the only way the actor knows where to go.

Synthetic grid task: configurable size (e.g., 8x8), randomly placed walls and goal,
actor starts at (0,0). Goal: actor reaches goal cell within max_steps (default 50).
"""
from __future__ import annotations

from .base import Episode, Task


class CoopNavigation(Task):
    name = "coop_navigation"

    def __init__(self, grid_size: int = 8, max_steps: int = 50, wall_density: float = 0.15) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.wall_density = wall_density

    def episodes(self, n: int, seed: int) -> list[Episode]:
        raise NotImplementedError

    def score(self, episode, final_answer):
        """Success = goal reached within max_steps. Score = 1 - (steps_used / max_steps) on success, 0 otherwise."""
        raise NotImplementedError
