"""Cooperative Navigation: Map agent sees grid, Actor agent moves. Communication mediates the gap."""
from __future__ import annotations

from .base import Agent, AgentAction, AgentObservation


class MapAgent(Agent):
    role = "map_holder"
    system_prompt = (
        "You see the full grid map. Your partner controls the moves but cannot see the map. "
        "Communicate the information they need to reach the goal in <= 50 steps."
    )

    def act(self, obs: AgentObservation) -> AgentAction:
        raise NotImplementedError


class ActorAgent(Agent):
    role = "actor"
    system_prompt = (
        "You move on a grid but cannot see it. Your partner can. Decide your next move based "
        "on what they tell you. Reach the goal in <= 50 steps."
    )

    def act(self, obs: AgentObservation) -> AgentAction:
        raise NotImplementedError
