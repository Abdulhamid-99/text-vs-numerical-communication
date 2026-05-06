"""Multi-Doc QA: each agent holds a disjoint passage and must communicate to answer."""
from __future__ import annotations

from .base import Agent, AgentAction, AgentObservation


class KnowledgeHolderAgent(Agent):
    role = "knowledge_holder"
    system_prompt = (
        "You are one of two agents. You hold a private passage. To answer the question, "
        "you must communicate with your partner. Be concise. The bit budget is limited."
    )

    def act(self, obs: AgentObservation) -> AgentAction:
        raise NotImplementedError
