"""Multi-Doc QA: each agent holds one private passage; both must collaborate to answer.

The agent:
  - Builds a chat-format message list including its system prompt, its private passage,
    the public question, and the prior transcript (mapped to user/assistant turns).
  - Calls the LLM client to generate one short response.
  - On the final round, asks the model for its best final answer (no commentary).
"""
from __future__ import annotations

from ..model.client import LLMClient
from .base import Agent, AgentAction, AgentObservation


_SYSTEM_PROMPT = """You are agent {agent_id} in a 2-agent collaboration. Your partner is {partner_id}.

You hold a private passage that your partner cannot see. To answer the question correctly, you must communicate efficiently with your partner. The communication budget is limited, so be concise.

Rules:
- Each round you produce one short message.
- Do not paste your entire passage verbatim. Summarize, quote selectively, or ask targeted questions.
- On the final round you will be asked to give your best answer; output ONLY the answer (no commentary, no explanation).
"""


class KnowledgeHolderAgent(Agent):
    role = "knowledge_holder"

    def __init__(self, agent_id: str, partner_id: str, client: LLMClient) -> None:
        self.agent_id = agent_id
        self.partner_id = partner_id
        self.client = client

    def act(self, obs: AgentObservation) -> AgentAction:
        passage = obs.private_context.get("passage", "")
        question = obs.task_state.get("question", "")
        is_final = bool(obs.task_state.get("is_final_round", False))

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT.format(
                    agent_id=self.agent_id, partner_id=self.partner_id
                ),
            },
            {
                "role": "user",
                "content": f"Your private passage:\n{passage}\n\nQuestion: {question}",
            },
        ]

        # Map the transcript onto chat turns from this agent's perspective:
        #   - messages I sent  -> assistant
        #   - messages from partner -> user (prefixed so the model sees who said what)
        for entry in obs.transcript:
            if entry["sender"] == self.agent_id:
                messages.append({"role": "assistant", "content": entry["text"]})
            else:
                messages.append(
                    {"role": "user", "content": f"[{entry['sender']}]: {entry['text']}"}
                )

        if is_final:
            messages.append(
                {
                    "role": "user",
                    "content": "Final round. Output ONLY your best final answer to the question. No explanation, no preamble — just the answer.",
                }
            )

        response = self.client.generate(messages).strip()

        return AgentAction(
            message=response,
            task_action=response if is_final else None,
            is_final=is_final,
        )
