"""Shared prompts and an episode-local text agent for protocol v2."""
from __future__ import annotations

from ..model.client import LLMClient
from .base import Agent, AgentAction, AgentObservation


_SYSTEM_PROMPT = """You are agent {agent_id} in a 2-agent collaboration. Your partner is {partner_id}.
You each have private information. Use the supplied records and messages to answer the question.
During communication, send the specific facts your partner needs in one concise message.
If your private record contains the requested value, copy that value exactly to your partner.
Do not invent values, ask riddles, encrypt values, or reinterpret a stored value as a cipher.
If you have no relevant information, say UNKNOWN.
The final answer is a separate local step. Follow its output format exactly.
"""

_RECORD_FINAL_SYSTEM = (
    "Answer the question using the supplied sources. Copy recorded values exactly, preserving all letters "
    "and leading zeros. Do not decode or reinterpret a recorded value as a cipher. "
    "Output only the requested answer, with no added punctuation, quotes, labels, or explanation. "
    "If no source supplies the answer, output UNKNOWN."
)


def build_messages(agent_id: str, partner_id: str, obs: AgentObservation,
                   own_history: list[str] | None = None) -> list[dict[str, str]]:
    """Same prompts across channels; private notes are never delivered to a partner."""
    passage = obs.private_context.get("passage", "")
    question = obs.task_state.get("question", "")
    record_key = obs.task_state.get("record_key")
    if record_key is not None:
        if not isinstance(record_key, str) or not record_key.strip() or f"item {record_key}?" not in question:
            raise ValueError("record_key must be an explicit key in the public question")
    if obs.task_state.get("is_final_round", False) and record_key is not None:
        # The one-query record protocol associates the received payload with
        # its PUBLIC key. Never retrieve or reconstruct a value from task gold.
        # This source formatting is identical for text and vector agents;
        # vector-only observations have no received text record to render.
        records = "\n".join(
            f"Lookup record: item {record_key}; value {entry['text']}"
            for entry in obs.transcript if entry["sender"] != agent_id
        )
        private = passage
        if own_history:
            private += "\nMy earlier private notes/messages:\n" + "\n".join(own_history)
        return [
            {"role": "system", "content": _RECORD_FINAL_SYSTEM},
            {"role": "user", "content": (
                f"Final round. Question: {question}\n\nPrivate source: {private}\n\n"
                f"Received records:\n{records or '(none)'}\n\nAnswer:"
            )},
        ]
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(agent_id=agent_id, partner_id=partner_id)},
        {"role": "user", "content": f"Your private passage:\n{passage}\n\nQuestion: {question}"},
    ]
    if own_history:
        messages.append({"role": "assistant", "content": "My earlier private notes/messages:\n" + "\n".join(own_history)})
    for entry in obs.transcript:
        if entry["sender"] != agent_id:
            messages.append({"role": "user", "content": f"[{entry['sender']}]: {entry['text']}"})
    if obs.task_state.get("is_final_round", False):
        messages.append({"role": "user", "content": (
            "Final round. Output ONLY the exact value or short phrase answering the question. "
            "Use the private records and received information. Preserve all letters and leading zeros. "
            "Do not explain, decode, add punctuation, or list alternatives. If unknown, output UNKNOWN."
        )})
    return messages


class KnowledgeHolderAgent(Agent):
    role = "knowledge_holder"

    def __init__(self, agent_id: str, partner_id: str, client: LLMClient) -> None:
        self.agent_id, self.partner_id, self.client = agent_id, partner_id, client
        self._own_history: list[str] = []

    def reset_episode(self) -> None:
        self._own_history.clear()

    def act(self, obs: AgentObservation) -> AgentAction:
        messages = build_messages(self.agent_id, self.partner_id, obs, self._own_history)
        response = self.client.generate(messages).strip()
        final = bool(obs.task_state.get("is_final_round", False))
        self._own_history.append(response)
        return AgentAction(message=response, task_action=response if final else None, is_final=final)
