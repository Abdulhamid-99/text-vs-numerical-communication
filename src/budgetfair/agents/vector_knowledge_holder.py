"""Multi-Doc QA agent that communicates via the vector channel.

Protocol per round:
  1. Agent sees its private passage + the question + any received vectors from
     the partner (lifted to source_dim and held as torch tensors).
  2. Agent calls `client.generate_with_prefix_embeds(messages, prefix_embeds)`
     using the lifted received vectors as virtual prefix tokens.
  3. Agent extracts the mean input-embedding of its own response via
     `client.embed_text(response)` and projects it down via the channel's
     GaussianProjection. This 128-dim vector is what gets transmitted.
  4. Agent returns AgentAction(message=text_response, vector=projected_np_vector).

The runner is responsible for transmitting the vector via VectorChannel; the
receiving agent calls `process_received_vector(decoded_low_dim_np_array)` which
lifts and stores the vector for use on the next call to `act()`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..comm.projection import GaussianProjection
from ..model.client import LLMClient
from .base import Agent, AgentAction, AgentObservation


_SYSTEM_PROMPT = """You are agent {agent_id} in a 2-agent collaboration. Your partner is {partner_id}.

You hold a private passage that your partner cannot see. To answer the question correctly, you must communicate efficiently with your partner. The communication budget is limited, so be concise.

Your messages are also encoded into a compact vector representation that your partner sees as a soft prefix; the more compact and on-topic your message, the better that vector serves your partner.

Rules:
- Each round you produce one short message.
- Do not paste your entire passage verbatim. Summarize, quote selectively, or ask targeted questions.
- On the final round you will be asked to give your best answer; output ONLY the answer (no commentary, no explanation).
"""


class VectorKnowledgeHolderAgent(Agent):
    role = "vector_knowledge_holder"

    def __init__(
        self,
        agent_id: str,
        partner_id: str,
        client: LLMClient,
        projection: GaussianProjection,
    ) -> None:
        if projection.source_dim != client.model.config.hidden_size:
            raise ValueError(
                f"projection.source_dim ({projection.source_dim}) must equal "
                f"model hidden_size ({client.model.config.hidden_size})"
            )
        self.agent_id = agent_id
        self.partner_id = partner_id
        self.client = client
        self.projection = projection
        # Lifted, source-dim representations of vectors received from partner.
        # Held until consumed (cleared at the start of each act()).
        self._pending_lifted_vectors: list[np.ndarray] = []

    def process_received_vector(self, low_dim_vec: np.ndarray) -> None:
        """Called by the runner when the partner's vector is delivered.

        Lifts via P^T and queues for use on the next act() call.
        """
        if low_dim_vec.shape != (self.projection.target_dim,):
            raise ValueError(
                f"expected ({self.projection.target_dim},), got {low_dim_vec.shape}"
            )
        lifted = self.projection.lift(low_dim_vec)
        self._pending_lifted_vectors.append(lifted.astype(np.float32))

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

        # Map any prior text transcript onto chat turns. (In a vector-only run
        # the text transcript will be empty; in a hybrid run we pass both.)
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
                    "content": (
                        "Final round. Output the SHORTEST EXACT phrase that answers the question — "
                        "typically 1 to 5 words.\n"
                        "Rules:\n"
                        "- No explanation, no preamble, no period at the end.\n"
                        "- Match the granularity the question expects (a person's name, an entity, "
                        "a year, a country — not a sentence).\n"
                        "- If only ONE entity answers the question, output ONLY that entity. "
                        "Do not list alternatives or add hedging.\n"
                        "- If the answer appears verbatim in the discussion or a passage, quote it exactly."
                    ),
                }
            )

        # Generate, optionally with prefix embeds from received vectors
        if self._pending_lifted_vectors:
            import torch

            stacked = np.stack(self._pending_lifted_vectors, axis=0)  # (n_prefix, d)
            prefix_t = torch.from_numpy(stacked)
            response = self.client.generate_with_prefix_embeds(messages, prefix_t)
        else:
            response = self.client.generate(messages)

        response = response.strip()
        # Clear the queue — vectors are consumed exactly once.
        self._pending_lifted_vectors = []

        # Extract sender-side projection of the response for transmission.
        # We embed the RESPONSE text (what we just produced); that's what the
        # partner receives an encoding of.
        embed_t = self.client.embed_text(response)  # torch tensor (hidden_dim,)
        embed_np = embed_t.detach().cpu().numpy().astype(np.float32)
        projected_np = self.projection.project(embed_np)  # (target_dim,)

        return AgentAction(
            message=response,
            vector=projected_np,
            task_action=response if is_final else None,
            is_final=is_final,
        )
