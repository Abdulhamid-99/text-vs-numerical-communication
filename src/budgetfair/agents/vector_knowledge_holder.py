"""Multi-Doc QA agent that communicates via the vector channel.

Protocol per round:
  1. Agent sees its private passage + the question + any received vectors from
     the partner (lifted to source_dim and held as torch tensors).
  2. Agent calls `client.generate_with_activation_injection(messages,
     injected_vectors, layer_index)` — the lifted received vectors are ADDED to
     the residual stream at a middle decoder layer (ref:1, activation injection).
  3. Agent extracts the last-token activation of its own response at the same
     middle layer via `client.extract_activation(response, layer_index)` and
     returns it RAW (source_dim); the VectorChannel projects + quantizes it.
  4. Agent returns AgentAction(message=text_response, vector=raw_hidden_vector).

The runner is responsible for transmitting the vector via VectorChannel; the
receiving agent calls `process_received_vector(decoded_low_dim_np_array)` which
lifts and stores the vector for use on the next call to `act()`.

NOTE (2026-06-17): the previous implementation injected a single lifted vector
as ONE inputs_embeds prefix token at layer 0, and extracted the sender vector as
a layer-0 mean input-embedding. A random-vector ablation proved that mechanism
transmitted zero usable information (real activations == norm-matched noise). It
has been replaced with middle-layer activation injection per ref:1.
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
        injection_layer: int,
        injection_strength: float = 1.0,
        random_injection: bool = False,
        random_seed: int = 0,
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
        # Middle decoder layer used for BOTH sender extraction and receiver
        # injection (same residual-stream point keeps the two sides in one space).
        self.injection_layer = injection_layer
        # Scales the injected vector to (mean token residual norm) * strength.
        self.injection_strength = injection_strength
        # ABLATION: when True, the received vector's CONTENT is replaced with a
        # random Gaussian of matched norm before lifting. Everything else in the
        # pipeline is identical (same bit cost, same budget, same lift, same
        # injection). If accuracy is unchanged vs the real channel, the channel
        # transmits no usable information. See vector_failure_decision doc.
        self.random_injection = random_injection
        self._rng = np.random.RandomState(random_seed + (1 if agent_id.endswith("b") else 0))
        # Lifted, source-dim representations of vectors received from partner.
        # Held until consumed (cleared at the start of each act()).
        self._pending_lifted_vectors: list[np.ndarray] = []

    def process_received_vector(self, low_dim_vec: np.ndarray) -> None:
        """Called by the runner when the partner's vector is delivered.

        Lifts via P^T and queues for use on the next act() call. Under the
        random_injection ablation, the delivered vector is replaced by a random
        Gaussian of matched L2 norm (content destroyed, magnitude preserved).
        """
        if low_dim_vec.shape != (self.projection.target_dim,):
            raise ValueError(
                f"expected ({self.projection.target_dim},), got {low_dim_vec.shape}"
            )
        if self.random_injection:
            norm = float(np.linalg.norm(low_dim_vec))
            rand = self._rng.randn(self.projection.target_dim).astype(np.float32)
            rn = float(np.linalg.norm(rand))
            low_dim_vec = (rand / rn * norm) if rn > 0 else rand
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

        # Generate, injecting received vectors into the residual stream at the
        # configured middle layer (activation injection, ref:1).
        if self._pending_lifted_vectors:
            import torch

            stacked = np.stack(self._pending_lifted_vectors, axis=0)  # (n, d)
            injected_t = torch.from_numpy(stacked)
            response = self.client.generate_with_activation_injection(
                messages,
                injected_t,
                layer_index=self.injection_layer,
                injection_strength=self.injection_strength,
            )
        else:
            response = self.client.generate(messages)

        response = response.strip()
        # Clear the queue — vectors are consumed exactly once.
        self._pending_lifted_vectors = []

        # Extract the response's activation at the SAME middle layer the receiver
        # injects into (last-token pooling). Return it RAW (source_dim, no
        # projection here). The VectorChannel applies the projection at encode()
        # time — keeping projection in one place.
        act_t = self.client.extract_activation(
            response, layer_index=self.injection_layer
        )  # torch tensor (hidden_dim,)
        act_np = act_t.detach().cpu().numpy().astype(np.float32)

        return AgentAction(
            message=response,
            vector=act_np,              # shape (hidden_dim,)
            task_action=response if is_final else None,
            is_final=is_final,
        )
