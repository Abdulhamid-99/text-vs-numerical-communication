"""Vector communication agent with explicit per-episode private memory.

Delivered vectors remain available through communication and final submission.
The runner must call reset_episode before a new episode. Generated text is kept
as private memory only; vector transport never exposes it to the peer.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Optional

import numpy as np
import torch

from ..comm.projection import GaussianProjection
from ..model.client import LLMClient
from .base import Agent, AgentAction, AgentObservation
from .knowledge_holder import build_messages


class VectorKnowledgeHolderAgent(Agent):
    role = "vector_knowledge_holder"

    def __init__(self, agent_id: str, partner_id: str, client: LLMClient,
                 projection: GaussianProjection, injection_layer: int,
                 injection_strength: float = 1.0, random_injection: bool = False,
                 random_seed: int = 0, pooling: str = "last_token",
                 extraction_layer: Optional[int] = None) -> None:
        if projection.source_dim != client.model.config.hidden_size:
            raise ValueError("projection source dimension must match model hidden_size")
        extraction_layer = injection_layer if extraction_layer is None else extraction_layer
        for label, layer in (("injection_layer", injection_layer), ("extraction_layer", extraction_layer)):
            if isinstance(layer, bool) or not isinstance(layer, Integral) or layer < 0:
                raise ValueError(f"{label} must be a nonnegative decoder-block index")
        if pooling not in ("mean", "last_token", "cls"):
            raise ValueError(f"Unknown pooling: {pooling!r}")
        if not math.isfinite(float(injection_strength)):
            raise ValueError("injection_strength must be finite")
        if not isinstance(random_injection, bool):
            raise ValueError("random_injection must be boolean")
        if isinstance(random_seed, bool) or not isinstance(random_seed, Integral) or not 0 <= random_seed < 2**32:
            raise ValueError("random_seed must be an integer in [0, 2**32)")
        self.agent_id, self.partner_id = agent_id, partner_id
        self.client, self.projection = client, projection
        self.injection_layer, self.extraction_layer = int(injection_layer), int(extraction_layer)
        self.injection_strength = float(injection_strength)
        self.pooling = pooling
        self.random_injection = random_injection
        self._random_seed = (int(random_seed) + (1 if agent_id.endswith("b") else 0)) % 2**32
        self.reset_episode()

    def reset_episode(self) -> None:
        """Clear all per-episode information and reset the independent ablation RNG."""
        self._own_history: list[str] = []
        # Historical field name retained for integration; vectors now persist
        # within an episode, not merely until the next act() call.
        self._pending_lifted_vectors: list[np.ndarray] = []
        self._rng = np.random.RandomState(self._random_seed)

    def process_received_vector(self, low_dim_vec: np.ndarray) -> None:
        if np.iscomplexobj(low_dim_vec):
            raise ValueError("received vector must be real")
        with np.errstate(over="ignore", invalid="ignore"):
            vector = np.asarray(low_dim_vec, dtype=np.float32)
        if vector.shape != (self.projection.target_dim,):
            raise ValueError(f"expected ({self.projection.target_dim},), got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError("received vector must contain finite float32 values")
        if self.random_injection:
            norm = np.linalg.norm(vector.astype(np.float64))
            random = self._rng.randn(self.projection.target_dim)
            random_norm = np.linalg.norm(random)
            with np.errstate(over="ignore", invalid="ignore"):
                vector = (random / random_norm * norm).astype(np.float32) if random_norm else np.zeros_like(vector)
            if not np.isfinite(vector).all():
                raise ValueError("norm-matched random vector overflows float32")
        lifted = self.projection.lift(vector)
        self._pending_lifted_vectors.append(lifted.copy())

    def act(self, obs: AgentObservation) -> AgentAction:
        is_final = bool(obs.task_state.get("is_final_round", False))
        messages = build_messages(self.agent_id, self.partner_id, obs, self._own_history)
        if self._pending_lifted_vectors:
            vectors = torch.from_numpy(np.stack(self._pending_lifted_vectors, axis=0))
            response = self.client.generate_with_activation_injection(messages, vectors,
                layer_index=self.injection_layer, injection_strength=self.injection_strength)
        else:
            response = self.client.generate(messages)
        response = response.strip()
        activation = None
        if not is_final:
            tensor = self.client.extract_activation(response, layer_index=self.extraction_layer,
                                                    pooling=self.pooling)
            activation = tensor.detach().cpu().numpy().astype(np.float32)
            if activation.shape != (self.projection.source_dim,) or not np.isfinite(activation).all():
                raise ValueError("extracted activation must be a finite model-width vector")
        self._own_history.append(response)
        return AgentAction(message=response, vector=activation,
                           task_action=response if is_final else None, is_final=is_final)
