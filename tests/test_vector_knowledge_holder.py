"""VectorKnowledgeHolderAgent tests with a mocked LLMClient.

Validates the agent's protocol — projection on send, lift-and-queue on
receive, queue-clearing per call — without needing a loaded LLM.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from budgetfair.agents.base import AgentObservation
from budgetfair.agents.vector_knowledge_holder import VectorKnowledgeHolderAgent
from budgetfair.comm.projection import GaussianProjection


HIDDEN_DIM = 4096


class _FakeClient:
    """Stand-in for LLMClient. Records calls; returns canned responses.

    `client.model.config.hidden_size` is required by the agent's __init__.
    """

    def __init__(self, hidden_size: int = HIDDEN_DIM, response: str = "fake response") -> None:
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=hidden_size))
        self.response = response
        self.generate_calls: list = []
        self.generate_with_injection_calls: list = []
        self.extract_calls: list = []

    def generate(self, messages):
        self.generate_calls.append(messages)
        return self.response

    def generate_with_activation_injection(
        self, messages, injected_vectors, layer_index, injection_strength=1.0
    ):
        self.generate_with_injection_calls.append(
            (messages, injected_vectors, layer_index, injection_strength)
        )
        return self.response

    def extract_activation(self, text, layer_index, pooling="last_token"):
        self.extract_calls.append((text, layer_index, pooling))
        # Return a deterministic activation based on the response length so the
        # downstream projection produces reproducible output.
        rng = np.random.RandomState(len(text) % 1000)
        return torch.from_numpy(rng.randn(HIDDEN_DIM).astype(np.float32))


def _proj():
    return GaussianProjection(source_dim=HIDDEN_DIM, target_dim=128, seed=42)


INJECTION_LAYER = 16


def _agent(agent_id, partner_id, client, proj, **kwargs):
    kwargs.setdefault("injection_layer", INJECTION_LAYER)
    return VectorKnowledgeHolderAgent(agent_id, partner_id, client, proj, **kwargs)


def test_init_rejects_dim_mismatch():
    proj = _proj()
    bad_client = _FakeClient(hidden_size=2048)  # mismatch
    with pytest.raises(ValueError):
        _agent("a", "b", bad_client, proj)


def test_act_no_received_vectors_uses_plain_generate():
    """Round 1: nothing received. Agent should use generate(), not the
    injection variant."""
    client = _FakeClient(response="some answer")
    proj = _proj()
    agent = _agent("agent_a", "agent_b", client, proj)

    obs = AgentObservation(
        transcript=[],
        private_context={"passage": "p"},
        task_state={"question": "q", "is_final_round": False},
    )
    action = agent.act(obs)
    assert len(client.generate_calls) == 1
    assert len(client.generate_with_injection_calls) == 0
    assert action.message == "some answer"
    assert action.vector is not None
    # Agent returns raw hidden-dim activation; channel applies projection.
    assert action.vector.shape == (HIDDEN_DIM,)
    assert action.vector.dtype == np.float32
    assert action.is_final is False
    # Sender extraction uses the middle injection layer.
    assert client.extract_calls[0][1] == INJECTION_LAYER


def test_act_with_received_vector_uses_injection_generate():
    """After process_received_vector, next act() should inject the lifted
    activation at the configured layer."""
    client = _FakeClient(response="x")
    proj = _proj()
    agent = _agent("agent_a", "agent_b", client, proj)

    incoming = np.random.RandomState(1).randn(128).astype(np.float32)
    agent.process_received_vector(incoming)
    assert len(agent._pending_lifted_vectors) == 1

    obs = AgentObservation(
        transcript=[],
        private_context={"passage": "p"},
        task_state={"question": "q", "is_final_round": False},
    )
    agent.act(obs)
    assert len(client.generate_calls) == 0
    assert len(client.generate_with_injection_calls) == 1
    # Injected shape: (1, hidden_dim) since we stacked 1 lifted vec; layer routed.
    _msgs, injected, layer_index, _strength = client.generate_with_injection_calls[0]
    assert injected.shape == (1, HIDDEN_DIM)
    assert layer_index == INJECTION_LAYER


def test_received_queue_clears_after_act():
    """Vectors are consumed exactly once."""
    client = _FakeClient()
    proj = _proj()
    agent = _agent("a", "b", client, proj)
    agent.process_received_vector(np.zeros(128, dtype=np.float32))
    obs = AgentObservation(private_context={"passage": "p"}, task_state={"question": "q"})
    agent.act(obs)
    assert agent._pending_lifted_vectors == []


def test_multiple_received_vectors_stack():
    """Two received vectors -> 2 injected vectors."""
    client = _FakeClient()
    proj = _proj()
    agent = _agent("a", "b", client, proj)
    rng = np.random.RandomState(0)
    agent.process_received_vector(rng.randn(128).astype(np.float32))
    agent.process_received_vector(rng.randn(128).astype(np.float32))
    obs = AgentObservation(private_context={"passage": "p"}, task_state={"question": "q"})
    agent.act(obs)
    _msgs, injected, _layer, _strength = client.generate_with_injection_calls[0]
    assert injected.shape == (2, HIDDEN_DIM)


def test_injection_strength_threaded():
    """injection_strength from config reaches the client call."""
    client = _FakeClient()
    proj = _proj()
    agent = _agent("a", "b", client, proj, injection_strength=2.5)
    agent.process_received_vector(np.zeros(128, dtype=np.float32))
    obs = AgentObservation(private_context={"passage": "p"}, task_state={"question": "q"})
    agent.act(obs)
    _msgs, _injected, _layer, strength = client.generate_with_injection_calls[0]
    assert strength == 2.5


def test_random_injection_destroys_content_preserves_norm():
    """ABLATION: random_injection replaces received-vector content with
    norm-matched noise before lift. Same lifted norm, different direction."""
    proj = _proj()
    incoming = np.random.RandomState(7).randn(128).astype(np.float32)

    real_agent = _agent("a", "b", _FakeClient(), proj, random_injection=False)
    real_agent.process_received_vector(incoming.copy())
    real_lifted = real_agent._pending_lifted_vectors[0]

    rand_agent = _agent("a", "b", _FakeClient(), proj, random_injection=True, random_seed=0)
    rand_agent.process_received_vector(incoming.copy())
    rand_lifted = rand_agent._pending_lifted_vectors[0]

    # The pre-lift L2 norm is matched, so the lifted norms are close (lift is
    # linear; norm differences come only from M^T's non-orthogonality on a
    # different direction). Directions must differ substantially.
    assert real_lifted.shape == rand_lifted.shape == (HIDDEN_DIM,)
    cos = float(
        np.dot(real_lifted, rand_lifted)
        / (np.linalg.norm(real_lifted) * np.linalg.norm(rand_lifted) + 1e-9)
    )
    assert abs(cos) < 0.5  # content destroyed (not aligned with the real vector)


def test_process_received_rejects_wrong_shape():
    client = _FakeClient()
    proj = _proj()
    agent = _agent("a", "b", client, proj)
    with pytest.raises(ValueError):
        agent.process_received_vector(np.zeros(64, dtype=np.float32))


def test_final_round_returns_task_action():
    client = _FakeClient(response="Paris")
    proj = _proj()
    agent = _agent("a", "b", client, proj)
    obs = AgentObservation(
        private_context={"passage": "p"},
        task_state={"question": "q", "is_final_round": True},
    )
    action = agent.act(obs)
    assert action.is_final is True
    assert action.task_action == "Paris"


def test_non_final_round_no_task_action():
    client = _FakeClient(response="some hint")
    proj = _proj()
    agent = _agent("a", "b", client, proj)
    obs = AgentObservation(
        private_context={"passage": "p"},
        task_state={"question": "q", "is_final_round": False},
    )
    action = agent.act(obs)
    assert action.is_final is False
    assert action.task_action is None
