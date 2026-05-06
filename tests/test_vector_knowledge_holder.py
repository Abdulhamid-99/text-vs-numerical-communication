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
        self.generate_with_prefix_calls: list = []
        self.embed_calls: list = []

    def generate(self, messages):
        self.generate_calls.append(messages)
        return self.response

    def generate_with_prefix_embeds(self, messages, prefix_embeds):
        self.generate_with_prefix_calls.append((messages, prefix_embeds))
        return self.response

    def embed_text(self, text, pooling="mean"):
        self.embed_calls.append((text, pooling))
        # Return a deterministic embedding based on the response length so the
        # downstream projection produces reproducible output.
        rng = np.random.RandomState(len(text) % 1000)
        return torch.from_numpy(rng.randn(HIDDEN_DIM).astype(np.float32))


def _proj():
    return GaussianProjection(source_dim=HIDDEN_DIM, target_dim=128, seed=42)


def test_init_rejects_dim_mismatch():
    proj = _proj()
    bad_client = _FakeClient(hidden_size=2048)  # mismatch
    with pytest.raises(ValueError):
        VectorKnowledgeHolderAgent("a", "b", bad_client, proj)


def test_act_no_received_vectors_uses_plain_generate():
    """Round 1: nothing received. Agent should use generate(), not the prefix
    variant."""
    client = _FakeClient(response="some answer")
    proj = _proj()
    agent = VectorKnowledgeHolderAgent("agent_a", "agent_b", client, proj)

    obs = AgentObservation(
        transcript=[],
        private_context={"passage": "p"},
        task_state={"question": "q", "is_final_round": False},
    )
    action = agent.act(obs)
    assert len(client.generate_calls) == 1
    assert len(client.generate_with_prefix_calls) == 0
    assert action.message == "some answer"
    assert action.vector is not None
    assert action.vector.shape == (128,)
    assert action.vector.dtype == np.float32
    assert action.is_final is False


def test_act_with_received_vector_uses_prefix_generate():
    """After process_received_vector, next act() should pass prefix embeds."""
    client = _FakeClient(response="x")
    proj = _proj()
    agent = VectorKnowledgeHolderAgent("agent_a", "agent_b", client, proj)

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
    assert len(client.generate_with_prefix_calls) == 1
    # Prefix embed shape: (1, hidden_dim) since we stacked 1 lifted vec
    _msgs, prefix = client.generate_with_prefix_calls[0]
    assert prefix.shape == (1, HIDDEN_DIM)


def test_received_queue_clears_after_act():
    """Vectors are consumed exactly once."""
    client = _FakeClient()
    proj = _proj()
    agent = VectorKnowledgeHolderAgent("a", "b", client, proj)
    agent.process_received_vector(np.zeros(128, dtype=np.float32))
    obs = AgentObservation(private_context={"passage": "p"}, task_state={"question": "q"})
    agent.act(obs)
    assert agent._pending_lifted_vectors == []


def test_multiple_received_vectors_stack():
    """Two received vectors -> 2 prefix embeds."""
    client = _FakeClient()
    proj = _proj()
    agent = VectorKnowledgeHolderAgent("a", "b", client, proj)
    rng = np.random.RandomState(0)
    agent.process_received_vector(rng.randn(128).astype(np.float32))
    agent.process_received_vector(rng.randn(128).astype(np.float32))
    obs = AgentObservation(private_context={"passage": "p"}, task_state={"question": "q"})
    agent.act(obs)
    _msgs, prefix = client.generate_with_prefix_calls[0]
    assert prefix.shape == (2, HIDDEN_DIM)


def test_process_received_rejects_wrong_shape():
    client = _FakeClient()
    proj = _proj()
    agent = VectorKnowledgeHolderAgent("a", "b", client, proj)
    with pytest.raises(ValueError):
        agent.process_received_vector(np.zeros(64, dtype=np.float32))


def test_final_round_returns_task_action():
    client = _FakeClient(response="Paris")
    proj = _proj()
    agent = VectorKnowledgeHolderAgent("a", "b", client, proj)
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
    agent = VectorKnowledgeHolderAgent("a", "b", client, proj)
    obs = AgentObservation(
        private_context={"passage": "p"},
        task_state={"question": "q", "is_final_round": False},
    )
    action = agent.act(obs)
    assert action.is_final is False
    assert action.task_action is None
