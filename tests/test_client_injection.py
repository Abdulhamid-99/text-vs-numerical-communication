"""Tests for LLMClient's activation-injection mechanism (ref:1).

These exercise the forward-hook lifecycle and scaling WITHOUT loading a real LLM:
we build a tiny torch model whose decoder "block" is the identity, attach the
real LLMClient._injection_hook / _decoder_layers methods to a bare instance, and
assert the hook is registered, modifies the residual stream correctly, and is
removed (no leaked hooks).
"""
from types import SimpleNamespace

import torch

from budgetfair.model.client import LLMClient


HIDDEN = 32


class _FakeBlock(torch.nn.Module):
    """Identity decoder block: returns a bare tensor like Llama/Qwen >=4.54."""

    def forward(self, x):  # noqa: ANN001
        return x


class _FakeInner(torch.nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeBlock() for _ in range(n_layers)])


class _FakeModel(torch.nn.Module):
    """Mimics a CausalLM wrapper: decoder layers live at .model.layers."""

    def __init__(self, n_layers: int = 4) -> None:
        super().__init__()
        self.model = _FakeInner(n_layers)
        self.config = SimpleNamespace(hidden_size=HIDDEN)
        self.device = torch.device("cpu")


def _bare_client(n_layers: int = 4) -> LLMClient:
    """An LLMClient instance with only .model set (no weights loaded)."""
    client = LLMClient.__new__(LLMClient)  # bypass __init__ (no model download)
    client.model = _FakeModel(n_layers)
    return client


def test_decoder_layers_located():
    client = _bare_client(n_layers=5)
    layers = client._decoder_layers()
    assert isinstance(layers, torch.nn.ModuleList)
    assert len(layers) == 5


def test_hook_registered_and_removed():
    """The injection hook must be registered inside the context and removed
    on exit — no leaked hooks."""
    client = _bare_client(n_layers=4)
    block = client._decoder_layers()[2]
    inj = torch.ones(HIDDEN)

    assert len(block._forward_hooks) == 0  # clean to start
    with client._injection_hook(layer_index=2, injected_vectors=inj, injection_strength=1.0):
        assert len(block._forward_hooks) == 1  # registered
    assert len(block._forward_hooks) == 0  # removed in finally


def test_hook_removed_on_exception():
    """Even if the body raises, the finally block removes the hook."""
    client = _bare_client(n_layers=4)
    block = client._decoder_layers()[1]
    inj = torch.ones(HIDDEN)
    try:
        with client._injection_hook(layer_index=1, injected_vectors=inj):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert len(block._forward_hooks) == 0


def test_injection_modifies_prefill_not_decode():
    """During the hook, a multi-token (prefill) forward is modified; a
    single-token (decode-step) forward is left untouched."""
    client = _bare_client(n_layers=3)
    block = client._decoder_layers()[1]
    inj = torch.ones(HIDDEN)

    # Multi-token residual stream (batch=1, seq=5, hidden).
    prefill = torch.randn(1, 5, HIDDEN)
    # Single-token decode step.
    decode = torch.randn(1, 1, HIDDEN)

    with client._injection_hook(layer_index=1, injected_vectors=inj, injection_strength=1.0):
        out_prefill = block(prefill)
        out_decode = block(decode)

    # Prefill changed (injection added); decode unchanged (skipped).
    assert not torch.allclose(out_prefill, prefill)
    assert torch.allclose(out_decode, decode)


def test_injection_scaled_to_token_norm():
    """The added vector magnitude tracks (mean token norm * strength), not the
    raw injected norm — an unscaled lift would be negligible or destructive."""
    client = _bare_client(n_layers=3)
    block = client._decoder_layers()[1]
    # Tiny raw injection (norm ~1) but the residual tokens are large (norm ~100).
    inj = torch.ones(HIDDEN)  # norm = sqrt(32) ~ 5.66
    prefill = torch.full((1, 4, HIDDEN), 0.0)
    # Give each token a controlled norm by setting one component large.
    prefill[..., 0] = 100.0  # per-token norm ~100

    with client._injection_hook(layer_index=1, injected_vectors=inj, injection_strength=1.0):
        out = block(prefill)

    delta = (out - prefill)[0, 0]  # the added vector at position 0
    added_norm = torch.linalg.vector_norm(delta).item()
    # Scaled to ~ mean token norm (≈100), NOT the raw injection norm (≈5.66).
    assert 80.0 < added_norm < 120.0


def test_injection_strength_scales_magnitude():
    """Doubling injection_strength roughly doubles the added magnitude."""
    client = _bare_client(n_layers=3)
    block = client._decoder_layers()[1]
    inj = torch.ones(HIDDEN)
    prefill = torch.full((1, 4, HIDDEN), 0.0)
    prefill[..., 0] = 50.0

    with client._injection_hook(layer_index=1, injected_vectors=inj, injection_strength=1.0):
        out1 = block(prefill)
    with client._injection_hook(layer_index=1, injected_vectors=inj, injection_strength=2.0):
        out2 = block(prefill)

    n1 = torch.linalg.vector_norm((out1 - prefill)[0, 0]).item()
    n2 = torch.linalg.vector_norm((out2 - prefill)[0, 0]).item()
    assert abs(n2 - 2 * n1) < 0.05 * n2  # within 5%


def test_tuple_output_block_supported():
    """If a (future/older) decoder block returns a tuple, the hook injects into
    element 0 and preserves the rest."""
    client = _bare_client(n_layers=2)

    class _TupleBlock(torch.nn.Module):
        def forward(self, x):  # noqa: ANN001
            return (x, "extra")

    client.model.model.layers[0] = _TupleBlock()
    block = client._decoder_layers()[0]
    inj = torch.ones(HIDDEN)
    prefill = torch.randn(1, 4, HIDDEN)

    with client._injection_hook(layer_index=0, injected_vectors=inj):
        out = block(prefill)

    assert isinstance(out, tuple)
    assert out[1] == "extra"
    assert not torch.allclose(out[0], prefill)
