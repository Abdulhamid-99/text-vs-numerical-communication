"""VectorChannel + GaussianProjection integration.

Validates that the *primary* thesis config (source_dim=4096, target_dim=128,
4-bit) gives the expected bit cost (544 bits/message) and survives a
quantize-dequantize roundtrip with the projected vector.
"""
import numpy as np
import pytest

from budgetfair.comm.projection import GaussianProjection
from budgetfair.comm.vector_channel import VectorChannel


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_primary_config_bit_cost_544():
    """Primary thesis setting: project 4096 -> 128, 4-bit. 544 bits/message."""
    proj = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    ch = VectorChannel(dim=128, bits_per_component=4, projection=proj)
    assert ch.bit_cost_for_message() == 128 * 4 + 32 == 544


def test_mid_config_bit_cost_1056():
    """Mid ablation: project 4096 -> 256, 4-bit. 1056 bits/message ~ text avg."""
    proj = GaussianProjection(source_dim=4096, target_dim=256, seed=42)
    ch = VectorChannel(dim=256, bits_per_component=4, projection=proj)
    assert ch.bit_cost_for_message() == 256 * 4 + 32 == 1056


def test_rich_config_bit_cost_2080():
    """Rich ablation (MLT-aligned): project 4096 -> 512, 4-bit. 2080 bits/msg."""
    proj = GaussianProjection(source_dim=4096, target_dim=512, seed=42)
    ch = VectorChannel(dim=512, bits_per_component=4, projection=proj)
    assert ch.bit_cost_for_message() == 512 * 4 + 32 == 2080


def test_encode_accepts_source_dim_input_when_projection_set():
    """With projection, encode takes (source_dim,) not (target_dim,)."""
    proj = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    ch = VectorChannel(dim=128, bits_per_component=4, projection=proj)
    v = np.random.RandomState(0).randn(4096).astype(np.float32)
    msg = ch.encode(v, "agent_a", "agent_b")
    assert msg.bit_cost == 544
    # 4-bit packed into 128/2 = 64 bytes + 4 byte scale = 68 bytes
    assert len(msg.payload) == 64 + 4


def test_encode_rejects_target_dim_input_when_projection_set():
    """If projection is set, the input must be source_dim shaped."""
    proj = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    ch = VectorChannel(dim=128, bits_per_component=4, projection=proj)
    with pytest.raises(ValueError):
        ch.encode(np.zeros(128, dtype=np.float32), "a", "b")


def test_no_projection_still_works():
    """Backward compat: VectorChannel(dim=4096) without projection unchanged."""
    ch = VectorChannel(dim=4096, bits_per_component=8)  # no projection arg
    v = np.random.RandomState(0).randn(4096).astype(np.float32)
    msg = ch.encode(v, "a", "b")
    assert msg.bit_cost == 4096 * 8 + 32
    decoded = ch.decode(msg)
    assert _cosine(v, decoded) > 0.99


def test_target_dim_mismatch_rejected_at_init():
    """projection.target_dim must match channel dim."""
    proj = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    with pytest.raises(ValueError):
        VectorChannel(dim=256, bits_per_component=4, projection=proj)


def test_metadata_records_projection_info():
    proj = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    ch = VectorChannel(dim=128, bits_per_component=4, projection=proj)
    msg = ch.encode(np.zeros(4096, dtype=np.float32), "a", "b")
    assert msg.metadata["projection_source_dim"] == 4096
    assert msg.metadata["projection_seed"] == 42
    assert msg.metadata["dim"] == 128


def test_metadata_records_no_projection():
    ch = VectorChannel(dim=4096, bits_per_component=8)
    msg = ch.encode(np.zeros(4096, dtype=np.float32), "a", "b")
    assert msg.metadata["projection_source_dim"] is None
    assert msg.metadata["projection_seed"] is None


def test_roundtrip_preserves_projected_signal_4bit():
    """Encode 4096 -> project 128 -> 4-bit quantize -> dequantize -> compare to
    the *projected* vector (not the original 4096). At 4-bit the cosine should
    still be well above 0.9 for a Gaussian input.
    """
    proj = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    ch = VectorChannel(dim=128, bits_per_component=4, projection=proj)
    rng = np.random.RandomState(0)
    v = rng.randn(4096).astype(np.float32)

    expected_projected = proj.project(v)
    msg = ch.encode(v, "a", "b")
    recovered = ch.decode(msg)

    assert recovered.shape == (128,)
    assert _cosine(expected_projected, recovered) > 0.95


def test_two_senders_with_same_seed_can_communicate():
    """Sender and receiver can build the channel independently from the same
    config; identical (source_dim, target_dim, seed) yields identical projections.
    The receiver's `decode` returns a vector that an independently-constructed
    *receiver-side* projection knows how to interpret.
    """
    proj_sender = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    proj_receiver = GaussianProjection(source_dim=4096, target_dim=128, seed=42)
    np.testing.assert_array_equal(proj_sender.matrix, proj_receiver.matrix)

    ch_send = VectorChannel(dim=128, bits_per_component=4, projection=proj_sender)
    ch_recv = VectorChannel(dim=128, bits_per_component=4, projection=proj_receiver)

    v = np.random.RandomState(0).randn(4096).astype(np.float32)
    msg = ch_send.encode(v, "a", "b")
    decoded = ch_recv.decode(msg)
    assert decoded.shape == (128,)
    # Sanity: round-trip the receiver's view through its own projection of v
    expected = proj_receiver.project(v)
    assert _cosine(expected, decoded) > 0.95
