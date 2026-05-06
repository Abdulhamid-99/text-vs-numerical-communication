"""VectorChannel encode/decode roundtrip + bit-cost correctness.

The thesis comparison hinges on this channel's bit cost matching the formula
   C_vector = d * b + 32
exactly (32 bits for the fp32 scale needed to dequantize). Quantization error
must also be small enough that the vector still carries the sender's signal —
we sanity-check via cosine similarity on a random vector.
"""
import numpy as np
import pytest

from budgetfair.comm.vector_channel import VectorChannel


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def test_bit_cost_8bit():
    ch = VectorChannel(dim=4096, bits_per_component=8)
    # 4096 components * 8 bits + 32 bits scale = 32,800 bits
    assert ch.bit_cost_for_message() == 4096 * 8 + 32 == 32_800


def test_bit_cost_4bit():
    ch = VectorChannel(dim=4096, bits_per_component=4)
    assert ch.bit_cost_for_message() == 4096 * 4 + 32 == 16_416


def test_bit_cost_arbitrary_dim():
    assert VectorChannel(dim=512, bits_per_component=8).bit_cost_for_message() == 512 * 8 + 32
    assert VectorChannel(dim=2048, bits_per_component=4).bit_cost_for_message() == 2048 * 4 + 32


def test_payload_size_8bit_matches_formula():
    """Wire-format byte count must equal (dim + 4): dim bytes int8 + 4 bytes scale."""
    ch = VectorChannel(dim=4096, bits_per_component=8)
    v = np.random.RandomState(0).randn(4096).astype(np.float32)
    msg = ch.encode(v, "agent_a", "agent_b")
    assert len(msg.payload) == 4096 + 4
    # And bit_cost from the message matches exactly
    assert msg.bit_cost == ch.bit_cost_for_message()


def test_payload_size_4bit_matches_formula():
    ch = VectorChannel(dim=4096, bits_per_component=4)
    v = np.random.RandomState(0).randn(4096).astype(np.float32)
    msg = ch.encode(v, "a", "b")
    # 4096 / 2 packed bytes + 4 scale bytes = 2052 bytes
    assert len(msg.payload) == 4096 // 2 + 4


def test_roundtrip_8bit_preserves_signal():
    """Quantization is lossy but should preserve cosine similarity > 0.99 for a
    typical Gaussian vector at 8-bit symmetric quantization."""
    rng = np.random.RandomState(0)
    v = rng.randn(4096).astype(np.float32)
    ch = VectorChannel(dim=4096, bits_per_component=8)
    msg = ch.encode(v, "a", "b")
    v_back = ch.decode(msg)
    assert v_back.shape == v.shape
    assert v_back.dtype == np.float32
    assert _cosine(v, v_back) > 0.99


def test_roundtrip_4bit_preserves_direction():
    """4-bit quantization is much more lossy but cosine should still be > 0.95
    for a smooth Gaussian vector."""
    rng = np.random.RandomState(1)
    v = rng.randn(4096).astype(np.float32)
    ch = VectorChannel(dim=4096, bits_per_component=4)
    msg = ch.encode(v, "a", "b")
    v_back = ch.decode(msg)
    assert _cosine(v, v_back) > 0.95


def test_roundtrip_handles_zero_vector():
    """An all-zero vector must roundtrip without divide-by-zero errors."""
    ch = VectorChannel(dim=512, bits_per_component=8)
    v = np.zeros(512, dtype=np.float32)
    msg = ch.encode(v, "a", "b")
    v_back = ch.decode(msg)
    np.testing.assert_array_equal(v_back, v)


def test_roundtrip_handles_extreme_outliers():
    """One huge component shouldn't break quantization; symmetric scale handles it."""
    ch = VectorChannel(dim=4096, bits_per_component=8)
    v = np.random.RandomState(0).randn(4096).astype(np.float32)
    v[0] = 1000.0  # outlier
    msg = ch.encode(v, "a", "b")
    v_back = ch.decode(msg)
    # The outlier is still recovered approximately
    assert abs(v_back[0] - v[0]) / abs(v[0]) < 0.02  # within 2%


def test_encode_rejects_wrong_shape():
    ch = VectorChannel(dim=4096, bits_per_component=8)
    with pytest.raises(ValueError):
        ch.encode(np.zeros(2048, dtype=np.float32), "a", "b")


def test_encode_rejects_wrong_bits_per_component():
    with pytest.raises(ValueError):
        VectorChannel(dim=4096, bits_per_component=16)
    with pytest.raises(ValueError):
        VectorChannel(dim=4096, bits_per_component=2)


def test_decode_rejects_short_payload():
    from budgetfair.comm.base import TransmittedMessage
    ch = VectorChannel(dim=4096, bits_per_component=8)
    bad = TransmittedMessage(
        payload=b"\x00\x00\x00",  # 3 bytes < 4 (scale alone)
        bit_cost=0, sender_id="a", receiver_id="b", timestamp=0.0, metadata={},
    )
    with pytest.raises(ValueError):
        ch.decode(bad)


def test_metadata_carries_channel_config():
    ch = VectorChannel(dim=4096, bits_per_component=8, layer_index=18)
    msg = ch.encode(np.zeros(4096, dtype=np.float32), "a", "b")
    assert msg.metadata["channel"] == "vector"
    assert msg.metadata["dim"] == 4096
    assert msg.metadata["bits_per_component"] == 8
    assert msg.metadata["layer_index"] == 18
