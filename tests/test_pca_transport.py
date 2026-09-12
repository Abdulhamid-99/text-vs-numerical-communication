"""Exercise the real tiny causal model through the compressed receiver boundary."""
import hashlib

import numpy as np
import pytest

from .test_span_transport import client, public, single_cpu_thread, hooks, capture_generation
from budgetfair.comm.pca_packet import load_asset, encode_coefficients, encode_projected
from budgetfair.comm.tensor_packet import encode_tensor, decode_packet
from budgetfair.model.span_transport import encode_message, receive_packet
from budgetfair.model.pca_transport import receive_projected_packet


def make_asset(tmp_path, dimensions=16, layer=0):
    path = tmp_path / "basis.npz"
    np.savez(path, mean=np.zeros(16, np.float32), components=np.eye(16, dtype=np.float32)[:dimensions],
             layer=np.array(layer, np.int32), schema=np.array(1, np.int32))
    return load_asset(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_identity_numeric_basis_matches_uncompressed_injection_and_actual_logits(client, public, tmp_path):
    asset = make_asset(tmp_path)
    original, _ = encode_message(client, public, "ABBA", layer=0)
    compressed = encode_projected(decode_packet(original).array, asset, dimensions=16, codec="float32")
    reference, _, reference_logits = capture_generation(client,
        lambda: receive_packet(client, public, original, expected_layer=0))
    result, _, actual_logits = capture_generation(client,
        lambda: receive_projected_packet(client, public, compressed, asset, expected_layer=0,
                                        expected_dimensions=16, expected_codec="float32"))
    assert result["output_ids"] == reference["output_ids"]
    assert len(actual_logits) == len(reference_logits) == 3
    for before, after in zip(reference_logits, actual_logits):
        assert np.array_equal(before.cpu().numpy(), after.cpu().numpy())
    assert result["wire_bits"] == len(compressed) * 8
    assert result["packet_sha256"] == hashlib.sha256(compressed).hexdigest()
    assert result["injections"][0]["applied_dtype"] == "torch.float32"
    assert hooks(client) == {}


def test_known_int4_coefficients_match_independently_built_hidden_span(client, public, tmp_path):
    asset = make_asset(tmp_path, dimensions=8)
    coefficients = np.array([[-7, -1, 0, 1, 2, 3, 5, 7], [7, 5, 3, 2, 1, 0, -1, -7]], np.float32)
    # Row maxima7 make wire scales exactly1, so these values are exactly representable.
    packet = encode_coefficients(coefficients, asset, codec="int4")
    expected = np.zeros((2, 16), np.float32)
    expected[:, :8] = coefficients
    reference = receive_packet(client, public, encode_tensor(expected, role="hidden_states", layer_index=0), expected_layer=0)
    result = receive_projected_packet(client, public, packet, asset, expected_layer=0,
                                      expected_dimensions=8, expected_codec="int4")
    assert result["injections"][0]["applied_array_sha256"] == hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest()
    assert result["output_ids"] == reference["output_ids"]
    assert result["wire_bits"] == 640
    assert result["local_reconstructed_buffer_bytes"] * 8 == 1248
    assert result["receiver_input_ids"] == [1, 3, 10, 10, 8, 9]
    assert hooks(client) == {}


@pytest.mark.parametrize("failure", ["basis", "layer", "codec", "rank", "truncated"])
def test_invalid_compressed_input_rejected_before_model_execution(client, public, tmp_path, monkeypatch, failure):
    asset = make_asset(tmp_path, dimensions=8)
    packet = encode_coefficients(np.ones((2, 8), np.float32), asset, codec="int4")
    layer, rank, codec = 0, 8, "int4"
    if failure == "basis": packet = packet[:4] + bytes(32) + packet[36:]
    elif failure == "layer": layer = 1
    elif failure == "codec": codec = "int8"
    elif failure == "rank": rank = 4
    elif failure == "truncated": packet = packet[:-1]
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid compressed packet reached the model")
    monkeypatch.setattr(client.model, "generate", forbidden)
    with pytest.raises(ValueError):
        receive_projected_packet(client, public, packet, asset, expected_layer=layer,
                                 expected_dimensions=rank, expected_codec=codec)
    assert hooks(client) == {}


def test_model_failure_does_not_leave_compressed_state_hook_installed(client, public, tmp_path, monkeypatch):
    asset = make_asset(tmp_path, dimensions=8)
    packet = encode_coefficients(np.ones((2, 8), np.float32), asset, codec="int4")
    def fail(**kwargs):
        raise RuntimeError("Injected actual model execution failure")
    monkeypatch.setattr(client.model, "generate", fail)
    with pytest.raises(RuntimeError, match="execution failure"):
        receive_projected_packet(client, public, packet, asset, expected_layer=0,
                                 expected_dimensions=8, expected_codec="int4")
    assert hooks(client) == {}
