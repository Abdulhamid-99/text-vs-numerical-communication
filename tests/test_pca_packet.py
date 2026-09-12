"""Independent numeric fixtures and malformed-input tests for the PCA channel."""
import hashlib
import struct

import numpy as np
import pytest

from budgetfair.comm.pca_packet import load_asset, project, encode_coefficients, encode_projected, reconstruct


def asset_file(tmp_path, **overrides):
    fields = dict(mean=np.array([10, 20, 30, 40], np.float32),
                  components=np.eye(4, dtype=np.float32)[:2],
                  layer=np.array(4, np.int32), schema=np.array(1, np.int32))
    fields.update(overrides)
    path = tmp_path / "basis.npz"
    np.savez(path, **fields)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, sha


def basis(tmp_path, **overrides):
    path, sha = asset_file(tmp_path, **overrides)
    return load_asset(path, expected_sha256=sha)


def test_known_packet_bytes_and_independent_reconstruction(tmp_path):
    asset = basis(tmp_path)
    source = np.array([[11, 18, 900, -900], [10, 20, -77, 88]], np.float32)
    packet = encode_projected(source, asset, dimensions=2, codec="int8")
    body = struct.pack("<ff4b", np.float32(2 / 127), 1.0, 64, -127, 0, 0)
    expected = (b"PCA1" + bytes.fromhex(asset.fingerprint) +
                struct.pack("<4sBBBBiQII", b"BFTP", 1, 1, 0, 2, 4, len(body), 2, 2) + body)
    assert packet == expected
    decoded, meta = reconstruct(packet, asset, expected_dimensions=2, expected_codec="int8")
    # Discarded coordinates reconstruct the public mean, never the source values.
    scale = float(np.float32(2 / 127))
    np.testing.assert_array_equal(decoded, np.array([[10 + 64 * scale, 20 - 127 * scale, 30, 40],
                                                    [10, 20, 30, 40]], np.float32))
    assert meta["wire_bits"] == len(expected) * 8
    assert meta["pca_frame_bytes"] == 36
    assert meta["shared_basis_bytes"] == asset.setup_bytes
    assert meta["reconstructed_hidden_sha256"] == hashlib.sha256(decoded.astype("<f4").tobytes()).hexdigest()


def test_projection_generalizes_to_unseen_low_rank_values_without_token_lookup(tmp_path):
    asset = basis(tmp_path)
    unseen = np.array([[14.25, 19.75, 30, 40], [-10.5, 100.25, 30, 40]], np.float32)
    packet = encode_projected(unseen, asset, dimensions=2, codec="float32")
    decoded, _ = reconstruct(packet, asset, expected_dimensions=2, expected_codec="float32")
    np.testing.assert_array_equal(decoded, unseen)


def test_zero_coefficients_deliver_shared_mean_not_private_states(tmp_path):
    asset = basis(tmp_path)
    packet = encode_coefficients(np.zeros((5, 2), np.float32), asset, codec="int4")
    decoded, _ = reconstruct(packet, asset, expected_dimensions=2, expected_codec="int4")
    np.testing.assert_array_equal(decoded, np.tile(asset.mean, (5, 1)))


@pytest.mark.parametrize("override", [
    {"mean": np.array([np.nan, 20, 30, 40], np.float32)},
    {"mean": np.zeros(3, np.float32)},
    {"mean": np.zeros(4, np.float64)},
    {"components": np.ones((2, 4), np.float32)},
    {"components": np.eye(4, dtype=np.float64)[:2]},
    {"layer": np.array(-1, np.int32)},
    {"layer": np.array(True)},
    {"schema": np.array(2, np.int32)},
    {"schema": np.array(1, np.int64)},
    {"unexpected": np.zeros(1)},
])
def test_invalid_public_assets_rejected(tmp_path, override):
    with pytest.raises(ValueError):
        basis(tmp_path, **override)


def test_asset_must_match_frozen_plan_and_is_read_only(tmp_path):
    path, sha = asset_file(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        load_asset(path, expected_sha256="0" * 64)
    asset = load_asset(path, expected_sha256=sha)
    with pytest.raises(ValueError):
        asset.mean[0] = 99
    with pytest.raises(ValueError):
        asset.components[0, 0] = 99


@pytest.mark.parametrize("mutation", ["truncated", "trailing", "bad_magic", "other_basis", "wrong_layer", "wrong_role"])
def test_bad_wire_cannot_be_reconstructed(tmp_path, mutation):
    asset = basis(tmp_path)
    packet = bytearray(encode_coefficients(np.ones((2, 2), np.float32), asset, codec="int4"))
    if mutation == "truncated": packet = packet[:-1]
    elif mutation == "trailing": packet += b"\0"
    elif mutation == "bad_magic": packet[0] ^= 1
    elif mutation == "other_basis": packet[4] ^= 1
    elif mutation == "wrong_layer": struct.pack_into("<i", packet, 36 + 8, 3)
    elif mutation == "wrong_role": packet[36 + 6] = 2
    with pytest.raises(ValueError):
        reconstruct(bytes(packet), asset, expected_dimensions=2, expected_codec="int4")


@pytest.mark.parametrize("dimensions,codec", [(1, "int4"), (True, "int4"), (0, "int4"), (3, "int4"), (2, "int8")])
def test_receiver_enforces_public_rank_and_codec(tmp_path, dimensions, codec):
    asset = basis(tmp_path)
    packet = encode_coefficients(np.ones((2, 2), np.float32), asset, codec="int4")
    with pytest.raises(ValueError):
        reconstruct(packet, asset, expected_dimensions=dimensions, expected_codec=codec)


def test_finite_input_that_overflows_projection_is_explicit_failure(tmp_path):
    maximum = np.finfo(np.float32).max
    asset = basis(tmp_path, mean=np.full(4, -maximum, np.float32))
    with pytest.raises(ValueError, match="nonfinite"):
        project(np.full((2, 4), maximum, np.float32), asset, 2)


def test_finite_packet_that_overflows_reconstruction_is_explicit_failure(tmp_path):
    maximum = np.finfo(np.float32).max
    asset = basis(tmp_path, mean=np.full(4, maximum, np.float32))
    packet = encode_coefficients(np.full((1, 2), maximum, np.float32), asset, codec="float32")
    with pytest.raises(ValueError, match="nonfinite"):
        reconstruct(packet, asset, expected_dimensions=2, expected_codec="float32")


def test_receiver_rejects_original_array_instead_of_wire_bytes(tmp_path):
    asset = basis(tmp_path)
    with pytest.raises(ValueError):
        reconstruct(np.ones((2, 2), np.float32), asset, expected_dimensions=2, expected_codec="float32")
