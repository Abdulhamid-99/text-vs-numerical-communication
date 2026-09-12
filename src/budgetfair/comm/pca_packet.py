"""Numeric PCA message packets with a charged frozen-basis fingerprint.

The shared asset is learned from calibration hidden states, not from held-out
answers. Decoding is solely a linear reconstruction; there is no token lookup,
nearest-token dictionary, text fallback or access to private evaluation values.
Asset transfer is separately reported setup cost; every message charges its
36-byte frame and the complete underlying tensor packet.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from numbers import Integral
from pathlib import Path
import struct
import zipfile

import numpy as np

from .tensor_packet import encode_tensor, decode_packet

FRAME = struct.Struct("<4s32s")
MAGIC = b"PCA1"
MAX_ASSET_BYTES = 128 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _matrix(array):
    _require(isinstance(array, np.ndarray) and array.dtype == np.float32 and array.ndim == 2,
             "Expected a float32 matrix")
    _require(min(array.shape) > 0 and np.isfinite(array).all(), "Empty or nonfinite matrix")
    return array


def _sha(array):
    return hashlib.sha256(array.astype("<f4", copy=False).tobytes()).hexdigest()


@dataclass(frozen=True)
class PCAAsset:
    mean: np.ndarray
    components: np.ndarray
    layer: int
    fingerprint: str
    setup_bytes: int


def load_asset(path: Path | str, *, expected_sha256: str) -> PCAAsset:
    """Load a plan-bound, finite, orthonormal numeric asset with allocation limits."""
    path = Path(path)
    _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64,
             "A complete frozen asset SHA256 is required")
    _require(0 < path.stat().st_size <= MAX_ASSET_BYTES, "PCA asset file exceeds limit")
    raw = path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    _require(fingerprint == expected_sha256, "PCA asset checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        _require(len(entries) == 4 and {item.filename for item in entries} ==
                 {"mean.npy", "components.npy", "layer.npy", "schema.npy"}, "Unknown PCA asset members")
        _require(sum(item.file_size for item in entries) <= MAX_ASSET_BYTES, "Expanded PCA asset exceeds limit")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        mean, components = archive["mean"].copy(), archive["components"].copy()
        layer, schema = archive["layer"], archive["schema"]
    _matrix(components)
    _require(components.shape[0] <= 2048 and components.shape[1] <= 8192,
             "PCA dimensions exceed supported limits")
    _require(mean.dtype == np.float32 and mean.shape == (components.shape[1],) and np.isfinite(mean).all(),
             "Invalid PCA mean")
    _require(layer.shape == () and layer.dtype == np.int32 and int(layer) >= 0, "Invalid PCA layer")
    _require(schema.shape == () and schema.dtype == np.int32 and int(schema) == 1, "Unknown PCA schema")
    _require(np.allclose(components.astype(np.float64) @ components.astype(np.float64).T,
                        np.eye(components.shape[0]), atol=2e-6, rtol=2e-6), "Nonorthogonal PCA basis")
    mean.setflags(write=False)
    components.setflags(write=False)
    return PCAAsset(mean, components, int(layer), fingerprint, len(raw))


def project(states: np.ndarray, asset: PCAAsset, dimensions: int) -> np.ndarray:
    _matrix(states)
    _require(not isinstance(dimensions, bool) and isinstance(dimensions, Integral)
             and 1 <= dimensions <= asset.components.shape[0], "Unsupported PCA dimension")
    _require(states.shape[1] == asset.mean.shape[0], "Source width differs from basis")
    with np.errstate(over="ignore", invalid="ignore"):
        values = ((states.astype(np.float64) - asset.mean.astype(np.float64)) @
                  asset.components[:dimensions].astype(np.float64).T).astype(np.float32)
    return _matrix(values)


def encode_coefficients(coefficients: np.ndarray, asset: PCAAsset, *, codec: str) -> bytes:
    _matrix(coefficients)
    _require(1 <= coefficients.shape[0] <= 128 and coefficients.shape[1] <= asset.components.shape[0],
             "Compressed shape exceeds calibrated protocol")
    inner = encode_tensor(coefficients, codec=codec, role="generic", layer_index=asset.layer)
    return FRAME.pack(MAGIC, bytes.fromhex(asset.fingerprint)) + inner


def encode_projected(states: np.ndarray, asset: PCAAsset, *, dimensions: int, codec: str) -> bytes:
    return encode_coefficients(project(states, asset, dimensions), asset, codec=codec)


def reconstruct(packet: bytes, asset: PCAAsset, *, expected_dimensions: int,
                expected_codec: str) -> tuple[np.ndarray, dict]:
    """Receiver decoder: only serialized coefficients and the fixed public asset."""
    _require(not isinstance(expected_dimensions, bool) and isinstance(expected_dimensions, Integral)
             and 1 <= expected_dimensions <= asset.components.shape[0], "Invalid public projection dimension")
    _require(type(packet) is bytes and len(packet) > FRAME.size, "Missing PCA packet")
    magic, fingerprint = FRAME.unpack_from(packet)
    _require(magic == MAGIC and fingerprint.hex() == asset.fingerprint, "Packet references a different frozen basis")
    decoded = decode_packet(packet[FRAME.size:], max_elements=128 * asset.components.shape[0])
    coefficients = _matrix(decoded.array)
    _require(decoded.role == "generic" and decoded.layer_index == asset.layer, "Wrong PCA role or layer")
    _require(decoded.codec == expected_codec and coefficients.shape[1] == expected_dimensions,
             "Packet codec/dimension differs from the public plan")
    _require(coefficients.shape[0] <= 128, "Private span exceeds maximum")
    with np.errstate(over="ignore", invalid="ignore"):
        hidden = (coefficients.astype(np.float64) @ asset.components[:expected_dimensions].astype(np.float64)
                  + asset.mean.astype(np.float64)).astype(np.float32)
    _matrix(hidden)
    return hidden, {"dimensions": int(expected_dimensions), "private_token_rows": coefficients.shape[0],
                   "codec": decoded.codec, "basis_sha256": asset.fingerprint,
                   "wire_bits": 8 * len(packet), "packet_sha256": hashlib.sha256(packet).hexdigest(),
                   "pca_frame_bytes": FRAME.size, "tensor_packet_bytes": decoded.packet_bytes,
                   "decoded_coefficients_sha256": _sha(coefficients), "reconstructed_hidden_sha256": _sha(hidden),
                   "shared_basis_bytes": asset.setup_bytes}
