"""Self-describing numeric tensor packets; no text/token-ID side channel.

Wire v1 is little-endian: ``<4sBBBBiQ`` (20 bytes), then ``rank`` uint32
dimensions, then body. Header fields: magic BFTP, version, codec (0=f32,
1=int8, 2=int4), role (0=generic, 1=token_embeddings, 2=hidden_states), rank,
decoder layer (-1=not applicable), body byte length. Rank is 2..8.

Float32 body is the exact C-order IEEE-754 float32 bit representation.
Quantized body starts with one positive float32 scale per last-axis row,
followed by signed symmetric codes. Int4 packs low nibble first, with a zero
high padding nibble for each odd-width row. Reserved -128/-8 codes are invalid.
Quantization is round-to-nearest-even and is lossy; it makes no semantic claim.

Every byte, including header, dimensions, scales and padding, is charged:
``bit_cost = 8 * len(packet)``. No framing overhead is silently omitted from
this application packet. Network-level headers, if used, are a caller concern.
Integrity/authenticity is also external: structurally valid bit changes need
an independently recorded packet hash to be detected.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import struct

import numpy as np

MAGIC = b"BFTP"
VERSION = 1
HEADER = struct.Struct("<4sBBBBiQ")
MAX_RANK = 8
MAX_ELEMENTS = 16_777_216
MAX_PACKET_BYTES = 128 * 1024 * 1024
CODECS = {"float32": 0, "int8": 1, "int4": 2}
ROLES = {"generic": 0, "token_embeddings": 1, "hidden_states": 2}
_CODEC_NAMES = {value: name for name, value in CODECS.items()}
_ROLE_NAMES = {value: name for name, value in ROLES.items()}
_FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class DecodedTensorPacket:
    """Decoded owned array and explicit wire metadata; not a sender reference."""

    array: np.ndarray
    codec: str
    role: str
    layer_index: int | None
    packet_bytes: int
    version: int = VERSION

    @property
    def bit_cost(self) -> int:
        return 8 * self.packet_bytes


def _limit(value, name: str, ceiling: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= ceiling:
        raise ValueError(f"{name} must be an integer in 1..{ceiling}")
    return int(value)


def _shape(shape: tuple[int, ...], max_elements: int) -> int:
    if not 2 <= len(shape) <= MAX_RANK:
        raise ValueError(f"tensor rank must be in 2..{MAX_RANK}")
    elements = 1
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, Integral) or not 1 <= dimension <= 0xFFFFFFFF:
            raise ValueError("tensor dimensions must be positive uint32 integers")
        # Validate before multiplication/allocation, independent of integer width.
        if dimension > max_elements // elements:
            raise ValueError("tensor element count exceeds allocation limit")
        elements *= int(dimension)
    return elements


def _layer(role: str, layer_index: int | None) -> int:
    if layer_index is None:
        if role == "hidden_states":
            raise ValueError("hidden_states requires a nonnegative decoder layer_index")
        return -1
    if isinstance(layer_index, bool) or not isinstance(layer_index, Integral) or not 0 <= layer_index <= 0x7FFFFFFF:
        raise ValueError("layer_index must be None or a nonnegative int32")
    if role == "token_embeddings":
        raise ValueError("token_embeddings has no decoder layer_index")
    return int(layer_index)


def _body_size(codec: str, shape: tuple[int, ...], elements: int) -> int:
    if codec == "float32":
        return 4 * elements
    rows, width = elements // shape[-1], shape[-1]
    return rows * 4 + (elements if codec == "int8" else rows * ((width + 1) // 2))


def encode_tensor(array: np.ndarray, *, codec: str = "float32", role: str = "generic",
                  layer_index: int | None = None) -> bytes:
    """Encode only a finite float32 ndarray (rank >=2), returning wire bytes.

    Noncontiguous and big-endian float32 arrays are supported. Other dtypes
    are rejected instead of silently narrowing or accepting text/token IDs.
    Callers must explicitly detach/copy/cast framework tensors before encoding.
    """
    if not isinstance(array, np.ndarray) or array.dtype.kind != "f" or array.dtype.itemsize != 4:
        raise TypeError("array must be a float32 numpy ndarray")
    if not isinstance(codec, str) or codec not in CODECS:
        raise ValueError("codec must be float32, int8, or int4")
    if not isinstance(role, str) or role not in ROLES:
        raise ValueError("unknown tensor role")
    layer = _layer(role, layer_index)
    shape = tuple(array.shape)
    elements = _shape(shape, MAX_ELEMENTS)
    body_size = _body_size(codec, shape, elements)
    if HEADER.size + 4 * len(shape) + body_size > MAX_PACKET_BYTES:
        raise ValueError("packet exceeds byte limit")
    if not np.isfinite(array).all():
        raise ValueError("tensor components must be finite")
    if codec == "float32":
        body = array.astype("<f4", copy=False).tobytes(order="C")
    else:
        rows = np.asarray(array, dtype=np.float32).reshape(-1, shape[-1]).astype(np.float64)
        qmax = 127 if codec == "int8" else 7
        maxima = np.max(np.abs(rows), axis=1)
        minimum = float(np.nextafter(np.float32(0), np.float32(1)))
        scales = np.where(maxima == 0, 1.0, np.maximum(maxima / qmax, minimum)).astype(np.float32)
        too_large = scales.astype(np.float64) * qmax > _FLOAT32_MAX
        scales[too_large] = np.nextafter(scales[too_large], np.float32(0))
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("quantization scale is not positive finite float32")
        quantized = np.clip(np.rint(rows / scales.astype(np.float64)[:, None]), -qmax, qmax).astype(np.int8)
        if codec == "int8":
            codes = quantized.tobytes(order="C")
        else:
            packed = (quantized[:, 0::2] & 15).astype(np.uint8)
            high = (quantized[:, 1::2] & 15).astype(np.uint8)
            packed[:, :high.shape[1]] |= high << 4
            codes = packed.tobytes(order="C")
        body = scales.astype("<f4", copy=False).tobytes() + codes
    if len(body) != body_size:
        raise RuntimeError("internal tensor packet length mismatch")
    header = HEADER.pack(MAGIC, VERSION, CODECS[codec], ROLES[role], len(shape), layer, len(body))
    return header + struct.pack("<" + "I" * len(shape), *shape) + body


def decode_packet(packet: bytes, *, max_elements: int = MAX_ELEMENTS,
                  max_packet_bytes: int = MAX_PACKET_BYTES) -> DecodedTensorPacket:
    """Validate before allocation; decode bytes only, never an original tensor.

    Limits may be lowered by the receiver, but cannot exceed the hard ceilings.
    Empty dimensions, malformed metadata, trailing/truncated bytes, reserved
    codes, nonfinite scales/data and reconstructed float32 overflow all fail.
    """
    if type(packet) is not bytes:
        raise TypeError("packet must be bytes")
    max_elements = _limit(max_elements, "max_elements", MAX_ELEMENTS)
    max_packet_bytes = _limit(max_packet_bytes, "max_packet_bytes", MAX_PACKET_BYTES)
    if len(packet) > max_packet_bytes:
        raise ValueError("packet exceeds byte limit")
    if len(packet) < HEADER.size:
        raise ValueError("truncated tensor header")
    magic, version, codec_id, role_id, rank, wire_layer, body_length = HEADER.unpack_from(packet)
    if magic != MAGIC or version != VERSION:
        raise ValueError("unsupported tensor packet magic/version")
    if codec_id not in _CODEC_NAMES or role_id not in _ROLE_NAMES:
        raise ValueError("unsupported tensor codec/role")
    if not 2 <= rank <= MAX_RANK:
        raise ValueError("invalid tensor rank")
    codec, role = _CODEC_NAMES[codec_id], _ROLE_NAMES[role_id]
    layer = None if wire_layer == -1 else wire_layer
    _layer(role, layer)
    offset = HEADER.size + 4 * rank
    if len(packet) < offset:
        raise ValueError("truncated tensor dimensions")
    shape = struct.unpack_from("<" + "I" * rank, packet, HEADER.size)
    elements = _shape(shape, max_elements)
    expected_body = _body_size(codec, shape, elements)
    if body_length != expected_body or len(packet) != offset + expected_body:
        raise ValueError("tensor packet body length does not match shape/codec/exact wire length")
    if codec == "float32":
        result = np.frombuffer(packet, dtype="<f4", count=elements, offset=offset)
        if not np.isfinite(result).all():
            raise ValueError("float32 packet contains nonfinite components")
        array = result.astype(np.float32, copy=True).reshape(shape)
    else:
        rows, width = elements // shape[-1], shape[-1]
        scales = np.frombuffer(packet, dtype="<f4", count=rows, offset=offset)
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("wire scales must be positive finite float32")
        codes_offset = offset + rows * 4
        if codec == "int8":
            quantized = np.frombuffer(packet, dtype=np.int8, count=elements, offset=codes_offset).reshape(rows, width)
            if np.any(quantized == -128):
                raise ValueError("reserved int8 code -128")
        else:
            packed = np.frombuffer(packet, dtype=np.uint8, offset=codes_offset).reshape(rows, (width + 1) // 2)
            low, high = packed & 15, packed >> 4
            if width % 2 and np.any(high[:, -1] != 0):
                raise ValueError("nonzero int4 row padding")
            if np.any(low == 8) or np.any(high[:, :width // 2] == 8):
                raise ValueError("reserved int4 code -8")
            quantized = np.empty((rows, width), dtype=np.int8)
            quantized[:, 0::2] = np.where(low > 7, low.astype(np.int16) - 16, low)
            used_high = high[:, :width // 2]
            quantized[:, 1::2] = np.where(used_high > 7, used_high.astype(np.int16) - 16, used_high)
        reconstructed = quantized.astype(np.float64) * scales.astype(np.float64)[:, None]
        if not np.isfinite(reconstructed).all() or np.any(np.abs(reconstructed) > _FLOAT32_MAX):
            raise ValueError("dequantized tensor overflows finite float32")
        array = reconstructed.astype(np.float32).reshape(shape)
    return DecodedTensorPacket(array, codec, role, layer, len(packet))


def decode_tensor(packet: bytes, **limits) -> np.ndarray:
    """Convenience tensor-only view of a fully validated numeric packet."""
    return decode_packet(packet, **limits).array
