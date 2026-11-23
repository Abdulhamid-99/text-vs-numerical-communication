"""Independent scalar BFTP decoder used by the held-out reconstruction audit."""

import hashlib
import struct

import numpy as np


class VerificationError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise VerificationError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def f32bytes(array):
    return np.asarray(array, dtype="<f4").tobytes(order="C")


def matrix(value):
    need(
        isinstance(value, np.ndarray)
        and value.dtype == np.float32
        and value.ndim == 2
        and 1 <= value.shape[0] <= 128
        and 1 <= value.shape[1] <= 8192
        and np.isfinite(value).all(),
        "Invalid finite source matrix",
    )
    return value


def decode_packet(raw, codec, layer, role, shape):
    need(type(raw) is bytes and len(raw) >= 28, "Truncated tensor packet")
    magic, version, number, got_role, rank, got_layer, body_size, rows, width = (
        struct.unpack_from("<4sBBBBiQII", raw)
    )
    need(
        (magic, version, number, got_role, rank, got_layer, rows, width)
        == (
            b"BFTP",
            1,
            ("float32", "int8", "int4").index(codec),
            role,
            2,
            layer,
            *shape,
        ),
        "Packet header",
    )
    need(
        1 <= rows <= 128 and 1 <= width <= 8192 and len(raw) == 28 + body_size,
        "Packet shape/size invalid",
    )
    if codec == "float32":
        need(body_size == rows * width * 4, "Float body size")
        result = np.frombuffer(raw, dtype="<f4", offset=28).reshape(rows, width).copy()
    else:
        scale_bytes = rows * 4
        expected_bytes = rows * width if codec == "int8" else rows * ((width + 1) // 2)
        need(body_size == scale_bytes + expected_bytes, "Quantized body size")
        scales = np.frombuffer(raw, dtype="<f4", count=rows, offset=28)
        need(np.isfinite(scales).all() and (scales > 0).all(), "Invalid scales")
        packed = raw[28 + scale_bytes :]
        if codec == "int8":
            codes = np.frombuffer(packed, dtype=np.int8).reshape(rows, width)
            need(not np.any(codes == -128), "Reserved int8 code")
        else:
            codes = np.empty((rows, width), dtype=np.int8)
            for i in range(rows):
                for j in range((width + 1) // 2):
                    byte = packed[i * ((width + 1) // 2) + j]
                    for k, nibble in enumerate((byte & 15, byte >> 4)):
                        if 2 * j + k == width:
                            need(nibble == 0, "Nonzero int4 pad")
                        else:
                            need(nibble != 8, "Reserved int4 code")
                            codes[i, 2 * j + k] = nibble if nibble < 8 else nibble - 16
        result = (codes.astype(np.float64) * scales.astype(np.float64)[:, None]).astype(
            np.float32
        )
    return matrix(result)
