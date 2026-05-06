"""Vector channel: a fixed-dimension hidden state extracted from a chosen transformer layer,
optionally projected to a lower dimension via a fixed random Gaussian projection,
symmetrically quantized to b bits per component, and transmitted. Receiver-side
injection (how the receiver model consumes the dequantized vector) is left to the
model integration layer; this module owns just the wire format.

Cost formula (from thesis), per message:
    C_vector_payload = d_t * b
    C_vector_metadata = 32 bits (fp32 scale; required to dequantize)
    C_vector = C_vector_payload + C_vector_metadata
              = d_t * b + 32

where d_t is the *transmitted* dimension (after projection if any).

Examples (Llama-3.1-8B and Qwen2.5-7B both have hidden_size = 4096):
    No projection, b=8:    4096 * 8 + 32 = 32,800 bits/message  (alternative ablation)
    Project 128, b=4:       128 *  4 + 32 =     544 bits/message  (PRIMARY config)
    Project 256, b=4:       256 *  4 + 32 =   1,056 bits/message  (mid ablation)
    Project 512, b=4:       512 *  4 + 32 =   2,080 bits/message  (rich ablation)

Quantization is symmetric per-tensor linear:
    scale = max(|v_min|, |v_max|) / (2^(b-1) - 1)   # so values in [-127..127] for b=8
    q     = clip(round(v / scale), -2^(b-1) + 1, 2^(b-1) - 1).astype(int_b)

Design references: C2C [ref:17] uses a learnable projection at 128 per head;
MLT [ref:9] projects 4096 -> 512; Q-KVComm uses adaptive 4-8 bit quantization.
The d=128 / b=4 primary setting is closest to C2C's effective per-head dim.
See ../../../06_Notes_Prompts_GapAnalysis/bit_budget_research_memo_2026-05-06.md
for the literature review and design rationale.
"""
from __future__ import annotations

import struct
import time
from typing import Optional

import numpy as np

from .base import Channel, TransmittedMessage
from .projection import GaussianProjection


class VectorChannel(Channel):
    name = "vector"

    def __init__(
        self,
        dim: int,
        bits_per_component: int = 8,
        layer_index: int = 18,
        projection: Optional[GaussianProjection] = None,
    ) -> None:
        """`dim` is the *transmitted* dimension — the quantization happens at this
        size. If `projection` is provided, the encoder projects the input
        (shape (projection.source_dim,)) down to `dim` first; the decoder
        returns the projected (low-dim) vector.

        If `projection` is None, the encoder expects input already at shape (dim,).
        """
        if bits_per_component not in (8, 4):
            raise ValueError(
                f"bits_per_component must be 8 or 4, got {bits_per_component}. "
                "Other widths require non-trivial bit-packing not yet implemented."
            )
        if projection is not None and projection.target_dim != dim:
            raise ValueError(
                f"projection.target_dim ({projection.target_dim}) must equal "
                f"channel dim ({dim})"
            )
        self.dim = dim
        self.bits_per_component = bits_per_component
        self.layer_index = layer_index
        self.projection = projection

    @property
    def _qmax(self) -> int:
        """Maximum representable quantized magnitude. Symmetric: [-qmax, qmax]."""
        return 2 ** (self.bits_per_component - 1) - 1  # 127 for 8-bit; 7 for 4-bit

    def bit_cost_for_message(self) -> int:
        """Bits transmitted per single vector message under this channel's settings."""
        return self.dim * self.bits_per_component + 32  # +32 bits for the fp32 scale

    def encode(
        self,
        content: np.ndarray,
        sender_id: str,
        receiver_id: str,
    ) -> TransmittedMessage:
        """Quantize a hidden-state vector to int8/int4 + scale, package as bytes.

        Input shape:
          - If `projection` is set: (projection.source_dim,)  -- typically (4096,)
          - If `projection` is None: (self.dim,)              -- transmitted directly

        Output payload:
            quantized int8/int4 (size depends on bits) || fp32 scale (4 bytes)
        bit_cost equals `bit_cost_for_message()` regardless of source dim.
        """
        v = np.asarray(content, dtype=np.float32)
        if self.projection is not None:
            v = self.projection.project(v)  # shape (target_dim,) == (self.dim,)
        if v.shape != (self.dim,):
            raise ValueError(f"expected shape ({self.dim},), got {v.shape}")

        abs_max = float(np.max(np.abs(v))) if v.size else 0.0
        scale = abs_max / self._qmax if abs_max > 0 else 1.0  # dequantize back to zeros if all-zero

        quantized_f = np.round(v / scale)
        np.clip(quantized_f, -self._qmax, self._qmax, out=quantized_f)

        if self.bits_per_component == 8:
            payload_int = quantized_f.astype(np.int8)
            payload_bytes = payload_int.tobytes()
        else:  # 4-bit: pack two values per byte (still symmetric, range [-7..7])
            qi = quantized_f.astype(np.int8)
            # nibble-pack: low nibble = qi[2k], high nibble = qi[2k+1]
            if self.dim % 2 != 0:
                raise ValueError("4-bit packing requires even dim")
            low = qi[0::2] & 0x0F
            high = (qi[1::2] & 0x0F) << 4
            payload_bytes = (low | high).astype(np.uint8).tobytes()

        # Pack scale as little-endian fp32 (4 bytes = 32 bits)
        scale_bytes = struct.pack("<f", scale)

        return TransmittedMessage(
            payload=payload_bytes + scale_bytes,
            bit_cost=self.bit_cost_for_message(),
            sender_id=sender_id,
            receiver_id=receiver_id,
            timestamp=time.time(),
            metadata={
                "channel": self.name,
                "dim": self.dim,
                "bits_per_component": self.bits_per_component,
                "layer_index": self.layer_index,
                "projection_source_dim": (
                    self.projection.source_dim if self.projection is not None else None
                ),
                "projection_seed": (
                    self.projection.seed if self.projection is not None else None
                ),
            },
        )

    def decode(self, msg: TransmittedMessage) -> np.ndarray:
        """Bytes -> int8/int4 -> fp32 vector. Used by the receiver to feed the model."""
        if not isinstance(msg.payload, (bytes, bytearray)):
            raise TypeError(f"VectorChannel.decode expected bytes, got {type(msg.payload).__name__}")
        buf = bytes(msg.payload)

        # Last 4 bytes = fp32 scale
        if len(buf) < 4:
            raise ValueError(f"payload too short ({len(buf)} bytes) to contain fp32 scale")
        scale = struct.unpack("<f", buf[-4:])[0]
        body = buf[:-4]

        if self.bits_per_component == 8:
            if len(body) != self.dim:
                raise ValueError(f"expected {self.dim} body bytes for 8-bit, got {len(body)}")
            qi = np.frombuffer(body, dtype=np.int8).astype(np.float32)
        else:  # 4-bit unpack
            expected = self.dim // 2
            if len(body) != expected:
                raise ValueError(f"expected {expected} body bytes for 4-bit, got {len(body)}")
            packed = np.frombuffer(body, dtype=np.uint8)
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            # Sign-extend nibbles ([-8..7]); since we clip to [-7..7] in encode, no -8 should appear.
            low = np.where(low > 7, low.astype(np.int16) - 16, low.astype(np.int16))
            high = np.where(high > 7, high.astype(np.int16) - 16, high.astype(np.int16))
            qi = np.empty(self.dim, dtype=np.int16)
            qi[0::2] = low
            qi[1::2] = high
            qi = qi.astype(np.float32)

        return qi * scale

    # Backwards-compat shim (the test scaffold called bit_cost() with no arg).
    def bit_cost(self) -> int:  # type: ignore[override]
        return self.bit_cost_for_message()
