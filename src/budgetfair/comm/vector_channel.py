"""Vector channel: a fixed-dimension hidden state extracted from a chosen transformer layer,
quantized to b bits per component, transmitted to a receiver that injects it via activation
addition / KV-cache splice / adapter.

Cost formula (from thesis):
    C_vector = d * b   (plus < 32 bits metadata)

Design follows KVComm [ref:3] (selective KV sharing, Llama-3.1) and Cache-to-Cache [ref:17]
(direct KV-cache communication, Qwen2.5).
"""
from __future__ import annotations

import numpy as np

from .base import Channel, TransmittedMessage


class VectorChannel(Channel):
    name = "vector"

    def __init__(self, dim: int, bits_per_component: int = 8, layer_index: int = 18) -> None:
        self.dim = dim
        self.bits_per_component = bits_per_component
        self.layer_index = layer_index

    def encode(self, content: np.ndarray, sender_id: str, receiver_id: str) -> TransmittedMessage:
        """Quantize the hidden state. content is shape (dim,) float32; returns int8 bytes."""
        raise NotImplementedError

    def decode(self, msg: TransmittedMessage) -> np.ndarray:
        """Dequantize back to float for receiver-side injection."""
        raise NotImplementedError

    def bit_cost(self) -> int:
        return self.dim * self.bits_per_component
