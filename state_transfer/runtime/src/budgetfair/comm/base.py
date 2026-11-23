"""Channel interface. A Channel encodes a sender's payload, computes its bit cost,
optionally passes it through the network simulator, and decodes for the receiver.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TransmittedMessage:
    """One message on the wire. `payload` is the encoded form (bytes for text, np.ndarray for vector).
    `bit_cost` is the size that counts against the budget (per the thesis cost formulas).
    """
    payload: Any
    bit_cost: int
    sender_id: str
    receiver_id: str
    timestamp: float
    metadata: dict


class Channel(ABC):
    name: str

    @abstractmethod
    def encode(self, content: Any, sender_id: str, receiver_id: str) -> TransmittedMessage:
        """Convert agent output to a transmittable form and compute bit cost."""

    @abstractmethod
    def decode(self, msg: TransmittedMessage) -> Any:
        """Convert the transmittable form back to something the receiver can consume
        (string for text channel, hidden-state injection or KV-cache splice for vector)."""
