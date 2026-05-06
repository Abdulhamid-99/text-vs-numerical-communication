"""NetworkSimulator applies (bandwidth, latency, packet_loss) to TransmittedMessages.

Three named conditions used in the experiment matrix:
  - clean:    packet_loss = 0.0
  - lossy_5:  packet_loss = 0.05
  - lossy_10: packet_loss = 0.10

Bandwidth and latency are reserved for future RQ4 work; defaults are unconstrained.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..comm.base import TransmittedMessage


@dataclass
class NetworkProfile:
    name: str
    bandwidth_bps: int | None = None  # None = unconstrained
    latency_ms: float = 0.0
    packet_loss: float = 0.0


class NetworkSimulator:
    def __init__(self, profile: NetworkProfile, rng_seed: int = 0) -> None:
        self.profile = profile
        self.rng = random.Random(rng_seed)

    def transmit(self, msg: TransmittedMessage) -> TransmittedMessage | None:
        """Apply network effects. Returns None if the packet was dropped."""
        raise NotImplementedError
