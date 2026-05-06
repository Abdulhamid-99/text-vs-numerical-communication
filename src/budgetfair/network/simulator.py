"""NetworkSimulator applies (bandwidth, latency, packet_loss) to TransmittedMessages.

Three named conditions used in the experiment matrix:
  - clean:    packet_loss = 0.0
  - lossy_5:  packet_loss = 0.05
  - lossy_10: packet_loss = 0.10

Bandwidth and latency are recorded but not actually slept on — the runner totals
the per-message simulated delay separately. We don't want to wait real seconds for
1000-episode runs.
"""
from __future__ import annotations

import dataclasses
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
        self.dropped_count = 0
        self.delivered_count = 0

    def transmit(self, msg: TransmittedMessage) -> TransmittedMessage | None:
        """Apply network effects. Returns None if the packet was dropped.

        On success, returns a copy of the message with simulated transmission delay
        merged into its metadata; the runner can sum these for per-episode latency.
        """
        if 0.0 < self.profile.packet_loss <= 1.0:
            if self.rng.random() < self.profile.packet_loss:
                self.dropped_count += 1
                return None

        # Compute simulated delay: latency + (bits / bandwidth) if bandwidth is set.
        delay_ms = self.profile.latency_ms
        if self.profile.bandwidth_bps and self.profile.bandwidth_bps > 0:
            delay_ms += (msg.bit_cost / self.profile.bandwidth_bps) * 1000.0

        delivered = dataclasses.replace(
            msg,
            metadata={**msg.metadata, "sim_delay_ms": delay_ms, "network_profile": self.profile.name},
        )
        self.delivered_count += 1
        return delivered
