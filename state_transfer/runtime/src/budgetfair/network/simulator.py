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
import math
import random
from dataclasses import dataclass

from ..comm.base import TransmittedMessage


@dataclass
class NetworkProfile:
    name: str
    bandwidth_bps: int | None = None  # None = unconstrained
    latency_ms: float = 0.0
    packet_loss: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (("latency_ms", self.latency_ms), ("packet_loss", self.packet_loss)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if not 0 <= self.packet_loss <= 1:
            raise ValueError("packet_loss must be in [0, 1]")
        if self.bandwidth_bps is not None and (
            isinstance(self.bandwidth_bps, bool) or not isinstance(self.bandwidth_bps, (int, float))
            or not math.isfinite(self.bandwidth_bps) or self.bandwidth_bps <= 0
        ):
            raise ValueError("bandwidth_bps must be finite positive, or None for unlimited")


class NetworkSimulator:
    def __init__(self, profile: NetworkProfile, rng_seed: int = 0) -> None:
        profile.__post_init__()
        self.profile = profile
        self.rng = random.Random(rng_seed)
        self.dropped_count = 0
        self.delivered_count = 0
        self.last_delay_ms = 0.0

    def delay_ms(self, bits: int) -> float:
        if isinstance(bits, bool) or not isinstance(bits, int) or bits < 0:
            raise ValueError("wire bit cost must be a non-negative integer")
        delay = self.profile.latency_ms
        if self.profile.bandwidth_bps is not None:
            delay += bits / self.profile.bandwidth_bps * 1000.0
        if not math.isfinite(delay):
            raise ValueError("simulated delay overflow")
        return delay

    def transmit(self, msg: TransmittedMessage) -> TransmittedMessage | None:
        """Apply network effects. Returns None if the packet was dropped.

        On success, returns a copy of the message with simulated transmission delay
        merged into its metadata; the runner can sum these for per-episode latency.
        """
        self.last_delay_ms = self.delay_ms(msg.bit_cost)
        if 0.0 < self.profile.packet_loss <= 1.0:
            if self.rng.random() < self.profile.packet_loss:
                self.dropped_count += 1
                return None

        # Compute simulated delay: latency + (bits / bandwidth) if bandwidth is set.
        delay_ms = self.last_delay_ms

        delivered = dataclasses.replace(
            msg,
            metadata={**msg.metadata, "sim_delay_ms": delay_ms, "network_profile": self.profile.name},
        )
        self.delivered_count += 1
        return delivered
