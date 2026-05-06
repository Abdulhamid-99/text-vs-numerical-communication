"""Network simulator correctness. Loss rates must match the configured probabilities
to within sampling noise — otherwise the robustness sweep means nothing.
"""
from budgetfair.comm.base import TransmittedMessage
from budgetfair.network.simulator import NetworkProfile, NetworkSimulator


def _msg() -> TransmittedMessage:
    return TransmittedMessage(
        payload=b"x",
        bit_cost=8,
        sender_id="a",
        receiver_id="b",
        timestamp=0.0,
        metadata={},
    )


def test_clean_network_passes_through():
    sim = NetworkSimulator(NetworkProfile(name="clean", packet_loss=0.0), rng_seed=0)
    out = sim.transmit(_msg())
    assert out is not None
    assert out.metadata["network_profile"] == "clean"
    assert sim.dropped_count == 0
    assert sim.delivered_count == 1


def test_full_loss_drops_everything():
    sim = NetworkSimulator(NetworkProfile(name="lossy", packet_loss=1.0), rng_seed=0)
    assert sim.transmit(_msg()) is None
    assert sim.dropped_count == 1


def test_5pct_loss_within_expected_band():
    """Run 1000 trials at 5% loss; binomial std is sqrt(1000 * 0.05 * 0.95) ≈ 6.9.
    We expect ~50 drops; allow a generous ±20 (≈3 sigma) to keep this from flaking."""
    sim = NetworkSimulator(NetworkProfile(name="lossy_5", packet_loss=0.05), rng_seed=42)
    for _ in range(1000):
        sim.transmit(_msg())
    assert 30 <= sim.dropped_count <= 70, f"got {sim.dropped_count} drops"


def test_bandwidth_adds_serialization_delay():
    """At 1000 bps, an 8-bit message takes 8 ms to serialize. Plus configured latency."""
    sim = NetworkSimulator(
        NetworkProfile(name="rate_limited", bandwidth_bps=1000, latency_ms=2.0),
        rng_seed=0,
    )
    out = sim.transmit(_msg())
    assert out is not None
    # 2 ms latency + (8 bits / 1000 bps) * 1000 ms = 2 + 8 = 10 ms
    assert abs(out.metadata["sim_delay_ms"] - 10.0) < 1e-6
