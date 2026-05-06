"""Per-cell aggregation: mean / std / 95% CI on P, C, T, P_loss for each cell of the matrix."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CellSummary:
    model: str
    task: str
    channel: str
    budget: str
    network: str
    n_episodes: int
    success_rate: float
    mean_score: float
    mean_bits: float
    mean_latency_ms: float

    # 95% bootstrap CIs
    score_ci_low: float
    score_ci_high: float


def aggregate_cell(results: list) -> CellSummary:
    raise NotImplementedError
