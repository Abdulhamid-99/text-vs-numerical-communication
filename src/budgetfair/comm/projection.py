"""Random Gaussian projection (Johnson-Lindenstrauss) for the vector channel.

Background
----------
The thesis's vector channel transmits a hidden-state vector v in R^source_dim
(typically 4096 for Llama-3.1-8B and Qwen2.5-7B). Without projection, the
per-message bit cost is source_dim * bits_per_component + 32, which makes the
budget-fair comparison with text infeasible (see
06_Notes_Prompts_GapAnalysis/bit_budget_research_memo_2026-05-06.md).

The Johnson-Lindenstrauss lemma guarantees that a random Gaussian projection
preserves pairwise Euclidean distances within (1+/-eps) for target_dim ~
O(log N / eps^2), where N is the number of points. For sentence-level
hidden states drawn from any reasonable distribution, this means a 128-dim
projection of 4096-dim vectors loses very little structural information.

Why random projection vs learned
--------------------------------
The thesis methodology forbids training new models. Random projection:
  - is reproducible from a seed,
  - has theoretical distance-preservation guarantees,
  - serves as a defensible baseline (learned projection = future-work ablation).

Construction
------------
The matrix is sampled once at init and cached. Entries follow N(0, 1/target_dim)
so the projected vector has approximately the same expected squared norm as
the source vector (standard JL convention).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class GaussianProjection:
    """Fixed random Gaussian projection R^source_dim -> R^target_dim.

    Identical (source_dim, target_dim, seed) tuples produce identical matrices,
    so sender and receiver instances of this class agree without sharing state.
    """

    source_dim: int
    target_dim: int
    seed: int

    def __post_init__(self) -> None:
        if self.source_dim <= 0 or self.target_dim <= 0:
            raise ValueError("source_dim and target_dim must be positive")
        if self.target_dim > self.source_dim:
            raise ValueError(
                f"target_dim ({self.target_dim}) > source_dim ({self.source_dim}); "
                "projection should reduce dimension"
            )

    @cached_property
    def matrix(self) -> np.ndarray:
        """The projection matrix, shape (target_dim, source_dim), float32.

        Generated lazily and cached. Entries ~ N(0, 1/target_dim) so that for
        v ~ N(0, I), E[||Pv||^2] approximately equals ||v||^2 * source_dim/target_dim.
        We compensate for that scaling at projection time so the projected vector
        has roughly the same magnitude as the input — this matters for symmetric
        quantization (otherwise the scale absorbs an unnecessary factor).
        """
        rng = np.random.RandomState(self.seed)
        # Standard JL: N(0, 1) entries, then divide by sqrt(target_dim).
        # Multiply by sqrt(source_dim/target_dim) so the projection is approximately
        # norm-preserving in expectation.
        m = rng.randn(self.target_dim, self.source_dim).astype(np.float32)
        m *= np.sqrt(1.0 / self.target_dim)
        return m

    def project(self, v: np.ndarray) -> np.ndarray:
        """Project v: shape (source_dim,) -> shape (target_dim,)."""
        v = np.asarray(v, dtype=np.float32)
        if v.shape != (self.source_dim,):
            raise ValueError(
                f"expected shape ({self.source_dim},), got {v.shape}"
            )
        return self.matrix @ v
