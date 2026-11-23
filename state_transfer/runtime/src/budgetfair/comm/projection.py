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
O(log N / eps^2), where N is the number of points and failure probability also
enters the dimension bound. A fixed 128-dimensional projection does not by
itself guarantee semantic preservation, invertibility, or useful communication.

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
from numbers import Integral

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
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or not 0 <= self.seed < 2**32:
            raise ValueError("projection seed must be an integer in [0, 2**32)")
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               for value in (self.source_dim, self.target_dim)):
            raise ValueError("source_dim and target_dim must be integers")
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

        Generated lazily from its own RNG. Entries have variance 1/target_dim;
        for any fixed v independent of the matrix, E[||Pv||^2] = ||v||^2.
        """
        rng = np.random.RandomState(self.seed)
        # Standard JL: N(0, 1) entries, then divide by sqrt(target_dim).
        m = rng.randn(self.target_dim, self.source_dim).astype(np.float32)
        m *= np.sqrt(1.0 / self.target_dim)
        return m

    def project(self, v: np.ndarray) -> np.ndarray:
        """Project v: shape (source_dim,) -> shape (target_dim,)."""
        v = self._finite_array(v)
        if v.shape != (self.source_dim,):
            raise ValueError(
                f"expected shape ({self.source_dim},), got {v.shape}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            projected = self.matrix @ v
        if not np.isfinite(projected).all():
            raise ValueError("projection produced nonfinite components")
        return projected

    def lift(self, w: np.ndarray) -> np.ndarray:
        """Lift w: shape (target_dim,) -> shape (source_dim,) via M^T.

        This implementation retains the declared target_dim/source_dim amplitude
        convention. E[M^T M] = I_source, so the resulting expectation is
        (target_dim/source_dim)*v, not an unbiased inverse of v. A rank-reducing
        projection cannot reconstruct every source vector. Residual injection
        separately normalizes this lifted direction.
        """
        w = self._finite_array(w)
        if w.shape != (self.target_dim,):
            raise ValueError(
                f"expected shape ({self.target_dim},), got {w.shape}"
            )
        scale = float(self.target_dim) / float(self.source_dim)
        with np.errstate(over="ignore", invalid="ignore"):
            lifted = scale * (self.matrix.T @ w)
        if not np.isfinite(lifted).all():
            raise ValueError("lift produced nonfinite components")
        return lifted

    @staticmethod
    def _finite_array(value: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(value):
            raise ValueError("projection components must be real")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("projection components must be finite float32 values")
        return array
