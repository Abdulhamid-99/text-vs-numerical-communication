"""Statistical tests for the hypotheses (H1–H4) per the thesis methodology:

  - Paired t-test or Wilcoxon signed-rank for text vs vector at fixed (model, task, budget, network).
  - Two-way ANOVA on Channel x Budget within each (model, task, network).
  - Cohen's d for primary effect-size estimates.
  - Holm-Bonferroni correction across the family of tests at p<0.05.
"""
from __future__ import annotations


def paired_test(text_scores: list[float], vector_scores: list[float]) -> dict:
    """Returns {t, p, df, cohens_d, normality_ok, test_used}.
    Auto-selects t-test if Shapiro-Wilk p > 0.05 on the differences, else Wilcoxon.
    """
    raise NotImplementedError


def two_way_anova(df) -> dict:
    """Two-way ANOVA Channel x Budget. df is a long-format DataFrame with columns
    (model, task, channel, budget, network, episode_id, score)."""
    raise NotImplementedError


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Returns reject/keep decisions in original order."""
    raise NotImplementedError
