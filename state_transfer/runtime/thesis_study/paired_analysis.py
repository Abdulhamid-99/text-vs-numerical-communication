"""Independent CPU statistics for family-paired, completed task outcomes.

Does not import the experiment, generate data, load an LLM, or validate producers'
scores. Callers must independently score saved outputs and supply frozen IDs.
"""
from collections.abc import Mapping
from math import comb, isfinite, sqrt
from numbers import Integral, Real
from statistics import NormalDist

import numpy as np
from scipy.stats import beta


def _probability(value, name, *, open_interval=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if not (0 < value < 1 if open_interval else 0 <= value <= 1):
        raise ValueError(f"{name} is outside the probability interval")
    return float(value)


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _ids(values):
    values = list(values)
    if not values or any(not isinstance(x, str) or not x or x != x.strip() for x in values):
        raise ValueError("Expected IDs must be nonempty exact strings")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate family or hypothesis IDs")
    return sorted(values)


def _binary(value):
    if not isinstance(value, (bool, Integral)) or value not in (0, 1):
        raise ValueError("Outcome must be a binary integer/bool; missing and fractional values are not failures")
    return int(value)


def _paired(left, right, expected_family_ids):
    ids = _ids(expected_family_ids)
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError("Outcomes must be mappings keyed by family ID")
    if set(left) != set(ids) or set(right) != set(ids):
        raise ValueError("Missing, extra, or mismatched paired families")
    return ids, np.array([_binary(left[i]) for i in ids]), np.array([_binary(right[i]) for i in ids])


def outcomes_from_rows(rows, expected_family_ids):
    """One already-scored completed row per family in ONE selected condition.

    'completed' describes execution, not task success. Empty observed answers may
    validly score zero. Technical failures/pending rows cannot be silently scored.
    A caller selecting multiple policies into one condition encounters duplicates.
    """
    ids = _ids(expected_family_ids)
    result = {}
    for row in rows:
        identity = row["family_id"]
        if identity in result:
            raise ValueError("Repeated family; variants/replays must not inflate n")
        if row.get("execution_status") != "completed":
            raise ValueError("Technical or incomplete execution: analysis unavailable, not zero")
        if row.get("execution_kind") not in ("actual", "policy_replay"):
            raise ValueError("Explicit actual/policy_replay provenance required")
        if row["execution_kind"] == "policy_replay" and not row.get("source_result_id"):
            raise ValueError("Policy replay must retain its actual source result ID")
        result[identity] = _binary(row["success"])
    if set(result) != set(ids):
        raise ValueError("Frozen expected family set is incomplete or contains extra rows")
    return {i: result[i] for i in ids}


def clopper_pearson(successes, n, confidence=0.95):
    successes, n = _count(successes, "successes"), _count(n, "n")
    if n == 0 or successes > n:
        raise ValueError("Require 0 <= successes <= n and nonempty n")
    alpha = 1 - _probability(confidence, "confidence", open_interval=True)
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n-successes+1))
    upper = 1.0 if successes == n else float(beta.ppf(1-alpha/2, successes+1, n-successes))
    return [lower, upper]


def wilson(successes, n, confidence=0.95):
    successes, n = _count(successes, "successes"), _count(n, "n")
    if n == 0 or successes > n:
        raise ValueError("Require 0 <= successes <= n and nonempty n")
    confidence = _probability(confidence, "confidence", open_interval=True)
    z = NormalDist().inv_cdf((1+confidence)/2)
    p = successes/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0.0, center-half), min(1.0, center+half)]


def exact_mcnemar(left_only, right_only):
    """Two-sided exact conditional binomial test, including zero discordances."""
    b, c = _count(left_only, "left_only"), _count(right_only, "right_only")
    m = b+c
    if m == 0:
        return 1.0
    tail_numerator = sum(comb(m, k) for k in range(min(b, c)+1))
    return min(1.0, 2*tail_numerator/(1 << m))


def paired_binary(left, right, *, expected_family_ids, confidence=0.95,
                  bootstrap_replicates=20000, seed=20260912, strata=None):
    """Effect is LEFT minus RIGHT; rows are families, never policy replicas.

    Primary CI is conservative: simultaneous marginal Clopper-Pearson intervals
    for p10 and p01 (Bonferroni) imply a valid interval for p10-p01 under the
    Bernoulli family-sampling model. It is not
    called a Newcombe/score interval. Bootstrap CI is supplementary and flagged
    when degenerate, avoiding false certainty at a boundary.
    """
    ids, a, b = _paired(left, right, expected_family_ids)
    confidence = _probability(confidence, "confidence", open_interval=True)
    reps = _count(bootstrap_replicates, "bootstrap_replicates")
    if reps < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    seed = _count(seed, "seed")
    n = len(ids)
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n11, n00 = int(((a == 1) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())
    simultaneous_confidence = 1-(1-confidence)/2
    l10, u10 = clopper_pearson(n10, n, simultaneous_confidence)
    l01, u01 = clopper_pearson(n01, n, simultaneous_confidence)
    differences = a-b
    rng = np.random.default_rng(seed)
    if strata is None:
        groups = {"all_families": np.arange(n)}
    else:
        if not isinstance(strata, Mapping) or set(strata) != set(ids):
            raise ValueError("Strata must identify every frozen family exactly")
        if any(not isinstance(strata[i], str) or not strata[i].strip() for i in ids):
            raise ValueError("Stratum labels must be nonempty strings")
        groups = {label: np.array([j for j,i in enumerate(ids) if strata[i] == label])
                  for label in sorted(set(strata.values()))}
    bootstrap = np.zeros(reps)
    for indices in groups.values():
        sampled = indices[rng.integers(0,len(indices),size=(reps,len(indices)))]
        bootstrap += differences[sampled].sum(axis=1)/n
    boot_ci = np.quantile(bootstrap, [(1-confidence)/2, (1+confidence)/2]).tolist()
    return {
        "n_families": n, "family_ids": ids, "effect_direction": "left_minus_right",
        "left_successes": int(a.sum()), "right_successes": int(b.sum()),
        "left_rate": float(a.mean()), "right_rate": float(b.mean()),
        "left_wilson_ci": wilson(int(a.sum()), n, confidence),
        "right_wilson_ci": wilson(int(b.sum()), n, confidence),
        "left_exact_ci": clopper_pearson(int(a.sum()), n, confidence),
        "right_exact_ci": clopper_pearson(int(b.sum()), n, confidence),
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "difference": (n10-n01)/n,
        "difference_ci": [max(-1.0, l10-u01), min(1.0, u10-l01)],
        "difference_ci_method": "conservative paired discordance CP with Bonferroni union bound",
        "confidence": confidence, "intervals_are_multiplicity_adjusted": False,
        "mcnemar_exact_two_sided_p": exact_mcnemar(n10, n01),
        "bootstrap_difference_ci": boot_ci, "bootstrap_degenerate": boot_ci[0] == boot_ci[1],
        "bootstrap_replicates": reps, "bootstrap_seed": seed,
        "bootstrap_strata_counts": {label:len(indices) for label,indices in groups.items()},
        "bootstrap_scope": "resample paired family differences within fixed strata; conditional on frozen policies/masks"
    }


def holm_adjust(pvalues, *, expected_hypotheses, alpha=0.05):
    expected = _ids(expected_hypotheses)
    if not isinstance(pvalues, Mapping) or set(pvalues) != set(expected):
        raise ValueError("All prespecified hypotheses must be present; missing cells cannot be discarded")
    alpha = _probability(alpha, "alpha", open_interval=True)
    ordered = sorted(((_probability(pvalues[k], "pvalue"), k) for k in expected))
    adjusted, running = {}, 0.0
    for rank, (pvalue, key) in enumerate(ordered):
        running = max(running, (len(ordered)-rank)*pvalue)
        adjusted[key] = {"raw_p": pvalue, "adjusted_p": min(1.0, running),
                         "reject": min(1.0, running) <= alpha}
    return {"method": "Holm", "n_hypotheses": len(expected), "alpha": alpha,
            "hypotheses": {k: adjusted[k] for k in expected}}


def expected_one_packet_loss(delivered, no_packet, *, expected_family_ids, loss_probability):
    """Analytic POLICY expectation; fractional values are not actual outcomes.

    Assumes a post-admission, whole-packet Bernoulli drop independent of content,
    with no retransmission. No second model evaluation is performed here.
    """
    ids, a, b = _paired(delivered, no_packet, expected_family_ids)
    p = _probability(loss_probability, "loss_probability")
    values = (1-p)*a+p*b
    return {"execution_kind": "analytic_policy_expectation", "n_families": len(ids),
            "loss_probability": p, "expected_rate": float(values.mean()),
            "family_expectations": {key: float(value) for key, value in zip(ids, values)},
            "mcnemar_applicable": False}


def verify_replay_reference(actual, replay):
    """Minimal provenance guard; does not certify raw byte or GPU determinism.

    Required fingerprints must be computed from full effective model inputs and
    run identity by the producer/verifier, not copied blindly from a plan.
    Different evaluation targets still require rescoring the actual saved answer.
    """
    if actual.get("execution_kind") != "actual" or actual.get("execution_status") != "completed":
        raise ValueError("Replay source must be a completed actual model evaluation")
    if replay.get("execution_kind") != "policy_replay":
        raise ValueError("Derived evaluation must be explicitly labelled policy_replay")
    if not actual.get("result_id") or replay.get("source_result_id") != actual["result_id"]:
        raise ValueError("Replay does not identify its actual source result")
    for field in ("family_id", "effective_input_sha256", "model_run_identity_sha256", "output_sha256"):
        if not actual.get(field) or replay.get(field) != actual[field]:
            raise ValueError(f"Replay identity mismatch: {field}")
    for field in ("effective_input_sha256", "model_run_identity_sha256", "output_sha256"):
        value = actual[field]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Malformed fingerprint: {field}")
    return {"verified_metadata_reference": True, "new_model_generations": 0,
            "scope": "metadata identity only; independent byte reconstruction and saved-output rescoring still required"}
