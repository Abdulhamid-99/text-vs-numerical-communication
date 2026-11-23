"""Budget replay, paired confidence intervals, cost and calibration spectra."""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

from evidence import completed_manifest, read, require, rows, sha, write_csv


def close(a, b):
    return math.isclose(float(a), float(b), abs_tol=2e-12, rel_tol=2e-11)


def cp(k, n, confidence):
    tail = (1 - confidence) / 2
    return (
        0.0 if k == 0 else beta.ppf(tail, k, n - k + 1),
        1.0 if k == n else beta.ppf(1 - tail, k + 1, n - k),
    )


def compute(plan_file, analysis, output):
    checks = defaultdict(int)

    def check(condition, label):
        require(condition, label)
        checks[label] += 1

    plan = read(plan_file)
    manifest = completed_manifest(analysis, "heldout_separate_evaluation")
    for name in (
        "sweep_summary.json",
        "budget_sweep_replays.jsonl",
        "graded_actual.jsonl",
        "policy_replays.jsonl",
        "actual_summary.json",
        "statistics.json",
    ):
        check(
            sha(analysis / name) == manifest[name]["sha256"],
            "final_analysis_hash_match",
        )
    sweep = read(analysis / "sweep_summary.json")
    replay = rows(analysis / "budget_sweep_replays.jsonl")
    actual_rows = rows(analysis / "graded_actual.jsonl")
    actual = {row["receiver_record_id"]: row for row in actual_rows}
    policy = rows(analysis / "policy_replays.jsonl")
    stats = read(analysis / "statistics.json")
    actual_summary = read(analysis / "actual_summary.json")
    check(stats["plan_sha256"] == sha(plan_file), "statistics_final_plan_binding")
    caps = plan["sweep_budgets_bits"]
    check(
        len(caps) == 15 and len(replay) == 11520 and len(sweep) == 180,
        "complete_frozen_sweep_shape",
    )
    grouped = defaultdict(list)
    for row in replay:
        offered = actual[row["offered_receiver_record_id"]]
        admitted = offered["wire_bits"] <= row["budget_bits"]
        expected_id = (
            row["offered_receiver_record_id"]
            if admitted
            else row["no_message_record_id"]
        )
        chosen = actual[expected_id]
        check(
            row["receiver_record_id"] == expected_id
            and row["admitted"] == admitted
            and row["delivered"] == admitted,
            "sweep_atomic_admission",
        )
        check(
            row["success"] == chosen["score_original"]["success"]
            and row["strict_success"] == chosen["score_original"]["strict_success"]
            and row["effective_input_sha256"] == chosen["effective_input_sha256"]
            and row["output_sha256"] == chosen["output_sha256"],
            "sweep_observed_response_reuse",
        )
        check(
            row["sent_bits"]
            == row["received_bits"]
            == (offered["wire_bits"] if admitted else 0)
            and row["lost_bits"] == 0
            and row["loss_rate"] == 0,
            "sweep_bit_accounting",
        )
        grouped[(row["model"], row["task"], row["channel"], row["budget_bits"])].append(
            row
        )
    cap_rows = []
    for saved in sorted(
        sweep, key=lambda x: (x["model"], x["task"], x["channel"], x["budget_bits"])
    ):
        group = grouped[
            (saved["model"], saved["task"], saved["channel"], saved["budget_bits"])
        ]
        check(
            len(group) == len({x["family_id"] for x in group}) == 64,
            "sweep_family_denominator",
        )
        result = {
            key: saved[key] for key in ("model", "task", "channel", "budget_bits")
        }
        result.update(
            n_families=64,
            successes=sum(x["success"] for x in group),
            strict_successes=sum(x["strict_success"] for x in group),
            delivered_count=sum(x["delivered"] for x in group),
            mean_sent_bits=sum(x["sent_bits"] for x in group) / 64,
            mean_planned_wire_bits=sum(x["planned_wire_bits"] for x in group) / 64,
        )
        result.update(
            accuracy=result["successes"] / 64,
            delivery_rate=result["delivered_count"] / 64,
        )
        for key in (
            "successes",
            "strict_successes",
            "delivered_count",
            "mean_sent_bits",
            "mean_planned_wire_bits",
            "accuracy",
        ):
            check(close(result[key], saved[key]), "sweep_summary_reconciles")
        cap_rows.append(result)
    write_csv(output / "budget_sweep.csv", cap_rows)

    contrasts = []
    for saved in stats["primary_contrasts"]:
        left = {
            x["family_id"]: int(x["success"])
            for x in policy
            if x["model"] == saved["model"]
            and x["task"] == saved["task"]
            and x["regime"] == saved["regime"]
            and x["loss_rate"] == 0
            and x["channel"] == saved["left_channel"]
        }
        right = {
            x["family_id"]: int(x["success"])
            for x in policy
            if x["model"] == saved["model"]
            and x["task"] == saved["task"]
            and x["regime"] == saved["regime"]
            and x["loss_rate"] == 0
            and x["channel"] == "text"
        }
        check(
            set(left) == set(right) == set(saved["family_ids"]) and len(left) == 64,
            "contrast_family_pairing",
        )
        counts = {
            "n" + str(a) + str(b): sum(left[k] == a and right[k] == b for k in left)
            for a in (0, 1)
            for b in (0, 1)
        }
        check(
            all(counts[k] == saved[k] for k in counts), "discordance_counts_recomputed"
        )
        n10, n01 = counts["n10"], counts["n01"]
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        l10, u10 = cp(n10, 64, 0.975)
        l01, u01 = cp(n01, 64, 0.975)
        interval = [max(-1, l10 - u01), min(1, u10 - l01)]
        check(
            all(close(a, b) for a, b in zip(interval, saved["difference_ci"]))
            and close(p, saved["mcnemar_exact_two_sided_p"]),
            "independent_scipy_interval_and_mcnemar",
        )
        result = {
            key: saved[key]
            for key in (
                "model",
                "task",
                "regime",
                "left_channel",
                "right_channel",
                "n_families",
            )
        }
        result.update(
            left_successes=sum(left.values()),
            right_successes=sum(right.values()),
            **counts,
            difference_pp=100 * (n10 - n01) / 64,
            ci_low_pp=100 * interval[0],
            ci_high_pp=100 * interval[1],
            raw_p=p,
            global_holm_p=saved["global_holm"]["adjusted_p"],
            intervals_pointwise_not_familywise=True,
        )
        contrasts.append(result)
    ordered = sorted(contrasts, key=lambda x: x["raw_p"])
    running = 0
    for i, row in enumerate(ordered):
        running = max(running, min(1, (16 - i) * row["raw_p"]))
        check(close(running, row["global_holm_p"]), "global_holm_recomputed")
    write_csv(output / "primary_contrasts.csv", contrasts)
    write_csv(
        output / "high_budget_contrasts.csv",
        [x for x in contrasts if x["regime"] == "high"],
    )

    pca_rows, variance_rows = [], []
    for model in plan["models"]:
        probe = Path(__file__).resolve().parent / "calibration" / model
        seal = read(probe / "artifact_manifest.json")
        check(
            sha(probe / "artifact_manifest.json")
            == plan["models"][model]["pca_development_manifest_sha256"],
            "pca_probe_seal_bound_to_final_plan",
        )
        check(
            sha(probe / "configuration.json")
            == seal["files"]["configuration.json"]["sha256"],
            "pca_config_sealed",
        )
        config = read(probe / "configuration.json")
        trace = rows(probe / "calibration_trace.jsonl")
        check(
            sha(probe / "calibration_trace.jsonl")
            == seal["files"]["calibration_trace.jsonl"]["sha256"],
            "pca_trace_sealed",
        )
        for task in plan["tasks"]:
            binding = plan["models"][model]["tasks"][task]
            descriptor_path = plan_file.parent / "assets" / binding["descriptor_file"]
            basis_path = plan_file.parent / "assets" / binding["basis_file"]
            desc = read(descriptor_path)
            fit = read(probe / (task + "_fit.json"))
            check(
                sha(basis_path) == binding["basis_sha256"]
                and sha(descriptor_path) == binding["descriptor_sha256"],
                "pca_basis_and_descriptor_bound_to_plan",
            )
            check(
                sha(probe / (task + "_fit.json"))
                == seal["files"][task + "_fit.json"]["sha256"],
                "pca_fit_sealed",
            )
            selected = [x for x in trace if x["task"] == task]
            check(
                len(selected) == 64
                and len({x["family_id"] for x in selected}) == 32
                and set(x["family_id"] for x in selected)
                == set(desc["calibration_family_ids"]),
                "pca_calibration_sample_counts",
            )
            check(
                {x["variant"] for x in selected} == {"original", "alternate"},
                "pca_calibration_two_worlds",
            )
            singular = np.array(fit["singular_values"], dtype=np.float64)
            variance = np.cumsum(singular**2) / np.sum(singular**2)
            check(
                np.allclose(
                    variance,
                    fit["cumulative_explained_variance"],
                    atol=1e-12,
                    rtol=1e-12,
                ),
                "pca_variance_recomputed_from_recorded_spectrum",
            )
            token_rows = sum(x["encoding"]["shape"][0] for x in selected)
            check(token_rows == fit["rows"], "pca_token_rows_reconcile")
            pca_rows.append(
                dict(
                    model=model,
                    task=task,
                    calibration_families=32,
                    worlds_per_family=2,
                    calibration_arrays=64,
                    token_rows=token_rows,
                    hidden_width=fit["width"],
                    rank=512,
                    codec=config["codec"],
                    explained_variance=variance[511],
                    explained_variance_percent=100 * variance[511],
                    calibration_relative_squared_reconstruction_error=1 - variance[511],
                    basis_sha256=binding["basis_sha256"],
                )
            )
            for rank in (1, 8, 16, 32, 64, 128, 256, 512):
                variance_rows.append(
                    dict(
                        model=model,
                        task=task,
                        rank=rank,
                        explained_variance_percent=100 * variance[rank - 1],
                        scope="calibration_spectrum_only_no_new_receiver_evaluation",
                    )
                )
    write_csv(output / "pca_calibration.csv", pca_rows)
    write_csv(output / "pca_variance_by_rank.csv", variance_rows)

    costs = []
    for model in plan["models"]:
        for task in plan["tasks"]:
            means = {
                channel: next(
                    x["mean_wire_bits"]
                    for x in actual_summary
                    if x["model"] == model
                    and x["task"] == task
                    and x["condition"] == channel + "_generated"
                )
                for channel in ("text", "raw", "pca")
            }
            ratio = means["pca"] / means["text"]
            current_codes_per_row = 256 if model == "qwen" else 512
            spans = [
                (x["wire_bits"] // 8 - 64) // (4 + current_codes_per_row)
                for x in actual_rows
                if x["model"] == model
                and x["task"] == task
                and x["condition"] == "pca_generated"
            ]
            check(
                len(spans) == 64
                and all(1 <= n <= 128 for n in spans)
                and close(
                    8 * (64 + np.mean(spans) * (4 + current_codes_per_row)),
                    means["pca"],
                ),
                "pca_fixed_frame_and_body_accounting",
            )
            costs.append(
                dict(
                    model=model,
                    task=task,
                    mean_text_bits=means["text"],
                    mean_raw_bits=means["raw"],
                    mean_pca_bits=means["pca"],
                    pca_to_text_ratio=ratio,
                    pca_relative_reduction_to_mean_text_percent=100 * (1 - 1 / ratio),
                    pca_share_of_current_bits_at_mean_text_percent=100 / ratio,
                    fixed_pca_framing_bits=512,
                    mean_token_rows=float(np.mean(spans)),
                    hypothetical_rank1_mean_bits=512 + 40 * float(np.mean(spans)),
                    caveat="Mean byte arithmetic only; current fixed 512-bit framing already exceeds mean text, so rank reduction alone cannot break even; accuracy untested",
                )
            )
    write_csv(output / "mean_cost_break_even.csv", costs)
    return dict(checks)
