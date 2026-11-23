"""Family-level statistics with independent reconstruction of all policy references.

CPU only. Scoring is rerun from saved answers with the frozen scorer; no saved
success flag is accepted without comparison. Model token decoding is a separate
audit and is never implied by this statistical check.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import statistics

from common import Stage, read_json, read_rows, sha, verify_seal, write_json
from private_tasks import SCORER_VERSION, agent_view, score_answer
from paired_analysis import clopper_pearson, holm_adjust, paired_binary, wilson
from analyze_study import (LOSS_DRAW_VERSION, object_sha, require, stratum,
                           task_response, task_scores, validate_analysis_source, validate_families, validate_plan, validate_record)


def exact_index(rows, key, expected, name):
    result = {}
    for row in rows:
        identity = key(row)
        require(identity not in result, "Duplicate " + name)
        result[identity] = row
    require(set(result) == set(expected), "Missing/extra " + name)
    return result


def _draw(seed, fid):
    # Independent implementation: first seven bytes, discard their low 3 bits.
    require(isinstance(seed, str) and seed and "|" not in seed and isinstance(fid, str) and fid and "|" not in fid,
            "Invalid mask identity")
    prefix = hashlib.sha256((seed + "|" + fid).encode("utf-8")).digest()[:7]
    return (int.from_bytes(prefix, "big") >> 3) * (2.0 ** -53)


def verify_policy_rows(plan, families, actual, rows, lineage):
    """Reconstruct admission, post-admission loss, selected observation and costs."""
    expected = {(m, fid, c, regime, rate) for m in plan["models"] for fid in families
                for c in ("text", "raw", "pca") for regime in ("high", "low") for rate in plan["loss_rates"]}
    index = exact_index(rows, lambda r: (r["model"], r["family_id"], r["channel"], r["regime"], r["loss_rate"]), expected, "policy cell")
    for (model, fid, channel, regime, rate), row in index.items():
        require(type(row["loss_rate"]) in (int, float) and math.isfinite(row["loss_rate"]), "Invalid replay loss-rate type")
        require(row["budget_bits"] is None or type(row["budget_bits"]) is int, "Invalid replay budget type")
        require(type(row["loss_uniform"]) in (int, float) and math.isfinite(row["loss_uniform"]), "Invalid replay draw type")
        family = families[fid]
        require(row["task"] == family["task"] and all(row[k] == v for k, v in stratum(family).items()), "Wrong replay task/stratum")
        require(row.get("execution_kind") == "policy_replay" and row.get("execution_status") == "completed"
                and row.get("derived_policy_replay") is True, "Replay is mislabeled as an independent observation")
        source = actual[(model, fid, channel + "_generated")]
        none = actual[(model, fid, "no_comm")]
        for field in ("plan_sha256", "model_settings_sha256", "b_view_sha256"):
            require(source[field] == none[field], "Replay references different effective run/private B view")
        cap = None if regime == "high" else plan["models"][model]["tasks"][family["task"]]["low_budget_bits"]
        draw = _draw(lineage["loss_seed"], fid)
        require(row["budget_bits"] == cap and row["loss_uniform"] == draw, "Replay cap/mask differs from frozen rule")
        valid = source["source_status"] == "valid"
        require(source["source_status"] in ("valid", "source_contract_failure"), "Unavailable technical source state")
        bits = source["wire_bits"]
        require((valid and bits > 0) or (not valid and bits == 0), "Invalid source packet accounting")
        admitted = valid and (cap is None or bits <= cap)
        dropped = admitted and draw < rate
        delivered = admitted and not dropped
        selected = source if delivered else none
        status = ("source_contract_failure" if not valid else "budget_blocked" if not admitted
                  else "packet_lost" if dropped else "delivered")
        expected_fields = {"planned_wire_bits": bits, "sent_bits": bits if admitted else 0,
            "received_bits": bits if delivered else 0, "lost_bits": bits if dropped else 0,
            "admitted": admitted, "packet_lost": dropped, "delivered": delivered, "admission_status": status,
            "source_status": source["source_status"], "receiver_record_id": selected["receiver_record_id"],
            "source_result_id": selected["receiver_record_id"], "offered_receiver_record_id": source["receiver_record_id"],
            "no_message_record_id": none["receiver_record_id"], "success": selected["score_original"]["success"],
            "strict_success": selected["score_original"]["strict_success"], "receiver_score": selected["score_original"],
            "receiver_seconds": selected["receiver_seconds"], "planner_seconds": selected["planner_seconds"],
            "effective_input_sha256": selected["effective_input_sha256"],
            "output_sha256": selected["output_sha256"]}
        for field, value in expected_fields.items():
            require(row.get(field) == value and (type(row[field]) is type(value) if type(value) in (bool, int) else True),
                    "Policy reconstruction mismatch: " + field)
    return index


def rate_summary(values):
    require(values and all(type(v) is bool for v in values), "Empty/nonboolean outcome inventory")
    n, successes = len(values), sum(values)
    return {"n_families": n, "successes": successes, "rate": successes / n,
            "wilson95": wilson(successes, n), "clopper_pearson95": clopper_pearson(successes, n),
            "intervals_are_pointwise": True}


def build_statistics(plan, family_rows, graded_actual, policy_replays, *, lineage, plan_hash,
                     bootstrap_replicates=20000):
    validate_plan(plan)
    require(lineage["plan_sha256"] == plan_hash, "Statistical input differs from protocol")
    families = validate_families(plan, {"schema": 1, "split": "heldout", "families": family_rows})
    conditions = {condition: channel for channel, names in plan["conditions"].items() for condition in names}
    expected_actual = {(m, fid, c) for m in plan["models"] for fid in families for c in conditions}
    actual = exact_index(graded_actual, lambda r: (r["model"], r["family_id"], r["condition"]), expected_actual, "actual result cell")
    result_ids = set()
    for (model, fid, condition), row in actual.items():
        validate_record(row)
        family = families[fid]
        require(row["receiver_record_id"] == model + "_" + fid + "_" + condition and row["receiver_record_id"] not in result_ids,
                "Nonunique/wrong actual result ID")
        result_ids.add(row["receiver_record_id"])
        require(row["channel"] == conditions[condition] and row["task"] == family["task"], "Actual task/channel mismatch")
        require(row["scorer_version"] == SCORER_VERSION and row["plan_sha256"] == plan_hash, "Actual scorer/plan mismatch")
        require(row["model_settings_sha256"] == object_sha(plan["models"][model]) and
                row["b_view_sha256"] == object_sha(agent_view(family["episodes"]["original"], "B")), "Actual effective identity mismatch")
        require(all(row[k] == v for k, v in stratum(family).items()), "Actual stratum differs from family")
        require(isinstance(row["answer"], str), "Unavailable model answer")
        adapted = task_response(family, row)
        require(all(row[k] == v for k, v in adapted.items()), "Stored navigation/map diagnostics differ from independent replay")
        scores = task_scores(family, adapted)
        require(row["score_original"] == scores["original"] and row["score_alternate"] == scores["alternate"], "Saved task grades differ from scorer replay")
        require(not (row["score_original"]["success"] and row["score_alternate"]["success"]), "One answer succeeds in both disjoint counterfactual worlds")
        require(row["output_sha256"] == object_sha({"ids": row["output_ids"], "answer": row["answer"]}), "Output identity mismatch")
        if condition == "no_comm":
            require(row["source_status"] == "not_applicable" and row["wire_bits"] == 0, "Invalid no-message control")
    policies = verify_policy_rows(plan, families, actual, policy_replays, lineage)
    primary, pvalues, descriptive_actual, descriptive_strata, causal = [], {}, [], [], []
    expected_hypotheses = {"|".join((model, task, regime, channel + "_vs_text")) for model in plan["models"]
                           for task in plan["tasks"] for regime in ("high", "low") for channel in ("raw", "pca")}
    require(len(expected_hypotheses) == 16, "Expected exactly 16 frozen primary contrasts")
    for model in plan["models"]:
        for task in plan["tasks"]:
            ids = sorted(fid for fid, f in families.items() if f["task"] == task)
            strata = {fid: stratum(families[fid])["stratum"] for fid in ids}
            for regime in ("high", "low"):
                text = {fid: policies[(model, fid, "text", regime, 0.0)]["success"] for fid in ids}
                for channel in ("raw", "pca"):
                    numeric = {fid: policies[(model, fid, channel, regime, 0.0)]["success"] for fid in ids}
                    identity = "|".join((model, task, regime, channel + "_vs_text"))
                    comparison = paired_binary(numeric, text, expected_family_ids=ids, strata=strata,
                        seed=lineage["bootstrap_seed"], bootstrap_replicates=bootstrap_replicates)
                    pvalues[identity] = comparison["mcnemar_exact_two_sided_p"]
                    primary.append({"hypothesis_id": identity, "model": model, "task": task, "regime": regime,
                                    "loss_rate": 0.0, "left_channel": channel, "right_channel": "text", **comparison})
            for condition, channel in conditions.items():
                group = [actual[(model, fid, condition)] for fid in ids]
                summary = {"model": model, "task": task, "channel": channel, "condition": condition,
                    "n_families": len(ids), "success_original": sum(r["score_original"]["success"] for r in group),
                    "success_alternate": sum(r["score_alternate"]["success"] for r in group),
                    "strict_success_original": sum(r["score_original"]["strict_success"] for r in group),
                    "original": rate_summary([r["score_original"]["success"] for r in group]),
                    "alternate": rate_summary([r["score_alternate"]["success"] for r in group]),
                    "strict_original": rate_summary([r["score_original"]["strict_success"] for r in group]),
                    "mean_wire_bits": statistics.mean(r["wire_bits"] for r in group),
                    "total_wire_bits": sum(r["wire_bits"] for r in group),
                    "mean_receiver_seconds": statistics.mean(r["receiver_seconds"] for r in group),
                    "total_receiver_seconds": sum(r["receiver_seconds"] for r in group),
                    "mean_planner_seconds": statistics.mean(r["planner_seconds"] for r in group),
                    "total_planner_seconds": sum(r["planner_seconds"] for r in group),
                    "generation_limit_count": sum(r["generation_limit_reached"] for r in group),
                    "original_status_counts": dict(Counter(r["score_original"]["status"] for r in group)),
                    "source_status_counts": dict(Counter(r["source_status"] for r in group)),
                    "map_exact_original": None if group[0]["map_exact_original"] is None else rate_summary([r["map_exact_original"] for r in group]),
                    "map_exact_alternate": None if group[0]["map_exact_alternate"] is None else rate_summary([r["map_exact_alternate"] for r in group]),
                    "model_map_strict_format": None if group[0]["model_map_strict_format"] is None else rate_summary([r["model_map_strict_format"] for r in group]),
                    "execution_kind": "actual", "inferential_role": "descriptive_control"}
                descriptive_actual.append(summary)
                for label in sorted(set(strata.values())):
                    subset = [r for r in group if strata[r["family_id"]] == label]
                    descriptive_strata.append({"model": model, "task": task, "channel": channel, "condition": condition,
                        "stratum": label, "original": rate_summary([r["score_original"]["success"] for r in subset]),
                        "alternate": rate_summary([r["score_alternate"]["success"] for r in subset]), "inferential_role": "descriptive_stratum"})
            for channel in ("text", "raw", "pca"):
                original = [actual[(model, fid, channel + "_generated")] for fid in ids]
                swapped = [actual[(model, fid, channel + "_swapped")] for fid in ids]
                joint = [a["score_original"]["success"] and b["score_alternate"]["success"] for a, b in zip(original, swapped)]
                loss = [a["score_original"]["success"] and not b["score_original"]["success"] for a, b in zip(original, swapped)]
                stubborn = [a["score_original"]["success"] and b["score_original"]["success"] for a, b in zip(original, swapped)]
                causal.append({"model": model, "task": task, "channel": channel, "n_families": len(ids),
                    "joint_world_following": rate_summary(joint), "original_correct_then_swapped_original_incorrect": rate_summary(loss),
                    "original_correct_then_swapped_still_original": rate_summary(stubborn),
                    "swapped_originality_rate_change": (sum(b["score_original"]["success"] for b in swapped) - sum(a["score_original"]["success"] for a in original)) / len(ids),
                    "no_comm_joint_success_count": 0, "no_comm_disjoint_world_invariant_verified": True,
                    "interpretation": "Paired finite synthetic worlds with identical B information; conditional causal controls, not general semantic capacity.",
                    "inferential_role": "secondary_descriptive_no_additional_confirmatory_pvalues"})
    holm = holm_adjust(pvalues, expected_hypotheses=expected_hypotheses)
    for contrast in primary:
        contrast["global_holm"] = holm["hypotheses"][contrast["hypothesis_id"]]
    policy_descriptions = []
    for model in plan["models"]:
        for task in plan["tasks"]:
            ids = sorted(fid for fid, family in families.items() if family["task"] == task)
            for channel in ("text", "raw", "pca"):
                for regime in ("high", "low"):
                    for rate in plan["loss_rates"]:
                        group = [policies[(model, fid, channel, regime, rate)] for fid in ids]
                        policy_descriptions.append({"model": model, "task": task, "channel": channel, "regime": regime,
                            "loss_rate": rate, **rate_summary([r["success"] for r in group]),
                            "admission_counts": dict(Counter(r["admission_status"] for r in group)),
                            "total_sent_bits": sum(r["sent_bits"] for r in group), "total_received_bits": sum(r["received_bits"] for r in group),
                            "execution_kind": "policy_replay", "inferential_role": "primary_if_loss_zero_otherwise_descriptive"})
    return {"schema": 1, "status": "completed", "plan_sha256": plan_hash, "scorer_version": SCORER_VERSION,
        "loss_draw_rule": LOSS_DRAW_VERSION, "primary_contrasts": primary, "holm": holm,
        "descriptive_actual": descriptive_actual, "descriptive_strata": descriptive_strata,
        "secondary_counterfactual_controls": causal, "descriptive_policy": policy_descriptions,
        "counts": {"actual_receiver_rows": len(actual), "policy_replay_rows": len(policies), "primary_contrasts": 16,
                   "independent_families_per_task": plan["families_per_task"], "additional_generations_from_replays": 0},
        "scope": {"sampling_unit": "paired family within task/model; variants, channels, caps, and loss masks are repeated measures",
                  "effect_direction": "numeric minus text", "multiplicity": "One global Holm family of 16 within-model contrasts, not a test between model checkpoints",
                  "confidence_intervals": "Pointwise, not simultaneous. Wilson and Clopper-Pearson are Bernoulli-model summaries; fixed-stratum paired bootstrap preserves the design mix.",
                  "bootstrap": "Conditional on frozen caps and one paired loss realization; degenerate intervals are flagged, never evidence of exact equivalence.",
                  "loss": "Simulated deterministic whole-packet masks, no new receiver generation or measured network delivery",
                  "small_effects": "n=64 is not claimed to establish equivalence or adequate power for 5-10 percentage-point effects",
                  "model_token_decoding_independently_verified_here": False}}


def run(plan_path, data, analysis, output):
    stage = Stage(output, "independent_statistics_finalization", requested=16)
    try:
        verify_seal(data)
        verify_seal(analysis)
        plan = read_json(plan_path)
        validate_analysis_source(plan)
        result = build_statistics(plan, read_json(data / "private_families.json")["families"],
            read_rows(analysis / "graded_actual.jsonl"), read_rows(analysis / "policy_replays.jsonl"),
            lineage=read_json(data / "private_lineage.json"), plan_hash=sha(plan_path))
        write_json(output / "statistics.json", result)
        write_json(output / "inputs.json", {"plan_sha256": sha(plan_path), "families_sha256": sha(data / "private_families.json"),
            "graded_actual_sha256": sha(analysis / "graded_actual.jsonl"), "policy_replays_sha256": sha(analysis / "policy_replays.jsonl"),
            "finalizer_sha256": sha(Path(__file__)), "gpu_used": False})
        stage.update(n_completed=16)
        stage.finish()
    except BaseException as error:
        stage.fail(error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "data", "analysis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.data, args.analysis, args.output)
