"""Fail-closed evaluation of sealed actual calls and labelled policy replays.

This evaluator reconciles source/partition/receiver bytes and reruns task scoring.
It does not independently retokenize model continuations; that is a separate audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import statistics

from common import (Stage, append_row, read_json, read_rows, sha, verify_seal,
                    within, write_json)
from private_tasks import (NAV_TASK, QA_TASK, SCORER_VERSION, agent_view,
                           score_answer, validate_split_isolation)

LOSS_DRAW_VERSION = "sha256_first53_utf8_seed_pipe_family_div_2pow53"
EXPECTED_CONDITIONS = {
    "text": ["text_generated", "text_oracle", "text_swapped"],
    "raw": ["raw_generated", "raw_oracle", "raw_swapped", "raw_zero", "no_comm"],
    "pca": ["pca_generated", "pca_oracle", "pca_swapped", "pca_mean"],
}


def require(test, message):
    if not test:
        raise ValueError(message)


def nonnegative(value, name, *, integer=False):
    require(type(value) is int if integer else type(value) in (int, float), name + " has invalid type")
    require(math.isfinite(value) and value >= 0, name + " must be finite and nonnegative")
    return value


def unique(rows, key, description):
    output = {}
    for row in rows:
        identity = key(row)
        require(identity not in output, "Duplicate " + description)
        output[identity] = row
    return output


def validate_plan(plan):
    from navigation_agent import NAVIGATION_AGENT_VERSION
    require(plan.get("schema") == 1 and plan.get("status") == "frozen", "Expected frozen protocol")
    require(plan.get("scorer_version") == SCORER_VERSION, "Scorer version is not frozen in protocol")
    require(plan.get("navigation_agent_version") == NAVIGATION_AGENT_VERSION, "Navigation adapter is not frozen in protocol")
    require(plan.get("loss_draw_rule") == LOSS_DRAW_VERSION, "Loss mask algorithm differs from frozen rule")
    require(set(plan["models"]) == {"qwen", "llama"}, "Expected two prespecified model strata")
    require(len(plan["tasks"]) == 2 and set(plan["tasks"]) == {QA_TASK, NAV_TASK}, "Expected both prespecified tasks")
    require(plan["conditions"] == EXPECTED_CONDITIONS, "Condition inventory differs from prespecified design")
    nonnegative(plan["families_per_task"], "families_per_task", integer=True)
    require(plan["families_per_task"] > 0, "Empty study")
    require(plan["primary_regimes"] == ["high", "low"] and plan["loss_rates"] == [0.0, 0.05, 0.10], "Primary policy inventory differs")
    require(all(type(rate) in (int, float) and math.isfinite(rate) for rate in plan["loss_rates"]), "Invalid loss-rate types")
    for field in ("sweep_budgets_bits", "network_latency_ms", "network_bandwidth_bits_per_second"):
        require(plan[field] and len(set(plan[field])) == len(plan[field]), "Empty or duplicate policy settings")
        for value in plan[field]:
            nonnegative(value, field, integer=True)
            require(field != "network_bandwidth_bits_per_second" or value > 0, "Zero network bandwidth")
    for settings in plan["models"].values():
        require(set(settings["tasks"]) == set(plan["tasks"]), "Missing task settings")
        require(settings["raw_codec"] == "int4", "Raw codec differs from design")
        require(type(settings["generation"]["max_new_tokens"]) is int and settings["generation"]["max_new_tokens"] > 0,
                "Invalid model continuation bound")
        for task in plan["tasks"]:
            config = settings["tasks"][task]
            bits = config["baseline_development_bits"]
            require(bits and all(type(b) is int and b > 0 for b in bits), "Missing development baseline costs")
            # Integer arithmetic implements floor(mean), without float rounding.
            baseline = sum(bits) // len(bits)
            nonnegative(config["baseline_bits"], "baseline_bits", integer=True)
            nonnegative(config["low_budget_bits"], "low_budget_bits", integer=True)
            require(config["baseline_bits"] == baseline and config["low_budget_bits"] == baseline // 4,
                    "Common Low cap does not equal development baseline // 4")


def validate_families(plan, document):
    require(document.get("schema") == 1 and document.get("split") == "heldout", "Wrong family dataset")
    rows = document["families"]
    families = unique(rows, lambda f: f["family_id"], "family ID")
    require(Counter(f["task"] for f in rows) == Counter({t: plan["families_per_task"] for t in plan["tasks"]}),
            "Family counts per task differ from frozen plan")
    validate_split_isolation(rows)
    for family in rows:
        require(family["split"] == "heldout" and set(family["episodes"]) == {"original", "alternate"}, "Wrong family split/variants")
        original, alternate = (family["episodes"][v] for v in ("original", "alternate"))
        require(agent_view(original, "B") == agent_view(alternate, "B"), "Counterparts have different B information")
        for variant, episode in family["episodes"].items():
            require(episode["task"] == family["task"] and episode["variant"] == variant,
                    "Episode task or variant mismatch")
        stratum(family)
    expected_strata = Counter({"qa_rows_" + str(k): sum(4 + i % 3 == k for i in range(plan["families_per_task"])) for k in (4, 5, 6)})
    expected_strata.update({"navigation_size_" + str(k): sum(5 + i % 2 == k for i in range(plan["families_per_task"])) for k in (5, 6)})
    require(Counter(stratum(f)["stratum"] for f in rows) == expected_strata, "Heldout stratum allocation differs from frozen round-robin design")
    return families


def stratum(family):
    if family["task"] == QA_TASK:
        n = family["design"]["rows"]
        require(type(n) is int and n in (4, 5, 6), "Invalid QA row stratum")
        require(all(len(e["evaluator"]["directory"]) == n and len(e["evaluator"]["registry"]) == n
                    for e in family["episodes"].values()), "QA stratum differs from task contents")
        return {"stratum": "qa_rows_" + str(n), "qa_rows": n, "navigation_size": None}
    n = family["design"]["size"]
    require(type(n) is int and n in (5, 6), "Invalid navigation size stratum")
    require(all(len(e["evaluator"]["grid"]) == n and all(len(row) == n for row in e["evaluator"]["grid"])
                for e in family["episodes"].values()), "Navigation stratum differs from actual grid")
    return {"stratum": "navigation_size_" + str(n), "qa_rows": None, "navigation_size": n}


def loss_draw(seed, family_id):
    require(isinstance(seed, str) and seed and "|" not in seed, "Invalid private loss seed")
    require(isinstance(family_id, str) and family_id and "|" not in family_id, "Invalid loss family ID")
    digest = hashlib.sha256((seed + "|" + family_id).encode("utf-8")).digest()
    return (int.from_bytes(digest, "big") >> 203) / (1 << 53)


def validate_score(score):
    require(type(score.get("success")) is bool and type(score.get("strict_success")) is bool,
            "Scorer must return explicit boolean semantic and strict outcomes")
    require(score.get("scorer_version") == SCORER_VERSION and isinstance(score.get("status"), str), "Unbound scorer result")
    require(not score["strict_success"] or score["success"], "Strict success contradicts semantic outcome")


def task_response(family, result):
    """Recompute the B-only public adapter; never trust a serialized tool answer."""
    from navigation_agent import attach_task_response
    view = agent_view(family["episodes"]["original"], "B")
    recomputed = attach_task_response(view, {"answer": result["answer"]})
    require(result.get("task_answer") == recomputed["task_answer"] and
            result.get("navigation_planner") == recomputed.get("navigation_planner"), "Task adapter output differs from B-only recomputation")
    predicted = recomputed.get("navigation_planner", {}).get("predicted_grid")
    # The parser already proved that this is one valid JSON map (optionally fenced).
    # A canonical tool-generated route is not evidence of strict neural formatting.
    model_map_strict = None if family["task"] == QA_TASK else predicted is not None and not result["answer"].strip().startswith("```")
    return {"task_answer": recomputed["task_answer"], "navigation_planner": recomputed.get("navigation_planner"),
            "model_map_strict_format": model_map_strict,
            "map_exact_original": None if family["task"] == QA_TASK else predicted == family["episodes"]["original"]["evaluator"]["grid"],
            "map_exact_alternate": None if family["task"] == QA_TASK else predicted == family["episodes"]["alternate"]["evaluator"]["grid"]}


def task_scores(family, adapted):
    scores = {}
    for variant in ("original", "alternate"):
        result = score_answer(family["episodes"][variant], adapted["task_answer"])
        if family["task"] == NAV_TASK:
            result["route_format_strict_success"] = result["strict_success"]
            result["strict_success"] = result["success"] and adapted["model_map_strict_format"]
            result["strict_success_definition"] = "Executable true-environment route AND sole unfenced JSON map from B model"
        scores[variant] = result
    return scores


def validate_record(record):
    require(record.get("execution_status") == "completed" and record.get("execution_kind") == "actual",
            "Technical failure or unobserved result cannot become a task score")
    require(isinstance(record["receiver_record_id"], str) and record["receiver_record_id"], "Missing actual result ID")
    nonnegative(record["wire_bits"], "wire_bits", integer=True)
    nonnegative(record["receiver_seconds"], "receiver_seconds")
    nonnegative(record["planner_seconds"], "planner_seconds")
    validate_score(record["score_original"])
    validate_score(record["score_alternate"])


def policy(delivered_record, no_message_record, cap, rate, draw):
    if cap is not None:
        nonnegative(cap, "Budget", integer=True)
    require(type(rate) in (int, float) and math.isfinite(rate) and 0 <= rate <= 1, "Invalid loss probability")
    require(type(draw) in (int, float) and math.isfinite(draw) and 0 <= draw < 1, "Invalid loss draw")
    for record in (delivered_record, no_message_record):
        validate_record(record)
    for field in ("model", "task", "family_id", "plan_sha256", "model_settings_sha256", "b_view_sha256"):
        require(delivered_record[field] == no_message_record[field], "Cross-identity policy reference: " + field)
    require(delivered_record["condition"] == delivered_record["channel"] + "_generated", "Policy source is not generated-original")
    require(no_message_record["condition"] == "no_comm" and no_message_record["channel"] == "raw"
            and no_message_record["source_status"] == "not_applicable" and no_message_record["wire_bits"] == 0,
            "Fallback is not an actual same-family no-message observation")
    source_status = delivered_record["source_status"]
    require(source_status in ("valid", "source_contract_failure"), "Unknown/technical source failure")
    bits = delivered_record["wire_bits"]
    valid = source_status == "valid"
    require((valid and bits > 0) or (not valid and bits == 0), "Source status contradicts packet cost")
    admitted = valid and (cap is None or bits <= cap)
    dropped = admitted and draw < rate
    received = admitted and not dropped
    chosen = delivered_record if received else no_message_record
    status = "source_contract_failure" if not valid else "budget_blocked" if not admitted else "packet_lost" if dropped else "delivered"
    return {"planned_wire_bits": bits, "sent_bits": bits if admitted else 0,
        "received_bits": bits if received else 0, "lost_bits": bits if dropped else 0,
        "admitted": admitted, "packet_lost": dropped, "delivered": received, "admission_status": status,
        "offered_receiver_record_id": delivered_record["receiver_record_id"],
        "no_message_record_id": no_message_record["receiver_record_id"],
        "receiver_record_id": chosen["receiver_record_id"], "source_result_id": chosen["receiver_record_id"],
        "success": chosen["score_original"]["success"], "strict_success": chosen["score_original"]["strict_success"],
        "receiver_seconds": chosen["receiver_seconds"], "planner_seconds": chosen["planner_seconds"], "receiver_score": chosen["score_original"],
        "effective_input_sha256": chosen["effective_input_sha256"],
        "output_sha256": chosen["output_sha256"], "source_status": source_status,
        "execution_kind": "policy_replay", "execution_status": "completed", "derived_policy_replay": True}


def summarize(rows, keys, expected_by_task):
    require(bool(rows), "Empty policy results")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for labels, group in sorted(groups.items(), key=lambda item: str(item[0])):
        by_family = unique(group, lambda r: r["family_id"], "family within policy cell")
        require(set(by_family) == set(expected_by_task[group[0]["task"]]), "Incomplete policy denominator")
        require(all(type(r[k]) is bool for r in group for k in ("success", "strict_success", "admitted", "packet_lost", "delivered")), "Nonboolean policy field")
        sums = {key: sum(r[key] for r in group) for key in ("planned_wire_bits", "sent_bits", "received_bits", "lost_bits")}
        require(sums["sent_bits"] == sums["received_bits"] + sums["lost_bits"], "Bit conservation failure")
        output.append({**dict(zip(keys, labels)), "n_families": len(group), "successes": sum(r["success"] for r in group),
            "accuracy": statistics.mean(r["success"] for r in group), "strict_successes": sum(r["strict_success"] for r in group),
            **{"total_" + k: v for k, v in sums.items()},
            **{"mean_" + k: v / len(group) for k, v in sums.items()},
            "admitted_count": sum(r["admitted"] for r in group), "lost_packet_count": sum(r["packet_lost"] for r in group),
            "delivered_count": sum(r["delivered"] for r in group),
            "admission_counts": dict(Counter(r["admission_status"] for r in group)),
            "receiver_outcome_counts": dict(Counter(r["receiver_score"]["status"] for r in group)),
            "execution_kind": "policy_replay", "additional_model_generations": 0})
    return output


def object_sha(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def analysis_source_inventory():
    import common
    import finalize_statistics
    import navigation_agent
    import paired_analysis
    import private_tasks
    from budgetfair.comm import pca_packet, tensor_packet
    modules = {"common.py": common, "finalize_statistics.py": finalize_statistics,
               "navigation_agent.py": navigation_agent, "paired_analysis.py": paired_analysis,
               "private_tasks.py": private_tasks}
    result = {"thesis_study/" + name: sha(Path(module.__file__)) for name, module in modules.items()}
    result["thesis_study/analyze_study.py"] = sha(Path(__file__))
    result.update({"src/budgetfair/comm/" + name: sha(Path(module.__file__))
                   for name, module in (("tensor_packet.py", tensor_packet), ("pca_packet.py", pca_packet))})
    return result


def validate_analysis_source(plan):
    require(all(plan.get("code_sha256", {}).get(path) == digest for path, digest in analysis_source_inventory().items()),
            "Executed evaluator/scorer/adapter/statistics/codec bytes differ from frozen code")


def completed_stage(root, phase, requested, **identity):
    state = verify_seal(root)
    require(state["phase"] == phase and state["n_requested"] == requested and state["n_completed"] == requested,
            "Stage phase/count differs from exact planned inventory")
    require(all(state.get(k) == v for k, v in identity.items()), "Stage identity mismatch")
    return state


def packet_bytes(root, descriptor):
    require(set(descriptor) == {"file", "sha256", "wire_bits"}, "Unexpected packet descriptor")
    raw = within(root, descriptor["file"]).read_bytes()
    require(raw and hashlib.sha256(raw).hexdigest() == descriptor["sha256"], "Packet bytes/hash mismatch")
    require(type(descriptor["wire_bits"]) is int and descriptor["wire_bits"] == len(raw) * 8, "Packet cost differs from actual bytes")
    return raw


def validate_encoding(row, source, family, root, settings, asset):
    """Bind generated text/source arrays and numeric/control packets (CPU only)."""
    import numpy as np
    from budgetfair.comm.tensor_packet import decode_packet, encode_tensor
    from budgetfair.comm.pca_packet import FRAME, MAGIC, reconstruct
    fid, variant, kind = family["family_id"], row["variant"], row["kind"]
    require(row["id"] == fid + "_" + variant + "_" + kind and row["source_id"] == fid + "_" + variant,
            "Encoding source identity mismatch")
    require(row["family_id"] == fid and row["task"] == family["task"], "Encoding family/task mismatch")
    expected_status = source["source_status"] if kind == "generated" else "valid"
    require(row["source_status"] == expected_status, "Encoding source status mismatch")
    if expected_status != "valid":
        require(expected_status == "source_contract_failure" and not row["packets"], "Failed source has packet data")
        return {}
    keys = {"text", "raw", "pca"} | ({"raw_zero", "pca_mean"} if variant == "original" and kind == "generated" else set())
    require(set(row["packets"]) == keys, "Encoding packet inventory mismatch")
    packets = {k: packet_bytes(root, v) for k, v in row["packets"].items()}
    text = source["trace"]["text"] if kind == "generated" else family["episodes"][variant]["a_private_context"]
    require(packets["text"] == text.encode("utf-8"), "Text wire differs from actual A output / A-only oracle")
    array_path = within(root, row["array_file"])
    require(sha(array_path) == row["array_file_sha256"], "Raw capture file changed")
    array = np.load(array_path, allow_pickle=False)
    require(array.dtype == np.float32 and array.ndim == 2 and 1 <= array.shape[0] <= 128 and np.isfinite(array).all(), "Invalid raw source capture")
    trace = row["encoding"]
    require(trace["source_array_sha256"] == hashlib.sha256(array.astype("<f4").tobytes()).hexdigest()
            and trace["shape"] == list(array.shape), "Source capture/trace mismatch")
    require(trace["layer"] == settings["layer"] and trace["codec"] == settings["raw_codec"], "Source codec/layer mismatch")
    require(len(trace["message_token_ids"]) == array.shape[0] and trace["sender_input_ids"] ==
            row["encoding_prompt"]["prefix_ids"] + trace["message_token_ids"], "Source IDs do not match the A-only ordered payload span")
    require(trace["packet_sha256"] == row["packets"]["raw"]["sha256"] and trace["wire_bits"] == len(packets["raw"]) * 8,
            "Raw trace/wire mismatch")
    role = "token_embeddings" if settings["layer"] == -1 else "hidden_states"
    wire_layer = None if settings["layer"] == -1 else settings["layer"]
    require(packets["raw"] == encode_tensor(array, codec=settings["raw_codec"], role=role, layer_index=wire_layer), "Raw packet not derived from saved source array")
    nonnegative(row["encoding_seconds"], "encoding seconds")
    nonnegative(row["pca_encoding_seconds"], "PCA encoding seconds")
    decoded = {"raw": decode_packet(packets["raw"]).array}
    for key in ("pca", "pca_mean"):
        if key not in packets:
            continue
        magic, fingerprint = FRAME.unpack_from(packets[key])
        require(magic == MAGIC and fingerprint.hex() == asset.fingerprint, "PCA asset fingerprint differs")
        inner = decode_packet(packets[key][FRAME.size:])
        require(inner.array.shape == (array.shape[0], settings["pca_rank"]), "PCA row/rank mismatch")
        if key == "pca_mean":
            require(np.count_nonzero(inner.array) == 0, "PCA mean control has nonzero coefficients")
        # Also validates role/layer/codec and reconstructs bytes used at receiver.
        decoded[key], _ = reconstruct(packets[key], asset, expected_dimensions=settings["pca_rank"], expected_codec=settings["pca_codec"])
    if "raw_zero" in packets:
        require(packets["raw_zero"] == encode_tensor(np.zeros_like(array), codec=settings["raw_codec"], role=role, layer_index=wire_layer), "Zero control is not a same-shape zero packet")
        decoded["raw_zero"] = decode_packet(packets["raw_zero"]).array
    return {k: {"bytes": v, "decoded_sha256": None if k == "text" else hashlib.sha256(decoded[k].astype("<f4").tobytes()).hexdigest(),
                "rows": array.shape[0]} for k, v in packets.items()}


def validate_receiver(row, item, encoding, packet_cache, input_root, settings, channel, plan_hash, input_hash):
    require(row["id"] == item["id"] and all(row[k] == item[k] for k in ("family_id", "task", "condition", "source_status", "packet")),
            "Receiver row differs from frozen partition item")
    require(row["channel"] == channel and row["plan_sha256"] == plan_hash and row["input_sha256"] == input_hash,
            "Receiver plan/input/channel mismatch")
    condition = row["condition"]
    status = "not_applicable" if condition == "no_comm" else encoding["source_status"]
    require(row["source_status"] == status, "Receiver/source status mismatch")
    result = row["result"]
    require(result["status"] == "completed" and isinstance(result["answer"], str), "Technical failure is not a score")
    ids = result["output_ids"]
    require(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids), "Missing/invalid generated continuation IDs")
    require(len(ids) <= settings["generation"]["max_new_tokens"], "Receiver continuation exceeds frozen generation bound")
    # Empty decoded text with EOS output remains an observed abstention, not a technical error.
    nonnegative(row["seconds"], "receiver seconds")
    nonnegative(row["planner_seconds"], "planner seconds")
    require(row["task"] != QA_TASK or row["planner_seconds"] == 0, "QA does not invoke navigation planner")
    expected = None
    if condition != "no_comm" and status == "valid":
        key = condition if condition in ("raw_zero", "pca_mean") else channel
        expected = packet_cache[key]
        original = encoding["packets"][key]
        require(row["packet"] is not None, "Missing delivered packet")
        require(all(row["packet"][k] == original[k] for k in ("sha256", "wire_bits")), "Receiver packet descriptor differs from source")
        require(packet_bytes(input_root, row["packet"]) == expected["bytes"], "Receiver packet bytes differ from source")
    else:
        require(status in ("not_applicable", "source_contract_failure") and row["packet"] is None, "Absent packet status mismatch")
    raw = None if expected is None else expected["bytes"]
    require(result["wire_bits"] == (0 if raw is None else 8 * len(raw)) and
            result["packet_sha256"] == (None if raw is None else hashlib.sha256(raw).hexdigest()), "Receiver runtime wire accounting differs")
    injections = result["injections"]
    wanted = int(expected is not None and channel != "text")
    require(isinstance(injections, list) and len(injections) == wanted, "Incorrect actual injection count")
    public = row["receiver_prompt"]
    prefix, suffix = public["prefix_ids"], public["suffix_ids"]
    if raw is None or channel != "text":
        n = public["no_message_slots"] if raw is None else expected["rows"]
        require(result["receiver_input_ids"] == prefix + [public["placeholder_id"]] * n + suffix,
                "Numeric/absent receiver IDs contain data beyond public placeholder layout")
    if wanted:
        injection = injections[0]
        require(injection["layer"] == settings["layer"] and injection["positions"] == [len(prefix), len(prefix) + expected["rows"]]
                and injection["decoded_array_sha256"] == expected["decoded_sha256"], "Wrong injected layer/span/decoded numeric payload")
        nonnegative(injection["delta_l2"], "injection delta")


def run(plan_path, data, study, output):
    # The output stage exists before all evidence preflight, so refusals are durable.
    stage = Stage(output, "heldout_separate_evaluation", requested=0)
    try:
        plan = read_json(plan_path)
        validate_plan(plan)
        validate_analysis_source(plan)
        plan_hash = sha(plan_path)
        families = validate_families(plan, read_json(data / "private_families.json"))
        n = len(families)
        expected = n * len(plan["models"]) * sum(map(len, plan["conditions"].values()))
        stage.update(n_requested=expected)
        completed_stage(data, "heldout_data", n)
        completed_stage(study, "complete_isolated_heldout_study", len(plan["models"]) * 4)
        lineage = read_json(data / "private_lineage.json")
        require(lineage["plan_sha256"] == plan_hash, "Data differ from protocol")
        receipt = read_json(study / "input_receipt.json")
        require(receipt["plan_sha256"] == plan_hash and receipt["data_manifest_sha256"] == sha(data / "artifact_manifest.json"), "Study differs from frozen dataset")
        actual, index, sender_index, encoding_index = [], {}, {}, {}
        counts = {"a_generations": 0, "b_generations": 0, "encoder_forwards": 0, "injections": 0, "pca_injections": 0,
                  "source_contract_failures": 0, "output_tokens_B": 0, "output_tokens_A": 0,
                  "a_exact_private_message": 0, "a_strip_equal_private_message": 0,
                  "b_navigation_planner_calls": 0, "b_planner_seconds": 0.0}
        actual_ids = set()
        expected_by_task = {t: {fid for fid, f in families.items() if f["task"] == t} for t in plan["tasks"]}
        ainput_path = data / "A" / "input.json"
        ainput = read_json(ainput_path)
        aitems = unique(ainput["items"], lambda r: r["id"], "sender input ID")
        require(ainput["plan_sha256"] == plan_hash and ainput["role"] == "A", "Wrong sender input plan/role")
        require(set(aitems) == {fid + "_" + v for fid in families for v in ("original", "alternate")}, "Wrong sender input coverage")
        for model, settings in plan["models"].items():
            from budgetfair.comm.pca_packet import load_asset
            assets = {t: load_asset(within(plan_path.parent / "assets", settings["tasks"][t]["basis_file"]),
                                   expected_sha256=settings["tasks"][t]["basis_sha256"]) for t in plan["tasks"]}
            aroot = study / (model + "_sender") / "runtime" / "stage"
            astate = completed_stage(aroot, "heldout_sender", n * 2, model=model)
            identity = read_json(aroot / "identity.json")
            require(identity["plan_sha256"] == plan_hash and identity["input_sha256"] == sha(ainput_path)
                    and identity["model"] == model and identity["settings"] == settings, "Sender runtime identity mismatch")
            senders = unique(read_rows(aroot / "senders.jsonl"), lambda r: r["id"], "sender result ID")
            encodings = unique(read_rows(aroot / "encodings.jsonl"), lambda r: r["id"], "encoding ID")
            require(set(senders) == set(aitems), "Incomplete sender identity inventory")
            require(set(encodings) == {fid + "_" + variant + "_" + kind for fid in families
                                      for variant, kind in (("original", "generated"), ("alternate", "generated"), ("original", "oracle"))}, "Incomplete encoding identity inventory")
            cache = {}
            for sid, source in senders.items():
                item, family = aitems[sid], families[source["family_id"]]
                require(source["task"] == family["task"] == item["task"] and source["variant"] == item["variant"]
                        and item["family_id"] == family["family_id"] and item["view"] == agent_view(family["episodes"][source["variant"]], "A"), "Sender private-view binding mismatch")
                require(source["source_status"] in ("valid", "source_contract_failure"), "Technical sender status is not a scored source failure")
                length = nonnegative(source["payload_token_count"], "payload token count", integer=True)
                require((source["source_status"] == "valid") == (1 <= length <= 128), "Sender length/status mismatch")
                trace = source["trace"]
                require(isinstance(trace["text"], str) and trace["output_ids"] and all(type(i) is int and i >= 0 for i in trace["output_ids"]), "Missing sender continuation")
                require(len(trace["output_ids"]) <= settings["generation"]["max_new_tokens"], "Sender continuation exceeds frozen generation bound")
                nonnegative(source["seconds"], "sender seconds")
                requested_text = family["episodes"][source["variant"]]["a_private_context"]
                counts["a_exact_private_message"] += int(trace["text"] == requested_text)
                counts["a_strip_equal_private_message"] += int(trace["text"].strip() == requested_text.strip())
                sender_index[(model, sid)] = source
            for eid, enc in encodings.items():
                family = families[enc["family_id"]]
                cache[eid] = validate_encoding(enc, senders[enc["source_id"]], family, aroot, settings, assets[family["task"]])
                encoding_index[(model, eid)] = enc
            count_enc = sum(r["source_status"] == "valid" for r in encodings.values())
            require(astate["a_generations_completed"] == len(senders) and astate["encoding_forwards_completed"] == count_enc, "Sender stage counters differ from evidence")
            counts["a_generations"] += len(senders)
            counts["encoder_forwards"] += count_enc
            counts["source_contract_failures"] += sum(r["source_status"] == "source_contract_failure" for r in senders.values())
            counts["output_tokens_A"] += sum(len(r["trace"]["output_ids"]) for r in senders.values())
            for channel, conditions in plan["conditions"].items():
                root = study / (model + "_" + channel + "_receiver") / "runtime" / "stage"
                state = completed_stage(root, "heldout_receiver", n * len(conditions), model=model, channel=channel)
                input_root = study / (model + "_" + channel + "_input")
                input_hash = sha(input_root / "input.json")
                inputs = read_json(input_root / "input.json")
                require(inputs["plan_sha256"] == plan_hash and inputs["channel"] == channel and inputs["role"] == "B", "Wrong B partition identity")
                items = unique(inputs["items"], lambda r: (r["family_id"], r["condition"]), "B input cell")
                rows = unique(read_rows(root / "rows.jsonl"), lambda r: (r["family_id"], r["condition"]), "receiver cell")
                expected_cells = {(fid, condition) for fid in families for condition in conditions}
                require(set(items) == set(rows) == expected_cells, "Incomplete B input/result cell coverage")
                identity = read_json(root / "identity.json")
                require(identity["plan_sha256"] == plan_hash and identity["input_sha256"] == input_hash and identity["settings"] == settings
                        and identity["model"] == model and identity["channel"] == channel, "Receiver runtime identity mismatch")
                for key, row in rows.items():
                    family, item = families[row["family_id"]], items[key]
                    fid, condition = key
                    require(item["id"] == model + "_" + fid + "_" + condition and item["task"] == family["task"]
                            and item["view"] == agent_view(family["episodes"]["original"], "B"), "B partition private-view mismatch")
                    variant, kind = ("alternate" if condition.endswith("swapped") else "original"), ("oracle" if condition.endswith("oracle") else "generated")
                    eid = fid + "_" + variant + "_" + kind
                    validate_receiver(row, item, encodings[eid], cache[eid], input_root, settings, channel, plan_hash, input_hash)
                    result = row["result"]
                    adapted = task_response(family, result)
                    scores = task_scores(family, adapted)
                    original, alternate = scores["original"], scores["alternate"]
                    validate_score(original)
                    validate_score(alternate)
                    require(not (original["success"] and alternate["success"]), "Counterpart outcomes are not mutually exclusive")
                    record = {"model": model, "task": family["task"], "family_id": fid, **stratum(family),
                        "condition": condition, "channel": channel, "source_status": row["source_status"],
                        "receiver_record_id": row["id"], "score_original": original, "score_alternate": alternate,
                        "wire_bits": result["wire_bits"], "answer": result["answer"], **adapted,
                        "receiver_seconds": row["seconds"], "planner_seconds": row["planner_seconds"],
                        "output_tokens": len(result["output_ids"]), "output_ids": result["output_ids"],
                        "generation_limit_reached": len(result["output_ids"]) >= settings["generation"]["max_new_tokens"],
                        "execution_status": "completed", "execution_kind": "actual", "scorer_version": SCORER_VERSION,
                        "plan_sha256": plan_hash, "model_settings_sha256": object_sha(settings), "b_view_sha256": object_sha(item["view"]),
                        "effective_input_sha256": object_sha({"ids": result["receiver_input_ids"], "packet_sha256": result["packet_sha256"],
                                                             "layer": settings["layer"] if result["injections"] else None}),
                        "output_sha256": object_sha({"ids": result["output_ids"], "answer": result["answer"]}),
                        "evidence_file": (root / "rows.jsonl").relative_to(study).as_posix(), "evidence_sha256": sha(root / "rows.jsonl")}
                    require(record["receiver_record_id"] not in actual_ids, "Duplicate global actual result ID")
                    actual_ids.add(record["receiver_record_id"])
                    actual.append(record)
                    index[(model, fid, condition)] = record
                    append_row(output / "graded_actual.jsonl", record)
                    counts["b_generations"] += 1
                    counts["injections"] += len(result["injections"])
                    counts["pca_injections"] += int(channel == "pca") * len(result["injections"])
                    counts["output_tokens_B"] += len(result["output_ids"])
                    counts["b_navigation_planner_calls"] += int(family["task"] == NAV_TASK)
                    counts["b_planner_seconds"] += row["planner_seconds"]
                    stage.update(n_completed=len(actual))
                require(state["b_generations_completed"] == len(rows), "Receiver counter differs from actual rows")
        primary, sweep, latency = [], [], []
        for model, settings in plan["models"].items():
            for fid, family in families.items():
                task = family["task"]
                draw = loss_draw(lineage["loss_seed"], fid)
                none = index[(model, fid, "no_comm")]
                for channel in ("text", "raw", "pca"):
                    delivered = index[(model, fid, channel + "_generated")]
                    enc, source = encoding_index[(model, fid + "_original_generated")], sender_index[(model, fid + "_original")]
                    for regime, cap in (("high", None), ("low", settings["tasks"][task]["low_budget_bits"])):
                        for rate in plan["loss_rates"]:
                            replay = policy(delivered, none, cap, rate, draw)
                            row = {"model": model, "task": task, "family_id": fid, **stratum(family), "channel": channel,
                                   "regime": regime, "budget_bits": cap, "loss_rate": rate, "loss_uniform": draw, **replay}
                            primary.append(row)
                            append_row(output / "policy_replays.jsonl", row)
                            if rate == 0:
                                encoding_seconds = 0.0 if channel == "text" or enc["source_status"] != "valid" else enc["encoding_seconds"]
                                if channel == "pca" and enc["source_status"] == "valid":
                                    encoding_seconds += enc["pca_encoding_seconds"]
                                measured = source["seconds"] + encoding_seconds + replay["receiver_seconds"] + replay["planner_seconds"]
                                for delay in plan["network_latency_ms"]:
                                    for bandwidth in plan["network_bandwidth_bits_per_second"]:
                                        network_seconds = replay["sent_bits"] / bandwidth + (delay / 1000 if replay["sent_bits"] else 0)
                                        latency.append({"model": model, "task": task, "family_id": fid, **stratum(family),
                                            "channel": channel, "regime": regime, "bandwidth_bits_per_second": bandwidth,
                                            "one_way_latency_ms": delay, "measured_component_seconds": measured,
                                            "measured_sender_seconds": source["seconds"], "measured_encoding_seconds": encoding_seconds,
                                            "measured_receiver_model_seconds": replay["receiver_seconds"], "measured_planner_seconds": replay["planner_seconds"],
                                            "modeled_network_seconds": network_seconds, "modeled_total_seconds": measured + network_seconds,
                                            "success": replay["success"], "receiver_record_id": replay["receiver_record_id"],
                                            "execution_kind": "latency_model", "additional_model_generations": 0})
                    for cap in plan["sweep_budgets_bits"]:
                        row = {"model": model, "task": task, "family_id": fid, **stratum(family), "channel": channel,
                               "budget_bits": cap, "loss_rate": 0.0, "loss_uniform": draw, **policy(delivered, none, cap, 0, draw)}
                        sweep.append(row)
                        append_row(output / "budget_sweep_replays.jsonl", row)
        require(len(actual) == expected and counts["a_generations"] == n * 2 * len(plan["models"]), "Final actual count mismatch")
        require(len(primary) == n * len(plan["models"]) * 3 * 2 * len(plan["loss_rates"]), "Policy row count mismatch")
        require(len(sweep) == n * len(plan["models"]) * 3 * len(plan["sweep_budgets_bits"]), "Sweep row count mismatch")
        require(len(latency) == n * len(plan["models"]) * 3 * 2 * len(plan["network_latency_ms"]) * len(plan["network_bandwidth_bits_per_second"]), "Latency row count mismatch")
        write_json(output / "policy_summary.json", summarize(primary, ["model", "task", "channel", "regime", "loss_rate"], expected_by_task))
        write_json(output / "sweep_summary.json", summarize(sweep, ["model", "task", "channel", "budget_bits"], expected_by_task))
        write_json(output / "latency_model.json", latency)
        from finalize_statistics import build_statistics
        statistics_output = build_statistics(plan, list(families.values()), actual, primary, lineage=lineage, plan_hash=plan_hash)
        write_json(output / "statistics.json", statistics_output)
        write_json(output / "actual_summary.json", statistics_output["descriptive_actual"])
        write_json(output / "counts.json", {**counts, "actual_receiver_records": len(actual), "expected_receiver_records": expected,
            "actual_total_model_generations": counts["a_generations"] + counts["b_generations"],
            "policy_replay_rows": len(primary), "budget_sweep_replay_rows": len(sweep), "modeled_latency_rows": len(latency),
            "additional_generations_from_all_replays": 0, "independent_families_per_task": plan["families_per_task"],
            "all_required_stages_completed_and_sealed": True, "technical_failures_are_unavailable_not_zero": True,
            "scope": "Replay and latency rows reuse observed outputs. Families remain the sampling units; inference is within model/task."})
        write_json(output / "evaluation_provenance.json", {"plan_sha256": plan_hash, "scorer_version": SCORER_VERSION,
            "loss_draw_version": LOSS_DRAW_VERSION, "data_manifest_sha256": sha(data / "artifact_manifest.json"),
            "study_manifest_sha256": sha(study / "artifact_manifest.json"),
            "output_token_decode_independently_replayed": False,
            "numeric_validation": "Sender and B wire bytes and controls reconciled; raw quantizer replay uses implementation codec. Independent tokenizer/codec audit is separate.",
            "accounting_scope": "Complete per-message application packet only; shared assets, manifests, and transport framing excluded and separately reported."})
        stage.finish()
    except BaseException as error:
        stage.fail(error)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "data", "study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.plan, args.data, args.study, args.output)
