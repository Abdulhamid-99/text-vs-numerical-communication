"""Freeze development choices before fresh heldout creation; partition private inputs."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import secrets
import shutil
import statistics

from common import Stage, read_json, read_rows, write_json, sha, verify_seal, utc, within
from private_tasks import (QA_TASK, NAV_TASK, generate_qa_pair, generate_navigation_pair,
                          validate_split_isolation, agent_view, SCORER_VERSION, family_fingerprint, reference_answer)
from task_prompts import PROMPT_VERSION
from navigation_agent import NAVIGATION_AGENT_VERSION
from protocol_contracts import (TASKS, CONDITIONS, MODELS, LOSS_DRAW_RULE, LOSS_DRAW_DEFINITION,
    validate_protocol, validate_role_input, source_inventory, digest, safe_id, validate_prior_heldout_records)


def _commit(values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _normalized(values):
    if not isinstance(values, (list, set, tuple)) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("Malformed private exclusion values")
    return sorted({v.strip().upper() for v in values})


def actual_family_exclusions(family):
    """Derive QA identifiers/codes and NAV map/instance hashes from actual states.

    Never trust an inventory that omits a value from the private evaluator.
    Local family names may repeat in separately committed heldout attempts.
    """
    if family.get("split") != "heldout" or set(family.get("episodes", {})) != {"original", "alternate"}:
        raise ValueError("Prior input must contain complete heldout families")
    task = family["task"]
    safe_id(family["family_id"])
    derived = []
    answers = []
    for variant, episode in family["episodes"].items():
        if any(episode.get(key) != value for key, value in {"task": task, "family_id": family["family_id"],
                "split": "heldout", "variant": variant}.items()):
            raise ValueError("Prior episode identity differs from family")
        state = episode["evaluator"]
        if task == QA_TASK:
            directory, registry = state["directory"], state["registry"]
            if (not isinstance(directory, list) or not isinstance(registry, list) or
                    len(directory) != len(registry) or not 4 <= len(directory) <= 6 or
                    any(not isinstance(pair, list) or len(pair) != 2 for pair in directory + registry)):
                raise ValueError("Malformed prior QA state")
            for subject, relay in directory:
                if any(not isinstance(v, str) or re.fullmatch("[A-Z]{4}", v) is None for v in (subject, relay)):
                    raise ValueError("Malformed prior QA directory identifier")
                derived.extend((subject, relay))
            for relay, code in registry:
                if (not isinstance(relay, str) or re.fullmatch("[A-Z]{4}", relay) is None or
                        not isinstance(code, str) or re.fullmatch("[A-Z]{16}", code) is None):
                    raise ValueError("Malformed prior QA registry value")
                derived.extend((relay, code))
            if (episode["a_private_context"] != "\n".join(a + ">" + b for a, b in directory) or
                    episode["b_private_context"] != "\n".join(a + "=" + b for a, b in registry)):
                raise ValueError("Prior QA payload differs from actual evaluator")
        elif task == NAV_TASK:
            grid, start, goal = state["grid"], state["start"], state["goal"]
            if (not isinstance(grid, list) or len(grid) not in (5, 6) or
                    any(not isinstance(line, str) or len(line) != len(grid) or set(line) - {".", "#"} for line in grid) or
                    any(not isinstance(p, list) or len(p) != 2 or any(type(x) is not int or not 0 <= x < len(grid) for x in p) for p in (start, goal)) or
                    grid[start[0]][start[1]] != "." or grid[goal[0]][goal[1]] != "." or state["max_steps"] != 50):
                raise ValueError("Malformed prior navigation state")
            text = "\n".join(grid)
            if episode["a_private_context"] != text:
                raise ValueError("Prior NAV payload differs from evaluator")
            derived += ["nav_map_sha256:" + hashlib.sha256(text.encode()).hexdigest(),
                        "nav_instance_sha256:" + _commit([grid, start, goal])]
        else:
            raise ValueError("Unknown prior task")
        answers.append(reference_answer(episode))
    if agent_view(family["episodes"]["original"], "B") != agent_view(family["episodes"]["alternate"], "B"):
        raise ValueError("Prior counterpart B views differ")
    if task == QA_TASK and answers[0] == answers[1]:
        raise ValueError("Prior QA counterpart answers are identical")
    actual = _normalized(derived)
    saved = family.get("exclusion_inventory")
    if not isinstance(saved, list) or len(saved) != len(set(saved)) or _normalized(saved) != actual:
        raise ValueError("Prior exclusion inventory omits/adds private values")
    expected_private = actual if task == QA_TASK else []
    if _normalized(family.get("private_value_inventory")) != expected_private:
        raise ValueError("Prior private-value inventory differs from evaluator")
    return actual


def load_prior_heldout(directories=(), expected=None):
    """Verify exact sealed prior datasets; return public hashes and private union."""
    if not isinstance(directories, (list, tuple)):
        raise ValueError("Prior dataset paths must be a sequence")
    records, inventories, seen_paths = [], [], set()
    for directory in directories:
        directory = Path(directory)
        resolved = directory.resolve()
        if resolved in seen_paths:
            raise ValueError("Duplicate prior dataset path")
        seen_paths.add(resolved)
        state = verify_seal(directory)
        document = read_json(within(directory, "private_families.json"))
        lineage = read_json(within(directory, "private_lineage.json"))
        digest(lineage.get("plan_sha256"))
        if (state.get("phase") != "heldout_data" or set(document) != {"schema", "split", "families"} or
                document["schema"] != 1 or document["split"] != "heldout" or
                not isinstance(document["families"], list) or not document["families"] or
                state["n_requested"] != len(document["families"])):
            raise ValueError("Prior directory is not a complete sealed heldout dataset")
        families = document["families"]
        validate_split_isolation(families)
        counts = Counter(f["task"] for f in families)
        if set(counts) != set(TASKS) or len(set(counts.values())) != 1:
            raise ValueError("Prior dataset requires both task denominators")
        values = _normalized([v for family in families for v in actual_family_exclusions(family)])
        record = {"manifest_sha256": sha(directory / "artifact_manifest.json"),
            "families_sha256": sha(directory / "private_families.json"),
            "lineage_sha256": sha(directory / "private_lineage.json"), "source_plan_sha256": lineage["plan_sha256"],
            "family_fingerprints_sha256": _commit(sorted(family_fingerprint(f) for f in families)),
            "exclusions_sha256": _commit(values), "family_count": len(families), "exclusion_count": len(values)}
        records.append(record)
        inventories.append({"manifest_sha256": record["manifest_sha256"], "exclusion_values": values})
    records.sort(key=lambda r: r["manifest_sha256"])
    inventories.sort(key=lambda r: r["manifest_sha256"])
    validate_prior_heldout_records(records)
    if expected is not None:
        validate_prior_heldout_records(expected)
        if records != expected:
            raise ValueError("Missing, extra, or changed bound prior-heldout dataset")
    return records, inventories


def validate_prior_lineage(plan, lineage, families):
    expected = validate_prior_heldout_records(plan.get("prior_heldout_datasets", []))
    if lineage.get("prior_heldout_datasets", []) != expected:
        raise ValueError("Private prior-heldout commitments differ from plan")
    inventories = lineage.get("prior_heldout_inventories", [])
    if not isinstance(inventories, list) or len(inventories) != len(expected):
        raise ValueError("Missing/extra private prior exclusion inventories")
    union = set()
    for record, inventory in zip(expected, inventories):
        if (not isinstance(inventory, dict) or set(inventory) != {"manifest_sha256", "exclusion_values"} or
                inventory["manifest_sha256"] != record["manifest_sha256"]):
            raise ValueError("Private prior inventory binding differs")
        values = inventory["exclusion_values"]
        if values != _normalized(values) or len(values) != record["exclusion_count"] or _commit(values) != record["exclusions_sha256"]:
            raise ValueError("Private prior exclusions differ from frozen commitment")
        union.update(values)
    if expected:
        current = {v for f in families for v in actual_family_exclusions(f)}
        if current & union or type(lineage.get("prior_heldout_overlap_count")) is not int or lineage["prior_heldout_overlap_count"] != 0:
            raise ValueError("Current heldout reuses prior private values/map identities")
        if type(lineage.get("prior_heldout_exclusion_count")) is not int or lineage["prior_heldout_exclusion_count"] != len(union):
            raise ValueError("Wrong prior exclusion union count")
    return union

def unique_rows(rows, label):
    result = {}
    for row in rows:
        identity = safe_id(row.get("id"))
        if identity in result:
            raise ValueError("Duplicate " + label + " identity")
        result[identity] = row
    return result


def validate_heldout_data(plan_path, data, plan=None):
    plan = validate_protocol(plan_path) if plan is None else plan
    state = verify_seal(data)
    count = plan["families_per_task"] * len(TASKS)
    if state.get("phase") != "heldout_data" or state["n_requested"] != count:
        raise ValueError("Wrong heldout data stage/count")
    lineage = read_json(data / "private_lineage.json")
    if (lineage.get("plan_sha256") != sha(plan_path)
            or lineage.get("exclusion_lineage_sha256") != plan["exclusion_lineage_sha256"]):
        raise ValueError("Heldout values/lineage differ from frozen plan")
    document = read_json(data / "private_families.json")
    if set(document) != {"schema", "split", "families"} or document["schema"] != 1 or document["split"] != "heldout":
        raise ValueError("Invalid heldout family document")
    families = document["families"]
    if len(families) != count or Counter(f["task"] for f in families) != Counter({task: plan["families_per_task"] for task in TASKS}):
        raise ValueError("Wrong exact family denominator by task")
    validate_split_isolation(families)
    if any(f["split"] != "heldout" for f in families):
        raise ValueError("Non-heldout family in confirmatory input")
    validate_prior_lineage(plan, lineage, families)
    aitems, bitems = [], []
    for family in families:
        safe_id(family["family_id"])
        for variant in ("original", "alternate"):
            aitems.append({"id": family["family_id"] + "_" + variant,
                "family_id": family["family_id"], "variant": variant, "task": family["task"],
                "view": agent_view(family["episodes"][variant], "A")})
        if agent_view(family["episodes"]["original"], "B") != agent_view(family["episodes"]["alternate"], "B"):
            raise ValueError("Paired family must preserve B's private view")
        bitems.append({"family_id": family["family_id"], "task": family["task"],
            "view": agent_view(family["episodes"]["original"], "B")})
    expected_a = {"schema": 1, "role": "A", "split": "heldout", "plan_sha256": sha(plan_path), "items": aitems}
    if read_json(data / "A" / "input.json") != expected_a:
        raise ValueError("A partition differs from actual heldout families")
    validate_role_input(expected_a, plan, sha(plan_path), "A", model="qwen", input_root=data / "A")
    if read_json(data / "private_B_views.json") != {"schema": 1, "role": "B", "items": bitems}:
        raise ValueError("B partition differs from actual heldout families")
    return families, bitems


def freeze(code_root, development, probes, pcas, image_id, output, families_per_task=64, prior_heldout=()):
    if type(families_per_task) is not int or families_per_task < 1:
        raise ValueError("Positive heldout family count required")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ValueError("Immutable Docker image ID required")
    digest(image_id[7:])
    verify_seal(development)
    prior_records, _ = load_prior_heldout(prior_heldout)
    output.mkdir(parents=True, exist_ok=False)
    assets_root = output / "assets"
    assets_root.mkdir()
    models = {}
    for model in ("qwen", "llama"):
        verify_seal(probes[model])
        verify_seal(pcas[model])
        source = read_json(probes[model] / "configuration.json")
        pca = read_json(pcas[model] / "configuration.json")
        if source["model"] != model or pca["model"] != model or source["prompt_version"] != PROMPT_VERSION or pca["prompt_version"] != PROMPT_VERSION:
            raise ValueError("Development model/prompt provenance mismatch")
        if (source["data_manifest_sha256"] != sha(development / "artifact_manifest.json")
                or pca["data_manifest_sha256"] != source["data_manifest_sha256"]
                or pca["config_sha256"] != source["config_sha256"] or pca["model_config"] != source["model_config"]):
            raise ValueError("Development/PCA model and data binding differs")
        if pca["source_probe_manifest_sha256"] != sha(probes[model] / "artifact_manifest.json") or pca["layer"] != source["layer"]:
            raise ValueError("PCA is not bound to selected development source")
        encodings = read_rows(probes[model] / "encodings.jsonl")
        development_families = read_json(development / "development.json")["families"]
        calibration_families = read_json(development / "calibration.json")["families"]
        validate_split_isolation(calibration_families + development_families)
        families = {f["family_id"]: f for f in development_families}
        cells = [(row["family_id"], row["variant"]) for row in encodings]
        selected = {family for family, _variant in cells}
        if (len(cells) != len(set(cells)) or not selected or not selected <= set(families)
                or set(cells) != {(family, variant) for family in selected for variant in ("original", "alternate", "oracle")}
                or {families[family]["task"] for family in selected} != set(TASKS)):
            raise ValueError("Development source inventory is incomplete or duplicated")
        selected_counts = Counter(families[family]["task"] for family in selected)
        if len(set(selected_counts.values())) != 1:
            raise ValueError("Selected development requires equal positive family counts per task")
        for row in encodings:
            expected_file = row["family_id"] + "_" + row["variant"] + ".bin"
            if row.get("packet") != expected_file:
                raise ValueError("Development packet identity differs from source cell")
            packet_file = within(probes[model], "raw/" + expected_file)
            trace = row["trace"]
            if (trace.get("codec") != "int4" or trace.get("layer") != source["layer"]
                    or type(trace.get("wire_bits")) is not int or trace["wire_bits"] <= 0
                    or packet_file.stat().st_size * 8 != trace["wire_bits"] or sha(packet_file) != trace.get("packet_sha256")):
                raise ValueError("Baseline development cost must equal actual complete raw-int4 packet bytes")
        tasks = {}
        for task in TASKS:
            bits = [row["trace"]["wire_bits"] for row in encodings if row["variant"] == "original" and families[row["family_id"]]["task"] == task]
            if not bits or any(type(b) is not int or b <= 0 for b in bits):
                raise ValueError("Missing actual development baseline packet costs")
            baseline = int(statistics.mean(bits))
            basis = pcas[model] / (task + "_basis.npz")
            descriptor = pcas[model] / (task + "_descriptor.json")
            description = read_json(descriptor)
            expected_calibration = [family["family_id"] for family in calibration_families if family["task"] == task]
            if (not expected_calibration or description.get("calibration_family_ids") != expected_calibration
                    or description.get("calibration_data_sha256") != sha(development / "calibration.json")):
                raise ValueError("PCA basis does not bind the exact independent task calibration families")
            prefix = model + "_" + task
            for origin, target_name in ((basis, prefix + "_basis.npz"), (descriptor, prefix + "_descriptor.json")):
                shutil.copyfile(origin, assets_root / target_name)
            tasks[task] = {"basis_file": prefix + "_basis.npz", "basis_sha256": sha(basis),
                "descriptor_file": prefix + "_descriptor.json", "descriptor_sha256": sha(descriptor),
                "shared_setup_bytes": basis.stat().st_size + descriptor.stat().st_size,
                "baseline_bits": baseline, "low_budget_bits": baseline // 4,
                "baseline_development_bits": bits,
                "baseline_rule": "Floor of mean complete raw-int4 generated-payload bits on selected development families; common to all channels"}
        models[model] = {"config": source["model_config"], "generation": source["generation"],
            "allowed_device_placements": ["cuda:0"],
            "layer": source["layer"], "raw_codec": "int4", "pca_codec": pca["codec"],
            "pca_rank": pca["rank"], "tasks": tasks,
            "development_manifest_sha256": sha(probes[model] / "artifact_manifest.json"),
            "pca_development_manifest_sha256": sha(pcas[model] / "artifact_manifest.json")}
    plan = {"schema": 1, "status": "frozen", "frozen_utc": utc(), "heldout_created": False,
        "families_per_task": families_per_task, "tasks": list(TASKS), "models": models,
        "conditions": CONDITIONS, "prompt_version": PROMPT_VERSION, "scorer_version": SCORER_VERSION,
        "navigation_agent_version": NAVIGATION_AGENT_VERSION,
        "loss_draw_rule": LOSS_DRAW_RULE, "loss_draw_definition": LOSS_DRAW_DEFINITION,
        "code_sha256": source_inventory(code_root),
        "image_id": image_id, "development_data_manifest_sha256": sha(development / "artifact_manifest.json"),
        "exclusion_lineage_sha256": sha(development / "lineage.json"),
        "scope": "Synthetic private two-document joins and tool-assisted nonadaptive cooperative grids; B's LLM reconstructs a map and a public deterministic BFS plans only on that predicted map. Environment scoring uses the true private map separately. Same-checkpoint pairs.",
        "primary_regimes": ["high", "low"], "loss_rates": [0.0, 0.05, 0.10],
        "sweep_budgets_bits": [0,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576],
        "network_latency_ms": [0,50,100], "network_bandwidth_bits_per_second": [10000,100000,1000000],
        "budget_rule": "Atomic one-packet admission, full application bytes; oversize means zero sent bits and no message; equality admitted",
        "loss_rule": "One paired SHA256-seeded Bernoulli mask per family, shared across models/channels; 5/10 percent nested thresholds; sent lost packets still charged",
        "replay_rule": "Actual greedy receiver calls evaluated once per unique condition. Budget/loss policies deterministically reuse the exact delivered-condition or no_comm record ID. Derived rows are not extra generations, experiments, or independent observations.",
        "failure_rule": "Technical/model runtime exceptions abort and leave unsealed stage. Completed sender output outside 1..128 payload tokens is an explicit source_contract_failure, never silently truncated; affected delivery uses observed no-message output and reports flag.",
        "statistics": "Family is unit; Wilson95 marginal intervals; paired bootstrap differences; 16 primary exact McNemar comparisons with global Holm. Secondary causal/loss/sweep analysis descriptive. No a priori power claim.",
        "latency_scope": "Measured synchronized model/encoding seconds; simulated serialization/propagation only. No measured network or deployment p95 claim.",
        "heldout_tuning": "Prohibited; any implementation failure preserves evidence and requires explicit deviation and fresh heldout families"}
    if prior_records:
        plan["prior_heldout_datasets"] = prior_records
    write_json(output / "protocol.json", plan)
    write_json(output / "freeze_receipt.json", {"protocol_sha256": sha(output / "protocol.json"), "frozen_utc": plan["frozen_utc"],
        "heldout_existed_at_freeze": False, "assets": {p.name: sha(p) for p in assets_root.iterdir()}})
    validate_protocol(output / "protocol.json", code_root=code_root)
    return plan


def heldout(plan_path, development, output, prior_heldout=()):
    plan = validate_protocol(plan_path)
    verify_seal(development)
    if (sha(development / "lineage.json") != plan["exclusion_lineage_sha256"]
            or sha(development / "artifact_manifest.json") != plan["development_data_manifest_sha256"]):
        raise ValueError("Exclusion lineage changed after freeze")
    prior_records, prior_inventories = load_prior_heldout(prior_heldout, plan.get("prior_heldout_datasets", []))
    stage = Stage(output, "heldout_data", requested=plan["families_per_task"] * 2)
    try:
        seed = secrets.token_hex(32)
        lineage = read_json(development / "lineage.json")
        excluded = set(_normalized(lineage["all_exclusions"]))
        prior_values = {v for inventory in prior_inventories for v in inventory["exclusion_values"]}
        excluded.update(prior_values)
        initial_exclusions = sorted(excluded)
        families = []
        for task, generator, name in ((QA_TASK, generate_qa_pair, "qa"), (NAV_TASK, generate_navigation_pair, "navigation")):
            for index in range(plan["families_per_task"]):
                kwargs = {"rows": 4 + index % 3} if task == QA_TASK else {"size": 5 + index % 2}
                family = generator(seed, split="heldout", family_id=f"heldout_{name}_{index:05d}",
                                   excluded_values=excluded, **kwargs)
                derived = actual_family_exclusions(family) if prior_records else _normalized(family["exclusion_inventory"])
                if set(derived) & excluded:
                    raise ValueError("Generated family violates initial/previous-family exclusion union")
                excluded.update(derived)
                families.append(family)
                stage.update(n_completed=len(families))
        old = read_json(development / "calibration.json")["families"] + read_json(development / "development.json")["families"]
        isolation = validate_split_isolation(old + families)
        write_json(output / "private_families.json", {"schema": 1, "split": "heldout", "families": families})
        private_lineage = {"plan_sha256": sha(plan_path), "created_utc": utc(),
            "seed": seed, "exclusion_lineage_sha256": plan["exclusion_lineage_sha256"],
            "initial_exclusion_count": len(initial_exclusions), "isolation": isolation,
            "loss_seed": secrets.token_hex(32), "bootstrap_seed": 20260912}
        if prior_records:
            private_lineage.update(prior_heldout_datasets=prior_records,
                prior_heldout_inventories=prior_inventories,
                prior_heldout_exclusion_count=len(prior_values), prior_heldout_overlap_count=0,
                initial_exclusions_sha256=_commit(initial_exclusions))
        validate_prior_lineage(plan, private_lineage, families)
        # Revalidate prior seals after generation to detect concurrent changes.
        load_prior_heldout(prior_heldout, prior_records)
        write_json(output / "private_lineage.json", private_lineage)
        aitems, bitems = [], []
        for family in families:
            for variant in ("original", "alternate"):
                episode = family["episodes"][variant]
                aitems.append({"id": family["family_id"] + "_" + variant, "family_id": family["family_id"],
                    "variant": variant, "task": family["task"], "view": agent_view(episode, "A")})
            bitems.append({"family_id": family["family_id"], "task": family["task"],
                           "view": agent_view(family["episodes"]["original"], "B")})
        write_json(output / "A" / "input.json", {"schema": 1, "role": "A", "split": "heldout", "plan_sha256": sha(plan_path), "items": aitems})
        write_json(output / "private_B_views.json", {"schema": 1, "role": "B", "items": bitems})
        stage.finish()
        validate_heldout_data(plan_path, output, plan)
    except BaseException as error:
        stage.fail(error)
        raise


def prepare_receiver(plan_path, heldout_data, sender_output, output, model, channel):
    plan = validate_protocol(plan_path)
    if model not in MODELS or channel not in CONDITIONS:
        raise ValueError("Unknown model/channel")
    families, bitems = validate_heldout_data(plan_path, heldout_data, plan)
    state = verify_seal(sender_output)
    if (state.get("model") != model or state.get("phase") != "heldout_sender"
            or state["n_requested"] != 2 * len(families) or state.get("a_generations_completed") != 2 * len(families)):
        raise ValueError("Wrong sender output identity")
    identity = read_json(sender_output / "identity.json")
    if (identity.get("plan_sha256") != sha(plan_path) or identity.get("input_sha256") != sha(heldout_data / "A" / "input.json")
            or identity.get("model") != model or identity.get("settings") != plan["models"][model]):
        raise ValueError("Sender evidence is not bound to this model/plan/private input")
    sources = unique_rows(read_rows(sender_output / "encodings.jsonl"), "encoding")
    senders = unique_rows(read_rows(sender_output / "senders.jsonl"), "sender")
    aitems = {item["id"]: item for item in read_json(heldout_data / "A" / "input.json")["items"]}
    expected_sources = {identity + "_" + kind for identity, item in aitems.items()
                        for kind in (("generated", "oracle") if item["variant"] == "original" else ("generated",))}
    if set(senders) != set(aitems) or set(sources) != expected_sources:
        raise ValueError("Missing/extra sender or encoding counterpart")
    captures = 0
    for source_id, source in sources.items():
        kind = "oracle" if source_id.endswith("_oracle") else "generated"
        base = source_id.rsplit("_", 1)[0]
        item, generated = aitems[base], senders[base]
        if any(generated.get(key) != item[key] for key in ("id", "family_id", "task", "variant")):
            raise ValueError("Sender trace identity differs from actual A input")
        if any(source.get(key) != expected for key, expected in {
                "source_id": base, "family_id": item["family_id"], "task": item["task"],
                "variant": item["variant"], "kind": kind}.items()):
            raise ValueError("Encoding identity differs from actual source")
        trace = generated.get("trace", {})
        text = trace.get("text")
        if (not isinstance(text, str) or not isinstance(trace.get("output_ids"), list) or not trace["output_ids"]
                or type(generated.get("payload_token_count")) is not int or generated["payload_token_count"] < 0):
            raise ValueError("Sender trace is not a completed observed generation")
        valid_generated = 1 <= generated["payload_token_count"] <= 128
        status = "valid" if valid_generated else "source_contract_failure"
        if generated.get("source_status") != status or generated.get("source_exact") != (text == item["view"]["private_context"]):
            raise ValueError("Sender source-contract diagnostics differ from observations")
        expected_status = "valid" if kind == "oracle" else status
        if source.get("source_status") != expected_status:
            raise ValueError("Encoding source status differs from actual generation")
        if expected_status != "valid":
            if source.get("packets") != {} or any(key in source for key in ("encoding", "array_file", "array_file_sha256")):
                raise ValueError("Failed source must not carry fabricated encoding evidence")
            continue
        captures += 1
        expected_channels = {"text", "raw", "pca"} | ({"raw_zero", "pca_mean"} if kind == "generated" and item["variant"] == "original" else set())
        if set(source.get("packets", {})) != expected_channels:
            raise ValueError("Incomplete source/control channel packet set")
        array_name = "sources/" + source_id + ".npy"
        if source.get("array_file") != array_name or sha(within(sender_output, array_name)) != source.get("array_file_sha256"):
            raise ValueError("Source numeric array identity differs")
        for key, packet in source["packets"].items():
            expected_name = "packets/" + source_id + "_" + key + ".bin"
            if not isinstance(packet, dict) or set(packet) != {"file", "sha256", "wire_bits"} or packet["file"] != expected_name:
                raise ValueError("Unexpected source packet descriptor")
            path = within(sender_output, expected_name)
            if (type(packet["wire_bits"]) is not int or packet["wire_bits"] <= 0
                    or sha(path) != packet["sha256"] or path.stat().st_size * 8 != packet["wire_bits"]):
                raise ValueError("Source packet bytes/cost differ from actual file")
        source_text = within(sender_output, source["packets"]["text"]["file"]).read_bytes().decode("utf-8", errors="strict")
        if source_text != (item["view"]["private_context"] if kind == "oracle" else text):
            raise ValueError("Text packet differs from actual generated/oracle A message")
    if state.get("encoding_forwards_completed") != captures:
        raise ValueError("Source forward count differs from actual encoding inventory")
    output.mkdir(parents=True, exist_ok=False)
    (output / "packets").mkdir()
    items = []
    for b in bitems:
        for condition in plan["conditions"][channel]:
            variant = "alternate" if condition.endswith("swapped") else "original"
            kind = "oracle" if condition.endswith("oracle") else "generated"
            source = sources[b["family_id"] + "_" + variant + "_" + kind]
            packet = None
            status = "not_applicable" if condition == "no_comm" else source["source_status"]
            if condition != "no_comm" and status == "valid":
                key = condition if condition in ("raw_zero", "pca_mean") else channel
                old = source["packets"][key]
                origin = within(sender_output, old["file"])
                if sha(origin) != old["sha256"] or origin.stat().st_size * 8 != old["wire_bits"]:
                    raise ValueError("Sender packet changed during partitioning")
                name = b["family_id"] + "_" + condition + ".bin"
                shutil.copyfile(origin, output / "packets" / name)
                packet = {**old, "file": "packets/" + name}
            items.append({"id": model + "_" + b["family_id"] + "_" + condition, **b,
                          "condition": condition, "packet": packet, "source_status": status})
    write_json(output / "input.json", {"schema": 1, "role": "B", "split": "heldout", "plan_sha256": sha(plan_path),
                                     "channel": channel, "items": items})
    validate_role_input(read_json(output / "input.json"), plan, sha(plan_path), "B", model=model, channel=channel, input_root=output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    for arg in ("code-root", "development", "qwen-probe", "llama-probe", "qwen-pca", "llama-pca", "output"):
        f.add_argument("--" + arg, type=Path, required=True)
    f.add_argument("--image-id", required=True)
    f.add_argument("--families-per-task", type=int, default=64)
    f.add_argument("--prior-heldout", type=Path, action="append", default=[])
    h = sub.add_parser("heldout")
    for arg in ("plan", "development", "output"):
        h.add_argument("--" + arg, type=Path, required=True)
    h.add_argument("--prior-heldout", type=Path, action="append", default=[])
    a = p.parse_args()
    if a.command == "freeze":
        freeze(a.code_root, a.development, {"qwen": a.qwen_probe, "llama": a.llama_probe},
               {"qwen": a.qwen_pca, "llama": a.llama_pca}, a.image_id, a.output, a.families_per_task, a.prior_heldout)
    else:
        heldout(a.plan, a.development, a.output, a.prior_heldout)
