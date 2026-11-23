"""Stdlib-only frozen-plan and role-input contracts shared by host and worker.

No model loading, evaluation data imports, or inferred task answers occur here.
"""
from collections import Counter, defaultdict
from pathlib import Path
import os
import re

from common import read_json, sha, file_inventory, within, _reject_link_components
from private_tasks import QA_TASK, NAV_TASK, QA_PUBLIC, NAV_PUBLIC, SCORER_VERSION
from task_prompts import PROMPT_VERSION
from navigation_agent import NAVIGATION_AGENT_VERSION

TASKS = (QA_TASK, NAV_TASK)
MODELS = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct"}
CONDITIONS = {"text": ["text_generated", "text_oracle", "text_swapped"],
              "raw": ["raw_generated", "raw_oracle", "raw_swapped", "raw_zero", "no_comm"],
              "pca": ["pca_generated", "pca_oracle", "pca_swapped", "pca_mean"]}
LOSS_DRAW_RULE = "sha256_first53_utf8_seed_pipe_family_div_2pow53"
LOSS_DRAW_DEFINITION = "Take the first 53 most significant SHA256 bits of UTF-8(loss_seed + '|' + family_id), then divide by 2**53."


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")


def safe_id(value):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) is None:
        raise ValueError("Unsafe or missing family/item identifier")
    return value


def digest(value, length=64):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value) is None:
        raise ValueError("Invalid immutable SHA identity")
    return value


def validate_prior_heldout_records(records):
    """Public recovery commitments contain only immutable hashes and counts."""
    if not isinstance(records, list):
        raise ValueError("Prior-heldout commitments must be a list")
    hashes = {"manifest_sha256", "families_sha256", "lineage_sha256", "source_plan_sha256",
              "family_fingerprints_sha256", "exclusions_sha256"}
    counts = {"family_count", "exclusion_count"}
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != hashes | counts:
            raise ValueError("Prior-heldout commitments permit only hashes/counts")
        for name in hashes:
            digest(record[name])
        for name in counts:
            positive_integer(record[name], name)
        if record["manifest_sha256"] in seen:
            raise ValueError("Duplicate prior-heldout manifest")
        seen.add(record["manifest_sha256"])
    if records != sorted(records, key=lambda r: r["manifest_sha256"]):
        raise ValueError("Prior-heldout commitments must have canonical manifest order")
    return records


def source_inventory(code_root):
    root = Path(code_root)
    _reject_link_components(root)
    actual = {}
    # Never traverse probe/data directories: the image contains code only.
    _reject_link_components(root / "src")
    _reject_link_components(root / "thesis_study")
    for base, directories, files in os.walk(root / "src", followlinks=False):
        for name in directories:
            _reject_link_components(Path(base) / name)
        for name in files:
            if name.endswith(".py"):
                path = Path(base) / name
                relative = path.relative_to(root).as_posix()
                actual[relative] = sha(within(root, relative))
    for path in (root / "thesis_study").iterdir():
        if path.name.endswith(".py"):
            relative = path.relative_to(root).as_posix()
            actual[relative] = sha(within(root, relative))
    if not actual or not any(name.startswith("src/") for name in actual) or "thesis_study/study_worker.py" not in actual:
        raise ValueError("Incomplete runtime source tree")
    return dict(sorted(actual.items()))


def boundary_inventory(root):
    """Unlike sealed-stage inventories, an input boundary has no exempt file."""
    result = file_inventory(root)
    manifest = Path(root) / "artifact_manifest.json"
    if manifest.exists() or manifest.is_symlink():
        path = within(root, "artifact_manifest.json")
        result["artifact_manifest.json"] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return dict(sorted(result.items()))


def validate_plan_document(plan):
    if (type(plan.get("schema")) is not int or plan["schema"] != 1 or plan.get("status") != "frozen"
            or plan.get("heldout_created") is not False or plan.get("tasks") != list(TASKS)
            or plan.get("conditions") != CONDITIONS or set(plan.get("models", {})) != set(MODELS)):
        raise ValueError("Invalid frozen task/model/condition inventory")
    positive_integer(plan.get("families_per_task"), "families_per_task")
    if plan.get("prompt_version") != PROMPT_VERSION or plan.get("scorer_version") != SCORER_VERSION:
        raise ValueError("Frozen prompt/scorer version differs")
    if plan.get("navigation_agent_version") != NAVIGATION_AGENT_VERSION:
        raise ValueError("Frozen public navigation-agent version differs")
    if plan.get("loss_draw_rule") != LOSS_DRAW_RULE or plan.get("loss_draw_definition") != LOSS_DRAW_DEFINITION:
        raise ValueError("Frozen loss-draw algorithm differs")
    if not isinstance(plan.get("image_id"), str) or re.fullmatch(r"sha256:[0-9a-f]{64}", plan["image_id"]) is None:
        raise ValueError("Immutable image SHA required")
    if not isinstance(plan.get("code_sha256"), dict) or not plan["code_sha256"]:
        raise ValueError("Missing complete frozen code inventory")
    for name, expected in plan["code_sha256"].items():
        if not isinstance(name, str) or re.fullmatch(r"(?:src/[A-Za-z0-9_./-]+|thesis_study/[A-Za-z0-9_.-]+)\.py", name) is None or ".." in name.split("/"):
            raise ValueError("Unsafe or out-of-scope frozen code path")
        digest(expected)
    for field in ("development_data_manifest_sha256", "exclusion_lineage_sha256"):
        digest(plan.get(field))
    validate_prior_heldout_records(plan.get("prior_heldout_datasets", []))
    asset_names = set()
    for model, settings in plan["models"].items():
        for section in ("model", "tokenizer"):
            config = settings.get("config", {}).get(section, {})
            if config.get("name") != MODELS[model]:
                raise ValueError("Frozen model/tokenizer identity mismatch")
            digest(config.get("revision"), 40)
        if settings["config"]["model"].get("quantization") != "int8" or settings["config"]["model"].get("device") != "cuda:0":
            raise ValueError("Frozen study requires the declared int8 CUDA-0 checkpoint policy")
        if settings.get("allowed_device_placements") != ["cuda:0"]:
            raise ValueError("Offload policy must explicitly require CUDA-0")
        if type(settings.get("layer")) is not int or settings["layer"] < 0 or settings.get("raw_codec") != "int4":
            raise ValueError("Invalid frozen layer/raw codec")
        positive_integer(settings.get("pca_rank"), "pca_rank")
        if settings.get("pca_codec") not in ("int4", "int8") or set(settings.get("tasks", {})) != set(TASKS):
            raise ValueError("Invalid frozen PCA/task settings")
        generation = settings.get("generation", {})
        positive_integer(generation.get("max_new_tokens"), "max_new_tokens")
        if generation.get("do_sample") is not False or generation.get("temperature") != 0:
            raise ValueError("Exact-input replay requires the declared greedy generation policy")
        for record in settings["tasks"].values():
            positive_integer(record.get("baseline_bits"), "baseline_bits")
            if type(record.get("low_budget_bits")) is not int or record["low_budget_bits"] != record["baseline_bits"] // 4:
                raise ValueError("Low budget differs from frozen baseline rule")
            positive_integer(record.get("shared_setup_bytes"), "shared_setup_bytes")
            for kind, suffix in (("basis", ".npz"), ("descriptor", ".json")):
                name = record.get(kind + "_file")
                if not isinstance(name, str) or Path(name).name != name or any(c in name for c in "\\/:") or not name.endswith(suffix) or name in asset_names:
                    raise ValueError("Unsafe or duplicate public asset filename")
                asset_names.add(name)
                digest(record.get(kind + "_sha256"))
    return asset_names


def validate_protocol(plan_path, *, code_root=None):
    path = Path(plan_path)
    if path.name != "protocol.json":
        raise ValueError("Frozen protocol must be named protocol.json")
    plan = read_json(path)
    asset_names = validate_plan_document(plan)
    inventory = boundary_inventory(path.parent)
    allowed = {"protocol.json", "freeze_receipt.json"} | {"assets/" + name for name in asset_names}
    if set(inventory) != allowed:
        raise ValueError("Public plan directory contains missing/unexpected files; private data must never be mounted here")
    # Empty extra directories also violate the fixed public mount boundary.
    directories = {p.relative_to(path.parent).as_posix() for p in path.parent.rglob("*") if p.is_dir()}
    if directories != {"assets"}:
        raise ValueError("Unexpected directory inside public plan boundary")
    receipt = read_json(path.parent / "freeze_receipt.json")
    expected_assets = {name: inventory["assets/" + name]["sha256"] for name in asset_names}
    if (receipt.get("protocol_sha256") != sha(path) or receipt.get("assets") != expected_assets
            or receipt.get("heldout_existed_at_freeze") is not False or receipt.get("frozen_utc") != plan.get("frozen_utc")):
        raise ValueError("Protocol/asset bytes differ from freeze receipt")
    for model, settings in plan["models"].items():
        for task, record in settings["tasks"].items():
            for kind in ("basis", "descriptor"):
                relative = "assets/" + record[kind + "_file"]
                if inventory[relative]["sha256"] != record[kind + "_sha256"]:
                    raise ValueError("Public asset differs from plan")
            descriptor = read_json(path.parent / "assets" / record["descriptor_file"])
            expected = {"model": model, "task": task, "layer": settings["layer"],
                        "rank": settings["pca_rank"], "codec": settings["pca_codec"],
                        "basis_sha256": record["basis_sha256"], "heldout_seen": False}
            if any(descriptor.get(key) != value for key, value in expected.items()):
                raise ValueError("PCA descriptor has incompatible calibration/model semantics")
            if record["shared_setup_bytes"] != sum(inventory["assets/" + record[kind + "_file"]]["bytes"] for kind in ("basis", "descriptor")):
                raise ValueError("Shared PCA setup byte accounting differs")
    if code_root is not None and source_inventory(code_root) != plan["code_sha256"]:
        raise ValueError("Complete runtime code inventory differs from frozen plan")
    return plan


def _view(view, task, role):
    keys = {"task", "public_instruction", "private_context"} | ({"query"} if role == "B" else set())
    public = QA_PUBLIC if task == QA_TASK else NAV_PUBLIC if task == NAV_TASK else None
    if (not isinstance(view, dict) or set(view) != keys or public is None or view["task"] != task
            or view["public_instruction"] != public[role] or any(not isinstance(value, str) or not value.strip() for value in view.values())):
        raise ValueError("Unexpected fields/instructions at private role boundary")


def validate_role_input(document, plan, plan_sha, role, *, model, channel=None, input_root=None):
    keys = {"schema", "role", "split", "plan_sha256", "items"} | ({"channel"} if role == "B" else set())
    if (set(document) != keys or type(document.get("schema")) is not int or document["schema"] != 1
            or document.get("role") != role or document.get("split") != "heldout" or document.get("plan_sha256") != plan_sha
            or not isinstance(document.get("items"), list) or model not in plan["models"]):
        raise ValueError("Wrong role input schema/plan/model binding")
    if role == "B" and (channel not in CONDITIONS or document.get("channel") != channel):
        raise ValueError("Receiver channel differs from plan")
    tasks, count = set(plan["tasks"]), plan["families_per_task"]
    labels = ("original", "alternate") if role == "A" else plan["conditions"][channel]
    items = document["items"]
    if len(items) != count * len(tasks) * len(labels):
        raise ValueError("Wrong role item denominator")
    cells, identities, family_tasks, family_views = set(), set(), {}, {}
    expected_files = {"input.json"}
    for item in items:
        expected = {"id", "family_id", "variant", "task", "view"} if role == "A" else {"id", "family_id", "task", "condition", "view", "packet", "source_status"}
        if set(item) != expected:
            raise ValueError("Unexpected item fields")
        family, task = safe_id(item["family_id"]), item["task"]
        label = item["variant" if role == "A" else "condition"]
        identity = family + "_" + label if role == "A" else model + "_" + family + "_" + label
        if task not in tasks or label not in labels or item["id"] != identity or identity in identities:
            raise ValueError("Invalid/duplicate role identity or condition")
        _view(item["view"], task, role)
        identities.add(identity)
        cell = (task, family, label)
        if cell in cells or (family in family_tasks and family_tasks[family] != task):
            raise ValueError("Duplicate role cell or family reused across tasks")
        cells.add(cell)
        family_tasks[family] = task
        if role == "B":
            if family in family_views and family_views[family] != item["view"]:
                raise ValueError("B context must be identical across this family's conditions")
            family_views[family] = item["view"]
            descriptor, status = item["packet"], item["source_status"]
            if label == "no_comm":
                if descriptor is not None or status != "not_applicable":
                    raise ValueError("No-communication control must not contain a packet")
            elif status == "source_contract_failure":
                if descriptor is not None or label.endswith("oracle"):
                    raise ValueError("Invalid source-failure packet/oracle semantics")
            elif status == "valid":
                if not isinstance(descriptor, dict) or set(descriptor) != {"file", "sha256", "wire_bits"}:
                    raise ValueError("Valid source must have exactly one complete packet")
                positive_integer(descriptor["wire_bits"], "wire_bits")
                digest(descriptor["sha256"])
                expected_name = "packets/" + family + "_" + label + ".bin"
                if descriptor["file"] != expected_name:
                    raise ValueError("Packet filename differs from exact role identity")
                expected_files.add(expected_name)
                if input_root is not None:
                    path = within(input_root, expected_name)
                    if sha(path) != descriptor["sha256"] or path.stat().st_size * 8 != descriptor["wire_bits"]:
                        raise ValueError("Actual packet bytes differ from admitted descriptor")
            else:
                raise ValueError("Unrecognized source status cannot become a task failure")
    counts = Counter(family_tasks.values())
    if counts != Counter({task: count for task in tasks}):
        raise ValueError("Wrong exact family count per task")
    if cells != {(task, family, label) for family, task in family_tasks.items() for label in labels}:
        raise ValueError("Missing counterpart or condition cell")
    if input_root is not None and set(boundary_inventory(input_root)) != expected_files:
        raise ValueError("Unexpected/missing files in private role mount")
    return items
