"""Isolated numeric relay study: prepare -> send -> receive(group) -> analyze.

prepare creates NEW private evaluation data and must be called only after the
experiment owner authorizes the frozen design. Numeric receivers must mount
only their public plan/config/prompt/jobs and numeric packet directory; text
payloads, sender traces, source arrays and targets belong to other processes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import re
import secrets
import string
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from budgetfair.comm.tensor_packet import decode_packet, encode_tensor
from budgetfair.comm.pca_packet import (FRAME as PCA_FRAME, load_asset, encode_projected,
    encode_coefficients, reconstruct)
from budgetfair.eval.stats import clopper_pearson_interval, paired_binary_test, holm_bonferroni
from budgetfair.model.client import GenerationConfig, LLMClient
from budgetfair.model.span_transport import (PublicPrompt, encode_message, make_public_prompt,
    neutral_generation_kwargs, receive_packet, receive_text)
from budgetfair.model.pca_transport import receive_projected_packet
from budgetfair.run_state import atomic_json, source_identity, utc_now

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen", "llama")
PRIMARY = ("text_generated", "vector_generated", "vector_oracle", "vector_swapped", "vector_zero", "no_comm")
ENGINEERING = ("vector_dropped", "vector_budget544")
PCA_PRIMARY = ("vector_pca_generated", "vector_pca_oracle", "vector_pca_swapped", "vector_pca_mean")
CONDITIONS = PRIMARY + ENGINEERING
JOB_FIELDS = {"job_id", "case_id", "condition", "group", "filename", "packet_sha256", "packet_bits",
              "sent_bits", "delivered_bits", "delivery_status", "source_condition"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # Reject NaN/Infinity anywhere in the evidence document.
    json.dumps(value, allow_nan=False)
    return value


def append_json(path, value):
    import os
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    json.dumps(rows, allow_nan=False)
    return rows


def tests_identity():
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted((ROOT / "tests").rglob("*.py"))}


def safe_file(folder, name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
        raise ValueError("Unsafe artifact filename")
    path = Path(folder) / name
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(Path(folder).resolve()):
        raise ValueError("Artifact must be a regular file inside its declared directory")
    return path


class Stage:
    def __init__(self, directory, phase, **fields):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "run_status.json"
        self.data = {"schema_version": 1, "phase": phase, "status": "running", "stage": "preflight",
                     "started_utc": utc_now(), "finished_utc": None, "n_requested": None,
                     "n_completed": 0, "error": None, **fields}
        self.update()

    def update(self, **fields):
        self.data.update(fields)
        atomic_json(self.path, self.data)

    def fail(self, error):
        self.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    finished_utc=utc_now(), error={"type": type(error).__name__, "message": str(error),
                                                 "traceback": traceback.format_exc()})

    def complete(self):
        self.update(status="completed", stage="finished", finished_utc=utc_now())


def seal(directory):
    directory = Path(directory)
    files = {path.relative_to(directory).as_posix(): sha(path) for path in sorted(directory.rglob("*"))
             if path.is_file() and path != directory / "artifact_manifest.json"}
    atomic_json(directory / "artifact_manifest.json", {"schema_version": 1, "files": files})


def verify_seal(directory):
    directory = Path(directory)
    document = read_json(directory / "artifact_manifest.json")
    expected = document["files"]
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path != directory / "artifact_manifest.json"}
    if actual != set(expected):
        raise ValueError("Sealed artifact file inventory differs")
    for relative, digest in expected.items():
        path = directory / relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or sha(path) != digest:
            raise ValueError("Sealed artifact hash/path differs")


def validate_cases(document, count):
    cases = document["cases"]
    if len(cases) != count:
        raise ValueError("Private case count differs from the frozen plan")
    seen = set()
    for index, case in enumerate(cases):
        if set(case) != {"case_id", "target", "alternate"} or case["case_id"] != f"case_{index:06d}":
            raise ValueError("Invalid, duplicate or reordered case identity")
        for key in ("target", "alternate"):
            value = case[key]
            if not isinstance(value, str) or not re.fullmatch("[A-Z]{16}", value) or value in seen:
                raise ValueError("Private values must be disjoint 16-letter uppercase strings")
            seen.add(value)
    return cases


def model_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    LLMClient._loader_identity(config)
    model = config["model"]
    if not re.fullmatch("[0-9a-f]{40}", str(model.get("revision", ""))):
        raise ValueError("Model revision must be a full pinned 40-character commit")
    tokenizer = config.get("tokenizer", {}) or {}
    if tokenizer.get("name", model["name"]) != model["name"] or tokenizer.get("revision", model["revision"]) != model["revision"]:
        raise ValueError("Tokenizer must match the pinned model")
    return {key: value for key, value in config.items() if key in {"model", "tokenizer"}} | {
        "generation": {"max_new_tokens": 48, "do_sample": False, "temperature": 0.0, "top_p": 1.0}}


def calibration_values_digest(values):
    """Portable ordered calibration identity: UTF8 compact JSON, no newline."""
    return bytes_sha(json.dumps(values, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def validate_lineage(document, config, asset):
    values = document.get("calibration_values")
    if (not isinstance(values, list) or not values or
            any(not isinstance(value, str) or not re.fullmatch("[A-Z]{16}", value) for value in values) or
            len(set(values)) != len(values)):
        raise ValueError("PCA lineage requires unique explicit 16-uppercase calibration values")
    expected = {"model_name": config["model"]["name"], "model_revision": config["model"]["revision"],
                "layer": asset.layer, "asset_sha256": asset.fingerprint,
                "calibration_values_sha256": calibration_values_digest(values)}
    if any(document.get(key) != value for key, value in expected.items()) or type(document.get("layer")) is not int:
        raise ValueError("PCA lineage model/revision/layer/asset/calibration identity differs")
    return values


def validate_policy(plan, pca_enabled):
    count = plan["count"]
    expected = {"bootstrap_seed": 20260912, "bootstrap_resamples": 10000, "budget_control_bits": 544,
                "generation": {"max_new_tokens": 48, "do_sample": False, "neutral_checkpoint_defaults": True},
                "normalization": "strip().upper(); no punctuation removal or substring extraction",
                "partial_metric": "1 - unit-cost character Levenshtein distance / max(normalized lengths)"}
    for key, value in expected.items():
        if type(plan.get(key)) is not type(value) or plan[key] != value:
            raise ValueError("Frozen statistical/generation policy differs")
    raw = plan["scientific_gate"]
    raw_expected = {"positive_required": math.ceil(0.95 * count), "negative_maximum": 0,
                    "positive_original_conditions": ["vector_generated", "vector_oracle"],
                    "alternate_condition": "vector_swapped",
                    "zero_original_conditions": ["vector_zero", "no_comm", *ENGINEERING]}
    if any(type(raw.get(key)) is not type(value) or raw[key] != value for key, value in raw_expected.items()):
        raise ValueError("Frozen scientific gate differs")
    pca_expected = {"positive_required": math.ceil(0.95 * count), "negative_maximum": 0,
                    "positive_original_conditions": ["vector_pca_generated", "vector_pca_oracle"],
                    "alternate_condition": "vector_pca_swapped", "zero_original_conditions": ["vector_pca_mean"]}
    if plan.get("pca_success_gate") != (pca_expected if pca_enabled else None):
        raise ValueError("Frozen PCA scientific gate differs")


def verify_prepared(study):
    """Check the prepare-only snapshot; later sender directories may be added."""
    study = Path(study)
    public = study / "public"
    plan = read_json(study / "plan.json")
    status = read_json(study / "run_status.json")
    receipt = read_json(public / "prepare_receipt.json")
    manifest = read_json(study / "prepare_manifest.json")
    if (status["phase"] != "prepare" or status["status"] != "completed" or
            any(type(status.get(key)) is not int or status[key] != plan["count"] for key in ("n_requested", "n_completed")) or
            status["plan_sha256"] != sha(study / "plan.json") or
            sha(public / "plan.json") != status["plan_sha256"] or
            receipt["prepare_status_sha256"] != sha(study / "run_status.json") or
            receipt["prepare_manifest_sha256"] != sha(study / "prepare_manifest.json")):
        raise ValueError("Completed prepare identity/status differs")
    expected = {"run_status.json", "plan.json", "private_cases.json", "forbidden_values.json", "public/plan.json"}
    for model, record in plan["models"].items():
        expected.add("public/model_configs/" + record["filename"])
        if "pca" in record:
            expected.add("public/assets/" + record["pca"]["filename"])
            lineage = record["pca"]["calibration_lineage"]
            if lineage is not None:
                expected.add("calibration_lineage/" + lineage["private_copy"])
    if set(manifest["files"]) != expected:
        raise ValueError("Frozen prepare artifact inventory differs")
    for relative, digest in manifest["files"].items():
        path = study / relative
        if path.is_symlink() or not path.resolve().is_relative_to(study.resolve()) or sha(path) != digest:
            raise ValueError("Frozen prepare artifact hash/path differs")
    return plan


def prepare(study, forbidden_values, configs, *, purpose, count=64, layers=None, codecs=None, models=MODELS,
            pca_assets=None, pca_lineage=None, pca_dim=512, pca_codecs=None):
    """Explicitly authorized fresh-data creation; never called by other phases."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 4096:
        raise ValueError("count must be an integer in 1..4096")
    if purpose not in {"development", "heldout"} or (purpose == "heldout" and count != 64):
        raise ValueError("purpose must be development or heldout; heldout requires exactly64 cases")
    if not models or len(set(models)) != len(models) or not set(models) <= set(MODELS):
        raise ValueError("Select a nonempty unique subset of qwen/llama")
    pca_assets, pca_lineage = pca_assets or {}, pca_lineage or {}
    if pca_assets and set(pca_assets) != set(models):
        raise ValueError("Optional PCA requires an asset for every selected model or none")
    if isinstance(pca_dim, bool) or not isinstance(pca_dim, int) or pca_dim < 1:
        raise ValueError("PCA dimension must be a positive integer")
    layers = layers or {model: 4 for model in MODELS}
    codecs = codecs or {model: "int4" for model in MODELS}
    pca_codecs = pca_codecs or {model: codecs[model] for model in models}
    if not set(models) <= set(configs) or not set(models) <= set(layers) or not set(models) <= set(codecs):
        raise ValueError("Every selected pinned model requires configuration and transport choices")
    for model in models:
        if isinstance(layers[model], bool) or not isinstance(layers[model], int) or layers[model] < -1:
            raise ValueError("Invalid chosen model layer")
        if codecs[model] not in {"float32", "int8", "int4"}:
            raise ValueError("Unsupported chosen numeric codec")
    forbidden = read_json(forbidden_values)
    if isinstance(forbidden, dict):
        forbidden = forbidden["values"]
    if not isinstance(forbidden, list) or any(not isinstance(value, str) or not value for value in forbidden):
        raise ValueError("forbidden-values must be a JSON string list or {values: [...]} object")
    frozen_configs = {model: model_config(configs[model]) for model in models}
    assets, calibration_values = {}, set()
    for model, path in pca_assets.items():
        if pca_codecs.get(model) not in {"float32", "int8", "int4"}:
            raise ValueError("Unsupported chosen PCA coefficient codec")
        asset = load_asset(path, expected_sha256=sha(path))
        if asset.layer != layers[model] or pca_dim > asset.components.shape[0]:
            raise ValueError("PCA layer/rank differs from the selected model protocol")
        if purpose == "heldout" and model not in pca_lineage:
            raise ValueError("Heldout PCA requires frozen calibration lineage metadata")
        if model in pca_lineage:
            calibration_values.update(validate_lineage(read_json(pca_lineage[model]), frozen_configs[model], asset))
        assets[model] = asset
    stage = Stage(study, "prepare", purpose=purpose)
    try:
        study = Path(study)
        public = study / "public"
        (public / "model_configs").mkdir(parents=True)
        if assets:
            (public / "assets").mkdir()
            (study / "calibration_lineage").mkdir()
        (study / "forbidden_values.json").write_bytes(Path(forbidden_values).read_bytes())
        rng, used = secrets.SystemRandom(), {value.strip().upper() for value in forbidden} | calibration_values
        all_exclusions = sorted(used)
        def draw():
            for _ in range(100000):
                value = "".join(rng.choice(string.ascii_uppercase) for _ in range(16))
                if value not in used:
                    used.add(value)
                    return value
            raise RuntimeError("Unable to draw a new disjoint private value")
        cases = {"schema_version": 1, "cases": [{"case_id": f"case_{i:06d}", "target": draw(), "alternate": draw()} for i in range(count)]}
        validate_cases(cases, count)
        atomic_json(study / "private_cases.json", cases)
        config_records = {}
        for model in models:
            path = public / "model_configs" / f"{model}.yaml"
            path.write_text(yaml.safe_dump(frozen_configs[model], sort_keys=False), encoding="utf-8")
            config_records[model] = {"filename": path.name, "sha256": sha(path), "original_sha256": sha(configs[model]),
                                     "layer": layers[model], "codec": codecs[model]}
            if model in assets:
                asset = assets[model]
                target = public / "assets" / f"{model}_pca.npz"
                target.write_bytes(Path(pca_assets[model]).read_bytes())
                lineage = None
                if model in pca_lineage:
                    private_lineage = study / "calibration_lineage" / f"{model}.json"
                    private_lineage.write_bytes(Path(pca_lineage[model]).read_bytes())
                    lineage = {"source_path": str(Path(pca_lineage[model]).resolve()), "private_copy": f"{model}.json",
                               "sha256": sha(private_lineage)}
                config_records[model]["pca"] = {"filename": target.name, "sha256": sha(target),
                    "setup_bytes": asset.setup_bytes, "layer": asset.layer, "rank": asset.components.shape[0],
                    "hidden_size": asset.mean.shape[0], "dimensions": pca_dim, "codec": pca_codecs[model],
                    "calibration_lineage": lineage}
        plan = {"schema_version": 1, "created_utc": utc_now(), "status": "frozen", "count": count, "purpose": purpose, "model_order": list(models),
                "engineering_count": min(8, count), "case_ids": [case["case_id"] for case in cases["cases"]],
                "models": config_records, "private_cases_sha256": sha(study / "private_cases.json"),
                "forbidden_values_sha256": sha(forbidden_values), "forbidden_values_count": len(forbidden),
                "calibration_values_count": len(calibration_values), "effective_exclusions_count": len(all_exclusions),
                "effective_exclusions_sha256": calibration_values_digest(all_exclusions),
                "source_identity": source_identity(ROOT), "test_files": tests_identity(),
                "primary_conditions": list(PRIMARY + (PCA_PRIMARY if assets else ())), "engineering_conditions": list(ENGINEERING),
                "generation": {"max_new_tokens": 48, "do_sample": False, "neutral_checkpoint_defaults": True},
                "empty_sender_rule": "raw decoded output with strip()=='' is observed_empty; it has no packet",
                "engineering_source": "oracle packet isolates loss and admission from sender errors",
                "budget_control_bits": 544, "numeric_cost": "all serialized application packet bytes times8",
                "text_cost": "all transmitted UTF8 bytes times8; no claim to include TCP/IP framing",
                "normalization": "strip().upper(); no punctuation removal or substring extraction",
                "partial_metric": "1 - unit-cost character Levenshtein distance / max(normalized lengths)",
                "primary_contrast": "vector_generated minus no_comm separately per model; Holm across models",
                "secondary_contrast": "text_generated minus vector_generated separately per model",
                "bootstrap_seed": 20260912, "bootstrap_resamples": 10000,
                "scientific_gate": {"positive_required": math.ceil(0.95 * count), "negative_maximum": 0,
                    "positive_original_conditions": ["vector_generated", "vector_oracle"],
                    "alternate_condition": "vector_swapped", "zero_original_conditions": ["vector_zero", "no_comm", *ENGINEERING],
                    "interpretation": "Observed sample working criterion, not a confidence guarantee that population accuracy is at least95%; false leaves execution completed."},
                "pca_success_gate": {"positive_required": math.ceil(0.95 * count), "negative_maximum": 0,
                    "positive_original_conditions": ["vector_pca_generated", "vector_pca_oracle"],
                    "alternate_condition": "vector_pca_swapped", "zero_original_conditions": ["vector_pca_mean"]} if assets else None,
                "primary_pvalue_family": "All raw and enabled PCA generated-minus-no_comm contrasts across selected models; one Holm family",
                "isolation": "numeric and text receiver groups run in different processes; only selected group's packet directory mounted"}
        atomic_json(study / "plan.json", plan)
        atomic_json(public / "plan.json", plan)
        stage.update(n_requested=count, n_completed=count, plan_sha256=sha(study / "plan.json"))
        stage.complete()
        files = {path.relative_to(study).as_posix(): sha(path) for path in sorted(study.rglob("*")) if path.is_file()}
        atomic_json(study / "prepare_manifest.json", {"schema_version": 1, "files": files})
        atomic_json(public / "prepare_receipt.json", {"schema_version": 1, "status": "completed",
            "plan_sha256": sha(public / "plan.json"), "prepare_status_sha256": sha(study / "run_status.json"),
            "prepare_manifest_sha256": sha(study / "prepare_manifest.json")})
        return plan
    except BaseException as error:
        stage.fail(error)
        raise


def verify_plan(public, *, check_tests=False):
    public = Path(public)
    plan = read_json(public / "plan.json")
    receipt = read_json(public / "prepare_receipt.json")
    if receipt["schema_version"] != 1 or receipt["status"] != "completed" or receipt["plan_sha256"] != sha(public / "plan.json"):
        raise ValueError("Public plan differs from completed prepare receipt")
    if plan["schema_version"] != 1 or plan["status"] != "frozen" or not set(plan["models"]) <= set(MODELS) or not plan["models"]:
        raise ValueError("Unsupported frozen plan")
    if plan["model_order"] != list(plan["models"]) or plan["purpose"] not in {"development", "heldout"}:
        raise ValueError("Plan model order/purpose differs")
    if type(plan["count"]) is not int or not 1 <= plan["count"] <= 4096 or (plan["purpose"] == "heldout" and plan["count"] != 64) or plan["engineering_count"] != min(8, plan["count"]):
        raise ValueError("Plan episode counts differ from declared purpose")
    pca_enabled = all("pca" in record for record in plan["models"].values())
    if any("pca" in record for record in plan["models"].values()) != pca_enabled:
        raise ValueError("PCA assets must cover every selected model or none")
    validate_policy(plan, pca_enabled)
    if plan["primary_conditions"] != list(PRIMARY + (PCA_PRIMARY if pca_enabled else ())) or plan["engineering_conditions"] != list(ENGINEERING):
        raise ValueError("Frozen conditions changed")
    if plan["case_ids"] != [f"case_{i:06d}" for i in range(plan["count"])]:
        raise ValueError("Plan case identities differ")
    if source_identity(ROOT)["source_files"] != plan["source_identity"]["source_files"]:
        raise ValueError("Production source changed after plan freeze")
    if check_tests and tests_identity() != plan["test_files"]:
        raise ValueError("Test source changed after plan freeze")
    for model, record in plan["models"].items():
        path = safe_file(public / "model_configs", record["filename"])
        if sha(path) != record["sha256"]:
            raise ValueError("Pinned model configuration changed")
        if "pca" in record:
            info = record["pca"]
            asset_path = safe_file(public / "assets", info["filename"])
            if sha(asset_path) != info["sha256"] or asset_path.stat().st_size != info["setup_bytes"]:
                raise ValueError("Frozen PCA asset hash/size changed")
    return plan


def public_asset(public, plan, model):
    info = plan["models"][model].get("pca")
    if info is None:
        return None
    asset = load_asset(safe_file(Path(public) / "assets", info["filename"]), expected_sha256=info["sha256"])
    if (asset.layer != info["layer"] or asset.layer != plan["models"][model]["layer"] or
            asset.components.shape != (info["rank"], info["hidden_size"]) or not 1 <= info["dimensions"] <= info["rank"]):
        raise ValueError("PCA asset metadata differs from frozen model choice")
    return asset


def load_client(public, model, plan):
    client = LLMClient.from_config(safe_file(Path(public) / "model_configs", plan["models"][model]["filename"]))
    client.generation = GenerationConfig(max_new_tokens=48, do_sample=False, temperature=0.0, top_p=1.0)
    return client


def client_metadata(client, config_path):
    import platform
    import transformers
    kwargs = neutral_generation_kwargs(client)
    effective = kwargs["generation_config"].to_dict()
    effective.update({key: value for key, value in kwargs.items() if key not in {"generation_config", "logits_processor"}})
    effective["logits_processors"] = [type(value).__module__ + "." + type(value).__name__ for value in kwargs.get("logits_processor", [])]
    config = client.model.config
    document = {"model_config_file_sha256": sha(config_path), "full_input_model_config": yaml.safe_load(Path(config_path).read_text()),
                "loaded_model_config": config.to_dict() if hasattr(config, "to_dict") else vars(config),
                "effective_generation_config": effective, "loader_identity": getattr(client, "loader_config", None),
                "loader_fingerprint": getattr(client, "loader_config_fingerprint", None),
                "quantization": getattr(client, "quantization", None), "torch_dtype": getattr(client, "torch_dtype", None),
                "requested_device": getattr(client, "device", None), "hf_device_map": getattr(client, "hf_device_map", None),
                "torch_version": torch.__version__, "transformers_version": transformers.__version__, "python": platform.python_version(),
                "inference_backend": type(client).__module__ + "." + type(client).__name__}
    return json.loads(json.dumps(document, default=str, allow_nan=False))


@torch.inference_mode()
def generate_sender(client, value):
    messages = [{"role": "system", "content": "Copy the private record value exactly as one continuous 16-character uppercase string. Do not insert spaces or separators between letters. Output only the unchanged value, without labels, punctuation, quotes, or explanation."},
                {"role": "user", "content": f"Private lookup record:\nItem: ALPHA\nValue: {value}"}]
    inputs = client._chat_inputs(messages)
    output = client.model.generate(**inputs, **neutral_generation_kwargs(client))
    count = inputs.input_ids.shape[1]
    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] <= count:
        raise ValueError("Sender generation returned no valid continuation")
    ids = output[0, count:].tolist()
    text = client.tokenizer.decode(ids, skip_special_tokens=True)
    return {"messages": messages, "input_ids": inputs.input_ids[0].tolist(), "output_ids": ids, "text": text}


def expected_jobs(plan):
    return [(case, condition) for i, case in enumerate(plan["case_ids"])
            for condition in tuple(plan["primary_conditions"]) + (ENGINEERING if i < plan["engineering_count"] else ())]


def send(study, model):
    if model not in MODELS:
        raise ValueError("Unknown model")
    study = Path(study)
    stage = Stage(study / "sender" / model, "send", model=model)
    try:
        public = study / "public"
        plan = verify_plan(public, check_tests=True)
        verify_prepared(study)
        if sha(study / "plan.json") != sha(public / "plan.json") or sha(study / "private_cases.json") != plan["private_cases_sha256"]:
            raise ValueError("Private plan/cases changed")
        cases = validate_cases(read_json(study / "private_cases.json"), plan["count"])
        asset = public_asset(public, plan, model)
        packet_root = public / "receiver_inputs" / model
        packet_root.mkdir(parents=True, exist_ok=False)
        for group in ("numeric", "text"):
            (packet_root / group).mkdir()
        (stage.directory / "raw").mkdir()
        (stage.directory / "payloads").mkdir()
        stage.update(n_requested=len(expected_jobs(plan)), stage="model_loading", plan_sha256=sha(public / "plan.json"))
        client = load_client(public, model, plan)
        if asset is not None and asset.mean.shape != (client.model.config.hidden_size,):
            raise ValueError("PCA basis width differs from loaded sender model")
        atomic_json(stage.directory / "model_metadata.json", client_metadata(client, public / "model_configs" / plan["models"][model]["filename"]))
        prompt = make_public_prompt(client)
        prompt_document = {"prompt": asdict(prompt), "hidden_size": client.model.config.hidden_size}
        atomic_json(packet_root / "public_prompt.json", prompt_document)
        choice = plan["models"][model]
        layer, codec = choice["layer"], choice["codec"]
        role = "token_embeddings" if layer == -1 else "hidden_states"
        jobs, private_jobs = [], []

        def encode_source(case_id, kind, message):
            if not message.strip():
                append_json(stage.directory / "sources.jsonl", {"case_id": case_id, "source": kind,
                    "status": "observed_empty", "raw_path": None, "packet_sha256": None})
                return None, None
            raw_packet, capture = encode_message(client, prompt, message, layer=layer, codec="float32")
            array = decode_packet(raw_packet).array
            raw_path = stage.directory / "raw" / f"{case_id}_{kind}.npy"
            with raw_path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            packet = encode_tensor(array, codec=codec, role=role, layer_index=None if layer == -1 else layer)
            append_json(stage.directory / "sources.jsonl", {"case_id": case_id, "source": kind, "status": "encoded",
                "capture": capture, "raw_path": raw_path.relative_to(stage.directory).as_posix(), "raw_file_sha256": sha(raw_path),
                "raw_capture_packet_sha256": bytes_sha(raw_packet), "actual_packet_sha256": bytes_sha(packet),
                "actual_packet_bits": len(packet) * 8})
            return packet, array

        for index, case in enumerate(cases):
            stage.update(stage="sender_generation", case_id=case["case_id"])
            generated = {}
            for kind, value in (("generated", case["target"]), ("alternate", case["alternate"])):
                trace = generate_sender(client, value)
                if not isinstance(trace["text"], str) or not trace["output_ids"]:
                    raise ValueError("Invalid actual sender generation trace")
                generated[kind] = trace["text"]
                append_json(stage.directory / "sender_trace.jsonl", {"case_id": case["case_id"], "source": kind,
                    "observed_empty": not trace["text"].strip(), **trace})
            vector, raw = encode_source(case["case_id"], "generated", generated["generated"])
            swapped, alternate_raw = encode_source(case["case_id"], "alternate", generated["alternate"])
            oracle, oracle_raw = encode_source(case["case_id"], "oracle", case["target"])
            zero = None if raw is None else encode_tensor(np.zeros_like(raw), codec=codec, role=role,
                                                          layer_index=None if layer == -1 else layer)
            payloads = {"text_generated": generated["generated"].encode("utf-8") if generated["generated"].strip() else None,
                        "vector_generated": vector, "vector_oracle": oracle, "vector_swapped": swapped,
                        "vector_zero": zero, "no_comm": None, "vector_dropped": oracle, "vector_budget544": oracle}
            if asset is not None:
                dimensions = choice["pca"]["dimensions"]
                pca_codec = choice["pca"]["codec"]
                projected = lambda array: None if array is None else encode_projected(array, asset, dimensions=dimensions, codec=pca_codec)
                payloads.update(vector_pca_generated=projected(raw), vector_pca_oracle=projected(oracle_raw),
                    vector_pca_swapped=projected(alternate_raw), vector_pca_mean=None if raw is None else
                    encode_coefficients(np.zeros((raw.shape[0], dimensions), dtype=np.float32), asset, codec=pca_codec))
            conditions = tuple(plan["primary_conditions"]) + (ENGINEERING if index < plan["engineering_count"] else ())
            for condition in conditions:
                payload = payloads[condition]
                group = "text" if condition == "text_generated" else "numeric"
                status = "no_comm" if condition == "no_comm" else "observed_empty" if payload is None else "delivered"
                bits = 0 if payload is None else len(payload) * 8
                if condition == "vector_dropped":
                    status = "dropped"
                if condition == "vector_budget544":
                    if bits <= 544:
                        raise ValueError("Chosen numeric packet does not exercise the blocked544 control")
                    status = "blocked"
                filename = f"{case['case_id']}_{condition}." + ("utf8" if group == "text" else "bin")
                private_file = None
                if payload is not None:
                    private_file = "payloads/" + filename
                    with (stage.directory / private_file).open("xb") as stream:
                        stream.write(payload)
                    if status == "delivered":
                        with (packet_root / group / filename).open("xb") as stream:
                            stream.write(payload)
                source = "none" if condition == "no_comm" else "alternate" if condition in {"vector_swapped", "vector_pca_swapped"} else "zero_coefficients" if condition == "vector_pca_mean" else "zero" if condition == "vector_zero" else "oracle" if condition in {"vector_oracle", "vector_pca_oracle", *ENGINEERING} else "generated"
                job = {"job_id": f"{case['case_id']}_{condition}", "case_id": case["case_id"], "condition": condition,
                       "group": group, "filename": filename if status == "delivered" else None,
                       "packet_sha256": None if payload is None else bytes_sha(payload), "packet_bits": bits,
                       "sent_bits": bits if status in {"delivered", "dropped"} else 0,
                       "delivered_bits": bits if status == "delivered" else 0,
                       "delivery_status": status, "source_condition": source}
                jobs.append(job)
                private_jobs.append({**job, "private_payload": private_file})
                stage.update(n_completed=len(jobs), stage="packets", case_id=case["case_id"])
        document = {"schema_version": 1, "model": model, "plan_sha256": sha(public / "plan.json"),
                    "public_prompt_sha256": sha(packet_root / "public_prompt.json"), "jobs": jobs}
        atomic_json(packet_root / "jobs.json", document)
        atomic_json(stage.directory / "private_jobs.json", {"jobs": private_jobs})
        verify_plan(public, check_tests=True)
        verify_prepared(study)
        stage.update(jobs_sha256=sha(packet_root / "jobs.json"), inference_backend=type(client).__module__ + "." + type(client).__name__)
        stage.complete()
        seal(stage.directory)
        atomic_json(packet_root / "ready.json", {"status": "completed", "model": model,
            "plan_sha256": sha(public / "plan.json"), "jobs_sha256": sha(packet_root / "jobs.json"),
            "sender_manifest_sha256": sha(stage.directory / "artifact_manifest.json")})
        return document
    except BaseException as error:
        stage.fail(error)
        raise


def verify_jobs(public, model, plan, group, *, asset=None):
    root = Path(public) / "receiver_inputs" / model
    ready, document = read_json(root / "ready.json"), read_json(root / "jobs.json")
    if ready["status"] != "completed" or ready["model"] != model or document["model"] != model:
        raise ValueError("Sender public manifest is incomplete or belongs to another model")
    if ready["jobs_sha256"] != sha(root / "jobs.json") or document["plan_sha256"] != sha(Path(public) / "plan.json") or ready["plan_sha256"] != document["plan_sha256"]:
        raise ValueError("Public jobs/plan hash differs")
    if sha(root / "public_prompt.json") != document["public_prompt_sha256"]:
        raise ValueError("Public prompt hash differs")
    prompt_document = read_json(root / "public_prompt.json")
    width = prompt_document["hidden_size"]
    if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= 8192:
        raise ValueError("Invalid declared model width")
    prompt = PublicPrompt(**prompt_document["prompt"])
    if "pca" in plan["models"][model]:
        asset = asset or public_asset(public, plan, model)
        if asset.mean.shape != (width,):
            raise ValueError("Public PCA basis and declared model width differ")
    if [(job["case_id"], job["condition"]) for job in document["jobs"]] != expected_jobs(plan):
        raise ValueError("Missing, duplicated, extra or reordered planned jobs")
    for job in document["jobs"]:
        if set(job) != JOB_FIELDS or job["job_id"] != f"{job['case_id']}_{job['condition']}":
            raise ValueError("Unexpected public job fields or identity")
        expected_group = "text" if job["condition"] == "text_generated" else "numeric"
        if job["group"] != expected_group:
            raise ValueError("Incorrect receiver group")
        for key in ("packet_bits", "sent_bits", "delivered_bits"):
            if type(job[key]) is not int or job[key] < 0 or job[key] % 8:
                raise ValueError("Packet costs must be nonnegative whole bytes")
        status, bits = job["delivery_status"], job["packet_bits"]
        if status not in {"delivered", "dropped", "blocked", "no_comm", "observed_empty"}:
            raise ValueError("Unknown delivery status")
        if status in {"no_comm", "observed_empty"} and (bits or job["packet_sha256"] is not None):
            raise ValueError("No-packet condition has a fabricated packet")
        if status in {"delivered", "dropped", "blocked"} and (not bits or not re.fullmatch("[0-9a-f]{64}", str(job["packet_sha256"]))):
            raise ValueError("Encoded packet identity is missing")
        if job["sent_bits"] != (bits if status in {"delivered", "dropped"} else 0) or job["delivered_bits"] != (bits if status == "delivered" else 0):
            raise ValueError("Attempted/sent/delivered costs do not reconcile")
        if (status == "delivered") != (job["filename"] is not None):
            raise ValueError("Only delivered packets may be mounted for receivers")
        expected_status = {"no_comm": "no_comm", "vector_dropped": "dropped", "vector_budget544": "blocked"}.get(job["condition"])
        if expected_status is not None and status != expected_status:
            raise ValueError("Engineering/no-communication intervention differs")
        if expected_status is None and status not in {"delivered", "observed_empty"}:
            raise ValueError("Primary condition has an unexpected loss/admission status")
        if status == "blocked" and bits <= 544:
            raise ValueError("Blocked control packet would fit its budget")
        if job["condition"] in {"vector_oracle", "vector_pca_oracle"} and status != "delivered":
            raise ValueError("Oracle diagnostic packet is missing")
    chosen = [job for job in document["jobs"] if job["group"] == group]
    directory = root / group
    expected_files = {job["filename"] for job in chosen if job["filename"] is not None}
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != expected_files:
        raise ValueError("Selected receiver packet file inventory differs")
    for job in chosen:
        if job["filename"] is None:
            continue
        path = safe_file(directory, job["filename"])
        if sha(path) != job["packet_sha256"] or path.stat().st_size * 8 != job["packet_bits"]:
            raise ValueError("Delivered packet hash/size differs")
        packet = path.read_bytes()
        if group == "numeric":
            if job["condition"] in PCA_PRIMARY:
                if asset is None:
                    raise ValueError("PCA job lacks a frozen public basis")
                info = plan["models"][model]["pca"]
                hidden, metadata = reconstruct(packet, asset, expected_dimensions=info["dimensions"], expected_codec=info["codec"])
                if hidden.shape[1] != width or hidden.shape[0] > prompt.max_payload_tokens or metadata["wire_bits"] != job["packet_bits"]:
                    raise ValueError("Compressed packet shape/cost differs")
                if job["condition"] == "vector_pca_mean" and np.any(decode_packet(packet[PCA_FRAME.size:]).array != 0):
                    raise ValueError("Mean control must contain zero coefficients")
                continue
            decoded = decode_packet(packet, max_elements=128 * width)
            layer = plan["models"][model]["layer"]
            if decoded.codec != plan["models"][model]["codec"] or decoded.role != ("token_embeddings" if layer == -1 else "hidden_states") or decoded.layer_index != (None if layer == -1 else layer):
                raise ValueError("Numeric packet codec/role/layer differs")
            if decoded.array.ndim != 2 or decoded.array.shape[1] != width or decoded.array.shape[0] > prompt.max_payload_tokens:
                raise ValueError("Numeric span shape differs")
            if job["condition"] == "vector_zero" and np.any(decoded.array != 0):
                raise ValueError("Zero-vector control contains nonzero values")
        else:
            if not packet.decode("utf-8", errors="strict").strip():
                raise ValueError("Observed-empty text must not masquerade as a delivered packet")
    return chosen, prompt, document


def validate_result(result, job):
    if not isinstance(result, dict) or result.get("status") != "completed" or not isinstance(result.get("answer"), str):
        raise ValueError("Receiver produced no valid completed answer")
    for key in ("output_ids", "receiver_input_ids"):
        ids = result.get(key)
        if not isinstance(ids, list) or not ids or any(type(value) is not int or value < 0 for value in ids):
            raise ValueError("Receiver returned invalid actual token IDs/continuation")
    if type(result.get("wire_bits")) is not int or result["wire_bits"] != job["delivered_bits"]:
        raise ValueError("Receiver runtime cost differs from delivered packet")
    expected_sha = job["packet_sha256"] if job["delivery_status"] == "delivered" else None
    if result.get("packet_sha256") != expected_sha:
        raise ValueError("Receiver consumed packet identity differs")
    expected_injections = 1 if job["delivery_status"] == "delivered" and job["group"] == "numeric" else 0
    if not isinstance(result.get("injections"), list) or len(result["injections"]) != expected_injections:
        raise ValueError("Receiver injection count differs from delivery")
    json.dumps(result, allow_nan=False)


def receive(public, output, model, group):
    if model not in MODELS or group not in {"numeric", "text"}:
        raise ValueError("Specify one model and numeric|text receiver group")
    public = Path(public)
    stage = Stage(Path(output) / model / group, "receive", model=model, group=group)
    try:
        plan = verify_plan(public)
        asset = public_asset(public, plan, model)
        jobs, prompt, document = verify_jobs(public, model, plan, group, asset=asset)
        stage.update(stage="model_loading", n_requested=len(jobs), plan_sha256=sha(public / "plan.json"),
                     jobs_sha256=sha(public / "receiver_inputs" / model / "jobs.json"))
        client = load_client(public, model, plan)
        if asset is not None and asset.mean.shape != (client.model.config.hidden_size,):
            raise ValueError("PCA basis width differs from loaded receiver model")
        atomic_json(stage.directory / "model_metadata.json", client_metadata(client, public / "model_configs" / plan["models"][model]["filename"]))
        if asdict(make_public_prompt(client)) != asdict(prompt):
            # JSON arrays become lists; normalize only representation, not values.
            if json.dumps(asdict(make_public_prompt(client))) != json.dumps(asdict(prompt)):
                raise ValueError("Receiver tokenizer/public prompt differs from sender")
        row_path = stage.directory / "rows.jsonl"
        row_path.touch(exist_ok=False)
        for job in jobs:
            stage.update(stage="receiver_generation", job_id=job["job_id"])
            packet = None if job["filename"] is None else safe_file(public / "receiver_inputs" / model / group, job["filename"]).read_bytes()
            if packet is not None and bytes_sha(packet) != job["packet_sha256"]:
                raise ValueError("Packet changed after preflight")
            if packet is None:
                result = receive_packet(client, prompt, None, expected_layer=plan["models"][model]["layer"])
            elif group == "text":
                result = receive_text(client, prompt, packet)
            elif job["condition"] in PCA_PRIMARY:
                info = plan["models"][model]["pca"]
                result = receive_projected_packet(client, prompt, packet, asset,
                    expected_layer=plan["models"][model]["layer"], expected_dimensions=info["dimensions"], expected_codec=info["codec"])
            else:
                result = receive_packet(client, prompt, packet, expected_layer=plan["models"][model]["layer"])
            validate_result(result, job)
            append_json(row_path, {"job_id": job["job_id"], "case_id": job["case_id"], "condition": job["condition"],
                "model": model, "group": group, "execution_status": "completed", "delivery_status": job["delivery_status"],
                "packet_bits": job["packet_bits"], "sent_bits": job["sent_bits"], "delivered_bits": job["delivered_bits"],
                "source_packet_sha256": job["packet_sha256"], "result": result})
            stage.update(n_completed=stage.data["n_completed"] + 1)
        verify_plan(public)
        verify_jobs(public, model, plan, group, asset=asset)
        stage.update(inference_backend=type(client).__module__ + "." + type(client).__name__)
        stage.complete()
        seal(stage.directory)
        return stage.data
    except BaseException as error:
        stage.fail(error)
        raise


def string_scores(answer, target):
    if not isinstance(answer, str) or not isinstance(target, str) or not target.strip():
        raise ValueError("Scoring requires an actual string answer and nonempty private target")
    predicted, gold = answer.strip().upper(), target.strip().upper()
    previous = list(range(len(gold) + 1))
    for i, left in enumerate(predicted, 1):
        current = [i]
        for j, right in enumerate(gold, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
        previous = current
    return {"exact_match": predicted == gold, "raw_exact_match": answer == target,
            "edit_similarity": 1 - previous[-1] / max(len(predicted), len(gold)),
            "empty": not predicted, "unknown": predicted == "UNKNOWN"}


def analyze(study, receipts, output):
    study, receipts = Path(study), Path(receipts)
    stage = Stage(output, "analyze")
    try:
        public = study / "public"
        plan = verify_plan(public, check_tests=True)
        verify_prepared(study)
        if sha(study / "private_cases.json") != plan["private_cases_sha256"] or sha(study / "plan.json") != sha(public / "plan.json"):
            raise ValueError("Private evaluation cases/plan changed")
        cases = {case["case_id"]: case for case in validate_cases(read_json(study / "private_cases.json"), plan["count"])}
        metrics, summaries, contrasts = [], {}, {}
        for model in plan["model_order"]:
            asset = public_asset(public, plan, model)
            lineage = plan["models"][model].get("pca", {}).get("calibration_lineage")
            if lineage is not None and sha(safe_file(study / "calibration_lineage", lineage["private_copy"])) != lineage["sha256"]:
                raise ValueError("Private frozen PCA calibration lineage changed")
            sender = study / "sender" / model
            verify_seal(sender)
            sender_status = read_json(sender / "run_status.json")
            root = public / "receiver_inputs" / model
            if sender_status["status"] != "completed" or read_json(root / "ready.json")["sender_manifest_sha256"] != sha(sender / "artifact_manifest.json"):
                raise ValueError("Sender stage failed or evidence differs")
            if any(type(sender_status.get(key)) is not int or sender_status[key] != len(expected_jobs(plan)) for key in ("n_requested", "n_completed")):
                raise ValueError("Sender stage counters differ from complete planned jobs")
            private_jobs = read_json(sender / "private_jobs.json")["jobs"]
            all_jobs = read_json(root / "jobs.json")["jobs"]
            if [{key: row[key] for key in JOB_FIELDS} for row in private_jobs] != all_jobs:
                raise ValueError("Private source ledger differs from public jobs")
            for job in private_jobs:
                if job["private_payload"] is not None:
                    path = sender / job["private_payload"]
                    if not path.resolve().is_relative_to(sender.resolve()) or sha(path) != job["packet_sha256"] or path.stat().st_size * 8 != job["packet_bits"]:
                        raise ValueError("Private sent/blocked payload hash/cost differs")
            rows = []
            for group in ("numeric", "text"):
                jobs, _, _ = verify_jobs(public, model, plan, group, asset=asset)
                directory = receipts / model / group
                status = read_json(directory / "run_status.json")
                if status["status"] != "completed" or status["model"] != model or status["group"] != group:
                    raise ValueError("Receiver group is failed or incomplete")
                verify_seal(directory)
                if status["plan_sha256"] != sha(public / "plan.json") or status["jobs_sha256"] != sha(root / "jobs.json"):
                    raise ValueError("Receiver group belongs to different planned jobs")
                group_rows = read_rows(directory / "rows.jsonl")
                if status["n_requested"] != len(jobs) or status["n_completed"] != len(jobs) or [row["job_id"] for row in group_rows] != [job["job_id"] for job in jobs]:
                    raise ValueError("Missing, duplicate or reordered receiver results")
                for row, job in zip(group_rows, jobs):
                    if any(type(row[key]) is not int for key in ("packet_bits", "sent_bits", "delivered_bits")):
                        raise ValueError("Receiver cost fields must be integers, not booleans/floats")
                    for field in ("case_id", "condition", "group", "delivery_status", "packet_bits", "sent_bits", "delivered_bits"):
                        if row[field] != job[field]:
                            raise ValueError("Receiver result identity/cost differs from planned job")
                    if row["execution_status"] != "completed" or row["model"] != model or row["source_packet_sha256"] != job["packet_sha256"]:
                        raise ValueError("Receiver result is not completed matching evidence")
                    validate_result(row["result"], job)
                    row = {**row, **string_scores(row["result"]["answer"], cases[row["case_id"]]["target"]),
                           "alternate_exact_match": string_scores(row["result"]["answer"], cases[row["case_id"]]["alternate"])["exact_match"]}
                    rows.append(row)
                    metrics.append(row)
            by_condition = {condition: [row for row in rows if row["condition"] == condition] for condition in tuple(plan["primary_conditions"]) + ENGINEERING}
            summaries[model] = {}
            for condition, observations in by_condition.items():
                required = plan["count"] if condition in plan["primary_conditions"] else plan["engineering_count"]
                if len(observations) != required or required == 0:
                    raise ValueError("Condition denominator differs from frozen plan")
                successes = sum(row["exact_match"] for row in observations)
                summaries[model][condition] = {"n": required, "successes": successes, "exact_rate": successes / required,
                    "exact_ci95": list(clopper_pearson_interval(successes, required)),
                    "mean_edit_similarity": float(np.mean([row["edit_similarity"] for row in observations])),
                    "alternate_matches": sum(row["alternate_exact_match"] for row in observations),
                    "empty_answers": sum(row["empty"] for row in observations), "unknown_answers": sum(row["unknown"] for row in observations),
                    "delivery_counts": dict(Counter(row["delivery_status"] for row in observations)),
                    "packet_bits": sum(row["packet_bits"] for row in observations), "sent_bits": sum(row["sent_bits"] for row in observations),
                    "delivered_bits": sum(row["delivered_bits"] for row in observations)}
            contrasts[model] = {}
            pairs = [("primary", "vector_generated", "no_comm"), ("secondary", "text_generated", "vector_generated")]
            if asset is not None:
                pairs += [("pca_primary", "vector_pca_generated", "no_comm"), ("pca_vs_raw", "vector_pca_generated", "vector_generated")]
            for label, first, second in pairs:
                a, b = by_condition[first], by_condition[second]
                contrasts[model][label] = paired_binary_test([row["exact_match"] for row in a], [row["exact_match"] for row in b],
                    episode_ids=[row["case_id"] for row in a], episode_ids_b=[row["case_id"] for row in b],
                    n_bootstrap=plan["bootstrap_resamples"], seed=plan["bootstrap_seed"])
        gate_models = {}
        policy = plan["scientific_gate"]
        for model in plan["model_order"]:
            values = summaries[model]
            checks = {condition: values[condition]["successes"] >= policy["positive_required"]
                      for condition in policy["positive_original_conditions"]}
            checks["vector_swapped_alternate"] = values["vector_swapped"]["alternate_matches"] >= policy["positive_required"]
            checks.update({condition: values[condition]["successes"] <= policy["negative_maximum"] for condition in policy["zero_original_conditions"]})
            checks["dropped_statuses"] = values["vector_dropped"]["delivery_counts"] == {"dropped": plan["engineering_count"]}
            checks["blocked_statuses"] = values["vector_budget544"]["delivery_counts"] == {"blocked": plan["engineering_count"]}
            gate_models[model] = {"passed": all(checks.values()), "checks": checks,
                                  "observed_counts": {condition: {key: values[condition][key] for key in ("n", "successes", "alternate_matches")} for condition in CONDITIONS}}
        pca_gate = None
        if plan["pca_success_gate"] is not None:
            policy_pca = plan["pca_success_gate"]
            pca_models = {}
            for model in plan["model_order"]:
                values = summaries[model]
                checks = {condition: values[condition]["successes"] >= policy_pca["positive_required"] for condition in policy_pca["positive_original_conditions"]}
                checks["vector_pca_swapped_alternate"] = values["vector_pca_swapped"]["alternate_matches"] >= policy_pca["positive_required"]
                checks["vector_pca_mean"] = values["vector_pca_mean"]["successes"] == 0
                pca_models[model] = {"passed": all(checks.values()), "checks": checks,
                    "observed_counts": {condition: {key: values[condition][key] for key in ("n", "successes", "alternate_matches")} for condition in PCA_PRIMARY}}
            pca_gate = {"passed": all(value["passed"] for value in pca_models.values()), "policy": policy_pca, "models": pca_models}
        family = {(model, label): contrasts[model][label]["p"] for model in plan["model_order"]
                  for label in (("primary", "pca_primary") if plan["pca_success_gate"] is not None else ("primary",))}
        rejections = dict(zip(family, holm_bonferroni(list(family.values()))))
        report = {"status": "completed", "plan_sha256": sha(public / "plan.json"), "count": plan["count"],
                  "summaries": summaries, "contrasts": contrasts,
                  "purpose": plan["purpose"],
                  "scientific_gate": {"passed": all(value["passed"] for value in gate_models.values()), "policy": policy, "models": gate_models},
                  "pca_success_gate": pca_gate,
                  "pca_setup_bytes": {model: plan["models"][model].get("pca", {}).get("setup_bytes", 0) for model in plan["model_order"]},
                  "primary_holm_reject": {model: rejections[model, "primary"] for model in plan["model_order"]},
                  "pca_primary_holm_reject": {model: rejections[model, "pca_primary"] for model in plan["model_order"]} if pca_gate is not None else None,
                  "scope": "Generated lexical message transported as numeric spans; oracle diagnostics are separate. No claim of semantic compression or independent output-token replay.",
                  "all_planned_results_complete": True}
        for row in metrics:
            append_json(stage.directory / "episode_metrics.jsonl", row)
        verify_plan(public, check_tests=True)
        verify_prepared(study)
        atomic_json(stage.directory / "analysis.json", report)
        stage.update(n_requested=len(metrics), n_completed=len(metrics))
        stage.complete()
        seal(stage.directory)
        return report
    except BaseException as error:
        stage.fail(error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--forbidden-values", type=Path, required=True)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--purpose", choices=("development", "heldout"), required=True)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--pca-dim", type=int, default=512)
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--codec", choices=("float32", "int8", "int4"), default="int4")
    for model, default in (("qwen", "qwen2_5_7b.yaml"), ("llama", "llama_3_1_8b.yaml")):
        p.add_argument(f"--{model}-config", type=Path, default=ROOT / "configs" / default)
        p.add_argument(f"--{model}-layer", type=int)
        p.add_argument(f"--{model}-codec", choices=("float32", "int8", "int4"))
        p.add_argument(f"--{model}-pca-asset", type=Path)
        p.add_argument(f"--{model}-pca-lineage", type=Path)
        p.add_argument(f"--{model}-pca-codec", choices=("float32", "int8", "int4"))
    p = sub.add_parser("send")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p = sub.add_parser("receive")
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="Fresh group output is created below OUTPUT/model/group")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--group", choices=("numeric", "text"), required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--receipts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args.study, args.forbidden_values, {model: getattr(args, model + "_config") for model in MODELS}, purpose=args.purpose, count=args.count, models=args.models,
            layers={model: getattr(args, model + "_layer") if getattr(args, model + "_layer") is not None else args.layer for model in MODELS},
            codecs={model: getattr(args, model + "_codec") or args.codec for model in MODELS},
            pca_assets={model: getattr(args, model + "_pca_asset") for model in args.models if getattr(args, model + "_pca_asset") is not None},
            pca_lineage={model: getattr(args, model + "_pca_lineage") for model in args.models if getattr(args, model + "_pca_lineage") is not None}, pca_dim=args.pca_dim,
            pca_codecs={model: getattr(args, model + "_pca_codec") or getattr(args, model + "_codec") or args.codec for model in args.models})
    elif args.phase == "send":
        result = send(args.study, args.model)
    elif args.phase == "receive":
        result = receive(args.public, args.output, args.model, args.group)
    else:
        result = analyze(args.study, args.receipts, args.output)
    summary = {"phase": args.phase, "status": result.get("status", "completed")}
    if "scientific_gate" in result and "passed" in result["scientific_gate"]:
        summary["scientific_gate_passed"] = result["scientific_gate"]["passed"]
    if result.get("pca_success_gate") is not None and "passed" in result["pca_success_gate"]:
        summary["pca_success_gate_passed"] = result["pca_success_gate"]["passed"]
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
