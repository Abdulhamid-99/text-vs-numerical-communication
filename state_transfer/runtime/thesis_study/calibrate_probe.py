"""Disjoint task-specific PCA calibration and actual development inference.

Calibration sees only A's canonical messages. Development captures come from a
completed/sealed actual-generation probe. This is explicitly not heldout evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np

from budgetfair.model.client import LLMClient, GenerationConfig
from budgetfair.model.pca_transport import receive_projected_packet
from budgetfair.model.span_transport import receive_packet
from budgetfair.comm.pca_packet import load_asset, encode_projected, encode_coefficients
from budgetfair.comm.tensor_packet import encode_tensor
from common import Stage, read_json, read_rows, verify_seal, write_json, append_row, sha, timed, encode_with_states, within
from private_tasks import agent_view, sender_requested_message, score_answer, validate_split_isolation, QA_TASK, NAV_TASK
from task_prompts import receiver_parts, sender_encoding_prompt, PROMPT_VERSION
from navigation_agent import attach_task_response, NAVIGATION_AGENT_VERSION


def fit_pca(features, dimensions):
    import torch
    if features.dtype != np.float32 or features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Invalid calibration feature matrix")
    if type(dimensions) is not int or not 0 < dimensions < min(features.shape):
        raise ValueError("Insufficient independent calibration rows/width")
    started = time.perf_counter()
    data = torch.from_numpy(features).double()
    mean = data.mean(0).float()
    centered = data - mean.double()
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    threshold = torch.finfo(torch.float64).eps * max(features.shape) * singular[0]
    if not bool(singular[dimensions - 1] > threshold):
        raise ValueError("Calibration numerical rank is below requested dimensions")
    components = vh[:dimensions].float()
    pivot = components.abs().argmax(1)
    components *= torch.sign(components[torch.arange(dimensions), pivot])[:, None]
    variance = singular.square()
    stats = {"method": "Exact torch.linalg.svd CPU float64; float32 stored mean/components",
        "rows": features.shape[0], "width": features.shape[1], "dimensions": dimensions,
        "seconds": time.perf_counter() - started, "singular_values": singular.tolist(),
        "cumulative_explained_variance": (variance.cumsum(0) / variance.sum()).tolist(),
        "sign_rule": "Largest absolute loading is positive", "generation_performed": False}
    return mean.numpy().copy(), components.numpy().copy(), stats


def run(data, source_probe, config, output, model, rank=512):
    if type(rank) is not int or rank <= 0:
        raise ValueError("PCA rank must be a positive integer")
    expected_names = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct"}
    model_config = read_json(config)
    if model not in expected_names:
        raise ValueError("Unknown calibration model identity")
    for section in ("model", "tokenizer"):
        pinned = model_config.get(section, {})
        if (pinned.get("name") != expected_names[model] or not isinstance(pinned.get("revision"), str)
                or re.fullmatch(r"[0-9a-f]{40}", pinned["revision"]) is None):
            raise ValueError("Calibration requires the exact immutable model/tokenizer identity")
    data_status, probe_status = verify_seal(data), verify_seal(source_probe)
    if probe_status.get("purpose") != "development" or probe_status.get("model") != model:
        raise ValueError("Wrong source probe scope/model")
    calibration = read_json(data / "calibration.json")["families"]
    development = read_json(data / "development.json")["families"]
    if (not calibration or not development or any(f["split"] != "calibration" for f in calibration)
            or any(f["split"] != "development" for f in development)
            or set(f["task"] for f in calibration) != {QA_TASK, NAV_TASK}):
        raise ValueError("Expected both tasks on disjoint calibration/development splits")
    validate_split_isolation(calibration + development)
    all_development = {f["family_id"]: f for f in development}
    source_rows = read_rows(source_probe / "encodings.jsonl")
    ids = list(dict.fromkeys(row["family_id"] for row in source_rows))
    families = [all_development[key] for key in ids]
    source_cells = [(row["family_id"], row["variant"]) for row in source_rows]
    expected_cells = {(family["family_id"], variant) for family in families for variant in ("original", "alternate", "oracle")}
    if (not families or len(source_rows) != len(expected_cells) or set(source_cells) != expected_cells
            or set(f["task"] for f in families) != {QA_TASK, NAV_TASK}):
        raise ValueError("Missing complete calibration/development captures")
    stage = Stage(output, "task_pca_development", requested=2*len(calibration)+5*len(families), model=model,
                  purpose="development", calibration_completed=0, generations_completed=0)
    try:
        old = read_json(source_probe / "configuration.json")
        if (old["config_sha256"] != sha(config) or old["prompt_version"] != PROMPT_VERSION
                or old["data_manifest_sha256"] != sha(data / "artifact_manifest.json")
                or old.get("model") != model or old.get("model_config") != model_config):
            raise ValueError("Cached probe model/prompt differs")
        source_manifest = read_json(source_probe / "artifact_manifest.json")["files"]
        layer, codec = old["layer"], "int4" if model == "qwen" else "int8"
        if type(layer) is not int or layer < 0:
            raise ValueError("Invalid cached encoder block index")
        # Validate every existing capture before loading a model or performing
        # calibration forwards. Actual calibration numerical rank is checked
        # separately after fitting; source width alone can be known now.
        widths = set()
        for row in source_rows:
            relative = "raw/" + row["array"]
            array_path = within(source_probe, relative)
            if relative not in source_manifest or sha(array_path) != source_manifest[relative]["sha256"]:
                raise ValueError("Development state-file checksum differs before calibration")
            states = np.load(array_path, allow_pickle=False)
            if (states.dtype != np.float32 or states.ndim != 2 or min(states.shape) <= 0 or not np.isfinite(states).all()
                    or list(states.shape) != row["trace"]["shape"]
                    or hashlib.sha256(states.astype("<f4").tobytes()).hexdigest() != row["trace"]["source_array_sha256"]):
                raise ValueError("Development states differ from captured encoder before calibration")
            if rank >= states.shape[1]:
                raise ValueError("PCA rank must be smaller than the captured hidden width")
            widths.add(states.shape[1])
        if len(widths) != 1:
            raise ValueError("One checkpoint must have a consistent captured hidden width")
        client, load_seconds = timed(lambda: LLMClient.from_config(config))
        client.generation = GenerationConfig(max_new_tokens=192, temperature=0.0, do_sample=False)
        write_json(output / "configuration.json", {"model": model, "layer": layer, "codec": codec,
            "rank": rank, "config_sha256": sha(config), "model_config": read_json(config),
            "prompt_version": PROMPT_VERSION, "model_load_seconds": load_seconds,
            "navigation_agent_version": NAVIGATION_AGENT_VERSION,
            "source_probe_manifest_sha256": sha(source_probe / "artifact_manifest.json"),
            "data_manifest_sha256": sha(data / "artifact_manifest.json"),
            "sources": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}})
        captures = output / "calibration_states"
        captures.mkdir()
        by_task = defaultdict(list)
        count = 0
        for family in calibration:
            encoding_prompt = sender_encoding_prompt(client, family["task"])
            for variant, episode in family["episodes"].items():
                stage.update(stage="calibration_forward", case_id=family["family_id"], variant=variant)
                message = sender_requested_message(episode)
                (_packet, trace, states), seconds = timed(lambda: encode_with_states(
                    client, encoding_prompt, message, layer=layer, codec="int4"))
                filename = family["family_id"] + "_" + variant + ".npy"
                np.save(captures / filename, states, allow_pickle=False)
                by_task[family["task"]].append(states)
                append_row(output / "calibration_trace.jsonl", {"task": family["task"],
                    "family_id": family["family_id"], "variant": variant, "message": message,
                    "generation_performed": False, "encoding": trace, "seconds": seconds,
                    "encoding_prompt": asdict(encoding_prompt), "file": filename, "sha256": sha(captures / filename)})
                count += 1
                stage.update(n_completed=count, calibration_completed=count)
        assets = {}
        for task, arrays in by_task.items():
            stage.update(stage="exact_cpu_pca_fit", task=task)
            matrix = np.concatenate(arrays)
            mean, components, stats = fit_pca(matrix, rank)
            asset_path = output / (task + "_basis.npz")
            np.savez(asset_path, mean=mean, components=components,
                     layer=np.asarray(layer, dtype=np.int32), schema=np.asarray(1, dtype=np.int32))
            asset = load_asset(asset_path, expected_sha256=sha(asset_path))
            write_json(output / (task + "_fit.json"), stats)
            write_json(output / (task + "_descriptor.json"), {"task": task, "model": model,
                "layer": layer, "rank": rank, "codec": codec, "basis_sha256": asset.fingerprint,
                "basis_file_bytes": asset.setup_bytes, "calibration_family_ids": [f["family_id"] for f in calibration if f["task"] == task],
                "calibration_data_sha256": sha(data / "calibration.json"), "heldout_seen": False,
                "scope": "Task-specific PCA learned from A canonical messages on the independent calibration split"})
            assets[task] = asset
        packet_dir = output / "packets"
        packet_dir.mkdir()
        summary = defaultdict(lambda: defaultdict(lambda: {"n": 0, "successes": 0}))
        generations = 0
        for family in families:
            task = family["task"]
            prompt, parts = receiver_parts(client, agent_view(family["episodes"]["original"], "B"))
            asset = assets[task]
            arrays = {}
            for row in source_rows:
                if row["family_id"] == family["family_id"]:
                    if tuple(row["encoding_prompt"]["prefix_ids"]) != prompt.prefix_ids:
                        raise ValueError("Calibration/development receiver prefix differs")
                    relative = "raw/" + row["array"]
                    array_path = within(source_probe, relative)
                    if sha(array_path) != source_manifest[relative]["sha256"]:
                        raise ValueError("Development state-file checksum differs")
                    states = np.load(array_path, allow_pickle=False)
                    if (states.dtype != np.float32 or states.ndim != 2 or not np.isfinite(states).all()
                            or list(states.shape) != row["trace"]["shape"]
                            or hashlib.sha256(states.astype("<f4").tobytes()).hexdigest() != row["trace"]["source_array_sha256"]):
                        raise ValueError("Development states differ from actual captured encoder array")
                    arrays[row["variant"]] = states
            if set(arrays) != {"original", "alternate", "oracle"}:
                raise ValueError("Incomplete captured source states")
            conditions = [("pca_generated", "original"), ("pca_oracle", "oracle"),
                          ("pca_swapped", "alternate"), ("pca_mean", "original"), ("raw_zero", "original")]
            for condition, variant in conditions:
                stage.update(stage="receiver_generation", case_id=family["family_id"], condition=condition)
                array = arrays[variant]
                if condition == "raw_zero":
                    packet = encode_tensor(np.zeros_like(array), codec="int4", role="hidden_states", layer_index=layer)
                    fn = lambda: receive_packet(client, prompt, packet, expected_layer=layer)
                else:
                    packet = (encode_coefficients(np.zeros((array.shape[0], rank), dtype=np.float32), asset, codec=codec)
                              if condition == "pca_mean" else encode_projected(array, asset, dimensions=rank, codec=codec))
                    fn = lambda: receive_projected_packet(client, prompt, packet, asset, expected_layer=layer,
                                                          expected_dimensions=rank, expected_codec=codec)
                file = family["family_id"] + "_" + condition + ".bin"
                (packet_dir / file).write_bytes(packet)
                result, seconds = timed(fn)
                result = attach_task_response(agent_view(family["episodes"]["original"], "B"), result)
                episode = family["episodes"]["alternate" if condition == "pca_swapped" else "original"]
                score = score_answer(episode, result["task_answer"])
                append_row(output / "rows.jsonl", {"task": task, "family_id": family["family_id"],
                    "condition": condition, "result": result, "seconds": seconds, "score": score,
                    "score_original": score_answer(family["episodes"]["original"], result["task_answer"]),
                    "score_alternate": score_answer(family["episodes"]["alternate"], result["task_answer"]),
                    "packet": file, "packet_sha256": sha(packet_dir / file), "wire_bits": len(packet)*8,
                    "receiver_prompt": asdict(prompt), "receiver_prompt_text": parts})
                summary[task][condition]["n"] += 1
                summary[task][condition]["successes"] += int(score["success"])
                count += 1
                generations += 1
                stage.update(n_completed=count, generations_completed=generations)
        write_json(output / "summary.json", dict(summary))
        stage.finish()
        print({"status": "completed", "model": model, "summary": dict(summary)})
    except BaseException as error:
        stage.fail(error)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "source-probe", "config", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--model", choices=("qwen", "llama"), required=True)
    p.add_argument("--rank", type=int, default=512)
    a = p.parse_args()
    run(a.data, a.source_probe, a.config, a.output, a.model, a.rank)
