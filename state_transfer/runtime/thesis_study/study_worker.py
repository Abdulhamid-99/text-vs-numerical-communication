"""Offline isolated model stages. A and B inputs are mounted separately.

This program never receives a full evaluation family or private evaluator file.
GPU/driver/model exceptions fail and unseal the stage; they are never scored zero.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.metadata
import time
import numpy as np

from budgetfair.model.client import LLMClient, GenerationConfig
from budgetfair.model.span_transport import receive_text, receive_packet, neutral_generation_kwargs
from budgetfair.model.pca_transport import receive_projected_packet
from budgetfair.comm.tensor_packet import encode_tensor
from budgetfair.comm.pca_packet import load_asset, encode_projected, encode_coefficients
from common import Stage, read_json, write_json, append_row, sha, timed, generate, encode_with_states, within
from task_prompts import sender_messages, receiver_parts, sender_encoding_prompt, PROMPT_VERSION
from protocol_contracts import validate_protocol, validate_role_input
from private_tasks import NAV_TASK
from navigation_agent import attach_task_response


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_protocol(path, model):
    plan = validate_protocol(path, code_root=Path(__file__).parent.parent)
    if model not in plan["models"]:
        raise ValueError("Model absent from frozen protocol")
    return plan, plan["models"][model]


def load_client(settings):
    client, seconds = timed(lambda: LLMClient.from_config_dict(settings["config"]))
    client.generation = GenerationConfig(**settings["generation"])
    return client, seconds


def runtime_identity(client, settings):
    """Record executed policy and refuse unplanned placement before any generation."""
    placements = dict(getattr(client.model, "hf_device_map", {}) or {})
    canonical_device = lambda value: "cuda:0" if str(value) in ("0", "cuda:0") else str(value)
    if (settings["allowed_device_placements"] != ["cuda:0"] or not placements
            or any(canonical_device(value) != "cuda:0" for value in placements.values())):
        raise ValueError("Actual model placement includes missing/unplanned CPU, disk or GPU offload")
    parameter_devices = sorted({str(parameter.device) for parameter in client.model.parameters()})
    if not parameter_devices or any(device != "cuda:0" for device in parameter_devices):
        raise ValueError("Actual parameter placement differs from frozen CUDA-0 policy")
    def plain(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if hasattr(value, "to_dict"):
            return plain(value.to_dict())
        return {"class": type(value).__module__ + "." + type(value).__name__}
    effective = neutral_generation_kwargs(client)
    return {"hf_device_map": {name: str(device) for name, device in placements.items()},
        "parameter_devices": parameter_devices, "input_device": str(client._input_device()),
        "dependency_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "bitsandbytes")},
        "checkpoint_generation_defaults": plain(client.model.generation_config.to_dict()),
        "effective_generation_kwargs": plain(effective), "client_generation": asdict(client.generation),
        "generation_policy": "Fresh HF configuration retains special IDs only; use_model_defaults=False; explicit client generation and finite-logits guard."}


def validate_asset_root(plan_path, assets_root):
    # The only legal asset path is the already verified public plan subdirectory.
    if Path(assets_root).absolute() != (Path(plan_path).parent / "assets").absolute():
        raise ValueError("PCA assets must come from the verified public plan boundary")


def task_assets(root, settings):
    return {task: load_asset(within(root, record["basis_file"]), expected_sha256=record["basis_sha256"])
            for task, record in settings["tasks"].items()}


def sender(plan_path, input_path, assets_root, output, model):
    plan, settings = load_protocol(plan_path, model)
    data = read_json(input_path)
    items = validate_role_input(data, plan, sha(plan_path), "A", model=model, input_root=input_path.parent)
    validate_asset_root(plan_path, assets_root)
    stage = Stage(output, "heldout_sender", requested=len(items), model=model,
                  a_generations_completed=0, encoding_forwards_completed=0)
    try:
        assets = task_assets(assets_root, settings)
        client, load_seconds = load_client(settings)
        runtime = runtime_identity(client, settings)
        write_json(output / "identity.json", {"plan_sha256": sha(plan_path), "input_sha256": sha(input_path),
            "model": model, "settings": settings, "model_load_seconds": load_seconds,
            "runtime": runtime,
            "isolation": "Only A views, frozen code, public PCA assets and model weights are mounted"})
        (output / "sources").mkdir()
        (output / "packets").mkdir()
        n, captures = 0, 0
        for item in items:
            stage.update(stage="sender_generation", source_id=item["id"])
            trace, seconds = timed(lambda: generate(client, sender_messages(item["view"])))
            message = trace["text"]
            token_ids = client.tokenizer.encode(message, add_special_tokens=False)
            source_status = "valid" if 1 <= len(token_ids) <= 128 else "source_contract_failure"
            append_row(output / "senders.jsonl", {"id": item["id"], "family_id": item["family_id"],
                "task": item["task"], "variant": item["variant"], "trace": trace,
                "seconds": seconds, "payload_token_count": len(token_ids), "source_status": source_status,
                "source_exact": message == item["view"]["private_context"]})
            kinds = [("generated", message, source_status)]
            if item["variant"] == "original":
                kinds.append(("oracle", item["view"]["private_context"], "valid"))
            for kind, content, status in kinds:
                stem = item["id"] + "_" + kind
                record = {"id": stem, "source_id": item["id"], "family_id": item["family_id"],
                    "task": item["task"], "variant": item["variant"], "kind": kind,
                    "source_status": status, "packets": {}}
                if status == "valid":
                    public = sender_encoding_prompt(client, item["task"])
                    stage.update(stage="sender_encoding", source_id=stem)
                    (raw, encoding, states), encoding_seconds = timed(lambda: encode_with_states(
                        client, public, content, layer=settings["layer"], codec="int4"))
                    captures += 1
                    array_name = "sources/" + stem + ".npy"
                    np.save(output / array_name, states, allow_pickle=False)
                    asset = assets[item["task"]]
                    pca, pca_seconds = timed(lambda: encode_projected(states, asset, dimensions=settings["pca_rank"], codec=settings["pca_codec"]))
                    packets = {"text": content.encode("utf-8"), "raw": raw, "pca": pca}
                    if kind == "generated" and item["variant"] == "original":
                        packets["raw_zero"] = encode_tensor(np.zeros_like(states), codec="int4", role="hidden_states", layer_index=settings["layer"])
                        packets["pca_mean"] = encode_coefficients(np.zeros((states.shape[0], settings["pca_rank"]), dtype=np.float32), asset, codec=settings["pca_codec"])
                    for channel, packet in packets.items():
                        file = "packets/" + stem + "_" + channel + ".bin"
                        (output / file).write_bytes(packet)
                        record["packets"][channel] = {"file": file, "sha256": digest_bytes(packet), "wire_bits": len(packet)*8}
                    record.update(encoding=encoding, encoding_prompt=asdict(public), array_file=array_name,
                        array_file_sha256=sha(output / array_name), encoding_seconds=encoding_seconds,
                        pca_encoding_seconds=pca_seconds, source_generation_seconds=seconds if kind == "generated" else 0.0)
                append_row(output / "encodings.jsonl", record)
            n += 1
            stage.update(n_completed=n, a_generations_completed=n, encoding_forwards_completed=captures)
        stage.finish()
    except BaseException as error:
        stage.fail(error)
        raise


def receiver(plan_path, input_path, assets_root, output, model, channel):
    plan, settings = load_protocol(plan_path, model)
    data = read_json(input_path)
    items = validate_role_input(data, plan, sha(plan_path), "B", model=model, channel=channel, input_root=input_path.parent)
    validate_asset_root(plan_path, assets_root)
    stage = Stage(output, "heldout_receiver", requested=len(items), model=model, channel=channel,
                  b_generations_completed=0)
    try:
        assets = task_assets(assets_root, settings) if channel == "pca" else {}
        client, load_seconds = load_client(settings)
        runtime = runtime_identity(client, settings)
        write_json(output / "identity.json", {"plan_sha256": sha(plan_path), "input_sha256": sha(input_path),
            "model": model, "channel": channel, "settings": settings, "model_load_seconds": load_seconds,
            "runtime": runtime,
            "isolation": "Only B views, this channel's packets, public assets, frozen code and model weights are mounted"})
        n = 0
        for item in items:
            stage.update(stage="receiver_generation", case_id=item["family_id"], condition=item["condition"])
            public, parts = receiver_parts(client, item["view"])
            packet = None
            if item["packet"] is not None:
                descriptor = item["packet"]
                path = within(input_path.parent, descriptor["file"])
                packet = path.read_bytes()
                if digest_bytes(packet) != descriptor["sha256"] or len(packet)*8 != descriptor["wire_bits"]:
                    raise ValueError("Packet changed or byte cost differs")
            if packet is None:
                fn = lambda: receive_packet(client, public, None, expected_layer=settings["layer"])
            elif channel == "text":
                fn = lambda: receive_text(client, public, packet)
            elif channel == "raw":
                fn = lambda: receive_packet(client, public, packet, expected_layer=settings["layer"])
            elif channel == "pca":
                fn = lambda: receive_projected_packet(client, public, packet, assets[item["task"]],
                    expected_layer=settings["layer"], expected_dimensions=settings["pca_rank"], expected_codec=settings["pca_codec"])
            else:
                raise ValueError("Unsupported channel")
            result, seconds = timed(fn)
            if result["status"] != "completed" or not result["output_ids"]:
                raise ValueError("Incomplete receiver model generation")
            planner_started = time.perf_counter()
            result = attach_task_response(item["view"], result)
            planner_seconds = time.perf_counter() - planner_started if item["task"] == NAV_TASK else 0.0
            append_row(output / "rows.jsonl", {"id": item["id"], "family_id": item["family_id"],
                "task": item["task"], "condition": item["condition"], "channel": channel,
                "source_status": item["source_status"], "result": result, "seconds": seconds,
                "planner_seconds": planner_seconds,
                "receiver_prompt": asdict(public), "receiver_prompt_text": parts,
                "packet": item["packet"], "input_sha256": sha(input_path), "plan_sha256": sha(plan_path)})
            n += 1
            stage.update(n_completed=n, b_generations_completed=n)
        stage.finish()
    except BaseException as error:
        stage.fail(error)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("sender", "receiver"))
    for name in ("plan", "input", "assets", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--model", choices=("qwen", "llama"), required=True)
    p.add_argument("--channel", choices=("text", "raw", "pca"))
    a = p.parse_args()
    if a.mode == "sender":
        sender(a.plan, a.input, a.assets, a.output, a.model)
    else:
        receiver(a.plan, a.input, a.assets, a.output, a.model, a.channel)
