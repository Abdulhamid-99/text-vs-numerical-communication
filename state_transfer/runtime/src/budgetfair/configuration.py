"""Explicit supported protocol and strict experiment configuration validation."""
from __future__ import annotations

import math


def integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def boolean(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("experiment configuration must be a mapping")
    known = {
        "experiment": {"name", "seed", "description"},
        "agents": {"count", "roles"},
        "task": {"name", "dataset", "num_episodes", "max_communication_rounds", "question_type", "secret_mode",
                 "n_digits", "n_letters", "manifest_path", "split"},
        "channel": {"type", "budget_bits", "injection_strength", "random_injection"},
        "network": {"name", "bandwidth_bps", "latency_ms", "packet_loss"},
        "logging": {"output_dir", "log_level", "store_messages", "store_vectors"},
        "protocol": {"version", "direction", "final_agent", "sender_mode", "sender_value", "swap_contexts"},
    }
    unknown = set(cfg) - set(known) - {"model_config"}
    if unknown:
        raise ValueError(f"unsupported configuration keys: {sorted(unknown)}")
    for key in ("experiment", "task", "channel", "logging"):
        if key not in cfg:
            raise ValueError(f"missing {key} configuration")
    for section, allowed in known.items():
        if section not in cfg:
            continue
        values = cfg[section]
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f"unsupported fields or shape in {section}: {values!r}")
    if not isinstance(cfg.get("model_config"), str) or not cfg["model_config"]:
        raise ValueError("model_config must name a YAML file")
    integer("seed", cfg["experiment"].get("seed", 42))
    if integer("agents.count", cfg.get("agents", {}).get("count", 2), 1) != 2:
        raise ValueError("only exactly two agents are implemented")
    roles = cfg.get("agents", {}).get("roles", ["knowledge_holder_a", "knowledge_holder_b"])
    if roles != ["knowledge_holder_a", "knowledge_holder_b"]:
        raise ValueError("only knowledge_holder_a and knowledge_holder_b roles are implemented")
    task = cfg["task"]
    if task.get("name") not in {"secret_passing", "multidoc_qa"}:
        raise ValueError("unsupported task; available: secret_passing, multidoc_qa")
    irrelevant = {"dataset", "question_type"} if task["name"] == "secret_passing" else {"secret_mode", "n_digits", "n_letters"}
    if set(task) & irrelevant:
        raise ValueError(f"fields do not apply to {task['name']}: {sorted(set(task) & irrelevant)}")
    if task.get("split", "dev") not in {"dev", "test"}:
        raise ValueError("task.split must be dev or test")
    if task.get("split") == "test" and not task.get("manifest_path"):
        raise ValueError("a held-out test split requires a frozen manifest_path")
    if "manifest_path" in task and (not isinstance(task["manifest_path"], str) or not task["manifest_path"].strip()):
        raise ValueError("manifest_path must be a nonempty path string")
    integer("num_episodes", task.get("num_episodes"), 1)
    integer("max_communication_rounds", task.get("max_communication_rounds", 4), 1)
    for key in ("n_digits", "n_letters"):
        if key in task:
            integer(key, task[key], 1)
    channel = cfg["channel"]
    if channel.get("type") not in {"text", "vector", "hybrid"}:
        raise ValueError("channel.type must be text, vector or hybrid")
    if channel.get("budget_bits") is not None:
        integer("budget_bits", channel["budget_bits"])
    if "random_injection" in channel:
        boolean("random_injection", channel["random_injection"])
        if channel["random_injection"] and channel["type"] == "text":
            raise ValueError("random_injection only applies to a vector channel")
    if "injection_strength" in channel:
        strength = channel["injection_strength"]
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or strength < 0:
            raise ValueError("injection_strength must be finite non-negative")
        if channel["type"] == "text":
            raise ValueError("injection_strength has no effect on a text channel")
    for key in ("store_messages", "store_vectors"):
        if key in cfg["logging"]:
            boolean(key, cfg["logging"][key])
    if cfg["logging"].get("log_level", "INFO") != "INFO":
        raise ValueError("only log_level INFO is supported; events are always durable")
    protocol = cfg.get("protocol", {})
    if protocol.get("version", 2) != 2:
        raise ValueError("only protocol version 2 is supported")
    if protocol.get("direction", "bidirectional") not in {"bidirectional", "sender_to_receiver"}:
        raise ValueError("invalid communication direction")
    if protocol.get("final_agent", "agent_b") != "agent_b":
        raise ValueError("the predeclared evaluated receiver is agent_b")
    if protocol.get("sender_mode", "generated") not in {"generated", "oracle"}:
        raise ValueError("sender_mode must be generated or oracle")
    if protocol.get("sender_value", "original") not in {"original", "alternate"}:
        raise ValueError("sender_value must be original or alternate")
    boolean("swap_contexts", protocol.get("swap_contexts", False))
    if task["name"] == "secret_passing" and protocol.get("swap_contexts", False):
        raise ValueError("swap_contexts applies only to the two-passage QA task")
    if task["name"] != "secret_passing" and (protocol.get("sender_mode", "generated") != "generated" or protocol.get("sender_value", "original") != "original"):
        raise ValueError("oracle/alternate controls require a secret_passing task")
    if protocol.get("sender_mode") == "oracle" and protocol.get("direction") != "sender_to_receiver":
        raise ValueError("oracle controls require sender_to_receiver direction")


def validate_vector_config(model_cfg: dict, channel_type: str) -> None:
    if set(model_cfg) - {"model", "tokenizer", "generation", "vector_channel"}:
        raise ValueError("unsupported top-level model configuration fields")
    vc = model_cfg.get("vector_channel", {})
    allowed = {"injection_layer", "extraction_layer", "injection_strength", "layer_index", "hidden_dim",
               "projected_dim", "bits_per_component", "projection_seed", "pooling"}
    if not isinstance(vc, dict) or set(vc) - allowed:
        raise ValueError("unsupported vector_channel configuration fields")
    if channel_type in {"vector", "hybrid"} and not {"hidden_dim", "projected_dim", "injection_layer"} <= vc.keys():
        raise ValueError("vector protocol requires explicit hidden_dim, projected_dim and injection_layer")
    for key in ("hidden_dim", "projected_dim", "bits_per_component"):
        if key in vc:
            integer(key, vc[key], 1)
    for key in ("injection_layer", "extraction_layer", "layer_index", "projection_seed"):
        if key in vc:
            integer(key, vc[key])
    if "extraction_layer" in vc and "layer_index" in vc and vc["extraction_layer"] != vc["layer_index"]:
        raise ValueError("legacy layer_index conflicts with extraction_layer; remove the legacy field or make them agree")
    if vc.get("pooling", "last_token") not in {"mean", "last_token", "cls"}:
        raise ValueError("unsupported vector pooling")
    if vc.get("bits_per_component", 4) not in {4, 8}:
        raise ValueError("vector wire supports only 4 or 8 bits per component")
    strength = vc.get("injection_strength", 1.0)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or strength < 0:
        raise ValueError("vector injection_strength must be finite non-negative")
