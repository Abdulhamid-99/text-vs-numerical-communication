"""Protocol v2: paid communication, then a separately evaluated local answer.

Every attempt owns a new output directory. Exceptions remain exceptions and are
also recorded durably; absent or failed episodes are never scored as zero.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np
import yaml

from .agents.base import AgentAction, AgentObservation
from .agents.knowledge_holder import KnowledgeHolderAgent
from .agents.vector_knowledge_holder import VectorKnowledgeHolderAgent
from .budget.controller import BudgetController, OverbudgetPolicy
from .comm.projection import GaussianProjection
from .comm.text_channel import TextChannel
from .comm.vector_channel import VectorChannel
from .configuration import validate_config, validate_vector_config
from .logging_utils.logger import EpisodeLogger
from .model.client import GenerationConfig, LLMClient
from .network.simulator import NetworkProfile, NetworkSimulator
from .run_state import RunState, atomic_json, fingerprint, source_identity, utc_now
from .tasks.multidoc_qa import MultiDocQA
from .tasks.secret_passing import SecretPassingTask


def _resolve(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root / path


def _other(agent_id: str) -> str:
    if agent_id not in {"agent_a", "agent_b"}:
        raise ValueError("unknown agent identity")
    return "agent_b" if agent_id == "agent_a" else "agent_a"


def _append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def run_experiment(config: Path, project_root: Path, *, preloaded_client=None) -> Path:
    """Run one immutable-config cell. Optional model reuse never reuses agents/state."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("logging"), dict) or not cfg["logging"].get("output_dir"):
        raise ValueError("a configuration must declare logging.output_dir before an attempt can be created")
    output_dir = _resolve(cfg["logging"]["output_dir"], project_root)
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
    state = RunState(output_dir, config_hash)
    client = None
    previous_trace_callback = None
    try:
        validate_config(cfg)
        code_root = Path(__file__).resolve().parents[2]
        source = source_identity(code_root)
        task_cfg, channel_cfg = cfg["task"], cfg["channel"]
        seed = cfg["experiment"].get("seed", 42)
        n_requested = task_cfg["num_episodes"]
        max_rounds = task_cfg.get("max_communication_rounds", 4)
        protocol = {"version": 2, "direction": "bidirectional", "final_agent": "agent_b",
                    "sender_mode": "generated", "sender_value": "original", "swap_contexts": False,
                    **cfg.get("protocol", {})}
        channel_type = channel_cfg["type"]
        budget = BudgetController(channel_cfg.get("budget_bits"), OverbudgetPolicy.BLOCK)
        net_cfg = cfg.get("network", {})
        profile = NetworkProfile(name=net_cfg.get("name", "configured"),
                                 bandwidth_bps=net_cfg.get("bandwidth_bps"),
                                 latency_ms=net_cfg.get("latency_ms", 0.0),
                                 packet_loss=net_cfg.get("packet_loss", 0.0))
        state.update(n_requested=n_requested, stage="load_data")
        manifest_path = task_cfg.get("manifest_path")
        if manifest_path is not None:
            manifest_path = _resolve(manifest_path, project_root)
        manifest_file_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path else None
        if task_cfg["name"] == "secret_passing":
            task = SecretPassingTask(secret_mode=task_cfg.get("secret_mode", "both"),
                                     n_digits=task_cfg.get("n_digits", 4),
                                     n_letters=task_cfg.get("n_letters", 5),
                                     manifest_path=manifest_path, split=task_cfg.get("split", "dev"))
        else:
            kwargs = {"dataset": task_cfg.get("dataset", "hotpotqa_dev"),
                      "question_type": task_cfg.get("question_type")}
            if manifest_path is not None:
                kwargs.update(manifest_path=manifest_path, split=task_cfg.get("split", "dev"))
            task = MultiDocQA(**kwargs)
        episodes = task.episodes(n_requested, seed)
        if len(episodes) != n_requested or len({ep.episode_id for ep in episodes}) != n_requested:
            raise ValueError("episode count/identity does not match the planned experiment")
        dataset_hash = fingerprint([asdict(ep) for ep in episodes])
        state.update(stage="load_model")
        model_cfg_path = _resolve(cfg["model_config"], project_root)
        model_file_hash = hashlib.sha256(model_cfg_path.read_bytes()).hexdigest()
        model_cfg = yaml.safe_load(model_cfg_path.read_text(encoding="utf-8-sig"))
        if not isinstance(model_cfg, dict):
            raise ValueError("model configuration must be a mapping")
        validate_vector_config(model_cfg, channel_type)
        LLMClient.loader_fingerprint(model_cfg)
        model_revision = model_cfg.get("model", {}).get("revision")
        tokenizer_revision = model_cfg.get("tokenizer", {}).get("revision", model_revision)
        if any(not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
               for revision in (model_revision, tokenizer_revision)):
            raise ValueError("experiment model and tokenizer revisions must be immutable 40-character commit SHAs")
        client = preloaded_client if preloaded_client is not None else LLMClient.from_config(model_cfg_path)
        previous_trace_callback = getattr(client, "trace_callback", None)
        if preloaded_client is not None:
            expected = model_cfg["model"]
            if client.name != expected["name"] or client.revision != expected["revision"]:
                raise ValueError("preloaded model identity differs from experiment configuration")
            if asdict(client.generation) != asdict(GenerationConfig(**(model_cfg.get("generation") or {}))):
                raise ValueError("preloaded generation configuration differs from frozen cell")
            if client.loader_config_fingerprint != LLMClient.loader_fingerprint(model_cfg):
                raise ValueError("preloaded model/tokenizer loading settings differ from frozen cell")
        def trace(event):
            _append_json(output_dir / "generation_trace.jsonl", {"run_id": state.data["run_id"],
                "episode_id": state.data["episode_id"], "stage": state.data["stage"], **event})
        client.trace_callback = trace
        vc = model_cfg.get("vector_channel", {})
        projection = None
        if channel_type == "text":
            channel = TextChannel()
        else:
            projection = GaussianProjection(source_dim=vc["hidden_dim"], target_dim=vc["projected_dim"], seed=vc.get("projection_seed", 42))
            channel = VectorChannel(dim=vc["projected_dim"], bits_per_component=vc.get("bits_per_component", 4),
                                    layer_index=vc.get("extraction_layer", vc.get("layer_index", vc["injection_layer"])), projection=projection)
        fallback = TextChannel() if channel_type == "hybrid" else None
        metadata = {"schema_version": 2, "run_id": state.data["run_id"], "status": "running",
                    "experiment_name": cfg["experiment"]["name"], "model": client.name, "model_revision": client.revision,
                    "model_config": model_cfg, "model_config_sha256": model_file_hash,
                    "loader_config_sha256": getattr(client, "loader_config_fingerprint", None),
                    "model_device_map": getattr(client, "hf_device_map", None),
                    "task": task.name, "dataset": task.dataset_id, "dataset_sha256": dataset_hash,
                    "dataset_manifest_sha256": manifest_file_hash,
                    "split": task_cfg.get("split", "dev"), "channel": channel_type, "budget_bits": budget.cap_bits,
                    "wall_time_definition": "per-episode inference, orchestration and durable logging; not pure model latency",
                    "sim_delay_definition": "sum of attempted-link serialization plus propagation, including network-lost packets; budget-blocked packets incur none",
                    "network_profile": profile.name, "network": asdict(profile), "n_requested": n_requested,
                    "n_episodes": len(episodes), "episode_ids": [ep.episode_id for ep in episodes],
                    "episode_sha256": {ep.episode_id: fingerprint(asdict(ep)) for ep in episodes},
                    "max_rounds": max_rounds, "seed": seed, "protocol": protocol,
                    "config_file": str(config), "config_sha256": config_hash, "effective_config": cfg,
                    "effective_config_sha256": fingerprint(cfg),
                    "code_root": str(code_root), **source}
        atomic_json(output_dir / "run_metadata.json", metadata)
        state.update(status="running", stage="initialize_logging")
        logger = EpisodeLogger(output_dir.parent, output_dir.name,
                               store_messages=cfg["logging"].get("store_messages", True))
        atomic_json(output_dir / "episodes_manifest.json", {"dataset_sha256": dataset_hash, "episodes": [asdict(ep) for ep in episodes]})
        n_correct = 0
        for ep in episodes:
            state.update(stage="initialize_episode", episode_id=ep.episode_id)
            started = time.perf_counter()
            budget.reset()
            episode_seed = int(hashlib.sha256(f"{seed}:{ep.episode_id}".encode()).hexdigest()[:8], 16)
            network = NetworkSimulator(profile, rng_seed=episode_seed)
            if channel_type == "text":
                agents = {identity: KnowledgeHolderAgent(identity, _other(identity), client) for identity in ("agent_a", "agent_b")}
            else:
                agents = {identity: VectorKnowledgeHolderAgent(identity, _other(identity), client, projection,
                    injection_layer=vc["injection_layer"],
                    extraction_layer=vc.get("extraction_layer", vc.get("layer_index", vc["injection_layer"])),
                    pooling=vc.get("pooling", "last_token"),
                    injection_strength=channel_cfg.get("injection_strength", vc.get("injection_strength", 1.0)),
                    random_injection=channel_cfg.get("random_injection", False), random_seed=episode_seed)
                    for identity in ("agent_a", "agent_b")}
            if task.name == "secret_passing":
                sender_key = "alternate_sender_text" if protocol["sender_value"] == "alternate" else "sender_text"
                if sender_key not in ep.initial_state:
                    raise ValueError("alternate control requires a frozen alternate sender record")
                contexts = {"agent_a": {"passage": ep.initial_state[sender_key]},
                            "agent_b": {"passage": "No private lookup record is available to you."}}
            else:
                a, b = ep.initial_state["passage_a"], ep.initial_state["passage_b"]
                if protocol["swap_contexts"]:
                    a, b = b, a
                contexts = {"agent_a": {"passage": a}, "agent_b": {"passage": b}}
            transcript = []
            n_attempted = n_messages = n_dropped = n_blocked = 0
            delay_ms = 0.0
            sender_messages = []
            def observation(identity, final=False):
                public_state = {"question": ep.initial_state["question"], "is_final_round": final}
                if task_cfg["name"] == "secret_passing" and "record_key" in ep.initial_state:
                    public_state["record_key"] = ep.initial_state["record_key"]
                return AgentObservation(transcript=list(transcript), private_context=contexts[identity],
                    task_state=public_state)
            communication_agents = ("agent_a",) if protocol["direction"] == "sender_to_receiver" else ("agent_a", "agent_b")
            for round_index in range(max_rounds):
                for identity in communication_agents:
                    state.update(stage="communicate_generate")
                    if protocol["sender_mode"] == "oracle":
                        value = ep.initial_state["alternate_ground_truth"] if protocol["sender_value"] == "alternate" else ep.ground_truth
                        # Explicit oracle control: evaluator-supplied source content, never a generated message claim.
                        vector = None
                        if projection is not None:
                            vector = client.extract_activation(value,
                                layer_index=vc.get("extraction_layer", vc.get("layer_index", vc["injection_layer"])),
                                pooling=vc.get("pooling", "last_token")).detach().cpu().numpy().astype(np.float32)
                        action = AgentAction(message=value, vector=vector)
                    else:
                        action = agents[identity].act(observation(identity))
                    if identity == "agent_a":
                        sender_messages.append(action.message or "")
                    state.update(stage="encode")
                    fallback_used = channel_type == "hybrid" and budget.remaining_bits() < channel.bit_cost_for_message()
                    active_channel = fallback if fallback_used else channel
                    payload = (action.message or "") if channel_type == "text" or fallback_used else action.vector
                    msg = active_channel.encode(payload, sender_id=identity, receiver_id=_other(identity))
                    if msg.bit_cost != len(msg.payload) * 8:
                        raise ValueError("reported bit cost differs from actual wire bytes")
                    n_attempted += 1
                    allowed = budget.attempt_spend(msg.bit_cost)
                    state.update(stage="transmit")
                    delivered = network.transmit(msg) if allowed else None
                    simulated_delay = network.last_delay_ms if allowed else 0.0
                    delay_ms += simulated_delay
                    dropped = bool(allowed and delivered is None)
                    state.update(stage="message_log")
                    logger.log_message(episode_id=ep.episode_id, round=round_index, sender=identity, receiver=_other(identity),
                        channel="text_fallback" if fallback_used else channel_type, bit_cost=msg.bit_cost,
                        allowed=allowed, blocked=not allowed, dropped=dropped, payload_text=action.message,
                        payload_sha256=hashlib.sha256(msg.payload).hexdigest(), sim_delay_ms=simulated_delay, is_final=False,
                        sender_mode=protocol["sender_mode"])
                    if cfg["logging"].get("store_vectors", False) and projection is not None and not fallback_used:
                        directory = output_dir / "vector_artifacts"
                        directory.mkdir(exist_ok=True)
                        prefix = f"{len(sender_messages):04d}_{round_index}_{identity}_{hashlib.sha256(ep.episode_id.encode()).hexdigest()[:16]}"
                        np.save(directory / (prefix + "_source.npy"), np.asarray(action.vector), allow_pickle=False)
                        (directory / (prefix + "_wire.bin")).write_bytes(msg.payload)
                    if not allowed:
                        n_blocked += 1
                        continue
                    if dropped:
                        n_dropped += 1
                        continue
                    n_messages += 1
                    state.update(stage="decode")
                    decoded = active_channel.decode(delivered)
                    if channel_type == "text" or fallback_used:
                        transcript.append({"sender": identity, "text": decoded})
                    else:
                        state.update(stage="receive")
                        agents[_other(identity)].process_received_vector(decoded)
            # Final answer is an observation-local evaluation, NEVER a free partner message.
            state.update(stage="final_generate")
            final_action = agents["agent_b"].act(observation("agent_b", final=True))
            if final_action.task_action is None or not isinstance(final_action.task_action, str):
                raise ValueError("evaluated receiver did not produce a final task action")
            final_answer = final_action.task_action.strip()
            state.update(stage="score")
            success, score = task.score(ep, final_answer)
            if not isinstance(success, bool) or not np.isfinite(score):
                raise ValueError("scorer returned an invalid measurement")
            n_correct += int(success)
            alternate = ep.initial_state.get("alternate_ground_truth")
            state.update(stage="finalize_episode")
            logger.finalize_episode(episode_id=ep.episode_id, success=success, score=score,
                final_answer=final_answer, ground_truth=ep.ground_truth, bits_used=budget.state.spent_bits,
                n_attempted=n_attempted, n_messages=n_messages, n_dropped=n_dropped, n_blocked=n_blocked,
                sim_delay_ms=delay_ms, wall_time_ms=round((time.perf_counter()-started)*1000.0, 3),
                final_agent="agent_b", status="completed", episode_sha256=fingerprint(asdict(ep)),
                evaluation_status="abstained" if not final_answer else "scored",
                alternate_ground_truth=alternate or "", alternate_match=bool(alternate and final_answer.upper()==alternate.upper()),
                sender_contains_target=bool(task.name=="secret_passing" and any(ep.ground_truth.upper() in m.upper() for m in sender_messages)))
            state.update(n_completed=state.data["n_completed"] + 1, stage="episode_completed")
            click.echo(f"[{state.data['n_completed']}/{n_requested}] {ep.episode_id} answer={final_answer!r} success={success} bits={budget.state.spent_bits}")
        state.update(stage="verify_artifacts")
        if hashlib.sha256(config.read_bytes()).hexdigest() != config_hash or source_identity(code_root)["source_sha256"] != source["source_sha256"]:
            raise RuntimeError("source/configuration changed during the attempt; measurements cannot be certified")
        if hashlib.sha256(model_cfg_path.read_bytes()).hexdigest() != model_file_hash or (
            manifest_path is not None and hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_file_hash
        ):
            raise RuntimeError("model configuration or data manifest changed during the attempt")
        metadata.update(status="completed", n_completed=n_requested)
        atomic_json(output_dir / "run_metadata.json", metadata)
        state.update(status="completed", stage="completed", episode_id=None, finished_utc=utc_now())
        hashes = {path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(output_dir.rglob("*")) if path.is_file()}
        atomic_json(output_dir / "artifact_manifest.json", {"schema_version": 2, "run_id": state.data["run_id"], "files": hashes})
        click.echo(f"=== DONE === Completed {n_requested} episodes; exact match {n_correct}/{n_requested}. Results: {output_dir}")
        return output_dir
    except BaseException as error:
        state.fail(error)
        raise
    finally:
        if client is not None:
            client.trace_callback = previous_trace_callback


@click.group()
def cli() -> None:
    """Budget-fair experiment runner with durable, validated outcomes."""


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--project-root", default="/app", type=click.Path(file_okay=False, path_type=Path))
def run(config: Path, project_root: Path) -> None:
    run_experiment(config, project_root)


if __name__ == "__main__":
    cli()
