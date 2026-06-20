"""Experiment orchestrator. Reads a YAML config, runs episodes, logs results.

Pilot scope: 1 model, 1 task (multidoc_qa), text channel, no budget cap.
For the full matrix, the runner will be expanded to iterate over all cells.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import click
import yaml

from .agents.base import AgentObservation
from .agents.knowledge_holder import KnowledgeHolderAgent
from .agents.vector_knowledge_holder import VectorKnowledgeHolderAgent
from .budget.controller import BudgetController, OverbudgetPolicy
from .comm.projection import GaussianProjection
from .comm.text_channel import TextChannel
from .comm.vector_channel import VectorChannel
from .logging_utils.logger import EpisodeLogger
from .model.client import LLMClient
from .network.simulator import NetworkProfile, NetworkSimulator
from .tasks.multidoc_qa import MultiDocQA
from .tasks.secret_passing import SecretPassingTask


def _resolve(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (project_root / p)


def _other(agent_id: str) -> str:
    return "agent_b" if agent_id == "agent_a" else "agent_a"


@click.group()
def cli() -> None:
    """budgetfair CLI."""


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--project-root",
    default="/app",
    type=click.Path(file_okay=False, path_type=Path),
    help="Anchor for relative paths in the config (default /app inside container).",
)
def run(config: Path, project_root: Path) -> None:
    """Run an experiment from a YAML config (pilot version)."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))

    # === Resolve subcomponents ============================================
    model_cfg_path = _resolve(cfg["model_config"], project_root)
    click.echo(f"[1/4] Loading model from {model_cfg_path} ...")
    client = LLMClient.from_config(model_cfg_path)
    click.echo(f"      {client.name} loaded.")

    task_cfg = cfg["task"]
    if task_cfg["name"] == "multidoc_qa":
        task = MultiDocQA(dataset=task_cfg["dataset"], question_type=task_cfg.get("question_type"))
    elif task_cfg["name"] == "secret_passing":
        task = SecretPassingTask(secret_mode=task_cfg.get("secret_mode", "both"))
    else:
        raise NotImplementedError(f"Task {task_cfg['name']!r} not supported.")

    n_episodes = int(task_cfg["num_episodes"])
    seed = int(cfg["experiment"].get("seed", 42))
    click.echo(f"[2/4] Loading {n_episodes} episodes from {task.dataset_id} ...")
    episodes = task.episodes(n_episodes, seed)
    click.echo(f"      {len(episodes)} episodes ready.")

    channel_cfg = cfg["channel"]
    channel_type = channel_cfg["type"]
    if channel_type == "text":
        channel = TextChannel()
        projection = None
    elif channel_type in ("vector", "hybrid"):
        # Pull projection / quantization params from the model config's
        # vector_channel block (filled by configs/<model>.yaml).
        model_cfg = yaml.safe_load(model_cfg_path.read_text(encoding="utf-8"))
        vc_cfg = model_cfg.get("vector_channel", {})
        hidden_dim = int(vc_cfg.get("hidden_dim", 4096))
        projected_dim = int(vc_cfg.get("projected_dim", 128))
        bits_per_component = int(vc_cfg.get("bits_per_component", 4))
        projection_seed = int(vc_cfg.get("projection_seed", 42))
        # Activation-injection params (ref:1). Default to a middle layer if a
        # legacy config omits injection_layer.
        injection_layer = int(vc_cfg.get("injection_layer", vc_cfg.get("layer_index", 16)))
        # Experiment channel block can override the model-config default (for sweeps).
        injection_strength = float(
            channel_cfg.get("injection_strength", vc_cfg.get("injection_strength", 1.0))
        )
        projection = GaussianProjection(
            source_dim=hidden_dim,
            target_dim=projected_dim,
            seed=projection_seed,
        )
        channel = VectorChannel(
            dim=projected_dim,
            bits_per_component=bits_per_component,
            layer_index=int(vc_cfg.get("layer_index", 0)),
            projection=projection,
        )
        click.echo(
            f"      vector channel: {hidden_dim}->{projected_dim}, "
            f"{bits_per_component}-bit, {channel.bit_cost_for_message()} bits/msg"
        )
        click.echo(
            f"      activation injection: layer {injection_layer}, "
            f"strength {injection_strength}"
        )
    else:
        raise NotImplementedError(f"Channel type {channel_type!r} not supported.")
    budget = BudgetController(
        cap_bits=channel_cfg.get("budget_bits"),
        policy=OverbudgetPolicy.BLOCK,
    )

    network_cfg = cfg.get("network", {}) or {}
    network = NetworkSimulator(
        NetworkProfile(
            name=str(network_cfg.get("name", "clean")),
            bandwidth_bps=network_cfg.get("bandwidth_bps"),
            latency_ms=float(network_cfg.get("latency_ms", 0.0)),
            packet_loss=float(network_cfg.get("packet_loss", 0.0)),
        ),
        rng_seed=seed,
    )

    log_cfg = cfg["logging"]
    output_dir = _resolve(log_cfg["output_dir"], project_root)
    logger = EpisodeLogger(output_dir.parent, output_dir.name)
    click.echo(f"[3/4] Logging to {logger.output_dir}")

    if channel_type == "text":
        agent_a = KnowledgeHolderAgent("agent_a", "agent_b", client)
        agent_b = KnowledgeHolderAgent("agent_b", "agent_a", client)
    else:  # vector or hybrid — both use the vector agent (it produces both message+vector)
        assert projection is not None
        random_injection = bool(channel_cfg.get("random_injection", False))
        if random_injection:
            click.echo("      ABLATION: random-vector injection ON (content destroyed, norm matched)")
        agent_a = VectorKnowledgeHolderAgent(
            "agent_a", "agent_b", client, projection,
            injection_layer=injection_layer, injection_strength=injection_strength,
            random_injection=random_injection, random_seed=seed,
        )
        agent_b = VectorKnowledgeHolderAgent(
            "agent_b", "agent_a", client, projection,
            injection_layer=injection_layer, injection_strength=injection_strength,
            random_injection=random_injection, random_seed=seed,
        )
    # For hybrid, build a fallback text channel too.
    fallback_text_channel = TextChannel() if channel_type == "hybrid" else None
    agents_by_id = {agent_a.agent_id: agent_a, agent_b.agent_id: agent_b}

    # === Run metadata =====================================================
    run_metadata = {
        "experiment_name": cfg["experiment"]["name"],
        "model": client.name,
        "model_revision": client.revision,
        "task": task.name,
        "dataset": task.dataset_id,
        "channel": channel.name,
        "budget_bits": budget.cap_bits,
        "network_profile": network.profile.name,
        "n_episodes": len(episodes),
        "max_rounds": int(task_cfg.get("max_communication_rounds", 4)),
        "seed": seed,
        "config_file": str(config),
    }
    (logger.output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    # === Episode loop =====================================================
    max_rounds = int(task_cfg.get("max_communication_rounds", 4))
    click.echo(f"[4/4] Running {len(episodes)} episodes x {max_rounds} rounds ...\n")

    n_correct = 0
    for ep_idx, ep in enumerate(episodes):
        episode_start = time.time()
        budget.reset()
        transcript: list[dict[str, str]] = []
        n_messages = 0
        n_dropped = 0
        n_blocked = 0
        final_answer: Optional[str] = None
        final_answers: dict[str, str] = {}  # per-agent final answers (for directional tasks)

        if task.name == "secret_passing":
            # Directional: agent_a is the SENDER (holds the secret), agent_b is the
            # RECEIVER (no info — must rely entirely on the channel). Only the
            # receiver's answer is scored.
            contexts = {
                "agent_a": {"passage": ep.initial_state["sender_text"]},
                "agent_b": {"passage": "(You have NO information about the answer. "
                                       "You can only rely on what your partner transmits to you.)"},
            }
        else:
            contexts = {
                "agent_a": {"passage": ep.initial_state["passage_a"]},
                "agent_b": {"passage": ep.initial_state["passage_b"]},
            }

        for round_idx in range(max_rounds):
            is_final = round_idx == max_rounds - 1
            for agent in (agent_a, agent_b):
                obs = AgentObservation(
                    transcript=list(transcript),
                    private_context=contexts[agent.agent_id],
                    task_state={
                        "question": ep.initial_state["question"],
                        "is_final_round": is_final,
                    },
                )
                action = agent.act(obs)
                # Encode the appropriate payload per channel type.
                used_text_fallback = False
                if channel_type == "text":
                    msg = channel.encode(
                        action.message or "",
                        sender_id=agent.agent_id,
                        receiver_id=_other(agent.agent_id),
                    )
                elif channel_type == "vector":
                    msg = channel.encode(
                        action.vector,
                        sender_id=agent.agent_id,
                        receiver_id=_other(agent.agent_id),
                    )
                else:  # hybrid: vector if budget allows, else text
                    vector_cost = channel.bit_cost_for_message()
                    if budget.remaining_bits() >= vector_cost:
                        msg = channel.encode(
                            action.vector,
                            sender_id=agent.agent_id,
                            receiver_id=_other(agent.agent_id),
                        )
                    else:
                        # Fall back to text — a 1-byte message fits any positive budget.
                        msg = fallback_text_channel.encode(
                            action.message or "",
                            sender_id=agent.agent_id,
                            receiver_id=_other(agent.agent_id),
                        )
                        used_text_fallback = True
                allowed = budget.attempt_spend(msg.bit_cost)
                delivered = network.transmit(msg) if allowed else None

                logger.log_message(
                    episode_id=ep.episode_id,
                    round=round_idx,
                    sender=agent.agent_id,
                    receiver=_other(agent.agent_id),
                    channel=channel.name if not used_text_fallback else "text_fallback",
                    bit_cost=msg.bit_cost,
                    allowed=allowed,
                    dropped=(delivered is None),
                    payload_text=action.message,  # logged for both channels (debugging)
                    is_final=is_final,
                )

                if not allowed:
                    n_blocked += 1
                    continue
                if delivered is None:
                    n_dropped += 1
                    continue

                n_messages += 1
                if channel_type == "text" or used_text_fallback:
                    text = (channel if channel_type == "text" else fallback_text_channel).decode(delivered)
                    transcript.append({"sender": agent.agent_id, "text": text})
                else:  # vector
                    decoded_vec = channel.decode(delivered)  # (target_dim,)
                    receiver_agent = agents_by_id[_other(agent.agent_id)]
                    receiver_agent.process_received_vector(decoded_vec)
                if is_final and action.task_action is not None:
                    # Tie-break: take the longer final answer. Rationale: in pilot
                    # smoke testing both agents emit a final answer; the longer one
                    # tends to carry more synthesis (e.g. "The Yoruba and the Itsekiri"
                    # from agent_a vs "Itsekiri" from agent_b). Length is a noisy proxy
                    # for informativeness but defensibly better than just-take-agent_b.
                    candidate = action.task_action.strip()
                    if final_answer is None or len(candidate) > len(final_answer):
                        final_answer = candidate
                    final_answers[agent.agent_id] = candidate

        # Directional tasks score the RECEIVER (agent_b); others use the tie-break.
        if task.name == "secret_passing":
            scored_answer = final_answers.get("agent_b")
            final_answer = scored_answer  # log/report the scored (receiver) answer
        else:
            scored_answer = final_answer
        success, score = task.score(ep, scored_answer)
        n_correct += int(success)
        wall_time_ms = (time.time() - episode_start) * 1000.0

        logger.finalize_episode(
            episode_id=ep.episode_id,
            success=success,
            score=score,
            final_answer=final_answer,
            ground_truth=ep.ground_truth,
            bits_used=budget.state.spent_bits,
            n_messages=n_messages,
            n_dropped=n_dropped,
            n_blocked=n_blocked,
            wall_time_ms=round(wall_time_ms, 1),
        )
        click.echo(
            f"[{ep_idx+1}/{len(episodes)}] {ep.episode_id} "
            f"pred={final_answer!r} gold={ep.ground_truth!r} "
            f"em={success} bits={budget.state.spent_bits} "
            f"msgs={n_messages} time={wall_time_ms:.0f}ms"
        )

    click.echo(
        f"\n=== DONE === EM accuracy: {n_correct}/{len(episodes)} = "
        f"{n_correct / max(1, len(episodes)):.1%}"
    )
    click.echo(f"Results: {logger.output_dir}")


if __name__ == "__main__":
    cli()
