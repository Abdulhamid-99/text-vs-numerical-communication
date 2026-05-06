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
from .budget.controller import BudgetController, OverbudgetPolicy
from .comm.text_channel import TextChannel
from .logging_utils.logger import EpisodeLogger
from .model.client import LLMClient
from .network.simulator import NetworkProfile, NetworkSimulator
from .tasks.multidoc_qa import MultiDocQA


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
    if task_cfg["name"] != "multidoc_qa":
        raise NotImplementedError(f"Task {task_cfg['name']!r} not supported in pilot.")
    task = MultiDocQA(dataset=task_cfg["dataset"])

    n_episodes = int(task_cfg["num_episodes"])
    seed = int(cfg["experiment"].get("seed", 42))
    click.echo(f"[2/4] Loading {n_episodes} episodes from {task.dataset_id} ...")
    episodes = task.episodes(n_episodes, seed)
    click.echo(f"      {len(episodes)} episodes ready.")

    channel_cfg = cfg["channel"]
    if channel_cfg["type"] != "text":
        raise NotImplementedError("Pilot supports text channel only.")
    channel = TextChannel()
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

    agent_a = KnowledgeHolderAgent("agent_a", "agent_b", client)
    agent_b = KnowledgeHolderAgent("agent_b", "agent_a", client)

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
                msg = channel.encode(
                    action.message or "",
                    sender_id=agent.agent_id,
                    receiver_id=_other(agent.agent_id),
                )
                allowed = budget.attempt_spend(msg.bit_cost)
                delivered = network.transmit(msg) if allowed else None

                logger.log_message(
                    episode_id=ep.episode_id,
                    round=round_idx,
                    sender=agent.agent_id,
                    receiver=_other(agent.agent_id),
                    channel=channel.name,
                    bit_cost=msg.bit_cost,
                    allowed=allowed,
                    dropped=(delivered is None),
                    payload_text=action.message,
                    is_final=is_final,
                )

                if not allowed:
                    n_blocked += 1
                    continue
                if delivered is None:
                    n_dropped += 1
                    continue

                n_messages += 1
                transcript.append(
                    {"sender": agent.agent_id, "text": channel.decode(delivered)}
                )
                if is_final and action.task_action is not None:
                    # Tie-break: take the LAST agent's answer (agent_b speaks second)
                    final_answer = action.task_action

        success, score = task.score(ep, final_answer)
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
