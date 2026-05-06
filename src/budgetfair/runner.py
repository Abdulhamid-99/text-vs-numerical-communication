"""Experiment orchestrator. Reads a YAML config, expands it into cells, runs each, logs results.

Not implemented yet — this is the pilot-study entry point we will fill in first.
"""
from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """budgetfair CLI."""


@cli.command()
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False))
def run(config: str) -> None:
    """Run an experiment from a YAML config.

    Pilot version: load model, instantiate two agents, run N episodes of one task on one channel,
    write JSONL message logs + summary CSV. No budget cap, no network simulator yet.
    """
    raise NotImplementedError("Pilot implementation lands in Phase 2 (target: 2026-05-15).")


if __name__ == "__main__":
    cli()
