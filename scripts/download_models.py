"""Download model weights from HuggingFace Hub into the HF cache.

Reads the model name + revision (commit sha) from a YAML config under configs/.
Always uses the pinned revision — null revisions are rejected to prevent silent
upgrades from invalidating previously-collected experiment results.

Usage (from inside the container, where HF_HOME is set to /root/.cache/huggingface
and the host cache is bind-mounted there):

    python scripts/download_models.py --config configs/llama_3_1_8b.yaml
    python scripts/download_models.py --config configs/qwen2_5_7b.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a model YAML config (e.g. configs/llama_3_1_8b.yaml).",
)
@click.option(
    "--allow-latest",
    is_flag=True,
    default=False,
    help="Allow null revision (download latest). DO NOT use for experiments.",
)
def main(config: Path, allow_latest: bool) -> None:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    name: str = cfg["model"]["name"]
    revision = cfg["model"].get("revision")

    if revision is None and not allow_latest:
        click.echo(
            f"ERROR: revision is null in {config}. Pin the commit sha first, or pass --allow-latest.",
            err=True,
        )
        sys.exit(2)

    click.echo(f"Downloading {name} @ {revision or 'latest'}")
    click.echo("This pulls ~16 GB. The bind-mounted HF cache means it survives rebuilds.\n")

    try:
        local_dir = snapshot_download(
            repo_id=name,
            revision=revision,
            # Skip large extras we don't use — original *.pth/*.bin alongside *.safetensors.
            ignore_patterns=["*.pth", "*.pt", "original/*", "*.gguf"],
        )
    except GatedRepoError as e:
        click.echo(
            f"ERROR: {name} is a gated model. Accept the license at "
            f"https://huggingface.co/{name} and run `huggingface-cli login` first.",
            err=True,
        )
        click.echo(str(e), err=True)
        sys.exit(3)
    except RepositoryNotFoundError as e:
        click.echo(f"ERROR: repo not found: {name}", err=True)
        click.echo(str(e), err=True)
        sys.exit(4)

    click.echo(f"\nOK. Local snapshot at: {local_dir}")


if __name__ == "__main__":
    main()
