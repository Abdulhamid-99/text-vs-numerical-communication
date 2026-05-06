"""Download model weights from HuggingFace into the local cache.

Pin the revisions in the YAML configs before running this in production — null revisions
mean "latest", which breaks reproducibility.
"""
from __future__ import annotations

import click


@click.command()
@click.option("--model", required=True, type=click.Choice(["llama-3.1-8b-instruct", "qwen2.5-7b-instruct"]))
def main(model: str) -> None:
    raise NotImplementedError("Stub: implement before pilot run on 2026-05-15.")


if __name__ == "__main__":
    main()
