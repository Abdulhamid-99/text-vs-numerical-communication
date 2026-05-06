"""Layer-selection pilot — pick the transformer layer whose mean-pooled hidden
state best encodes semantic similarity, per
docs/layer_selection_method.md.

Method: STS-B sentence pairs (validation split, 1500 pairs) -> for each layer
in the model, compute cosine(pool(s1), pool(s2)) for each pair, then Spearman-
correlate against the human similarity score. The winning layer is the one
with the highest Spearman rho.

Usage (inside the docker container):

    python scripts/select_layer.py --config configs/llama_3_1_8b.yaml \
        --num-pairs 200 --output data/results/layer_selection_llama.csv

Output:
  - CSV with columns: layer_index, spearman, pearson, mean_norm
  - Stdout: best layer + its scores
  - The user is then expected to update `vector_channel.layer_index` in the
    relevant config YAML to the printed best layer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _load_model(cfg: dict):
    name = cfg["model"]["name"]
    revision = cfg["model"].get("revision")
    quantization = cfg["model"].get("quantization", "int8")
    device = cfg["model"].get("device", "cuda:0")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        cfg["model"].get("torch_dtype", "bfloat16"), torch.bfloat16
    )

    bnb = None
    if quantization == "int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "int4":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    from inspect import signature
    kwargs = {
        "revision": revision,
        "quantization_config": bnb,
        "device_map": "auto" if bnb is not None else device,
    }
    params = signature(AutoModelForCausalLM.from_pretrained).parameters
    kwargs["dtype" if "dtype" in params else "torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def _hidden_states_mean_pool(
    model, tokenizer, text: str, max_length: int = 256
) -> np.ndarray:
    """Run text through the model, return mean-pooled hidden state PER LAYER.

    Output shape: (n_layers + 1, hidden_dim) — index 0 is the embedding output,
    index 1..n_layers are the n_layers transformer block outputs.
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(model.device)
    out = model(**inputs, output_hidden_states=True, return_dict=True)
    # out.hidden_states is a tuple of length (n_layers + 1)
    mask = inputs.attention_mask.unsqueeze(-1).to(out.hidden_states[0].dtype)
    pooled = []
    for hs in out.hidden_states:
        # (1, T, d) * (1, T, 1) -> mean over non-pad
        p = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled.append(p.squeeze(0).float().cpu().numpy())
    return np.stack(pooled, axis=0)  # (n_layers+1, d)


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--num-pairs", default=200, help="Number of STS-B pairs to use.")
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="CSV output path (relative to /app inside container).",
)
def main(config: Path, num_pairs: int, output: Path) -> None:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    click.echo(f"[1/4] Loading model {cfg['model']['name']} ...")
    model, tokenizer = _load_model(cfg)
    click.echo("      model loaded.")

    click.echo("[2/4] Loading STS-B validation pairs ...")
    from datasets import load_dataset

    ds = load_dataset("glue", "stsb", split="validation")
    # Sort by idx for determinism, take first N
    pairs = sorted(ds, key=lambda r: r["idx"])[:num_pairs]
    click.echo(f"      {len(pairs)} pairs loaded.")

    click.echo(f"[3/4] Computing layer-wise pooled hidden states ...")
    # We'll accumulate per-layer cosine similarities for each pair
    # First pass to figure out n_layers by encoding one sentence
    first_pooled = _hidden_states_mean_pool(model, tokenizer, pairs[0]["sentence1"])
    n_layers_plus_1 = first_pooled.shape[0]
    hidden_dim = first_pooled.shape[1]
    click.echo(f"      n_layers+1 = {n_layers_plus_1}, hidden_dim = {hidden_dim}")

    cosines = np.zeros((len(pairs), n_layers_plus_1), dtype=np.float64)
    norms = np.zeros((len(pairs) * 2, n_layers_plus_1), dtype=np.float64)
    human_scores = np.zeros(len(pairs), dtype=np.float64)

    for i, pair in enumerate(tqdm(pairs, desc="pairs")):
        h1 = _hidden_states_mean_pool(model, tokenizer, pair["sentence1"])
        h2 = _hidden_states_mean_pool(model, tokenizer, pair["sentence2"])
        # Per-layer cosine
        n1 = np.linalg.norm(h1, axis=1) + 1e-12
        n2 = np.linalg.norm(h2, axis=1) + 1e-12
        cosines[i] = np.sum(h1 * h2, axis=1) / (n1 * n2)
        norms[i * 2] = n1
        norms[i * 2 + 1] = n2
        human_scores[i] = pair["label"]

    click.echo("[4/4] Computing per-layer correlations ...")
    rows = []
    for layer_idx in range(n_layers_plus_1):
        cos_layer = cosines[:, layer_idx]
        sp = spearmanr(cos_layer, human_scores).correlation
        pe = pearsonr(cos_layer, human_scores).statistic
        mean_norm = float(np.mean(norms[:, layer_idx]))
        rows.append((layer_idx, sp, pe, mean_norm))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("layer_index,spearman,pearson,mean_norm\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f}\n")

    # Find best by Spearman; ignore layer 0 (embedding output) since it's trivially
    # high correlation in some models due to subword overlap
    best_idx, best_sp, best_pe, best_norm = max(rows[1:], key=lambda r: r[1])
    click.echo(f"\n=== BEST LAYER ===")
    click.echo(f"layer_index = {best_idx}")
    click.echo(f"spearman    = {best_sp:.4f}")
    click.echo(f"pearson     = {best_pe:.4f}")
    click.echo(f"mean_norm   = {best_norm:.4f}")
    click.echo(f"\nFull table written to: {output}")
    click.echo(
        f"To use this in experiments, set vector_channel.layer_index = {best_idx} "
        f"in {config}"
    )


if __name__ == "__main__":
    main()
