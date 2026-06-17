"""Measure end-to-end vector channel round-trip fidelity (for the methods section).

Pipeline: random source-dim vector -> project(4096->128) -> 4-bit quantize ->
dequantize -> lift(128->4096). Reports:
  - cosine(original, reconstructed)         (end-to-end)
  - cosine(projected, dequantized)          (quantization alone)
  - JL floor sqrt(target/source)            (= expected cosine of a random projection)

Usage: python scripts/measure_roundtrip.py --source-dim 4096 --target-dim 128 --bits 4
"""
from __future__ import annotations

import click
import numpy as np

from budgetfair.comm.projection import GaussianProjection
from budgetfair.comm.vector_channel import VectorChannel


@click.command()
@click.option("--source-dim", default=4096)
@click.option("--target-dim", default=128)
@click.option("--bits", default=4)
@click.option("--trials", default=200)
@click.option("--seed", default=42)
def main(source_dim: int, target_dim: int, bits: int, trials: int, seed: int) -> None:
    proj = GaussianProjection(source_dim=source_dim, target_dim=target_dim, seed=seed)
    ch = VectorChannel(dim=target_dim, bits_per_component=bits, projection=proj)
    rng = np.random.RandomState(seed)

    e2e, quant = [], []
    for _ in range(trials):
        v = rng.randn(source_dim).astype(np.float32)
        projected = proj.project(v)
        msg = ch.encode(v, "a", "b")          # projects then quantizes
        dequant = ch.decode(msg)              # (target_dim,)
        lifted = proj.lift(dequant)           # (source_dim,)

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        e2e.append(cos(v, lifted))
        quant.append(cos(projected, dequant))

    jl_floor = float(np.sqrt(target_dim / source_dim))
    print(f"source_dim={source_dim} target_dim={target_dim} bits={bits} trials={trials}")
    print(f"end-to-end cosine (orig vs reconstructed): mean={np.mean(e2e):.4f} std={np.std(e2e):.4f}")
    print(f"quantization-only cosine (proj vs dequant): mean={np.mean(quant):.4f} std={np.std(quant):.4f}")
    print(f"JL floor sqrt(target/source) = {jl_floor:.4f}  (random-projection baseline)")
    verdict = "NEAR-RANDOM (channel carries ~no info)" if np.mean(e2e) <= jl_floor * 1.3 else "carries signal"
    print(f"VERDICT: end-to-end fidelity is {verdict}")


if __name__ == "__main__":
    main()
