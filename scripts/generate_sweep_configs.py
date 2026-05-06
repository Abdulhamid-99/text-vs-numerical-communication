"""Generate all sweep configs for the H1/H2 + network + Qwen + SVD experiments.

Run once locally; produces YAML files under configs/sweep/. Each YAML is a
pre-built experiment runnable directly via `python -m budgetfair.runner run`.
"""
from __future__ import annotations

from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "configs" / "sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LLAMA = "configs/llama_3_1_8b.yaml"
QWEN = "configs/qwen2_5_7b.yaml"

NETWORKS = [
    ("clean", 0.0),
    ("loss5", 0.05),
    ("loss10", 0.10),
]
BUDGETS = [("high", 8000), ("low", 2000)]
CHANNELS_LLAMA = ["text", "vector", "hybrid"]
CHANNELS_QWEN = ["text", "vector"]  # skip hybrid for Qwen to save time

EPISODES = 30
SEED = 42


def cfg(name, model, channel, budget_bits, packet_loss, agents_role):
    return f"""experiment:
  name: {name}
  description: "sweep cell: {name}"
  seed: {SEED}
model_config: {model}
agents: {{ count: 2, roles: [{agents_role}_a, {agents_role}_b] }}
task: {{ name: multidoc_qa, dataset: hotpotqa_dev, num_episodes: {EPISODES}, max_communication_rounds: 4 }}
channel: {{ type: {channel}, budget_bits: {budget_bits} }}
network: {{ bandwidth_bps: null, latency_ms: 0, packet_loss: {packet_loss} }}
logging: {{ output_dir: data/results/sweep_{name}, log_level: INFO, store_messages: true, store_vectors: false }}
"""


def role_for(channel):
    return "knowledge_holder" if channel == "text" else "vector_knowledge_holder"


count = 0
for model_name, model_path, channels in [("llama", LLAMA, CHANNELS_LLAMA), ("qwen", QWEN, CHANNELS_QWEN)]:
    for ch in channels:
        for bname, bbits in BUDGETS:
            for nname, ploss in NETWORKS:
                # Skip the 4 H1 cells (already run): llama × {text,vector} × {high,low} × clean
                if model_name == "llama" and ch in ("text", "vector") and nname == "clean":
                    continue
                name = f"{model_name}_{ch}_{bname}_{nname}"
                role = role_for(ch)
                path = OUT_DIR / f"{name}.yaml"
                path.write_text(cfg(name, model_path, ch, bbits, ploss, role), encoding="utf-8")
                count += 1
print(f"wrote {count} configs to {OUT_DIR}")
print(f"List of cells:")
for p in sorted(OUT_DIR.glob("*.yaml")):
    print(f"  {p.name}")
