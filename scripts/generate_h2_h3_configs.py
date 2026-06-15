"""Generate H2 (budget sweep) and H3 (task-type contrast) configs."""
from pathlib import Path

OUT = Path(__file__).parent.parent / "configs" / "sweep_h2h3"
OUT.mkdir(parents=True, exist_ok=True)
LLAMA = "configs/llama_3_1_8b.yaml"


def cfg(name, channel, budget_bits, agents_role, question_type=None):
    qt = f"  question_type: {question_type}\n" if question_type else ""
    return f"""experiment:
  name: {name}
  description: "H2/H3 cell: {name}"
  seed: 42
model_config: {LLAMA}
agents: {{ count: 2, roles: [{agents_role}_a, {agents_role}_b] }}
task:
  name: multidoc_qa
  dataset: hotpotqa_dev
  num_episodes: 30
  max_communication_rounds: 4
{qt}channel: {{ type: {channel}, budget_bits: {budget_bits} }}
network: {{ bandwidth_bps: null, latency_ms: 0, packet_loss: 0.0 }}
logging: {{ output_dir: data/results/h2h3_{name}, log_level: INFO, store_messages: true, store_vectors: false }}
"""


def role(channel):
    return "knowledge_holder" if channel == "text" else "vector_knowledge_holder"


count = 0

# H2 — Budget sweep across 7 levels x 3 channels (Llama, clean)
H2_BUDGETS = [
    ("0p5k", 500),
    ("1k", 1000),
    ("3k", 3000),
    ("4k", 4000),
    ("6k", 6000),
    ("12k", 12000),
    ("16k", 16000),
]
for ch in ["text", "vector", "hybrid"]:
    for bname, bbits in H2_BUDGETS:
        name = f"h2_{ch}_{bname}"
        (OUT / f"{name}.yaml").write_text(cfg(name, ch, bbits, role(ch)), encoding="utf-8")
        count += 1

# H3 — Task-type contrast: bridge vs comparison
# 2 channels x 2 types x 2 budgets (High, Low) = 8 cells
for ch in ["text", "vector"]:
    for qtype in ["bridge", "comparison"]:
        for bname, bbits in [("high", 8000), ("low", 2000)]:
            name = f"h3_{ch}_{qtype}_{bname}"
            (OUT / f"{name}.yaml").write_text(
                cfg(name, ch, bbits, role(ch), question_type=qtype), encoding="utf-8"
            )
            count += 1

print(f"wrote {count} configs to {OUT}")
