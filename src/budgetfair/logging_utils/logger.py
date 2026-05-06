"""Episode logger — appends one JSONL line per message, plus an episode-summary row.

JSONL message schema:
  {ts, episode_id, sender, receiver, channel, payload_kind, bit_cost, dropped,
   payload_text, payload_vector_path}

Vectors are written to data/results/<run>/vectors/<episode_id>/<msg_idx>.npy.gz
to keep the JSONL small.
"""
from __future__ import annotations

from pathlib import Path


class EpisodeLogger:
    def __init__(self, output_dir: Path, run_name: str) -> None:
        self.output_dir = Path(output_dir) / run_name
        self.run_name = run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_message(self, **fields) -> None:
        raise NotImplementedError

    def finalize_episode(self, **fields) -> None:
        raise NotImplementedError
