"""Episode logger — appends one JSONL line per message, plus an episode-summary row.

JSONL message schema (open-ended, but these fields should always be present):
  {ts, episode_id, sender, receiver, channel, bit_cost, dropped, payload_text}

Vectors will be written to data/results/<run>/vectors/<episode_id>/<msg_idx>.npy.gz
when the vector channel is wired up; for the text-only pilot they don't appear.

Episode summary CSV schema (one row per episode):
  episode_id, success, score, bits_used, n_messages, n_dropped, wall_time_ms
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class EpisodeLogger:
    MESSAGES_FILE = "messages.jsonl"
    EPISODES_FILE = "episodes.csv"

    def __init__(self, output_dir: Path | str, run_name: str) -> None:
        self.output_dir = Path(output_dir) / run_name
        self.run_name = run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.messages_path = self.output_dir / self.MESSAGES_FILE
        self.episodes_path = self.output_dir / self.EPISODES_FILE
        self._episode_fieldnames: list[str] | None = None

    def log_message(self, **fields: Any) -> None:
        """Append one message-event row to messages.jsonl. UTF-8, no ensure_ascii so
        Arabic and other non-Latin scripts stay readable in the file."""
        with self.messages_path.open("a", encoding="utf-8") as f:
            json.dump(fields, f, ensure_ascii=False, default=str)
            f.write("\n")

    def finalize_episode(self, **fields: Any) -> None:
        """Append one row to episodes.csv. The first call defines the header order."""
        if self._episode_fieldnames is None:
            self._episode_fieldnames = list(fields.keys())
            write_header = not self.episodes_path.exists()
        else:
            # Maintain stable column order across calls.
            extra = [k for k in fields.keys() if k not in self._episode_fieldnames]
            self._episode_fieldnames.extend(extra)
            write_header = False

        with self.episodes_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._episode_fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(fields)
