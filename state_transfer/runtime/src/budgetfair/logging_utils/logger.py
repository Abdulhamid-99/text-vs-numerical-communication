"""Strict per-attempt research logging. Existing evidence is never resumed implicitly."""
from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_number(value: Any, name: str, *, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative {'integer' if integer else 'number'}")
    if not math.isfinite(value) or value < 0 or (integer and int(value) != value):
        raise ValueError(f"Invalid {name}: {value!r}")


class EpisodeLogger:
    MESSAGES_FILE = "messages.jsonl"
    EPISODES_FILE = "episodes.csv"

    def __init__(self, output_dir: Path | str, run_name: str, *, store_messages: bool = True) -> None:
        if not isinstance(run_name, str) or not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
            raise ValueError("run_name must be a single non-empty directory name")
        if not isinstance(store_messages, bool):
            raise ValueError("store_messages must be boolean")
        self.output_dir = Path(output_dir) / run_name
        self.run_name = run_name
        self.store_messages = store_messages
        if self.output_dir.exists():
            present = {p.name for p in self.output_dir.iterdir()}
            if present - {"run_status.json", "run_metadata.json"} or "run_status.json" not in present:
                raise FileExistsError(f"Attempt directory already exists: {self.output_dir}; choose a new attempt")
            status = json.loads((self.output_dir / "run_status.json").read_text(encoding="utf-8"))
            if status.get("status") not in {"initializing", "running"} or status.get("n_completed", 0) != 0:
                raise FileExistsError(f"Cannot resume previous attempt: {self.output_dir}")
        else:
            self.output_dir.mkdir(parents=True, exist_ok=False)
        with (self.output_dir / ".logger_claim").open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "instance_id": uuid4().hex}, stream)
        self.messages_path = self.output_dir / self.MESSAGES_FILE
        self.episodes_path = self.output_dir / self.EPISODES_FILE
        self._episode_fieldnames: list[str] | None = None
        self._episode_rows: list[dict[str, Any]] = []
        self._episode_ids: set[str] = set()
        self._messages: list[str] = []

    def log_message(self, **fields: Any) -> None:
        """Validate before recording; no implicit string fallback for invalid values."""
        for key in ("bit_cost", "round", "sim_delay_ms"):
            if key in fields:
                _finite_number(fields[key], key, integer=key != "sim_delay_ms")
        for key in ("allowed", "blocked", "dropped", "is_final"):
            if key in fields and not isinstance(fields[key], bool):
                raise ValueError(f"{key} must be boolean")
        serialized = json.dumps(fields, ensure_ascii=False, allow_nan=False)
        if not self.store_messages:
            return
        new_messages = [*self._messages, serialized]
        _atomic_text(self.messages_path, "\n".join(new_messages) + "\n")
        self._messages = new_messages

    def finalize_episode(self, **fields: Any) -> None:
        """Publish one unique completed episode with an immutable column schema."""
        required = {"episode_id", "success", "score"}
        if not required <= fields.keys():
            raise ValueError(f"Missing episode fields: {sorted(required - fields.keys())}")
        ep_id = fields["episode_id"]
        if not isinstance(ep_id, str) or not ep_id:
            raise ValueError("episode_id must be non-empty text")
        if ep_id in self._episode_ids:
            raise ValueError(f"Duplicate episode_id: {ep_id}")
        if not isinstance(fields["success"], bool):
            raise ValueError("success must be boolean; execution errors belong in run_status")
        _finite_number(fields["score"], "score")
        if fields["score"] > 1:
            raise ValueError("score must be in [0, 1]")
        for key in ("status", "execution_status"):
            if key in fields and fields[key] != "completed":
                raise ValueError("Only completed measurements belong in episodes.csv")
        for key in ("bits_used", "n_messages", "n_dropped", "n_blocked", "n_attempted", "n_sent", "n_delivered"):
            if key in fields:
                _finite_number(fields[key], key, integer=True)
        if "wall_time_ms" in fields:
            _finite_number(fields["wall_time_ms"], "wall_time_ms")
        if "sim_delay_ms" in fields:
            _finite_number(fields["sim_delay_ms"], "sim_delay_ms")
        accounting = {"n_attempted", "n_messages", "n_dropped", "n_blocked"}
        if accounting <= fields.keys() and fields["n_attempted"] != sum(fields[k] for k in accounting - {"n_attempted"}):
            raise ValueError("Episode message counters do not reconcile")
        json.dumps(fields, allow_nan=False)
        if self._episode_fieldnames is not None and set(fields) != set(self._episode_fieldnames):
            raise ValueError("Episode schema changed; use a new versioned attempt")
        names = self._episode_fieldnames or list(fields)
        rows = [*self._episode_rows, dict(fields)]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        _atomic_text(self.episodes_path, stream.getvalue())
        self._episode_fieldnames = names
        self._episode_rows = rows
        self._episode_ids.add(ep_id)
