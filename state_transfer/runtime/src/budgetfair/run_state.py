"""Durable attempt state and provenance, including failures before model loading."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    tmp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def source_identity(root: Path) -> dict:
    hashes = {}
    for folder in ("src", "scripts"):
        for path in sorted((root / folder).rglob("*.py")):
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"commit_hash": commit, "source_sha256": fingerprint(hashes), "source_files": hashes,
            "python": platform.python_version()}


class RunState:
    def __init__(self, output_dir: Path, config_sha256: str) -> None:
        # Existing directories belong to previous attempts, even if incomplete.
        output_dir.mkdir(parents=True, exist_ok=False)
        self.path = output_dir / "run_status.json"
        self.data = {"schema_version": 2, "run_id": uuid.uuid4().hex,
                     "status": "initializing", "stage": "validate", "started_utc": utc_now(),
                     "finished_utc": None, "config_sha256": config_sha256,
                     "n_requested": None, "n_completed": 0, "episode_id": None, "error": None}
        self.update()

    def update(self, **values) -> None:
        self.data.update(values)
        atomic_json(self.path, self.data)

    def fail(self, error: BaseException) -> None:
        self.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                    finished_utc=utc_now(), error={"type": type(error).__name__, "message": str(error),
                        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))})
