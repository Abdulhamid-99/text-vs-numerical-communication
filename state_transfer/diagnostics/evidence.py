"""Read the CPU inputs used in the supplementary analyses."""

import csv
import hashlib
import json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def completed_manifest(folder, phase):
    manifest = read(folder / "artifact_manifest.json")["files"]
    status_file = folder / "run_status.json"
    require(
        sha(status_file) == manifest["run_status.json"]["sha256"],
        "Stage status digest differs",
    )
    status = read(status_file)
    require(
        status["status"] == "completed"
        and status["error"] is None
        and status["phase"] == phase
        and status["n_completed"] == status["n_requested"],
        "Evidence stage is incomplete or has the wrong phase",
    )
    return manifest


def write_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
