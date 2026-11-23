"""Fail-closed artifacts and task-neutral ordered-state interfaces for the thesis study."""
from __future__ import annotations

import hashlib
import errno
import json
import os
from pathlib import Path
import stat
import sys
import time
import traceback
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _replace_json_atomically(source, destination):
    """Tolerate a brief Windows bind-reader denial without dropping atomicity.

    A reader that omits FILE_SHARE_DELETE can temporarily deny os.replace on
    Docker Desktop. Only access/sharing permission denials are retried. The
    existing destination is never removed and persistent failure still raises.
    """
    started = time.monotonic()
    deadline = started + 3.0
    retries = 0
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if (error.errno not in (errno.EACCES, errno.EPERM)
                    and getattr(error, "winerror", None) not in (5, 32, 33)):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0 or retries >= 30:
                raise
            retries += 1
            print(f"Atomic JSON replace temporarily denied: {destination}; "
                  f"retry {retries}/30 within 3 seconds "
                  f"(errno={error.errno}, winerror={getattr(error, 'winerror', None)})",
                  file=sys.stderr, flush=True)
            time.sleep(min(0.1, remaining))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    with temp.open("wb") as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())
    _replace_json_atomically(temp, path)


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON member: " + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def read_json(path):
    return _strict_json(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [_strict_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def append_row(path, row):
    with Path(path).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _reject_link_components(path):
    # Check lexical components BEFORE resolve, including parent directory links.
    path = Path(path).absolute()
    for component in (path, *path.parents):
        info = component.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError("Symbolic/reparse evidence path is not permitted")


def within(root, relative):
    _reject_link_components(root)
    root = Path(root).resolve()
    if Path(relative).is_absolute():
        raise ValueError("Absolute evidence path is not permitted")
    lexical = root / relative
    try:
        _reject_link_components(lexical)
    except FileNotFoundError as error:
        raise ValueError("Missing evidence path") from error
    path = lexical.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Unsafe or missing evidence path")
    return path


def file_inventory(root):
    root = Path(root)
    _reject_link_components(root)
    records = {}
    for base, directories, files in os.walk(root, followlinks=False):
        for name in directories:
            _reject_link_components(Path(base) / name)
        for name in files:
            path = Path(base) / name
            _reject_link_components(path)
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("Nonregular evidence file")
            if path != root / "artifact_manifest.json":
                records[path.relative_to(root).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return dict(sorted(records.items()))


def _validate_status(status, *, require_completed=False):
    for field in ("n_requested", "n_completed"):
        if type(status.get(field)) is not int or status[field] < 0:
            raise ValueError("Stage counts must be nonnegative integers")
    if status["n_completed"] > status["n_requested"]:
        raise ValueError("Stage completed count exceeds request")
    if require_completed and (status.get("status") != "completed" or status.get("error") is not None
                              or status["n_completed"] != status["n_requested"]):
        raise ValueError("Cannot certify incomplete or errored stage")


def seal(root):
    root = Path(root)
    status = read_json(root / "run_status.json")
    _validate_status(status, require_completed=True)
    write_json(root / "artifact_manifest.json", {"schema": 1, "files": file_inventory(root)})


def verify_seal(root):
    root = Path(root)
    status = read_json(root / "run_status.json")
    _validate_status(status, require_completed=True)
    expected = read_json(root / "artifact_manifest.json")
    if (status["status"] != "completed" or status["n_completed"] != status["n_requested"]
            or expected != {"schema": 1, "files": file_inventory(root)}):
        raise ValueError("Incomplete, changed or unsealed stage: " + str(root))
    return status


class Stage:
    def __init__(self, directory, phase, requested=0, **fields):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.state = {"schema": 1, "phase": phase, "status": "running", "stage": "preflight",
                      "n_requested": requested, "n_completed": 0, "started_utc": utc(),
                      "finished_utc": None, "error": None, **fields}
        self.update()

    def update(self, **fields):
        candidate = {**self.state, **fields}
        _validate_status(candidate)
        self.state = candidate
        write_json(self.directory / "run_status.json", self.state)

    def finish(self):
        if self.state.get("status") != "running" or self.state.get("error") is not None:
            raise ValueError("Only a running, error-free stage may finish")
        if self.state["n_completed"] != self.state["n_requested"]:
            raise ValueError("Stage result count does not match request")
        self.update(status="completed", stage="finished", finished_utc=utc())
        seal(self.directory)

    def fail(self, error):
        self.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    finished_utc=utc(), error={"type": type(error).__name__, "message": str(error),
                                             "traceback": traceback.format_exc()})


def timed(call):
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def public_prompt(client, prefix_text, suffix_text):
    """A encoding sees prefix only; B-owned information belongs after the payload."""
    from budgetfair.model.span_transport import PublicPrompt
    encode = lambda value: tuple(client.tokenizer.encode(value, add_special_tokens=False))
    placeholder = encode("?")
    if len(placeholder) != 1:
        raise ValueError("Expected exactly one public placeholder token")
    result = PublicPrompt(encode(prefix_text), encode(suffix_text), placeholder[0])
    result.validate(client.model.config.vocab_size)
    return result


def render_receiver_parts(client, system, before, after):
    marker = "{{THESIS_MESSAGE_BOUNDARY_V1}}"
    if any(marker in part for part in (system, before, after)):
        raise ValueError("Payload marker in input context")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": before + marker + after}]
    rendered = client.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if rendered.count(marker) != 1:
        raise ValueError("Nonunique message boundary")
    prefix, suffix = rendered.split(marker)
    return prefix, suffix, messages


def generate(client, messages):
    import torch
    from budgetfair.model.span_transport import neutral_generation_kwargs
    inputs = client._chat_inputs(messages)
    with torch.inference_mode():
        output = client.model.generate(**inputs, **neutral_generation_kwargs(client))
    count = inputs.input_ids.shape[1]
    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] <= count:
        raise ValueError("Model produced no valid continuation")
    ids = output[0, count:].tolist()
    return {"messages": messages, "input_ids": inputs.input_ids[0].tolist(), "output_ids": ids,
            "text": client.tokenizer.decode(ids, skip_special_tokens=True)}


def encode_with_states(client, prompt, message, *, layer, codec="int4"):
    import numpy as np
    from budgetfair.model.span_transport import encode_message
    captured = []
    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[0, len(prompt.prefix_ids):].detach().float().cpu().numpy().copy())
    handle = client._decoder_block(layer).register_forward_hook(capture)
    try:
        packet, trace = encode_message(client, prompt, message, layer=layer, codec=codec)
    finally:
        handle.remove()
    if len(captured) != 1 or not np.isfinite(captured[0]).all():
        raise ValueError("Expected one finite source-state capture")
    array_hash = hashlib.sha256(captured[0].astype("<f4").tobytes()).hexdigest()
    if array_hash != trace["source_array_sha256"]:
        raise ValueError("Independent capture differs from encoded source")
    return packet, trace, captured[0]
