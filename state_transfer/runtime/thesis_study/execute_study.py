"""Host controller for plan-pinned, offline, separately mounted model stages."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import uuid
import json
import re

from common import Stage, read_json, write_json, sha, verify_seal, utc, file_inventory
from prepare_study import prepare_receiver, validate_heldout_data
from protocol_contracts import validate_protocol, validate_role_input, boundary_inventory


def mount_source_identity(value):
    """Normalize documented Docker Desktop Windows spellings, preserving POSIX case."""
    text = str(value).replace("\\", "/").rstrip("/")
    for prefix in ("/host_mnt/", "/run/desktop/mnt/host/"):
        if text.lower().startswith(prefix) and re.match(r"[A-Za-z]/", text[len(prefix):]):
            text = text[len(prefix)] + ":/" + text[len(prefix) + 2:]
            break
    if re.match(r"^[A-Za-z]:/", text):
        return text.casefold()
    return str(Path(text).resolve())


def verify_mounts(actual, expected):
    rows = actual.get("Mounts", [])
    mounts = {row.get("Destination"): row for row in rows}
    if len(mounts) != len(rows) or set(mounts) != set(expected):
        raise ValueError("Unexpected or duplicated model-container mount")
    for destination, source in expected.items():
        row = mounts[destination]
        if (row.get("Type") != "bind" or row.get("RW") is not (destination == "/out")
                or mount_source_identity(row.get("Source", "")) != mount_source_identity(source)):
            raise ValueError("Actual container mount source/read-only boundary differs: " + destination)


def checked(args, *, log=None):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    if log is not None:
        Path(log).write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Command failed with exit {result.returncode}: {args[0]} {args[1]}; output saved at {log}")
    return result.stdout


def stage_container(plan_path, weights, input_root, output_root, model, mode, channel=None):
    plan = validate_protocol(plan_path)
    if mode not in ("sender", "receiver") or (mode == "sender" and channel is not None):
        raise ValueError("Invalid isolated stage mode/channel")
    document = read_json(input_root / "input.json")
    items = validate_role_input(document, plan, sha(plan_path), "A" if mode == "sender" else "B",
        model=model, channel=channel, input_root=input_root)
    output_root.mkdir(parents=True, exist_ok=False)
    name = "thesis-" + model + "-" + mode + "-" + uuid.uuid4().hex[:12]
    runtime_output = output_root / "runtime"
    runtime_output.mkdir()
    args = ["docker", "create", "--name", name, "--pull=never", "--network", "none", "--gpus", "device=0",
        "--read-only", "--tmpfs", "/tmp:rw,size=2g", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
        "--mount", f"type=bind,source={input_root.resolve()},target=/input,readonly",
        "--mount", f"type=bind,source={plan_path.parent.resolve()},target=/runplan,readonly",
        "--mount", f"type=bind,source={weights.resolve()},target=/modelhub,readonly",
        "--mount", f"type=bind,source={runtime_output.resolve()},target=/out",
        "-e", "PYTHONPATH=/code/src", "-e", "HF_HUB_CACHE=/modelhub", "-e", "HF_HOME=/tmp/hf",
        "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "OMP_NUM_THREADS=4", "-e", "MKL_NUM_THREADS=4", "-e", "OPENBLAS_NUM_THREADS=4",
        "--entrypoint", "python", plan["image_id"], "/code/thesis_study/study_worker.py", mode,
        "--plan", "/runplan/protocol.json", "--input", "/input/input.json", "--assets", "/runplan/assets",
        "--output", "/out/stage", "--model", model]
    if channel:
        args += ["--channel", channel]
    receipt = {"started_utc": utc(), "status": "creating", "argv": args, "model": model, "mode": mode,
        "channel": channel, "plan_sha256": sha(plan_path), "input_inventory": boundary_inventory(input_root),
        "public_plan_inventory": boundary_inventory(plan_path.parent),
        "exit_code": None, "container_id": None, "error": None}
    write_json(output_root / "invocation.json", receipt)
    container_created = False
    try:
        container_id = checked(args, log=output_root / "create.log").strip()
        container_created = True
        inspection = checked(["docker", "inspect", container_id], log=output_root / "inspect.json")
        actual = json.loads(inspection)[0]
        if actual["Image"] != plan["image_id"] or actual["HostConfig"]["NetworkMode"] != "none" or not actual["HostConfig"]["ReadonlyRootfs"]:
            raise ValueError("Actual container identity/isolation differs from plan")
        verify_mounts(actual, {"/input": input_root.resolve(), "/runplan": plan_path.parent.resolve(),
            "/modelhub": weights.resolve(), "/out": runtime_output.resolve()})
        receipt.update(container_id=container_id, status="running", image_id=actual["Image"])
        write_json(output_root / "invocation.json", receipt)
        with (output_root / "stdout.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(["docker", "start", "--attach", container_id], stdout=log, stderr=subprocess.STDOUT, text=True)
        final_text = checked(["docker", "inspect", container_id], log=output_root / "inspect_final.json")
        final = json.loads(final_text)[0]
        exit_code = final["State"]["ExitCode"]
        receipt.update(exit_code=exit_code, docker_start_exit_code=process.returncode,
                       oom_killed=final["State"]["OOMKilled"], finished_utc=utc())
        if process.returncode or exit_code or final["State"]["OOMKilled"]:
            raise RuntimeError(f"Model stage failed (exit={exit_code}, attach={process.returncode}, OOM={final['State']['OOMKilled']})")
        state = verify_seal(runtime_output / "stage")
        identity = read_json(runtime_output / "stage" / "identity.json")
        if (state.get("phase") != "heldout_" + mode or state.get("model") != model
                or state["n_requested"] != len(items) or (mode == "receiver" and state.get("channel") != channel)
                or identity.get("model") != model or identity.get("plan_sha256") != receipt["plan_sha256"]
                or identity.get("input_sha256") != sha(input_root / "input.json")
                or identity.get("settings") != plan["models"][model]
                or (mode == "receiver" and identity.get("channel") != channel)):
            raise ValueError("Completed child stage identity/count differs from actual invocation")
        if boundary_inventory(input_root) != receipt["input_inventory"]:
            raise ValueError("Read-only stage input changed")
        if boundary_inventory(plan_path.parent) != receipt["public_plan_inventory"]:
            raise ValueError("Public protocol/assets changed during isolated stage")
        receipt.update(status="completed", output_manifest_sha256=sha(runtime_output / "stage" / "artifact_manifest.json"))
        write_json(output_root / "invocation.json", receipt)
        return runtime_output / "stage"
    except BaseException as error:
        receipt.update(status="failed", finished_utc=utc(), error={"type": type(error).__name__, "message": str(error)})
        write_json(output_root / "invocation.json", receipt)
        raise
    finally:
        if container_created:
            # A cancelled docker attach does not stop the container. Stop the
            # exact container created here before removal, even on host error.
            stop = subprocess.run(["docker", "stop", "--time", "10", name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            remove = subprocess.run(["docker", "rm", name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (output_root / "cleanup.log").write_text("STOP:\n" + stop.stdout + "\nREMOVE:\n" + remove.stdout, encoding="utf-8")
            receipt["cleanup"] = {"stop_exit_code": stop.returncode, "remove_exit_code": remove.returncode}
            if remove.returncode:
                receipt.update(status="failed", error={"type": "ContainerCleanupError", "message": "Own container could not be removed; inspect cleanup.log"})
            write_json(output_root / "invocation.json", receipt)
            if remove.returncode:
                raise RuntimeError("Container cleanup failed; stage cannot be certified")


def run(plan_path, data, weights, output):
    plan = validate_protocol(plan_path)
    validate_heldout_data(plan_path, data, plan)
    stage = Stage(output, "complete_isolated_heldout_study", requested=8, stages={})
    try:
        write_json(output / "input_receipt.json", {"plan_sha256": sha(plan_path), "data_manifest_sha256": sha(data / "artifact_manifest.json")})
        count = 0
        for model in ("qwen", "llama"):
            stage.update(stage="sender", model=model)
            sender_output = stage_container(plan_path, weights, data / "A", output / (model + "_sender"), model, "sender")
            stage.state["stages"][model + "_sender"] = {"status": "completed", "manifest_sha256": sha(sender_output / "artifact_manifest.json")}
            count += 1
            stage.update(n_completed=count)
            for channel in ("text", "raw", "pca"):
                stage.update(stage="receiver_partition", model=model, channel=channel)
                inputs = output / (model + "_" + channel + "_input")
                prepare_receiver(plan_path, data, sender_output, inputs, model, channel)
                stage.update(stage="receiver", model=model, channel=channel)
                receiver_output = stage_container(plan_path, weights, inputs, output / (model + "_" + channel + "_receiver"), model, "receiver", channel)
                stage.state["stages"][model + "_" + channel] = {"status": "completed", "manifest_sha256": sha(receiver_output / "artifact_manifest.json")}
                count += 1
                stage.update(n_completed=count)
        stage.finish()
    except BaseException as error:
        stage.fail(error)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("plan", "data", "weights", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    a = p.parse_args()
    run(a.plan, a.data, a.weights, a.output)
