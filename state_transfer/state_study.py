"""Verify immutable inputs, run an isolated two-agent smoke, or reproduce the full state study."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
PLAN = ROOT / "frozen" / "plan" / "protocol.json"
DATA = ROOT / "frozen" / "heldout"
PLAN_HASH = "c130f965ebe9a14f95071200657df3789560a4c37120d6ea73c19ae806fd12bb"
DATA_HASH = "609feb781e7be46d26db8bd742e6f1a777a0338138c359cce522c03b0e4bd42f"
sys.dont_write_bytecode = True
sys.path[:0] = [str(RUNTIME / "thesis_study"), str(RUNTIME / "src")]

from common import read_json, write_json, sha, seal
from prepare_study import validate_heldout_data
from protocol_contracts import validate_protocol


def verify():
    if sha(PLAN) != PLAN_HASH or sha(DATA / "artifact_manifest.json") != DATA_HASH:
        raise ValueError("The supplied original protocol or data manifest differs")
    plan = validate_protocol(PLAN, code_root=RUNTIME)
    families, _ = validate_heldout_data(PLAN, DATA, plan)
    return plan, families


def cache_check(cache, labels, *, sizes_only=False):
    manifest = read_json(ROOT / "checkpoint_hashes.json")
    for label in labels:
        model = manifest[label]
        path = cache / ("models--" + model["name"].replace("/", "--")) / "snapshots" / model["revision"]
        for name, expected_hash in model["files"].items():
            file = path / name
            if not file.is_file() or file.stat().st_size <= 0:
                raise ValueError(f"Missing pinned checkpoint file {label}/{name}")
            if sizes_only and name.endswith(".safetensors"):
                continue
            with file.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
                    raise ValueError(f"Pinned checkpoint file differs: {label}/{name}")


def prepare_image_run(output, image, cache):
    original, families = verify()
    if not cache.is_dir():
        raise ValueError("Pass the hub cache directory containing models--* folders")
    protected = [ROOT, cache]
    if any(output == path or output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError("Choose a fresh output outside the release and model cache")
    image_id = subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
    if re.fullmatch("sha256:[0-9a-f]{64}", image_id) is None:
        raise ValueError("Could not resolve the local Docker image identity")
    output.mkdir(parents=True, exist_ok=False)
    plan_dir, data = output / "plan", output / "data"
    shutil.copytree(PLAN.parent, plan_dir)
    shutil.copytree(DATA, data)
    plan_path = plan_dir / "protocol.json"
    if image_id != original["image_id"]:
        # The historical inputs stay untouched. Only the image field and the
        # derived hashes that bind inputs to this execution need new values.
        write_json(plan_path, {**original, "image_id": image_id})
        execution_hash = sha(plan_path)
        receipt_file = plan_dir / "freeze_receipt.json"
        write_json(receipt_file, {**read_json(receipt_file), "protocol_sha256": execution_hash})
        for name in ("A/input.json", "private_lineage.json"):
            file = data / name
            write_json(file, {**read_json(file), "plan_sha256": execution_hash})
        seal(data)
    copied = validate_protocol(plan_path, code_root=RUNTIME)
    validate_heldout_data(plan_path, data, copied)
    write_json(output / "execution_binding.json", {
        "published_protocol_sha256": PLAN_HASH, "execution_protocol_sha256": sha(plan_path),
        "published_image": original["image_id"], "execution_image": image_id,
        "original_image": image_id == original["image_id"],
        "source_hashes_and_scientific_settings_unchanged": True,
        "private_families_sha256": sha(data / "private_families.json"),
    })
    return plan_path, data, image_id, families


def container_cache_check(cache, image, labels, *, sizes_only=False):
    # Historical snapshots may contain Linux symlinks on an NTFS host.
    # The Linux workers must see and validate their actual resolved bytes.
    program = ("import sys; sys.path.insert(0,'/release'); from state_study import cache_check; "
               "from pathlib import Path; cache_check(Path('/modelhub'), sys.argv[1:], sizes_only="
               + repr(sizes_only) + ")")
    cmd = ["docker", "run", "--rm", "--pull=never", "--network", "none", "--read-only",
           "--mount", f"type=bind,source={ROOT},target=/release,readonly",
           "--mount", f"type=bind,source={cache},target=/modelhub,readonly",
           "--entrypoint", "python", image, "-B", "-c", program, *labels]
    subprocess.run(cmd, check=True)


def run_smoke(plan_path, image_id, cache, output, families):
    from private_tasks import QA_TASK, agent_view, score_answer
    family = next(row for row in families if row["task"] == QA_TASK)
    episode = family["episodes"]["original"]
    for role, agent in [("sender", "A"), ("receiver", "B")]:
        incoming, outgoing = output / (role + "_input"), output / (role + "_output")
        incoming.mkdir()
        outgoing.mkdir()
        write_json(incoming / "view.json", agent_view(episode, agent))
        if role == "receiver":
            shutil.copyfile(output / "sender_output" / "packet.bin", incoming / "packet.bin")
        cmd = ["docker", "run", "--rm", "--pull=never", "--gpus", "device=0", "--network", "none",
               "--read-only", "--tmpfs", "/tmp:rw,size=2g", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges:true"]
        for file, dest, writable in [(plan_path.parent, "/runplan", False), (cache, "/modelhub", False),
                    (ROOT / "smoke_worker.py", "/smoke_worker.py", False), (incoming, "/input", False),
                    (outgoing, "/out", True)]:
            cmd += ["--mount", f"type=bind,source={file},target={dest}" + ("" if writable else ",readonly")]
        for var in ["PYTHONPATH=/code/src:/code/thesis_study", "HF_HUB_CACHE=/modelhub", "HF_HOME=/tmp/hf",
                    "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "PYTHONDONTWRITEBYTECODE=1",
                    "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4"]:
            cmd += ["-e", var]
        subprocess.run(cmd + ["--entrypoint", "python", image_id, "/smoke_worker.py", role], check=True)
    sender = read_json(output / "sender_output/receipt.json")
    receiver = read_json(output / "receiver_output/receipt.json")
    if not (sender["packet_sha256"] == receiver["packet_sha256"] == sha(output / "receiver_input/packet.bin")
            and sender["packet_bytes"] == receiver["packet_bytes"] and receiver["injections"] == 1):
        raise ValueError("Packet identity, length or receiver injection mismatch")
    result = {"status": "completed", "family": family["family_id"], "sender": sender, "receiver": receiver,
              "task_score": score_answer(episode, receiver["answer"])}
    write_json(output / "smoke_result.json", result)
    print(json.dumps(result, indent=2))


def analyze(plan_path, data, study, output):
    plan_path, data, study, output = (p.resolve() for p in (plan_path, data, study, output))
    plan = validate_protocol(plan_path, code_root=RUNTIME)
    if output.exists() or output.with_name(output.name + "_statistics").exists():
        raise ValueError("Analysis and statistics need fresh output directories")
    if any(output.is_relative_to(p) for p in (ROOT, data, study, plan_path.parent)):
        raise ValueError("Keep analysis output separate from its inputs")
    output.parent.mkdir(exist_ok=True, parents=True)
    base = ["docker", "run", "--rm", "--pull=never", "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,size=2g", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true"]
    for path, dest, writable in [(plan_path.parent, "/runplan", False), (data, "/private_data", False),
                (study, "/completed_study", False), (output.parent, "/out", True)]:
        base += ["--mount", f"type=bind,source={path},target={dest}" + ("" if writable else ",readonly")]
    for var in ["PYTHONPATH=/code/src", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=1",
                "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1"]:
        base += ["-e", var]
    base += ["--entrypoint", "python", plan["image_id"]]
    arguments = ["--plan", "/runplan/protocol.json", "--data", "/private_data"]
    analysis_path = "/out/" + output.name
    subprocess.run(base + ["/code/thesis_study/analyze_study.py", *arguments,
                   "--study", "/completed_study", "--output", analysis_path], check=True)
    subprocess.run(base + ["/code/thesis_study/finalize_statistics.py", *arguments,
                   "--analysis", analysis_path, "--output", analysis_path + "_statistics"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    cache_cmd = commands.add_parser("verify-cache")
    cache_cmd.add_argument("--cache", type=Path, required=True)
    cache_cmd.add_argument("--image", help="Check inside this local Linux image")
    for name in ("smoke", "run"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--image", required=True)
        cmd.add_argument("--cache", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    cmd = commands.add_parser("analyze")
    for name in ("plan", "data", "study", "output"):
        cmd.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        plan, families = verify()
        print(f"Verified {len(plan['code_sha256'])} source files, four PCA bases and {len(families)} paired families.")
    elif args.command == "verify-cache":
        verify()
        if args.image:
            container_cache_check(args.cache.resolve(), args.image, ("qwen", "llama"))
        else:
            cache_check(args.cache.resolve(), ("qwen", "llama"))
        print("Both frozen checkpoint and tokenizer inventories match.")
    elif args.command == "analyze":
        analyze(args.plan, args.data, args.study, args.output)
    else:
        cache, output = args.cache.resolve(), args.output.resolve()
        plan, data, image, families = prepare_image_run(output, args.image, cache)
        if args.command == "smoke":
            container_cache_check(cache, image, ["qwen"], sizes_only=True)
            run_smoke(plan, image, cache, output, families)
        else:
            container_cache_check(cache, image, ["qwen", "llama"])
            import execute_study
            execute_study.run(plan, data, cache, output / "study")
            analyze(plan, data, output / "study", output / "analysis")


if __name__ == "__main__":
    main()

