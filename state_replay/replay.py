"""Rescore saved receiver outputs and reconstruct state-transfer policies on CPU."""
from collections import defaultdict
import argparse
import csv
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent / "state_transfer"
PLAN_SHA = "c130f965ebe9a14f95071200657df3789560a4c37120d6ea73c19ae806fd12bb"
FAMILIES_SHA = "294727edf65868175e48ac607fbc40d4a86c7f49ab9de76df3d812b11dc7b9c8"
LINEAGE_SHA = "6d4efb36276af0ed61597a66d23981fdd1df5e08f015576935a9219518110006"
RECORDS = {
    "graded_actual.jsonl": "237a893b9b78c74c5d07f1c33a5a4cc6492451904b193e015bb5dbd36ec46def",
    "policy_replays.jsonl": "81f2bd963b776fc2e9bbd8003a52edbb576bd0c90325f493787e296e204f6bf7",
    "statistics.json": "c2f9fbc00a24e170e0311428c69b2d7f1702425ac78bae454994e95fc580cd08",
    "run_status.json": "2fda46b1cf26853c188b075ed33bcece9d8514e7fc8b81d3fd38d50c1fae94a5",
    "packet_index.csv": "d35e6f4e45391018dac2560f8136b75c0509b38501c41ab24f9fbd3f99745ffd",
}


def checked(data, expected, name):
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"Recorded digest mismatch: {name}")
    return data


def match(observed, expected, path="statistics"):
    """Compare every key, record and value, tolerating library roundoff in floats."""
    if type(observed) is not type(expected):
        raise ValueError(f"Type mismatch in {path}")
    if isinstance(expected, dict):
        if observed.keys() != expected.keys():
            raise ValueError(f"Fields differ in {path}")
        for key, value in expected.items():
            match(observed[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if len(observed) != len(expected):
            raise ValueError(f"Length differs in {path}")
        for i, (left, right) in enumerate(zip(observed, expected)):
            match(left, right, f"{path}[{i}]")
    elif isinstance(expected, float):
        if not math.isclose(observed, expected, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError(f"Value differs in {path}")
    elif observed != expected:
        raise ValueError(f"Value differs in {path}")


def verify_sweep(actual, families, plan):
    from analyze_study import policy, summarize

    expected_families = defaultdict(list)
    for family in families:
        expected_families[family["task"]].append(family["family_id"])
    actual_by_key = {(r["model"], r["family_id"], r["condition"]): r for r in actual}
    rows = []
    for source in actual:
        if source["condition"] != source["channel"] + "_generated":
            continue
        model, fid = source["model"], source["family_id"]
        fallback = actual_by_key[model, fid, "no_comm"]
        for cap in plan["sweep_budgets_bits"]:
            # At zero loss every allowed draw selects the same delivery.
            row = policy(source, fallback, cap=cap, rate=0.0, draw=0.0)
            rows.append({**row, "budget_bits": cap, "channel": source["channel"],
                         "model": model, "task": source["task"], "family_id": fid})
    group_keys = ["model", "task", "channel", "budget_bits"]
    summaries = summarize(rows, group_keys, expected_families)
    computed = {tuple(str(row[k]) for k in group_keys): row for row in summaries}
    with ZipFile(HERE.parent / "evidence" / "Supplement_S2.zip") as z:
        table = list(csv.DictReader(io.StringIO(z.read("budget_sweep.csv").decode("utf-8-sig"))))
    if len(table) != 180 or len(computed) != 180:
        raise ValueError("Expected 180 budget sweep cells")
    for reference in table:
        row = computed.pop(tuple(reference[k] for k in group_keys))
        row["delivery_rate"] = row["delivered_count"] / row["n_families"]
        for key in reference.keys() - set(group_keys):
            if row[key] != float(reference[key]):
                raise ValueError(f"S2 mismatch for {key} in {reference}")
    return len(rows), len(table)


def verify_packet_costs(actual, index_bytes, packet_root):
    indexed = list(csv.DictReader(io.StringIO(index_bytes.decode("ascii"))))
    packets = {r["receiver_record_id"]: r for r in indexed}
    if len(packets) != len(indexed):
        raise ValueError("Duplicate measured packet identity")
    for r in actual:
        if r["condition"] == "no_comm":
            if r["wire_bits"] != 0:
                raise ValueError("Nonzero cost for a no-message observation")
            continue
        entry = packets.pop(r["receiver_record_id"])
        size = int(entry["bytes"])
        if size <= 0 or size * 8 != r["wire_bits"]:
            raise ValueError(f"Recorded wire cost differs from measured size for {r['receiver_record_id']}")
        if packet_root is not None:
            partition = packet_root / f"{r['model']}_{r['channel']}_input" / "packets"
            packet = partition / f"{r['family_id']}_{r['condition']}.bin"
            raw = checked(packet.read_bytes(), entry["sha256"], packet.name)
            if len(raw) != size:
                raise ValueError(f"Measured wire bytes differ for {packet.name}")
    if packets or len(indexed) != 2816:
        raise ValueError("Unexpected packet inventory")
    return len(indexed)


def load_json(path, digest):
    return json.loads(checked(path.read_bytes(), digest, path.name))


def replay(packet_root=None):
    plan = load_json(STUDY / "frozen/plan/protocol.json", PLAN_SHA)
    families = load_json(STUDY / "frozen/heldout/private_families.json", FAMILIES_SHA)["families"]
    lineage = load_json(STUDY / "frozen/heldout/private_lineage.json", LINEAGE_SHA)
    runtime = STUDY / "runtime/thesis_study"
    for name in ("common", "analyze_study", "finalize_statistics", "navigation_agent",
                 "paired_analysis", "private_tasks"):
        module = runtime / f"{name}.py"
        checked(module.read_bytes(), plan["code_sha256"]["thesis_study/" + module.name], module.name)
    sys.path.insert(0, str(runtime))
    from finalize_statistics import build_statistics

    with ZipFile(HERE / "observations.zip") as z:
        if sorted(z.namelist()) != sorted(RECORDS):
            raise ValueError("Incomplete or unexpected observation archive")
        files = {name: checked(z.read(name), digest, name) for name, digest in RECORDS.items()}
    status = json.loads(files["run_status.json"])
    if (status["status"], status["n_requested"], status["n_completed"], status["error"]) != (
        "completed", 3072, 3072, None
    ):
        raise ValueError("Held-out evaluation failed or is incomplete")
    actual = [json.loads(line) for line in files["graded_actual.jsonl"].splitlines()]
    policies = [json.loads(line) for line in files["policy_replays.jsonl"].splitlines()]
    result = build_statistics(plan, families, actual, policies, lineage=lineage, plan_hash=PLAN_SHA)
    match(result, json.loads(files["statistics.json"]))
    rows, cells = verify_sweep(actual, families, plan)
    packets = verify_packet_costs(actual, files["packet_index.csv"], packet_root)
    return {
        "status": "verified",
        "rescored_receiver_answers": len(actual),
        "reconstructed_primary_policy_rows": len(policies),
        "primary_contrasts": len(result["primary_contrasts"]),
        "significant_contrasts_favoring_text": sum(
            r["global_holm"]["reject"] and r["difference"] < 0 for r in result["primary_contrasts"]
        ),
        "sweep_reconstructed_policy_rows": rows,
        "sweep_cells_matching_supplement_s2": cells,
        "packet_size_records_checked": packets,
        "serialized_packets_rehashed": packets if packet_root is not None else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, help="Directory holding the archived receiver input partitions")
    print(json.dumps(replay(parser.parse_args().packet_root), indent=2))
