"""Independent isolated-harness contracts using synthetic development data only.

SystemRandom, model loading and inference are explicitly replaced. These tests
never create a real held-out dataset or perform pretrained/GPU inference.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from transformers import BatchEncoding, GenerationConfig as HFGenerationConfig

from budgetfair.comm.tensor_packet import decode_packet, encode_tensor
from budgetfair.comm.pca_packet import FRAME, reconstruct
from budgetfair.model.span_transport import PublicPrompt


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_numeric_relay_study.py"


class SyntheticRandom:
    def __init__(self):
        self.offset = 0

    def choice(self, alphabet):
        value = alphabet[(self.offset // 16) % len(alphabet)]
        self.offset += 1
        return value


class UnitInference:
    def __init__(self):
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=16, vocab_size=64),
                                     generation_config=HFGenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=0))
        self.quantization = "TEST_DOUBLE_NO_PRETRAINED_WEIGHTS"

    def _generation_kwargs(self):
        return {"max_new_tokens": 48, "do_sample": False, "temperature": 1.0, "top_p": 1.0, "logits_processor": []}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("isolated_relay_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    # Fingerprint a disposable source tree so other agents' legitimate edits
    # cannot alter this synthetic plan. Production ROOT is not edited.
    root = tmp_path / "source"
    for directory in ("src", "scripts", "tests"):
        (root / directory).mkdir(parents=True)
    (root / "src/frozen.py").write_text("VERSION = 1\n")
    (root / "tests/test_frozen.py").write_text("def test_original(): pass\n")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module.secrets, "SystemRandom", SyntheticRandom)
    forbidden = tmp_path / "forbidden.json"
    forbidden.write_text(json.dumps(["A" * 16]))
    config = tmp_path / "unit_model.yaml"
    config.write_text(yaml.safe_dump({"model": {"name": "UNIT-INFERENCE-NOT-PRETRAINED", "revision": "0" * 40,
        "quantization": "none", "device": "cpu", "torch_dtype": "float32", "max_context_len": 512}}))
    configs = {model: config for model in module.MODELS}
    study = tmp_path / "private_study"
    plan = module.prepare(study, forbidden, configs, purpose="development", count=2,
                          layers={"qwen": 4, "llama": 0}, models=("qwen", "llama"))
    public_prompt = PublicPrompt((1, 3), (8, 9), 10, max_payload_tokens=128, no_message_slots=16)
    state = SimpleNamespace(module=module, study=study, public=study / "public", root=root, plan=plan,
        configs=configs, forbidden=forbidden, receipts=tmp_path / "receipts", tmp=tmp_path,
        loads=[], received=[], empty_sender=False, monkeypatch=monkeypatch,
        real_generate_sender=module.generate_sender)

    def load_client(public, model, plan):
        state.loads.append((str(public), model))
        return UnitInference()

    def generate_sender(client, value):
        text = "" if state.empty_sender else value
        return {"messages": [{"role": "user", "content": "private:" + value}],
                "input_ids": [1, 3] + [ord(c) for c in value], "output_ids": [ord(c) for c in text] or [2], "text": text}

    def encode_message(client, public, message, *, layer, codec):
        value = np.repeat(np.array([ord(c) - 64 for c in message], dtype=np.float32)[:, None], 16, axis=1)
        packet = encode_tensor(value, codec=codec, role="hidden_states", layer_index=layer)
        return packet, {"source_array_sha256": module.bytes_sha(value.astype("<f4").tobytes()),
                        "sender_input_ids": [1, 3] + [ord(c) for c in message], "message_token_ids": [ord(c) for c in message],
                        "wire_bits": len(packet) * 8, "packet_sha256": module.bytes_sha(packet)}

    def result(public, packet, answer, injected):
        rows = decode_packet(packet).array.shape[0] if injected else public.no_message_slots
        return {"answer": answer, "output_ids": [ord(c) for c in answer] or [2],
                "receiver_input_ids": list(public.prefix_ids) + [public.placeholder_id] * rows + list(public.suffix_ids),
                "wire_bits": len(packet) * 8 if packet is not None else 0,
                "packet_sha256": module.bytes_sha(packet) if packet is not None else None,
                "injections": [{"positions": [2, 2 + rows]}] if injected else [], "status": "completed"}

    def receive_packet(client, public, packet, *, expected_layer):
        state.received.append(("numeric", packet))
        if packet is None:
            return result(public, None, "UNKNOWN", False)
        decoded = decode_packet(packet)
        assert decoded.layer_index == expected_layer
        answer = "UNKNOWN" if np.all(decoded.array == 0) else "".join(chr(round(float(x)) + 64) for x in decoded.array[:, 0])
        return result(public, packet, answer, True)

    def receive_text(client, public, packet):
        state.received.append(("text", packet))
        return result(public, packet, packet.decode("utf-8"), False)

    monkeypatch.setattr(module, "load_client", load_client)
    monkeypatch.setattr(module, "make_public_prompt", lambda client: public_prompt)
    monkeypatch.setattr(module, "generate_sender", generate_sender)
    monkeypatch.setattr(module, "encode_message", encode_message)
    monkeypatch.setattr(module, "receive_packet", receive_packet)
    monkeypatch.setattr(module, "receive_text", receive_text)
    return state


def send_all(h):
    for model in h.plan["model_order"]:
        h.module.send(h.study, model)


def receive_all(h):
    for model in h.plan["model_order"]:
        for group in ("numeric", "text"):
            h.module.receive(h.public, h.receipts, model, group)


def rewrite_public_jobs(h, model, edit):
    directory = h.public / "receiver_inputs" / model
    document = h.module.read_json(directory / "jobs.json")
    edit(document)
    h.module.atomic_json(directory / "jobs.json", document)
    ready = h.module.read_json(directory / "ready.json")
    ready["jobs_sha256"] = h.module.sha(directory / "jobs.json")
    h.module.atomic_json(directory / "ready.json", ready)


@pytest.mark.parametrize("prediction,target,exact,partial", [
    (" abcd ", "ABCD", True, 1.0), ("ABXD", "ABCD", False, 0.75),
    ("ABC", "ABCD", False, 0.75), ("ABCDE", "ABCD", False, 0.8),
    ("ABCD!", "ABCD", False, 0.8), ("", "ABCD", False, 0.0),
])
def test_exact_and_edit_metrics_have_independent_hand_calculated_values(harness, prediction, target, exact, partial):
    result = harness.module.string_scores(prediction, target)
    assert result["exact_match"] is exact and result["edit_similarity"] == pytest.approx(partial)


def test_prepare_freezes_disjoint_values_models_layers_source_tests_and_purpose(harness):
    h = harness
    cases = h.module.read_json(h.study / "private_cases.json")["cases"]
    values = [case[key] for case in cases for key in ("target", "alternate")]
    assert len(values) == len(set(values)) == 4 and "A" * 16 not in values
    assert h.plan["purpose"] == "development" and h.plan["models"]["qwen"]["layer"] == 4 and h.plan["models"]["llama"]["layer"] == 0
    assert h.plan["forbidden_values_sha256"] == h.module.sha(h.forbidden)
    assert h.plan["source_identity"]["source_files"] and h.plan["test_files"]
    assert h.module.sha(h.study / "plan.json") == h.module.sha(h.public / "plan.json")


def test_prepare_refuses_overwrite_and_small_runs_cannot_masquerade_as_heldout(harness):
    h = harness
    before = h.module.sha(h.study / "private_cases.json")
    with pytest.raises(FileExistsError):
        h.module.prepare(h.study, h.forbidden, h.configs, purpose="development", count=2)
    with pytest.raises(ValueError, match="heldout"):
        h.module.prepare(h.tmp / "mislabelled", h.forbidden, h.configs, purpose="heldout", count=2)
    assert h.module.sha(h.study / "private_cases.json") == before


def test_selected_single_model_development_plan_is_supported(harness):
    h = harness
    plan = h.module.prepare(h.tmp / "single", h.forbidden, h.configs, purpose="development", count=1, models=("qwen",))
    assert plan["model_order"] == ["qwen"] and set(plan["models"]) == {"qwen"}
    assert h.module.verify_plan(h.tmp / "single/public")["engineering_count"] == 1


def test_sender_public_manifest_has_no_gold_or_private_trace_and_controls_charge_real_packets(harness):
    h = harness
    document = h.module.send(h.study, "qwen")
    private_cases = h.module.read_json(h.study / "private_cases.json")["cases"]
    serialized = json.dumps(document)
    for case in private_cases:
        assert case["target"] not in serialized and case["alternate"] not in serialized
    assert all(set(job) == h.module.JOB_FIELDS for job in document["jobs"])
    for job in document["jobs"]:
        if job["condition"] == "vector_dropped":
            assert job["sent_bits"] == job["packet_bits"] > 544 and job["delivered_bits"] == 0 and job["filename"] is None
        elif job["condition"] == "vector_budget544":
            assert job["packet_bits"] > 544 and job["sent_bits"] == job["delivered_bits"] == 0 and job["filename"] is None
    assert (h.study / "sender/qwen/sender_trace.jsonl").is_file()
    assert not list((h.public / "receiver_inputs/qwen/numeric").glob("*.utf8"))


def test_raw_private_source_arrays_reproduce_actual_wire_and_capture_identity(harness):
    h = harness
    h.module.send(h.study, "qwen")
    sender = h.study / "sender/qwen"
    sources = h.module.read_rows(sender / "sources.jsonl")
    assert len(sources) == 6
    for source in sources:
        array = np.load(sender / source["raw_path"], allow_pickle=False)
        assert array.dtype == np.float32
        raw_packet = encode_tensor(array, role="hidden_states", layer_index=4)
        actual = encode_tensor(array, codec="int4", role="hidden_states", layer_index=4)
        assert h.module.bytes_sha(raw_packet) == source["raw_capture_packet_sha256"]
        assert h.module.bytes_sha(actual) == source["actual_packet_sha256"]
        assert len(actual) * 8 == source["actual_packet_bits"]
    assert not list((h.public / "receiver_inputs/qwen").rglob("*.npy"))


def test_numeric_receiver_operates_without_private_files_or_text_payload_directory(harness):
    h = harness
    h.module.send(h.study, "qwen")
    isolated = h.tmp / "numeric_mount"
    isolated.mkdir()
    shutil.copy2(h.public / "plan.json", isolated / "plan.json")
    shutil.copy2(h.public / "prepare_receipt.json", isolated / "prepare_receipt.json")
    shutil.copytree(h.public / "model_configs", isolated / "model_configs")
    target = isolated / "receiver_inputs/qwen"
    target.mkdir(parents=True)
    source = h.public / "receiver_inputs/qwen"
    for filename in ("public_prompt.json", "jobs.json", "ready.json"):
        shutil.copy2(source / filename, target / filename)
    shutil.copytree(source / "numeric", target / "numeric")
    assert not (target / "text").exists() and not (isolated / "private_cases.json").exists()
    status = h.module.receive(isolated, h.receipts, "qwen", "numeric")
    assert status["status"] == "completed" and status["n_completed"] == 14
    assert all(kind == "numeric" for kind, _ in h.received)
    assert sum(packet is None for _, packet in h.received) == 6  # no_comm + drop + blocked, each2.
    rows = h.module.read_rows(h.receipts / "qwen/numeric/rows.jsonl")
    assert all("exact_match" not in row and "success" not in row for row in rows)


@pytest.mark.parametrize("problem", ["missing_job", "duplicate_job", "extra_file", "corrupt_packet"])
def test_invalid_public_jobs_fail_before_model_loading_and_preserve_failed_status(harness, problem):
    h = harness
    h.module.send(h.study, "qwen")
    directory = h.public / "receiver_inputs/qwen"
    if problem == "missing_job":
        rewrite_public_jobs(h, "qwen", lambda doc: doc["jobs"].pop())
    elif problem == "duplicate_job":
        rewrite_public_jobs(h, "qwen", lambda doc: doc["jobs"].append(doc["jobs"][0]))
    elif problem == "extra_file":
        (directory / "numeric/unplanned.bin").write_bytes(b"extra")
    else:
        document = h.module.read_json(directory / "jobs.json")
        job = next(job for job in document["jobs"] if job["condition"] == "vector_generated")
        path = directory / "numeric" / job["filename"]
        path.write_bytes(b"BAD!" + path.read_bytes()[4:])
        rewrite_public_jobs(h, "qwen", lambda doc: next(row for row in doc["jobs"] if row["job_id"] == job["job_id"]).update(packet_sha256=h.module.sha(path)))
    before = len(h.loads)
    with pytest.raises(ValueError):
        h.module.receive(h.public, h.receipts, "qwen", "numeric")
    assert len(h.loads) == before and not h.received
    status = h.module.read_json(h.receipts / "qwen/numeric/run_status.json")
    assert status["status"] == "failed" and status["n_completed"] == 0 and status["error"]["traceback"]


def test_loading_failure_has_durable_running_record_before_load_and_no_fabricated_rows(harness):
    h = harness
    def fail(*args, **kwargs):
        status = h.module.read_json(h.study / "sender/qwen/run_status.json")
        assert status["status"] == "running" and status["stage"] == "model_loading"
        raise RuntimeError("loading unavailable")
    h.monkeypatch.setattr(h.module, "load_client", fail)
    with pytest.raises(RuntimeError, match="loading unavailable"):
        h.module.send(h.study, "qwen")
    status = h.module.read_json(h.study / "sender/qwen/run_status.json")
    assert status["status"] == "failed" and status["n_completed"] == 0
    assert not (h.public / "receiver_inputs/qwen/ready.json").exists()


def test_receiver_exception_preserves_partial_rows_without_zero_filling(harness):
    h = harness
    h.module.send(h.study, "qwen")
    original = h.module.receive_packet
    calls = []
    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("inference crashed")
        return original(*args, **kwargs)
    h.monkeypatch.setattr(h.module, "receive_packet", failing)
    with pytest.raises(RuntimeError, match="inference crashed"):
        h.module.receive(h.public, h.receipts, "qwen", "numeric")
    directory = h.receipts / "qwen/numeric"
    assert h.module.read_json(directory / "run_status.json")["n_completed"] == 1
    assert len(h.module.read_rows(directory / "rows.jsonl")) == 1
    assert not (directory / "artifact_manifest.json").exists()


def test_empty_sender_is_observed_empty_not_a_runtime_failure_or_synthetic_packet(harness):
    h = harness
    h.empty_sender = True
    jobs = h.module.send(h.study, "qwen")["jobs"]
    for job in jobs:
        if job["condition"] in {"text_generated", "vector_generated", "vector_swapped", "vector_zero"}:
            assert job["delivery_status"] == "observed_empty" and job["filename"] is None and job["packet_bits"] == 0
    assert all(trace["observed_empty"] for trace in h.module.read_rows(h.study / "sender/qwen/sender_trace.jsonl"))
    assert h.module.receive(h.public, h.receipts, "qwen", "text")["status"] == "completed"


def test_complete_two_model_group_analysis_uses_actual_outcomes_and_planned_denominators(harness):
    h = harness
    send_all(h)
    receive_all(h)
    report = h.module.analyze(h.study, h.receipts, h.tmp / "analysis")
    assert report["purpose"] == "development" and report["all_planned_results_complete"]
    assert report["scientific_gate"]["passed"] is True
    for model in h.module.MODELS:
        assert report["summaries"][model]["vector_generated"]["successes"] == 2
        assert report["summaries"][model]["vector_swapped"]["successes"] == 0
        assert report["summaries"][model]["vector_swapped"]["alternate_matches"] == 2
        assert report["summaries"][model]["no_comm"]["successes"] == 0
        assert report["contrasts"][model]["primary"]["n"] == 2
        assert report["summaries"][model]["vector_dropped"]["sent_bits"] > 0
        assert report["summaries"][model]["vector_budget544"]["sent_bits"] == 0


@pytest.mark.parametrize("missing", ["numeric", "text"])
def test_analysis_rejects_missing_receiver_group(harness, missing):
    h = harness
    send_all(h)
    for model in h.module.MODELS:
        h.module.receive(h.public, h.receipts, model, "text" if missing == "numeric" else "numeric")
    with pytest.raises((ValueError, FileNotFoundError)):
        h.module.analyze(h.study, h.receipts, h.tmp / "incomplete_analysis")
    assert h.module.read_json(h.tmp / "incomplete_analysis/run_status.json")["status"] == "failed"
    assert not (h.tmp / "incomplete_analysis/analysis.json").exists()


@pytest.mark.parametrize("change", ["tampered_answer", "duplicate_row", "boolean_cost"])
def test_analysis_rejects_tampered_or_inconsistent_receipts(harness, change):
    h = harness
    send_all(h)
    receive_all(h)
    directory = h.receipts / "qwen/numeric"
    path = directory / "rows.jsonl"
    rows = h.module.read_rows(path)
    if change == "tampered_answer":
        rows[0]["result"]["answer"] = "tampered"
    elif change == "duplicate_row":
        rows.append(rows[0])
    else:
        next(row for row in rows if row["condition"] == "no_comm")["sent_bits"] = False
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if change != "tampered_answer":
        h.module.seal(directory)  # synthetic re-signed fixture tests semantic checks, not SHA alone.
    with pytest.raises(ValueError):
        h.module.analyze(h.study, h.receipts, h.tmp / "invalid_analysis")


def test_analysis_recomputes_score_from_answer_and_does_not_trust_stored_success_flag(harness):
    h = harness
    send_all(h)
    receive_all(h)
    directory = h.receipts / "qwen/numeric"
    rows = h.module.read_rows(directory / "rows.jsonl")
    row = next(row for row in rows if row["condition"] == "vector_generated")
    row["result"]["answer"] = "WRONG"
    row["success"] = True
    (directory / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    h.module.seal(directory)  # Deliberate synthetic source; token replay is separately declared pending.
    report = h.module.analyze(h.study, h.receipts, h.tmp / "rescored")
    assert report["summaries"]["qwen"]["vector_generated"]["successes"] == 1
    assert report["status"] == "completed" and report["scientific_gate"]["passed"] is False


def test_inconsistent_completed_sender_counter_is_not_accepted(harness):
    h = harness
    send_all(h)
    receive_all(h)
    sender = h.study / "sender/qwen"
    status = h.module.read_json(sender / "run_status.json")
    status["n_completed"] = 1
    h.module.atomic_json(sender / "run_status.json", status)
    h.module.seal(sender)
    ready_path = h.public / "receiver_inputs/qwen/ready.json"
    ready = h.module.read_json(ready_path)
    ready["sender_manifest_sha256"] = h.module.sha(sender / "artifact_manifest.json")
    h.module.atomic_json(ready_path, ready)
    with pytest.raises(ValueError, match="Sender stage counters"):
        h.module.analyze(h.study, h.receipts, h.tmp / "bad_counter")


def test_source_change_after_freeze_is_rejected_before_sending(harness):
    h = harness
    (h.root / "src/frozen.py").write_text("VERSION = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        h.module.send(h.study, "qwen")
    assert not h.loads


def enable_pca(h, *, wrong_layer=False, missing_model=False, lineage_change=None, calibration_values=None, pca_codecs=None):
    assets, lineages = {}, {}
    for model, layer in (("qwen", 4), ("llama", 0)):
        path = h.tmp / f"{model}_synthetic_basis.npz"
        np.savez(path, mean=np.ones(16, dtype=np.float32), components=np.eye(16, dtype=np.float32)[:4],
                 layer=np.array(0 if wrong_layer else layer, dtype=np.int32), schema=np.array(1, dtype=np.int32))
        assets[model] = path
        lineage = h.tmp / f"{model}_synthetic_lineage.json"
        values = calibration_values or ["Z" * 16]
        document = {"purpose": "synthetic CPU calibration fixture", "model_name": "UNIT-INFERENCE-NOT-PRETRAINED",
                    "model_revision": "0" * 40, "layer": layer, "asset_sha256": h.module.sha(path),
                    "calibration_values": values, "calibration_values_sha256": h.module.calibration_values_digest(values)}
        if lineage_change is not None:
            lineage_change(document)
        lineage.write_text(json.dumps(document))
        lineages[model] = lineage
    if missing_model:
        assets.pop("llama")
    study = h.tmp / "pca_study"
    plan = h.module.prepare(study, h.forbidden, h.configs, purpose="development", count=2,
        layers={"qwen": 4, "llama": 0}, pca_assets=assets, pca_lineage=lineages, pca_dim=4, pca_codecs=pca_codecs)
    h.study, h.public, h.plan = study, study / "public", plan
    h.receipts = h.tmp / "pca_receipts"

    def receive_projected(client, prompt, packet, asset, *, expected_layer, expected_dimensions, expected_codec):
        hidden, metadata = reconstruct(packet, asset, expected_dimensions=expected_dimensions, expected_codec=expected_codec)
        local = encode_tensor(hidden, codec="float32", role="hidden_states", layer_index=expected_layer)
        result = h.module.receive_packet(client, prompt, local, expected_layer=expected_layer)
        result.update(metadata)
        result["local_reconstructed_buffer_bytes"] = len(local)
        return result
    h.monkeypatch.setattr(h.module, "receive_projected_packet", receive_projected)


def test_optional_pca_known_basis_preserves_raw_arms_and_correct_packet_costs(harness):
    h = harness
    enable_pca(h)
    assert h.plan["primary_conditions"][:6] == list(h.module.PRIMARY)
    assert h.plan["primary_conditions"][6:] == list(h.module.PCA_PRIMARY)
    send_all(h)
    root = h.public / "receiver_inputs/qwen"
    jobs = h.module.read_json(root / "jobs.json")["jobs"]
    raw = next(job for job in jobs if job["condition"] == "vector_generated")
    projected = next(job for job in jobs if job["condition"] == "vector_pca_generated")
    mean = next(job for job in jobs if job["condition"] == "vector_pca_mean")
    packet = (root / "numeric" / projected["filename"]).read_bytes()
    assert projected["packet_bits"] == len(packet) * 8 < raw["packet_bits"]
    assert packet[:4] == b"PCA1"
    coefficients = decode_packet((root / "numeric" / mean["filename"]).read_bytes()[FRAME.size:]).array
    assert coefficients.shape == (16, 4) and np.all(coefficients == 0)
    assert (h.study / "calibration_lineage/qwen.json").is_file()
    assert not (h.public / "calibration_lineage").exists()
    receive_all(h)
    report = h.module.analyze(h.study, h.receipts, h.tmp / "pca_analysis")
    assert report["scientific_gate"]["passed"] and report["pca_success_gate"]["passed"]
    assert all(value > 0 for value in report["pca_setup_bytes"].values())
    for model in h.module.MODELS:
        values = report["summaries"][model]
        assert values["vector_pca_generated"]["successes"] == 2
        assert values["vector_pca_swapped"]["alternate_matches"] == 2
        assert values["vector_pca_mean"]["successes"] == 0
        assert report["contrasts"][model]["pca_vs_raw"]["n"] == 2
    rows = h.module.read_rows(h.receipts / "qwen/numeric/rows.jsonl")
    actual = next(row for row in rows if row["condition"] == "vector_pca_generated")
    assert actual["result"]["wire_bits"] == projected["packet_bits"]
    assert actual["result"]["local_reconstructed_buffer_bytes"] * 8 > actual["result"]["wire_bits"]


@pytest.mark.parametrize("problem", ["wrong_layer", "missing_model"])
def test_pca_assets_must_match_layers_and_cover_all_selected_models(harness, problem):
    with pytest.raises(ValueError):
        enable_pca(harness, **{problem: True})
    assert not (harness.tmp / "pca_study/private_cases.json").exists()


def test_tampered_frozen_basis_is_rejected_before_receiver_model_loading(harness):
    h = harness
    enable_pca(h)
    send_all(h)
    path = h.public / "assets/qwen_pca.npz"
    path.write_bytes(path.read_bytes() + b"modified")
    before = len(h.loads)
    with pytest.raises(ValueError, match="PCA asset"):
        h.module.receive(h.public, h.receipts, "qwen", "numeric")
    assert len(h.loads) == before and not h.received


def test_compression_gate_failure_is_separate_from_raw_success_and_execution_status(harness):
    h = harness
    enable_pca(h)
    send_all(h)
    receive_all(h)
    directory = h.receipts / "qwen/numeric"
    rows = h.module.read_rows(directory / "rows.jsonl")
    next(row for row in rows if row["condition"] == "vector_pca_generated")["result"]["answer"] = "WRONG"
    (directory / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    h.module.seal(directory)
    report = h.module.analyze(h.study, h.receipts, h.tmp / "failed_pca_gate")
    assert report["status"] == "completed" and report["scientific_gate"]["passed"] is True
    assert report["pca_success_gate"]["passed"] is False


@pytest.mark.parametrize("phase", ["send", "receive", "analyze"])
def test_rewriting_both_plan_copies_does_not_redefine_completed_prepare(harness, phase):
    h = harness
    if phase != "send":
        send_all(h)
    if phase == "analyze":
        receive_all(h)
    plan = h.module.read_json(h.study / "plan.json")
    plan["created_utc"] = "rewritten after prepare"
    h.module.atomic_json(h.study / "plan.json", plan)
    h.module.atomic_json(h.public / "plan.json", plan)
    before = len(h.loads)
    with pytest.raises(ValueError, match="prepare"):
        if phase == "send":
            h.module.send(h.study, "qwen")
        elif phase == "receive":
            h.module.receive(h.public, h.receipts, "qwen", "numeric")
        else:
            h.module.analyze(h.study, h.receipts, h.tmp / "rewritten_analysis")
    assert len(h.loads) == before


@pytest.mark.parametrize("field,value", [("bootstrap_resamples", 1), ("bootstrap_seed", 0),
                                        ("scientific_gate", {"positive_required": 0}),
                                        ("pca_success_gate", {"positive_required": 0})])
def test_statistical_policy_cannot_be_relaxed_by_reanchoring_a_public_receipt(harness, field, value):
    h = harness
    plan = h.module.read_json(h.public / "plan.json")
    plan[field] = value
    h.module.atomic_json(h.public / "plan.json", plan)
    receipt = h.module.read_json(h.public / "prepare_receipt.json")
    receipt["plan_sha256"] = h.module.sha(h.public / "plan.json")
    h.module.atomic_json(h.public / "prepare_receipt.json", receipt)
    with pytest.raises(ValueError, match="policy|gate"):
        h.module.verify_plan(h.public)


@pytest.mark.parametrize("change", ["failed_status", "counter", "manifest_file", "changed_forbidden"])
def test_sender_requires_completed_exact_prepare_snapshot_before_loading(harness, change):
    h = harness
    if change in {"failed_status", "counter"}:
        status = h.module.read_json(h.study / "run_status.json")
        status["status" if change == "failed_status" else "n_completed"] = "failed" if change == "failed_status" else 1
        h.module.atomic_json(h.study / "run_status.json", status)
    elif change == "manifest_file":
        manifest = h.module.read_json(h.study / "prepare_manifest.json")
        manifest["files"].pop("private_cases.json")
        h.module.atomic_json(h.study / "prepare_manifest.json", manifest)
    else:
        (h.study / "forbidden_values.json").write_text("[]")
    with pytest.raises(ValueError, match="prepare"):
        h.module.send(h.study, "qwen")
    assert not h.loads


@pytest.mark.parametrize("field,value", [("model_name", "OTHER"), ("model_revision", "1" * 40),
    ("layer", 19), ("asset_sha256", "f" * 64), ("calibration_values_sha256", "f" * 64),
    ("calibration_values", ["Z" * 16, "Z" * 16])])
def test_pca_lineage_rejects_wrong_model_basis_or_calibration_before_data_creation(harness, field, value):
    h = harness
    with pytest.raises(ValueError, match="lineage"):
        enable_pca(h, lineage_change=lambda doc: doc.update({field: value}))
    assert not (h.tmp / "pca_study").exists()


def test_calibration_values_are_automatically_excluded_and_identity_is_portable(harness):
    h = harness
    # Synthetic RNG would draw B first after forbidden A; calibration must veto B.
    enable_pca(h, calibration_values=["B" * 16])
    cases = h.module.read_json(h.study / "private_cases.json")["cases"]
    assert all(case[key] not in {"A" * 16, "B" * 16} for case in cases for key in ("target", "alternate"))
    assert h.plan["calibration_values_count"] == 1 and h.plan["effective_exclusions_count"] == 2
    import hashlib
    assert h.module.calibration_values_digest(["B" * 16]) == hashlib.sha256(b'["BBBBBBBBBBBBBBBB"]').hexdigest()
    assert h.module.verify_prepared(h.study)["effective_exclusions_sha256"] == h.plan["effective_exclusions_sha256"]


def test_pca_coefficient_codecs_are_independent_per_model_and_do_not_change_raw_codec(harness):
    h = harness
    enable_pca(h, pca_codecs={"qwen": "int4", "llama": "int8"})
    send_all(h)
    costs = {}
    for model, codec in (("qwen", "int4"), ("llama", "int8")):
        root = h.public / "receiver_inputs" / model
        jobs = h.module.read_json(root / "jobs.json")["jobs"]
        raw = next(job for job in jobs if job["condition"] == "vector_generated")
        pca = next(job for job in jobs if job["condition"] == "vector_pca_generated")
        assert decode_packet((root / "numeric" / raw["filename"]).read_bytes()).codec == "int4"
        assert decode_packet((root / "numeric" / pca["filename"]).read_bytes()[FRAME.size:]).codec == codec
        costs[model] = pca["packet_bits"]
    assert costs["llama"] > costs["qwen"]
    receive_all(h)
    report = h.module.analyze(h.study, h.receipts, h.tmp / "mixed_pca_codecs")
    assert report["pca_success_gate"]["passed"] and report["scientific_gate"]["passed"]


@pytest.mark.parametrize("decoded_text,normalized_exact,raw_exact", [
    ("ABCDEFGHIJKLMNOP", True, True),
    ("A B C D E F G H I J K L M N O P", False, False),
    ("ABCD\tEFGHIJKLMNOP", False, False),
    ("ABCDEFGHIJKLMNOP!", False, False),
    ("\nABCDEFGHIJKLMNOP\n", True, False),
    (" abcdefghijklmnop ", True, False),
    ("", False, False),
])
def test_real_generate_sender_preserves_raw_decoding_and_actual_continuation_ids(harness, decoded_text, normalized_exact, raw_exact):
    h = harness
    private_value = "ABCDEFGHIJKLMNOP"
    prompt_ids = [11, 12, 13]
    continuation_ids = list(range(100, 100 + len(decoded_text))) + [2]
    calls = []
    client = UnitInference()
    client.model.config.vocab_size = 512

    def chat_inputs(messages):
        calls.append(("chat", messages))
        return BatchEncoding({"input_ids": torch.tensor([prompt_ids]), "attention_mask": torch.ones((1, 3), dtype=torch.long)})

    def generate(**kwargs):
        calls.append(("generate", kwargs))
        assert kwargs["input_ids"].tolist() == [prompt_ids]
        assert kwargs["max_new_tokens"] == 48 and kwargs["use_model_defaults"] is False
        return torch.tensor([prompt_ids + continuation_ids])

    def decode(ids, *, skip_special_tokens):
        calls.append(("decode", ids))
        assert ids == continuation_ids and skip_special_tokens is True
        return decoded_text

    client._chat_inputs = chat_inputs
    client.model.generate = generate
    client.tokenizer = SimpleNamespace(decode=decode)
    trace = h.real_generate_sender(client, private_value)
    assert [kind for kind, _ in calls] == ["chat", "generate", "decode"]  # one actual generation, no retry until correct
    assert trace["input_ids"] == prompt_ids and trace["output_ids"] == continuation_ids
    assert trace["text"] == decoded_text  # no whitespace deletion, casing change, or gold replacement
    assert trace["messages"] == calls[0][1]
    assert trace["messages"][0]["role"] == "system" and private_value not in trace["messages"][0]["content"]
    assert trace["messages"][1]["role"] == "user" and private_value in trace["messages"][1]["content"]
    scores = h.module.string_scores(trace["text"], private_value)
    assert scores["exact_match"] is normalized_exact
    assert scores["raw_exact_match"] is raw_exact  # permitted outer-strip/case normalization does not alter raw evidence


@pytest.mark.parametrize("separator", [" ", "\t", "."])
def test_sender_formatting_errors_reach_text_and_numeric_channels_unchanged_and_costed(harness, separator):
    h = harness
    original_sender = h.module.generate_sender
    original_encoder = h.module.encode_message
    captured_messages = []

    def malformed_sender(client, value):
        trace = original_sender(client, value)
        observed = separator.join(value)
        return {**trace, "text": observed, "output_ids": [ord(char) for char in observed] + [2]}

    def capture_actual_message(client, public, message, **kwargs):
        captured_messages.append(message)
        return original_encoder(client, public, message, **kwargs)

    h.monkeypatch.setattr(h.module, "generate_sender", malformed_sender)
    h.monkeypatch.setattr(h.module, "encode_message", capture_actual_message)
    send_all(h)
    cases = h.module.read_json(h.study / "private_cases.json")["cases"]
    expected_capture = [message for _model in h.plan["model_order"] for case in cases
                        for message in (separator.join(case["target"]), separator.join(case["alternate"]), case["target"])]
    assert captured_messages == expected_capture
    for model in h.plan["model_order"]:
        root = h.public / "receiver_inputs" / model
        jobs = h.module.read_json(root / "jobs.json")["jobs"]
        traces = h.module.read_rows(h.study / "sender" / model / "sender_trace.jsonl")
        assert all(len(row["text"]) == 31 and row["output_ids"] == [ord(c) for c in row["text"]] + [2] for row in traces)
        for case in cases:
            selected = {job["condition"]: job for job in jobs if job["case_id"] == case["case_id"]}
            text_job, raw_job, oracle_job = (selected[key] for key in ("text_generated", "vector_generated", "vector_oracle"))
            exact_bytes = separator.join(case["target"]).encode("utf-8")
            assert (root / "text" / text_job["filename"]).read_bytes() == exact_bytes
            assert text_job["packet_bits"] == len(exact_bytes) * 8 == 248
            raw_packet = (root / "numeric" / raw_job["filename"]).read_bytes()
            assert decode_packet(raw_packet).array.shape == (31, 16)
            assert raw_job["packet_bits"] == len(raw_packet) * 8 == 3200 > oracle_job["packet_bits"] == 1760
            assert text_job["delivery_status"] == raw_job["delivery_status"] == "delivered"
    receive_all(h)
    report = h.module.analyze(h.study, h.receipts, h.tmp / "literal_formatting_errors")
    for model in h.plan["model_order"]:
        assert report["summaries"][model]["text_generated"]["successes"] == 0
        assert report["summaries"][model]["vector_generated"]["successes"] == 0
        assert report["summaries"][model]["vector_oracle"]["successes"] == 2
    assert report["status"] == "completed" and report["scientific_gate"]["passed"] is False
