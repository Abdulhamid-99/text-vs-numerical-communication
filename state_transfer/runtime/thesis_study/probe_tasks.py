"""Actual pretrained development probe; never a heldout study or isolated receiver proof."""
import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import re

import numpy as np

from budgetfair.model.client import LLMClient, GenerationConfig
from budgetfair.model.span_transport import receive_text, receive_packet
from common import Stage, read_json, verify_seal, write_json, append_row, sha, timed, generate, encode_with_states
from private_tasks import agent_view, sender_requested_message, score_answer, validate_split_isolation, QA_TASK, NAV_TASK
from task_prompts import sender_messages, receiver_parts, sender_encoding_prompt, PROMPT_VERSION
from navigation_agent import attach_task_response, NAVIGATION_AGENT_VERSION


def run(data, config, output, model, count=4, layer=None, comparison_layer=None):
    stage = Stage(output, "task_development", requested=0, model=model, purpose="development")
    try:
        if type(count) is not int or count <= 0:
            raise ValueError("Development count must be a positive integer per task")
        for value in (layer, comparison_layer):
            if value is not None and (type(value) is not int or not 0 <= value <= 31):
                raise ValueError("Development layer must be a block index in 0..31")
        layer = (4 if model == "qwen" else 0) if layer is None else layer
        if comparison_layer == layer:
            raise ValueError("Comparison layer must differ from selected layer")
        model_names = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct"}
        model_config = read_json(config)
        if model not in model_names or model_config.get("model", {}).get("name") != model_names[model]:
            raise ValueError("Model label differs from the actual frozen model configuration")
        tokenizer = model_config.get("tokenizer", {})
        if tokenizer.get("name") != model_names[model]:
            raise ValueError("Tokenizer model identity differs from frozen study model")
        for item in (model_config["model"], tokenizer):
            if not isinstance(item.get("revision"), str) or re.fullmatch(r"[0-9a-fA-F]{40}", item["revision"]) is None:
                raise ValueError("Model and tokenizer revisions must be immutable 40-character commit IDs")
        verify_seal(data)
        document = read_json(Path(data) / "development.json")
        if document.get("split") != "development" or not isinstance(document.get("families"), list) or not document["families"]:
            raise ValueError("Expected nonempty development data; never probe heldout/calibration")
        if any(family.get("split") != "development" for family in document["families"]):
            raise ValueError("All probe families must belong to development")
        validate_split_isolation(document["families"])
        by_task = defaultdict(list)
        for family in document["families"]:
            if len(by_task[family["task"]]) < count:
                by_task[family["task"]].append(family)
        if set(by_task) != {QA_TASK, NAV_TASK} or any(len(items) != count for items in by_task.values()):
            raise ValueError("Require the requested number of families for both supported tasks")
        families = [family for items in by_task.values() for family in items]
        stage.update(n_requested=len(families) * (7 if comparison_layer is None else 10), stage="model_loading")
        client, load_seconds = timed(lambda: LLMClient.from_config(config))
        client.generation = GenerationConfig(max_new_tokens=192, temperature=0.0, do_sample=False)
        write_json(stage.directory / "configuration.json", {"model": model, "layer": layer,
            "comparison_layer": comparison_layer,
            "navigation_agent_version": NAVIGATION_AGENT_VERSION,
            "model_config": model_config, "config_sha256": sha(config), "prompt_version": PROMPT_VERSION,
            "model_load_seconds": load_seconds, "generation": vars(client.generation),
            "source_hashes": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
            "data_manifest_sha256": sha(Path(data) / "artifact_manifest.json"),
            "scope": "Development in one process; private evaluator data are accessible to the probe. Not an isolation claim."})
        raw = stage.directory / "raw"
        raw.mkdir()
        summary = defaultdict(lambda: defaultdict(lambda: {"n": 0, "successes": 0}))
        counter = 0
        for family in families:
            task = family["task"]
            originals = family["episodes"]
            source = {}
            comparisons = {}
            encoding_prompt = sender_encoding_prompt(client, task)
            for variant, episode in originals.items():
                stage.update(stage="sender_generation", case_id=episode["family_id"], variant=variant)
                trace, seconds = timed(lambda: generate(client, sender_messages(agent_view(episode, "A"))))
                message = trace["text"]
                append_row(stage.directory / "senders.jsonl", {"family_id": episode["family_id"], "task": task,
                    "variant": variant, "trace": trace, "seconds": seconds,
                    "source_exact": message == sender_requested_message(episode)})
                if not message:
                    raise ValueError("Observed empty development sender; preserve as failed transport preparation")
                (packet, encoding, states), encode_seconds = timed(lambda: encode_with_states(
                    client, encoding_prompt, message, layer=layer, codec="int4"))
                stem = f"{episode['family_id']}_{variant}"
                np.save(raw / (stem + ".npy"), states, allow_pickle=False)
                (raw / (stem + ".bin")).write_bytes(packet)
                source[variant] = (message, packet)
                append_row(stage.directory / "encodings.jsonl", {"family_id": episode["family_id"],
                    "variant": variant, "trace": encoding, "encode_seconds": encode_seconds,
                    "encoding_prompt": asdict(encoding_prompt), "array": stem + ".npy", "packet": stem + ".bin"})
                if comparison_layer is not None:
                    (other_packet, other_trace, other_states), other_seconds = timed(lambda: encode_with_states(
                        client, encoding_prompt, message, layer=comparison_layer, codec="int4"))
                    comparisons[variant] = other_packet
                    other_stem = stem + "_comparison"
                    np.save(raw / (other_stem + ".npy"), other_states, allow_pickle=False)
                    (raw / (other_stem + ".bin")).write_bytes(other_packet)
                    append_row(stage.directory / "comparison_encodings.jsonl", {"family_id": episode["family_id"],
                        "variant": variant, "trace": other_trace, "encode_seconds": other_seconds,
                        "encoding_prompt": asdict(encoding_prompt), "array": other_stem + ".npy", "packet": other_stem + ".bin"})
            episode = originals["original"]
            prompt, parts = receiver_parts(client, agent_view(episode, "B"))
            if prompt.prefix_ids != encoding_prompt.prefix_ids:
                raise ValueError("Receiver prefix differs from the public sender encoding prefix")
            oracle = sender_requested_message(episode)
            (oracle_packet, oracle_encoding, oracle_states), oracle_encode_seconds = timed(
                lambda: encode_with_states(client, encoding_prompt, oracle, layer=layer, codec="int4"))
            np.save(raw / (episode["family_id"] + "_oracle.npy"), oracle_states, allow_pickle=False)
            (raw / (episode["family_id"] + "_oracle.bin")).write_bytes(oracle_packet)
            append_row(stage.directory / "encodings.jsonl", {"family_id": episode["family_id"],
                "variant": "oracle", "trace": oracle_encoding, "encode_seconds": oracle_encode_seconds,
                "encoding_prompt": asdict(encoding_prompt), "array": episode["family_id"] + "_oracle.npy",
                "packet": episode["family_id"] + "_oracle.bin"})
            if comparison_layer is not None:
                (other_packet, other_trace, other_states), other_seconds = timed(lambda: encode_with_states(
                    client, encoding_prompt, oracle, layer=comparison_layer, codec="int4"))
                comparisons["oracle"] = other_packet
                other_stem = episode["family_id"] + "_oracle_comparison"
                np.save(raw / (other_stem + ".npy"), other_states, allow_pickle=False)
                (raw / (other_stem + ".bin")).write_bytes(other_packet)
                append_row(stage.directory / "comparison_encodings.jsonl", {"family_id": episode["family_id"],
                    "variant": "oracle", "trace": other_trace, "encode_seconds": other_seconds,
                    "encoding_prompt": asdict(encoding_prompt), "array": other_stem + ".npy", "packet": other_stem + ".bin"})
            conditions = [
                ("text_generated", lambda: receive_text(client, prompt, source["original"][0].encode("utf-8")), episode),
                ("raw_generated", lambda: receive_packet(client, prompt, source["original"][1], expected_layer=layer), episode),
                ("text_oracle", lambda: receive_text(client, prompt, oracle.encode("utf-8")), episode),
                ("raw_oracle", lambda: receive_packet(client, prompt, oracle_packet, expected_layer=layer), episode),
                ("no_comm", lambda: receive_packet(client, prompt, None, expected_layer=layer), episode),
                ("text_swapped", lambda: receive_text(client, prompt, source["alternate"][0].encode("utf-8")), originals["alternate"]),
                ("raw_swapped", lambda: receive_packet(client, prompt, source["alternate"][1], expected_layer=layer), originals["alternate"]),
            ]
            if comparison_layer is not None:
                conditions.extend([
                    ("raw_comparison_generated", lambda: receive_packet(client, prompt, comparisons["original"], expected_layer=comparison_layer), episode),
                    ("raw_comparison_oracle", lambda: receive_packet(client, prompt, comparisons["oracle"], expected_layer=comparison_layer), episode),
                    ("raw_comparison_swapped", lambda: receive_packet(client, prompt, comparisons["alternate"], expected_layer=comparison_layer), originals["alternate"]),
                ])
            for condition, fn, evaluation in conditions:
                stage.update(stage="receiver_generation", case_id=episode["family_id"], condition=condition)
                result, seconds = timed(fn)
                result = attach_task_response(agent_view(episode, "B"), result)
                score = score_answer(evaluation, result["task_answer"])
                append_row(stage.directory / "rows.jsonl", {"family_id": episode["family_id"], "task": task,
                    "condition": condition, "result": result, "score": score, "seconds": seconds,
                    "score_original": score_answer(originals["original"], result["task_answer"]),
                    "score_alternate": score_answer(originals["alternate"], result["task_answer"]),
                    "receiver_prompt": asdict(prompt), "receiver_prompt_text": parts})
                summary[task][condition]["n"] += 1
                summary[task][condition]["successes"] += int(score["success"])
                counter += 1
                stage.update(n_completed=counter)
        write_json(stage.directory / "summary.json", dict(summary))
        stage.finish()
        print({"status": "completed", "model": model, "summary": dict(summary)})
    except BaseException as error:
        stage.fail(error)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", choices=("qwen", "llama"), required=True)
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--layer", type=int)
    p.add_argument("--comparison-layer", type=int)
    a = p.parse_args()
    run(a.data, a.config, a.output, a.model, a.count, a.layer, a.comparison_layer)
