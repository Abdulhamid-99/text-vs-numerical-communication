"""Development-only navigation text-oracle presentation diagnostic; no heldout or A-generation claim."""
from pathlib import Path
import argparse
from collections import defaultdict
from dataclasses import asdict

from budgetfair.model.client import LLMClient, GenerationConfig
from budgetfair.model.span_transport import receive_text
from common import Stage, read_json, verify_seal, write_json, append_row, timed, sha, render_receiver_parts, public_prompt
from private_tasks import NAV_TASK, agent_view, score_answer, reference_answer
from task_prompts import receiver_parts


def wall_coordinates(private_ascii_map):
    grid = private_ascii_map.splitlines()
    if not grid or any(len(row) != len(grid[0]) or set(row) - {".", "#"} for row in grid):
        raise ValueError("Invalid sender-owned ASCII map")
    walls = [(r,c) for r,row in enumerate(grid) for c,char in enumerate(row) if char == "#"]
    return f"size={len(grid)}x{len(grid[0])}\nwalls=" + ";".join(f"{r},{c}" for r,c in walls)


DELTAS = "Coordinates are (row,column), both zero-based. N changes (r,c) to (r-1,c), S to (r+1,c), E to (r,c+1), and W to (r,c-1). Check every destination cell, and stop exactly at GOAL. Never enter a wall or leave the grid. "
SYSTEM_ASCII = "Find a grid route. Each ASCII line is a row: the first line is row0. Each character is a cell: the first character is column0. '.' means open and '#' means wall. " + DELTAS + "Return only a JSON object with start and steps keys; steps is a list of N,E,S,W moves, at most50. If no map is supplied, return UNKNOWN."
SYSTEM_WALLS = "Find a grid route using the supplied map. size=HxW gives the number of rows and columns. walls lists blocked coordinates as row,column pairs separated by semicolons. EVERY other in-bounds cell is open. " + DELTAS + "Return only a JSON object with start and steps keys; steps is a list of N,E,S,W moves, at most50. If no map is supplied, return UNKNOWN."


def explicit_parts(client, view, system):
    before = "MAP data:\n```text\n"
    after = "\n```\n\n" + view["private_context"] + "\n" + view["query"] + "\nThe last position must equal the goal. Reply ONLY with {\"start\":[row,column],\"steps\":[...]}. No explanation."
    prefix, suffix, template = render_receiver_parts(client, system, before, after)
    return public_prompt(client, prefix, suffix), {"prefix_text": prefix, "suffix_text": suffix, "template_messages": template}


def run(data, config, output, model, count=8):
    verify_seal(data)
    families = [f for f in read_json(data / "development.json")["families"] if f["task"] == NAV_TASK][:count]
    if len(families) != count or count < 1 or any(f["split"] != "development" for f in families):
        raise ValueError("Incomplete navigation development input")
    stage = Stage(output, "navigation_text_oracle_format_diagnostic", requested=3*count, model=model)
    try:
        client, load_seconds = timed(lambda: LLMClient.from_config(config))
        client.generation = GenerationConfig(max_new_tokens=192, temperature=0.0, do_sample=False)
        write_json(output / "configuration.json", {"model": model, "config_sha256": sha(config), "model_config": read_json(config),
            "source_sha256": sha(__file__), "model_load_seconds": load_seconds, "data_manifest_sha256": sha(data / "artifact_manifest.json"),
            "scope": "Paired text-oracle presentation diagnostic, development only; canonical source from A-owned map; zero sender generations"})
        summary = defaultdict(lambda: {"n": 0, "successes": 0})
        n = 0
        for family in families:
            episode = family["episodes"]["original"]
            view = agent_view(episode, "B")
            ascii_map = episode["a_private_context"]
            if not score_answer(episode, reference_answer(episode))["success"]:
                raise ValueError("Independent BFS oracle does not solve generated map")
            formats = [("ascii_v2", ascii_map, receiver_parts(client, view)),
                       ("ascii_explicit_coordinates", ascii_map, explicit_parts(client, view, SYSTEM_ASCII)),
                       ("wall_coordinates", wall_coordinates(ascii_map), explicit_parts(client, view, SYSTEM_WALLS))]
            for condition, message, (prompt, parts) in formats:
                stage.update(stage="receiver_generation", case_id=family["family_id"], condition=condition)
                if len(client.tokenizer.encode(message, add_special_tokens=False)) > 128:
                    raise ValueError("Diagnostic message exceeds common payload token limit")
                result, seconds = timed(lambda: receive_text(client, prompt, message.encode("utf-8")))
                score = score_answer(episode, result["answer"])
                append_row(output / "rows.jsonl", {"family_id": family["family_id"], "condition": condition,
                    "message": message, "result": result, "score": score, "seconds": seconds,
                    "receiver_prompt": asdict(prompt), "receiver_prompt_text": parts})
                summary[condition]["n"] += 1
                summary[condition]["successes"] += int(score["success"])
                n += 1
                stage.update(n_completed=n)
        write_json(output / "summary.json", dict(summary))
        stage.finish()
        print(dict(summary))
    except BaseException as error:
        stage.fail(error)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "config", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--model", choices=("qwen", "llama"), required=True)
    p.add_argument("--count", type=int, default=8)
    a = p.parse_args()
    run(a.data, a.config, a.output, a.model, a.count)
