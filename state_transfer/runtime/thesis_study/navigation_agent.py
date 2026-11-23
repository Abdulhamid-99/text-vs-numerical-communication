"""Public deterministic planner over B's MODEL-PREDICTED map only.

No evaluator object, A context, seed, packet or source trace is an input. The
model prediction is parsed without access to the true map. Environment grading
is a separate process and may reject this planner's executable predicted route.
"""
from __future__ import annotations
from collections import deque
import json
import re

NAVIGATION_AGENT_VERSION = "predicted_map_bfs_v1_preheldout"
MOVES = (('N', -1, 0), ('E', 0, 1), ('S', 1, 0), ('W', 0, -1))


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("Duplicate JSON key")
        out[key] = value
    return out


def private_goal_context(view):
    if set(view) != {"task", "public_instruction", "private_context", "query"}:
        raise ValueError("Unexpected private fields at navigation-agent boundary")
    match = re.fullmatch(r"Grid: ([0-9]+) rows, ([0-9]+) columns\.\nStart: ([0-9]+),([0-9]+)\nGoal: ([0-9]+),([0-9]+)", view["private_context"])
    if match is None:
        raise ValueError("Malformed B-owned navigation task")
    rows, columns, sr, sc, gr, gc = map(int, match.groups())
    if not 2 <= rows <= 6 or rows != columns or not (0 <= sr < rows and 0 <= gr < rows and 0 <= sc < columns and 0 <= gc < columns):
        raise ValueError("B-owned dimensions/coordinates are outside supported public bounds")
    return rows, columns, (sr, sc), (gr, gc)


def predict_map(raw_answer, rows, columns):
    """Parse a sole JSON grid; a single surrounding Markdown fence is allowed."""
    if not isinstance(raw_answer, str):
        return None, "invalid_model_map"
    value = raw_answer.strip()
    if value.upper() == "UNKNOWN" or not value:
        return None, "no_model_map"
    if value.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", value, re.DOTALL)
        if match is None:
            return None, "invalid_model_map"
        value = match.group(1)
    try:
        document = json.loads(value, object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    except (ValueError, TypeError):
        return None, "invalid_model_map"
    if (not isinstance(document, dict) or set(document) != {"grid"} or not isinstance(document["grid"], list)
            or len(document["grid"]) != rows
            or any(not isinstance(row, str) or len(row) != columns or set(row) - {".", "#"} for row in document["grid"])):
        return None, "invalid_model_map"
    return document["grid"], "parsed"


def plan_from_prediction(view, raw_answer):
    """Return a route over predicted terrain, or an explicit failed-planning status."""
    rows, columns, start, goal = private_goal_context(view)
    grid, parse_status = predict_map(raw_answer, rows, columns)
    trace = {"version": NAVIGATION_AGENT_VERSION, "status": parse_status, "predicted_grid": grid,
             "start": list(start), "goal": list(goal), "max_moves": 50,
             "planner_kind": "Deterministic BFS over model-predicted map, N/E/S/W neighbor order",
             "environment_oracle_available": False, "expanded_cells": 0, "route": None}
    if grid is None:
        return {"task_answer": "UNKNOWN", "navigation_planner": trace}
    if grid[start[0]][start[1]] != "." or grid[goal[0]][goal[1]] != ".":
        trace["status"] = "predicted_endpoint_blocked"
        return {"task_answer": "UNKNOWN", "navigation_planner": trace}
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        current, route = queue.popleft()
        trace["expanded_cells"] += 1
        if current == goal:
            answer = {"start": list(start), "steps": route}
            trace.update(status="route_planned", route=answer)
            return {"task_answer": json.dumps(answer, separators=(",", ":")), "navigation_planner": trace}
        if len(route) >= 50:
            continue
        for action, dr, dc in MOVES:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < columns and grid[nr][nc] == "." and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append(((nr, nc), route + [action]))
    trace["status"] = "no_route_on_predicted_map"
    return {"task_answer": "UNKNOWN", "navigation_planner": trace}


def attach_task_response(view, result):
    """Preserve actual neural continuation separately from the agent's tool output."""
    if any(name in result for name in ("task_answer", "navigation_planner")):
        raise ValueError("Task adapter already applied or supplied in neural result")
    if view["task"] == "private_multidocument_qa_v1":
        return {**result, "task_answer": result["answer"]}
    if view["task"] != "private_planned_navigation_v1":
        raise ValueError("Unsupported agent task")
    return {**result, **plan_from_prediction(view, result["answer"])}
