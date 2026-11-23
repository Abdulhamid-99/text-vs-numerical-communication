"""Deterministic private-information task families, independent of model/runtime code.

Private seeds are inputs to dataset creation, never agent-visible task metadata.
Each family contains original/alternate episodes with an identical B view.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import re
from typing import Iterable


SCHEMA_VERSION = 1
SCORER_VERSION = "private_task_semantic_v3_strict_subset_preheldout"
SPLITS = frozenset({"calibration", "development", "heldout"})
QA_TASK = "private_multidocument_qa_v1"
NAV_TASK = "private_planned_navigation_v1"
ACTIONS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
QA_PUBLIC = {
    "A": "Your private directory maps subjects to relays: SUBJECT>RELAY. Send every directory row exactly as supplied, one per line. Your partner has a separate relay-to-code registry and a private question. Send only the directory; do not guess the code.",
    "B": "Combine the received SUBJECT>RELAY directory with your private RELAY=CODE registry. Find the queried subject's relay, then its code. Return only the complete unchanged 16-letter code. If the directory is absent or does not determine the subject's relay, return UNKNOWN. Do not guess.",
}
NAV_PUBLIC = {
    "A": "Send your entire grid exactly as supplied, one row per line. A dot is traversable; # is blocked. Row 0 is the top and column 0 the left. Your partner privately knows its start and goal. Send only the grid, without a route or explanation.",
    "B": "Use the received grid: . is traversable and # is blocked; row 0 is top, column 0 is left. Plan unit moves N,E,S,W from your private start to goal. Return only JSON {\"start\":[row,column],\"steps\":[\"N\",\"E\"]}, with at most 50 moves. Any executable route ending at the goal within 50 moves is valid; revisits are allowed. If the map is absent, return UNKNOWN. Do not invent a map.",
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class _Stream:
    """Local SHA256 counter stream; no global RNG state or Python hash dependence."""
    def __init__(self, seed, task, split, family_id):
        if isinstance(seed, bool) or not isinstance(seed, (str, int)) or seed == "":
            raise ValueError("seed must be a nonempty private string or integer")
        if split not in SPLITS:
            raise ValueError("split must be calibration, development or heldout")
        if not isinstance(family_id, str) or not family_id.strip():
            raise ValueError("family_id must be a nonempty string")
        self.key = hashlib.sha256(_canonical([SCHEMA_VERSION, seed, task, split, family_id]).encode()).digest()
        self.counter = 0

    def below(self, n):
        if type(n) is not int or n <= 0:
            raise ValueError("Invalid random range")
        limit = (1 << 256) - ((1 << 256) % n)
        while True:
            value = int.from_bytes(hashlib.sha256(self.key + self.counter.to_bytes(16, "big")).digest(), "big")
            self.counter += 1
            if value < limit:
                return value % n

    def shuffle(self, items):
        for index in range(len(items) - 1, 0, -1):
            other = self.below(index + 1)
            items[index], items[other] = items[other], items[index]


def _exclusions(values):
    result = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("Excluded values must be strings")
        result.add(value.strip().upper())
    return result


def _opaque(stream, length, unavailable):
    for _ in range(10000):
        value = "".join(chr(65 + stream.below(26)) for _ in range(length))
        if value not in unavailable:
            unavailable.add(value)
            return value
    raise ValueError("Opaque namespace exhausted by exclusions")


def _episode(task, split, family_id, variant, a_context, b_context, query, evaluator):
    return {"schema_version": SCHEMA_VERSION, "task": task, "family_id": family_id,
            "split": split, "variant": variant, "a_private_context": a_context,
            "b_private_context": b_context, "query": query, "evaluator": evaluator}


def generate_qa_pair(seed, *, split, family_id, rows=4, excluded_values: Iterable[str] = ()):
    """Generate a private two-document join and a critical-fact counterpart.

    A knows subject->relay; B knows relay->code and the selected subject. The
    alternate swaps two directory assignments; code values/order and B stay fixed.
    """
    if type(rows) is not int or not 4 <= rows <= 6:
        raise ValueError("QA directory must have 4 to 6 rows")
    stream = _Stream(seed, QA_TASK, split, family_id)
    unavailable = _exclusions(excluded_values)
    subjects = [_opaque(stream, 4, unavailable) for _ in range(rows)]
    relays = [_opaque(stream, 4, unavailable) for _ in range(rows)]
    codes = [_opaque(stream, 16, unavailable) for _ in range(rows)]
    assigned = list(relays)
    stream.shuffle(assigned)
    queried = stream.below(rows)
    changed = (queried + 1 + stream.below(rows - 1)) % rows
    directory = [[subject, relay] for subject, relay in zip(subjects, assigned)]
    alternate_directory = deepcopy(directory)
    alternate_directory[queried][1], alternate_directory[changed][1] = (
        alternate_directory[changed][1], alternate_directory[queried][1])
    order = list(range(rows))
    stream.shuffle(order)
    directory = [directory[index] for index in order]
    alternate_directory = [alternate_directory[index] for index in order]
    registry = [[relay, code] for relay, code in zip(relays, codes)]
    stream.shuffle(registry)
    b_context = "\n".join(f"{relay}={code}" for relay, code in registry)
    query = f"What is the code for subject {subjects[queried]}?"
    episodes = {}
    for variant, facts in (("original", directory), ("alternate", alternate_directory)):
        episodes[variant] = _episode(QA_TASK, split, family_id, variant,
            "\n".join(f"{subject}>{relay}" for subject, relay in facts), b_context, query,
            {"directory": facts, "registry": deepcopy(registry), "queried_subject": subjects[queried]})
    return {"schema_version": SCHEMA_VERSION, "task": QA_TASK, "split": split, "family_id": family_id,
            "episodes": episodes, "private_value_inventory": sorted(subjects + relays + codes),
            "exclusion_inventory": sorted(subjects + relays + codes),
            "design": {"documents": 2, "join_hops": 2, "rows": rows,
                       "counterpart": "swap queried and distractor relay assignments; same B view",
                       "seed_is_not_exported": True}}


def _reachable(grid, start, goal):
    # Generation only tests reachability; it never stores a route or answer.
    frontier, reached = [tuple(start)], {tuple(start)}
    while frontier:
        r, c = frontier.pop()
        if [r, c] == goal:
            return True
        for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            q = (r + dr, c + dc)
            if (0 <= q[0] < len(grid) and 0 <= q[1] < len(grid[0])
                    and grid[q[0]][q[1]] == "." and q not in reached):
                reached.add(q)
                frontier.append(q)
    return False


def generate_navigation_pair(seed, *, split, family_id, size=6,
                             excluded_values: Iterable[str] = ()):
    """Random compact paired maps with disjoint executable first actions.

    Counterparts share all cells except two critical neighbors of the start.
    Exactly one first move is possible in each map, and those moves differ.
    Both maps retain a route to the private goal. No route is stored by generation.
    """
    if type(size) is not int or size not in (5, 6):
        raise ValueError("Navigation size must be 5 or 6")
    stream = _Stream(seed, NAV_TASK, split, family_id)
    unavailable = _exclusions(excluded_values)
    for _attempt in range(10000):
        start = [1 + stream.below(size - 2), 1 + stream.below(size - 2)]
        possible_goals = [[r, c] for r in range(size) for c in range(size)
                          if abs(r - start[0]) + abs(c - start[1]) >= 3]
        goal = possible_goals[stream.below(len(possible_goals))]
        directions = list(ACTIONS.values())
        stream.shuffle(directions)
        base = [["." if stream.below(100) < 65 else "#" for _ in range(size)] for _ in range(size)]
        base[start[0]][start[1]] = base[goal[0]][goal[1]] = "."
        for dr, dc in ACTIONS.values():
            base[start[0] + dr][start[1] + dc] = "#"
        maps = []
        for dr, dc in directions[:2]:
            candidate = deepcopy(base)
            candidate[start[0] + dr][start[1] + dc] = "."
            maps.append(["".join(row) for row in candidate])
        if not all(_reachable(grid, start, goal) for grid in maps):
            continue
        b_context = f"Grid: {size} rows, {size} columns.\nStart: {start[0]},{start[1]}\nGoal: {goal[0]},{goal[1]}"
        map_ids = ["nav_map_sha256:" + hashlib.sha256("\n".join(grid).encode()).hexdigest() for grid in maps]
        instance_ids = ["nav_instance_sha256:" + hashlib.sha256(_canonical([grid, start, goal]).encode()).hexdigest() for grid in maps]
        inventory = map_ids + instance_ids
        if any(value.upper() in unavailable for value in inventory):
            continue
        episodes = {}
        for variant, grid in zip(("original", "alternate"), maps):
            episodes[variant] = _episode(NAV_TASK, split, family_id, variant, "\n".join(grid),
                b_context, "Return a complete executable route from start to goal, using at most 50 moves.",
                {"grid": grid, "start": start[:], "goal": goal[:], "max_steps": 50})
        return {"schema_version": SCHEMA_VERSION, "task": NAV_TASK, "split": split, "family_id": family_id,
                "episodes": episodes, "private_value_inventory": [], "exclusion_inventory": inventory,
                "design": {"mode": "nonadaptive_planned_route", "size": size,
                           "counterpart": "two critical cells swap; identical B view and disjoint executable first actions",
                           "topology": "random 65-percent-open proposal conditioned on paired reachability",
                           "seed_is_not_exported": True, "generation_attempts": _attempt + 1}}
    raise ValueError("Could not construct a new reachable navigation family under exclusions")


def agent_view(episode, agent):
    """Fresh allowlisted view; never pass the full episode to a model API."""
    if agent not in ("A", "B"):
        raise ValueError("agent must be A or B")
    prompts = QA_PUBLIC if episode["task"] == QA_TASK else NAV_PUBLIC if episode["task"] == NAV_TASK else None
    if prompts is None:
        raise ValueError("Unknown task")
    view = {"task": episode["task"], "public_instruction": prompts[agent],
            "private_context": episode["a_private_context" if agent == "A" else "b_private_context"]}
    if agent == "B":
        view["query"] = episode["query"]
    return view


def sender_requested_message(episode):
    """Canonical A-only oracle payload; uses neither B context nor evaluator."""
    return agent_view(episode, "A")["private_context"]


def reference_answer(episode):
    """Independent reference: relational lookup or BFS over actual evaluator map."""
    state = episode["evaluator"]
    if episode["task"] == QA_TASK:
        directory, registry = dict(state["directory"]), dict(state["registry"])
        if len(directory) != len(state["directory"]) or len(registry) != len(state["registry"]):
            raise ValueError("Ambiguous evaluator records")
        return registry[directory[state["queried_subject"]]]
    if episode["task"] != NAV_TASK:
        raise ValueError("Unknown task")
    grid, start, goal = state["grid"], tuple(state["start"]), tuple(state["goal"])
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        position, route = queue.popleft()
        if position == goal:
            return {"start": list(start), "steps": route}
        if len(route) >= state["max_steps"]:
            continue
        for action, (dr, dc) in ACTIONS.items():
            candidate = (position[0] + dr, position[1] + dc)
            r, c = candidate
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == "." and candidate not in seen:
                seen.add(candidate)
                queue.append((candidate, route + [action]))
    raise ValueError("Evaluator map has no route within the action budget")


def _position(value):
    return isinstance(value, list) and len(value) == 2 and all(type(x) is int for x in value)


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _score_navigation_strict(state, answer):
    """Execute all proposed moves, with independent status and cycle diagnostics.

    Reaching the goal at the END within <=50 moves is the success definition.
    An explicit wrong start, malformed route, or any invalid attempted move fails.
    Routes longer than the budget are timeout without partial acceptance.
    """
    grid = state["grid"]
    start, goal, budget = list(state["start"]), list(state["goal"]), state["max_steps"]
    if (not grid or any(not isinstance(row, str) or len(row) != len(grid[0]) or set(row) - {".", "#"} for row in grid)
            or not _position(start) or not _position(goal) or type(budget) is not int or not 0 <= budget <= 50):
        raise ValueError("Invalid evaluator map, coordinates or budget")
    for r, c in (start, goal):
        if not 0 <= r < len(grid) or not 0 <= c < len(grid[0]) or grid[r][c] != ".":
            raise ValueError("Evaluator endpoint is outside a traversable cell")
    result = {"success": False, "status": "invalid_format", "executed_steps": 0,
              "proposed_steps": None, "position": start[:], "goal": goal,
              "cycle_count": 0, "cycle_detected": False, "failed_step": None}
    if isinstance(answer, str):
        if answer.strip().casefold() in ("", "unknown"):
            result["status"] = "abstained"
            return result
        try:
            answer = json.loads(answer, object_pairs_hook=_unique_json_object)
        except (ValueError, TypeError):
            return result
    if (not isinstance(answer, dict) or set(answer) != {"start", "steps"}
            or not _position(answer["start"]) or not isinstance(answer["steps"], list)):
        return result
    steps = answer["steps"]
    result["proposed_steps"] = len(steps)
    if answer["start"] != start:
        result["status"] = "wrong_start"
        return result
    if any(not isinstance(action, str) or action not in ACTIONS for action in steps):
        result["status"] = "invalid_action"
        result["failed_step"] = next(i + 1 for i, action in enumerate(steps) if not isinstance(action, str) or action not in ACTIONS)
        return result
    if len(steps) > budget:
        result["status"] = "timeout"
        return result
    seen = {tuple(start)}
    for index, action in enumerate(steps, 1):
        dr, dc = ACTIONS[action]
        candidate = [result["position"][0] + dr, result["position"][1] + dc]
        r, c = candidate
        if not 0 <= r < len(grid) or not 0 <= c < len(grid[0]):
            result.update(status="out_of_bounds", failed_step=index)
            return result
        if grid[r][c] == "#":
            result.update(status="blocked", failed_step=index)
            return result
        result["position"] = candidate
        result["executed_steps"] = index
        if tuple(candidate) in seen:
            result["cycle_count"] += 1
            result["cycle_detected"] = True
        seen.add(tuple(candidate))
    result["success"] = result["position"] == goal
    result["status"] = "success" if result["success"] else "wrong_goal"
    return result


def _route_candidate(answer):
    """Parse independently of map/gold; never select a good route from a list.

    One JSON object may be surrounded by plain prose/fences. Multiple containers,
    malformed containers, duplicate keys, arrays and nested wrapper objects are
    not rescued by looking for an inner route. Nonfinite JSON is refused.
    """
    if isinstance(answer, dict):
        return answer, True, "one_object", 1
    if not isinstance(answer, str):
        return None, False, "invalid_format", 0
    decoder = json.JSONDecoder(object_pairs_hook=_unique_json_object,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    text = answer.strip()
    if not text or text.casefold() == "unknown":
        return None, False, "abstained", 0
    candidates, spans, cursor = [], [], 0
    while cursor < len(text):
        indexes = [index for char in "{[" if (index := text.find(char, cursor)) >= 0]
        if not indexes:
            break
        start = min(indexes)
        try:
            value, end = decoder.raw_decode(text, start)
        except (ValueError, TypeError):
            return None, False, "invalid_format", len(candidates)
        candidates.append(value)
        spans.append((start, end))
        cursor = end
    if len(candidates) > 1:
        return None, False, "ambiguous_answer", len(candidates)
    if not candidates or not isinstance(candidates[0], dict):
        return None, False, "invalid_format", len(candidates)
    start, end = spans[0]
    surrounding = text[:start] + text[end:]
    if any(character in surrounding for character in "{}[]"):
        return None, False, "invalid_format", 1
    return candidates[0], start == 0 and end == len(text), "one_object", 1


def score_navigation(state, answer):
    candidate, strict_format, extraction, candidate_count = _route_candidate(answer)
    # Even a malformed response does not turn malformed evaluator state into a
    # task failure: validation is still delegated to the execution scorer.
    result = _score_navigation_strict(state, candidate)
    if extraction != "one_object":
        result["status"] = extraction
    if result["status"] == "invalid_format":
        strict_format = False
    result.update(scorer_version=SCORER_VERSION, strict_format=strict_format,
                  strict_success=result["success"] and strict_format,
                  extraction_status=extraction, candidate_count=candidate_count,
                  extracted_route=candidate)
    return result


def score_answer(episode, answer):
    if episode["task"] == NAV_TASK:
        return score_navigation(episode["evaluator"], answer)
    if episode["task"] != QA_TASK:
        raise ValueError("Unknown task")
    gold = reference_answer(episode)
    if not isinstance(answer, str):
        return {"success": False, "status": "invalid_format", "raw_exact": False,
                "strict_success": False, "scorer_version": SCORER_VERSION,
                "candidates": [], "candidate_count": 0, "candidate_occurrences": 0,
                "extracted_answer": None}
    # Unicode word boundaries prevent extracting 16 letters from a longer
    # identifier (including non-ASCII neighbors); case-sensitive by declaration.
    occurrences = re.findall(r"(?<!\w)[A-Z]{16}(?!\w)", answer)
    candidates = list(dict.fromkeys(occurrences))
    chosen = candidates[0] if len(candidates) == 1 else None
    success = chosen == gold
    normalized = answer.strip().casefold()
    status = ("success" if success else "abstained" if normalized in ("", "unknown") else
              "ambiguous_answer" if len(candidates) > 1 else "no_candidate" if not candidates else "wrong_answer")
    return {"success": success, "status": status, "raw_exact": answer == gold,
            "strict_success": success and answer.strip() == gold, "scorer_version": SCORER_VERSION,
            "candidates": candidates, "candidate_count": len(candidates),
            "candidate_occurrences": len(occurrences), "extracted_answer": chosen}


def family_fingerprint(family):
    """Content identity for a frozen family; never include this in agent prompts."""
    return hashlib.sha256(_canonical(family).encode()).hexdigest()


def validate_split_isolation(families):
    """Check family IDs/private tokens across splits; report NAV view reuse openly.

    QA values and exact NAV map/instance fingerprints belong in the exclusion
    ledger. This checks exact overlap, not broad graph-isomorphism equivalence.
    """
    family_splits, value_splits, nav_views = {}, {}, {}
    for family in families:
        split = family["split"]
        if split not in SPLITS or set(family["episodes"]) != {"original", "alternate"}:
            raise ValueError("Invalid split or incomplete counterpart family")
        if any(episode["split"] != split or episode["family_id"] != family["family_id"] for episode in family["episodes"].values()):
            raise ValueError("Episode members disagree with their family partition")
        identity = (family["task"], family["family_id"])
        if identity in family_splits:
            raise ValueError("Duplicate family_id, within or across dataset splits")
        family_splits[identity] = split
        for value in family.get("exclusion_inventory", family["private_value_inventory"]):
            if value in value_splits and value_splits[value] != split:
                raise ValueError("Private values or navigation map/instance identities cross dataset splits")
            value_splits[value] = split
        if family["task"] == NAV_TASK:
            for episode in family["episodes"].values():
                key = hashlib.sha256(_canonical([episode["a_private_context"], episode["b_private_context"]]).encode()).hexdigest()
                nav_views.setdefault(key, set()).add(split)
    return {"family_count": len(family_splits), "exclusion_value_count": len(value_splits),
            "cross_split_navigation_view_reuse": sum(len(splits) > 1 for splits in nav_views.values()),
            "navigation_topology_generalization_is_not_established": bool(nav_views)}
