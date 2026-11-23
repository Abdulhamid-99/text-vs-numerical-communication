"""Development v2: explicit data blocks; task information remains disjoint."""
from private_tasks import QA_TASK, NAV_TASK
from common import render_receiver_parts, public_prompt

PROMPT_VERSION = "task_prompts_v4_predicted_map_planner_sender_shape"
SYSTEMS = {
    QA_TASK: "You solve a two-table lookup. The first data block is agent A's directory. EVERY line, including the first, is a data row: AAAA>BBBB means subject AAAA has relay BBBB. The second block is your private registry: BBBB=CCCC means relay BBBB has code CCCC. Find the requested subject in the directory, take its relay, and then look up that relay in the registry. Copy the complete 16-letter code exactly. Output only that code. If the directory contains no usable subject-to-relay data, output UNKNOWN.",
    NAV_TASK: "You are the map-reading component of a navigation agent. Recover the complete ASCII map from the GRID data block. Each line is one row. Preserve every '.' and '#' character and the exact row order. Return ONLY JSON with one key grid whose value is the list of map-row strings. A separate deterministic route planner will navigate using YOUR recovered map and the private start/goal. Your job is to report the map accurately; do not invent, simplify, repair or plan a route. If no usable map was supplied, output UNKNOWN.",
}


def sender_messages(view):
    content = view["private_context"]
    if view["task"] == NAV_TASK:
        rows = content.splitlines()
        if not rows or any(len(row) != len(rows[0]) or set(row) - {".", "#"} for row in rows):
            raise ValueError("Malformed sender-owned map")
        content = (f"Copy the complete map between MAP_BEGIN and MAP_END. It has {len(rows)} rows, "
                   f"each with exactly {len(rows[0])} characters. Preserve repeated rows; do not add or remove any row or character. "
                   "Reply with ONLY the map lines, without markers or a code fence.\nMAP_BEGIN\n" + content + "\nMAP_END")
    return [{"role": "system", "content": view["public_instruction"]},
            {"role": "user", "content": content}]


def _before(task):
    if task == QA_TASK:
        return "DIRECTORY data (all lines are records):\n```text\n"
    if task == NAV_TASK:
        return "GRID data (every character is one cell):\n```text\n"
    raise ValueError("Unsupported task")


def receiver_parts(client, view):
    task = view["task"]
    if task == QA_TASK:
        after = "\n```\n\nREGISTRY data:\n```text\n" + view["private_context"] + "\n```\n\n" + view["query"] + "\nReply with ONLY the 16-letter code. No explanation, table, or steps.\nCode:"
    elif task == NAV_TASK:
        after = "\n```\n\nPrivate navigation task:\n" + view["private_context"] + "\n" + view["query"] + "\nThe route planner handles that task. Return ONLY the recovered map as {\"grid\":[\"row0\",\"row1\",...]}, with the actual map characters. No route or explanation.\nMap JSON:"
    else:
        raise ValueError("Unsupported task")
    prefix, suffix, messages = render_receiver_parts(client, SYSTEMS[task], _before(task), after)
    return public_prompt(client, prefix, suffix), {"prefix_text": prefix, "suffix_text": suffix,
                                                  "template_messages": messages}


def sender_encoding_prompt(client, task):
    prefix, suffix, _ = render_receiver_parts(client, SYSTEMS[task], _before(task), "\n```\nAnswer:")
    return public_prompt(client, prefix, suffix)
