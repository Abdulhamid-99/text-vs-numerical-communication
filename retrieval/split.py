"""Outcome-free ArguAna split audit; deliberately imports no study code."""

import collections
import hashlib
import json
import math
import unicodedata


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized_content(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def content_hash(value):
    return digest(normalized_content(value).encode("utf-8"))


def debate(identifier):
    return identifier.rsplit("-", 1)[0]


def build(corpus, queries, relevance, seed=73021):
    cpos = {row["_id"]: i for i, row in enumerate(corpus)}
    qpos = {row["_id"]: i for i, row in enumerate(queries)}
    if len(cpos) != len(corpus) or len(qpos) != len(queries):
        raise ValueError("Duplicate source IDs")
    if set(relevance) - set(qpos):
        raise ValueError("Unknown qrel query IDs")
    ch = [content_hash(row["text"]) for row in corpus]
    qh = [content_hash(row["text"]) for row in queries]
    groups = {debate(row["_id"]) for row in queries}
    parent = {g: g for g in groups}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    content_owner = {}
    query_content = set(qh)
    protected_content = set(qh)
    missing, eligible = [], []
    per_query_content = []
    for i, row in enumerate(queries):
        qid, g = row["_id"], debate(row["_id"])
        targets = relevance.get(qid, [])
        absent = sorted(d for d in targets if d not in cpos)
        if not targets or absent:
            missing.append(
                {
                    "query_index": i,
                    "query_id": qid,
                    "missing_target_ids": absent,
                    "reason": "missing_target" if absent else "no_positive_qrel",
                }
            )
        else:
            eligible.append(i)
        # Include available contents even from ineligible rows when joining
        # debate groups. Exclusion cannot reopen a content link.
        hashes = {qh[i]} | {ch[cpos[d]] for d in targets if d in cpos}
        per_query_content.append(hashes)
        protected_content.update(hashes)
        for h in hashes:
            if h in content_owner:
                union(g, content_owner[h])
            else:
                content_owner[h] = g

    component_groups = collections.defaultdict(set)
    for g in sorted(groups):
        component_groups[find(g)].add(g)
    component_ids = {
        root: digest(json.dumps(sorted(gs), separators=(",", ":")).encode())
        for root, gs in component_groups.items()
    }
    active_components = {
        component_ids[find(debate(queries[i]["_id"]))] for i in eligible
    }
    if len(active_components) < 2:
        raise ValueError("Too few independent components for development/test")
    ordered = sorted(
        active_components, key=lambda s: digest(f"split-{seed}-{s}".encode())
    )
    dev_components = set(
        ordered[: min(len(ordered) - 1, math.ceil(0.30 * len(ordered)))]
    )
    assignment = {i: component_ids[find(debate(queries[i]["_id"]))] for i in eligible}
    dev = [i for i in eligible if assignment[i] in dev_components]
    test = [i for i in eligible if assignment[i] not in dev_components]

    all_query_ids = set(qpos)
    removed_receiver = [
        i
        for i, row in enumerate(corpus)
        if row["_id"] in all_query_ids or ch[i] in query_content
    ]
    receiver = [i for i in range(len(corpus)) if i not in set(removed_receiver)]
    receiver_set = set(receiver)
    conflicts = [
        {"query_index": i, "target_id": d, "corpus_index": cpos[d]}
        for i in eligible
        for d in relevance[queries[i]["_id"]]
        if cpos[d] not in receiver_set
    ]
    if conflicts:
        raise ValueError(
            "Receiver exclusion would remove relevant targets: " + json.dumps(conflicts)
        )
    fit_all = [i for i, row in enumerate(corpus) if row["_id"].startswith("training-")]
    fit = [i for i in fit_all if ch[i] not in protected_content]
    fit_removed = [i for i in fit_all if i not in set(fit)]

    dev_hash = set().union(*(per_query_content[i] for i in dev))
    test_hash = set().union(*(per_query_content[i] for i in test))
    assert not dev_hash & test_hash
    assert not {ch[i] for i in fit} & protected_content
    assert not {ch[i] for i in receiver} & query_content
    assert not {corpus[i]["_id"] for i in receiver} & all_query_ids
    assert not set(dev) & set(test)
    assert set(dev) | set(test) == set(eligible)
    assert not (
        {debate(queries[i]["_id"]) for i in dev}
        & {debate(queries[i]["_id"]) for i in test}
    )

    return {
        "schema": 1,
        "purpose": "outcome-free internal split of modified BEIR ArguAna; not official BEIR scores",
        "seed": seed,
        "development_component_fraction": 0.30,
        "normalization": "Unicode NFKC, casefold, whitespace collapse; text field only",
        "assignment_rule": "Union entire debate groups sharing normalized query or relevant-document content; hash component IDs; first ceil(30%) development",
        "role_boundary": "Indices, original IDs, relevance and component IDs are host-audit metadata only. Sender sees query text; receiver sees static filtered corpus plus packet. Neither gets source query ID.",
        "fit_corpus": fit,
        "dev": dev,
        "test": test,
        "receiver_corpus": receiver,
        "dev_groups": sorted({debate(queries[i]["_id"]) for i in dev}),
        "test_groups": sorted({debate(queries[i]["_id"]) for i in test}),
        "query_component_ids": {str(i): assignment[i] for i in eligible},
        "dev_components": sorted(dev_components),
        "test_components": sorted(active_components - dev_components),
        "components": {
            component_ids[k]: sorted(gs) for k, gs in component_groups.items()
        },
        "excluded_missing_target": missing,
        "removed_receiver_indices": removed_receiver,
        "removed_fit_content_overlap_indices": fit_removed,
        "counts": {
            "original_corpus": len(corpus),
            "original_queries": len(queries),
            "evaluable_queries": len(eligible),
            "excluded_missing_target": len(missing),
            "original_debate_groups": len(groups),
            "independent_components": len(active_components),
            "dev": len(dev),
            "test": len(test),
            "dev_components": len(dev_components),
            "test_components": len(active_components - dev_components),
            "fit_before_filter": len(fit_all),
            "fit_corpus": len(fit),
            "fit_overlap_removed": len(fit_removed),
            "receiver_corpus": len(receiver),
            "receiver_removed": len(removed_receiver),
            "query_ids_in_original_corpus": len(all_query_ids & set(cpos)),
            "receiver_original_query_content_removed": sum(
                ch[i] in query_content for i in removed_receiver
            ),
            "retained_target_links": sum(
                len(relevance[queries[i]["_id"]]) for i in eligible
            ),
            "crosssplit_protected_content_overlap": 0,
            "fit_protected_content_overlap": 0,
            "receiver_query_content_overlap": 0,
            "queries_single_positive": sum(
                len(relevance[queries[i]["_id"]]) == 1 for i in eligible
            ),
        },
    }
