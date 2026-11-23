"""Replay the fixed 56-byte switching rule using the frozen evaluation order.

The NPZ contains integer byte lengths only for each query's shortest complete
text message, and integer SciFact cluster labels. ArguAna lengths come from
the original policy controls. BRIGHT and SciFact lengths were measured on the
serialized full-text files in per_query.jsonl order. SciFact group numbers
preserve the original evaluator's ordered dependency-group list.
"""
from pathlib import Path
import hashlib
import io
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "transfer"))
from metrics import paired_interval

INPUT_SHA256 = "e99279bac254e4a63f45e574f1671084b2c6ea67de8bbbad71462d3c75a5c553"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(z, name, rows):
    table = np.load(io.BytesIO(z.read(name)), allow_pickle=False)
    require(table.shape == (rows, 3) and np.isfinite(table).all(), f"Incomplete metrics: {name}")
    return table


def run(z):
    inputs = (HERE / "routing_inputs.npz").read_bytes()
    require(hashlib.sha256(inputs).hexdigest() == INPUT_SHA256, "Routing input hash mismatch")
    output = {}
    with np.load(io.BytesIO(inputs), allow_pickle=False) as data:
        for domain, rows in (("arguana", 989), ("bright", 103), ("scifact", 300)):
            lengths = data[domain + "_full_bytes"]
            require(lengths.shape == (rows,) and np.issubdtype(lengths.dtype, np.integer)
                    and (lengths >= 8).all(), f"{domain}: invalid packet length records")
            # Each sign packet is 8 header bytes plus 384 packed signs (48 bytes).
            use_text = lengths <= 56
            costs = np.minimum(lengths, 56)
            if domain == "arguana":
                require(not use_text.any(), "Unexpected complete-text branch in ArguAna")
                with np.load(io.BytesIO(z.read("arguana/test_metrics.npz")), allow_pickle=False) as arrays:
                    scores = arrays["raw_r384_b1"]
                require(scores.shape == (rows, 3) and np.isfinite(scores).all(),
                        "Incomplete ArguAna query metrics")
                audit = json.loads(z.read("audits/runtime_audit/policy_actual_audit/POLICY_RUNTIME_AUDIT.json"))
                require(audit["all_policy_packets_equal_primary_numeric"]
                        and audit["matched_runtime_policy_minus_always_numeric"] == [0, 0, 0],
                        "Missing matched-runtime packet verification")
                np.testing.assert_allclose(scores.mean(axis=0), audit["primary_numeric_metrics"],
                                           rtol=0, atol=1e-12)
            else:
                numeric = "numeric_primary" if domain == "bright" else "selected_numeric"
                switch = "adaptive_switch_numeric_primary" if domain == "bright" else "adaptive_switch"
                sign = metrics(z, f"{domain}/metrics_{numeric}.npy", rows)
                full = metrics(z, f"{domain}/metrics_full_query_64.npy", rows)
                recorded = metrics(z, f"{domain}/metrics_{switch}.npy", rows)
                scores = np.where(use_text[:, None], full, sign)
                np.testing.assert_array_equal(scores, recorded)
                result = json.loads(z.read(f"{domain}/results.json"))
                ref = result["summaries"][switch]
                np.testing.assert_allclose([costs.mean(), scores[:, 2].mean()],
                                           [ref["wire_bytes_mean"], ref["mean"]["ndcg10"]],
                                           rtol=0, atol=1e-12)
            row = {"queries": rows, "complete_text_routes": int(use_text.sum()),
                   "numerical_routes": int((~use_text).sum()), "complete_text_fits_64": int((lengths <= 64).sum()),
                   "mean_wire_bytes": float(costs.mean()), "ndcg10": float(scores[:, 2].mean())}
            if domain == "scifact":
                groups = data["scifact_group"]
                require(groups.shape == (rows,) and np.array_equal(np.unique(groups), np.arange(247)),
                        "Incomplete SciFact bootstrap components")
                ids = list(range(rows))
                components = [np.flatnonzero(groups == g).tolist() for g in range(247)]
                ci = paired_interval(scores[:, 2], sign[:, 2], ids, components)
                ref = result["contrasts"]["adaptive_switch_minus_selected_numeric"]
                for key in ("delta_query_mean", "ci95_cluster_percentile"):
                    np.testing.assert_allclose(ci[key], ref[key], rtol=0, atol=1e-12)
                    row[key] = ci[key]
            output[domain] = row
    return output

