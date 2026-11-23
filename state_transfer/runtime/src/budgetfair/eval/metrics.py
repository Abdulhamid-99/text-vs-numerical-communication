"""Validated per-cell summaries, separate from inferential paired comparisons."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .stats import wilson_interval, _bootstrap_difference

@dataclass
class CellSummary:
    model: str
    task: str
    channel: str
    budget: str
    network: str
    n_episodes: int
    success_rate: float
    mean_score: float
    mean_bits: float
    mean_latency_ms: float
    score_ci_low: float | None
    score_ci_high: float | None
    success_ci_low: float
    success_ci_high: float
    score_ci_method: str

def _get(row,key,default=None):
    return row.get(key,default) if isinstance(row,dict) else getattr(row,key,default)

def _numeric(value,name):
    if isinstance(value,bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        number=float(value)
    except (ValueError,TypeError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number

def _success(value):
    if value in (True,1,"True","true","1"): return 1
    if value in (False,0,"False","false","0"): return 0
    raise ValueError("success must be an explicit binary outcome")

def aggregate_cell(results, *, metadata=None, confidence=0.95, n_bootstrap=10000, seed=42):
    if not isinstance(results,(list,tuple)) or not results:
        raise ValueError("cannot summarize an empty cell")
    meta=dict(metadata or {})
    aliases={"model":"model","task":"task","channel":"channel","budget":"budget_bits","network":"network_profile"}
    context={}
    for key,alias in aliases.items():
        labels={str(_get(row,key,_get(row,alias))) for row in results if _get(row,key,_get(row,alias)) is not None}
        if len(labels)>1:
            raise ValueError(f"mixed {key} cells require separate summaries")
        declared=meta.get(key,meta.get(alias))
        if declared is not None and labels and str(declared) not in labels:
            raise ValueError(f"declared {key} conflicts with result rows")
        context[key]=str(declared) if declared is not None else next(iter(labels),"unspecified")
    identifiers=[_get(row,"episode_id") for row in results]
    if any(not isinstance(i,str) or not i for i in identifiers) or len(set(identifiers))!=len(identifiers):
        raise ValueError("cell episode identities must be nonempty and unique")
    scores=np.array([_numeric(_get(row,"score"),"score") for row in results])
    if np.any((scores<0)|(scores>1)):
        raise ValueError("task scores must lie in [0,1]")
    bits=np.array([_numeric(_get(row,"bits_used"),"bits_used") for row in results])
    latency=np.array([_numeric(_get(row,"wall_time_ms",_get(row,"latency_ms")),"wall_time_ms") for row in results])
    if np.any(bits<0) or np.any(bits!=np.floor(bits)) or np.any(latency<0):
        raise ValueError("bits must be nonnegative integers and latency nonnegative")
    successes=sum(_success(_get(row,"success")) for row in results)
    success_ci=wilson_interval(successes,len(results),confidence)
    if np.all(np.isin(scores,[0.,1.])):
        ci=wilson_interval(int(scores.sum()),len(scores),confidence); method="Wilson binomial"
    elif len(scores)>1:
        ci=_bootstrap_difference(scores,confidence,n_bootstrap,seed); method="percentile bootstrap mean score"
    else:
        ci=(None,None); method="undefined for one continuous observation"
    return CellSummary(**context,n_episodes=len(results),success_rate=successes/len(results),
        mean_score=float(scores.mean()),mean_bits=float(bits.mean()),mean_latency_ms=float(latency.mean()),
        score_ci_low=ci[0],score_ci_high=ci[1],success_ci_low=success_ci[0],success_ci_high=success_ci[1],
        score_ci_method=method)
