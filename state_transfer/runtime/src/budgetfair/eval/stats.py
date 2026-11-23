"""Prespecified paired inference for matched experimental instances.

Binary primary comparisons use exact McNemar and paired percentile bootstrap
effect intervals. No Shapiro-driven test selection or unsupported ANOVA claims.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import beta, binomtest, norm, ttest_rel, wilcoxon

def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int,float,np.integer,np.floating)) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    return float(value)

def _pairs(a,b):
    x,y=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if x.ndim!=1 or y.ndim!=1 or len(x)!=len(y) or not len(x):
        raise ValueError("paired samples must be nonempty one-dimensional equal-length sequences")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("paired samples must be finite")
    return x,y

def _trials(successes,n):
    if any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) for v in (successes,n)) or n<=0 or not 0<=successes<=n:
        raise ValueError("n must be positive; successes must be an integer between zero and n")

def wilson_interval(successes,n,confidence=0.95):
    _trials(successes,n); confidence=_probability(confidence,"confidence")
    z=float(norm.ppf((1+confidence)/2)); p=successes/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return max(0.,center-half),min(1.,center+half)

def clopper_pearson_interval(successes,n,confidence=0.95):
    _trials(successes,n); alpha=1-_probability(confidence,"confidence")
    low=0. if successes==0 else float(beta.ppf(alpha/2,successes,n-successes+1))
    high=1. if successes==n else float(beta.ppf(1-alpha/2,successes+1,n-successes))
    return low,high

def _bootstrap_difference(difference,confidence,n_bootstrap,seed):
    if isinstance(n_bootstrap,bool) or not isinstance(n_bootstrap,int) or n_bootstrap<100:
        raise ValueError("n_bootstrap must be an integer >= 100")
    if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    rng=np.random.default_rng(seed)
    unique,counts=np.unique(difference,return_counts=True)
    if len(unique)<=3:
        draws=rng.multinomial(len(difference),counts/len(difference),size=n_bootstrap)
        means=draws@unique/len(difference)
    else:
        means=np.empty(n_bootstrap)
        chunk=max(1, min(1000, 500000//len(difference)))
        for start in range(0,n_bootstrap,chunk):
            count=min(chunk,n_bootstrap-start)
            means[start:start+count]=difference[rng.integers(0,len(difference),size=(count,len(difference)))].mean(axis=1)
    alpha=(1-confidence)/2
    return [float(v) for v in np.quantile(means,[alpha,1-alpha])]

def paired_binary_test(success_a,success_b,*,episode_ids=None,episode_ids_b=None,
                       confidence=0.95,n_bootstrap=10000,seed=42):
    a,b=_pairs(success_a,success_b)
    if not np.all(np.isin(a,[0.,1.])) or not np.all(np.isin(b,[0.,1.])):
        raise ValueError("binary outcomes must be bool or 0/1")
    confidence=_probability(confidence,"confidence")
    if episode_ids is not None:
        ids=list(episode_ids)
        if len(ids)!=len(a) or len(set(ids))!=len(ids) or any(not isinstance(i,str) or not i for i in ids):
            raise ValueError("episode IDs must be nonempty, unique, and align with outcomes")
        if episode_ids_b is not None and ids!=list(episode_ids_b):
            raise ValueError("paired episode identities/order differ")
    elif episode_ids_b is not None:
        raise ValueError("episode_ids_b requires episode_ids")
    a_only=int(np.sum((a==1)&(b==0))); b_only=int(np.sum((a==0)&(b==1)))
    discordant=a_only+b_only
    p=1. if discordant==0 else float(binomtest(a_only,discordant,p=0.5,alternative="two-sided").pvalue)
    ci=_bootstrap_difference(a-b,confidence,n_bootstrap,seed)
    successes_a,successes_b=int(a.sum()),int(b.sum())
    return {"test_used":"mcnemar_exact","n":len(a),"p":p,
        "discordant_a_only":a_only,"discordant_b_only":b_only,"discordant_total":discordant,
        "successes_a":successes_a,"successes_b":successes_b,
        "rate_a":float(a.mean()),"rate_b":float(b.mean()),
        "difference":float((a-b).mean()),"mean_difference":float((a-b).mean()),
        "difference_ci":ci,"ci_low":ci[0],"ci_high":ci[1],"confidence":confidence,
        "ci_method":"paired_percentile_bootstrap","bootstrap_seed":int(seed),"bootstrap_resamples":n_bootstrap,
        "wilson_a":list(wilson_interval(successes_a,len(a),confidence)),
        "wilson_b":list(wilson_interval(successes_b,len(a),confidence)),
        "exact_binomial_a":list(clopper_pearson_interval(successes_a,len(a),confidence)),
        "exact_binomial_b":list(clopper_pearson_interval(successes_b,len(a),confidence)),
        "pairing":"verified IDs" if episode_ids is not None else "positional; caller must ensure matched instances",
        "boundary_ci_note":"a degenerate paired bootstrap interval is not an equivalence claim"}

def paired_test(text_scores,vector_scores,*,method="auto",confidence=0.95,n_bootstrap=10000,seed=42):
    a,b=_pairs(text_scores,vector_scores)
    binary=np.all(np.isin(a,[0.,1.])) and np.all(np.isin(b,[0.,1.]))
    if method=="mcnemar_exact" or (method=="auto" and binary):
        return paired_binary_test(a,b,confidence=confidence,n_bootstrap=n_bootstrap,seed=seed)
    if method=="auto":
        raise ValueError("continuous paired outcomes require a prespecified method: paired_t or wilcoxon")
    if method not in ("paired_t","wilcoxon") or len(a)<2:
        raise ValueError("unsupported paired method or fewer than two observations")
    confidence=_probability(confidence,"confidence")
    difference=a-b
    if not np.all(np.isfinite(difference)):
        raise ValueError("paired differences are nonfinite")
    deviation=float(np.std(difference,ddof=1)); effect=float(difference.mean())
    if np.all(difference==0):
        statistic,p=0.,1.
    elif method=="paired_t":
        result=ttest_rel(a,b)
        statistic=float(result.statistic) if np.isfinite(result.statistic) else None
        p=float(result.pvalue)
    else:
        result=wilcoxon(difference,alternative="two-sided",zero_method="wilcox",method="auto")
        statistic,p=float(result.statistic),float(result.pvalue)
    return {"test_used":method,"n":len(a),"t":statistic,"p":p,"df":len(a)-1 if method=="paired_t" else None,
            "cohens_d":effect/deviation if deviation>0 else (0. if effect==0 else None),
            "mean_difference":effect,"difference_ci":_bootstrap_difference(difference,confidence,n_bootstrap,seed),
            "normality_ok":None,"assumption_note":"method prespecified; normality was not used to select a test"}

def two_way_anova(df):
    raise ValueError("two_way_anova is outside the implemented analysis protocol; use prespecified paired comparisons. A repeated-measures design and its assumptions must be specified before adding ANOVA.")

def holm_bonferroni(p_values,alpha=0.05):
    alpha=_probability(alpha,"alpha")
    values=np.asarray(p_values,dtype=float)
    if values.ndim!=1 or not len(values) or not np.all(np.isfinite(values)) or np.any((values<0)|(values>1)):
        raise ValueError("p_values must be a nonempty finite sequence in [0,1]")
    order=np.argsort(values,kind="stable"); reject=[False]*len(values)
    for rank,index in enumerate(order):
        if values[index]<=alpha/(len(values)-rank):
            reject[int(index)]=True
        else:
            break
    return reject
