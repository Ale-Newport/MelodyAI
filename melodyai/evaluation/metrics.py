"""Retrieval, rejection and calibration metrics."""

from __future__ import annotations

import numpy as np


def retrieval_metrics(ranks: list[int | None], ks=(1, 3, 5, 10)) -> dict:
    """``ranks``: 1-based rank of the correct song per query (None = not retrieved)."""
    r = np.array([x if x is not None else 10**9 for x in ranks], dtype=float)
    if not len(r):
        return {}
    out = {f"top{k}": float(np.mean(r <= k)) for k in ks}
    out["mrr"] = float(np.mean(np.where(r < 10**9, 1.0 / r, 0.0)))
    out["n"] = int(len(r))
    return out


def latency_stats(seconds: list[float]) -> dict:
    a = np.asarray(seconds, dtype=float)
    if not len(a):
        return {}
    return {"mean_ms": float(a.mean() * 1000), "p50_ms": float(np.percentile(a, 50) * 1000),
            "p95_ms": float(np.percentile(a, 95) * 1000)}


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """Probability that a positive score exceeds a negative one (Mann-Whitney U)."""
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv)) + 1.0
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def threshold_for_far(neg_scores: np.ndarray, far: float) -> float:
    """Smallest threshold whose false-accept rate on ``neg_scores`` is <= ``far``."""
    if not len(neg_scores):
        return 0.0
    s = np.sort(neg_scores)
    k = int(np.ceil((1 - far) * len(s))) - 1
    return float(s[min(max(k, 0), len(s) - 1)] + 1e-6)


def rejection_metrics(correct_scores: np.ndarray, wrong_scores: np.ndarray, ooc_scores: np.ndarray,
                      threshold: float) -> dict:
    """Behaviour of an accept threshold.

    correct_scores: top-1 score of in-catalog queries whose top-1 is correct
    wrong_scores:   top-1 score of in-catalog queries whose top-1 is wrong
    ooc_scores:     top-1 score of out-of-catalog / non-melodic queries
    """
    n_in = len(correct_scores) + len(wrong_scores)
    return {
        "threshold": round(float(threshold), 4),
        "accepted_correct_rate": float(np.mean(correct_scores >= threshold)) if len(correct_scores) else None,
        "accepted_wrong_rate_in_catalog": float(np.mean(wrong_scores >= threshold)) if len(wrong_scores) else 0.0,
        "false_accept_rate_ooc": float(np.mean(ooc_scores >= threshold)) if len(ooc_scores) else None,
        "precision_of_accepts": float(
            np.sum(correct_scores >= threshold) / max(1, np.sum(correct_scores >= threshold)
                                                      + np.sum(wrong_scores >= threshold)
                                                      + np.sum(ooc_scores >= threshold))),
        "coverage_in_catalog": float(
            (np.sum(correct_scores >= threshold) + np.sum(wrong_scores >= threshold)) / max(1, n_in)),
        "auc_correct_vs_ooc": roc_auc(correct_scores, ooc_scores),
    }


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """Expected calibration error with equal-width bins + reliability diagram data."""
    bins = np.linspace(0, 1, n_bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if m.any():
            conf, acc = float(probs[m].mean()), float(labels[m].mean())
            total += m.mean() * abs(conf - acc)
            rows.append({"bin": [round(lo, 2), round(hi, 2)], "n": int(m.sum()), "confidence": conf, "accuracy": acc})
    brier = float(np.mean((probs - labels) ** 2)) if len(probs) else None
    return {"ece": float(total), "brier": brier, "reliability": rows, "n": int(len(probs))}
