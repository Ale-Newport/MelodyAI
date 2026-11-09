"""Fusion of per-engine rankings and abstention.

Engines produce song candidates with *relative* scores in [0, 1] (not probabilities).
Fusion strategies:

- ``weighted``  sum_e  w_e * reliability_e * score_e(song)
- ``rrf``       Reciprocal Rank Fusion (Cormack et al., 2009): sum_e 1 / (k + rank_e)
- ``combmnz``   (sum of scores) * (number of engines that returned the song)
- ``learned``   logistic regression over per-song features, trained by
                ``melodyai.evaluation`` on labelled synthetic queries (optional)

Abstention decides between ``match`` / ``uncertain`` / ``no_match`` from the fused top
score and the margin to the runner-up, using thresholds stored in
``configs/calibration.json`` (produced by ``make evaluate``), or conservative defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from melodyai.config import ROOT_DIR

CALIBRATION_PATH = ROOT_DIR / "configs" / "calibration.json"


@dataclass
class EngineOutput:
    engine: str                       # "melody" | "lyrics"
    candidates: list[dict]            # each: {"song_id", "score", ...}
    reliability: float = 1.0          # input quality heuristic in [0, 1]
    meta: dict = field(default_factory=dict)

    def scores(self) -> dict[str, float]:
        return {c["song_id"]: float(c["score"]) for c in self.candidates}

    def ranks(self) -> dict[str, int]:
        return {c["song_id"]: i + 1 for i, c in enumerate(self.candidates)}


DEFAULT_WEIGHTS = {"melody": 1.0, "lyrics": 1.0}


def song_features(outputs: list[EngineOutput], song_id: str) -> list[float]:
    """Feature vector used by the learned fusion model (order matters)."""
    feats = []
    for name in ("melody", "lyrics"):
        o = next((x for x in outputs if x.engine == name), None)
        if o is None:
            feats += [0.0, 0.0, 0.0]
            continue
        s, r = o.scores(), o.ranks()
        feats += [s.get(song_id, 0.0), 1.0 / r[song_id] if song_id in r else 0.0, o.reliability]
    feats.append(float(sum(1 for o in outputs if song_id in o.scores()) > 1))
    return feats


def fuse(outputs: list[EngineOutput], method: str = "weighted", weights: dict | None = None,
         k: int = 60, model=None) -> list[dict]:
    outputs = [o for o in outputs if o.candidates]
    if not outputs:
        return []
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    songs = sorted({c["song_id"] for o in outputs for c in o.candidates})
    fused = []
    wsum = sum(weights.get(o.engine, 1.0) * max(o.reliability, 1e-3) for o in outputs)
    for sid in songs:
        per = {o.engine: o.scores().get(sid) for o in outputs}
        if method == "weighted":
            val = sum(weights.get(o.engine, 1.0) * max(o.reliability, 1e-3) * o.scores().get(sid, 0.0)
                      for o in outputs) / wsum
        elif method == "rrf":
            val = sum(1.0 / (k + o.ranks()[sid]) for o in outputs if sid in o.ranks())
        elif method == "combmnz":
            hits = [o.scores()[sid] for o in outputs if sid in o.scores()]
            val = sum(hits) * len(hits) / len(outputs)
        elif method == "learned":
            if model is None:
                raise ValueError("learned fusion requires a trained model")
            val = float(model.predict_proba(np.array([song_features(outputs, sid)]))[0, 1])
        else:
            raise ValueError(f"Unknown fusion method {method}")
        fused.append({"song_id": sid, "score": float(val), "engine_scores": per,
                      "agreement": sum(1 for v in per.values() if v is not None)})
    fused.sort(key=lambda d: -d["score"])
    # Map RRF/CombMNZ onto [0, 1] as a fraction of their maximum attainable value
    # (rank 1 in every engine / score 1 in every engine), keeping the order unchanged.
    if method in ("rrf", "combmnz") and fused:
        max_val = len(outputs) / (k + 1) if method == "rrf" else float(len(outputs))
        for f in fused:
            f["raw_fusion_value"] = f["score"]
            f["score"] = f["score"] / max_val
    return fused


DEFAULT_THRESHOLDS = {
    # key -> (min_score_for_match, min_margin_for_match, min_score_uncertain)
    "melody": {"accept": 0.55, "margin": 0.04, "uncertain": 0.40},
    "lyrics": {"accept": 0.60, "margin": 0.10, "uncertain": 0.40},
    "fused": {"accept": 0.50, "margin": 0.05, "uncertain": 0.35},
    # score = verified peak-overlap fraction (true clips ~0.6-0.95, unknown recordings <~0.46)
    "fingerprint": {"accept": 0.50, "margin": 0.10, "uncertain": 0.42},
    # open-world lyrics search (melodyai.world.search): alignment score + popularity tie-break
    "world": {"accept": 0.62, "margin": 0.08, "uncertain": 0.45},
    # hummed melodies of real songs (melodyai.world.melodies): two-stage relative score + popularity tie-break.
    # Real songs' melodies are hard to tell apart: a confident match needs a large lead (446 synthetic hums of a
    # 25k-song index: 25 confident matches, 1 wrong); everything else is shown as "possible songs".
    "world_melody": {"accept": 0.5, "margin": 0.3, "uncertain": 0.45},
}


def load_calibration(path: Path = CALIBRATION_PATH) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def decide(ranked: list[dict], kind: str, algorithm: str | None = None, calibration: dict | None = None) -> dict:
    """Abstention decision for a ranked list of {"song_id", "score"} dicts."""
    cal = calibration if calibration is not None else load_calibration()
    th = dict(DEFAULT_THRESHOLDS.get(kind, DEFAULT_THRESHOLDS["fused"]))
    source = "default"
    key = f"{kind}:{algorithm}" if algorithm else kind
    if key in cal.get("thresholds", {}):
        th.update(cal["thresholds"][key])
        source = "calibrated"
    if not ranked:
        return {"status": "no_match", "reason": "no candidates", "thresholds": th, "threshold_source": source}
    top = ranked[0]["score"]
    second = ranked[1]["score"] if len(ranked) > 1 else 0.0
    margin = top - second
    if top >= th["accept"] and margin >= th["margin"]:
        status, conf = "match", "high" if (top >= th["accept"] + 0.1 and margin >= 2 * th["margin"]) else "medium"
    elif top >= th["uncertain"]:
        status, conf = "uncertain", "low"
    else:
        status, conf = "no_match", None
    out = {"status": status, "confidence": conf, "top_score": round(top, 4), "margin": round(margin, 4),
           "thresholds": th, "threshold_source": source}
    prob = calibrated_probability(cal, key, top, margin)
    if prob is not None:
        out["calibrated_probability"] = prob
    return out


def calibrated_probability(cal: dict, key: str, top: float, margin: float) -> float | None:
    """Platt-scaled P(top-1 correct) if a calibration model for ``key`` exists."""
    m = cal.get("platt", {}).get(key)
    if not m:
        return None
    z = m["coef"][0] * top + m["coef"][1] * margin + m["intercept"]
    return round(float(1.0 / (1.0 + np.exp(-z))), 4)
