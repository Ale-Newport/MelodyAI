"""Alignment explanations for visualisation (Melody Explorer / results view).

Given a query and a matched reference passage, recompute a subsequence DTW alignment
on the reference region around the match and return the *actual* data used: the two
contours (key-aligned), the accumulated cost matrix and the warping path.
"""

from __future__ import annotations

import numpy as np

from melodyai.config import CONTOUR_RATE
from melodyai.features.extract import QueryFeatures, ReferenceFeatures
from melodyai.features.melody import midi_to_name
from melodyai.matching.algorithms import SubsequenceDTW


def _downsample_matrix(M: np.ndarray, max_rows: int, max_cols: int) -> np.ndarray:
    r = max(1, int(np.ceil(M.shape[0] / max_rows)))
    c = max(1, int(np.ceil(M.shape[1] / max_cols)))
    return M[::r, ::c]


def explain_alignment(query: QueryFeatures, ref: ReferenceFeatures, t0: float | None, t1: float | None,
                      margin_s: float = 3.0) -> dict | None:
    if len(query.contour) < 4 or len(ref.contour) < 4:
        return None
    times = ref.contour_times
    if t0 is None or t1 is None:
        lo, hi = 0, len(ref.contour)
    else:
        lo = int(np.searchsorted(times, t0 - margin_s))
        hi = int(np.searchsorted(times, t1 + margin_s))
    seg = ref.contour[lo:hi]
    if len(seg) < 4:
        seg, lo = ref.contour, 0
    sdtw = SubsequenceDTW()
    q_ds = sdtw._down(query.contour)  # same resolution as the reference inside match_song
    m = sdtw.match_song(q_ds, seg, keep_cost=True)
    if m is None:
        return None
    res, scale, shift = m
    step = CONTOUR_RATE / sdtw.rate
    q_scaled_len = max(4, int(round(len(q_ds) * scale)))
    from melodyai.features.melody import resample_contour

    q_al = resample_contour(q_ds - np.median(q_ds), q_scaled_len) + shift
    r_ds = sdtw._down(seg)
    seg_t0 = float(times[lo]) if len(times) > lo else 0.0
    cost = res.cost
    cost_norm = None
    if cost is not None:
        finite = np.where(np.isfinite(cost), cost, np.nan)
        cost_norm = _downsample_matrix(finite, 120, 240)
        mx = np.nanmax(cost_norm) if np.isfinite(cost_norm).any() else 1.0
        cost_norm = np.round(np.nan_to_num(cost_norm / (mx + 1e-9), nan=1.0), 3).tolist()
    path = res.path
    return {
        "rate_hz": sdtw.rate,
        "query_contour": np.round(q_al, 3).tolist(),
        "reference_contour": np.round(r_ds, 3).tolist(),
        "reference_t0": round(seg_t0, 3),
        "reference_match": [round(seg_t0 + res.start / sdtw.rate, 3), round(seg_t0 + (res.end + 1) / sdtw.rate, 3)],
        "path": path[:: max(1, len(path) // 400)].tolist(),
        "distance": round(res.distance, 4),
        "tempo_scale": scale,
        # Semitones to add to the query to reach the reference key (folded into one octave).
        "query_transposition_semitones": round(float((shift - np.median(q_ds) + 6) % 12 - 6), 2),
        "reference_median_note": midi_to_name(float(np.median(r_ds))),
        "accumulated_cost": cost_norm,
        "segment_frames_per_point": step,
    }
