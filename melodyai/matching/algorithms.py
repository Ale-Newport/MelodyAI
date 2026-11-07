"""Melody matching algorithms (classical).

All algorithms share one interface: ``search(query, index, top_k) -> list[Candidate]``
ordered by increasing distance. Scores are *relative similarities* in [0, 1]
(``exp(-distance / tau)``), NOT probabilities.

==============  ====================================================================
``dtw``         Window DTW: key-normalised, time-normalised contour vs. every
                overlapping reference window, banded DTW with key refinement.
``subseq_dtw``  Subsequence DTW of the query against whole-song contours with
                multi-scale tempo search and locally-centred references.
``chroma_dtw``  CQT chroma of query and reference audio, 12 circular key shifts,
                cosine-cost subsequence DTW (baseline from the chroma QBH literature).
``note_align``  Smith-Waterman alignment of note interval + rhythm-ratio sequences
                (tolerates missing/extra notes; tempo invariant by construction).
``ls``          Linear scaling only: Euclidean distance between window vectors
                (fast candidate generator, no warping).
==============  ====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from melodyai.config import CONTOUR_RATE
from melodyai.features.extract import CHROMA_RATE, QueryFeatures
from melodyai.features.melody import NoteSequence, resample_contour
from melodyai.features.windows import query_vector
from melodyai.indexing.melody_index import MelodyIndex
from melodyai.matching.dtw import (
    abs_cost,
    banded_dtw_batch,
    cosine_cost,
    local_alignment,
    subsequence_dtw,
    sw_start,
)


@dataclass
class Candidate:
    song_id: str
    distance: float
    score: float
    start_time: float | None = None
    end_time: float | None = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "song_id": self.song_id,
            "distance": round(float(self.distance), 5),
            "score": round(float(self.score), 5),
            "start_time": None if self.start_time is None else round(float(self.start_time), 3),
            "end_time": None if self.end_time is None else round(float(self.end_time), 3),
            "details": self.details,
        }


def _rank(cands: list[Candidate], top_k: int) -> list[Candidate]:
    cands.sort(key=lambda c: c.distance)
    return cands[:top_k]


class WindowDTW:
    name = "dtw"
    label = "Pitch contour + window DTW"
    tau = 1.0

    def __init__(self, band_frac: float = 0.15, clip: float = 4.0, penalty: float = 0.15,
                 min_ratio: float = 0.4, max_ratio: float = 2.6, prefilter: int = 30000) -> None:
        self.band_frac, self.clip, self.penalty = band_frac, clip, penalty
        self.min_ratio, self.max_ratio, self.prefilter = min_ratio, max_ratio, prefilter

    def window_distances(self, query: QueryFeatures, index: MelodyIndex):
        M = index.matrix()
        if not len(M) or len(query.contour) < 4:
            return M, np.zeros(0, dtype=int), np.zeros(0)
        qv = query_vector(query.contour, index.n_points)
        ratio = (M.t1 - M.t0) / max(query.voiced_duration, 1e-3)
        sel = np.flatnonzero((ratio >= self.min_ratio) & (ratio <= self.max_ratio))
        if not len(sel):
            sel = np.arange(len(M))
        W = M.vectors[sel].astype(np.float64)
        if len(sel) > self.prefilter:  # linear-scaling prefilter for large catalogs
            d_ls = np.mean(np.abs(W - qv[None, :]), axis=1)
            keep = np.argsort(d_ls)[: self.prefilter]
            sel, W = sel[keep], W[keep]
        band = max(2, int(self.band_frac * index.n_points))
        d0 = banded_dtw_batch(qv, W, band, self.clip, self.penalty, np.zeros(1))
        # Key refinement: residual offset after median centring (handles melodies whose
        # median note differs slightly between query and window boundaries).
        shifts = np.median(qv[None, :] - W, axis=1)
        d1 = banded_dtw_batch(qv, W, band, self.clip, self.penalty, shifts)
        return M, sel, np.minimum(d0, d1)

    def search(self, query: QueryFeatures, index: MelodyIndex, top_k: int = 10) -> list[Candidate]:
        M, sel, d = self.window_distances(query, index)
        if not len(sel):
            return []
        best: dict[int, int] = {}
        for k in np.argsort(d):
            s = int(M.song_idx[sel[k]])
            if s not in best:
                best[s] = k
            if len(best) >= top_k:
                break
        out = []
        for s, k in best.items():
            w = sel[k]
            out.append(Candidate(
                M.song_ids[s], float(d[k]), float(np.exp(-d[k] / self.tau)), float(M.t0[w]), float(M.t1[w]),
                {"window_notes": [int(M.start_note[w]), int(M.end_note[w])],
                 "tempo_ratio": round(float((M.t1[w] - M.t0[w]) / max(query.voiced_duration, 1e-3)), 3)},
            ))
        return _rank(out, top_k)


class LinearScaling(WindowDTW):
    name = "ls"
    label = "Linear scaling (no DTW)"
    tau = 1.0

    def window_distances(self, query, index):
        M = index.matrix()
        if not len(M) or len(query.contour) < 4:
            return M, np.zeros(0, dtype=int), np.zeros(0)
        qv = query_vector(query.contour, index.n_points)
        ratio = (M.t1 - M.t0) / max(query.voiced_duration, 1e-3)
        sel = np.flatnonzero((ratio >= self.min_ratio) & (ratio <= self.max_ratio))
        W = M.vectors[sel].astype(np.float64)
        diff = qv[None, :] - W
        diff -= np.median(diff, axis=1, keepdims=True)
        return M, sel, np.mean(np.minimum(np.abs(diff), self.clip), axis=1)


SCALES = (0.55, 0.75, 1.0, 1.35, 1.8)


class SubsequenceDTW:
    name = "subseq_dtw"
    label = "Pitch contour + subsequence DTW (whole song)"
    tau = 1.0

    def __init__(self, rate: float = 10.0, scales=SCALES, clip: float = 4.0, penalty: float = 0.2) -> None:
        self.rate, self.scales, self.clip, self.penalty = rate, scales, clip, penalty

    def _down(self, c: np.ndarray) -> np.ndarray:
        step = int(round(CONTOUR_RATE / self.rate))
        n = len(c) // step
        return np.median(c[: n * step].reshape(n, step), axis=1) if n else c

    def match_song(self, q: np.ndarray, ref_contour: np.ndarray, shifts: np.ndarray | None = None,
                   keep_cost: bool = False):
        """Best subsequence alignment of contour ``q`` inside ``ref_contour``.

        Key: the query is median-centred and compared against the reference shifted by
        each candidate local key (``shifts``: median pitches of the reference windows,
        rounded to semitones). Tempo: the query is linearly rescaled by each factor in
        ``scales``; DTW absorbs the remaining local timing deviations.
        """
        r = self._down(ref_contour)
        if shifts is None:
            shifts = np.unique(np.round(r))
        best = None
        qc = q - np.median(q)
        for f in self.scales:
            L = max(4, int(round(len(qc) * f)))
            if L > len(r) * 1.3:
                continue
            qs = resample_contour(qc, L)
            for s in shifts:
                res = subsequence_dtw(abs_cost(qs + s, r, self.clip), self.penalty)
                if best is None or res.distance < best[0].distance:
                    best = (res, f, qs, s)
        if best is None:
            return None
        res, f, qs, s = best
        # Fine key refinement along the optimal path, then re-align once.
        s2 = s + float(np.median(r[res.path[:, 1]] - (qs[res.path[:, 0]] + s)))
        res2 = subsequence_dtw(abs_cost(qs + s2, r, self.clip), self.penalty, keep_cost=keep_cost)
        if res2.distance <= res.distance:
            return res2, f, s2
        if keep_cost:
            res = subsequence_dtw(abs_cost(qs + s, r, self.clip), self.penalty, keep_cost=True)
        return res, f, s

    def search(self, query: QueryFeatures, index: MelodyIndex, top_k: int = 10) -> list[Candidate]:
        if len(query.contour) < 4:
            return []
        q = self._down(query.contour)
        out = []
        for sid, ref in list(index.refs.items()):
            if len(ref.contour) < 4:
                continue
            try:
                offs = index.windows_of(sid).offsets
            except KeyError:
                offs = np.zeros(0)
            shifts = np.unique(np.round(offs)) if len(offs) else None
            m = self.match_song(q, ref.contour, shifts)
            if m is None:
                continue
            res, f, _ = m
            step = 1.0 / self.rate
            t0 = float(ref.contour_times[0]) if len(ref.contour_times) else 0.0
            out.append(Candidate(sid, res.distance, float(np.exp(-res.distance / self.tau)),
                                 t0 + res.start * step, t0 + (res.end + 1) * step, {"tempo_scale": f}))
        return _rank(out, top_k)


class ChromaDTW:
    name = "chroma_dtw"
    label = "CQT chroma + subsequence DTW"
    tau = 0.25

    def __init__(self, scales=(0.7, 1.0, 1.4), penalty: float = 0.05) -> None:
        self.scales, self.penalty = scales, penalty

    def search(self, query: QueryFeatures, index: MelodyIndex, top_k: int = 10) -> list[Candidate]:
        Q = query.chroma
        if Q is None or len(Q) < 5:
            return []
        Q = Q[np.linalg.norm(Q, axis=1) > 1e-6]
        if len(Q) < 5:
            return []
        out = []
        for sid, ref in list(index.refs.items()):
            R = ref.chroma
            if R is None or len(R) < 5:
                continue
            best = None
            for f in self.scales:
                L = max(4, int(round(len(Q) * f)))
                if L > len(R) * 1.3:
                    continue
                Qs = np.stack([resample_contour(Q[:, k], L) for k in range(12)], axis=1)
                for k in range(12):
                    res = subsequence_dtw(cosine_cost(np.roll(Qs, k, axis=1), R), self.penalty)
                    if best is None or res.distance < best[0].distance:
                        best = (res, f, k)
            if best is None:
                continue
            res, f, k = best
            out.append(Candidate(sid, res.distance, float(np.exp(-res.distance / self.tau)),
                                 res.start / CHROMA_RATE, (res.end + 1) / CHROMA_RATE,
                                 {"tempo_scale": f, "chroma_shift": k}))
        return _rank(out, top_k)


def merge_repeats(notes: NoteSequence, tol: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Collapse consecutive notes of (almost) equal pitch: segmentation of repeated notes
    in hummed queries is unreliable, so both sides are compared without them."""
    if not len(notes):
        return np.zeros(0), np.zeros(0)
    p, ioi = [notes.pitches[0]], [notes.ioi()[0]]
    for pitch, d in zip(notes.pitches[1:], notes.ioi()[1:]):
        if abs(pitch - p[-1]) < tol:
            ioi[-1] += d
        else:
            p.append(pitch)
            ioi.append(d)
    return np.asarray(p), np.maximum(np.asarray(ioi), 1e-3)


class NoteAlignment:
    name = "note_align"
    label = "Note intervals + Smith-Waterman"
    tau = 0.35

    def __init__(self, gap: float = 0.7, pitch_tol: float = 1.5, rhythm_weight: float = 0.2) -> None:
        self.gap, self.pitch_tol, self.rhythm_weight = gap, pitch_tol, rhythm_weight

    def _features(self, notes: NoteSequence):
        p, ioi = merge_repeats(notes)
        iv = np.diff(p)
        rr = np.diff(np.log(ioi))  # tempo-invariant duration ratio between consecutive notes
        return iv, rr

    def similarity(self, qf, rf):
        qi, qr = qf
        ri, rr = rf
        S = 1.0 - np.abs(qi[:, None] - ri[None, :]) / self.pitch_tol
        R = 1.0 - np.abs(qr[:, None] - rr[None, :]) / 1.0
        S = (1 - self.rhythm_weight) * np.clip(S, -1, 1) + self.rhythm_weight * np.clip(R, -1, 1)
        return S

    def search(self, query: QueryFeatures, index: MelodyIndex, top_k: int = 10) -> list[Candidate]:
        qf = self._features(query.notes)
        if len(qf[0]) < 3:
            return []
        out = []
        for sid, ref in list(index.refs.items()):
            rf = self._features(ref.notes)
            if len(rf[0]) < 2:
                continue
            S = self.similarity(qf, rf)
            best, bi, bj, H = local_alignment(S, self.gap)
            sim = best / len(qf[0])
            dist = 1.0 - min(1.0, sim)
            _, j0 = sw_start(H, S, bi, bj, self.gap)
            # Map merged-note indices back to times in the reference.
            p, ioi = merge_repeats(ref.notes)
            starts = ref.notes.onsets[0] + np.concatenate([[0], np.cumsum(ioi)])
            out.append(Candidate(sid, dist, float(np.exp(-dist / self.tau)),
                                 float(starts[min(j0, len(starts) - 1)]), float(starts[min(bj + 1, len(starts) - 1)]),
                                 {"aligned_score": round(best, 3)}))
        return _rank(out, top_k)


ALGORITHMS = {a.name: a for a in (WindowDTW(), SubsequenceDTW(), ChromaDTW(), NoteAlignment(), LinearScaling())}
