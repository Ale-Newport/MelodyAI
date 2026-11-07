"""Overlapping melody windows for segment-level matching.

Every reference is cut into windows of several lengths (in notes) with a small hop,
so that a query covering *any* passage (e.g. a chorus in the middle of the song) has
a window that spans approximately the same musical content. Each window stores a
time-normalised, key-normalised contour vector of ``n_points`` samples (uniform
scaling, as in "linear scaling" QBH systems) which DTW then compares with a band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from melodyai.features.melody import NoteSequence, center, notes_to_contour, resample_contour

DEFAULT_LENGTHS = (6, 9, 13, 18, 25)
N_POINTS = 96


def hop_for(length: int) -> int:
    return max(1, length // 5)


@dataclass
class Windows:
    start_note: np.ndarray
    end_note: np.ndarray     # exclusive
    t0: np.ndarray           # seconds in the reference
    t1: np.ndarray
    vectors: np.ndarray      # (M, n_points) float32, median-centred
    offsets: np.ndarray      # (M,) median pitch subtracted from each window (absolute key)

    def __len__(self) -> int:
        return len(self.start_note)


def build_windows(notes: NoteSequence, lengths=DEFAULT_LENGTHS, n_points: int = N_POINTS) -> Windows:
    starts, ends, t0s, t1s, vecs, offs = [], [], [], [], [], []
    n = len(notes)
    ioi = notes.ioi()
    for L in lengths:
        if n < 3:
            break
        L_eff = min(L, n)
        for s in range(0, max(1, n - L_eff + 1), hop_for(L_eff)):
            e = s + L_eff
            sub = notes.slice(s, e)
            c = notes_to_contour(sub)
            med = float(np.median(c))
            starts.append(s)
            ends.append(e)
            t0s.append(notes.onsets[s])
            t1s.append(notes.onsets[e - 1] + ioi[e - 1])
            vecs.append(resample_contour(c - med, n_points))
            offs.append(med)
        if L >= n:
            break
    if not starts:
        z = np.zeros(0)
        return Windows(z.astype(int), z.astype(int), z, z, np.zeros((0, n_points), np.float32), z)
    return Windows(
        np.asarray(starts), np.asarray(ends), np.asarray(t0s), np.asarray(t1s),
        np.asarray(vecs, dtype=np.float32), np.asarray(offs),
    )


def query_vector(contour: np.ndarray, n_points: int = N_POINTS) -> np.ndarray:
    return resample_contour(center(contour), n_points).astype(np.float64)
