"""Pitch-invariant constellation hashes (anchor + 3 targets, "quads").

For every anchor peak A we take its ``fanout`` nearest later peaks inside a target zone
(time 1..``MAX_DT`` frames, frequency within ``MAX_DF`` bins) and hash every ordered
triple (P, Q, R) of them:

    hash = (round(f_P - f_A), round(f_Q - f_A), round(f_R - f_A),
            dt_AP // 2, dt_PQ // 2, dt_QR // 2)                       -> uint32

Only frequency *differences* appear, so a pitch shift (a translation on the log-frequency
axis) leaves the hash unchanged. Time differences are not tempo-invariant; the matcher
compensates by rescaling query peak times over a grid of speed factors before hashing
(see :mod:`melodyai.fingerprint.index`). Each hash keeps its anchor time and frequency,
used later to verify a consistent time offset *and* a constant pitch offset.

Why quads and not pairs/triplets: dropping the absolute frequency (needed for pitch
invariance) shrinks the hash space; with triplets only ~2.3 M distinct values exist, which
would make posting lists explode in a catalog of millions of tracks. Quads give ~3.5e9
distinct values (fits in 32 bits) at the cost of needing four peaks to survive distortion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from melodyai.fingerprint.peaks import Peaks

MAX_DF = 30          # bins (15 semitones at 24 bins/octave)
MAX_DT = 48          # frames (~1.5 s)
DT_STEP = 2          # time-difference quantisation (frames)
FANOUT = 5           # targets per anchor -> C(5, 3) = 10 hashes per anchor

_DF_RANGE = 2 * MAX_DF + 1          # 61
_DT_RANGE = MAX_DT // DT_STEP + 1   # 25
assert _DF_RANGE**3 * _DT_RANGE**3 < 2**32


@dataclass
class Hashes:
    h: np.ndarray    # uint32 hash values
    t: np.ndarray    # float32 anchor time (frames; fractional after time scaling)
    f: np.ndarray    # float32 anchor log-frequency bin

    def __len__(self) -> int:
        return len(self.h)

    @classmethod
    def empty(cls) -> Hashes:
        return cls(np.zeros(0, np.uint32), np.zeros(0, np.float32), np.zeros(0, np.float32))


def pack(dfs: list[np.ndarray], dts: list[np.ndarray]) -> np.ndarray:
    v = np.zeros(len(dfs[0]), dtype=np.uint64)
    for d in dfs:
        v = v * _DF_RANGE + (d + MAX_DF).astype(np.uint64)
    for q in dts:
        v = v * _DT_RANGE + q.astype(np.uint64)
    return v.astype(np.uint32)


def constellation_hashes(peaks: Peaks, time_scale: float = 1.0, fanout: int = FANOUT) -> Hashes:
    """Quad hashes of a peak constellation; ``time_scale`` multiplies peak times first."""
    n = len(peaks)
    if n < 4:
        return Hashes.empty()
    t = peaks.t.astype(np.float64) * time_scale
    f = peaks.f.astype(np.float64)
    # Candidate targets: the next ``window`` peaks in time order (peaks are time-sorted).
    window = 6 * fanout
    idx = np.arange(n)[:, None] + np.arange(1, window + 1)[None, :]
    valid = idx < n
    idx = np.where(valid, idx, 0)
    dt = t[idx] - t[:, None]
    df = f[idx] - f[:, None]
    valid &= (dt >= 1.0) & (dt <= MAX_DT) & (np.abs(df) <= MAX_DF)
    # Position of the k-th valid target of each anchor (k < fanout), and how many exist.
    rank = np.cumsum(valid, axis=1)
    count = np.minimum(rank[:, -1], fanout)
    target = np.stack([idx[np.arange(n), np.argmax(rank == k + 1, axis=1)] for k in range(fanout)], axis=1)
    hs, ts, fs = [], [], []
    for a, b, c in combinations(range(fanout), 3):
        anchors = np.flatnonzero(count >= c + 1)
        if not len(anchors):
            continue
        P, Q, R = target[anchors, a], target[anchors, b], target[anchors, c]
        dfs = [np.rint(f[X] - f[anchors]).astype(np.int64) for X in (P, Q, R)]
        dts = [np.clip(np.floor((t[Y] - t[X]) / DT_STEP), 0, _DT_RANGE - 1).astype(np.int64)
               for X, Y in ((anchors, P), (P, Q), (Q, R))]
        good = np.all([np.abs(d) <= MAX_DF for d in dfs], axis=0)
        hs.append(pack([d[good] for d in dfs], [q[good] for q in dts]))
        ts.append(t[anchors][good])
        fs.append(f[anchors][good])
    if not hs:
        return Hashes.empty()
    return Hashes(np.concatenate(hs), np.concatenate(ts).astype(np.float32), np.concatenate(fs).astype(np.float32))
