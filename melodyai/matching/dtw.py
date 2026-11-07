"""Dynamic Time Warping kernels (numba-accelerated).

Step pattern: {(1,1), (1,0), (0,1)} with an additive ``penalty`` on the two
non-diagonal moves. The penalty discourages extreme warping (holding one frame
of a sequence for many frames of the other) without forbidding it. Accumulated
cost is normalised by the path length, so distances are comparable across
sequence lengths (units = cost units per aligned frame pair, e.g. semitones).

Variants:
- :func:`dtw`               classic global alignment, optional Sakoe-Chiba band
- :func:`subsequence_dtw`   query fully aligned, free start/end in the reference
                            (Müller, FMP ch. 7.2)
- :func:`banded_dtw_batch`  many equal-length pairs in parallel (window matching)
- :func:`local_alignment`   Smith-Waterman on note-level similarity matrices
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

DIAG, UP, LEFT = 0, 1, 2


@njit(cache=True)
def _accumulate(C, subseq, penalty, band):
    n, m = C.shape
    D = np.full((n, m), np.inf)
    L = np.zeros((n, m))
    P = np.zeros((n, m), dtype=np.int8)
    for i in range(n):
        if band >= 0 and not subseq:
            centre = i * (m - 1) / max(n - 1, 1)
            j0 = max(0, int(np.floor(centre - band)))
            j1 = min(m, int(np.ceil(centre + band)) + 1)
        else:
            j0, j1 = 0, m
        for j in range(j0, j1):
            c = C[i, j]
            if i == 0 and (j == 0 or subseq):
                D[i, j] = c
                L[i, j] = 1.0
                continue
            best = np.inf
            bl = 0.0
            bp = DIAG
            if i > 0 and j > 0 and D[i - 1, j - 1] < np.inf:
                best = D[i - 1, j - 1] + c
                bl = L[i - 1, j - 1] + 1.0
                bp = DIAG
            if i > 0 and D[i - 1, j] < np.inf:
                v = D[i - 1, j] + c + penalty
                if v < best:
                    best, bl, bp = v, L[i - 1, j] + 1.0, UP
            if j > 0 and D[i, j - 1] < np.inf:
                v = D[i, j - 1] + c + penalty
                if v < best:
                    best, bl, bp = v, L[i, j - 1] + 1.0, LEFT
            D[i, j] = best
            L[i, j] = bl
            P[i, j] = bp
    return D, L, P


@njit(cache=True)
def _backtrack(P, i, j, subseq):
    path = np.empty((i + j + 2, 2), dtype=np.int64)
    k = 0
    while True:
        path[k, 0] = i
        path[k, 1] = j
        k += 1
        if i == 0 and (j == 0 or subseq):
            break
        p = P[i, j]
        if p == DIAG:
            i -= 1
            j -= 1
        elif p == UP:
            i -= 1
        else:
            j -= 1
    return path[:k][::-1]


@dataclass
class DTWResult:
    distance: float          # normalised accumulated cost
    path: np.ndarray         # (K, 2) aligned index pairs (query, reference)
    start: int               # reference start index
    end: int                 # reference end index (inclusive)
    cost: np.ndarray | None = None  # accumulated cost matrix (optional, for visualisation)


def abs_cost(q: np.ndarray, r: np.ndarray, clip: float = np.inf) -> np.ndarray:
    """Robust absolute-difference cost (clipped so single wrong notes do not dominate)."""
    return np.minimum(np.abs(q[:, None] - r[None, :]), clip)


def cosine_cost(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    return 1.0 - qn @ rn.T


def dtw(C: np.ndarray, penalty: float = 0.0, band: int = -1, keep_cost: bool = False) -> DTWResult:
    C = np.ascontiguousarray(C, dtype=np.float64)
    D, L, P = _accumulate(C, False, penalty, band)
    n, m = C.shape
    d = D[n - 1, m - 1] / max(L[n - 1, m - 1], 1.0)
    path = _backtrack(P, n - 1, m - 1, False) if np.isfinite(d) else np.zeros((0, 2), dtype=np.int64)
    return DTWResult(float(d), path, 0, m - 1, D if keep_cost else None)


def subsequence_dtw(C: np.ndarray, penalty: float = 0.0, keep_cost: bool = False) -> DTWResult:
    C = np.ascontiguousarray(C, dtype=np.float64)
    D, L, P = _accumulate(C, True, penalty, -1)
    n = C.shape[0]
    norm = D[n - 1] / np.maximum(L[n - 1], 1.0)
    j = int(np.argmin(norm))
    path = _backtrack(P, n - 1, j, True)
    return DTWResult(float(norm[j]), path, int(path[0, 1]), j, D if keep_cost else None)


@njit(cache=True)
def _banded_pair(q, r, band, clip, penalty, shift):
    n = q.shape[0]
    INF = 1e18
    D = np.full((n, n), INF)
    L = np.zeros((n, n))
    for i in range(n):
        j0 = max(0, i - band)
        j1 = min(n, i + band + 1)
        for j in range(j0, j1):
            c = abs(q[i] - shift - r[j])
            if c > clip:
                c = clip
            if i == 0 and j == 0:
                D[i, j] = c
                L[i, j] = 1.0
                continue
            best = INF
            bl = 0.0
            if i > 0 and j > 0 and D[i - 1, j - 1] < INF:
                best = D[i - 1, j - 1] + c
                bl = L[i - 1, j - 1] + 1.0
            if i > 0 and D[i - 1, j] < INF:
                v = D[i - 1, j] + c + penalty
                if v < best:
                    best = v
                    bl = L[i - 1, j] + 1.0
            if j > 0 and D[i, j - 1] < INF:
                v = D[i, j - 1] + c + penalty
                if v < best:
                    best = v
                    bl = L[i, j - 1] + 1.0
            D[i, j] = best
            L[i, j] = bl
    return D[n - 1, n - 1] / L[n - 1, n - 1]


@njit(parallel=True, cache=True)
def banded_dtw_batch(q, W, band, clip, penalty, shifts):
    """Distance between query ``q`` (N,) and each row of ``W`` (M, N), minimised over key ``shifts``.

    ``shifts`` has shape (M,) or (1,): per-window key offsets to subtract from the query.
    """
    M = W.shape[0]
    out = np.empty(M)
    for k in prange(M):
        s = shifts[k] if shifts.shape[0] == M else shifts[0]
        out[k] = _banded_pair(q, W[k], band, clip, penalty, s)
    return out


@njit(cache=True)
def _sw(S, gap):
    n, m = S.shape
    H = np.zeros((n + 1, m + 1))
    best = 0.0
    bi, bj = 0, 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            v = max(0.0, H[i - 1, j - 1] + S[i - 1, j - 1], H[i - 1, j] - gap, H[i, j - 1] - gap)
            H[i, j] = v
            if v > best:
                best, bi, bj = v, i, j
    return best, bi, bj, H


def local_alignment(S: np.ndarray, gap: float = 0.6) -> tuple[float, int, int, np.ndarray]:
    """Smith-Waterman local alignment over a similarity matrix ``S`` (positive = match).

    Returns (best_score, end_i, end_j, H). End indices are 1-based into S.
    """
    best, bi, bj, H = _sw(np.ascontiguousarray(S, dtype=np.float64), float(gap))
    return float(best), int(bi), int(bj), H


def sw_start(H: np.ndarray, S: np.ndarray, bi: int, bj: int, gap: float) -> tuple[int, int]:
    """Trace back a Smith-Waterman alignment to its start cell (0-based indices into S)."""
    i, j = bi, bj
    while i > 0 and j > 0 and H[i, j] > 0:
        if np.isclose(H[i, j], H[i - 1, j - 1] + S[i - 1, j - 1]):
            i, j = i - 1, j - 1
        elif np.isclose(H[i, j], H[i - 1, j] - gap):
            i -= 1
        else:
            j -= 1
    return i, j


def warmup() -> None:
    """Trigger numba compilation (cached on disk afterwards)."""
    q = np.random.rand(8)
    banded_dtw_batch(q, np.random.rand(2, 8), 2, 3.0, 0.1, np.zeros(1))
    subsequence_dtw(abs_cost(q, np.random.rand(12)), 0.1)
    dtw(abs_cost(q, np.random.rand(9)), 0.1, 3)
    local_alignment(np.random.rand(4, 5) - 0.5)
