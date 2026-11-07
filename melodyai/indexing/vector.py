"""Vector indexes for embedding search (inner product on L2-normalised vectors).

Backends
--------
* ``numpy`` — exact search with a BLAS matrix product (``NumpyFlat``) and an approximate
  inverted-file index (``NumpyIVF``: k-means coarse quantiser + ``nprobe`` lists). Always
  available; at demo scale (tens of thousands of vectors) exact search takes ~1 ms.
* ``faiss`` — ``IndexFlatIP`` (exact) and ``IndexHNSWFlat`` (approximate).

Backend choice (``MELODYAI_VECTOR_BACKEND`` = auto | numpy | faiss): ``auto`` uses FAISS when it
is installed **except on macOS**. There, the PyTorch and faiss-cpu wheels each bundle their own
``libomp``; loading both in one process aborts with "OMP: Error #15", and the commonly suggested
``KMP_DUPLICATE_LIB_OK=TRUE`` workaround segfaulted in our tests (see docs/LIMITATIONS.md).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np


def vector_backend() -> str:
    choice = os.environ.get("MELODYAI_VECTOR_BACKEND", "auto").lower()
    has_faiss = importlib.util.find_spec("faiss") is not None
    if choice == "faiss":
        if not has_faiss:
            raise RuntimeError("MELODYAI_VECTOR_BACKEND=faiss but faiss is not installed")
        return "faiss"
    if choice == "numpy":
        return "numpy"
    return "faiss" if has_faiss and sys.platform != "darwin" else "numpy"


class NumpyFlat:
    name = "numpy-flat (exact)"

    def __init__(self, E: np.ndarray) -> None:
        self.E = np.ascontiguousarray(E, dtype=np.float32)

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sims = self.E @ q.astype(np.float32)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return sims[top], top


def _kmeans(X: np.ndarray, k: int, iters: int, seed: int) -> np.ndarray:
    """Spherical k-means (cosine), plain NumPy."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(X @ C.T, axis=1)
        for j in range(k):
            m = assign == j
            if m.any():
                c = X[m].sum(axis=0)
                C[j] = c / (np.linalg.norm(c) + 1e-9)
            else:
                C[j] = X[rng.integers(len(X))]
    return C


class NumpyIVF:
    """Approximate search: probe only the ``nprobe`` closest of ``nlist`` k-means cells."""

    def __init__(self, E: np.ndarray, nlist: int | None = None, nprobe: int = 8, seed: int = 0) -> None:
        self.E = np.ascontiguousarray(E, dtype=np.float32)
        n = len(self.E)
        self.nlist = nlist or max(1, min(256, int(np.sqrt(n))))
        self.nprobe = min(nprobe, self.nlist)
        self.name = f"numpy-ivf (approximate, nlist={self.nlist}, nprobe={self.nprobe})"
        self.C = _kmeans(self.E, self.nlist, 8, seed) if n > self.nlist else self.E.copy()
        assign = np.argmax(self.E @ self.C.T, axis=1)
        self.lists = [np.flatnonzero(assign == j) for j in range(len(self.C))]

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = q.astype(np.float32)
        cells = np.argsort(-(self.C @ q))[: self.nprobe]
        ids = np.concatenate([self.lists[c] for c in cells]) if len(cells) else np.zeros(0, int)
        if not len(ids):
            return np.zeros(0, np.float32), ids
        sims = self.E[ids] @ q
        k = min(k, len(ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return sims[top], ids[top]


class FaissIndex:
    def __init__(self, E: np.ndarray, approximate: bool) -> None:
        import faiss

        E = np.ascontiguousarray(E, dtype=np.float32)
        if approximate:
            idx = faiss.IndexHNSWFlat(E.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efSearch = 128
            self.name = "faiss-hnsw (approximate, M=32, efSearch=128)"
        else:
            idx = faiss.IndexFlatIP(E.shape[1])
            self.name = "faiss-flat (exact)"
        idx.add(E)
        self.idx = idx

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sims, ids = self.idx.search(q[None, :].astype(np.float32), k)
        ok = ids[0] >= 0
        return sims[0][ok], ids[0][ok]


def build_vector_index(E: np.ndarray, approximate: bool = False, backend: str | None = None):
    backend = backend or vector_backend()
    if backend == "faiss":
        return FaissIndex(E, approximate)
    return NumpyIVF(E) if approximate else NumpyFlat(E)
