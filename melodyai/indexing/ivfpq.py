"""IVF-PQ approximate nearest-neighbour index for inner-product search (pure NumPy).

Why our own implementation: on macOS the faiss-cpu and PyTorch wheels cannot share a process
(duplicate libomp, see LIMITATIONS.md), and the neural encoder needs PyTorch. This index keeps
the classic FAISS ``IndexIVFPQ`` design so results transfer, and :func:`make_ivfpq` returns a FAISS
index instead wherever FAISS is safe to use.

Design (Jégou et al., 2011, "Product quantization for nearest neighbor search"):
* coarse quantiser: ``nlist`` spherical k-means centroids; a vector is stored in the list of its
  closest centroid;
* the residual (vector - centroid) is split into ``m`` sub-vectors, each replaced by the id of the
  nearest of 256 sub-centroids -> ``m`` bytes per vector (16 bytes for m=16 vs 512 for float32 d=128);
* query: probe the ``nprobe`` closest lists; for inner product the score decomposes exactly as
  ``q.c + sum_j q_j . codebook_j[code_j]``, so one 256 x m lookup table per query serves every list.

Storage: lists are kept as one array sorted by list id + offsets (``main``) plus an append-only
``delta`` for incremental additions (scanned with the same scoring); ``consolidate()`` merges them.
``save()``/``load()`` use .npy files (memory-mapped on load). Removal is by id filter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _kmeans(X: np.ndarray, k: int, iters: int, rng: np.random.Generator, spherical: bool) -> np.ndarray:
    C = X[rng.choice(len(X), size=k, replace=len(X) < k)].astype(np.float32).copy()
    for _ in range(iters):
        if spherical:
            assign = np.argmax(X @ C.T, axis=1)
        else:  # squared L2 via the expansion |x|^2 - 2 x.c + |c|^2
            assign = np.argmin(-2.0 * (X @ C.T) + (C * C).sum(1)[None, :], axis=1)
        sums = np.zeros_like(C)
        np.add.at(sums, assign, X)
        cnt = np.bincount(assign, minlength=k)
        empty = cnt == 0
        C[~empty] = sums[~empty] / cnt[~empty, None]
        if empty.any():
            C[empty] = X[rng.choice(len(X), size=int(empty.sum()))]
        if spherical:
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    return C


class IVFPQ:
    def __init__(self, dim: int, nlist: int = 1024, m: int = 16, nprobe: int = 16, seed: int = 0) -> None:
        if dim % m:
            raise ValueError("dim must be divisible by m")
        self.dim, self.nlist, self.m, self.nprobe, self.seed = dim, nlist, m, nprobe, seed
        self.dsub = dim // m
        self.coarse: np.ndarray | None = None      # (nlist, dim)
        self.codebooks: np.ndarray | None = None   # (m, 256, dsub)
        self.codes = np.zeros((0, m), np.uint8)
        self.ids = np.zeros(0, np.int64)
        self.offsets = np.zeros(nlist + 1, np.int64)
        self._dcodes: list[np.ndarray] = []
        self._dids: list[np.ndarray] = []
        self._dlists: list[np.ndarray] = []
        self.removed: set[int] = set()

    # ------------------------------------------------------------------ training / adding
    @property
    def trained(self) -> bool:
        return self.coarse is not None

    def __len__(self) -> int:
        return len(self.ids) + sum(len(x) for x in self._dids)

    def train(self, X: np.ndarray, iters: int = 10) -> None:
        rng = np.random.default_rng(self.seed)
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.nlist = min(self.nlist, max(1, len(X) // 39))  # >= ~39 points per centroid
        self.offsets = np.zeros(self.nlist + 1, np.int64)
        self.coarse = _kmeans(X, self.nlist, iters, rng, spherical=True)
        R = X - self.coarse[np.argmax(X @ self.coarse.T, axis=1)]
        self.codebooks = np.stack([
            _kmeans(np.ascontiguousarray(R[:, j * self.dsub:(j + 1) * self.dsub]), 256, iters, rng, spherical=False)
            for j in range(self.m)])

    def _assign(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X), np.int64)
        for i in range(0, len(X), 65536):
            out[i:i + 65536] = np.argmax(X[i:i + 65536] @ self.coarse.T, axis=1)
        return out

    def _encode(self, X: np.ndarray, lists: np.ndarray) -> np.ndarray:
        R = X - self.coarse[lists]
        codes = np.empty((len(X), self.m), np.uint8)
        for j in range(self.m):
            sub = R[:, j * self.dsub:(j + 1) * self.dsub]
            cb = self.codebooks[j]
            d = -2.0 * (sub @ cb.T) + (cb * cb).sum(1)[None, :]
            codes[:, j] = np.argmin(d, axis=1)
        return codes

    def add(self, X: np.ndarray, ids: np.ndarray) -> None:
        if not self.trained:
            raise RuntimeError("train() first")
        X = np.ascontiguousarray(X, dtype=np.float32)
        lists = self._assign(X)
        for i in range(0, len(X), 65536):
            self._dcodes.append(self._encode(X[i:i + 65536], lists[i:i + 65536]))
        self._dids.append(np.asarray(ids, np.int64))
        self._dlists.append(lists)

    def remove(self, ids) -> None:
        self.removed.update(int(i) for i in ids)

    def consolidate(self) -> None:
        """Merge the delta into the list-sorted main arrays (and drop removed ids)."""
        if not self._dids and not self.removed:
            return
        main_lists = np.repeat(np.arange(self.nlist), np.diff(self.offsets))
        codes = np.concatenate([np.asarray(self.codes)] + self._dcodes)
        ids = np.concatenate([np.asarray(self.ids)] + self._dids)
        lists = np.concatenate([main_lists] + self._dlists)
        if self.removed:
            keep = ~np.isin(ids, np.fromiter(self.removed, np.int64))
            codes, ids, lists = codes[keep], ids[keep], lists[keep]
            self.removed = set()
        o = np.argsort(lists, kind="stable")
        self.codes, self.ids = codes[o], ids[o]
        self.offsets = np.r_[0, np.cumsum(np.bincount(lists, minlength=self.nlist))]
        self._dcodes, self._dids, self._dlists = [], [], []

    # ------------------------------------------------------------------ search
    def search(self, q: np.ndarray, k: int, nprobe: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        nprobe = min(nprobe or self.nprobe, self.nlist)
        q = np.asarray(q, np.float32)
        cs = self.coarse @ q
        probe = np.argpartition(-cs, nprobe - 1)[:nprobe]
        lut = np.einsum("jkd,jd->jk", self.codebooks, q.reshape(self.m, self.dsub))  # (m, 256)
        cols = np.arange(self.m)
        scores, idss = [], []
        for li in probe:
            a, b = self.offsets[li], self.offsets[li + 1]
            if b > a:
                c = np.asarray(self.codes[a:b])
                scores.append(cs[li] + lut[cols, c].sum(axis=1))
                idss.append(np.asarray(self.ids[a:b]))
        if self._dids:
            dl = np.concatenate(self._dlists)
            sel = np.isin(dl, probe)
            if sel.any():
                c = np.concatenate(self._dcodes)[sel]
                scores.append(cs[dl[sel]] + lut[cols, c].sum(axis=1))
                idss.append(np.concatenate(self._dids)[sel])
        if not scores:
            return np.zeros(0, np.float32), np.zeros(0, np.int64)
        s, i = np.concatenate(scores), np.concatenate(idss)
        if self.removed:
            keep = ~np.isin(i, np.fromiter(self.removed, np.int64))
            s, i = s[keep], i[keep]
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return s[top], i[top]

    # ------------------------------------------------------------------ persistence
    @property
    def nbytes(self) -> int:
        return int(np.asarray(self.codes).nbytes + np.asarray(self.ids).nbytes + sum(x.nbytes for x in self._dcodes)
                   + sum(x.nbytes for x in self._dids) + self.coarse.nbytes + self.codebooks.nbytes)

    def save(self, folder: Path | str) -> None:
        self.consolidate()
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("coarse", "codebooks", "codes", "ids", "offsets"):
            np.save(folder / f"{name}.npy", np.asarray(getattr(self, name)))
        (folder / "meta.json").write_text(json.dumps({"dim": self.dim, "nlist": self.nlist, "m": self.m,
                                                      "nprobe": self.nprobe, "seed": self.seed}))

    @classmethod
    def load(cls, folder: Path | str) -> IVFPQ:
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        idx = cls(meta["dim"], meta["nlist"], meta["m"], meta["nprobe"], meta["seed"])
        idx.coarse = np.load(folder / "coarse.npy")
        idx.codebooks = np.load(folder / "codebooks.npy")
        idx.codes = np.load(folder / "codes.npy", mmap_mode="r")
        idx.ids = np.load(folder / "ids.npy", mmap_mode="r")
        idx.offsets = np.load(folder / "offsets.npy")
        return idx


def make_ivfpq(dim: int, nlist: int = 1024, m: int = 16, nprobe: int = 16):
    """FAISS IndexIVFPQ where FAISS is usable, else the NumPy implementation (same parameters)."""
    from melodyai.indexing.vector import vector_backend

    if vector_backend() == "faiss":
        import faiss

        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quant, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = nprobe
        return index
    return IVFPQ(dim, nlist, m, nprobe)
