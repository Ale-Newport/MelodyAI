"""Neural embedding search and neural + DTW re-ranking.

Every reference window of the :class:`MelodyIndex` is embedded once per model and cached
per song (``<store>/indexes/neural/<model_id>/<song_id>.npy``), so adding or removing a
song only embeds that song. The vector index is rebuilt from the cached embeddings when
the catalog version changes (cheap: a concatenation plus, for IVF/HNSW, a small build).

Index kinds: ``exact`` and ``approx``; the backend (NumPy or FAISS) is chosen by
:func:`melodyai.indexing.vector.vector_backend`.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np

from melodyai.config import torch_device
from melodyai.features.extract import QueryFeatures
from melodyai.features.windows import query_vector
from melodyai.indexing.melody_index import MelodyIndex
from melodyai.indexing.vector import build_vector_index, vector_backend
from melodyai.matching.algorithms import Candidate, WindowDTW, _rank
from melodyai.matching.dtw import banded_dtw_batch


def model_id(path: Path) -> str:
    h = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]
    return f"{Path(path).stem}-{h}"


class NeuralEmbedder:
    def __init__(self, checkpoint: Path, device: str = "auto") -> None:
        from melodyai.models.encoder import load_encoder

        self.device = torch_device(device)
        self.checkpoint = Path(checkpoint)
        self.model, self.ckpt = load_encoder(self.checkpoint, self.device)
        self.id = model_id(self.checkpoint)
        self.n_points = self.model.config["n_points"]
        self._lock = threading.Lock()

    def embed(self, X: np.ndarray, is_query: bool = False) -> np.ndarray:
        import torch

        with self._lock, torch.no_grad():
            out = []
            for i in range(0, len(X), 1024):
                x = torch.from_numpy(np.ascontiguousarray(X[i:i + 1024], dtype=np.float32)).to(self.device)
                out.append(self.model(x, is_query=is_query).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.model.config["dim"]), np.float32)


class NeuralIndex:
    def __init__(self, embedder: NeuralEmbedder, melody_index: MelodyIndex, cache_dir: Path | None = None,
                 kind: str = "exact") -> None:
        self.embedder = embedder
        self.melody_index = melody_index
        self.cache_dir = Path(cache_dir) / embedder.id if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        self._emb: dict[str, np.ndarray] = {}
        self._built_version = -1
        self._index = None
        self._E: np.ndarray | None = None
        self._lock = threading.RLock()
        self.stats = {"songs_embedded": 0}

    def song_embeddings(self, song_id: str) -> np.ndarray:
        if song_id in self._emb:
            return self._emb[song_id]
        p = self.cache_dir / f"{song_id}.npy" if self.cache_dir else None
        if p is not None and p.exists():
            e = np.load(p)
        else:
            e = self.embedder.embed(self.melody_index.windows_of(song_id).vectors)
            self.stats["songs_embedded"] += 1
            if p is not None:
                np.save(p, e)
        self._emb[song_id] = e
        return e

    def forget(self, song_id: str) -> None:
        self._emb.pop(song_id, None)
        if self.cache_dir:
            (self.cache_dir / f"{song_id}.npy").unlink(missing_ok=True)

    def _ensure(self):
        with self._lock:
            if self._built_version == self.melody_index.version and self._E is not None:
                return
            M = self.melody_index.matrix()
            parts = [self.song_embeddings(s) for s in M.song_ids]
            for s in list(self._emb):
                if s not in self.melody_index:
                    self._emb.pop(s)
            E = np.concatenate(parts) if parts else np.zeros((0, self.embedder.model.config["dim"]), np.float32)
            self._E = np.ascontiguousarray(E, dtype=np.float32)
            self._index = build_vector_index(self._E, approximate=self.kind == "approx") if len(E) else None
            self._built_version = self.melody_index.version

    def search_windows(self, qv: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        self._ensure()
        if self._index is None:
            return np.zeros(0), np.zeros(0, dtype=int)
        e = self.embedder.embed(qv[None, :], is_query=True)[0]
        return self._index.search(e, min(k, len(self._E)))

    @property
    def index_name(self) -> str:
        self._ensure()
        return getattr(self._index, "name", "empty")


class NeuralMatcher:
    """Embedding nearest-neighbour search, grouped per song (max similarity)."""

    tau = 1.0

    def __init__(self, nindex: NeuralIndex, name: str = "neural", k_windows: int = 300) -> None:
        self.nindex = nindex
        self.name = name
        self.label = f"Neural embeddings ({nindex.kind} search, {vector_backend()})"
        self.k_windows = k_windows

    def window_hits(self, query: QueryFeatures):
        qv = query_vector(query.contour, self.nindex.embedder.n_points)
        sims, ids = self.nindex.search_windows(qv, self.k_windows)
        return qv, sims, ids

    def search(self, query: QueryFeatures, index: MelodyIndex, top_k: int = 10) -> list[Candidate]:
        if len(query.contour) < 4:
            return []
        _, sims, ids = self.window_hits(query)
        M = index.matrix()
        best: dict[int, tuple[float, int]] = {}
        for s, w in zip(sims, ids):
            song = int(M.song_idx[w])
            if song not in best or s > best[song][0]:
                best[song] = (float(s), int(w))
        out = [Candidate(M.song_ids[s], 1.0 - sim, max(0.0, sim), float(M.t0[w]), float(M.t1[w]),
                         {"cosine": round(sim, 4)}) for s, (sim, w) in best.items()]
        return _rank(out, top_k)


class NeuralRerank(NeuralMatcher):
    """Neural candidate windows re-ranked with the window DTW distance."""

    def __init__(self, nindex: NeuralIndex, name: str = "neural_rerank", k_windows: int = 300) -> None:
        super().__init__(nindex, name, k_windows)
        self.label = "Neural embeddings + DTW re-ranking"
        self.dtw = WindowDTW()

    def search(self, query: QueryFeatures, index: MelodyIndex, top_k: int = 10) -> list[Candidate]:
        if len(query.contour) < 4:
            return []
        qv, sims, ids = self.window_hits(query)
        if not len(ids):
            return []
        M = index.matrix()
        W = M.vectors[ids].astype(np.float64)
        band = max(2, int(self.dtw.band_frac * index.n_points))
        d0 = banded_dtw_batch(qv, W, band, self.dtw.clip, self.dtw.penalty, np.zeros(1))
        d1 = banded_dtw_batch(qv, W, band, self.dtw.clip, self.dtw.penalty, np.median(qv[None, :] - W, axis=1))
        d = np.minimum(d0, d1)
        best: dict[int, tuple[float, int, float]] = {}
        for dist, w, sim in zip(d, ids, sims):
            song = int(M.song_idx[w])
            if song not in best or dist < best[song][0]:
                best[song] = (float(dist), int(w), float(sim))
        out = [Candidate(M.song_ids[s], dist, float(np.exp(-dist / self.dtw.tau)), float(M.t0[w]), float(M.t1[w]),
                         {"cosine": round(sim, 4)}) for s, (dist, w, sim) in best.items()]
        return _rank(out, top_k)


def attach_neural(engine, catalog, checkpoint: Path, cache_root: Path | None = None) -> dict:
    """Register neural algorithms on a RecognitionEngine and keep them in sync with the catalog."""
    from melodyai.matching.ensemble import MelodyEnsemble

    emb = NeuralEmbedder(checkpoint)
    cache = cache_root / "indexes" / "neural" if cache_root else None
    flat = NeuralIndex(emb, catalog.melody_index, cache, "exact")
    approx = NeuralIndex(emb, catalog.melody_index, None, "approx")
    approx._emb = flat._emb  # share per-song embeddings
    catalog.subscribe(lambda ref: flat.forget(ref.song_id), flat.forget)  # re-embed replaced songs lazily
    engine.register_algorithm(NeuralMatcher(flat, "neural"))
    engine.register_algorithm(NeuralMatcher(approx, "neural_approx"))
    rerank = NeuralRerank(flat, "neural_rerank")
    engine.register_algorithm(rerank)
    engine.register_algorithm(MelodyEnsemble(
        "ensemble_neural", {"dtw": engine.algorithms["dtw"], "note_align": engine.algorithms["note_align"],
                            "subseq_dtw": engine.algorithms["subseq_dtw"], "neural_rerank": rerank},
        label="Ensemble + neural re-ranking"))
    return {"model_id": emb.id, "checkpoint": str(checkpoint), "device": emb.device, "vector_backend": vector_backend(),
            "val": emb.ckpt.get("val"), "epoch": emb.ckpt.get("epoch")}


def detach_neural(engine) -> None:
    for name in ("neural", "neural_approx", "neural_rerank", "ensemble_neural"):
        engine.unregister_algorithm(name)
