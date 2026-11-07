"""Two-stage melody search for large catalogs.

Stage 1 (candidate generation, sub-linear): the query contour is embedded by the neural encoder
and looked up in an IVF-PQ index of reference-window embeddings; window hits are grouped per
song (max similarity) and the best ``n_candidates`` songs are kept.

Stage 2 (verification, cost independent of catalog size): only those songs are scored with the
classical matchers that performed best on the benchmark (window DTW and note alignment), whose
relative scores are averaged as in the default ensemble.

Reference features are fetched through ``loader(song_id) -> ReferenceFeatures`` (e.g. from the
on-disk feature store) with an LRU cache, so the full catalog never has to sit in memory.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np

from melodyai.features.extract import QueryFeatures, ReferenceFeatures
from melodyai.features.windows import DEFAULT_LENGTHS, N_POINTS, build_windows, query_vector
from melodyai.indexing.melody_index import MelodyIndex
from melodyai.matching.algorithms import ALGORITHMS, Candidate

# Stage-1 windows: mid lengths with wider hops (~75 per song instead of ~290) keep the ANN index
# small; stage 2 re-scores candidates with the full multi-length window set.
ANN_WINDOW_LENGTHS = (9, 13, 18)
ANN_WINDOW_HOPS = {9: 3, 13: 5, 18: 7}


class LRU:
    def __init__(self, loader, size: int = 5000) -> None:
        self.loader, self.size, self.d = loader, size, OrderedDict()

    def __call__(self, key):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        v = self.loader(key)
        self.d[key] = v
        if len(self.d) > self.size:
            self.d.popitem(last=False)
        return v


class ScalableMelodySearch:
    name = "scalable"
    label = "Two-stage: neural IVF-PQ candidates + DTW/note-alignment verification"
    tau = 1.0

    def __init__(self, embedder, ann, window_song: np.ndarray, song_ids: list[str], loader,
                 n_candidates: int = 50, k_windows: int = 600, nprobe: int = 32,
                 verifiers=("dtw", "note_align")) -> None:
        self.embedder = embedder              # .embed(X, is_query) -> normalised embeddings
        self.ann = ann                        # IVFPQ (or FAISS) over window embeddings; ids = window ids
        self.window_song = window_song        # window id -> song index
        self.song_ids = song_ids
        self.refs = LRU(loader)
        self._windows = LRU(lambda sid: build_windows(self.refs(sid).notes, DEFAULT_LENGTHS, N_POINTS))
        self.n_candidates, self.k_windows, self.nprobe = n_candidates, k_windows, nprobe
        self.verifiers = {v: ALGORITHMS[v] for v in verifiers}
        self.last_timings: dict = {}

    def candidates(self, query: QueryFeatures, n: int | None = None) -> list[tuple[str, float]]:
        qv = query_vector(query.contour, self.embedder.n_points)
        e = self.embedder.embed(qv[None, :].astype(np.float32), is_query=True)[0]
        sims, wids = (self.ann.search(e, self.k_windows, self.nprobe) if hasattr(self.ann, "codebooks")
                      else _faiss_search(self.ann, e, self.k_windows))
        best: dict[int, float] = {}
        for s, w in zip(sims, wids):
            si = int(self.window_song[w])
            if si not in best or s > best[si]:
                best[si] = float(s)
        ranked = sorted(best.items(), key=lambda x: -x[1])[: (n or self.n_candidates)]
        return [(self.song_ids[i], s) for i, s in ranked]

    def search(self, query: QueryFeatures, index: MelodyIndex | None = None, top_k: int = 10,
               candidates: list[tuple[str, float]] | None = None) -> list[Candidate]:
        if len(query.contour) < 4:
            return []
        t0 = time.perf_counter()
        cands = candidates[: self.n_candidates] if candidates is not None else self.candidates(query)
        t1 = time.perf_counter()
        sub = MelodyIndex()
        for sid, _ in cands:
            ref: ReferenceFeatures = self.refs(sid)
            sub.refs[sid] = ref
            sub._windows[sid] = self._windows(sid)
        sub._matrix = None
        per = {name: {c.song_id: c for c in alg.search(query, sub, len(cands))} for name, alg in self.verifiers.items()}
        out = []
        ann_sim = dict(cands)
        for sid, _ in cands:
            scores = [per[n][sid].score if sid in per[n] else 0.0 for n in per]
            loc = max((per[n][sid] for n in per if sid in per[n]), key=lambda c: c.score, default=None)
            score = float(np.mean(scores))
            out.append(Candidate(sid, 1.0 - score, score, loc.start_time if loc else None,
                                 loc.end_time if loc else None,
                                 {"ann_similarity": round(ann_sim[sid], 4),
                                  "verifier_scores": {n: round(s, 4) for n, s in zip(per, scores)}}))
        out.sort(key=lambda c: c.distance)
        self.last_timings = {"stage1_s": t1 - t0, "stage2_s": time.perf_counter() - t1, "candidates": len(cands)}
        return out[:top_k]


def _faiss_search(index, e: np.ndarray, k: int):
    s, i = index.search(e[None, :].astype(np.float32), k)
    ok = i[0] >= 0
    return s[0][ok], i[0][ok]


def ann_window_vectors(notes, n_points: int = N_POINTS) -> np.ndarray:
    return build_windows(notes, ANN_WINDOW_LENGTHS, n_points, ANN_WINDOW_HOPS).vectors
