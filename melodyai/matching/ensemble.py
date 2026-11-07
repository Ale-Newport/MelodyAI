"""Melody ensemble: combine several melody algorithms into one ranking.

Members are run independently; their relative scores are fused by a weighted mean
(missing = 0), so the final score stays in [0, 1] and the order is consistent with it.
"""

from __future__ import annotations

from melodyai.features.extract import QueryFeatures
from melodyai.indexing.melody_index import MelodyIndex
from melodyai.matching.algorithms import Candidate


class MelodyEnsemble:
    def __init__(self, name: str, members: dict, weights: dict | None = None, depth: int = 20,
                 label: str | None = None) -> None:
        self.name = name
        self.members = members
        self.weights = weights or {k: 1.0 for k in members}
        self.depth = depth
        self.label = label or "Ensemble (" + " + ".join(members) + ")"
        self.tau = 1.0

    def search(self, query: QueryFeatures, index: MelodyIndex, top_k: int = 10) -> list[Candidate]:
        per: dict[str, dict[str, Candidate]] = {}
        for name, alg in self.members.items():
            per[name] = {c.song_id: c for c in alg.search(query, index, self.depth)}
        songs = {s for d in per.values() for s in d}
        wsum = sum(self.weights.values())
        out = []
        for sid in songs:
            score = sum(self.weights[n] * (per[n][sid].score if sid in per[n] else 0.0) for n in per) / wsum
            # location from the best-scoring member that found the song
            loc = max((per[n][sid] for n in per if sid in per[n]), key=lambda c: c.score)
            out.append(Candidate(sid, 1.0 - score, score, loc.start_time, loc.end_time,
                                 {"member_scores": {n: round(per[n][sid].score, 4) if sid in per[n] else None
                                                    for n in per}}))
        out.sort(key=lambda c: c.distance)
        return out[:top_k]
