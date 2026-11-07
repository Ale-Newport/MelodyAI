"""In-memory melody index with incremental add/remove.

Per-song :class:`ReferenceFeatures` are computed once (feature store on disk) and the
window matrix is assembled lazily. Adding or removing a song only touches that song:
other songs' features are never recomputed (see ``tests/integration/test_catalog.py``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from melodyai.features.extract import ReferenceFeatures
from melodyai.features.windows import DEFAULT_LENGTHS, N_POINTS, Windows, build_windows


@dataclass
class WindowMatrix:
    song_ids: list[str]
    song_idx: np.ndarray
    start_note: np.ndarray
    end_note: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    vectors: np.ndarray
    offsets: np.ndarray

    def __len__(self) -> int:
        return len(self.song_idx)


class MelodyIndex:
    def __init__(self, lengths=DEFAULT_LENGTHS, n_points: int = N_POINTS) -> None:
        self.lengths = tuple(lengths)
        self.n_points = n_points
        self.refs: dict[str, ReferenceFeatures] = {}
        self._windows: dict[str, Windows] = {}
        self._matrix: WindowMatrix | None = None
        self._lock = threading.RLock()
        self.version = 0  # bumped on every mutation (used by dependent indexes)

    def __len__(self) -> int:
        return len(self.refs)

    def __contains__(self, song_id: str) -> bool:
        return song_id in self.refs

    def add(self, ref: ReferenceFeatures) -> int:
        """Add or replace one reference. Returns the number of windows created."""
        w = build_windows(ref.notes, self.lengths, self.n_points)
        with self._lock:
            self.refs[ref.song_id] = ref
            self._windows[ref.song_id] = w
            self._matrix = None
            self.version += 1
        return len(w)

    def remove(self, song_id: str) -> bool:
        with self._lock:
            found = self.refs.pop(song_id, None) is not None
            self._windows.pop(song_id, None)
            if found:
                self._matrix = None
                self.version += 1
            return found

    def windows_of(self, song_id: str) -> Windows:
        return self._windows[song_id]

    def matrix(self) -> WindowMatrix:
        with self._lock:
            if self._matrix is None:
                ids = list(self._windows)
                parts = [self._windows[s] for s in ids]
                if not parts or sum(len(p) for p in parts) == 0:
                    z = np.zeros(0)
                    self._matrix = WindowMatrix(ids, z.astype(int), z.astype(int), z.astype(int), z, z,
                                                np.zeros((0, self.n_points), np.float32), z)
                else:
                    self._matrix = WindowMatrix(
                        ids,
                        np.concatenate([np.full(len(p), i) for i, p in enumerate(parts)]),
                        np.concatenate([p.start_note for p in parts]),
                        np.concatenate([p.end_note for p in parts]),
                        np.concatenate([p.t0 for p in parts]),
                        np.concatenate([p.t1 for p in parts]),
                        np.concatenate([p.vectors for p in parts]).astype(np.float32),
                        np.concatenate([p.offsets for p in parts]),
                    )
            return self._matrix
