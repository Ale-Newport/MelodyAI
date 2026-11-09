"""Melodies of real songs (Lakh MIDI, CC-BY 4.0) searched by humming or whistling.

The open world (:mod:`melodyai.world.search`) recognises real songs by their *words*. Hummed or whistled
queries have no words, so this module adds a melodic reference set: the vocal lines of the Lakh MIDI
"matched" subset (MIDI transcriptions matched to Million Song Dataset entries, so each has a title and an
artist), built by :mod:`melodyai.world.lakh`.

Storage (one folder, default ``<store>/world_melodies``):

* ``melodies.db`` -- SQLite, one row per indexed melody (a song may have up to two: two MIDI transcriptions,
  or two plausible vocal tracks of one file): song key / title / artist (the same
  :func:`melodyai.world.text.song_key` as the lyrics index, so both open-world engines name a song alike),
  Million Song Dataset id, MIDI md5, chosen track and why, and the notes (zlib-compressed float32).
* ``ann/`` + ``window_mid.npy`` -- IVF-PQ index (:class:`melodyai.indexing.ivfpq.IVFPQ`) over neural
  embeddings of mid-length melody windows (:func:`melodyai.matching.scalable.ann_window_vectors`) and the
  melody of every window.
* ``meta.json`` -- build info: encoder checkpoint id (queries must be embedded by the same model), counts.

Search is :class:`melodyai.matching.scalable.ScalableMelodySearch`: IVF-PQ candidates, then window DTW and
note alignment on the best ``n_candidates`` melodies only, whose notes are read from SQLite on demand (LRU).
Results are grouped per song (best version).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from melodyai.features.extract import QueryFeatures, reference_from_notes
from melodyai.features.melody import NoteSequence
from melodyai.world.index import world_song_id

SOURCE = "Lakh MIDI (CC-BY 4.0)"
# People hum songs they know: among melodies that align about as well, the more popular song (Deezer rank of the
# same song in the world lyrics index) is the better guess. On 300 held-out songs of the 25k-song index this lifted
# top-1 for popular songs from 36 % to 52 % and left the whole sample unchanged (35 %).
POPULARITY_WEIGHT = 0.05

_SCHEMA = """
CREATE TABLE IF NOT EXISTS melodies(
    mid INTEGER PRIMARY KEY,
    key TEXT NOT NULL,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    msd_id TEXT,
    md5 TEXT,
    track INTEGER,
    track_name TEXT,
    selection TEXT,            -- "lyrics" (synced lyric events) | "model" | "model-2nd"
    track_score REAL,
    n_notes INTEGER,
    duration_s REAL,
    notes BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_melodies_key ON melodies(key);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID;
"""


def pack_notes(notes: NoteSequence) -> bytes:
    a = np.stack([notes.onsets, notes.durations, notes.pitches]).astype(np.float32)
    return zlib.compress(a.tobytes(), 6)


def unpack_notes(blob: bytes) -> NoteSequence:
    a = np.frombuffer(zlib.decompress(blob), dtype=np.float32).reshape(3, -1)
    return NoteSequence(a[0], a[1], a[2])


@dataclass
class MelodyMatch:
    song_id: str               # "world:" + world_song_id(title, artist)
    key: str
    title: str
    artist: str
    score: float
    start_time: float | None
    end_time: float | None
    mid: int
    msd_id: str | None
    selection: str
    ann_similarity: float
    meta: dict | None = None             # metadata of the same song in the world lyrics index, if any

    def to_dict(self) -> dict:
        from melodyai.world.search import song_links

        m = self.meta or {}
        title, artist = m.get("title") or self.title, m.get("artist") or self.artist
        return {"song_id": self.song_id, "title": title, "artist": artist, "score": round(self.score, 4),
                "start_time": self.start_time, "end_time": self.end_time, "melody_id": self.mid,
                "msd_id": self.msd_id, "selection": self.selection, "ann_similarity": round(self.ann_similarity, 4),
                "album": m.get("album"), "language": m.get("language"), "duration_s": m.get("duration_s"),
                "cover_url": m.get("cover_url"), "preview_url": None, "deezer_id": m.get("deezer_id"),
                "rank": m.get("rank") or 0, "links": song_links(title, artist, m.get("link")), "source": SOURCE}


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


class WorldMelodyIndex:
    """Read side of a built Lakh melody index (see :mod:`melodyai.world.lakh` for the build)."""

    def __init__(self, folder: str | Path, embedder=None, checkpoint: str | Path | None = None,
                 n_candidates: int = 40, nprobe: int = 32, meta_lookup=None) -> None:
        self.folder = Path(folder)
        self.meta_lookup = meta_lookup       # (title, artist) -> WorldSong | None (the world lyrics index)
        self._embedder, self._checkpoint = embedder, checkpoint
        self.n_candidates, self.nprobe = n_candidates, nprobe
        self._lock = threading.Lock()
        self._search = None
        self._conn: sqlite3.Connection | None = None
        self._meta: dict | None = None
        self.error = ""

    # ------------------------------------------------------------------ state
    @property
    def built(self) -> bool:
        return (self.folder / "meta.json").is_file() and (self.folder / "melodies.db").is_file()

    @property
    def meta(self) -> dict:
        if self._meta is None:
            try:
                self._meta = json.loads((self.folder / "meta.json").read_text())
            except (OSError, ValueError):
                self._meta = {}
        return self._meta

    def stats(self) -> dict:
        m = self.meta
        return {"built": self.built, "songs": m.get("songs", 0), "melodies": m.get("melodies", 0),
                "windows": m.get("windows", 0), "built_at": m.get("built_at"), "source": SOURCE,
                "loaded": self._search is not None, "error": self.error}

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(f"file:{self.folder / 'melodies.db'}?mode=ro", uri=True,
                                         check_same_thread=False)
        return self._conn

    def _load(self):
        """Load the ANN index and the encoder once (a few seconds); None if not built or unusable."""
        with self._lock:
            if self._search is not None or not self.built or self.error:
                return self._search
            try:
                from melodyai.indexing.ivfpq import IVFPQ
                from melodyai.matching.scalable import ScalableMelodySearch

                emb = self._embedder
                if emb is None:
                    from melodyai.config import models_dir
                    from melodyai.matching.neural import NeuralEmbedder

                    name = self.meta.get("checkpoint") or "melody_encoder.pt"     # the encoder of the build
                    emb = NeuralEmbedder(Path(self._checkpoint or models_dir() / name))
                if self.meta.get("encoder") and getattr(emb, "id", None) and emb.id != self.meta["encoder"]:
                    raise RuntimeError("the melody index was built with another encoder: rebuild it")
                ann = IVFPQ.load(self.folder / "ann")
                window_mid = np.load(self.folder / "window_mid.npy")
                mids = [int(x) for x in np.load(self.folder / "mids.npy")]
                self._search = ScalableMelodySearch(emb, ann, window_mid, mids, self._ref, n_candidates=self.n_candidates,
                                                    nprobe=self.nprobe)
            except Exception as e:  # noqa: BLE001 - a broken index must not break recognition
                self.error = f"{type(e).__name__}: {e}"[:300]
                self._search = None
            return self._search

    def _ref(self, mid: int):
        row = self._db().execute("SELECT notes FROM melodies WHERE mid = ?", (int(mid),)).fetchone()
        return reference_from_notes(str(mid), unpack_notes(row[0]), audio=np.zeros(1))

    def warm(self) -> bool:
        return self._load() is not None

    # ------------------------------------------------------------------ search
    def search(self, query: QueryFeatures, top_k: int = 5) -> list[MelodyMatch]:
        search = self._load()
        if search is None or len(query.contour) < 4:
            return []
        cands = search.search(query, None, max(top_k * 3, 10))
        if not cands:
            return []
        ids = [int(c.song_id) for c in cands]
        rows = {r[0]: r for r in self._db().execute(
            f"SELECT mid, key, title, artist, msd_id, selection FROM melodies WHERE mid IN ({','.join('?' * len(ids))})",
            ids).fetchall()}
        best: dict[str, MelodyMatch] = {}
        for c in cands:
            r = rows.get(int(c.song_id))
            if r is None:
                continue
            mid, key, title, artist, msd_id, selection = r
            if key in best and best[key].score >= c.score:
                continue
            best[key] = MelodyMatch(f"world:{world_song_id(title, artist)}", key, title, artist, float(c.score),
                                    c.start_time, c.end_time, mid, msd_id, selection,
                                    float(c.details.get("ann_similarity", 0.0)) if c.details else 0.0)
        if self.meta_lookup is not None:
            from melodyai.world.search import popularity_bonus

            for m in best.values():
                try:
                    ws = self.meta_lookup(m.title, m.artist)
                except Exception:  # noqa: BLE001 - metadata is optional
                    ws = None
                if ws is not None:
                    m.meta = ws.to_meta()
                    m.score = min(1.0, m.score + POPULARITY_WEIGHT / 0.03 * popularity_bonus(ws.rank))
        return sorted(best.values(), key=lambda m: (-m.score, m.key))[:top_k]
