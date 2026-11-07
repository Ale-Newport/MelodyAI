"""Catalog service: import, incremental indexing, deletion, reindexing and status.

Storage layout under ``store_dir``::

    catalog.db                      SQLite metadata (songs, lyrics, recognitions, jobs)
    lyrics_fts.db                   SQLite FTS5 lyrics index
    media/<song_id>/reference.*     reference file as imported (MIDI or audio)
    features/<FEATURE_VERSION>/<song_id>.npz   extracted melody features

Feature files are versioned: when ``FEATURE_VERSION`` changes, songs indexed with the
previous version are flagged ``stale`` at startup and can be re-indexed incrementally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select

from melodyai.audio.io import AudioError, decode_audio
from melodyai.catalog.importers import (
    AUDIO_EXTENSIONS,
    MIDI_EXTENSIONS,
    ImportErrorMelody,
    load_midi_melody,
)
from melodyai.catalog.models import Lyrics, Song, make_session_factory, new_id
from melodyai.config import ANALYSIS_SR, FEATURE_VERSION, store_dir
from melodyai.features.extract import (
    REFERENCE_TYPES,
    ReferenceFeatures,
    reference_from_audio,
    reference_from_notes,
)
from melodyai.fingerprint.index import FingerprintIndex, fingerprint_audio
from melodyai.indexing.melody_index import MelodyIndex
from melodyai.lyrics.search import LyricsIndex

log = logging.getLogger(__name__)

MAX_REFERENCE_BYTES = 60 * 1024 * 1024
MAX_REFERENCE_SECONDS = 15 * 60


class DuplicateSongError(ValueError):
    def __init__(self, existing_id: str) -> None:
        super().__init__(f"Duplicate reference file (already imported as song {existing_id})")
        self.existing_id = existing_id


class NotFoundError(KeyError):
    pass


def song_to_dict(s: Song) -> dict:
    return {
        "id": s.id, "title": s.title, "artist": s.artist, "album": s.album, "year": s.year,
        "language": s.language, "duration_s": s.duration_s, "reference_type": s.reference_type,
        "reference_filename": s.reference_filename, "composition_id": s.composition_id, "source": s.source,
        "license": s.license, "content_hash": s.content_hash, "index_status": s.index_status,
        "index_error": s.index_error, "feature_version": s.feature_version, "n_notes": s.n_notes,
        "n_windows": s.n_windows, "listen_url": s.listen_url, "has_lyrics": s.lyrics is not None,
        "lyrics_language": s.lyrics.language if s.lyrics else None, "extra": s.extra,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


@dataclass
class ComputedSong:
    """Everything indexing needs for one song; computable in a worker process."""
    ref: ReferenceFeatures
    peaks: object | None = None
    hashes: object | None = None
    duration_s: float | None = None


def compute_song(song_id: str, reference_type: str, data: bytes, extra: dict | None = None,
                 preview: bytes | None = None) -> ComputedSong:
    """Melody features + audio fingerprint of a reference (pure function, no catalog state)."""
    extra = extra or {}
    if reference_type == "midi":
        mel = load_midi_melody(data, extra.get("midi_track"))
        samples = decode_audio(preview).samples if preview else None
        ref = reference_from_notes(song_id, mel.notes, audio=samples, reference_type="midi")
    else:
        audio = decode_audio(data, ANALYSIS_SR, max_bytes=MAX_REFERENCE_BYTES, max_duration=MAX_REFERENCE_SECONDS)
        samples = audio.samples
        ref = reference_from_audio(song_id, audio, reference_type)
    if len(ref.notes) < 3:
        raise ImportErrorMelody("No usable melody could be extracted from the reference")
    peaks = hashes = None
    if samples is not None and len(samples) >= ANALYSIS_SR:
        peaks, hashes = fingerprint_audio(samples, ANALYSIS_SR)
    return ComputedSong(ref, peaks, hashes, len(samples) / ANALYSIS_SR if samples is not None else None)


class Catalog:
    def __init__(self, root: Path | str | None = None, database_url: str | None = None) -> None:
        self.root = Path(root or store_dir())
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "media").mkdir(exist_ok=True)
        self.features_dir = self.root / "features" / FEATURE_VERSION
        self.features_dir.mkdir(parents=True, exist_ok=True)
        url = database_url or os.environ.get("MELODYAI_DATABASE_URL") or f"sqlite:///{self.root / 'catalog.db'}"
        self.Session = make_session_factory(url)
        self.melody_index = MelodyIndex()
        self.lyrics_index = LyricsIndex(str(self.root / "lyrics_fts.db"))
        self.fp_index = FingerprintIndex(self.root / "fingerprints")
        self.stats = {"features_computed": 0, "features_loaded": 0, "fingerprints_computed": 0}
        self._lock = threading.RLock()
        self._on_add: list[Callable[[ReferenceFeatures], None]] = []
        self._on_remove: list[Callable[[str], None]] = []
        self.load()

    # ------------------------------------------------------------------ listeners
    def subscribe(self, on_add: Callable[[ReferenceFeatures], None], on_remove: Callable[[str], None]) -> None:
        """Dependent indexes (e.g. neural embeddings) follow catalog changes through these hooks."""
        self._on_add.append(on_add)
        self._on_remove.append(on_remove)

    # ------------------------------------------------------------------ loading
    def feature_path(self, song_id: str) -> Path:
        return self.features_dir / f"{song_id}.npz"

    def load(self) -> None:
        """Load stored features of indexed songs; flag songs from older feature versions as stale."""
        with self.Session() as db:
            songs = db.scalars(select(Song)).all()
            lyric_ids = set(db.scalars(select(Lyrics.song_id)).all())
            for s in songs:
                if s.index_status == "indexed" and s.feature_version != FEATURE_VERSION:
                    s.index_status = "stale"
                    continue
                if s.index_status != "indexed":
                    continue
                p = self.feature_path(s.id)
                if not p.exists():
                    s.index_status = "stale"
                    continue
                self.melody_index.add(ReferenceFeatures.load(p))
                self.stats["features_loaded"] += 1
            db.commit()
            # Lyrics FTS is persistent; make sure it matches the DB (e.g. after manual edits).
            for sid in list(self.lyrics_index.docs):
                if sid not in lyric_ids:
                    self.lyrics_index.remove(sid)
            for ly in db.scalars(select(Lyrics)).all():
                if ly.song_id not in self.lyrics_index:
                    self.lyrics_index.add(ly.song_id, ly.text, ly.language)
            indexed = {s.id: s.reference_type for s in songs if s.index_status == "indexed"}
        # Fingerprint index: drop deleted songs, back-fill songs indexed before fingerprints existed
        # (or whose unflushed delta was lost when the process stopped).
        for sid in [x for x in self.fp_index.slot if x not in indexed]:
            self.fp_index.remove(sid)
        for sid, rtype in indexed.items():
            if sid not in self.fp_index:
                self._fingerprint(sid, rtype)
        self.fp_index.flush()

    def audio_for_fingerprint(self, song_id: str, reference_type: str):
        """The recording to fingerprint: the reference audio, or the rendered preview of a MIDI."""
        try:
            path = self.listen_file(song_id) if reference_type == "midi" else self.reference_file(song_id)
        except NotFoundError:
            return None
        if path is None:
            return None
        try:
            return decode_audio(path.read_bytes(), ANALYSIS_SR, max_bytes=MAX_REFERENCE_BYTES,
                                max_duration=MAX_REFERENCE_SECONDS).samples
        except AudioError:
            return None

    def _fingerprint(self, song_id: str, reference_type: str, samples=None) -> int:
        samples = samples if samples is not None else self.audio_for_fingerprint(song_id, reference_type)
        if samples is None or len(samples) < ANALYSIS_SR:
            return 0
        peaks, hashes = fingerprint_audio(samples, ANALYSIS_SR)
        self.fp_index.add(song_id, hashes, len(samples) / ANALYSIS_SR, peaks)
        self.stats["fingerprints_computed"] += 1
        return len(hashes)

    def flush_indexes(self) -> None:
        """Persist in-memory index deltas (called after batch operations and at shutdown)."""
        self.fp_index.flush()

    # ------------------------------------------------------------------ queries
    def list_songs(self, q: str | None = None, status: str | None = None, offset: int = 0,
                   limit: int = 200) -> tuple[list[dict], int]:
        with self.Session() as db:
            stmt = select(Song)
            if q:
                like = f"%{q.lower()}%"
                stmt = stmt.where(func.lower(Song.title).like(like) | func.lower(Song.artist).like(like)
                                  | func.lower(func.coalesce(Song.album, "")).like(like))
            if status:
                stmt = stmt.where(Song.index_status == status)
            total = db.scalar(select(func.count()).select_from(stmt.subquery()))
            rows = db.scalars(stmt.order_by(Song.created_at, Song.title).offset(offset).limit(limit)).all()
            return [song_to_dict(s) for s in rows], int(total or 0)

    def get_song(self, song_id: str) -> dict:
        with self.Session() as db:
            s = db.get(Song, song_id)
            if s is None:
                raise NotFoundError(song_id)
            d = song_to_dict(s)
            d["lyrics"] = {"text": s.lyrics.text, "language": s.lyrics.language, "source": s.lyrics.source,
                           "license": s.lyrics.license} if s.lyrics else None
            return d

    def songs_by_ids(self, ids: list[str]) -> dict[str, dict]:
        with self.Session() as db:
            rows = db.scalars(select(Song).where(Song.id.in_(ids))).all() if ids else []
            return {s.id: song_to_dict(s) for s in rows}

    def reference_file(self, song_id: str) -> Path:
        with self.Session() as db:
            s = db.get(Song, song_id)
            if s is None:
                raise NotFoundError(song_id)
            folder = self.root / "media" / s.id
            files = sorted(folder.glob("reference.*")) if folder.exists() else []
            if not files:
                raise NotFoundError(f"no reference file for {song_id}")
            return files[0]

    def listen_file(self, song_id: str) -> Path | None:
        """Audio rendering for playback (reference audio, or the synthesised preview of a MIDI)."""
        folder = self.root / "media" / song_id
        for name in ("preview.wav", "reference.wav", "reference.mp3", "reference.flac", "reference.ogg",
                     "reference.m4a"):
            if (folder / name).exists():
                return folder / name
        return None

    def status(self) -> dict:
        with self.Session() as db:
            rows = db.execute(select(Song.index_status, func.count()).group_by(Song.index_status)).all()
            n_lyrics = db.scalar(select(func.count()).select_from(Lyrics))
        by_status = {k: v for k, v in rows}
        return {
            "songs": sum(by_status.values()), "by_status": by_status, "with_lyrics": int(n_lyrics or 0),
            "feature_version": FEATURE_VERSION, "melody_index": {
                "songs": len(self.melody_index), "windows": len(self.melody_index.matrix()),
                "window_lengths": list(self.melody_index.lengths), "version": self.melody_index.version},
            "lyrics_index": {"songs": len(self.lyrics_index)}, "fingerprint_index": self.fp_index.stats(),
            "stats": dict(self.stats),
        }

    def find_duplicates(self) -> list[dict]:
        """Possible duplicates by normalised title+artist or by identical composition id."""
        groups: dict[tuple, list[str]] = {}
        with self.Session() as db:
            for s in db.scalars(select(Song)).all():
                groups.setdefault(("title", s.title.strip().lower(), (s.artist or "").strip().lower()), []).append(s.id)
                if s.composition_id:
                    groups.setdefault(("composition", s.composition_id), []).append(s.id)
        return [{"key": list(k), "song_ids": v} for k, v in groups.items() if len(v) > 1]

    # ------------------------------------------------------------------ mutations
    def add_song(self, meta: dict, reference: bytes, filename: str, lyrics: dict | None = None,
                 index: bool = True, preview_wav: bytes | None = None, precomputed: ComputedSong | None = None) -> dict:
        ext = Path(filename).suffix.lower()
        if ext not in MIDI_EXTENSIONS | AUDIO_EXTENSIONS:
            raise ImportErrorMelody(f"Unsupported reference format '{ext}'")
        if not reference:
            raise ImportErrorMelody("Empty reference file")
        if len(reference) > MAX_REFERENCE_BYTES:
            raise ImportErrorMelody("Reference file too large")
        rtype = meta.get("reference_type") or ("midi" if ext in MIDI_EXTENSIONS else "monophonic_audio")
        if rtype not in REFERENCE_TYPES:
            raise ImportErrorMelody(f"reference_type must be one of {REFERENCE_TYPES}")
        if (ext in MIDI_EXTENSIONS) != (rtype == "midi"):
            raise ImportErrorMelody("reference_type 'midi' requires a .mid file (and vice versa)")
        digest = hashlib.sha256(reference).hexdigest()
        # Validate the file before storing anything (precomputed features imply it was decoded already).
        if precomputed is None:
            if ext in MIDI_EXTENSIONS:
                load_midi_melody(reference, meta.get("midi_track"))
            else:
                try:
                    decode_audio(reference, ANALYSIS_SR, max_bytes=MAX_REFERENCE_BYTES,
                                 max_duration=MAX_REFERENCE_SECONDS)
                except AudioError as e:
                    raise ImportErrorMelody(str(e)) from e
        with self._lock, self.Session() as db:
            existing = db.scalar(select(Song).where(Song.content_hash == digest))
            if existing is not None:
                raise DuplicateSongError(existing.id)
            sid = meta.get("id") or new_id()
            folder = self.root / "media" / sid
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"reference{ext}").write_bytes(reference)
            if preview_wav:
                (folder / "preview.wav").write_bytes(preview_wav)
            extra = {k: v for k, v in meta.items() if k in ("midi_track", "sections", "generator")}
            song = Song(
                id=sid, title=meta.get("title") or Path(filename).stem, artist=meta.get("artist") or "Unknown",
                album=meta.get("album"), year=meta.get("year"), language=meta.get("language"),
                reference_type=rtype, reference_filename=Path(filename).name[:300],
                composition_id=meta.get("composition_id"), source=meta.get("source"),
                license=meta.get("license"), content_hash=digest, index_status="pending",
                listen_url=meta.get("listen_url"), extra_json=json.dumps(extra),
            )
            db.add(song)
            if lyrics and lyrics.get("text"):
                db.add(Lyrics(song_id=sid, language=lyrics.get("language") or meta.get("language") or "en",
                              text=lyrics["text"], source=lyrics.get("source") or meta.get("source"),
                              license=lyrics.get("license") or meta.get("license")))
            db.commit()
        if lyrics and lyrics.get("text"):
            self.lyrics_index.add(sid, lyrics["text"], lyrics.get("language") or meta.get("language") or "en")
        if precomputed is not None:
            self._store_computed(sid, precomputed)
        elif index:
            self.index_song(sid)
        return self.get_song(sid)

    def known_hashes(self) -> set[str]:
        with self.Session() as db:
            return set(db.scalars(select(Song.content_hash)).all())

    def index_song(self, song_id: str) -> dict:
        """Compute features for ONE song and add it to the indexes (incremental)."""
        with self.Session() as db:
            s = db.get(Song, song_id)
            if s is None:
                raise NotFoundError(song_id)
            s.index_status = "processing"
            db.commit()
            rtype, extra = s.reference_type, s.extra
        try:
            preview = self.listen_file(song_id) if rtype == "midi" else None
            comp = compute_song(song_id, rtype, self.reference_file(song_id).read_bytes(), extra,
                                preview.read_bytes() if preview else None)
            self._store_computed(song_id, comp)
        except Exception as e:
            log.exception("indexing failed for %s", song_id)
            with self.Session() as db:
                s = db.get(Song, song_id)
                if s is not None:
                    s.index_status, s.index_error = "error", str(e)[:2000]
                    db.commit()
        return self.get_song(song_id)

    def _store_computed(self, song_id: str, comp: ComputedSong) -> None:
        ref = comp.ref
        ref.song_id = song_id
        if comp.hashes is not None:
            self.fp_index.add(song_id, comp.hashes, comp.duration_s, comp.peaks)
            self.stats["fingerprints_computed"] += 1
        ref.meta = {**ref.meta, "fingerprint_hashes": len(comp.hashes) if comp.hashes is not None else 0}
        ref.save(self.feature_path(song_id))
        self.stats["features_computed"] += 1
        n_windows = self.melody_index.add(ref)
        for cb in self._on_add:
            cb(ref)
        with self.Session() as db:
            s = db.get(Song, song_id)
            s.index_status, s.index_error = "indexed", None
            s.feature_version, s.n_notes, s.n_windows = FEATURE_VERSION, len(ref.notes), n_windows
            s.duration_s = round(float(ref.notes.end_time), 3)
            if ref.meta:
                s.extra_json = json.dumps({**s.extra, **ref.meta})
            db.commit()

    def reindex(self, song_ids: list[str] | None = None, only_needed: bool = True,
                progress: Callable[[float, str], None] | None = None) -> dict:
        with self.Session() as db:
            stmt = select(Song.id, Song.index_status)
            if song_ids:
                stmt = stmt.where(Song.id.in_(song_ids))
            rows = db.execute(stmt).all()
        todo = [sid for sid, st in rows if not only_needed or st in ("pending", "stale", "error") or song_ids]
        done = 0
        for i, sid in enumerate(todo):
            self.index_song(sid)
            done += 1
            if progress:
                progress((i + 1) / len(todo), f"indexed {sid}")
        self.flush_indexes()
        return {"reindexed": done, "requested": len(todo)}

    def update_song(self, song_id: str, fields: dict) -> dict:
        allowed = {"title", "artist", "album", "year", "language", "composition_id", "source", "license",
                   "listen_url"}
        with self.Session() as db:
            s = db.get(Song, song_id)
            if s is None:
                raise NotFoundError(song_id)
            for k, v in fields.items():
                if k in allowed:
                    setattr(s, k, v)
            db.commit()
        return self.get_song(song_id)

    def set_lyrics(self, song_id: str, text: str | None, language: str = "en", source: str | None = None,
                   license: str | None = None) -> dict:
        with self.Session() as db:
            s = db.get(Song, song_id)
            if s is None:
                raise NotFoundError(song_id)
            if s.lyrics is not None:
                db.delete(s.lyrics)
                db.flush()
            if text and text.strip():
                db.add(Lyrics(song_id=song_id, language=language, text=text, source=source, license=license))
            db.commit()
        if text and text.strip():
            self.lyrics_index.add(song_id, text, language)
        else:
            self.lyrics_index.remove(song_id)
        return self.get_song(song_id)

    def delete_song(self, song_id: str) -> None:
        with self._lock, self.Session() as db:
            s = db.get(Song, song_id)
            if s is None:
                raise NotFoundError(song_id)
            db.delete(s)
            db.commit()
        self.melody_index.remove(song_id)
        self.lyrics_index.remove(song_id)
        self.fp_index.remove(song_id)
        for cb in self._on_remove:
            cb(song_id)
        self.feature_path(song_id).unlink(missing_ok=True)
        shutil.rmtree(self.root / "media" / song_id, ignore_errors=True)
