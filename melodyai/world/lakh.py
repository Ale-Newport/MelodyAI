"""Build the world melody index (:mod:`melodyai.world.melodies`) from the Lakh MIDI dataset.

Lakh MIDI (Colin Raffel, 2016; CC-BY 4.0, https://colinraffel.com/projects/lmd/) "LMD-matched": 45,129 MIDI
files matched to 31,034 Million Song Dataset tracks. Inputs (downloaded by ``melodyai lakh-download`` into
``<store>/lakh/download``; the archive is unpacked into ``<store>/lakh/raw``):

* ``lmd_matched/<A>/<B>/<C>/<MSD track id>/<md5>.mid`` -- the MIDI files (a file may be matched to several tracks);
* ``match_scores.json`` -- ``{msd_track_id: {md5: match confidence}}``;
* ``unique_tracks.txt`` -- Million Song Dataset ``track id<SEP>song id<SEP>artist<SEP>title``.

Build steps (:func:`build_lakh_index`):

1. Group the matched tracks by :func:`melodyai.world.text.song_key` (title + primary artist): several MSD tracks
   are releases of one song. Per song, keep the MIDI files with the best match scores.
2. Extract at most ``max_melodies`` (2) vocal lines per song, in parallel: the track carrying synced lyric events
   when there is one, else the best track of the vocal-line classifier
   (:func:`melodyai.features.melody_track.rank_melody_tracks`), plus its runner-up when that is nearly as
   likely, plus the best track of the second-best MIDI file. Notes are reduced to a monophonic line (skyline);
   lines with fewer than ``MIN_NOTES`` notes, shorter than ``MIN_SECONDS`` or with a tiny pitch range are dropped.
3. A seeded sample of ``holdout`` songs is stored apart (table ``holdout``) and NOT indexed: the evaluation hums
   them as real songs that are not in the index (false-match rate).
4. Embed mid-length windows of every melody with the neural encoder, train an IVF-PQ index on a sample and add
   all windows (:class:`melodyai.indexing.ivfpq.IVFPQ`).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from melodyai.features.melody import NoteSequence
from melodyai.world.melodies import open_db, pack_notes
from melodyai.world.text import song_key

LMD_URL = "http://hog.ee.columbia.edu/craffel/lmd/"
DOWNLOADS = {"lmd_matched.tar.gz": LMD_URL + "lmd_matched.tar.gz",
             "match_scores.json": LMD_URL + "match_scores.json",
             "unique_tracks.txt": "http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt"}
MIN_NOTES = 40
MIN_SECONDS = 20.0
MIN_RANGE = 5.0               # semitones
SECOND_TRACK_MARGIN = 1.0     # logit gap under which the classifier's runner-up track is indexed too

_HOLDOUT_SCHEMA = """
CREATE TABLE IF NOT EXISTS holdout(key TEXT PRIMARY KEY, title TEXT, artist TEXT, msd_id TEXT, md5 TEXT,
                                   selection TEXT, notes BLOB NOT NULL);
"""


@dataclass
class SongFiles:
    key: str
    title: str
    artist: str
    files: list[tuple[str, str, str]]       # (md5, msd track id, path), best match score first


def load_titles(unique_tracks: Path, wanted: set[str]) -> dict[str, tuple[str, str]]:
    """MSD track id -> (artist, title) for the wanted ids."""
    out = {}
    with open(unique_tracks, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("<SEP>")
            if len(parts) == 4 and parts[0] in wanted:
                out[parts[0]] = (parts[2].strip(), parts[3].strip())
    return out


def plan_songs(raw_dir: Path, match_scores: Path, unique_tracks: Path, max_files: int = 2) -> list[SongFiles]:
    """Songs (grouped by song key) with their best-matching MIDI files, most confident songs first."""
    scores: dict[str, dict[str, float]] = json.loads(Path(match_scores).read_text())
    titles = load_titles(unique_tracks, set(scores))
    by_key: dict[str, list[tuple[float, str, str, str, str, str]]] = defaultdict(list)
    for trid, files in scores.items():
        if trid not in titles:
            continue
        artist, title = titles[trid]
        if not title or not artist:
            continue
        key = song_key(title, artist)
        if not key or key.startswith("|") or key.endswith("|"):
            continue
        folder = Path(raw_dir) / "lmd_matched" / trid[2] / trid[3] / trid[4] / trid
        for md5, sc in files.items():
            path = folder / f"{md5}.mid"
            if path.is_file():
                by_key[key].append((float(sc), md5, trid, str(path), title, artist))
    songs = []
    for key, rows in by_key.items():
        rows.sort(key=lambda r: (-r[0], r[1]))
        seen, files = set(), []
        for _sc, md5, trid, path, _, _ in rows:
            if md5 not in seen:
                seen.add(md5)
                files.append((md5, trid, path))
            if len(files) >= max_files:
                break
        songs.append((rows[0][0], SongFiles(key, rows[0][4], rows[0][5], files)))
    songs.sort(key=lambda x: (-x[0], x[1].key))
    return [s for _, s in songs]


def _line(inst) -> NoteSequence | None:
    from melodyai.catalog.importers import _skyline

    s = np.array([n.start for n in inst.notes])
    e = np.array([n.end for n in inst.notes])
    p = np.array([n.pitch for n in inst.notes], dtype=float)
    notes = _skyline(s, e, p)
    if len(notes) < MIN_NOTES or notes.end_time - notes.onsets[0] < MIN_SECONDS:
        return None
    if np.percentile(notes.pitches, 95) - np.percentile(notes.pitches, 5) < MIN_RANGE:
        return None
    return notes


def extract_song(song: SongFiles, max_melodies: int = 2) -> list[dict]:
    """Up to ``max_melodies`` vocal-line candidates of one song (see the module docstring)."""
    import warnings

    import pretty_midi

    from melodyai.features.melody_track import rank_melody_tracks, vocal_track_from_lyrics

    out: list[dict] = []
    for fi, (md5, trid, path) in enumerate(song.files):
        if len(out) >= max_melodies:
            break
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pm = pretty_midi.PrettyMIDI(path)
        except Exception:  # noqa: BLE001 - a few Lakh files are corrupt
            continue
        picks: list[tuple[int, str, float]] = []
        lyr = vocal_track_from_lyrics(pm)
        ranked = rank_melody_tracks(pm)
        if lyr is not None:
            picks.append((lyr, "lyrics", next((s for i, s in ranked if i == lyr), 0.0)))
        elif ranked:
            picks.append((ranked[0][0], "model", ranked[0][1]))
            if fi == 0 and len(ranked) > 1 and ranked[0][1] - ranked[1][1] < SECOND_TRACK_MARGIN:
                picks.append((ranked[1][0], "model-2nd", ranked[1][1]))
        for idx, why, score in picks:
            if len(out) >= max_melodies:
                break
            inst = pm.instruments[idx]
            notes = _line(inst)
            if notes is None:
                continue
            out.append({"key": song.key, "title": song.title, "artist": song.artist, "msd_id": trid, "md5": md5,
                        "track": idx, "track_name": (inst.name or "")[:60], "selection": why,
                        "track_score": round(float(score), 3), "notes": notes})
    return out


def _extract_chunk(args) -> list[dict]:
    songs, max_melodies = args
    out = []
    for s in songs:
        out.extend(extract_song(s, max_melodies))
    return out


def build_lakh_index(raw_dir: str | Path, download_dir: str | Path, out_dir: str | Path, *, workers: int = 8,
                     limit: int | None = None, holdout: int = 300, max_melodies: int = 2, seed: int = 7,
                     checkpoint: str | Path | None = None, nlist: int = 2048, m: int = 16, reembed: bool = False,
                     progress: Callable[[float, str], None] | None = None) -> dict:
    """Extract, store and index the Lakh melodies (replaces any previous index in ``out_dir``).

    ``reembed``: keep the melodies already extracted in ``out_dir`` and rebuild only the ANN index.
    """
    from melodyai.config import models_dir
    from melodyai.indexing.ivfpq import IVFPQ
    from melodyai.matching.neural import NeuralEmbedder
    from melodyai.matching.scalable import ann_window_vectors

    tick = progress or (lambda p, msg: None)
    t0 = time.perf_counter()
    raw_dir, download_dir, out_dir = Path(raw_dir), Path(download_dir), Path(out_dir)
    rng = np.random.default_rng(seed)
    if reembed and (out_dir / "melodies.db").is_file():
        # keep the extracted melodies, recompute only the embeddings / IVF-PQ index (e.g. with a new encoder)
        prev = json.loads((out_dir / "meta.json").read_text()) if (out_dir / "meta.json").is_file() else {}
        for name in ("meta.json", "window_mid.npy", "mids.npy"):
            (out_dir / name).unlink(missing_ok=True)
        conn = open_db(out_dir / "melodies.db")
        conn.executescript(_HOLDOUT_SCHEMA)
        n_mel = conn.execute("SELECT COUNT(*) FROM melodies").fetchone()[0]
        n_songs = {k for (k,) in conn.execute("SELECT DISTINCT key FROM melodies")}
        selections = defaultdict(int, dict(conn.execute("SELECT selection, COUNT(*) FROM melodies GROUP BY selection")))
        songs = [None] * int(prev.get("planned_songs") or len(n_songs))
    else:
        songs = plan_songs(raw_dir, download_dir / "match_scores.json", download_dir / "unique_tracks.txt")
        if limit:
            songs = songs[: int(limit)]
        held = set(rng.choice(len(songs), size=min(holdout, len(songs) // 10), replace=False).tolist()) if holdout else set()
        tick(0.01, f"{len(songs)} songs planned")

        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("melodies.db", "melodies.db-wal", "melodies.db-shm", "meta.json", "window_mid.npy", "mids.npy"):
            (out_dir / name).unlink(missing_ok=True)
        conn = open_db(out_dir / "melodies.db")
        conn.executescript(_HOLDOUT_SCHEMA)
        held_keys = {songs[i].key for i in held}

        chunk = 50
        jobs = [(songs[i:i + chunk], max_melodies) for i in range(0, len(songs), chunk)]
        n_mel, n_held, n_songs = 0, 0, set()
        selections: dict[str, int] = defaultdict(int)
        with ProcessPoolExecutor(max(1, workers)) as ex:
            for k, mels in enumerate(ex.map(_extract_chunk, jobs)):
                rows, hrows = [], []
                for mel in mels:
                    if mel["key"] in held_keys:
                        if mel["selection"] != "model-2nd":
                            hrows.append((mel["key"], mel["title"], mel["artist"], mel["msd_id"], mel["md5"],
                                          mel["selection"], pack_notes(mel["notes"])))
                        continue
                    notes: NoteSequence = mel["notes"]
                    rows.append((mel["key"], mel["title"], mel["artist"], mel["msd_id"], mel["md5"], mel["track"],
                                 mel["track_name"], mel["selection"], mel["track_score"], len(notes),
                                 round(notes.end_time, 2), pack_notes(notes)))
                    selections[mel["selection"]] += 1
                    n_songs.add(mel["key"])
                conn.executemany("INSERT INTO melodies(key, title, artist, msd_id, md5, track, track_name, selection, "
                                 "track_score, n_notes, duration_s, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                conn.executemany("INSERT OR IGNORE INTO holdout VALUES (?,?,?,?,?,?,?)", hrows)
                conn.commit()
                n_mel += len(rows)
                n_held += len(hrows)
                tick(0.02 + 0.5 * (k + 1) / len(jobs), f"Extracting melodies {min((k + 1) * chunk, len(songs))}/{len(songs)}")
    t_extract = time.perf_counter() - t0

    # ---- embeddings + IVF-PQ
    from melodyai.world.melodies import unpack_notes

    ckpt = Path(checkpoint) if checkpoint else models_dir() / "melody_encoder.pt"
    if not ckpt.is_absolute() and not ckpt.exists():
        ckpt = models_dir() / ckpt
    emb = NeuralEmbedder(ckpt)
    mids = [r[0] for r in conn.execute("SELECT mid FROM melodies ORDER BY mid")]

    def vectors(batch: list[int]):
        out = []
        for mid, blob in conn.execute(f"SELECT mid, notes FROM melodies WHERE mid IN ({','.join('?' * len(batch))}) "
                                      "ORDER BY mid", batch):
            v = ann_window_vectors(unpack_notes(blob)).astype(np.float32)
            out.append((mid, v))
        return out

    train_mids = [mids[i] for i in rng.permutation(len(mids))[:1500]]
    T = np.concatenate([v for _, v in vectors(train_mids)])
    T = emb.embed(T[rng.permutation(len(T))[:300_000]], is_query=False)
    ann = IVFPQ(T.shape[1], nlist=min(nlist, max(16, len(T) // 40)), m=m, nprobe=32)
    ann.train(T)
    tick(0.6, "Index trained")
    window_mid: list[np.ndarray] = []
    next_wid = 0
    step = 400
    mid_pos = {mid: i for i, mid in enumerate(mids)}
    for b in range(0, len(mids), step):
        for mid, v in vectors(mids[b:b + step]):
            if not len(v):
                continue
            E = emb.embed(v, is_query=False)
            ann.add(E, np.arange(next_wid, next_wid + len(v)))
            window_mid.append(np.full(len(v), mid_pos[mid], np.int32))
            next_wid += len(v)
        tick(0.6 + 0.38 * min(1.0, (b + step) / max(1, len(mids))), f"Embedding melodies {min(b + step, len(mids))}/{len(mids)}")
    ann.consolidate()
    ann.save(out_dir / "ann")
    np.save(out_dir / "window_mid.npy", np.concatenate(window_mid) if window_mid else np.zeros(0, np.int32))
    np.save(out_dir / "mids.npy", np.array(mids, dtype=np.int64))
    n_held = conn.execute("SELECT COUNT(*) FROM holdout").fetchone()[0]
    meta = {"songs": len(n_songs), "melodies": n_mel, "windows": int(next_wid), "holdout_songs": n_held,
            "planned_songs": len(songs), "selections": dict(selections), "encoder": emb.id, "checkpoint": ckpt.name,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "extract_s": round(t_extract, 1), "elapsed_s": round(time.perf_counter() - t0, 1),
            "ann_mb": round(ann.nbytes / 1e6, 1), "source": "Lakh MIDI LMD-matched (CC-BY 4.0)"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    conn.close()
    tick(1.0, f"Indexed {n_mel} melodies of {len(n_songs)} songs")
    return meta
