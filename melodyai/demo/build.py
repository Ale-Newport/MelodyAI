"""Build the reproducible demo catalog (MIDI + synthesised audio + metadata + lyrics).

    python -m melodyai.cli demo-data            # generate files under data/demo and import them

Files written to ``data/demo``::

    catalog.json        manifest (metadata, generator parameters, sections, file names)
    midi/<id>.mid       reference melodies
    audio/<id>.wav      piano renderings (22.05 kHz) for listening
    lyrics/<id>.json    original lyrics (20 of the 48 songs)
    checksums.json      SHA-256 of every MIDI file (reproducibility check)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from melodyai.audio.io import wav_bytes
from melodyai.audio.synth import render
from melodyai.catalog.importers import notes_to_midi_bytes
from melodyai.config import data_dir
from melodyai.demo.generator import ARTISTS, generate_catalog, title_for
from melodyai.demo.lyrics_bank import LYRICS

DEMO_SEED = 1000
DEMO_SIZE = 48
LISTEN_SR = 22050
# Lyrics are attached to every other song for the first 40 songs, so songs with and
# without lyrics are interleaved across the catalog.
LYRIC_SLOTS = list(range(0, 40, 2))


def demo_dir() -> Path:
    return data_dir() / "demo"


def build_demo_files(out: Path | None = None, n: int = DEMO_SIZE, with_audio: bool = True) -> dict:
    out = out or demo_dir()
    for sub in ("midi", "audio", "lyrics"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    melodies = generate_catalog(n, DEMO_SEED)
    songs, checksums, used_titles = [], {}, set()
    for i, m in enumerate(melodies):
        sid = f"demo-{i:02d}"
        lyr = LYRICS[LYRIC_SLOTS.index(i)] if i in LYRIC_SLOTS else None
        title = lyr["title"] if lyr else title_for(i, m.seed)
        bump = 0
        while title in used_titles:  # keep titles unique
            bump += 1
            title = title_for(i, m.seed + bump)
        used_titles.add(title)
        midi = notes_to_midi_bytes(m.notes, tempo=m.tempo, name="melody")
        (out / "midi" / f"{sid}.mid").write_bytes(midi)
        checksums[sid] = hashlib.sha256(midi).hexdigest()
        if with_audio and not (out / "audio" / f"{sid}.wav").exists():
            y = render(m.notes, "piano", sr=LISTEN_SR, seed=m.seed)
            (out / "audio" / f"{sid}.wav").write_bytes(wav_bytes(y, LISTEN_SR))
        if lyr:
            (out / "lyrics" / f"{sid}.json").write_text(json.dumps(
                {"language": lyr["language"], "text": lyr["text"], "source": "MelodyAI original lyrics",
                 "license": "CC0-1.0"}, ensure_ascii=False, indent=1))
        songs.append({
            "id": sid, "title": title, "artist": ARTISTS[i % len(ARTISTS)], "album": "MelodyAI Demo Catalog",
            "year": 2026, "language": lyr["language"] if lyr else None, "composition_id": f"melodyai-demo-{m.seed}",
            "source": f"MelodyAI procedural generator (seed={m.seed})", "license": "CC0-1.0",
            "reference_type": "midi", "reference_file": f"midi/{sid}.mid", "audio_file": f"audio/{sid}.wav",
            "lyrics_file": f"lyrics/{sid}.json" if lyr else None,
            "generator": {"seed": m.seed, "key": m.key, "scale": m.scale, "meter": m.meter, "tempo": m.tempo,
                          "form": m.form},
            "sections": m.sections, "n_notes": len(m.notes), "duration_s": round(m.duration, 2),
        })
    manifest = {"name": "MelodyAI demo catalog", "version": 1, "base_seed": DEMO_SEED, "size": n,
                "license": "CC0-1.0", "note": "Original procedurally generated melodies and original lyrics.",
                "songs": songs}
    (out / "catalog.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    (out / "checksums.json").write_text(json.dumps(checksums, indent=1))
    return manifest


def import_demo(catalog, out: Path | None = None, progress=None) -> dict:
    """Import the demo files into a :class:`melodyai.catalog.service.Catalog` (idempotent)."""
    from melodyai.catalog.service import DuplicateSongError

    out = out or demo_dir()
    manifest = json.loads((out / "catalog.json").read_text())
    added = skipped = 0
    for k, s in enumerate(manifest["songs"]):
        lyrics = json.loads((out / s["lyrics_file"]).read_text()) if s.get("lyrics_file") else None
        audio = out / s["audio_file"]
        meta = {k2: s.get(k2) for k2 in ("id", "title", "artist", "album", "year", "language", "composition_id",
                                          "source", "license", "reference_type", "sections", "generator")}
        try:
            catalog.add_song(meta, (out / s["reference_file"]).read_bytes(), Path(s["reference_file"]).name,
                             lyrics=lyrics, preview_wav=audio.read_bytes() if audio.exists() else None)
            added += 1
        except DuplicateSongError:
            skipped += 1
        if progress:
            progress((k + 1) / len(manifest["songs"]), s["title"])
    return {"added": added, "skipped": skipped, "total": len(manifest["songs"])}
