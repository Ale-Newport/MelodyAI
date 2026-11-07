"""Importers: MIDI, audio, lyrics (TXT/JSON) and bulk metadata (CSV/JSON)."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass

import numpy as np

from melodyai.features.melody import NoteSequence

MIDI_EXTENSIONS = {".mid", ".midi"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".mp4", ".aac", ".webm"}
METADATA_FIELDS = ("title", "artist", "album", "year", "language", "composition_id", "source", "license",
                   "reference_type", "reference_file", "lyrics_file", "listen_url")


class ImportErrorMelody(ValueError):
    pass


@dataclass
class MidiMelody:
    notes: NoteSequence
    track_index: int
    track_name: str
    n_tracks: int


def _skyline(starts, ends, pitches) -> NoteSequence:
    order = np.lexsort((-pitches, starts))
    out: list[list[float]] = []
    for k in order:
        s, e, p = float(starts[k]), float(ends[k]), float(pitches[k])
        if out and s < out[-1][0] + out[-1][1] - 1e-3:
            last = out[-1]
            if p > last[2]:
                last[1] = s - last[0]
                if last[1] <= 0.03:
                    out.pop()
                out.append([s, e - s, p])
            continue
        out.append([s, e - s, p])
    return NoteSequence.from_list([tuple(n) for n in out if n[1] > 0.03])


def load_midi_melody(data: bytes, track: int | None = None) -> MidiMelody:
    """Extract a monophonic melody from a MIDI file.

    Track selection: explicit ``track`` index, else a non-drum track whose name suggests
    a melody ("melody", "lead", "vocal", "voice"), else the non-drum track with the highest
    median pitch among tracks with >= 8 notes. Overlapping notes are reduced with the
    skyline rule.
    """
    import pretty_midi

    try:
        pm = pretty_midi.PrettyMIDI(io.BytesIO(data))
    except Exception as e:
        raise ImportErrorMelody(f"Invalid MIDI file: {e}") from e
    insts = [(i, inst) for i, inst in enumerate(pm.instruments) if not inst.is_drum and inst.notes]
    if not insts:
        raise ImportErrorMelody("MIDI file contains no pitched notes")
    if track is not None:
        chosen = [x for x in insts if x[0] == track]
        if not chosen:
            raise ImportErrorMelody(f"Track {track} not found or empty")
        idx, inst = chosen[0]
    else:
        named = [x for x in insts if any(k in (x[1].name or "").lower() for k in ("melody", "lead", "vocal", "voice"))]
        if named:
            idx, inst = named[0]
        else:
            rich = [x for x in insts if len(x[1].notes) >= 8] or insts
            idx, inst = max(rich, key=lambda x: np.median([n.pitch for n in x[1].notes]))
    starts = np.array([n.start for n in inst.notes])
    ends = np.array([n.end for n in inst.notes])
    pitches = np.array([n.pitch for n in inst.notes], dtype=float)
    notes = _skyline(starts, ends, pitches)
    if len(notes) < 3:
        raise ImportErrorMelody("Melody track has fewer than 3 notes")
    return MidiMelody(notes, idx, inst.name or f"track {idx}", len(pm.instruments))


def notes_to_midi_bytes(notes: NoteSequence, program: int = 0, tempo: float = 120.0, name: str = "melody") -> bytes:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    inst = pretty_midi.Instrument(program=program, name=name)
    for o, d, p in zip(notes.onsets, notes.durations, notes.pitches):
        inst.notes.append(pretty_midi.Note(velocity=96, pitch=int(round(p)), start=float(o), end=float(o + d)))
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def parse_lyrics(data: bytes, filename: str = "lyrics.txt") -> dict:
    """Return {"text", "language"?, "source"?, "license"?} from TXT or structured JSON.

    JSON formats accepted: {"text": "..."} or {"lines": ["...", ...]}, optional
    "language", "source", "license".
    """
    try:
        raw = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ImportErrorMelody("Lyrics must be UTF-8 text") from e
    if filename.lower().endswith(".json"):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ImportErrorMelody(f"Invalid lyrics JSON: {e}") from e
        text = obj.get("text") or "\n".join(obj.get("lines", []))
        out = {k: obj[k] for k in ("language", "source", "license") if k in obj}
        out["text"] = text
    else:
        out = {"text": raw}
    if not out["text"].strip():
        raise ImportErrorMelody("Lyrics are empty")
    if len(out["text"]) > 50_000:
        raise ImportErrorMelody("Lyrics too long (max 50k characters)")
    return out


def parse_metadata(data: bytes, filename: str) -> list[dict]:
    """Parse a CSV or JSON manifest into a list of song metadata dicts."""
    raw = data.decode("utf-8-sig")
    if filename.lower().endswith(".json"):
        obj = json.loads(raw)
        rows = obj["songs"] if isinstance(obj, dict) else obj
    else:
        rows = list(csv.DictReader(io.StringIO(raw)))
    out = []
    for r in rows:
        row = {k: (str(v).strip() if v is not None else None) for k, v in r.items() if k in METADATA_FIELDS}
        if not row.get("title"):
            raise ImportErrorMelody("Every metadata row needs a 'title'")
        if row.get("year"):
            try:
                row["year"] = int(row["year"])
            except ValueError:
                row["year"] = None
        out.append(row)
    return out
