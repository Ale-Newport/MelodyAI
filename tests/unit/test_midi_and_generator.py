import io
import json

import numpy as np
import pretty_midi
import pytest

from melodyai.catalog.importers import (
    ImportErrorMelody,
    load_midi_melody,
    notes_to_midi_bytes,
    parse_lyrics,
    parse_metadata,
)
from melodyai.demo.generator import _window_signatures, generate_catalog, generate_melody, too_similar
from melodyai.features.melody import NoteSequence


def test_midi_roundtrip():
    notes = NoteSequence.from_list([(0, 0.4, 60), (0.5, 0.4, 64), (1.0, 0.9, 67), (2.0, 0.4, 72)])
    mel = load_midi_melody(notes_to_midi_bytes(notes))
    assert mel.notes.pitches.tolist() == [60, 64, 67, 72]
    assert np.allclose(mel.notes.onsets, notes.onsets, atol=0.01)


def _multi_track_midi():
    pm = pretty_midi.PrettyMIDI()
    chords = pretty_midi.Instrument(0, name="Piano")
    for k in range(8):
        for p in (48, 52, 55):
            chords.notes.append(pretty_midi.Note(80, p, k * 1.0, k * 1.0 + 0.9))
    lead = pretty_midi.Instrument(73, name="Lead Flute")
    for k, p in enumerate([72, 74, 76, 79, 76, 74, 72, 71]):
        lead.notes.append(pretty_midi.Note(90, p, k * 1.0, k * 1.0 + 0.5))
    drums = pretty_midi.Instrument(0, is_drum=True)
    drums.notes.append(pretty_midi.Note(100, 36, 0, 0.1))
    pm.instruments += [chords, lead, drums]
    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()


def test_midi_melody_track_selection_and_skyline():
    data = _multi_track_midi()
    mel = load_midi_melody(data)
    assert mel.track_name == "Lead Flute"
    assert mel.notes.pitches.tolist() == [72, 74, 76, 79, 76, 74, 72, 71]
    chords = load_midi_melody(data, track=0)  # skyline keeps the top note of each chord
    assert set(chords.notes.pitches.tolist()) == {55}


def test_invalid_midi():
    with pytest.raises(ImportErrorMelody):
        load_midi_melody(b"MThd garbage")


def test_lyrics_and_metadata_parsers():
    assert parse_lyrics(b"la la\nlo", "a.txt")["text"] == "la la\nlo"
    j = parse_lyrics(json.dumps({"lines": ["uno", "dos"], "language": "es"}).encode(), "a.json")
    assert j == {"language": "es", "text": "uno\ndos"}
    with pytest.raises(ImportErrorMelody):
        parse_lyrics(b"   ", "a.txt")
    rows = parse_metadata(b"title,artist,year,reference_file\nSong,Me,2020,a.mid\n", "m.csv")
    assert rows == [{"title": "Song", "artist": "Me", "year": 2020, "reference_file": "a.mid"}]
    with pytest.raises(ImportErrorMelody):
        parse_metadata(b'[{"artist": "x"}]', "m.json")


def test_generator_is_deterministic_and_singable():
    a, b = generate_melody(42), generate_melody(42)
    assert np.array_equal(a.notes.pitches, b.notes.pitches) and np.array_equal(a.notes.onsets, b.notes.onsets)
    assert not np.array_equal(generate_melody(43).notes.pitches, a.notes.pitches)
    for m in generate_catalog(10):
        assert np.ptp(m.notes.pitches) <= 26  # about two octaves at most
        assert any(s["name"] == "chorus" for s in m.sections)


def test_catalog_melodies_are_mutually_distinct():
    cat = generate_catalog(16)
    sigs = [_window_signatures(m.notes) for m in cat]
    for i in range(len(cat)):
        for j in range(i + 1, len(cat)):
            assert not too_similar(sigs[i], sigs[j])
