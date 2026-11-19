"""Hostile or malformed uploads are refused cheaply instead of exhausting memory or returning 500."""

import io
import json

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from melodyai.audio.io import AudioError, decode_audio
from melodyai.catalog.importers import ImportErrorMelody, parse_lyrics, parse_metadata
from melodyai.catalog.service import compute_song


def _wav(sr: int, seconds: float) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(sr * seconds), dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_long_audio_is_refused_from_its_header_before_decoding():
    data = _wav(4000, 200)                         # 1.6 MB of PCM, 200 s
    with pytest.raises(AudioError, match="too long"):
        decode_audio(data, max_duration=60)


def test_tiny_sample_rate_is_refused():
    with pytest.raises(AudioError, match="Sample rate too low"):
        decode_audio(_wav(10, 200), max_duration=None)


def test_audio_within_limits_still_decodes():
    audio = decode_audio(_wav(22050, 2.0), max_duration=60)
    assert abs(audio.duration - 2.0) < 0.01


def test_midi_reference_longer_than_the_limit_is_refused():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for i, pitch in enumerate((60, 62, 64, 65)):
        inst.notes.append(pretty_midi.Note(velocity=96, pitch=pitch, start=i * 0.5, end=i * 0.5 + 0.4))
    inst.notes.append(pretty_midi.Note(velocity=96, pitch=67, start=3600.0, end=3600.5))   # one note at 1 h
    pm.instruments.append(inst)
    buf = io.BytesIO()
    pm.write(buf)
    with pytest.raises(ImportErrorMelody, match="too long"):
        compute_song("x", "midi", buf.getvalue())


@pytest.mark.parametrize("payload", ["[]", '{"text": 5}', '{"lines": [1, 2]}', '{"lines": "abc"}'])
def test_malformed_lyrics_json_is_an_import_error(payload):
    with pytest.raises(ImportErrorMelody):
        parse_lyrics(payload.encode(), "lyrics.json")


def test_valid_lyrics_json():
    out = parse_lyrics(json.dumps({"lines": ["one", "two"], "language": "en"}).encode(), "l.json")
    assert out == {"text": "one\ntwo", "language": "en"}


@pytest.mark.parametrize("payload", ['["a"]', '{"songs": 5}', "{not json"])
def test_malformed_manifest_json_is_an_import_error(payload):
    with pytest.raises(ImportErrorMelody):
        parse_metadata(payload.encode(), "manifest.json")
