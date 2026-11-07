import numpy as np
import pytest

from melodyai.audio.synth import render
from melodyai.features.melody import NoteSequence, fold_octave_jumps
from melodyai.pitch.extractors import EXTRACTORS, get_extractor
from melodyai.pitch.notes import segment_notes, track_to_contour

SR = 16000
MELODY = [60, 62, 64, 67, 67, 65, 64, 62, 60, 72, 71]


def melody_notes(pitches=MELODY, ioi=0.4):
    return NoteSequence.from_list([(i * ioi, ioi * 0.95, p) for i, p in enumerate(pitches)])


def test_pyin_pure_tone():
    t = np.arange(SR) / SR
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    tr = get_extractor("pyin").extract(y, SR, "hum")
    assert tr.voiced_ratio > 0.8
    assert np.nanmedian(tr.midi) == pytest.approx(69.0, abs=0.1)


@pytest.mark.parametrize("timbre,profile,shift", [("hum", "hum", 0), ("voice", "hum", -5), ("whistle", "whistle", 24)])
def test_segmentation_recovers_melody(timbre, profile, shift):
    notes = melody_notes()
    y = render(notes.transpose(shift), timbre, sr=SR, seed=3)
    tr = get_extractor("pyin").extract(y, SR, profile)
    seg = segment_notes(tr, y, SR)
    got = np.round(seg.pitches - shift).astype(int).tolist()
    # Repeated notes (67, 67) may merge for legato timbres; everything else must match.
    expected_merged = [p for i, p in enumerate(MELODY) if i == 0 or p != MELODY[i - 1]]
    assert got in (MELODY, expected_merged)
    # Onsets are close to the true onsets (within 80 ms) for matched notes.
    if len(got) == len(MELODY):
        assert np.max(np.abs(seg.onsets - notes.onsets)) < 0.08


def test_silence_and_noise_give_no_notes(rng):
    for y in (np.zeros(SR * 2, dtype=np.float32), (0.3 * rng.standard_normal(SR * 2)).astype(np.float32)):
        tr = get_extractor("pyin").extract(y, SR, "hum")
        assert len(segment_notes(tr, y, SR)) <= 1
        c, _ = track_to_contour(tr, 20.0)
        assert len(c) < 10


def test_octave_folding_short_excursion_only():
    c = np.full(200, 60.0)
    c[50:53] = 72.0     # 3-frame octave error -> folded
    c[120:160] = 72.0   # 40-frame genuine octave leap -> kept
    out = fold_octave_jumps(c)
    assert np.all(out[50:53] == 60.0)
    assert np.all(out[125:155] == 72.0)


@pytest.mark.skipif(not EXTRACTORS["crepe"].available(), reason="torchcrepe not installed")
def test_crepe_extractor():
    y = render(melody_notes(), "voice", sr=SR, seed=1)
    tr = get_extractor("crepe").extract(y, SR, "hum")
    seg = segment_notes(tr, y, SR)
    assert len(seg) >= 8
    assert abs(np.median(seg.pitches) - np.median(MELODY)) < 1.0


def test_unknown_extractor():
    with pytest.raises(ValueError):
        get_extractor("nope")
