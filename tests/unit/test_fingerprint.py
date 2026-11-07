"""Audio fingerprinting: robustness to TikTok-style edits, unknown clips, persistence, incremental updates."""

import librosa
import numpy as np
import pytest

from melodyai.audio.synth import add_noise
from melodyai.demo.generator import generate_catalog, generate_melody
from melodyai.evaluation.datasets import render_polyphonic
from melodyai.fingerprint.hashing import constellation_hashes
from melodyai.fingerprint.index import FingerprintIndex, fingerprint_audio
from melodyai.fingerprint.peaks import extract_peaks

SR = 16000


@pytest.fixture(scope="module")
def mixes():
    return [render_polyphonic(m)[0] for m in generate_catalog(8)]


@pytest.fixture(scope="module")
def index(mixes):
    idx = FingerprintIndex()
    for i, y in enumerate(mixes):
        p, h = fingerprint_audio(y, SR)
        idx.add(f"s{i}", h, len(y) / SR, p)
    return idx


def clip(y, start_s, dur_s=10.0):
    a = int(start_s * SR)
    return y[a:a + int(dur_s * SR)]


def test_hashes_are_pitch_invariant():
    """Shifting every peak by a constant number of log bins leaves the hashes unchanged."""
    p = extract_peaks(render_polyphonic(generate_melody(1001))[0][: SR * 8], SR)
    h0 = constellation_hashes(p)
    p.f = p.f + 5.0
    h1 = constellation_hashes(p)
    assert len(h0) > 500 and np.array_equal(h0.h, h1.h)


@pytest.mark.parametrize("edit,speed,pitch", [
    ("none", 1.0, 0.0),
    ("speed_up_1.25", 1.25, 3.86),       # TikTok "sped up": faster AND higher
    ("slow_down_0.85", 0.85, -2.81),     # "slowed": slower AND lower
    ("noise_5db", 1.0, 0.0),
])
def test_identifies_edited_clips(mixes, index, edit, speed, pitch):
    rng = np.random.default_rng(3)
    for i in (1, 4, 6):
        c = clip(mixes[i], 12.0)
        if edit.startswith("speed") or edit.startswith("slow"):
            c = librosa.resample(c, orig_sr=SR * speed, target_sr=SR)
        elif edit == "noise_5db":
            c = add_noise(c, 5, rng)
        m = index.search(c, SR, top_k=3)
        assert m[0].song_id == f"s{i}", edit
        assert abs(m[0].speed - speed) < 0.04
        assert abs(m[0].pitch_semitones - pitch) < 0.6
        # The offset points at this passage or at a repetition of it (songs repeat sections).
        assert 0 <= m[0].offset_s <= len(mixes[i]) / SR
        assert m[0].score >= 0.5 and m[0].score - (m[1].score if len(m) > 1 else 0) >= 0.1


def test_pitch_shift_without_tempo_change(mixes, index):
    c = librosa.effects.pitch_shift(clip(mixes[2], 8.0), sr=SR, n_steps=2)
    m = index.search(c, SR, top_k=3)
    assert m[0].song_id == "s2" and abs(m[0].pitch_semitones - 2.0) < 0.6 and abs(m[0].speed - 1.0) < 0.06


def test_unknown_clip_scores_low(mixes, index):
    """Clips of recordings that are not indexed stay below the acceptance threshold after verification."""
    assert index.search(clip(mixes[3], 5.0), SR)[0].score >= 0.6
    for k in range(4):
        r = index.search(clip(render_polyphonic(generate_melody(900_000 + k))[0], 5.0), SR)
        assert not r or r[0].score < 0.5


def test_persistence_incremental_add_remove_and_compaction(tmp_path, mixes):
    idx = FingerprintIndex(tmp_path, flush_postings=1, max_segments=3)
    for i, y in enumerate(mixes[:5]):
        p, h = fingerprint_audio(y, SR)
        idx.add(f"s{i}", h, None, p)
    assert idx.stats()["segments"] <= 3  # flushed per song, compacted when too many
    q = clip(mixes[4], 3.0)
    assert idx.search(q, SR)[0].song_id == "s4"
    idx.remove("s4")
    assert all(m.song_id != "s4" for m in idx.search(q, SR))
    reopened = FingerprintIndex(tmp_path)
    assert len(reopened) == 4 and "s4" not in reopened
    assert all(m.song_id != "s4" for m in reopened.search(q, SR))
    assert reopened.search(clip(mixes[1], 3.0), SR)[0].song_id == "s1"
    before = reopened.n_postings
    reopened.compact()  # physically drops the deleted song's postings
    assert reopened.n_postings < before and reopened.stats()["segments"] == 1
    assert reopened.search(clip(mixes[1], 3.0), SR)[0].song_id == "s1"
    # memory-mapped segments: data is on disk, not copied into RAM
    assert isinstance(FingerprintIndex(tmp_path).segments[0].h, np.memmap)


def test_silence_gives_no_match(index):
    assert index.search(np.zeros(SR * 5, dtype=np.float32), SR) == []
