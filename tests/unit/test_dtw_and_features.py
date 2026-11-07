import numpy as np
import pytest

from melodyai.features.melody import (
    NoteSequence,
    center,
    notes_to_contour,
    pitch_class_chroma,
    resample_contour,
)
from melodyai.features.windows import build_windows, query_vector
from melodyai.matching.dtw import abs_cost, banded_dtw_batch, dtw, local_alignment, subsequence_dtw

C, D, E, F, G, A = 60, 62, 64, 65, 67, 69


def seq(pitches, ioi=0.5):
    return NoteSequence.from_list([(i * ioi, ioi, p) for i, p in enumerate(pitches)])


def test_contour_rendering():
    c = notes_to_contour(seq([60, 62], ioi=0.5), rate=20)
    assert len(c) == 20 and np.all(c[:10] == 60) and np.all(c[10:] == 62)
    rests = NoteSequence.from_list([(0, 0.2, 60), (1.0, 0.2, 62)])
    assert np.isnan(notes_to_contour(rests, 20, keep_rests=True)).sum() > 5


def test_A_transposition_invariance():
    """Test A: C-D-E-G and F-G-A-C (same intervals) are equivalent after key normalisation."""
    q1 = query_vector(notes_to_contour(seq([C, D, E, G])))
    q2 = query_vector(notes_to_contour(seq([F, G, A, C + 12])))
    assert np.allclose(q1, q2)
    r = dtw(abs_cost(q1, q2))
    assert r.distance == pytest.approx(0.0, abs=1e-9)
    # ...whereas a different interval pattern is not equivalent.
    q3 = query_vector(notes_to_contour(seq([C, E, D, C])))
    assert dtw(abs_cost(q1, q3)).distance > 0.5


def test_B_tempo_invariance():
    """Test B: the same melody at different speeds keeps a high similarity."""
    mel = [C, D, E, G, E, D, C, G]
    slow = notes_to_contour(seq(mel, 0.6))
    fast = notes_to_contour(seq(mel, 0.3))
    a, b = query_vector(slow), query_vector(fast)
    # residual cost comes only from interpolating note transitions of the faster version
    assert dtw(abs_cost(a, b), 0.1, band=10).distance < 0.15
    # Non-uniform tempo (accelerando) is absorbed by warping.
    accel = NoteSequence.from_list([(sum(0.6 - 0.04 * k for k in range(i)), 0.6 - 0.04 * i, p) for i, p in enumerate(mel)])
    c = query_vector(notes_to_contour(accel))
    # Uneven note lengths move the median by ~1 semitone; the matchers search a small
    # residual key shift, so do the same here.
    best = min(dtw(abs_cost(a, c + s), 0.1, band=15).distance for s in np.arange(-2, 2.01, 0.5))
    assert best < 0.35


def test_dtw_path_is_monotonic_and_complete():
    a, b = np.random.default_rng(0).normal(size=30), np.random.default_rng(1).normal(size=40)
    r = dtw(abs_cost(a, b), 0.1)
    assert tuple(r.path[0]) == (0, 0) and tuple(r.path[-1]) == (29, 39)
    assert np.all(np.diff(r.path, axis=0) >= 0) and np.all(np.diff(r.path, axis=0).sum(axis=1) >= 1)


def test_subsequence_dtw_locates_segment():
    rng = np.random.default_rng(3)
    ref = rng.integers(55, 75, size=300).astype(float).repeat(1)
    q = ref[120:160] + 0.1 * rng.normal(size=40)
    r = subsequence_dtw(abs_cost(q, ref), 0.1)
    assert abs(r.start - 120) <= 2 and abs(r.end - 159) <= 2
    assert r.distance < 0.2


def test_banded_batch_matches_reference_dtw():
    rng = np.random.default_rng(4)
    q = rng.normal(size=32)
    W = rng.normal(size=(5, 32))
    batch = banded_dtw_batch(q, W, 4, np.inf, 0.1, np.zeros(1))
    single = [dtw(abs_cost(q, w), 0.1, band=4).distance for w in W]
    assert np.allclose(batch, single)
    # Per-window shifts are applied to the query.
    shifted = banded_dtw_batch(q + 3.0, W, 4, np.inf, 0.1, np.full(5, 3.0))
    assert np.allclose(shifted, single)


def test_D_different_melodies_far_apart():
    a = query_vector(notes_to_contour(seq([C, D, E, F, G, A, G, F])))
    b = query_vector(notes_to_contour(seq([G, E, C, E, G, C + 12, A, F])))
    assert dtw(abs_cost(a, b), 0.15, band=14).distance > 1.0


def test_local_alignment_tolerates_gaps():
    S = -np.ones((5, 8))
    for i, j in [(0, 2), (1, 3), (2, 5), (3, 6), (4, 7)]:  # one insertion in the reference
        S[i, j] = 1.0
    best, bi, bj, _ = local_alignment(S, gap=0.5)
    assert best == pytest.approx(4.5) and (bi, bj) == (5, 8)


def test_windows_cover_song_and_are_key_normalised():
    notes = seq([C, D, E, G] * 10)
    w = build_windows(notes, lengths=(6, 12), n_points=32)
    assert len(w) > 10
    assert np.allclose(np.median(w.vectors, axis=1), 0, atol=1.0)
    assert w.t0.min() == 0 and w.t1.max() == pytest.approx(notes.end_time, abs=0.6)


def test_resample_and_center_and_chroma():
    assert len(resample_contour(np.arange(10.0), 64)) == 64
    assert np.median(center(np.array([60.0, 62, 64]))) == 0
    ch = pitch_class_chroma(np.array([60.0, np.nan, 67.0]))
    assert ch[0].argmax() == 0 and ch[1].sum() == 0 and ch[2].argmax() == 7
