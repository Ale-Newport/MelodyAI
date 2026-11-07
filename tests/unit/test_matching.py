"""Functional retrieval tests on the procedural catalog (synthetic queries)."""

import numpy as np
import pytest

from melodyai.audio.io import Audio
from melodyai.demo.generator import generate_melody
from melodyai.demo.queries import CONDITIONS, excerpt, non_melodic_query, perturb, synth_query
from melodyai.engine import default_algorithms
from melodyai.features.extract import QueryFeatures, analyze_query
from melodyai.fusion.fusion import decide

ALGS = default_algorithms()
SYMBOLIC_ALGS = ["dtw", "subseq_dtw", "note_align", "ensemble"]


def rank_of(cands, sid):
    ids = [c.song_id for c in cands]
    return ids.index(sid) + 1 if sid in ids else None


@pytest.mark.parametrize("alg", SYMBOLIC_ALGS)
def test_A_transposed_query_retrieves_song(melodies, melody_index, alg):
    """Test A at catalog level: a passage transposed by +5/-7 semitones is still found."""
    for i in (0, 5, 11):
        m = melodies[i]
        sub = m.notes.slice(10, 22)
        for shift in (5, -7):
            q = QueryFeatures.from_notes(sub.transpose(shift))
            assert rank_of(ALGS[alg].search(q, melody_index, 5), f"s{i}") == 1


@pytest.mark.parametrize("alg", SYMBOLIC_ALGS)
@pytest.mark.parametrize("factor", [0.7, 1.45])
def test_B_tempo_changed_query(melodies, melody_index, alg, factor):
    for i in (2, 7):
        q = QueryFeatures.from_notes(melodies[i].notes.slice(5, 18).time_scale(factor))
        assert rank_of(ALGS[alg].search(q, melody_index, 5), f"s{i}") == 1


@pytest.mark.parametrize("alg", SYMBOLIC_ALGS)
def test_C_mid_song_chorus(melodies, melody_index, alg):
    """Test C: a query from a chorus in the middle of the song finds the song and its location."""
    for i in (1, 4, 9):
        m = melodies[i]
        chorus = [s for s in m.sections if s["name"] == "chorus" and s["start"] > 5][0]
        k0 = int(np.searchsorted(m.notes.onsets, chorus["start"]))
        q = QueryFeatures.from_notes(m.notes.slice(k0, k0 + 12).transpose(3))
        cands = ALGS[alg].search(q, melody_index, 5)
        assert cands[0].song_id == f"s{i}"
        if alg != "note_align":  # interval alignment may match a repeated chorus elsewhere
            assert cands[0].start_time is not None


def test_C_location_matches_a_chorus(melodies, melody_index):
    m = melodies[3]
    chorus = [s for s in m.sections if s["name"] == "chorus" and s["start"] > 5][0]
    k0 = int(np.searchsorted(m.notes.onsets, chorus["start"]))
    q = QueryFeatures.from_notes(m.notes.slice(k0, k0 + 12))
    c = ALGS["dtw"].search(q, melody_index, 1)[0]
    # The matched passage lies inside one of the chorus sections (choruses repeat).
    choruses = [s for s in m.sections if s["name"] == "chorus"]
    assert any(s["start"] - 1.0 <= c.start_time <= s["end"] for s in choruses)


def test_D_unknown_melody_scores_lower(melodies, melody_index):
    """Test D: melodies that are not in the catalog do not match as strongly as true matches."""
    alg = ALGS["dtw"]
    true_scores, foreign_scores = [], []
    for i in range(6):
        q = QueryFeatures.from_notes(melodies[i].notes.slice(8, 20))
        true_scores.append(alg.search(q, melody_index, 1)[0].score)
        other = generate_melody(900_000 + i)  # seeds outside the catalog range
        qf = QueryFeatures.from_notes(other.notes.slice(8, 20))
        foreign_scores.append(alg.search(qf, melody_index, 1)[0].score)
    assert min(true_scores) > max(foreign_scores)


def test_E_non_melodic_audio_gives_no_confident_match(melody_index, rng):
    """Test E: noise, silence, a single held tone and a glissando never produce a 'match'."""
    for kind in ("noise", "silence", "single_tone", "glissando"):
        q = analyze_query(Audio(non_melodic_query(kind, rng), 16000), "hum", with_chroma=False)
        assert not q.informative or kind == "glissando"
        cands = ALGS["ensemble"].search(q, melody_index, 5) if q.informative else []
        ranked = [{"song_id": c.song_id, "score": c.score} for c in cands]
        assert decide(ranked, "melody", "ensemble", calibration={})["status"] != "match"


@pytest.mark.slow
@pytest.mark.parametrize("cond", ["hum_clean", "whistle", "hum_detuned"])
def test_audio_queries_end_to_end(melodies, melody_index, cond):
    """Synthesised hum/whistle audio -> pYIN -> matching. Requires >= 7/8 top-1 hits."""
    rng = np.random.default_rng(7)
    spec = CONDITIONS[cond]
    hits = 0
    for i in range(8):
        m = melodies[i]
        ex, _ = excerpt(m.notes, rng, 6, 10)
        pn, _ = perturb(ex, spec, rng)
        y = synth_query(pn, spec, rng)
        q = analyze_query(Audio(y, 16000), "whistle" if spec.timbre == "whistle" else "hum", with_chroma=False)
        hits += rank_of(ALGS["ensemble"].search(q, melody_index, 5), f"s{i}") == 1
    assert hits >= 7


def test_alignment_explanation_is_consistent(melodies, melody_index):
    """The visualised path must index valid positions of both plotted contours."""
    from melodyai.matching.explain import explain_alignment

    m = melodies[6]
    q = QueryFeatures.from_notes(m.notes.slice(12, 26).transpose(3).time_scale(1.15))
    c = ALGS["dtw"].search(q, melody_index, 1)[0]
    a = explain_alignment(q, melody_index.refs["s6"], c.start_time, c.end_time)
    path = np.asarray(a["path"])
    assert path[:, 0].max() < len(a["query_contour"]) and path[:, 1].max() < len(a["reference_contour"])
    assert abs(a["query_transposition_semitones"] + 3) < 0.6
    assert a["reference_match"][0] <= c.start_time + 1.0
