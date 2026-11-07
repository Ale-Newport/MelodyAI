import pytest

from melodyai.demo.lyrics_bank import LYRICS
from melodyai.fusion.fusion import EngineOutput, decide, fuse
from melodyai.lyrics.normalize import normalize_text, tokenize
from melodyai.lyrics.search import LyricsIndex


@pytest.fixture(scope="module")
def lyr_index():
    idx = LyricsIndex(":memory:")
    for i, l in enumerate(LYRICS):
        idx.add(f"L{i}", l["text"], l["language"])
    return idx


def top(idx, q, **kw):
    r = idx.search(q, 3, **kw)
    return r[0] if r else None


def test_normalisation():
    assert normalize_text("¡Corazón, CANCIÓN!") == "corazon cancion"
    assert tokenize("Don't stop—it's ﬁne") == ["dont", "stop", "its", "fine"]


def test_exact_and_partial_line(lyr_index):
    c = top(lyr_index, "turn the lantern turn the lantern slow")
    assert c.song_id == "L1" and c.coverage > 0.95
    assert "lantern" in c.matched_text.lower()


def test_F_misremembered_words(lyr_index):
    """Test F: substituted/misspelled/missing words still find the song."""
    queries = {
        "the lighthouse keeper counts the boats at nite": "L1",
        "copper wire hummin on the hills": "L3",
        "paper boat along the gutters run racing down to meet the evening": "L0",
        "corre corre rio claro hasta el puerto de almeria": "L13",
        "farolito de la esquina que alumbraba mi ninez": "L12",
    }
    for q, sid in queries.items():
        c = top(lyr_index, q)
        assert c is not None and c.song_id == sid, q
        assert c.score >= 0.6


def test_topic_overlap_is_not_a_match(lyr_index):
    """Shared generic words ('night', 'love', 'home') must not yield a confident match."""
    res = lyr_index.search("the night is long and i love my home", 5)
    ranked = [{"song_id": c.song_id, "score": c.score} for c in res]
    assert decide(ranked, "lyrics", calibration={})["status"] != "match"


def test_stopword_only_query_and_language_filter(lyr_index):
    assert lyr_index.search("the and of", 3) == []
    c = top(lyr_index, "luna vuelve pronto", language="es")
    assert c.song_id == "L17"


def test_remove_lyrics(lyr_index):
    idx = LyricsIndex(":memory:")
    idx.add("x", LYRICS[0]["text"])
    assert top(idx, "paper boats carry me back home").song_id == "x"
    idx.remove("x")
    assert idx.search("paper boats carry me back home") == []


def test_lyrics_index_persistence(tmp_path):
    p = str(tmp_path / "l.db")
    a = LyricsIndex(p)
    a.add("x", LYRICS[5]["text"])
    b = LyricsIndex(p)
    assert top(b, "we built a kite from yesterdays news").song_id == "x"


def _out(engine, pairs, rel=1.0):
    return EngineOutput(engine, [{"song_id": s, "score": v} for s, v in pairs], rel)


@pytest.mark.parametrize("method", ["weighted", "rrf", "combmnz"])
def test_fusion_complementary_evidence(method):
    """Melody slightly prefers B, lyrics strongly prefer A: fused result should be A,
    and A should rank above songs supported by only one engine."""
    mel = _out("melody", [("B", 0.62), ("A", 0.60), ("C", 0.3)])
    lyr = _out("lyrics", [("A", 0.9), ("D", 0.3)])
    fused = fuse([mel, lyr], method)
    assert fused[0]["song_id"] == "A"
    assert fused[0]["agreement"] == 2
    assert all(0 <= f["score"] <= 1 for f in fused)


def test_fusion_reliability_weighting():
    mel = _out("melody", [("B", 0.8), ("A", 0.5)], rel=1.0)
    lyr = _out("lyrics", [("A", 0.6)], rel=0.05)  # unreliable transcript
    assert fuse([mel, lyr], "weighted")[0]["song_id"] == "B"


def test_decide_statuses():
    assert decide([], "melody", calibration={})["status"] == "no_match"
    assert decide([{"song_id": "a", "score": 0.9}, {"song_id": "b", "score": 0.3}], "melody",
                  calibration={})["status"] == "match"
    assert decide([{"song_id": "a", "score": 0.6}, {"song_id": "b", "score": 0.59}], "melody",
                  calibration={})["status"] == "uncertain"
    assert decide([{"song_id": "a", "score": 0.2}], "melody", calibration={})["status"] == "no_match"
    cal = {"thresholds": {"melody:dtw": {"accept": 0.95, "margin": 0.0, "uncertain": 0.9}},
           "platt": {"melody:dtw": {"coef": [10.0, 5.0], "intercept": -5.0}}}
    d = decide([{"song_id": "a", "score": 0.9}], "melody", "dtw", calibration=cal)
    assert d["status"] == "uncertain" and d["threshold_source"] == "calibrated"
    assert 0 < d["calibrated_probability"] < 1
