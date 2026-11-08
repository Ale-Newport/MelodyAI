"""RecognitionEngine + open-world search: when the search runs, what it is asked, and who wins the resolution.

No network: the world search is a fake returning :class:`melodyai.world.search.WorldResult` objects, the ASR
is a fake transcriber. Every title, artist and lyric below is invented.
"""

import threading
import time

import numpy as np
import pytest

from melodyai import engine as engine_mod
from melodyai.audio.io import Audio
from melodyai.demo.generator import generate_catalog
from melodyai.demo.queries import CONDITIONS, perturb, synth_query
from melodyai.engine import (
    WORLD_SONG_SOURCE,
    EngineSettings,
    RecognitionEngine,
    WorldTask,
    world_query_text,
    world_result_item,
)
from melodyai.lyrics.asr import Transcript
from melodyai.lyrics.search import LyricsIndex
from melodyai.world.search import WorldCandidate, WorldResult

LOCAL_LYRICS = ("Quiet harbour lamps are glowing\nFolding maps of silver weather\n"
                "Every sailor counts the herons\nQuiet harbour lamps are glowing")
SPEC_SONG_KEYS = {"id", "title", "artist", "album", "language", "reference_type", "source", "duration_s",
                  "cover_url", "preview_url", "deezer_id", "has_audio", "links"}
SPEC_WORLD_REF_KEYS = {"rank", "score", "coverage", "matched_text", "line_range", "sources"}


def cand(n: int, score: float, title: str | None = None) -> WorldCandidate:
    title = title or f"Invented Song {n}"
    return WorldCandidate(
        song_id=f"world:w{n:015x}", title=title, artist=f"Made Up Band {n}", album="Imaginary Album", language="en",
        duration_s=200.0 + n, cover_url=f"https://cdn.example.test/{n}.jpg", preview_url=f"https://cdn.example.test/{n}.mp3",
        rank=1000 * n, links={"deezer": f"https://www.deezer.com/track/{n}", "spotify": "https://open.spotify.com/search/x",
                              "youtube": "https://www.youtube.com/results?search_query=x"},
        score=score, coverage=min(1.0, score + 0.05), char_similarity=0.9, matched_text="invented line one / two",
        line_range=(2, 3), sources=["index", "lrclib"], alignment_score=score)


class FakeWorld:
    """Stands in for WorldSearch: records every call, answers with fixed candidates."""

    deadline_s = 2.0

    def __init__(self, scores=(0.9, 0.5), delay_s=0.0, error: Exception | None = None, status: str | None = None):
        self.scores, self.delay_s, self.error, self.status = scores, delay_s, error, status
        self.calls: list[dict] = []
        self.started = threading.Event()

    def search(self, text, language=None, top_k=5, online=True):
        self.calls.append({"text": text, "language": language, "top_k": top_k, "online": online,
                           "t": time.perf_counter(), "thread": threading.current_thread().name})
        self.started.set()
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        cands = [cand(i + 1, s) for i, s in enumerate(self.scores)]
        status = self.status or ("ok" if cands else "no_candidates")
        sources = {"index": {"status": "ok", "n": len(cands), "elapsed_s": 0.01, "message": ""},
                   "lrclib": {"status": "ok" if online else "skipped", "n": 0, "elapsed_s": 0.0, "message": ""}}
        return WorldResult(status, text, online, cands if status == "ok" else [], sources, 0.05)


def make_engine(world=None, lyrics: dict | None = None, melody_index=None, **settings) -> RecognitionEngine:
    from melodyai.indexing.melody_index import MelodyIndex

    lyr = LyricsIndex(":memory:")
    for sid, text in (lyrics or {}).items():
        lyr.add(sid, text, "en")
    settings.setdefault("asr_enabled", False)
    return RecognitionEngine(melody_index if melody_index is not None else MelodyIndex(), lyr,
                             EngineSettings(**settings), world=world)


def fake_asr(eng, monkeypatch, transcript: Transcript, calls: list):
    monkeypatch.setattr(eng, "asr_ready", lambda: True)

    def transcribe(audio, language=None):
        calls.append(language)
        return transcript

    monkeypatch.setattr(eng, "transcribe", transcribe)


def sung(segments: list[str], language="en") -> Transcript:
    segs = [{"start": float(i), "end": float(i) + 1, "text": s, "avg_logprob": -0.2, "no_speech_prob": 0.01,
             "compression_ratio": 1.2} for i, s in enumerate(segments)]
    return Transcript(" ".join(segments), language, 0.95, True, "", segs, "large-v3-turbo", 0.1)


def tone(seconds=3.0, sr=16000) -> Audio:
    t = np.arange(int(seconds * sr)) / sr
    return Audio((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)


# ---------------------------------------------------------------------------- helpers
def test_world_query_text_joins_sung_phrases_with_newlines():
    tr = sung(["  first invented phrase ", "", "second invented phrase"])
    assert world_query_text(tr) == "first invented phrase\nsecond invented phrase"
    plain = Transcript("only plain text here", "en", 0.9, True, "", [], "tiny")
    assert world_query_text(plain) == "only plain text here"


def test_world_result_item_matches_the_api_contract():
    item = world_result_item(1, cand(3, 0.81).to_dict())
    assert item["rank"] == 1 and item["song_id"] == item["song"]["id"] and item["score"] == 0.81
    assert set(item["song"]) == SPEC_SONG_KEYS
    assert item["song"]["reference_type"] == "online" and item["song"]["source"] == WORLD_SONG_SOURCE
    assert item["song"]["has_audio"] is False and set(item["song"]["links"]) == {"deezer", "spotify", "youtube"}
    assert set(item["engines"]) == {"world"} and set(item["engines"]["world"]) == SPEC_WORLD_REF_KEYS
    assert item["engines"]["world"]["line_range"] == [2, 3]


def test_world_task_never_raises():
    t = WorldTask(None, "some words", None, 5, True)
    assert t.result()["status"] == "disabled" and t.result()["online"] is False
    t = WorldTask(FakeWorld(error=RuntimeError("boom")), "some words", None, 5, True)
    r = t.result()
    assert r["status"] == "error" and "boom" in r["message"] and r["candidates"] == []


def test_world_task_times_out(monkeypatch):
    monkeypatch.setattr(engine_mod, "WORLD_JOIN_GRACE_S", 0.0)
    w = FakeWorld(delay_s=1.0)
    w.deadline_s = 0.1
    t0 = time.perf_counter()
    r = WorldTask(w, "some words here", None, 5, True).result()
    assert r["status"] == "error" and r["message"] == "timed out" and time.perf_counter() - t0 < 0.8


# ---------------------------------------------------------------------------- typed lyrics
def test_typed_lyrics_world_match_resolves_lyrics_online():
    w = FakeWorld(scores=(0.88, 0.41))
    eng = make_engine(w, top_k=3)
    r = eng.recognize(text="  marble kites over the copper valley  ", mode="lyrics", language="en", online=True)
    assert w.calls[0]["text"] == "marble kites over the copper valley" and w.calls[0]["online"] is True
    assert w.calls[0]["language"] == "en" and w.calls[0]["top_k"] >= 5
    assert w.calls[0]["thread"] == "melodyai-world-search"
    assert r["resolved_mode"] == "lyrics_online" and r["status"] == "match" and r["decision"]["status"] == "match"
    world = r["engines"]["world"]
    assert world["status"] == "ok" and world["decision"]["status"] == "match" and len(world["candidates"]) == 2
    assert [x["song_id"] for x in r["results"]] == [c["song_id"] for c in world["candidates"]]
    top = r["results"][0]
    assert set(top["song"]) == SPEC_SONG_KEYS and top["song"]["title"] == "Invented Song 1"
    assert set(top["engines"]["world"]) == SPEC_WORLD_REF_KEYS
    assert r["message"] == "Match found." and r["score_kind"] in ("relative", "calibrated_probability")


def test_online_flag_is_passed_through():
    w = FakeWorld()
    r = make_engine(w).recognize(text="marble kites over the copper valley", mode="lyrics", online=False)
    assert w.calls[0]["online"] is False
    assert r["engines"]["world"]["online"] is False and r["engines"]["world"]["sources"]["lrclib"]["status"] == "skipped"


def test_local_lyrics_match_wins_and_world_stays_as_also_found():
    w = FakeWorld(scores=(0.95, 0.3))
    eng = make_engine(w, lyrics={"local1": LOCAL_LYRICS})
    r = eng.recognize(text="every sailor counts the herons", mode="lyrics")
    assert r["resolved_mode"] == "lyrics" and r["status"] == "match" and r["results"][0]["song_id"] == "local1"
    assert r["engines"]["world"]["decision"]["status"] == "match"  # rule (a): reported, not used
    assert all("world" not in x["engines"] for x in r["results"])


def test_world_uncertain_beats_local_no_match_with_lower_score():
    w = FakeWorld(scores=(0.55, 0.52))
    r = make_engine(w, lyrics={"local1": LOCAL_LYRICS}).recognize(text="marble kites over the copper valley",
                                                                  mode="lyrics")
    assert r["engines"]["world"]["decision"]["status"] == "uncertain"
    assert r["resolved_mode"] == "lyrics_online" and r["status"] == "uncertain"


def test_weak_world_list_does_not_replace_a_better_local_list():
    w = FakeWorld(scores=(0.2, 0.1))
    eng = make_engine(w, lyrics={"local1": LOCAL_LYRICS})
    r = eng.recognize(text="quiet harbour lamps folding maps of weather", mode="lyrics")
    local_top = r["results"][0]
    assert local_top["song_id"] == "local1" and r["resolved_mode"] == "lyrics"
    assert r["engines"]["world"]["decision"]["status"] == "no_match"


def test_no_world_candidates_keeps_local_and_reports_status():
    w = FakeWorld(scores=())
    r = make_engine(w).recognize(text="marble kites over the copper valley", mode="lyrics")
    assert r["engines"]["world"]["status"] == "no_candidates" and r["resolved_mode"] == "lyrics"
    assert r["status"] == "no_match" and r["results"] == []


def test_no_world_search_configured_reports_disabled():
    r = make_engine(None).recognize(text="marble kites over the copper valley", mode="lyrics")
    assert r["engines"]["world"]["status"] == "disabled" and r["engines"]["world"]["decision"]["status"] == "no_match"


def test_world_failure_never_breaks_recognition():
    w = FakeWorld(error=RuntimeError("source exploded"))
    r = make_engine(w, lyrics={"local1": LOCAL_LYRICS}).recognize(text="every sailor counts the herons", mode="lyrics")
    assert r["status"] == "match" and r["results"][0]["song_id"] == "local1"
    assert r["engines"]["world"]["status"] == "error"


# ---------------------------------------------------------------------------- audio: ASR, modes, overlap
def test_transcript_feeds_the_world_search_with_segments_and_language(monkeypatch):
    w = FakeWorld(scores=(0.9,))
    eng = make_engine(w, asr_enabled=True)
    calls: list = []
    fake_asr(eng, monkeypatch, sung(["marble kites over the valley", "copper rain on the window"], "es"), calls)
    r = eng.recognize(tone(), mode="lyrics")
    assert calls == [None]
    assert w.calls[0]["text"] == "marble kites over the valley\ncopper rain on the window"
    assert w.calls[0]["language"] == "es"
    assert r["engines"]["lyrics"]["source"] == "asr" and r["resolved_mode"] == "lyrics_online"


def test_no_speech_means_no_world_search(monkeypatch):
    w = FakeWorld()
    eng = make_engine(w, asr_enabled=True)
    fake_asr(eng, monkeypatch, Transcript("", None, 0.0, False, "no speech", [], "tiny"), [])
    r = eng.recognize(tone(), mode="lyrics")
    assert not w.calls and "world" not in r["engines"]


def test_clip_without_fingerprint_match_is_transcribed_for_the_world(monkeypatch):
    w = FakeWorld(scores=(0.9, 0.3))
    eng = make_engine(w, asr_enabled=True)
    calls: list = []
    fake_asr(eng, monkeypatch, sung(["dancing paper tigers on the roof", "dancing paper tigers"]), calls)
    r = eng.recognize(tone(), mode="clip")
    assert calls and w.calls and w.calls[0]["text"].startswith("dancing paper tigers")
    assert r["resolved_mode"] == "lyrics_online" and r["status"] == "match"
    assert any("fingerprint index is empty" in x for x in r["warnings"])


class FakeFp:
    """Fingerprint index stand-in with one fixed candidate."""

    def __init__(self, score: float):
        self.score = score

    def __len__(self):
        return 1

    def search(self, samples, sr, top_k=10):
        d = {"song_id": "rec1", "score": self.score, "evidence": 12, "speed": 1.0, "pitch_semitones": 0.0,
             "start_time": 4.0}
        return [type("M", (), {"to_dict": lambda self: dict(d)})()]


@pytest.mark.parametrize("mode", ["clip", "auto"])
def test_fingerprint_match_ends_the_search(monkeypatch, mode):
    w = FakeWorld()
    eng = make_engine(w, asr_enabled=True)
    eng.fp_index = FakeFp(0.9)
    calls: list = []
    fake_asr(eng, monkeypatch, sung(["dancing paper tigers on the roof"]), calls)
    r = eng.recognize(tone(), mode=mode)
    assert r["resolved_mode"] == "clip" and r["status"] == "match" and not calls and not w.calls


def test_unsure_fingerprint_clip_falls_back_to_the_sung_words(monkeypatch):
    w = FakeWorld(scores=(0.9, 0.3))
    eng = make_engine(w, asr_enabled=True)
    eng.fp_index = FakeFp(0.3)  # below the fingerprint "uncertain" threshold
    calls: list = []
    fake_asr(eng, monkeypatch, sung(["dancing paper tigers on the roof"]), calls)
    r = eng.recognize(tone(), mode="clip")
    assert calls and w.calls and r["engines"]["fingerprint"]["candidates"]
    assert r["resolved_mode"] == "lyrics_online" and r["results"][0]["song_id"].startswith("world:")


def test_clip_without_asr_keeps_previous_behaviour():
    w = FakeWorld()
    r = make_engine(w).recognize(tone(), mode="clip")  # ASR disabled: nothing to search with
    assert not w.calls and r["resolved_mode"] == "clip" and r["status"] == "no_match"
    assert "lyrics" not in r["engines"] and "world" not in r["engines"]


def test_hum_mode_never_transcribes_but_typed_text_searches_the_world(monkeypatch):
    w = FakeWorld(scores=(0.9, 0.2))
    eng = make_engine(w, asr_enabled=True)
    calls: list = []
    fake_asr(eng, monkeypatch, sung(["should never be used here"]), calls)
    r = eng.recognize(tone(), mode="hum")
    assert not calls and not w.calls and "world" not in r["engines"]
    r = eng.recognize(tone(), text="marble kites over the copper valley", mode="hum")
    assert not calls and len(w.calls) == 1 and r["resolved_mode"] == "lyrics_online"


def test_world_search_overlaps_with_melody_matching(monkeypatch):
    """Typed text starts the search before the melody engine runs; the engine joins it afterwards."""
    w = FakeWorld(scores=(0.9, 0.2), delay_s=0.3)
    eng = make_engine(w)
    seen = {}
    real = engine_mod.analyze_query

    def spy(*a, **k):
        seen["world_started"] = w.started.is_set()
        return real(*a, **k)

    monkeypatch.setattr(engine_mod, "analyze_query", spy)
    r = eng.recognize(tone(), text="marble kites over the copper valley", mode="combined")
    assert seen["world_started"] is True and r["engines"]["world"]["status"] == "ok"


# ---------------------------------------------------------------------------- melody-only local answers
@pytest.fixture(scope="module")
def melodic():
    from melodyai.features.extract import reference_from_notes
    from melodyai.indexing.melody_index import MelodyIndex

    songs = generate_catalog(6, base_seed=1000)
    idx = MelodyIndex()
    for i, m in enumerate(songs):
        idx.add(reference_from_notes(f"m{i}", m.notes))
    rng = np.random.default_rng(7)
    spec = CONDITIONS["hum_clean"]
    pn, _ = perturb(songs[2].notes.slice(8, 22), spec, rng)
    return idx, Audio(synth_query(pn, spec, rng), 16000)


def test_world_match_beats_a_melody_only_match(melodic):
    idx, hum = melodic
    eng = make_engine(FakeWorld(scores=(0.9, 0.3)), melody_index=idx)
    local = make_engine(None, melody_index=idx).recognize(hum, mode="hum")
    assert local["status"] == "match" and local["results"][0]["song_id"] == "m2"  # precondition
    r = eng.recognize(hum, text="marble kites over the copper valley", mode="hum")
    assert r["resolved_mode"] == "lyrics_online" and r["results"][0]["song_id"].startswith("world:")  # rule (b)
    assert "alignment" not in r and r["engines"]["melody"]["candidates"]


def test_melody_only_match_beats_an_uncertain_world(melodic):
    idx, hum = melodic
    r = make_engine(FakeWorld(scores=(0.56, 0.5)), melody_index=idx).recognize(
        hum, text="marble kites over the copper valley", mode="hum")
    assert r["engines"]["world"]["decision"]["status"] == "uncertain"
    assert r["resolved_mode"] == "hum" and r["results"][0]["song_id"] == "m2" and r["status"] == "match"  # rule (c)


# ---------------------------------------------------------------------------- the rules themselves
@pytest.mark.parametrize("resolved, local, lyr_top, remote, world_top, local_top, expected", [
    ("lyrics", "match", "a", "match", 0.95, 0.7, False),         # (a) lyrics-backed local match
    ("clip", "match", None, "match", 0.95, 0.7, False),          # (a) fingerprint-backed local match
    ("combined", "match", "a", "match", 0.95, 0.7, False),       # (a) fused, the lyrics engine agrees
    ("combined", "match", "b", "match", 0.95, 0.7, True),        # fused match without lyrics support -> (b)
    ("hum", "match", None, "match", 0.9, 0.8, True),             # (b)
    ("hum", "match", None, "uncertain", 0.9, 0.8, False),        # (c)
    ("hum", "uncertain", None, "uncertain", 0.5, 0.45, True),    # (d) same status, higher score
    ("hum", "uncertain", None, "uncertain", 0.5, 0.52, False),   # (d) lower score
    ("hum", "uncertain", None, "no_match", 0.4, 0.3, False),     # (d) worse status
    ("none", "no_match", None, "no_match", 0.2, 0.0, True),      # (d) nothing local at all
])
def test_prefer_world_rules(resolved, local, lyr_top, remote, world_top, local_top, expected):
    eng = make_engine(None)
    ranked = [{"song_id": "a", "score": local_top}] if resolved != "none" else []
    lyr_c = [{"song_id": lyr_top, "score": 0.9}] if lyr_top else []
    world = {"candidates": [{"song_id": "world:w1", "score": world_top}], "decision": {"status": remote}}
    assert eng.prefer_world(resolved, ranked, {"status": local}, lyr_c, world) is expected


def test_prefer_world_needs_candidates():
    eng = make_engine(None)
    world = {"candidates": [], "decision": {"status": "no_match"}}
    assert eng.prefer_world("none", [], {"status": "no_match"}, [], world) is False
