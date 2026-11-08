"""Real-song evaluation harness (melodyai.evaluation.world_eval): variants, metrics, calibration, config, CLI.

No network: LRCLIB / Deezer answers come from ``httpx.MockTransport``. Every song, artist and lyric line below is
invented.
"""

import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import yaml

from melodyai import cli
from melodyai.evaluation import world_eval as we
from melodyai.lyrics.normalize import normalize_text, tokenize
from melodyai.world.index import WorldLyricsIndex, WorldSong
from melodyai.world.search import WorldCandidate, WorldResult
from melodyai.world.web import HostLimit, WebClient

FAST = {"lrclib.net": HostLimit(1000, 1000.0, 3), "api.deezer.com": HostLimit(1000, 1000.0, 6),
        "api.genius.com": HostLimit(1000, 1000.0, 3)}

PARADE = [
    "Walking down the harbor with a lantern in my hand",
    "Counting all the boats that never made it back to land",
    "Paper moon parade, shine across the bay",
    "Paper moon parade, carry me away",
    "Grandpa built a radio from bottle caps and wire",
    "Every station whistled like a kettle on the fire",
    "Paper moon parade, shine across the bay",
    "Paper moon parade, carry me away",
    "Seven silver herons dancing on the pier",
    "Paper moon parade, shine across the bay",
    "Paper moon parade, carry me away",
]
FARO = [
    "Camino por la orilla con un mapa sin color",
    "Las gaviotas me preguntan donde se quedo tu voz",
    "Faro de papel alumbra mi ciudad",
    "Faro de papel no me dejes naufragar",
    "Mi abuela cose velas con hilo de algodon",
    "El tranvia de las ocho canta una vieja cancion",
    "Faro de papel alumbra mi ciudad",
    "Faro de papel no me dejes naufragar",
]
GLASS = [
    "Glass canary singing in the attic of my mind",
    "Dusty letters folded in a drawer I never find",
    "Glass canary fly before the winter rain",
    "Glass canary call my name again",
    "Painted shutters rattle when the evening trains go by",
    "Glass canary fly before the winter rain",
    "Glass canary call my name again",
]
ALL_LINES = PARADE + FARO + GLASS

SONGS_CFG = [
    {"title": "Paper Moon Parade", "artist": "The Lantern Keepers", "language": "en", "tags": ["pop", "2010s"]},
    {"title": "Faro de Papel", "artist": "Marea Baja", "language": "es", "tags": ["pop", "long_tail"]},
    {"title": "Glass Canary", "artist": "Ivy Rook", "language": "en", "tags": ["tiktok"]},
]


def lrclib_record(rid, title, artist, lines, duration=200.0):
    return {"id": rid, "trackName": title, "artistName": artist, "albumName": "Album", "duration": duration,
            "instrumental": False, "plainLyrics": "\n".join(lines), "syncedLyrics": None}


LRCLIB_DB = [lrclib_record(11, "Paper Moon Parade", "The Lantern Keepers", PARADE),
             lrclib_record(12, "Faro de Papel", "Marea Baja", FARO),
             lrclib_record(13, "Glass Canary", "Ivy Rook", GLASS)]


def handler(request: httpx.Request) -> httpx.Response:
    url = request.url
    if url.host == "lrclib.net" and url.path == "/api/get":
        t, a = url.params.get("track_name", "").lower(), url.params.get("artist_name", "").lower()
        for r in LRCLIB_DB:
            if r["trackName"].lower() == t and r["artistName"].lower() == a:
                return httpx.Response(200, json=r)
        return httpx.Response(404, json={"code": 404})
    if url.host == "lrclib.net" and url.path == "/api/search":
        phrase = normalize_text(url.params.get("track_name") or url.params.get("q") or "")
        hits = [r for r in LRCLIB_DB if phrase and (normalize_text(r["trackName"]) in phrase
                                                     or phrase in normalize_text(r["trackName"]))]
        return httpx.Response(200, json=hits)
    if url.host == "api.deezer.com":
        return httpx.Response(200, json={"data": [], "total": 0})
    return httpx.Response(404)


@pytest.fixture()
def mock_web(monkeypatch):
    """Every WebClient the harness creates talks to the mock services, without rate limits."""
    transport = httpx.MockTransport(handler)
    real = WebClient

    def factory(cache_path=None, **kw):
        kw["host_limits"], kw["transport"] = FAST, transport
        return real(cache_path, **kw)

    monkeypatch.setattr(we, "WebClient", factory)
    return factory


@pytest.fixture()
def store(tmp_path):
    """A world index holding two of the three invented songs."""
    d = tmp_path / "store"
    idx = WorldLyricsIndex(d / "world.db")
    idx.upsert(WorldSong.create("Paper Moon Parade", "The Lantern Keepers", lyrics="\n".join(PARADE), rank=500000,
                                lrclib_id=11, source="sync:test"))
    idx.upsert(WorldSong.create("Faro de Papel", "Marea Baja", lyrics="\n".join(FARO), rank=300000, lrclib_id=12,
                                source="sync:test"))
    idx.close()
    return d


def write_config(tmp_path, **extra) -> str:
    cfg = {"name": "test", "seed": 3, "conditions": {"index": {}, "online": {"max_queries": 6}},
           "variants": ["chorus", "verse", "line", "misremembered", "negative"], "negatives": {"n": 4},
           "online": {"sources": ["index", "lrclib", "deezer"], "lrclib_rps": 1e6, "deadline_s": 5.0},
           "lyrics_cache": str(tmp_path / "eval_cache.db"), "songs": SONGS_CFG, **extra}
    path = tmp_path / "eval_world.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return str(path)


# ---------------------------------------------------------------------------- config
def test_repository_config_lists_songs_only():
    cfg = we.load_config(we.DEFAULT_CONFIG)
    songs = we.load_songs(cfg)
    assert 90 <= len(songs) <= 110
    langs = {lang: sum(s.language == lang for s in songs) for lang in ("es", "en")}
    assert langs["es"] > langs["en"] > 30
    groups = {g: sum(s.group == g for s in songs) for g in ("tiktok", "long_tail")}
    assert groups["tiktok"] >= 20 and groups["long_tail"] >= 15
    raw = yaml.safe_load(we.DEFAULT_CONFIG.read_text())
    assert all(set(s) <= {"title", "artist", "language", "tags"} for s in raw["songs"])
    assert not any(t.startswith("in_sync") for s in songs for t in s.tags)   # checked at run time, not hardcoded
    assert set(cfg["conditions"]) == {"index", "online", "both"}


def test_load_songs_rejects_lyrics_duplicates_and_bad_entries():
    ok = {"songs": [{"title": "Glass Canary", "artist": "Ivy Rook", "language": "EN", "tags": "tiktok"}]}
    s = we.load_songs(ok)[0]
    assert (s.language, s.tags, s.group) == ("en", ["tiktok"], "tiktok")
    with pytest.raises(ValueError, match="never lyrics"):
        we.load_songs({"songs": [{"title": "A", "artist": "B", "language": "en", "lyrics": "la la"}]})
    with pytest.raises(ValueError, match="duplicate"):
        we.load_songs({"songs": [{"title": "Glass Canary", "artist": "Ivy Rook", "language": "en"},
                                 {"title": "Glass Canary (Remastered 2011)", "artist": "Ivy Rook", "language": "en"}]})
    with pytest.raises(ValueError, match="language"):
        we.load_songs({"songs": [{"title": "A", "artist": "B"}]})


def test_load_config_merges_defaults_but_conditions_replace(tmp_path):
    path = write_config(tmp_path, asr={"max_songs": 2})
    cfg = we.load_config(path)
    assert list(cfg["conditions"]) == ["index", "online"]
    assert cfg["asr"]["max_songs"] == 2 and cfg["asr"]["voices"]["es"]          # nested defaults kept
    cfg = we.load_config(path, {"conditions": ["both"], "online": {"lrclib_rps": 0.2}})
    assert cfg["conditions"] == {"both": {}} and cfg["online"]["lrclib_rps"] == 0.2
    assert cfg["online"]["sources"] == ["index", "lrclib", "deezer"]


def test_select_songs_is_a_seeded_sample():
    assert we.select_songs(5, None, 1) == [0, 1, 2, 3, 4]
    a = we.select_songs(100, 30, 7)
    assert a == we.select_songs(100, 30, 7) and len(a) == 30 and a == sorted(a)
    assert a != we.select_songs(100, 30, 8)


# ---------------------------------------------------------------------------- variants
def test_chorus_block_finds_the_most_repeated_block():
    assert we.chorus_block(PARADE) == (2, 4)
    assert we.chorus_block(FARO) == (2, 4)
    no_repeat = ["one line about the river", "another about the hills", "a third about the sea"]
    assert we.chorus_block(no_repeat) is None
    # only one line repeats: that line + the next one
    lines = ["silver rain on the window", "the kettle hums alone", "silver rain on the window", "goodnight old town"]
    assert we.chorus_block(lines) == (0, 2)
    # repeated filler blocks are not a chorus
    assert we.chorus_block(["na na na", "oh oh oh", "na na na", "oh oh oh"]) is None


def test_verse_block_avoids_the_chorus_and_stays_in_the_first_half():
    avoid = [j for j, ln in enumerate(PARADE) if "parade" in ln.lower()]
    for seed in range(10):
        a, b = we.verse_block(PARADE, np.random.default_rng(seed), avoid)
        assert 2 <= b - a <= 3 and b <= max(b - a, len(PARADE) // 2)
        assert not set(range(a, b)) & set(avoid)
    assert we.verse_block(["hi"], np.random.default_rng(0)) is None


def test_text_queries_are_deterministic_and_per_song():
    song = we.EvalSong("Paper Moon Parade", "The Lantern Keepers", "en")
    q1 = we.text_queries(0, song, PARADE, we.TEXT_VARIANTS, seed=5)
    q2 = we.text_queries(0, song, PARADE, we.TEXT_VARIANTS, seed=5)
    assert [q.text for q in q1] == [q.text for q in q2]
    assert [q.variant for q in q1] == ["chorus", "verse", "line", "misremembered"]
    by = {q.variant: q for q in q1}
    assert by["chorus"].line_range == (2, 4) and by["misremembered"].line_range == by["verse"].line_range
    assert by["chorus"].text == " ".join(PARADE[2:4])
    assert 4 <= by["line"].n_words <= 14
    # another song index / other songs in the run do not change this song's queries (seeded by its key)
    q3 = we.text_queries(7, song, PARADE, we.TEXT_VARIANTS, seed=5)
    assert [q.text for q in q3] == [q.text for q in q1] and q3[0].qid == "s007-chorus"
    assert [q.text for q in we.text_queries(0, song, PARADE, we.TEXT_VARIANTS, seed=6)] != [q.text for q in q1]
    # newline joiner keeps the line breaks; the language hint is optional
    qn = we.text_queries(0, song, PARADE, ["chorus"], seed=5, joiner="\n", language_hint=True)[0]
    assert qn.text == "\n".join(PARADE[2:4]) and qn.language == "en" and q1[0].language is None


def test_misremember_drops_and_replaces_words():
    lines = PARADE[0:3]
    text, info = we.misremember(lines, "en", np.random.default_rng(4))
    n = sum(len(ln.split()) for ln in lines)
    assert info["words"] == n and info["dropped"] == round(0.15 * n)
    assert 1 <= info["replaced"] <= 2 and info["replaced"] == info.get("sound_alike", 0) + info.get("common_word", 0)
    assert len(text.split()) == n - info["dropped"]
    assert text != " ".join(lines)
    assert (text, info) == we.misremember(lines, "en", np.random.default_rng(4))
    assert all(isinstance(v, int) for v in info.values())            # counts only, never words


def test_sound_alike_changes_spelling_not_much_more():
    rng = np.random.default_rng(0)
    for word, lang in [("vida", "es"), ("llorar", "es"), ("corazon", "es"), ("night", "en"), ("phone", "en"),
                       ("kettle", "en"), ("xyz", "en")]:
        out = we.sound_alike(word, lang, rng)
        assert out and out != word and abs(len(out) - len(word)) <= 2


def test_negative_texts_are_everyday_sentences():
    negs = we.negative_texts(30, seed=1)
    assert len(negs) == 30 and [lang for lang, _ in negs[:4]] == ["es", "en", "es", "en"]
    assert negs == we.negative_texts(30, seed=1) and negs != we.negative_texts(30, seed=2)
    assert len({t for _, t in negs}) == 30
    assert all(12 <= len(tokenize(t)) <= 30 for _, t in negs)


def test_word_error_rate():
    assert we.word_error_rate([], []) == 0.0
    assert we.word_error_rate("a b c d".split(), "a b c d".split()) == 0.0
    assert we.word_error_rate("a b c d".split(), "a x c".split()) == 0.5
    assert we.word_error_rate(["a"], []) == 1.0


def test_asr_query_uses_joined_segments_and_keeps_numbers_only():
    class FakeSpeaker:
        def voice(self, language, k):
            return f"{language}-voice-{k}"

        def say(self, text, voice):
            self.spoken = text
            return np.zeros(we.SR * 2, dtype=np.float32)

        def sing(self, y, rng):
            return np.concatenate([y, y])

        def transcribe(self, y):
            return SimpleNamespace(segments=[{"text": " Paper moon parade, shine across the bay."},
                                             {"text": "Paper moon parade carry me "}],
                                   model="tiny", language="en", speech_detected=True, elapsed_s=0.5)

    sp = FakeSpeaker()
    song = we.EvalSong("Paper Moon Parade", "The Lantern Keepers", "en")
    q = we.asr_query(3, song, PARADE, "asr_sung", sp, seed=1, k=1)
    assert sp.spoken == " ".join(PARADE[2:4])
    assert q.text == "Paper moon parade, shine across the bay.\nPaper moon parade carry me"   # one line per segment
    assert q.variant == "asr_sung" and q.language == "en" and q.line_range == (2, 4)
    assert q.info["style"] == "sung" and q.info["audio_s"] == 4.0 and q.info["voice"] == "en-voice-1"
    assert q.info["wer"] == pytest.approx(1 / 13, abs=1e-3)        # "away" missing
    assert q.info["asr_reason"] == ""
    assert not any(isinstance(v, str) and len(v.split()) > 2 for v in q.info.values())


def test_asr_query_without_detected_speech_searches_nothing_like_the_engine():
    class Mute:
        def voice(self, language, k):
            return "v"

        def say(self, text, voice):
            return np.zeros(we.SR, dtype=np.float32)

        def transcribe(self, y):
            return SimpleNamespace(segments=[{"text": "ghost words from a filtered segment"}], model="tiny",
                                   language="en", speech_detected=False, elapsed_s=0.1)

    q = we.asr_query(3, we.EvalSong("Paper Moon Parade", "The Lantern Keepers", "en"), PARADE, "asr", Mute(),
                     seed=1, k=0)
    assert q.text == "" and q.info["speech_detected"] is False


# ---------------------------------------------------------------------------- scoring + metrics
def cand(title, artist, score, sources=("index",)):
    return WorldCandidate(song_id=f"world:{title}", title=title, artist=artist, album=None, language="en",
                          duration_s=None, cover_url=None, preview_url=None, rank=1000, links={}, score=score,
                          coverage=0.9, char_similarity=0.9, matched_text="(not stored)", line_range=(2, 3),
                          sources=list(sources), alignment_score=score)


def test_match_song_counts_covers_and_checks_artist():
    song = we.EvalSong("Paper Moon Parade", "The Lantern Keepers", "en")
    assert we.match_song(song, "Paper Moon Parade - Remastered 2011", "The Lantern Keepers feat. Bo") == (True, True)
    assert we.match_song(song, "Paper Moon Parade", "Acoustic Covers Band") == (True, False)
    assert we.match_song(song, "Glass Canary", "The Lantern Keepers") == (False, False)


def test_score_row_ranks_decisions_and_no_text():
    song = we.EvalSong("Paper Moon Parade", "The Lantern Keepers", "en")
    q = we.Query("s000-chorus", "chorus", " ".join(PARADE[2:4]), 0, None, (2, 4))
    res = WorldResult("ok", q.text, False, [cand("Glass Canary", "Ivy Rook", 0.8),
                                            cand("Paper Moon Parade", "The Lantern Keepers", 0.6)],
                      {"index": {"status": "ok", "n": 2, "elapsed_s": 0.01, "message": ""}}, 0.02)
    row = we.score_row(q, song, res, {"status": "match", "confidence": "medium"}, "index", 0.02, True)
    assert row["rank"] == 2 and row["artist_rank"] == 2 and not row["top1_correct"]
    assert row["false_match"] and not row["match_correct"] and row["found_by"] == ["index"]
    assert (row["top_score"], row["second_score"]) == (0.8, 0.6)
    dumped = json.dumps(row)
    assert "(not stored)" not in dumped and PARADE[2] not in dumped and "query_text" not in dumped
    neg = we.score_row(we.Query("neg-00", "negative", "the plumber will check the boiler", None, None), None,
                       WorldResult("no_candidates", "", False), {"status": "no_match"}, "index", 0.01, None)
    assert neg["kind"] == "negative" and not neg["false_match"] and neg["top_score"] == 0.0


def row(cond="index", variant="chorus", rank=1, decision="match", top=0.8, second=0.3, kind="song", song=0,
        lang="en", in_sync=True, elapsed=0.1, qid=None):
    correct = rank == 1
    return {"qid": qid or f"s{song:03d}-{variant}", "condition": cond, "variant": variant, "kind": kind,
            "song": song if kind == "song" else None, "language": lang, "group": "hits", "in_sync": in_sync,
            "status": "ok" if top else "no_candidates", "elapsed_s": elapsed, "rank": rank if kind == "song" else None,
            "top1_correct": kind == "song" and correct, "top1_artist_correct": kind == "song" and correct,
            "top_score": top, "second_score": second, "decision": decision,
            "match_correct": decision == "match" and correct and kind == "song",
            "false_match": decision == "match" and (kind == "negative" or not correct),
            "top": [{"title": "x"}] if top else [], "info": {},
            "sources": {"index": {"status": "ok", "n": 1 if top else 0, "elapsed_s": 0.01}},
            "requests": {"lrclib": 2} if cond == "online" else {}, "found_by": ["index"] if correct else []}


def test_song_and_negative_metrics_and_summaries():
    rows = [row(rank=1), row(rank=2, decision="match", song=1), row(rank=None, decision="no_match", top=0.3, song=2),
            row(rank=1, decision="uncertain", top=0.5, song=3, lang="es", in_sync=False),
            row(kind="negative", variant="negative", rank=None, decision="match", top=0.7, qid="neg-00"),
            row(kind="negative", variant="negative", rank=None, decision="no_match", top=0.2, qid="neg-01")]
    m = we.song_metrics([r for r in rows if r["kind"] == "song"])
    assert m["n"] == 4 and m["top1"] == 0.5 and m["top5"] == 0.75 and m["mrr"] == pytest.approx((1 + 0.5 + 1) / 4)
    assert m["match_correct"] == 0.25 and m["false_match"] == 0.25 and m["abstain"] == 0.5
    assert m["decision_accuracy"] == 0.5 and m["precision"] == 0.5
    n = we.negative_metrics([r for r in rows if r["kind"] == "negative"])
    assert n["n"] == 2 and n["false_match"] == 0.5 and n["max_top_score"] == 0.7
    s = we.summaries(rows)
    assert {(r["condition"], r["variant"]) for r in s["by_variant"]} == {("index", "chorus")}
    assert {r["language"]: r["n"] for r in s["by_language"]} == {"en": 3, "es": 1}
    assert s["negatives"][0]["n"] == 2 and s["overall"][0]["n"] == 4
    src = we.source_stats([*rows, row(cond="online", rank=1, song=9)])
    by = {(r["condition"], r["source"]): r for r in src}
    assert by[("index", "index")]["found_correct"] == 2 and by[("index", "index")]["proposed"] == 1.0
    assert by[("online", "index")]["requests_per_query"] == {"lrclib": 2.0}


def test_calibrate_world_respects_the_false_match_budget(tmp_path):
    rows = ([row(rank=1, top=0.7 + 0.01 * i, second=0.2, song=i) for i in range(20)]
            + [row(rank=2, top=0.62, second=0.55, song=20 + i) for i in range(5)]
            + [row(rank=None, top=0.45, second=0.1, song=30 + i, decision="no_match") for i in range(3)]
            + [row(kind="negative", variant="negative", rank=None, top=0.3 + 0.01 * i, second=0.0, decision="no_match",
                   qid=f"neg-{i:02d}") for i in range(10)])
    cal = we.calibrate_world(rows, max_fmr=0.05)
    th, fit = cal["thresholds"], cal["fit"]
    assert fit["fmr"] <= 0.05 and fit["negative_fmr"] <= 0.05
    assert fit["correct_matches"] == 20 and fit["false_matches"] == 0
    assert 0.62 < th["accept"] <= 0.70 or th["margin"] > 0.07
    assert 0.25 <= th["uncertain"] <= th["accept"]
    assert len(cal["holdout"]) == 2 and all(f["test"]["n_songs"] for f in cal["holdout"])
    assert cal["strict"]["fit"]["false_matches"] == 0 and cal["strict"]["fit"]["correct_matches"] == 20
    # the budgeted fit may trade a few false matches for more correct ones; the strict one never does
    risky = ([row(rank=1, top=0.8, second=0.3, song=i) for i in range(20)]
             + [row(rank=1, top=0.6, second=0.3, song=20 + i) for i in range(3)] + [row(rank=2, top=0.65, second=0.3, song=30)]
             + [row(kind="negative", variant="negative", rank=None, top=0.3, second=0.0, decision="no_match",
                    qid=f"neg-{i:02d}") for i in range(20)])
    cal2 = we.calibrate_world(risky, max_fmr=0.05)
    assert cal2["fit"]["correct_matches"] == 23 and cal2["fit"]["false_matches"] > 0
    assert cal2["strict"]["fit"]["false_matches"] == 0 and cal2["strict"]["fit"]["correct_matches"] == 20
    # nothing feasible but "never match" when every top is wrong
    bad = [row(rank=2, top=0.9, second=0.1, song=i) for i in range(5)]
    assert we.calibrate_world(bad)["fit"]["correct_matches"] == 0
    assert we.calibrate_world(bad)["thresholds"]["accept"] > 0.9
    # written only into thresholds["world"], everything else kept
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"note": "keep", "thresholds": {"lyrics": {"accept": 0.6}}, "platt": {"x": 1}}))
    we.write_world_calibration(cal, path, "run-1")
    out = json.loads(path.read_text())
    assert out["note"] == "keep" and out["platt"] == {"x": 1} and out["thresholds"]["lyrics"] == {"accept": 0.6}
    strict = cal["strict"]["thresholds"]
    assert out["thresholds"]["world"] == {"accept": strict["accept"], "margin": strict["margin"],
                                          "uncertain": max(strict["uncertain"], cal["current_thresholds"]["uncertain"])}
    assert out["world_calibration"]["fit"]["false_matches"] == 0
    assert out["world_calibration"]["source_run"] == "run-1"


def test_sample_queries_spreads_over_variants():
    qs = [we.Query(f"s{i:03d}-{v}", v, "x y z", i, None) for i in range(10) for v in ("chorus", "line")]
    qs += [we.Query(f"neg-{i:02d}", "negative", "x y z", None, None) for i in range(10)]
    s = we.sample_queries(qs, 9, seed=1, name="online")
    assert len(s) == 9 and {v: sum(q.variant == v for q in s) for v in ("chorus", "line", "negative")} == \
        {"chorus": 3, "line": 3, "negative": 3}
    assert [q.qid for q in s] == [q.qid for q in we.sample_queries(qs, 9, seed=1, name="online")]
    assert we.sample_queries(qs, None, 1, "x") == qs


def test_run_condition_paces_online_requests():
    class FakeSearch:
        def search(self, text, language=None, top_k=5, online=True):
            web.n += 3
            return WorldResult("ok", text, online, [cand("Glass Canary", "Ivy Rook", 0.9)],
                               {"lrclib": {"status": "ok", "n": 1, "elapsed_s": 0.0, "message": ""}}, 0.0)

    class FakeWeb:
        n = 0

        def stats(self):
            return {"hosts": {"lrclib.net": {"requests": self.n}}}

    web, slept = FakeWeb(), []
    songs = [we.EvalSong("Glass Canary", "Ivy Rook", "en")]
    qs = [we.Query("s000-line", "line", "glass canary fly before the winter rain", 0, None)]
    rows = we.run_condition("online", FakeSearch(), True, web, qs, songs, {0: False}, {}, lrclib_rps=0.5,
                            sleep=slept.append)
    assert rows[0]["requests"] == {"lrclib": 3} and rows[0]["top1_correct"] and rows[0]["in_sync"] is False
    assert len(slept) == 1 and 5.0 < slept[0] <= 6.0


# ---------------------------------------------------------------------------- end to end (no network)
def test_run_world_eval_end_to_end_without_lyrics_in_outputs(tmp_path, store, mock_web):
    md = tmp_path / "WORLD_RESULTS.md"
    res = we.run_world_eval(write_config(tmp_path), store=store, label="pilot", tts=False, markdown_path=md,
                            report_dir=tmp_path / "reports")
    rep = res["report"]
    assert rep["songs_with_lyrics"] == 3 and rep["index"]["songs"] == 2
    sync = {s["title"]: s["lyrics"]["in_sync"] for s in rep["songs"]}
    assert sync == {"Paper Moon Parade": True, "Faro de Papel": True, "Glass Canary": False}
    assert {s["title"]: s["lyrics"]["source"] for s in rep["songs"]}["Glass Canary"] == "lrclib"
    assert rep["plan"]["online"]["queries"] == 6 and rep["plan"]["index"]["queries"] == 3 * 4 + 4
    by = {(r["condition"], r["qid"]): r for r in rep["rows"]}
    assert by[("index", "s000-chorus")]["top1_correct"] and by[("index", "s001-chorus")]["top1_correct"]
    assert by[("index", "s002-chorus")]["rank"] is None          # not indexed: offline search cannot find it
    online = [r for r in rep["rows"] if r["condition"] == "online"]
    assert online and all(r["requests"].get("lrclib", 0) >= 0 for r in online)
    assert rep["calibration"]["thresholds"]["accept"] > 0
    assert rep["thresholds"]["accept"] == pytest.approx(0.62, abs=0.5)
    # privacy: no lyric line (and no matched / query text) in the JSON report or the Markdown
    report_text = (tmp_path / "reports" / f"{rep['run_id']}.json").read_text()
    md_text = md.read_text()
    for line in ALL_LINES:
        assert line not in report_text and line not in md_text
        assert normalize_text(line) not in normalize_text(report_text)
    assert "matched_text" not in report_text and "query_text" not in report_text
    assert "Pilot run" in md_text and "Glass Canary" in md_text and "## By query variant" in md_text
    assert "## Candidate sources" in md_text and "zero false matches" in md_text
    # the hand-written notes block survives a rewrite
    md.write_text(md_text.replace("_No notes yet._", "Kept by hand."))
    we.write_markdown(rep, md)
    assert "Kept by hand." in md.read_text()


def test_world_index_is_not_modified_by_the_evaluation(tmp_path, store, mock_web):
    before = WorldLyricsIndex(store / "world.db")
    n0 = len(before)
    before.close()
    we.run_world_eval(write_config(tmp_path), store=store, conditions="index,online", tts=False, markdown_path=None,
                      report_dir=tmp_path / "reports")
    after = WorldLyricsIndex(store / "world.db")
    assert len(after) == n0 and not after.has_key("Glass Canary", "Ivy Rook")
    after.close()


# ---------------------------------------------------------------------------- CLI wiring
def test_cli_evaluate_world_passes_options(monkeypatch, capsys):
    seen = {}

    def fake_run(config, **kw):
        seen.update(kw, config=config)
        return {"report_path": "r.json", "markdown_path": None,
                "report": {"summary": {"overall": [{"condition": "index", "n": 3, "top1": 1.0, "top5": 1.0,
                                                    "match_correct": 1.0, "false_match": 0.0}],
                                       "negatives": [{"condition": "index", "n": 2, "false_match": 0.0}]},
                           "calibration": {"thresholds": {"accept": 0.6, "margin": 0.05, "uncertain": 0.4}},
                           "calibration_written": None}}

    monkeypatch.setattr(we, "run_world_eval", fake_run)
    cli.main(["evaluate-world", "--conditions", "index,online", "--variants", "chorus,negative", "--limit", "3",
              "--no-tts", "--calibrate", "--seed", "5", "--online-max", "10", "--label", "pilot", "--markdown", "-"])
    assert seen["config"] == "configs/eval_world.yaml" and seen["conditions"] == "index,online"
    assert seen["variants"] == "chorus,negative" and seen["limit"] == 3 and seen["tts"] is False
    assert seen["calibrate"] is True and seen["seed"] == 5 and seen["online_max"] == 10
    assert seen["label"] == "pilot" and seen["markdown_path"] is None
    out = capsys.readouterr().out
    assert "top1=1.00" in out and "Saved results to r.json" in out


def test_cli_world_search_offline_prints_titles_not_lyrics(tmp_path, store, capsys):
    cli.main(["world-search", "paper moon parade shine across the bay", "--offline", "--store", str(store),
              "--cache", str(tmp_path / "cache.db")])
    out = capsys.readouterr().out
    assert "Paper Moon Parade - The Lantern Keepers" in out and "decision:" in out
    assert "source index" in out and PARADE[0] not in out and PARADE[2] not in out
    cli.main(["world-search", "paper moon parade shine across the bay", "--offline", "--store", str(store),
              "--cache", str(tmp_path / "cache.db"), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["candidates"][0]["title"] == "Paper Moon Parade" and "matched_text" not in data["candidates"][0]
    idx = WorldLyricsIndex(store / "world.db")
    assert len(idx) == 2                                          # --learn is off by default
    idx.close()
