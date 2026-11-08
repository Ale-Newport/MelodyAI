"""Open-world search: sources, lyric verification, merging, covers, learning, failures and deadlines.

No network: LRCLIB / Deezer / Genius answers come from ``httpx.MockTransport`` over a tiny invented catalog.
All lyrics, titles and artists below are invented.
"""

import json
import time
from types import SimpleNamespace

import httpx
import pytest

from melodyai.fusion.fusion import DEFAULT_THRESHOLDS, decide
from melodyai.lyrics.normalize import tokenize
from melodyai.world.deezer import DeezerClient
from melodyai.world.index import WorldLyricsIndex, WorldSong
from melodyai.world.lrclib import LrclibClient
from melodyai.world.search import COVER_SCORE_WINDOW, WorldSearch, popularity_bonus, song_links
from melodyai.world.sources import (
    Candidate,
    ClaudeSource,
    DeezerTitleSource,
    GeniusSource,
    LocalIndexSource,
    LrclibTitleSource,
    default_sources,
    query_phrases,
    title_in_query,
)
from melodyai.world.text import normalize_title
from melodyai.world.web import HostLimit, WebClient

FAST = {"lrclib.net": HostLimit(1000, 1000.0, 3), "api.deezer.com": HostLimit(1000, 1000.0, 6),
        "api.genius.com": HostLimit(1000, 1000.0, 3)}

# ---------------------------------------------------------------------------- invented catalog
LANTERNS = ("Paper lanterns drifting on the river\nWe count the silver lights until the morning\n"
            "Paper lanterns carry me home\nCarry me home across the water\n"
            "Paper lanterns carry me home\nNever let the candle fade away")
TORMENTA = ("Bailando sobre la tormenta de cristal\nTu voz me llama desde el puerto viejo\n"
            "Tormenta de cristal no me dejes caer\nLas gaviotas cantan en el muelle\n"
            "Tormenta de cristal no me dejes caer")
GARDEN = ("Copper garden in the rain\nWhispers under broken stairs\nCopper garden calls my name\n"
          "Running through the yellow fields")
CORAZON = ("Mi corazón de papel se moja en la lluvia\nCaminando por la avenida sin final\n"
           "Corazón de papel vuela conmigo\nLa ciudad se duerme y yo despierto")
# same title as LANTERNS, different song
OTHER_LANTERNS = ("Paper lanterns hanging in the hallway\nDusty boxes full of old photographs\n"
                  "Grandma hums a tune about the winter")

LRCLIB_DB = [
    {"id": 101, "trackName": "Paper Lanterns", "artistName": "Marea Alta", "albumName": "Noches de Sal",
     "duration": 201.0, "instrumental": False, "plainLyrics": LANTERNS, "syncedLyrics": None},
    {"id": 102, "trackName": "Paper Lanterns", "artistName": "The Attic Choir", "albumName": "Boxes",
     "duration": 180.0, "instrumental": False, "plainLyrics": OTHER_LANTERNS, "syncedLyrics": None},
    {"id": 103, "trackName": "Paper Lanterns (Acoustic Cover)", "artistName": "Lena Strand", "albumName": "Covers",
     "duration": 199.0, "instrumental": False, "plainLyrics": LANTERNS, "syncedLyrics": None},
    {"id": 201, "trackName": "Tormenta de Cristal", "artistName": "Los Faros", "albumName": "Puerto",
     "duration": 214.0, "instrumental": False, "plainLyrics": TORMENTA, "syncedLyrics": None},
    {"id": 301, "trackName": "Copper Garden", "artistName": "Ivy Lane", "albumName": "Stairs",
     "duration": 190.0, "instrumental": False, "plainLyrics": GARDEN, "syncedLyrics": None},
]

DEEZER_DB = [
    (9001, "Tormenta de Cristal", "Los Faros", 812345, 214),
    (9002, "Paper Lanterns", "Marea Alta", 650000, 201),
    (9003, "Paper Lanterns", "Lena Strand", 12000, 199),
    (9004, "Cristal", "Neon Ruta", 400000, 180),
    (9005, "Copper Garden", "Ivy Lane", 30000, 190),
]


def dz(tid, title, artist, rank, duration):
    return {"id": tid, "title": title, "title_short": title, "link": f"https://www.deezer.com/track/{tid}",
            "duration": duration, "rank": rank, "explicit_lyrics": False,
            "preview": f"https://cdnt-preview.dzcdn.net/api/1/1/{tid}.mp3", "type": "track",
            "artist": {"id": tid + 1, "name": artist, "type": "artist"},
            "album": {"id": 7, "title": "Album", "cover_medium": f"https://cdn-images.dzcdn.net/{tid}.jpg",
                      "type": "album"}}


class Catalog:
    """MockTransport handler answering LRCLIB / Deezer / Genius from the invented catalog."""

    def __init__(self, deezer_status=200, genius_hits=None, slow_s=0.0):
        self.calls: list[httpx.Request] = []
        self.deezer_status, self.genius_hits, self.slow_s = deezer_status, genius_hits or [], slow_s

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.slow_s:
            time.sleep(self.slow_s)
        p, q = request.url.path, dict(request.url.params)
        host = request.url.host
        if host == "lrclib.net" and p == "/api/search":
            words = set(tokenize(q.get("track_name") or q.get("q") or ""))
            artist = set(tokenize(q.get("artist_name") or ""))
            hits = [r for r in LRCLIB_DB if words and words <= set(tokenize(r["trackName"]))
                    and (not artist or artist <= set(tokenize(r["artistName"])))]
            return httpx.Response(200, json=hits)
        if host == "lrclib.net" and p == "/api/get":
            for r in LRCLIB_DB:
                if (tokenize(r["trackName"]) == tokenize(q.get("track_name", ""))
                        and tokenize(r["artistName"]) == tokenize(q.get("artist_name", ""))):
                    return httpx.Response(200, json=r)
            return httpx.Response(404, json={"statusCode": 404})
        if host == "api.deezer.com" and p == "/search":
            if self.deezer_status != 200:
                return httpx.Response(self.deezer_status, json={})
            words = set(tokenize(q.get("q", "")))
            hits = [dz(*t) for t in DEEZER_DB if set(tokenize(t[1])) <= words or set(tokenize(t[2])) & words]
            return httpx.Response(200, json={"data": hits, "total": len(hits)})
        if host == "api.genius.com":
            assert request.headers.get("Authorization") == "Bearer secret-token"
            return httpx.Response(200, json={"response": {"hits": self.genius_hits}})
        return httpx.Response(404, json={})

    def hosts(self):
        return [(c.url.host, c.url.path) for c in self.calls]


def make(handler, index=None, offline=False, **kw):
    web = WebClient(transport=httpx.MockTransport(handler), host_limits=FAST, offline=offline)
    index = index if index is not None else WorldLyricsIndex()
    sources = kw.pop("sources", None)
    return WorldSearch(index, web, sources, **kw), index, web


def indexed(*songs):
    idx = WorldLyricsIndex()
    for title, artist, lyrics, rank in songs:
        idx.upsert(WorldSong.create(title, artist, lyrics=lyrics, rank=rank, source="sync:test"))
    return idx


# ---------------------------------------------------------------------------- helpers
def test_query_helpers():
    toks = tokenize("oh oh tormenta de cristal no me dejes caer\ntormenta de cristal yeah")
    phrases = query_phrases(toks, 3)
    assert phrases and phrases[0] == "tormenta de cristal"
    assert title_in_query("Tormenta de Cristal (Remastered 2011)", toks) == 1.0
    assert title_in_query("Cristal", toks) == 1.0
    assert 0 < title_in_query("Cristal Azul", toks) < 1.0
    assert title_in_query("Something Else", toks) == 0.0
    assert popularity_bonus(0) == 0 and 0 < popularity_bonus(1000) < popularity_bonus(10 ** 6) <= 0.03
    links = song_links("Tormenta de Cristal", "Los Faros")
    assert links["spotify"].startswith("https://open.spotify.com/search/Tormenta%20de%20Cristal")
    assert links["youtube"].startswith("https://www.youtube.com/results?search_query=Tormenta+de+Cristal")
    assert links["deezer"].startswith("https://www.deezer.com/search/")
    assert song_links("a", "b", "https://www.deezer.com/track/1")["deezer"] == "https://www.deezer.com/track/1"
    assert DEFAULT_THRESHOLDS["world"] == {"accept": 0.62, "margin": 0.08, "uncertain": 0.45}


def test_no_query_for_too_few_words():
    ws, _, _ = make(Catalog())
    for text in ("", "oh yeah", "la la la la", "de la que"):
        assert ws.search(text).status == "no_query"


def test_index_only_offline_and_online_sources_skipped():
    idx = indexed(("Copper Garden", "Ivy Lane", GARDEN, 30000), ("Paper Lanterns", "Marea Alta", LANTERNS, 650000))
    handler = Catalog()
    ws, _, _ = make(handler, idx)
    res = ws.search("whispers under the broken stairs copper garden calls my name", online=False)
    assert res.status == "ok" and not res.online
    top = res.candidates[0]
    assert (top.title, top.artist) == ("Copper Garden", "Ivy Lane") and top.sources == ["index"]
    assert top.song_id == "world:" + idx.get_by_key("Copper Garden", "Ivy Lane").id
    assert top.score >= 0.62 and top.coverage > 0.6
    assert res.sources["index"]["status"] == "ok"
    assert res.sources["lrclib"]["status"] == res.sources["deezer"]["status"] == "skipped"
    assert res.sources["claude"]["status"] == "unavailable"
    assert handler.calls == []                      # nothing left the machine
    d = res.to_dict()
    assert json.loads(json.dumps(d))["candidates"][0]["title"] == "Copper Garden"


def test_online_title_search_verifies_learns_and_then_works_offline():
    handler = Catalog()
    ws, idx, web = make(handler)
    text = "bailando sobre la tormenta de cristal\ntu voz me llama desde el puerto\ntormenta de cristal no me dejes caer"
    res = ws.search(text, language="es")
    assert res.status == "ok", res.sources
    top = res.candidates[0]
    assert (top.title, top.artist) == ("Tormenta de Cristal", "Los Faros")
    assert "deezer" in top.sources
    # Deezer's candidate verified: the LRCLIB title search (a fallback) was not needed
    assert res.sources["lrclib"] == {"status": "skipped", "n": 0, "elapsed_s": 0.0, "message": "not needed"}
    assert not any(c.url.path == "/api/search" and "artist_name" not in dict(c.url.params) for c in handler.calls)
    assert top.rank == 812345 and top.cover_url and top.links["deezer"] == "https://www.deezer.com/track/9001"
    assert top.preview_url.endswith("9001.mp3") and top.language == "es"
    assert top.score > 0.62 and top.score <= 1.0
    # the song with only "cristal" in its title has no lyrics evidence
    assert all(c.title != "Cristal" for c in res.candidates)
    dec = decide([{"song_id": c.song_id, "score": c.score} for c in res.candidates], "world", calibration={})
    assert dec["status"] == "match"
    # learned: next time it is found offline from the local index
    learned = idx.get_by_key("Tormenta de Cristal", "Los Faros")
    assert learned is not None and learned.source == "lookup" and learned.rank == 812345 and learned.deezer_id == 9001
    assert learned.lrclib_id == 201
    n_calls = len(handler.calls)
    res2 = ws.search(text, online=False)
    assert res2.candidates[0].song_id == top.song_id and res2.candidates[0].sources == ["index"]
    assert len(handler.calls) == n_calls


def test_deezer_candidate_gets_lyrics_from_lrclib_get():
    handler = Catalog()
    web = WebClient(transport=httpx.MockTransport(handler), host_limits=FAST)
    idx = WorldLyricsIndex()
    ws = WorldSearch(idx, web, [LocalIndexSource(idx), DeezerTitleSource(DeezerClient(web))])
    res = ws.search("whispers under broken stairs\ncopper garden calls my name\nrunning through the fields")
    assert res.status == "ok"
    top = res.candidates[0]
    assert (top.title, top.artist, top.sources) == ("Copper Garden", "Ivy Lane", ["deezer"])
    assert ("lrclib.net", "/api/get") in handler.hosts()
    assert res.sources["lyrics_fetch"]["n"] >= 1


def test_same_title_different_song_ranks_below_and_covers_collapse():
    handler = Catalog()
    ws, _, _ = make(handler, learn=False)
    res = ws.search("paper lanterns drifting on the river\nwe count the silver lights until the morning\n"
                    "paper lanterns carry me home")
    assert res.status == "ok"
    titles = [(c.title, c.artist) for c in res.candidates]
    # the cover by Lena Strand has the same lyrics: collapsed into the more popular original
    assert titles[0] == ("Paper Lanterns", "Marea Alta")
    assert ("Paper Lanterns (Acoustic Cover)", "Lena Strand") not in titles
    assert ("Paper Lanterns", "Lena Strand") not in titles
    other = [c for c in res.candidates if c.artist == "The Attic Choir"]
    assert not other or other[0].score < res.candidates[0].score - 0.2


def test_unrelated_words_give_no_candidates():
    handler = Catalog()
    ws, _, _ = make(handler, learn=False)
    res = ws.search("the quarterly budget meeting moved to thursday afternoon please bring the spreadsheets")
    assert res.status == "no_candidates" and res.candidates == []


def test_source_failure_is_recorded_not_raised():
    handler = Catalog(deezer_status=500)
    ws, _, _ = make(handler)
    res = ws.search("bailando sobre la tormenta de cristal\ntormenta de cristal no me dejes caer")
    assert res.sources["deezer"]["status"] == "error"
    assert res.status == "ok" and res.candidates[0].title == "Tormenta de Cristal"


class _SlowSource:
    name, label, online, requires_key = "slow", "Slow", True, False

    def available(self):
        return True

    def propose(self, query_text, tokens, language):
        time.sleep(2.0)
        return [Candidate(title="Copper Garden", artist="Ivy Lane", source="slow")]


class _BrokenSource:
    name, label, online, requires_key = "broken", "Broken", False, False

    def available(self):
        raise RuntimeError("boom")

    def propose(self, query_text, tokens, language):
        raise AssertionError("never called")


def test_deadline_and_broken_sources():
    idx = indexed(("Copper Garden", "Ivy Lane", GARDEN, 30000))
    ws, _, _ = make(Catalog(), idx, sources=[LocalIndexSource(idx), _SlowSource(), _BrokenSource()],
                    deadline_s=0.6)
    t = time.perf_counter()
    res = ws.search("whispers under broken stairs copper garden calls my name")
    assert time.perf_counter() - t < 1.5
    assert res.sources["slow"] == {"status": "error", "n": 0, "elapsed_s": res.sources["slow"]["elapsed_s"],
                                   "message": "timed out"}
    assert res.sources["broken"]["status"] == "unavailable"
    assert res.status == "ok" and res.candidates[0].title == "Copper Garden"


def test_title_followed_only_by_version_words_groups_with_the_original():
    from melodyai.world.search import title_version_of

    assert title_version_of("Harbor Lights", "Harbor Lights (Spanish Mix)")
    assert title_version_of("Harbor Lights - 2013 Remaster", "Harbor Lights (Live)") is False   # same normalised title
    assert not title_version_of("Harbor Lights", "Harbor Lights Forever")
    sung = "Sunlight on the harbor wall\nSeagulls write my name in chalk"
    original = sung + "\nCopper bells across the bay\nLantern keepers count the waves\nNorthern ferries sleep at dawn"
    mix = sung + "\nLas campanas del puerto\nLos faroleros cuentan olas\nTransbordadores al amanecer"
    sequel = sung + "\nForty winters gone since then\nI still hear the foghorn call\nOld friends gather by the pier"
    idx = indexed(("Harbor Lights", "Quiet Pier", original, 950000), ("Harbor Lights (Spanish Mix)", "Flamenco Duo", mix, 600000),
                  ("Harbor Lights Forever", "Other Band", sequel, 100))
    ws, _, _ = make(Catalog(), idx, learn=False)
    res = ws.search("sunlight on the harbor wall\nseagulls write my name in chalk", online=False)
    artists = [c.artist for c in res.candidates]
    assert artists[0] == "Quiet Pier" and "Flamenco Duo" not in artists     # the mix collapsed into the original
    assert "Other Band" in artists                                          # another title is another song


def test_same_lyrics_under_other_titles_collapse_into_the_popular_recording():
    lyr = "Sunlight on the harbor wall\nSeagulls write my name in chalk\nSunlight on the harbor wall"
    idx = indexed(("Harbor Wall (Karaoke Version)", "Quiet Band", lyr, 100), ("Harbor Song", "Loud Band", lyr, 900000))
    ws, _, _ = make(Catalog(), idx, learn=False)
    res = ws.search("sunlight on the harbor wall seagulls write my name in chalk", online=False)
    assert [c.artist for c in res.candidates] == ["Loud Band"]
    top = res.candidates[0]
    assert top.score == min(1.0, top.alignment_score + popularity_bonus(900000))
    assert set(top.sources) == {"index"}


def test_genius_source_needs_token_and_is_verified():
    hits = [{"type": "song", "result": {"title": "Copper Garden", "primary_artist": {"name": "Ivy Lane"}}},
            {"type": "song", "result": {"title": "Not A Song Here", "primary_artist": {"name": "Nobody"}}}]
    handler = Catalog(genius_hits=hits)
    web = WebClient(transport=httpx.MockTransport(handler), host_limits=FAST)
    assert not GeniusSource(web, token="").available()
    g = GeniusSource(web, token="secret-token")
    assert g.available()
    idx = WorldLyricsIndex()
    ws = WorldSearch(idx, web, [g])
    res = ws.search("whispers under broken stairs copper garden calls my name")
    assert [c.title for c in res.candidates] == ["Copper Garden"]
    assert res.candidates[0].sources == ["genius"]
    genius_req = next(c for c in handler.calls if c.url.host == "api.genius.com")
    assert "secret-token" not in str(genius_req.url)
    # the token never reaches the response cache
    rows = web._db.execute("SELECT key FROM responses").fetchall()
    assert all("secret-token" not in r[0] for r in rows)


class _FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.kwargs = payload, stop_reason, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(stop_reason=self.stop_reason,
                               content=[SimpleNamespace(type="text", text=json.dumps(self.payload))])


def test_claude_source_with_fake_client():
    msgs = _FakeMessages({"songs": [{"title": "Tormenta de Cristal", "artist": "Los Faros", "confidence": 0.8},
                                    {"title": "", "artist": "x", "confidence": 1}]})
    src = ClaudeSource(api_key="", client=SimpleNamespace(messages=msgs), model="claude-opus-5-5")
    assert src.available()
    cands = src.propose("tu voz me llama desde el puerto viejo", [], "es")
    assert [(c.title, c.artist, c.prior) for c in cands] == [("Tormenta de Cristal", "Los Faros", 0.8)]
    kw = msgs.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_config"]["format"]["type"] == "json_schema" and kw["output_config"]["effort"] == "low"
    assert "temperature" not in kw and "thinking" not in kw and "top_p" not in kw
    assert "puerto viejo" in kw["messages"][0]["content"]
    refused = ClaudeSource(api_key="", client=SimpleNamespace(messages=_FakeMessages({"songs": []}, "refusal")))
    assert refused.propose("some words here", [], None) == []
    # no key -> unavailable, whatever the SDK
    assert not ClaudeSource(api_key="").available()


def test_default_sources_and_describe(monkeypatch):
    monkeypatch.delenv("MELODYAI_GENIUS_TOKEN", raising=False)
    monkeypatch.delenv("MELODYAI_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    web = WebClient(transport=httpx.MockTransport(Catalog()), host_limits=FAST)
    idx = WorldLyricsIndex()
    names = [s.name for s in default_sources(idx, web)]
    assert names == ["index", "lrclib", "deezer", "genius", "claude"]
    ws = WorldSearch(idx, web)
    desc = {d["name"]: d for d in ws.describe_sources()}
    assert desc["index"]["available"] and desc["lrclib"]["available"] and desc["deezer"]["available"]
    assert not desc["genius"]["available"] and not desc["claude"]["available"]
    assert desc["claude"]["requires_key"] and not desc["lrclib"]["requires_key"]
    assert [s.name for s in default_sources(idx, None)] == ["index", "genius", "claude"]
    offline = WorldSearch(idx, None)
    res = offline.search("whispers under broken stairs copper garden calls my name", online=True)
    assert not res.online and res.status == "no_candidates"


def test_lrclib_source_uses_title_phrases():
    handler = Catalog()
    web = WebClient(transport=httpx.MockTransport(handler), host_limits=FAST)
    src = LrclibTitleSource(LrclibClient(web), max_phrases=2)
    toks = tokenize("tormenta de cristal no me dejes caer tormenta de cristal")
    cands = src.propose("", toks, "es")
    assert cands and all(c.lyrics for c in cands) and cands[0].title == "Tormenta de Cristal"
    sent = [dict(c.url.params).get("track_name") for c in handler.calls if c.url.path == "/api/search"]
    assert "tormenta de cristal" in sent and len(sent) <= 2
    assert normalize_title(cands[0].title) == "tormenta de cristal"


# ---------------------------------------------------------------------------- versions vs different songs
FAROL_A = ("Dame tu farol esta noche de invierno\nLas gaviotas duermen en el muelle gris\n"
           "Dame tu farol que me pierdo en el puerto")
FAROL_B = ("Dame tu farol esta noche de verano\nBailamos descalzos sobre la arena\n"
           "Nadie sabe el nombre de la luna")


class _Fixed:
    """Offline source proposing fixed candidates with lyrics and Deezer metadata."""

    name, label, online, requires_key = "fixed", "Fixed", False, False

    def __init__(self, *cands):
        self.cands = cands

    def available(self):
        return True

    def propose(self, query_text, tokens, language):
        return list(self.cands)


def _meta(tid, title, artist, rank):
    from melodyai.world.deezer import TrackMeta

    return TrackMeta(deezer_id=tid, title=title, title_short=title, artist=artist, artist_id=tid, album=None,
                     duration_s=200.0, rank=rank, isrc=None, cover_url=None, preview_url=None, link=None)


def _two_songs(title_a, title_b, lyrics_b=FAROL_B):
    a = Candidate(title_a, "Banda Inventada", lyrics=FAROL_A, meta=_meta(1, title_a, "Banda Inventada", 40_000),
                  source="fixed")
    b = Candidate(title_b, "Estrella Ficticia", lyrics=lyrics_b, meta=_meta(2, title_b, "Estrella Ficticia", 950_000),
                  source="fixed")
    ws = WorldSearch(WorldLyricsIndex(), None, [_Fixed(a, b)], learn=False)
    return ws.search("dame tu farol esta noche de invierno", online=False)


def test_different_songs_sharing_a_title_or_a_line_are_not_merged():
    """Regression: a popular song with the same (or a containing) title and one similar line swallowed the sung
    song, took its alignment score and became a high-confidence match."""
    for title_b in ("Farol", "Mi Farol", "Otra Cancion"):
        res = _two_songs("Farol", title_b)
        top = res.candidates[0]
        assert top.artist == "Banda Inventada" and top.alignment_score > 0.99, title_b
        assert {c.artist for c in res.candidates} == {"Banda Inventada", "Estrella Ficticia"}
        other = next(c for c in res.candidates if c.artist == "Estrella Ficticia")
        assert other.alignment_score < 0.95


def test_versions_with_the_same_lyrics_collapse_but_keep_their_own_score():
    # same lyrics under another title: one song; the popular recording wins only if it aligns about as well
    res = _two_songs("Farol (Karaoke Version)", "Farol", lyrics_b=FAROL_A)
    assert [c.artist for c in res.candidates] == ["Estrella Ficticia"]
    assert set(res.candidates[0].sources) == {"fixed"}
    # a popular version that aligns clearly worse does not take over the better-aligned recording's score
    worse = FAROL_A.replace("esta noche de invierno", "aquella tarde de otono")
    res = _two_songs("Farol", "Farol (En Vivo)", lyrics_b=worse)
    top = res.candidates[0]
    assert top.artist == "Banda Inventada" and top.alignment_score > 0.95


def test_partly_translated_cover_with_the_same_title_collapses_into_the_original():
    sung = "dame tu farol esta noche de invierno"
    translated = FAROL_A.split("\n")[0] + "\nlantern of my winter nights\nthe harbor sleeps without you\nmy lantern"
    res = _two_songs("Farol", "Farol", lyrics_b=translated + "\n" + sung)   # both hold the sung line verbatim
    assert [c.artist for c in res.candidates] == ["Estrella Ficticia"]        # the popular recording stands for both
    assert res.candidates[0].alignment_score > 0.99


def test_decorated_titles_and_medleys_are_versions():
    from melodyai.world.search import base_title, medley_marked, title_version_of

    assert base_title("Farol (Acousique)") == "farol" and base_title("Farol - Directo en Lima") == "farol"
    assert title_version_of("Farol", "Farol (Acousique)") and title_version_of("Farol", "Farol - Sesiones del Puerto")
    assert title_version_of("Farol", "Mareas / Farol / Gaviotas")          # one of the songs a medley lists
    assert not title_version_of("Farol", "Mi Farol") and not title_version_of("Farol", "Farolito (Remix)")
    assert not title_version_of("Broken Stairs", "Copper Garden")          # same length, unrelated
    assert medley_marked("Mareas / Farol / Gaviotas") and medley_marked("Farol Mashup")
    assert medley_marked("Verano Mix 4") and medley_marked("Enganchados del Puerto")
    assert not medley_marked("Farol") and not medley_marked("24/7") and not medley_marked("AC/DC Blues")


def test_a_popular_medley_holding_the_sung_lines_never_stands_for_the_song():
    medley = FAROL_A + "\nOtra cancion empieza aqui\nY se termina en el muelle\nUna tercera para bailar"
    for title in ("Mareas / Farol / Gaviotas", "Verano Mix 4"):
        res = _two_songs("Farol", title, lyrics_b=medley)        # the medley is far more popular
        assert [c.artist for c in res.candidates] == ["Banda Inventada"], title
        assert res.candidates[0].title == "Farol"


def test_lyrics_overlap_and_version_marks():
    from melodyai.world.search import lyrics_overlap, version_marked

    assert lyrics_overlap(FAROL_A, FAROL_A.upper().replace("\n", "\n\n")) == 1.0
    assert lyrics_overlap(FAROL_A, FAROL_B) < 0.5
    assert lyrics_overlap(FAROL_A, "") == 0.0
    assert version_marked("Farol (Karaoke Version)", "X") and version_marked("Farol - En Vivo", "X")
    assert version_marked("Farol", "Karaoke Hits Band") and version_marked("Farol", "The Tribute Ensemble")
    assert not version_marked("Live Forever", "Sunny Garage")              # "live" is part of this title
    assert not version_marked("Farol", "Banda Inventada")


def test_popular_original_joins_only_if_it_aligns_as_well():
    """Regression: for an unpopular best match, the most popular same-title track by another artist was added
    whenever its lyrics scored >= 0.25, then merged into the sung song."""
    rec = {"A": {"id": 1, "trackName": "Farol", "artistName": "Banda Inventada", "albumName": "X", "duration": 200.0,
                 "instrumental": False, "plainLyrics": FAROL_A, "syncedLyrics": None},
           "B": {"id": 2, "trackName": "Farol", "artistName": "Estrella Ficticia", "albumName": "Y",
                 "duration": 190.0, "instrumental": False, "plainLyrics": FAROL_B, "syncedLyrics": None}}

    def handler(req):
        p, q = req.url.path, dict(req.url.params)
        if req.url.host == "lrclib.net" and p == "/api/get":
            for r in rec.values():
                if tokenize(r["artistName"]) == tokenize(q.get("artist_name", "")):
                    return httpx.Response(200, json=r)
            return httpx.Response(404, json={})
        if req.url.host == "lrclib.net" and p == "/api/search":
            return httpx.Response(200, json=[rec["A"]])
        if req.url.host == "api.deezer.com" and p == "/search":
            hits = [dz(11, "Farol", "Banda Inventada", 40_000, 200)]
            if q.get("q", "").strip().lower() == "farol":
                hits = [dz(22, "Farol", "Estrella Ficticia", 950_000, 190)] + hits
            return httpx.Response(200, json={"data": hits})
        return httpx.Response(404, json={})

    ws, _, _ = make(handler, learn=False)
    res = ws.search("dame tu farol esta noche de invierno")
    assert [(c.title, c.artist) for c in res.candidates][0] == ("Farol", "Banda Inventada")
    assert all(c.artist != "Estrella Ficticia" or c.score < res.candidates[0].score - 0.05 for c in res.candidates)
    dec = decide([{"song_id": c.song_id, "score": c.score} for c in res.candidates], "world", calibration={})
    assert dec["status"] == "match" and res.candidates[0].artist == "Banda Inventada"


# ---------------------------------------------------------------------------- deadlines and previews
def test_lyrics_lookups_start_as_soon_as_a_source_answers():
    """Regression: lyric lookups waited for the slowest source (up to PROPOSE_SHARE of the deadline), so a slow
    LRCLIB title search left too little time to fetch the lyrics of Deezer's (right) proposal."""
    base = Catalog()

    def handler(req):
        if req.url.host == "lrclib.net":
            time.sleep(2.0 if req.url.path == "/api/search" else 1.0)   # slow title search, slowish /api/get
        return base(req)

    web = WebClient(transport=httpx.MockTransport(handler), host_limits=FAST)
    idx = WorldLyricsIndex()
    ws = WorldSearch(idx, web, [LocalIndexSource(idx), LrclibTitleSource(LrclibClient(web)),
                                DeezerTitleSource(DeezerClient(web))], learn=False, deadline_s=3.0)
    res = ws.search("whispers under broken stairs\ncopper garden calls my name\nrunning through the fields")
    assert res.sources["lrclib"]["message"] == "not needed"           # a fallback: Deezer's candidate verified
    assert res.status == "ok" and res.candidates[0].title == "Copper Garden", res.sources
    assert res.sources["lyrics_fetch"]["n"] >= 1


def test_no_request_is_sent_after_the_search_returns():
    """Regression: abandoned sources kept sending requests (and holding rate-limit tokens) after the deadline."""
    started: list[float] = []

    def handler(req):
        started.append(time.perf_counter())
        time.sleep(0.4)
        return httpx.Response(200, json=[] if req.url.host == "lrclib.net" else {"data": []})

    web = WebClient(transport=httpx.MockTransport(handler),
                    host_limits={"lrclib.net": HostLimit(6, 1.3, 1), "api.deezer.com": HostLimit(40, 8.0, 1)})
    idx = WorldLyricsIndex()
    ws = WorldSearch(idx, web, [LrclibTitleSource(LrclibClient(web), max_phrases=4),
                                DeezerTitleSource(DeezerClient(web))], learn=False, deadline_s=1.0)
    res = ws.search("velvet harbor lanterns drifting over copper rivers tonight\nsilver gulls are calling the boats")
    returned = time.perf_counter()
    time.sleep(1.5)
    assert res.status == "no_candidates"
    assert started and all(t < returned for t in started), f"{sum(t >= returned for t in started)} sent late"


def test_web_requests_respect_the_deadline_and_give_back_tokens():
    from melodyai.world.web import WebError, request_deadline

    sent: list[str] = []

    def handler(req):
        sent.append(str(req.url))
        return httpx.Response(200, json=[])

    web = WebClient(transport=httpx.MockTransport(handler),
                    host_limits={"lrclib.net": HostLimit(capacity=1, refill_per_s=0.5, max_concurrent=2)})
    with request_deadline(time.monotonic() - 1):
        with pytest.raises(WebError) as err:
            LrclibClient(web).search(track_name="anything at all")
        assert err.value.kind == "deadline"
        # the sources' own worker threads inherit the deadline: nothing is sent
        with pytest.raises(WebError):
            LrclibTitleSource(LrclibClient(web)).propose(
                "", tokenize("tormenta de cristal no me dejes caer tormenta de cristal"), None)
    assert sent == []
    assert web.get_json("https://lrclib.net/api/search", {"q": "a"}) == []        # takes the only token
    with request_deadline(time.monotonic() + 1.0):                              # next token in 2 s: too late
        with pytest.raises(WebError) as err:
            web.get_json("https://lrclib.net/api/search", {"q": "b"})
        assert err.value.kind == "rate_limited"
    assert len(sent) == 1
    assert web._limiter("lrclib.net")._tokens > -0.5                           # the refused request gave it back


def test_expired_previews_are_not_returned_and_deezer_id_is():
    from melodyai.world.deezer import fresh_preview, storable_preview

    old = "https://cdnt-preview.dzcdn.net/api/1/1/a/b/c.mp3?hdnea=exp=1700000000~acl=/api/*~hmac=ab"
    new = old.replace("1700000000", str(int(time.time()) + 600))
    plain = "https://cdns-preview-1.dzcdn.net/stream/c-123-1.mp3"
    assert fresh_preview(old) is None and fresh_preview(new) == new and fresh_preview(plain) == plain
    assert fresh_preview(new, now=time.time() + 590) is None                     # about to expire
    assert storable_preview(new) is None and storable_preview(plain) == plain and storable_preview(None) is None
    idx = WorldLyricsIndex()
    idx.upsert(WorldSong.create("Copper Garden", "Ivy Lane", lyrics=GARDEN, rank=30000, deezer_id=9005,
                                preview_url=old, source="sync:test"))
    ws, _, _ = make(Catalog(), idx, learn=False)
    top = ws.search("whispers under the broken stairs copper garden calls my name", online=False).candidates[0]
    assert top.preview_url is None and top.deezer_id == 9005 and top.to_dict()["deezer_id"] == 9005


def test_claude_needs_its_own_key_or_an_explicit_opt_in(monkeypatch):
    from melodyai.world.sources import claude_api_key

    monkeypatch.delenv("MELODYAI_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MELODYAI_CLAUDE_SOURCE", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key-of-another-tool")
    assert claude_api_key() == "" and not ClaudeSource().available()           # a key in the shell is not consent
    monkeypatch.setenv("MELODYAI_CLAUDE_SOURCE", "1")
    assert claude_api_key() == "generic-key-of-another-tool"
    monkeypatch.delenv("MELODYAI_CLAUDE_SOURCE")
    monkeypatch.setenv("MELODYAI_ANTHROPIC_API_KEY", "melodyai-key")
    assert claude_api_key() == "melodyai-key"


def test_source_labels_are_english_for_the_ui_to_translate():
    web = WebClient(transport=httpx.MockTransport(Catalog()), host_limits=FAST)
    labels = {d["name"]: d["label"] for d in WorldSearch(WorldLyricsIndex(), web).describe_sources()}
    assert labels["lrclib"].startswith("LRCLIB") and all(lbl.isascii() for lbl in labels.values())


def test_result_says_why_online_sources_were_not_asked():
    """The UI tells "online search is off (Settings)" from "this server is offline" by ``offline_reason``."""
    idx = indexed(("Copper Garden", "Ivy Lane", GARDEN, 30000))
    text = "whispers under the broken stairs copper garden calls my name"
    ws, _, _ = make(Catalog(), idx, learn=False)
    on = ws.search(text)
    assert on.online and on.offline_reason == "" and on.to_dict()["offline_reason"] == ""
    off = ws.search(text, online=False)
    assert not off.online and off.offline_reason == "setting_off"
    assert off.sources["lrclib"] == {"status": "skipped", "n": 0, "elapsed_s": 0.0, "message": "online search off"}
    ws_offline, _, _ = make(Catalog(), idx, offline=True, learn=False)
    srv = ws_offline.search(text, online=True)
    assert not srv.online and srv.offline_reason == "server_offline"
    assert srv.sources["deezer"]["status"] == "skipped" and srv.sources["deezer"]["message"] == "server offline"
    assert srv.candidates[0].title == "Copper Garden"


# ---------------------------------------------------------------------------- early exit on a known song
def _synced(*songs):
    """An index filled like the popular-songs sync: popular songs with their Deezer ids."""
    idx = WorldLyricsIndex()
    for k, (title, artist, lyrics, rank) in enumerate(songs):
        idx.upsert(WorldSong.create(title, artist, lyrics=lyrics, rank=rank, deezer_id=7000 + k, source="sync:test"))
    return idx


def test_a_decisive_index_match_settles_the_search_without_online_requests():
    """Regression: a song the local index already matched perfectly still waited for the online sources (the
    typed query of a famous song took the whole 9 s deadline). A settled answer costs one cacheable title search
    (no more popular namesake here: Deezer's "Copper Garden" is the same artist), and none the next time."""
    idx = _synced(("Copper Garden", "Ivy Lane", GARDEN, 650_000), ("Paper Lanterns", "Marea Alta", LANTERNS, 650_000))
    handler = Catalog()
    ws, _, _ = make(handler, idx, learn=False)
    res = ws.search("whispers under broken stairs copper garden calls my name")
    assert res.status == "ok" and res.candidates[0].title == "Copper Garden" and res.candidates[0].sources == ["index"]
    for name in ("lrclib", "deezer"):
        assert res.sources[name] == {"status": "skipped", "n": 0, "elapsed_s": 0.0, "message": "local index match"}
    assert [(c.url.host, dict(c.url.params)["q"]) for c in handler.calls] == [("api.deezer.com", "copper garden")]
    # a partly remembered query needs a clearer lead, which it has over the other indexed song (title cached)
    res = ws.search("whispers under the stairs\ncopper garden is calling\nrunning to the yellow fields")
    assert res.candidates[0].title == "Copper Garden" and res.sources["deezer"]["message"] == "local index match"
    assert len(handler.calls) == 1
    # a slow title search does not hold the answer back for more than GRACE_S
    handler = Catalog(slow_s=2.0)
    ws, _, _ = make(handler, idx, learn=False)
    t = time.perf_counter()
    res = ws.search("whispers under broken stairs copper garden calls my name")
    assert time.perf_counter() - t < ws.GRACE_S + 0.5
    assert res.candidates[0].title == "Copper Garden" and res.sources["deezer"]["message"] == "local index match"
    assert all(c.url.host == "api.deezer.com" for c in handler.calls)


def test_a_settled_cover_still_looks_for_its_more_popular_original():
    """Regression (held-out query): the index held a popular cover (rank >= 300k, no version mark) but not the
    more popular original, so the settled answer skipped the online sources and named the cover's artist. Only
    a namesake by an artist the index knows unsettles the answer: Deezer may rank a cover channel's upload above
    the original (dev song)."""
    gets: list[str] = []

    def run(namesake_artist):
        idx = _synced(("Lighthouse", "Nina Vela", LIGHTHOUSE_COVER, 400_000),
                      ("Paper Lanterns", "Marea Alta", LANTERNS, 650_000))     # Marea Alta: an artist the index knows
        calls: list[str] = []

        def handler(req):
            p, q = req.url.path, dict(req.url.params)
            calls.append(req.url.host)
            if req.url.host == "api.deezer.com":
                hits = []
                if q["q"].strip().lower() == "lighthouse":
                    hits = [dz(31, "Lighthouse", "Nina Vela", 400_000, 190),
                            dz(32, "Lighthouse", namesake_artist, 900_000, 200)]
                return httpx.Response(200, json={"data": hits})
            if p == "/api/get":
                gets.append(q["artist_name"])
                if q["artist_name"] == namesake_artist:
                    return httpx.Response(200, json={"id": 7, "trackName": "Lighthouse", "artistName": namesake_artist,
                                                     "albumName": "x", "duration": 200.0, "instrumental": False,
                                                     "plainLyrics": LIGHTHOUSE_ORIGINAL, "syncedLyrics": None})
                return httpx.Response(404, json={})
            return httpx.Response(200, json=[])

        ws, _, _ = make(handler, idx, learn=False)
        text = "keepers of the lantern never sleeping the foghorn sings a lullaby for two"
        assert ws._index_verdict(_entries(ws, text), tokenize(text)) == "settled"
        return ws.search(text), calls

    res, _ = run("Marea Alta")
    top = res.candidates[0]
    assert (top.title, top.artist) == ("Lighthouse", "Marea Alta") and top.alignment_score > 0.99
    assert set(top.sources) == {"index", "deezer"} and gets == ["Marea Alta"]
    assert res.sources["deezer"]["status"] == "ok"                 # the answer was only probable: online asked
    gets.clear()
    res, calls = run("Lighthouse Hits Music")                       # unknown to the index: the answer stands
    assert (res.candidates[0].title, res.candidates[0].artist) == ("Lighthouse", "Nina Vela")
    assert res.sources["deezer"]["message"] == "local index match" and calls == ["api.deezer.com"] and gets == []


def test_a_probable_index_match_still_verifies_what_the_online_sources_propose():
    """Regression (dev query, a song the index lacks): a probable index answer gave the online sources 1 s for
    their proposals AND the lyric lookups. Deezer answered in ~0.5 s, the lookup of its (right) proposal was cut
    at the grace, and the index's near miss (0.90) became a confident match."""
    near = ("Whispers under broken stairs\nCopper garden calls my rain\nPaper boats along the shore\n"
            "Nobody is waiting there")
    other = ("Whispers under broken chairs\nCopper garden calls my train\nCandles on the window sill\n"
             "Every night the same")
    idx = _synced(("Broken Stairs", "Delta Nine", near, 650_000), ("Window Sill", "Neon Ruta", other, 640_000))
    base = Catalog()

    def slow(req):
        time.sleep(0.5 if req.url.host == "api.deezer.com" else 0.6)
        return base(req)

    ws, _, _ = make(slow, idx, learn=False)
    text = "whispers under broken stairs copper garden calls my name"
    assert ws._index_verdict(_entries(ws, text), tokenize(text)) == "probable"
    t = time.perf_counter()
    res = ws.search(text)
    assert time.perf_counter() - t < ws.HEAD_START_S + ws.GRACE_S + ws.PROBABLE_FETCH_S + 0.5
    top = res.candidates[0]
    assert (top.title, top.artist) == ("Copper Garden", "Ivy Lane") and top.alignment_score > 0.99, res.sources
    assert res.candidates[1].title == "Broken Stairs" and res.sources["lyrics_fetch"]["n"] == 1


def test_an_unpopular_or_weak_index_match_still_asks_the_online_sources():
    # an unpopular recording may be a cover: the original is only found online
    idx = _synced(("Copper Garden", "Ivy Lane", GARDEN, 900))
    handler = Catalog()
    ws, _, _ = make(handler, idx, learn=False)
    res = ws.search("whispers under broken stairs copper garden calls my name")
    assert res.sources["deezer"]["status"] == "ok" and handler.calls
    # nothing in the index for these words: the full online search finds the song
    handler = Catalog()
    ws, _, _ = make(handler, _synced(("Copper Garden", "Ivy Lane", GARDEN, 650_000)), learn=False)
    res = ws.search("bailando sobre la tormenta de cristal\ntormenta de cristal no me dejes caer")
    assert res.candidates[0].title == "Tormenta de Cristal" and res.sources["deezer"]["status"] == "ok"
    assert res.sources["lyrics_fetch"]["n"] >= 1 or "lrclib" in res.candidates[0].sources


def test_a_probable_index_match_gives_the_online_sources_only_a_short_grace():
    """Two indexed songs share the sung line (no clear lead), or the query has few informative words: the online
    sources may still add rivals, but only for ``GRACE_S``; nothing is sent after the search returns."""
    twin = "Copper garden calls my name\nSilver rivers under frozen bridges\nNobody waits for the evening ferry"
    idx = _synced(("Copper Garden", "Ivy Lane", GARDEN, 650_000), ("Frozen Bridges", "Delta Nine", twin, 640_000))
    started: list[float] = []
    base = Catalog()

    def slow(req):
        started.append(time.perf_counter())
        time.sleep(0.8)
        return base(req)

    ws, _, _ = make(slow, idx, learn=False)
    ws.GRACE_S = 0.4
    t = time.perf_counter()
    res = ws.search("copper garden calls my name")
    returned = time.perf_counter()
    assert returned - t < 1.6
    assert {c.title for c in res.candidates[:2]} == {"Copper Garden", "Frozen Bridges"}
    assert res.sources["deezer"]["status"] == "error" and res.sources["deezer"]["message"] == "timed out"
    time.sleep(1.0)
    assert started and all(s < returned for s in started)
    # a clear lead with too few informative words is probable as well
    text = "whispers under broken stairs copper garden"
    assert ws._index_verdict(_entries(ws, text), tokenize(text)) == "settled"
    ws.EXIT_MIN_WORDS = 7
    assert ws._index_verdict(_entries(ws, text), tokenize(text)) == "probable"


def _entries(ws, text):
    entries = {}
    for c in LocalIndexSource(ws.index).propose(text, tokenize(text), None):
        ws._add(entries, c)
    return entries


def test_index_verdict_needs_a_clearer_lead_for_a_partial_alignment():
    from melodyai.lyrics.align import AlignResult
    from melodyai.world.search import _Entry

    ws = WorldSearch(WorldLyricsIndex(), None, [], learn=False)

    def entry(key, score, rank=650_000, lyrics=""):
        e = _Entry(key, cands=[Candidate(key, "Artist " + key)], sources=["index"])
        e.world_song = WorldSong.create(key, "Artist " + key, lyrics=lyrics or f"{key} one\n{key} two", rank=rank)
        e.align = AlignResult(score, score, score, "", (0, 0), (0, 1))
        return e

    def verdict(*pairs, words=("whispers", "broken", "stairs")):
        return ws._index_verdict({k: entry(k, s, *rest) for k, s, *rest in pairs}, list(words))

    assert verdict(("alpha", 1.0), ("beta", 0.84)) == "settled"           # lead 0.16 >= 0.15
    assert verdict(("alpha", 1.0), ("beta", 0.9)) == "probable"           # near-perfect, no clear lead
    assert verdict(("alpha", 0.8), ("beta", 0.6)) is None                 # 0.8 needs a lead of 0.225
    assert verdict(("alpha", 0.8), ("beta", 0.5)) == "settled"
    assert verdict(("alpha", 0.6), ("beta", 0.25)) == "probable"          # partial, lead >= 0.3
    assert verdict(("alpha", 0.6), ("beta", 0.45)) is None
    assert verdict(("alpha", 0.45)) is None
    assert verdict(("alpha", 1.0, 900)) is None                           # unpopular: maybe a cover
    assert verdict(("alpha", 1.0), words=("whispers", "stairs")) == "probable"


# ---------------------------------------------------------------------------- LRCLIB title search as a fallback
def test_lrclib_title_search_runs_only_when_nothing_else_matched():
    """LRCLIB title search never found a song Deezer missed in the real-song check, and costs scarce LRCLIB
    requests: it runs only when the other sources' candidates, once verified, matched nothing."""
    handler = Catalog(deezer_status=500)                  # Deezer fails: nothing to verify
    ws, _, _ = make(handler, learn=False)
    res = ws.search("bailando sobre la tormenta de cristal\ntormenta de cristal no me dejes caer")
    assert res.sources["deezer"]["status"] == "error" and res.sources["lrclib"]["status"] == "ok"
    assert res.candidates[0].title == "Tormenta de Cristal" and res.candidates[0].sources == ["lrclib"]
    # Deezer proposes only a song whose lyrics do not match: the fallback runs as well (the text repeats a phrase)
    handler = Catalog()
    ws, _, _ = make(handler, learn=False)
    res = ws.search("silver gulls are calling neon ruta\nneon ruta the ferry leaves at noon")
    assert res.sources["deezer"]["n"] >= 1 and res.sources["lrclib"]["status"] == "ok"
    title_searches = [c for c in handler.calls if c.url.path == "/api/search" and "artist_name" not in dict(c.url.params)]
    assert title_searches
    # ... but a title search needs a repeated phrase (a chorus usually repeats the title), and says so
    handler = Catalog()
    ws, _, _ = make(handler, learn=False)
    res = ws.search("silver gulls are calling\nneon ruta cristal\nthe ferry leaves at noon")
    assert res.sources["lrclib"] == {"status": "skipped", "n": 0, "elapsed_s": 0.0, "message": "no repeated phrase"}
    assert not [c for c in handler.calls if c.url.path == "/api/search" and "artist_name" not in dict(c.url.params)]
    assert LrclibTitleSource.fallback and not getattr(DeezerTitleSource, "fallback", False)


def test_deezer_title_phrases_are_a_last_resort_when_no_window_finds_anything():
    """Regression (held-out query): a short line, not repeated, with one misheard word: every window and retry
    found nothing, title phrases were off (no repeated phrase), and the search gave up with requests to spare."""
    from melodyai.lyrics.normalize import normalize_text
    from melodyai.world.sources import query_phrases

    text = "whispers under broken chairs copper garden falls my name"
    phrase = query_phrases(tokenize(text), 2)[0]          # what Deezer, in this mock, resolves to the title
    sent: list[str] = []
    base = Catalog()

    def handler(req):
        if req.url.host != "api.deezer.com":
            return base(req)
        q = normalize_text(dict(req.url.params)["q"])
        sent.append(q)
        hits = [dz(9005, "Copper Garden", "Ivy Lane", 30000, 190), dz(9100, "Garden Party", "Somebody", 900_000, 200)]
        return httpx.Response(200, json={"data": hits if q == phrase else []})

    web = WebClient(transport=httpx.MockTransport(handler), host_limits=FAST)
    src = DeezerTitleSource(DeezerClient(web))
    ws = WorldSearch(WorldLyricsIndex(), web, [src], learn=False)
    res = ws.search(text)
    assert res.candidates and (res.candidates[0].title, res.candidates[0].artist) == ("Copper Garden", "Ivy Lane")
    assert res.candidates[0].alignment_score > 0.62
    i = sent.index(phrase)                  # after the windows and their retries, within the request budget
    assert sent.count(phrase) == 1 and 2 < i < src.max_requests
    assert all(c.title != "Garden Party" for c in res.candidates)    # its whole title was not sung: not proposed
    # a transcript that a window finds costs no title-phrase request
    handler = Catalog()
    web = WebClient(transport=httpx.MockTransport(handler), host_limits=FAST)
    text = "whispers under broken stairs copper garden calls my name"
    cands = DeezerTitleSource(DeezerClient(web)).propose(text, tokenize(text), None)
    assert cands and cands[0].title == "Copper Garden" and len(handler.calls) == 2      # the two windows only


# ---------------------------------------------------------------------------- weak evidence
def test_a_weak_partial_best_match_that_does_not_stand_out_is_capped():
    """Regression: wrong songs reached 0.64-0.74 on short generic lines (two common words matched out of four)."""
    from melodyai.lyrics.align import AlignResult
    from melodyai.world.search import LOW_EVIDENCE_CAP, _Entry

    def pair(key, score, matched):
        return _Entry(key, cands=[Candidate(key, "Artist")]), AlignResult(score, score, score, matched, (0, 0), (0, 4))

    toks = tokenize("velvet harbor under copper skies tonight")
    guard = WorldSearch._evidence_guard
    assert WorldSearch.matched_words(toks, AlignResult(1, 1, 1, "the velvet harbour / skies", (0, 1), (0, 6))) == 3
    out = guard(toks, [pair("a", 0.72, "the velvet harbor sleeps"), pair("b", 0.6, "copper rain")])
    assert out[0][1].score == out[0][0].align.score == pytest.approx(0.601) and out[1][1].score == 0.6  # just ahead
    out = guard(toks, [pair("a", 0.7, "the velvet harbor sleeps"), pair("b", 0.52, "copper rain")])
    assert out[0][1].score == LOW_EVIDENCE_CAP == 0.55
    # the cap lowers the confidence but keeps the order: a more popular runner-up does not overtake the best match
    a, b = pair("a", 0.623, "the velvet harbor sleeps"), pair("b", 0.548, "copper rain")
    a[0].meta, b[0].meta = _meta(1, "a", "Artist", 5_000), _meta(2, "b", "Artist", 950_000)
    ranked = WorldSearch._rank(guard(toks, [a, b]))
    assert [e.key for e, _ in ranked] == ["a", "b"] and 0 < ranked[0][1] - ranked[1][1] < DEFAULT_THRESHOLDS["world"][
        "margin"]
    assert ranked[0][1] < a[1].score + popularity_bonus(5_000)                 # it was capped
    for kept in ([pair("a", 0.72, "the velvet harbor sleeps"), pair("b", 0.5, "x y")],         # clear lead
                 [pair("a", 0.72, "velvet harbor under copper"), pair("b", 0.6, "x y")],         # 4 words matched
                 [pair("a", 0.95, "the velvet harbor sleeps"), pair("b", 0.9, "x y")],           # near-perfect
                 [pair("a", 0.72, "the velvet harbor sleeps")]):                                  # no rival
        assert [x[1].score for x in guard(toks, kept)] == [x[1].score for x in kept]
    assert guard(toks, []) == []


def test_lyric_lookups_stop_once_a_candidate_matches():
    """LRCLIB is a free shared service: after MIN_FETCH lookups, a verified near-perfect match ends them."""
    tracks = [dz(9100 + k, f"Song {chr(65 + k)}", f"Band {chr(65 + k)}", 500_000 - k, 190) for k in range(6)]
    lyrics = {"Song A": GARDEN}
    gets: list[str] = []

    def handler(req):
        q = dict(req.url.params)
        if req.url.host == "api.deezer.com":
            return httpx.Response(200, json={"data": tracks, "total": len(tracks)})
        if req.url.path == "/api/get":
            gets.append(q["track_name"])
            title = q["track_name"]
            if title != "Song A":
                time.sleep(0.3)                          # the matching lyrics arrive first
            return httpx.Response(200, json={"id": 1, "trackName": title, "artistName": q["artist_name"],
                                             "albumName": "x", "duration": 190.0, "instrumental": False,
                                             "plainLyrics": lyrics.get(title, OTHER_LANTERNS), "syncedLyrics": None})
        return httpx.Response(200, json=[])

    web = WebClient(transport=httpx.MockTransport(handler), host_limits=FAST)
    idx = WorldLyricsIndex()
    ws = WorldSearch(idx, web, [DeezerTitleSource(DeezerClient(web))], learn=False)
    res = ws.search("whispers under broken stairs\ncopper garden calls my name\nrunning through the yellow fields")
    assert res.candidates[0].title == "Song A" and res.candidates[0].score > 0.9
    assert len(gets) == ws.MIN_FETCH == 3 and "Song A" in gets
    assert res.sources["lyrics_fetch"]["message"].startswith("3 looked up")
    # nothing matches: every allowed lookup is made
    lyrics.clear()
    web = WebClient(transport=httpx.MockTransport(handler), host_limits=FAST)
    ws = WorldSearch(idx, web, [DeezerTitleSource(DeezerClient(web))], learn=False)
    res = ws.search("whispers under broken stairs\ncopper garden calls my name\nrunning through the yellow fields")
    msg = res.sources["lyrics_fetch"]["message"]
    assert ", 0 skipped" in msg and int(msg.split()[0]) > ws.MIN_FETCH


# ---------------------------------------------------------------------------- covers that match better than the original
LIGHTHOUSE_COVER = ("Salt on the window of the lighthouse\nWe painted the stairs a midnight blue\n"
                    "Keepers of the lantern never sleeping\nThe foghorn sings a lullaby for two\n"
                    "Gulls are circling over the harbour\nOne more winter waiting for the tide")
LIGHTHOUSE_ORIGINAL = ("Salt on the window of the lighthouse\nWe painted the stairs in navy and grey\n"
                       "Keepers of the lantern never sleeping\nThe foghorn sings a lullaby for two\n"
                       "Ships are passing far beyond the breakers\nNorthern lights are dancing on the bay\n"
                       "Nobody remembers who was first\nWe count the waves until the break of day")


def test_a_popular_original_stands_for_a_cover_that_matches_its_transcription_better():
    """Pilot regression (a Spanish hit): an unpopular cover's lyrics matched the sung line exactly (1.00), the
    original's LRCLIB transcription of that line differed (0.71), so the cover was returned. Records with the same
    title that share a good part of their lines are one song: the original stands for it, with the best score."""
    cover = Candidate("Lighthouse", "Harbour Covers", lyrics=LIGHTHOUSE_COVER,
                      meta=_meta(1, "Lighthouse", "Harbour Covers", 120_000), source="fixed")
    original = Candidate("Lighthouse", "Marea Alta", lyrics=LIGHTHOUSE_ORIGINAL,
                         meta=_meta(2, "Lighthouse", "Marea Alta", 900_000), source="fixed")
    ws = WorldSearch(WorldLyricsIndex(), None, [_Fixed(cover, original)], learn=False)
    text = "we painted the stairs a midnight blue keepers of the lantern"
    a_cover = ws.index.score_lyrics(text, LIGHTHOUSE_COVER).score
    a_orig = ws.index.score_lyrics(text, LIGHTHOUSE_ORIGINAL).score
    assert a_cover > 0.95 and a_cover - a_orig > COVER_SCORE_WINDOW         # too far apart for a version window
    res = ws.search(text, online=False)
    assert [c.artist for c in res.candidates] == ["Marea Alta"]
    assert abs(res.candidates[0].alignment_score - a_cover) < 1e-9
    # another song with that title and one shared line keeps its own place and score
    other = Candidate("Lighthouse", "Delta Nine", lyrics="Salt on the window of the lighthouse\nNothing else to say\n"
                      "Rain and rain and rain again", meta=_meta(3, "Lighthouse", "Delta Nine", 950_000), source="fixed")
    res = WorldSearch(WorldLyricsIndex(), None, [_Fixed(cover, other)], learn=False).search(text, online=False)
    assert res.candidates[0].artist == "Harbour Covers"


def test_a_recording_of_about_the_same_popularity_that_lacks_the_sung_verse_does_not_stand_for_it():
    """Regression (held-out query): of two records sharing the lyrics, the slightly more popular one was shown with
    the group's best alignment (1.00) although its own lyrics lack the sung verse (only the remix has it)."""
    chorus = ("Salt on the window of the lighthouse\nKeepers of the lantern never sleeping\n"
              "The foghorn sings a lullaby for two\n")
    remix = chorus + "Turn it up the night is young and electric\nEvery wave is dancing to the bass line\n" + chorus
    original = chorus + "Ships are passing far beyond the breakers\nNorthern lights are dancing on the bay\n" + chorus
    text = "the foghorn sings a lullaby for two turn it up the night is young and electric"

    def search(remix_rank):
        r = Candidate("Lighthouse (Club Remix)", "DJ Nova", lyrics=remix, source="fixed",
                      meta=_meta(1, "Lighthouse (Club Remix)", "DJ Nova", remix_rank))
        o = Candidate("Lighthouse", "Marea Alta", lyrics=original, meta=_meta(2, "Lighthouse", "Marea Alta", 700_000),
                      source="fixed")
        return WorldSearch(WorldLyricsIndex(), None, [_Fixed(r, o)], learn=False).search(text, online=False)

    ws = WorldSearch(WorldLyricsIndex(), None, [], learn=False)
    own = ws.index.score_lyrics(text, original).score
    assert 0.25 < own < 1.0 - COVER_SCORE_WINDOW                    # it is verified, but lacks half of the line
    res = search(600_000)
    assert [(c.title, c.artist) for c in res.candidates] == [("Lighthouse (Club Remix)", "DJ Nova")]
    assert res.candidates[0].alignment_score > 0.99
    # a little-known recording is still shown as the clearly more popular one (cf. the Lighthouse test above)
    res = search(40_000)
    assert [(c.title, c.artist) for c in res.candidates] == [("Lighthouse", "Marea Alta")]
    assert res.candidates[0].alignment_score > 0.99
