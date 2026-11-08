"""Open-world web clients: WebClient (rate limits, cache, Retry-After, offline), LRCLIB and Deezer parsing/lookup.

No network: every response comes from ``httpx.MockTransport``. All lyrics below are invented.
"""

import json
import logging
import sqlite3
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from melodyai.world.deezer import DeezerClient, TrackMeta
from melodyai.world.lrclib import (
    LrclibClient,
    LyricsRecord,
    artist_similarity,
    clean_title,
    primary_artist,
    strip_synced,
    title_similarity,
)
from melodyai.world.web import (
    DEFAULT_USER_AGENT,
    HostLimit,
    WebClient,
    WebError,
    cache_key,
    parse_retry_after,
)

FAST = {"lrclib.net": HostLimit(1000, 1000.0, 3), "api.deezer.com": HostLimit(1000, 1000.0, 6),
        "example.org": HostLimit(1000, 1000.0, 4)}

# ---------------------------------------------------------------------------- invented fixtures
PLAIN = "Paper lanterns drifting on the river\nWe count the lights until the morning\n\nPaper lanterns, carry me home"
SYNCED = ("[ar:Marea Alta]\n[ti:Paper Lanterns]\n[length: 03:21]\n[00:12.30] Paper lanterns drifting on the river\n"
          "[00:18.05]<00:18.05> We <00:18.40> count the lights\n[00:25.00]\n[00:31.10][01:40.00] Carry me home\n")


def lrc(rid, track, artist, *, album="Noches de Sal", duration=201.0, plain=PLAIN, synced=SYNCED, instrumental=False):
    return {"id": rid, "name": track, "trackName": track, "artistName": artist, "albumName": album,
            "duration": duration, "instrumental": instrumental, "hasWordSync": False,
            "plainLyrics": None if instrumental else plain, "syncedLyrics": None if instrumental else synced,
            "lyricsfile": "version: '1.0'\nmetadata:\n  title: x\nlines: []\n"}


def dz_track(tid, title, artist, rank, *, artist_id=11, title_short=None, duration=201, album="Noches de Sal",
             **extra):
    return {"id": tid, "readable": True, "title": title, "title_short": title_short or title, "title_version": "",
            "link": f"https://www.deezer.com/track/{tid}", "duration": duration, "rank": rank,
            "explicit_lyrics": False, "preview": f"https://cdnt-preview.dzcdn.net/api/1/1/{tid}.mp3?hdnea=exp=1~hmac=x",
            "md5_image": "abc", "artist": {"id": artist_id, "name": artist, "type": "artist"},
            "album": {"id": 9, "title": album, "cover": "https://api.deezer.com/album/9/image",
                      "cover_medium": "https://cdn-images.dzcdn.net/images/cover/abc/250x250.jpg", "type": "album"},
            "type": "track", **extra}


def json_response(body, status=200, headers=None):
    return lambda request: httpx.Response(status, json=body, headers=headers)


NOT_FOUND = json_response({"message": "Failed to find specified track", "name": "TrackNotFound", "statusCode": 404},
                          404)


class Router:
    """MockTransport handler: queued responders per URL path (the last one repeats); records requests."""

    def __init__(self):
        self.routes: dict[str, list] = {}
        self.calls: list[httpx.Request] = []
        self.lock = threading.Lock()

    def add(self, path, *responders):
        self.routes.setdefault(path, []).extend(responders)
        return self

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            self.calls.append(request)
            queue = self.routes.get(request.url.path)
            if not queue:
                responder = NOT_FOUND
            else:
                responder = queue[0] if len(queue) == 1 else queue.pop(0)
        return responder(request)

    def params(self, i=-1) -> dict:
        return dict(self.calls[i].url.params)


class FakeTime:
    def __init__(self):
        self.t = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.t

    def time(self):
        return 1.7e9 + self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def make_web(router, ft=None, **kw) -> WebClient:
    kw.setdefault("host_limits", FAST)
    web = WebClient(transport=httpx.MockTransport(router), **kw)
    if ft is not None:
        web._clock, web._sleep, web._now = ft.monotonic, ft.sleep, ft.time
    return web


# ---------------------------------------------------------------------------- WebClient
def test_user_agent_params_and_cache_key():
    router = Router().add("/a", json_response({"ok": 1}))
    web = make_web(router)
    assert web.get_json("https://example.org/a", {"b": "2", "a": "x y", "none": None}) == {"ok": 1}
    req = router.calls[0]
    assert req.headers["User-Agent"] == DEFAULT_USER_AGENT and DEFAULT_USER_AGENT.startswith("MelodyAI/")
    assert "github.com/Ale-Newport/MelodyAI" in DEFAULT_USER_AGENT
    assert dict(req.url.params) == {"a": "x y", "b": "2"}
    assert cache_key("https://h/p", {"b": 2, "a": "x y", "c": None}) == "https://h/p?a=x+y&b=2"
    assert parse_retry_after("7") == 7.0 and parse_retry_after(None) is None and parse_retry_after("soon") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now_epoch=1445412470.0) == pytest.approx(10.0)


def test_cache_hit_avoids_second_request_and_ttl_expiry(tmp_path):
    ft = FakeTime()
    router = Router().add("/a", json_response({"v": 1}))
    web = make_web(router, ft, cache_path=tmp_path / "web.db")
    assert web.get_json("https://example.org/a", {"q": "x"}, ttl_s=60) == {"v": 1}
    assert web.get_json("https://example.org/a", {"q": "x"}, ttl_s=60) == {"v": 1}
    assert len(router.calls) == 1
    st = web.stats()
    assert st["requests"] == 1 and st["cache_hits"] == 1 and st["hosts"]["example.org"]["cache_hits"] == 1
    assert st["cache"]["entries"] == 1
    ft.t += 61
    web.get_json("https://example.org/a", {"q": "x"}, ttl_s=60)
    assert len(router.calls) == 2
    web.get_json("https://example.org/a", {"q": "x"}, cache=False)   # bypasses the cache
    assert len(router.calls) == 3
    web.close()
    # persistent: a new client on the same file serves the entry without the network
    web2 = make_web(router, ft, cache_path=tmp_path / "web.db")
    assert web2.get_json("https://example.org/a", {"q": "x"}, ttl_s=60) == {"v": 1} and len(router.calls) == 3


def test_404_returns_none_and_is_negative_cached():
    ft = FakeTime()
    router = Router()   # every path -> 404
    web = make_web(router, ft)
    assert web.get_json("https://example.org/missing", negative_ttl_s=100) is None
    assert web.get_json("https://example.org/missing", negative_ttl_s=100) is None
    assert len(router.calls) == 1 and web.stats()["negative_hits"] == 1
    ft.t += 101
    assert web.get_json("https://example.org/missing") is None and len(router.calls) == 2


def test_retry_after_short_wait_retries_once():
    ft = FakeTime()
    router = Router().add("/a", json_response({"name": "RateLimitExceeded"}, 429, {"Retry-After": "1"}),
                          json_response({"v": 2}))
    web = make_web(router, ft)
    assert web.get_json("https://example.org/a", max_wait_s=4) == {"v": 2}
    assert len(router.calls) == 2
    assert len(ft.sleeps) == 1 and 1.0 <= ft.sleeps[0] <= 1.3 + 1e-9        # Retry-After + <= 30 % jitter
    st = web.stats()
    assert st["retries"] == 1 and st["rate_limited"] == 1 and st["errors"] == 0
    # 503 without Retry-After: default 1 s
    router.add("/b", lambda r: httpx.Response(503, text="busy"), json_response({"v": 3}))
    assert web.get_json("https://example.org/b") == {"v": 3}
    assert 1.0 <= ft.sleeps[-1] <= 1.3 + 1e-9


def test_retry_after_too_long_raises_and_blocks_the_host():
    ft = FakeTime()
    router = Router().add("/a", json_response({"name": "RateLimitExceeded"}, 429, {"Retry-After": "120"}))
    router.add("/b", json_response({"v": 1}))
    web = make_web(router, ft)
    with pytest.raises(WebError) as ei:
        web.get_json("https://example.org/a", max_wait_s=4)
    assert ei.value.kind == "rate_limited" and ei.value.retry_after == pytest.approx(120)
    assert len(router.calls) == 1 and ft.sleeps == []
    with pytest.raises(WebError):          # the host is blocked: nothing is sent while the block lasts
        web.get_json("https://example.org/b")
    assert len(router.calls) == 1
    ft.t += 121
    assert web.get_json("https://example.org/b") == {"v": 1} and len(router.calls) == 2
    # throttled again on the retry -> error after exactly two requests
    router.add("/c", json_response({}, 429, {"Retry-After": "1"}))
    with pytest.raises(WebError, match="rate limited"):
        web.get_json("https://example.org/c")
    assert len(router.calls) == 4


def test_http_network_and_decode_errors_are_not_cached():
    router = Router().add("/e500", lambda r: httpx.Response(500, json={"name": "UnknownError"}))
    router.add("/e400", lambda r: httpx.Response(400, text="Failed to deserialize query string"))
    router.add("/bad", lambda r: httpx.Response(200, text="<html>edge</html>"))

    def boom(request):
        raise httpx.ConnectError("unreachable", request=request)

    router.add("/down", boom)
    web = make_web(router)
    for path, kind, status in (("/e500", "http", 500), ("/e400", "http", 400), ("/bad", "decode", 200),
                               ("/down", "network", None)):
        for _ in range(2):
            with pytest.raises(WebError) as ei:
                web.get_json(f"https://example.org{path}")
            assert ei.value.kind == kind and ei.value.status == status and ei.value.host == "example.org"
    assert len(router.calls) == 8 and web.stats()["errors"] == 8 and web.stats()["cache"]["entries"] == 0


def test_offline_mode_serves_cache_only(tmp_path):
    ft = FakeTime()
    router = Router().add("/a", json_response({"v": 1}))
    online = make_web(router, ft, cache_path=tmp_path / "c.db")
    online.get_json("https://example.org/a", ttl_s=10)
    online.get_json("https://example.org/gone")       # negative entry
    online.close()
    offline = make_web(router, ft, cache_path=tmp_path / "c.db", offline=True)
    ft.t += 1000                                       # expired entries are still served offline
    assert offline.get_json("https://example.org/a", ttl_s=10) == {"v": 1}
    assert offline.get_json("https://example.org/gone") is None
    with pytest.raises(WebError) as ei:
        offline.get_json("https://example.org/never")
    assert ei.value.kind == "offline"
    assert len(router.calls) == 2 and offline.stats()["offline"] is True


def test_secret_headers_never_reach_cache_or_logs(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="melodyai.world.web")
    secret = "SECRET-TOKEN-987654"
    router = Router().add("/search", json_response({"response": {"hits": []}}))
    web = make_web(router, cache_path=tmp_path / "c.db")
    web.get_json("https://example.org/search", {"q": "words sung by user"}, headers={"Authorization": f"Bearer {secret}"})
    assert router.calls[0].headers["Authorization"] == f"Bearer {secret}"
    # same key without the header: a cache hit (headers are not part of the key)
    web.get_json("https://example.org/search", {"q": "words sung by user"})
    assert len(router.calls) == 1
    web.close()
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir())
    assert secret.encode() not in raw
    db = sqlite3.connect(tmp_path / "c.db")
    rows = db.execute("SELECT key, body FROM responses").fetchall()
    assert all(secret not in k and secret.encode() not in zlib.decompress(b) for k, b in rows)
    assert caplog.records and all(secret not in r.getMessage() for r in caplog.records)
    assert all("words" not in r.getMessage() for r in caplog.records)    # query text is not logged


def test_token_bucket_spaces_requests():
    ft = FakeTime()
    router = Router().add("/a", json_response({}))
    web = make_web(router, ft, host_limits={"example.org": HostLimit(capacity=2, refill_per_s=1.0, max_concurrent=4)})
    for i in range(4):
        web.get_json("https://example.org/a", {"i": i})
    assert ft.sleeps == pytest.approx([1.0, 1.0])        # burst of 2, then 1 request/s
    ft.t += 10                                           # bucket refills (capped at capacity)
    for i in range(2):
        web.get_json("https://example.org/a", {"j": i})
    assert len(ft.sleeps) == 2


def test_concurrency_cap_and_single_flight_are_thread_safe():
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    def slow(request):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return httpx.Response(200, json={"path": request.url.path, "i": request.url.params.get("i")})

    router = Router().add("/a", slow).add("/same", slow)
    web = make_web(router, host_limits={"example.org": HostLimit(1000, 1000.0, max_concurrent=2)})
    with ThreadPoolExecutor(max_workers=12) as pool:
        out = list(pool.map(lambda i: web.get_json("https://example.org/a", {"i": i}), range(24)))
    assert [o["i"] for o in out] == [str(i) for i in range(24)]
    assert state["max"] == 2
    with ThreadPoolExecutor(max_workers=10) as pool:
        same = list(pool.map(lambda _: web.get_json("https://example.org/same"), range(10)))
    assert all(s == {"path": "/same", "i": None} for s in same)
    assert sum(1 for c in router.calls if c.url.path == "/same") == 1    # collapsed into one request
    st = web.stats()
    assert st["requests"] == 25 and st["cache_hits"] == 9 and st["errors"] == 0


def test_cache_is_pruned_below_its_size_limit(tmp_path):
    ft = FakeTime()
    rnd = __import__("random").Random(0)

    def blob(request):
        ft.t += 1        # distinct fetch times -> deterministic "oldest first"
        return httpx.Response(200, json={"x": "".join(rnd.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(3000))})

    router = Router().add("/a", blob)
    web = make_web(router, ft, cache_path=tmp_path / "c.db", max_cache_mb=0.02)
    for i in range(40):
        web.get_json("https://example.org/a", {"i": i})
    st = web.stats()["cache"]
    assert 0 < st["entries"] < 40 and st["bytes"] <= st["max_bytes"]
    n = len(router.calls)
    web.get_json("https://example.org/a", {"i": 39})      # newest kept
    assert len(router.calls) == n
    web.get_json("https://example.org/a", {"i": 0})       # oldest pruned
    assert len(router.calls) == n + 1


def test_drop_fields_and_in_band_errors_are_not_stored():
    router = Router().add("/api/search", json_response([lrc(1, "A", "B")]))
    router.add("/err", json_response({"error": {"type": "Exception", "message": "Quota limit exceeded", "code": 4}}))
    web = make_web(router)
    data = web.get_json("https://lrclib.net/api/search", {"q": "a"}, drop_fields=("lyricsfile",))
    assert "lyricsfile" not in data[0] and data[0]["plainLyrics"] == PLAIN
    body = web._db.execute("SELECT body FROM responses").fetchone()[0]
    assert b"lyricsfile" not in zlib.decompress(body)
    assert web.get_json("https://example.org/err")["error"]["code"] == 4
    web.get_json("https://example.org/err")
    assert sum(1 for c in router.calls if c.url.path == "/err") == 2


# ---------------------------------------------------------------------------- LRCLIB
def test_strip_synced_removes_tags_metadata_and_blank_lines():
    assert strip_synced(SYNCED) == ("Paper lanterns drifting on the river\nWe count the lights\nCarry me home")
    assert strip_synced("") == "" and strip_synced("[offset:+120]\n[01:02:03]\n") == ""


def test_lyrics_record_parsing_and_text():
    r = LyricsRecord.from_json(lrc(7, "Paper Lanterns", "Marea Alta"))
    assert (r.id, r.track, r.artist, r.album, r.duration_s) == (7, "Paper Lanterns", "Marea Alta", "Noches de Sal", 201.0)
    assert r.text == PLAIN and r.has_lyrics and r.to_dict()["synced"] == SYNCED
    synced_only = LyricsRecord.from_json(lrc(8, "T", "A", plain=""))
    assert synced_only.text.startswith("Paper lanterns drifting") and "[" not in synced_only.text
    inst = LyricsRecord.from_json(lrc(9, "T", "A", instrumental=True))
    assert inst.instrumental and inst.text == "" and not inst.has_lyrics
    assert LyricsRecord.from_json({"trackName": "no id"}) is None and LyricsRecord.from_json([]) is None
    legacy = LyricsRecord.from_json({"id": "3", "name": "Old", "artistName": "X", "duration": None})
    assert legacy.track == "Old" and legacy.duration_s is None and legacy.text == ""


def test_title_and_artist_normalisation():
    assert clean_title("Paper Lanterns - Remastered 2011") == "Paper Lanterns"
    assert clean_title("Faro (feat. Luz Norte) [Live]") == "Faro"
    assert clean_title("Bailando con lobos") == "Bailando con lobos"
    assert title_similarity("Canción del Faro", "cancion del faro (en vivo)") == 1.0
    assert title_similarity("Paper Lanterns", "Paper Lanterns - Live at the Hall") == 1.0
    assert title_similarity("Dance with Me", "Dance with Somebody") < 0.85
    assert artist_similarity("Marea Alta feat. Luz Norte", "Luz Norte") == 1.0
    assert artist_similarity("The Night Owls", "Night Owls") == 1.0
    assert artist_similarity("Marea Alta", "Coro Lejano") < 0.5
    assert primary_artist("Marea Alta feat. Luz Norte") == "Marea Alta"
    assert primary_artist("Coro Lejano, Luz Norte") == "Coro Lejano" and primary_artist("Luz x Sal") == "Luz"
    assert primary_artist("Sal & Pimienta") == "Sal & Pimienta" and primary_artist("AC/DC") == "AC/DC"


def test_lrclib_get_sends_signature_and_negative_caches_404():
    router = Router().add("/api/get", NOT_FOUND)
    lr = LrclibClient(make_web(router))
    assert lr.get("Paper Lanterns", "Marea Alta", "Noches de Sal", 200.6) is None
    assert router.params() == {"track_name": "Paper Lanterns", "artist_name": "Marea Alta",
                               "album_name": "Noches de Sal", "duration": "201"}
    assert lr.get("Paper Lanterns", "Marea Alta", "Noches de Sal", 200.6) is None
    assert len(router.calls) == 1                              # negative cache
    lr.get("Paper Lanterns", "Marea Alta", duration_s=0.2)     # invalid durations are not sent (API -> 400)
    assert "duration" not in router.params() and "album_name" not in router.params()
    assert lr.get("", "Marea Alta") is None and len(router.calls) == 2


def test_lrclib_search_parameters():
    router = Router().add("/api/search", json_response([lrc(1, "Paper Lanterns", "Marea Alta"), {"bad": 1}]))
    lr = LrclibClient(make_web(router))
    recs = lr.search(q="paper lanterns marea")
    assert [r.id for r in recs] == [1] and router.params() == {"q": "paper lanterns marea"}
    lr.search(track_name="Paper Lanterns", artist_name="Marea Alta", album_name=" ")
    assert router.params() == {"track_name": "Paper Lanterns", "artist_name": "Marea Alta"}
    assert lr.search() == [] and lr.search(q="  ") == [] and len(router.calls) == 2


def test_find_uses_the_exact_signature_when_it_is_good():
    router = Router().add("/api/get", json_response(lrc(5, "Paper Lanterns", "Marea Alta", duration=201)))
    lr = LrclibClient(make_web(router))
    rec = lr.find("Paper Lanterns", "Marea Alta", duration_s=201, album="Noches de Sal")
    assert rec.id == 5 and len(router.calls) == 1


def test_find_falls_back_to_search_and_picks_the_best_record():
    records = [
        lrc(1, "Paper Lanterns", "Marea Alta", instrumental=True, duration=201),          # instrumental
        lrc(2, "Paper Lanterns", "Coro Lejano", duration=201),                            # other artist
        lrc(3, "Paper Lanterns (Karaoke Version)", "Karaoke Stars", duration=201),        # karaoke
        lrc(4, "Paper Lanterns - Remastered 2011", "Marea Alta", duration=206.0),         # 5 s off
        lrc(5, "paper lanterns", "Marea Alta, Luz Norte", duration=202.4, plain=None),    # synced only, 1.4 s off
        lrc(6, "Paper Lanterns", "Marea Alta", duration=201, plain=None, synced=None),    # no lyrics
        lrc(7, "Other Song", "Marea Alta", duration=201),                                 # other title
    ]
    router = Router().add("/api/get", NOT_FOUND).add("/api/search", json_response(records))
    lr = LrclibClient(make_web(router))
    rec = lr.find("Paper Lanterns - Remastered 2011", "Marea Alta feat. Luz Norte", duration_s=201)
    assert rec.id == 5 and rec.text.startswith("Paper lanterns drifting")
    assert router.params() == {"track_name": "Paper Lanterns", "artist_name": "Marea Alta"}   # clean + primary
    # duration unknown: 4 and 5 tie on similarity -> the lowest id (stable order)
    assert lr.best_record([LyricsRecord.from_json(r) for r in records], "Paper Lanterns", "Marea Alta").id == 4


def test_find_duration_tolerance_and_fallback():
    lr = LrclibClient(make_web(Router()))
    near = [LyricsRecord.from_json(lrc(1, "Faro", "Marea Alta", duration=211.0))]
    assert lr.best_record(near, "Faro", "Marea Alta", 201.0).id == 1          # 10 s: other cut, same lyrics
    far = [LyricsRecord.from_json(lrc(1, "Faro", "Marea Alta", duration=260.0))]
    assert lr.best_record(far, "Faro", "Marea Alta", 201.0) is None           # 59 s: not the same song
    fuzzy = [LyricsRecord.from_json(lrc(1, "Faros", "Marea Alta", duration=211.0))]
    assert lr.best_record(fuzzy, "Faro", "Marea Alta", 201.0) is None         # off tolerance needs strong agreement
    both = [LyricsRecord.from_json(lrc(1, "Faro", "Marea Alta", duration=211.0)),
            LyricsRecord.from_json(lrc(2, "Faro (Live)", "Marea Alta", duration=202.0))]
    assert lr.best_record(both, "Faro", "Marea Alta", 201.0).id == 2          # within +/-3 s preferred


def test_find_rejects_mismatched_signature_and_handles_errors():
    router = Router().add("/api/get", json_response(lrc(9, "Completely Different", "Marea Alta")))
    router.add("/api/search", json_response([lrc(3, "Faro", "Marea Alta", duration=None)]))
    lr = LrclibClient(make_web(router))
    assert lr.find("Faro", "Marea Alta").id == 3 and len(router.calls) == 2
    router400 = Router().add("/api/get", lambda r: httpx.Response(400, json={"name": "ValidationError"}))
    router400.add("/api/search", json_response([lrc(3, "Faro", "Marea Alta")]))
    assert LrclibClient(make_web(router400)).find("Faro", "Marea Alta").id == 3
    router500 = Router().add("/api/get", lambda r: httpx.Response(500))
    with pytest.raises(WebError):
        LrclibClient(make_web(router500)).find("Faro", "Marea Alta")
    assert LrclibClient(make_web(Router())).find("Faro", "Marea Alta") is None   # 404 + empty search
    assert LrclibClient(make_web(Router())).find("  ", "Marea Alta") is None


# ---------------------------------------------------------------------------- Deezer
def dz(router, ft=None):
    client = DeezerClient(make_web(router, ft))
    if ft is not None:
        client._sleep = ft.sleep
    return client


def test_track_meta_parsing():
    t = TrackMeta.from_json(dz_track(42, "Faro (Remix)", "Marea Alta", 512345, title_short="Faro", isrc="ESX012300001"))
    assert (t.deezer_id, t.title, t.title_short, t.artist, t.artist_id, t.album) == (
        42, "Faro (Remix)", "Faro", "Marea Alta", 11, "Noches de Sal")
    assert t.duration_s == 201.0 and t.rank == 512345 and t.isrc == "ESX012300001" and not t.explicit
    assert t.cover_url.endswith("250x250.jpg") and t.link == "https://www.deezer.com/track/42"
    assert t.preview_url.startswith("https://cdnt-preview") and t.to_dict()["deezer_id"] == 42
    assert TrackMeta.from_json({"id": 1, "title": "x", "type": "episode"}) is None
    assert TrackMeta.from_json({"id": None, "title": "x"}) is None
    bare = TrackMeta.from_json({"id": 2, "title": "Bare"})
    assert bare.artist == "" and bare.rank == 0 and bare.preview_url is None and bare.cover_url is None


def test_search_title_sends_the_plain_phrase_and_reranks_by_title():
    data = [dz_track(1, "Otra Canción", "Coro Lejano", 900),                       # Deezer relevance order
            dz_track(2, "Faro", "Marea Alta", 10),                                 # title words in the phrase
            dz_track(3, "El Faro del Sur (Remix)", "Marea Alta", 5, title_short="El Faro del Sur")]   # exact
    router = Router().add("/search", json_response({"data": data, "total": 3}))
    client = dz(router)
    res = client.search_title('el "faro"  del sur', limit=5)
    assert [t.deezer_id for t in res] == [3, 2, 1]
    assert router.params() == {"q": "el faro del sur", "limit": "5"}   # no field syntax (not honoured live)
    client.search_tracks("marea alta faro", limit=3)
    assert router.params() == {"q": "marea alta faro", "limit": "3"}
    assert client.search_title('  " ') == [] and client.search_tracks("") == [] and len(router.calls) == 2


def test_find_prefers_best_match_and_breaks_ties_by_rank():
    data = [dz_track(1, "Faro", "Marea Alta", 400_000),
            dz_track(2, "Faro (Remix)", "Marea Alta", 900_000, title_short="Faro"),
            dz_track(3, "Faro (Karaoke Version)", "Karaoke Stars", 999_999, title_short="Faro"),
            dz_track(4, "Faros", "Marea Alta", 950_000)]
    router = Router().add("/search", json_response({"data": data, "total": 4}))
    client = dz(router)
    best = client.find("Faro - Remastered 2011", "Marea Alta feat. Luz Norte")
    assert best.deezer_id == 2                                    # tie on similarity -> higher rank
    assert router.params() == {"q": "Marea Alta Faro", "limit": "5"}
    assert len(router.calls) == 1
    # nothing agreeing with "<artist> <title>" -> one title-only fallback with a wider page, then None
    router2 = Router().add("/search", json_response({"data": [dz_track(5, "Otra", "Otro", 1)], "total": 1}))
    assert dz(router2).find("Faro", "Marea Alta") is None and len(router2.calls) == 2
    assert router2.params(0) == {"q": "Marea Alta Faro", "limit": "5"}
    assert router2.params(1) == {"q": "Faro", "limit": "25"}
    assert dz(router2).find("Faro", " ") is None and len(router2.calls) == 2


def test_deezer_error_payloads():
    ft = FakeTime()
    quota = json_response({"error": {"type": "Exception", "message": "Quota limit exceeded", "code": 4}})
    ok = json_response({"data": [dz_track(1, "Faro", "Marea Alta", 10)], "total": 1})
    router = Router().add("/chart/0/tracks", quota, ok)
    client = dz(router, ft)
    assert [t.deezer_id for t in client.chart_tracks(limit=10)] == [1]
    assert ft.sleeps == [5.0] and len(router.calls) == 2           # the error payload was not cached
    client.chart_tracks(limit=10)
    assert len(router.calls) == 2                                  # the good one was
    router.add("/chart/0/artists", quota)
    with pytest.raises(WebError) as ei:
        client.chart_artists()
    assert ei.value.kind == "rate_limited" and len(router.calls) == 4
    router.add("/playlist/123/tracks", json_response({"error": {"type": "DataException", "message": "no data",
                                                                "code": 800}}))
    assert client.playlist_tracks(123) == []
    router.add("/search", json_response({"error": {"type": "Exception", "message": "Invalid query", "code": 600}}))
    with pytest.raises(WebError) as ei:
        client.search_tracks("x")
    assert ei.value.kind == "api" and "600" in str(ei.value)


def test_deezer_quota_retry_never_sleeps_past_the_request_deadline():
    from melodyai.world.web import request_deadline

    quota = json_response({"error": {"type": "Exception", "message": "Quota limit exceeded", "code": 4}})
    router = Router().add("/chart/0/tracks", quota)
    ft = FakeTime()
    client = dz(router, ft)
    with request_deadline(time.monotonic() + 1.0), pytest.raises(WebError) as ei:
        client.chart_tracks(limit=10)                  # waiting 5 s would overrun the 1 s deadline: give up
    assert ei.value.kind == "rate_limited" and ft.sleeps == [] and len(router.calls) == 1


def test_playlist_tracks_follow_pagination():
    total = 230

    def page(request):
        index = int(request.url.params.get("index", 0))
        limit = int(request.url.params["limit"])
        items = [dz_track(1000 + i, f"Song {i}", "Marea Alta", i) for i in range(index, min(total, index + limit))]
        body = {"data": items, "total": total, "checksum": "x"}
        if index + limit < total:
            body["next"] = f"https://api.deezer.com/playlist/77/tracks?limit={limit}&index={index + limit}"
        return httpx.Response(200, json=body)

    router = Router().add("/playlist/77/tracks", page)
    client = dz(router)
    tracks = client.playlist_tracks(77, limit=500)
    assert [t.deezer_id for t in tracks] == [1000 + i for i in range(total)]
    assert [(c.url.params.get("index"), c.url.params["limit"]) for c in router.calls] == [
        (None, "100"), ("100", "100"), ("200", "100")]
    n = len(router.calls)
    assert len(client.playlist_tracks(77, limit=150)) == 150
    assert [c.url.params["limit"] for c in router.calls[n:]] == ["50"]   # page 1 (limit 100) came from the cache


def test_charts_playlists_and_artist_top_parsing():
    router = Router()
    router.add("/chart/116/artists", json_response({"data": [
        {"id": 5, "name": "Marea Alta", "position": 1, "type": "artist"}, {"id": None, "name": "x"}], "total": 2}))
    router.add("/search/playlist", json_response({"data": [
        {"id": 908622995, "title": "Viral Hits", "nb_tracks": 60, "link": "https://www.deezer.com/playlist/908622995",
         "user": {"id": 2, "name": "Deezer Viral Editor", "type": "user"}, "type": "playlist"},
        {"id": 1, "title": "mine", "nb_tracks": 3, "user": {"id": 9, "name": "someone"}, "type": "playlist"}],
        "total": 2}))
    router.add("/artist/5/top", json_response({"data": [dz_track(1, "Faro", "Marea Alta", 10)], "total": 1}))
    client = dz(router)
    assert client.chart_artists(116, limit=10) == [{"id": 5, "name": "Marea Alta", "position": 1}]
    pls = client.search_playlists("tiktok viral", limit=10)
    assert pls[0] == {"id": 908622995, "title": "Viral Hits", "nb_tracks": 60, "user": "Deezer Viral Editor",
                      "user_id": 2, "link": "https://www.deezer.com/playlist/908622995"}
    assert router.params() == {"q": "tiktok viral", "limit": "10"}
    assert [t.title for t in client.artist_top(5, limit=50)] == ["Faro"]
    assert router.calls[-1].url.path == "/artist/5/top" and router.params()["limit"] == "50"


def test_clients_share_one_web_client_across_threads():
    router = Router().add("/api/get", json_response(lrc(5, "Faro", "Marea Alta")))
    router.add("/search", json_response({"data": [dz_track(1, "Faro", "Marea Alta", 10)], "total": 1}))
    web = make_web(router)
    lr, dzc = LrclibClient(web), DeezerClient(web)

    def job(i):
        return (lr.find("Faro", "Marea Alta", duration_s=201 + i % 3).id, dzc.find("Faro", "Marea Alta").deezer_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(job, range(30)))
    assert set(out) == {(5, 1)}
    st = web.stats()
    assert st["requests"] == 4 and st["errors"] == 0                 # 3 distinct durations + 1 Deezer query
    assert st["hosts"]["lrclib.net"]["requests"] == 3 and st["hosts"]["api.deezer.com"]["requests"] == 1
    json.dumps(st)   # stats are JSON-serialisable


# ---------------------------------------------------------------------------- privacy of the user's words
def test_httpx_request_log_never_shows_the_query(caplog):
    """Regression: with the server's root logger at INFO, httpx logged every upstream URL, query words included."""
    caplog.set_level(logging.INFO)
    router = Router().add("/search", json_response({"data": []}))
    web = make_web(router)
    web.get_json("https://example.org/search", {"q": "zebra moonlight saxophone"}, private=True)
    web.get_json("https://example.org/search", {"q": "zebra moonlight trombone"})
    assert len(router.calls) == 2
    assert all("zebra" not in r.getMessage() for r in caplog.records)


def test_private_requests_are_hashed_short_lived_and_purgeable(tmp_path):
    from melodyai.world.web import PRIVATE_TTL_S

    ft = FakeTime()
    router = Router().add("/search", json_response({"data": [1]})).add("/chart", json_response({"data": [2]}))
    web = make_web(router, ft, cache_path=tmp_path / "c.db")
    words = "zebra moonlight saxophone"
    assert web.get_json("https://example.org/search", {"q": words}, ttl_s=7 * 86400, private=True) == {"data": [1]}
    assert web.get_json("https://example.org/search", {"q": words}, private=True) == {"data": [1]}   # cached
    web.get_json("https://example.org/chart")
    assert len(router.calls) == 2
    rows = dict(web._db.execute("SELECT key, expires_at - fetched_at FROM responses").fetchall())
    assert not any("zebra" in k for k in rows)                                  # never stored in plain text
    private = [k for k in rows if k.startswith("h:")]
    assert len(private) == 1 and rows[private[0]] <= PRIVATE_TTL_S
    # plain-text search keys cached by older versions are purged with the private ones
    web._db.execute("INSERT INTO responses VALUES (?, 'api.deezer.com', 404, NULL, 10, 0, 1e12)",
                    ("https://api.deezer.com/search?q=older+words",))
    web._db.commit()
    assert web.purge_private() == 2
    assert [k for (k,) in web._db.execute("SELECT key FROM responses")] == ["https://example.org/chart"]
    # without a history, requests with the user's words are not cached at all
    web.cache_private = False
    for _ in range(2):
        web.get_json("https://example.org/search", {"q": words}, private=True)
    assert len(router.calls) == 4 and web.stats()["cache"]["entries"] == 1


def test_expired_rows_are_pruned_periodically_not_only_over_budget(tmp_path, monkeypatch):
    """Regression: expired rows (e.g. a searched phrase) stayed forever unless the cache exceeded its size budget."""
    from melodyai.world import web as web_mod

    monkeypatch.setattr(web_mod, "_PRUNE_EVERY", 5)
    ft = FakeTime()
    router = Router().add("/a", json_response({"v": 1}))
    web = make_web(router, ft, cache_path=tmp_path / "c.db")
    web.get_json("https://example.org/a", {"q": "a phrase somebody sang"}, ttl_s=60)
    ft.t += 400 * 86400
    for i in range(5):
        web.get_json("https://example.org/a", {"i": i})
    keys = [k for (k,) in web._db.execute("SELECT key FROM responses")]
    assert len(keys) == 5 and not any("phrase" in k for k in keys)
