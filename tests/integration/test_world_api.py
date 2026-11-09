"""Open-world search through the API: recognition resolution, the online setting and the /api/world endpoints.

Real app, temporary store, no network: LRCLIB / Deezer answers come from an ``httpx.MockTransport`` over a
tiny invented catalog, and the popular-songs sync is replaced by a fake. Every title, artist and lyric below
is invented (the local catalog uses the repository's invented demo lyrics).
"""

import json
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import ServerSettings
from app.main import create_app
from melodyai.catalog.importers import notes_to_midi_bytes
from melodyai.demo.generator import generate_catalog
from melodyai.demo.lyrics_bank import LYRICS
from melodyai.lyrics.asr import MODEL_CHOICES, MODEL_SIZES
from melodyai.lyrics.normalize import tokenize
from melodyai.world.index import WorldSong
from melodyai.world.lrclib import LrclibClient
from melodyai.world.search import WorldSearch
from melodyai.world.web import HostLimit, WebClient

FAST = {"lrclib.net": HostLimit(1000, 1000.0, 3), "api.deezer.com": HostLimit(1000, 1000.0, 6)}

HARBOR = ("Lantern harbor glowing in the fog\nWe tie the ribbons to the broken mast\n"
          "Lantern harbor calling out my name\nThe tide remembers every promise\n"
          "Lantern harbor calling out my name\nSalt and copper on the evening wind")
VALLEY = ("Copper valley nights are long and blue\nThe tractors sleep beneath the walnut trees\n"
          "Somebody whistles on the gravel road\nCopper valley nights I dream of you")
PARADE = ("Velvet thunder parade is rolling down the avenue\nPainted horses jumping through the hoops of fire\n"
          "Velvet thunder parade is rolling down the avenue\nDrummers in the rain and children on the wire")

LRCLIB_DB = [
    {"id": 501, "trackName": "Velvet Thunder Parade", "artistName": "Orla Brightwater", "albumName": "Carnival",
     "duration": 222.0, "instrumental": False, "plainLyrics": PARADE, "syncedLyrics": None},
]
DEEZER_DB = [(7001, "Velvet Thunder Parade", "Orla Brightwater", 640000, 222)]

SPEC_SONG_KEYS = {"id", "title", "artist", "album", "language", "reference_type", "source", "duration_s",
                  "cover_url", "preview_url", "deezer_id", "has_audio", "links"}


def dz(tid, title, artist, rank, duration):
    return {"id": tid, "title": title, "title_short": title, "link": f"https://www.deezer.com/track/{tid}",
            "duration": duration, "rank": rank, "explicit_lyrics": False,
            "preview": f"https://cdnt-preview.dzcdn.net/api/1/1/{tid}.mp3", "type": "track",
            "artist": {"id": tid + 1, "name": artist, "type": "artist"},
            "album": {"id": 3, "title": "Carnival", "cover_medium": f"https://cdn-images.dzcdn.net/{tid}.jpg",
                      "type": "album"}}


class Online:
    """MockTransport handler: LRCLIB title search / get and Deezer search over the invented catalog."""

    def __init__(self):
        self.calls: list[httpx.Request] = []
        self.lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            self.calls.append(request)
        p, q, host = request.url.path, dict(request.url.params), request.url.host
        if host == "lrclib.net" and p == "/api/search":
            words = set(tokenize(q.get("track_name") or q.get("q") or ""))
            return httpx.Response(200, json=[r for r in LRCLIB_DB if words and words <= set(tokenize(r["trackName"]))])
        if host == "lrclib.net" and p == "/api/get":
            for r in LRCLIB_DB:
                if tokenize(r["trackName"]) == tokenize(q.get("track_name", "")):
                    return httpx.Response(200, json=r)
            return httpx.Response(404, json={"statusCode": 404})
        if host == "api.deezer.com" and p == "/search":
            words = set(tokenize(q.get("q", "")))
            hits = [dz(*t) for t in DEEZER_DB if set(tokenize(t[1])) <= words]
            return httpx.Response(200, json={"data": hits, "total": len(hits)})
        if host == "api.deezer.com" and p.startswith("/track/"):
            tid = int(p.rsplit("/", 1)[1])
            for t in DEEZER_DB:
                if t[0] == tid:
                    return httpx.Response(200, json={**dz(*t), "preview": f"https://cdnt-preview.dzcdn.net/api/1/1/"
                                                                          f"{tid}.mp3?hdnea=exp=4102444800~hmac=x"})
            if tid == 666:                       # a payload pointing somewhere else must not become a redirect
                return httpx.Response(200, json={**dz(666, "Odd", "Odd", 1, 100), "preview": "https://evil.example/x"})
            return httpx.Response(200, json={"error": {"type": "DataException", "message": "no data", "code": 800}})
        return httpx.Response(404, json={})

    def count(self) -> int:
        with self.lock:
            return len(self.calls)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    store = tmp_path_factory.mktemp("worldstore")
    app = create_app(ServerSettings(store_dir=str(store), load_neural_model=False, serve_frontend=False))
    online = Online()
    with TestClient(app) as c:
        state = c.app.state.melody
        assert state.web.offline  # the test suite runs with MELODYAI_OFFLINE=1 (tests/conftest.py)
        assert state.world_index.path == str(store / "world.db") and state.engine.world is state.world
        # Same wiring as AppState, with the network replaced by the mock (and online capable).
        state.web.close()
        state.web = WebClient(None, transport=httpx.MockTransport(online), host_limits=FAST)
        state.world = WorldSearch(state.world_index, state.web)
        state.engine.world = state.world
        for title, artist, lyrics, rank in (("Lantern Harbor", "Mira Solenne", HARBOR, 500_000),
                                            ("Copper Valley Nights", "The Fennel Brothers", VALLEY, 200_000),
                                            ("Lighthouse Echo", "Somebody Else", LYRICS[1]["text"], 90_000)):
            state.world_index.upsert(WorldSong.create(title, artist, lyrics=lyrics, rank=rank, language="en",
                                                      source="sync:test"))
        for i, m in enumerate(generate_catalog(3, base_seed=1000)):
            files = {"reference": (f"s{i}.mid", notes_to_midi_bytes(m.notes), "audio/midi")}
            data = {"title": f"Song {i}", "artist": "API Tester", "license": "CC0-1.0", "lyrics_text": LYRICS[i]["text"]}
            assert c.post("/api/songs", files=files, data=data).status_code == 201
        yield c, state, online, store


def lines(text: str, a: int, b: int) -> str:
    return "\n".join(text.splitlines()[a:b])


@pytest.fixture()
def online_off(ctx):
    c = ctx[0]
    assert c.put("/api/settings", json={"online_search_enabled": False}).json()["online_search_enabled"] is False
    yield
    assert c.put("/api/settings", json={"online_search_enabled": True}).json()["online_search_enabled"] is True


# ---------------------------------------------------------------------------- recognition
def test_settings_default_online_search_on(ctx):
    assert ctx[0].get("/api/settings").json()["online_search_enabled"] is True


def test_typed_lyrics_from_the_world_index_resolve_lyrics_online(ctx):
    c = ctx[0]
    r = c.post("/api/recognize/lyrics", data={"text": lines(HARBOR, 1, 4)})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["resolved_mode"] == "lyrics_online" and res["status"] == "match"
    top = res["results"][0]
    assert top["song_id"].startswith("world:") and set(top["song"]) == SPEC_SONG_KEYS
    assert top["song"]["title"] == "Lantern Harbor" and top["song"]["artist"] == "Mira Solenne"
    assert top["song"]["reference_type"] == "online" and top["song"]["source"] == "LRCLIB + Deezer"
    assert top["song"]["has_audio"] is False and set(top["song"]["links"]) >= {"spotify", "youtube", "deezer"}
    assert top["song"]["deezer_id"] is None and top["song"]["preview_url"] is None
    assert set(top["engines"]["world"]) == {"rank", "score", "coverage", "matched_text", "line_range", "sources"}
    assert "index" in top["engines"]["world"]["sources"]
    world = res["engines"]["world"]
    assert world["status"] == "ok" and world["online"] is True and world["decision"]["status"] == "match"
    assert world["sources"]["index"]["status"] == "ok"
    # history keeps the online song
    assert c.get(f"/api/recognitions/{res['id']}").json()["results"][0]["song"]["title"] == "Lantern Harbor"
    hist = c.get("/api/recognitions").json()[0]
    assert hist["resolved_mode"] == "lyrics_online" and hist["top_song"]["title"] == "Lantern Harbor"


def test_typed_lyrics_found_online_by_title(ctx):
    c, state, online, _ = ctx
    before = online.count()
    res = c.post("/api/recognize", data={"mode": "auto", "text": lines(PARADE, 0, 2)}).json()
    assert online.count() > before
    assert res["resolved_mode"] == "lyrics_online" and res["status"] == "match"
    top = res["results"][0]
    assert top["song"]["title"] == "Velvet Thunder Parade"
    assert set(top["engines"]["world"]["sources"]) & {"lrclib", "deezer"}  # proposed by an online source
    assert top["song"]["cover_url"] and top["song"]["links"]["deezer"] == "https://www.deezer.com/track/7001"
    assert top["song"]["deezer_id"] == 7001
    assert state.world_index.has_key("Velvet Thunder Parade", "Orla Brightwater")  # learned for next time


def test_local_catalog_lyrics_match_wins_over_the_world(ctx):
    c = ctx[0]
    res = c.post("/api/recognize/lyrics", data={"text": "the lighthouse keeper counts the ships at nite"}).json()
    assert res["resolved_mode"] == "lyrics" and res["status"] == "match"
    assert res["results"][0]["song"]["title"] == "Song 1" and res["results"][0]["song"]["reference_type"] != "online"
    world = res["engines"]["world"]  # still reported: "also found online"
    assert world["candidates"] and world["candidates"][0]["title"] == "Lighthouse Echo"
    assert all(not r["song_id"].startswith("world:") for r in res["results"])


def test_online_setting_off_skips_online_sources(ctx, online_off):
    c, _, online, _ = ctx
    assert c.get("/api/world/status").json()["online_enabled"] is False
    before = online.count()
    res = c.post("/api/recognize/lyrics", data={"text": "painted horses jumping through the hoops of fire again"}).json()
    assert online.count() == before  # nothing left the server
    world = res["engines"]["world"]
    assert world["online"] is False
    assert world["sources"]["lrclib"]["status"] == "skipped" and world["sources"]["deezer"]["status"] == "skipped"
    assert world["sources"]["index"]["status"] == "ok"
    # the local world index still answers offline
    res = c.post("/api/recognize/lyrics", data={"text": lines(VALLEY, 1, 3)}).json()
    assert res["resolved_mode"] == "lyrics_online" and res["results"][0]["song"]["title"] == "Copper Valley Nights"


def test_world_ids_are_not_enriched_from_the_catalog(ctx, monkeypatch):
    c, state, _, _ = ctx
    asked: list[str] = []
    real_meta, real_listen = state.catalog.songs_by_ids, state.catalog.listen_file

    def songs_by_ids(ids):
        asked.extend(ids)
        return real_meta(ids)

    def listen_file(sid):
        asked.append(sid)
        return real_listen(sid)

    monkeypatch.setattr(state.catalog, "songs_by_ids", songs_by_ids)
    monkeypatch.setattr(state.catalog, "listen_file", listen_file)
    res = c.post("/api/recognize/lyrics", data={"text": lines(HARBOR, 0, 3)}).json()
    assert res["results"] and all(r["song_id"].startswith("world:") for r in res["results"])
    assert not any(s.startswith("world:") for s in asked)
    assert res["results"][0]["song"]["title"] == "Lantern Harbor"


# ---------------------------------------------------------------------------- /api/world
def test_world_status(ctx):
    c, state, _, _ = ctx
    st = c.get("/api/world/status").json()
    assert set(st) == {"songs", "online_enabled", "server_online", "sources", "last_sync", "db_bytes", "presets",
                       "running_sync", "melodies"}
    assert st["server_online"] is True and st["running_sync"] is None
    assert st["melodies"]["built"] is False and st["melodies"]["songs"] == 0     # no Lakh index in a fresh store
    assert st["songs"] == len(state.world_index) >= 3 and st["db_bytes"] > 0 and st["online_enabled"] is True
    names = {s["name"]: s for s in st["sources"]}
    assert {"index", "lrclib", "deezer"} <= set(names) and names["index"]["available"] is True
    assert all(set(s) == {"name", "label", "online", "requires_key", "available"} for s in st["sources"])
    assert [p["name"] for p in st["presets"]] == ["quick", "standard", "large"]
    assert all(set(p) == {"name", "label", "approx_songs", "approx_minutes"} for p in st["presets"])


def test_world_search_endpoint(ctx):
    c = ctx[0]
    for q in ("", "   ", "hi there"):
        r = c.post("/api/world/search", json={"q": q})
        assert r.status_code == 200 and r.json()["status"] == "no_query", q
        assert r.json()["decision"]["status"] == "no_match" and r.json()["candidates"] == []
    assert c.post("/api/world/search", json={}).json()["status"] == "no_query"
    r = c.post("/api/world/search", json={"q": lines(HARBOR, 3, 6), "language": "en"}).json()
    assert r["status"] == "ok" and r["candidates"][0]["title"] == "Lantern Harbor"
    assert r["decision"]["status"] == "match" and r["query_text"] and "elapsed_s" in r and "sources" in r
    assert c.post("/api/world/search", json={"q": "some words", "language": "english"}).status_code == 422
    assert c.post("/api/world/search", json={"q": "x" * 2001}).status_code == 422
    # the words never travel in a URL (access logs, cross-site <img> requests)
    assert c.get("/api/world/search", params={"q": lines(HARBOR, 3, 6)}).status_code == 405
    # a body that is not JSON (a cross-site "simple" form / text request) is not parsed
    r = c.post("/api/world/search", content=json.dumps({"q": lines(HARBOR, 3, 6)}), headers={"content-type": "text/plain"})
    assert r.status_code == 422


def test_world_search_decides_before_trimming_to_top_k(ctx):
    """Regression: the decision was computed on the list already cut to ``top_k``: with top_k=1 an ambiguous
    query (two songs with the same line) lost its runner-up and became a confident match."""
    c, state, _, _ = ctx
    shared = "Moonlit ferry crossing silver water\nPaper boats are sailing to the shore\n"
    for title, artist, rest in (
            ("Moonlit Ferry", "Ada Quill", "Anchors made of daisies\nThe harbor cat is dreaming\nWe paint the lighthouse"),
            ("Ferry Song", "Bruno Vale", "Grandpa sells balloons\nThe carousel is spinning\nA kite above the market")):
        state.world_index.upsert(WorldSong.create(title, artist, lyrics=shared + rest, rank=1000, language="en",
                                                  source="sync:test"))
    q = "moonlit ferry crossing silver water paper boats are sailing to the shore"
    full = c.post("/api/world/search", json={"q": q, "top_k": 5}).json()
    assert len(full["candidates"]) >= 2 and full["decision"]["status"] == "uncertain"
    one = c.post("/api/world/search", json={"q": q, "top_k": 1}).json()
    assert len(one["candidates"]) == 1 and one["decision"] == full["decision"]


def test_world_songs_endpoint(ctx):
    c = ctx[0]
    songs = c.get("/api/world/songs").json()
    assert len(songs) >= 3 and all("lyrics" not in s for s in songs)
    assert {"id", "title", "artist", "rank", "source", "added_at"} <= set(songs[0])
    assert songs == sorted(songs, key=lambda s: -s["rank"])  # most popular first
    only = c.get("/api/world/songs", params={"q": "lantern"}).json()
    assert [s["title"] for s in only] == ["Lantern Harbor"]
    assert len(c.get("/api/world/songs", params={"limit": 1, "offset": 1}).json()) == 1
    assert c.get("/api/world/songs", params={"limit": 0}).status_code == 422
    # an offset beyond SQLite's integer range is a validation error, not a 500
    assert c.get("/api/world/songs", params={"offset": 10**19}).status_code == 422
    assert c.get("/api/world/songs", params={"offset": 10_000_000}).json() == []


def test_world_sync_job_validation_and_conflict(ctx, monkeypatch):
    from app.api import world as world_api

    c, state, _, store = ctx
    release, calls = threading.Event(), []

    def fake_sync(index, lrclib, deezer, preset="quick", limit=None, progress=None, should_stop=None,
                  state_path=None):
        calls.append({"index": index, "lrclib": lrclib, "deezer": deezer, "preset": preset, "limit": limit,
                      "state_path": state_path})
        progress(0.5, "Lyrics 1/2")
        release.wait(20)
        index.meta_set("last_sync", json.dumps({"preset": preset, "status": "completed", "finished_at": "2026-01-01",
                                                "collected": 2, "added": 1, "no_lyrics": 1}))
        return {"status": "completed", "preset": preset, "collected": 2, "added": 1}

    monkeypatch.setattr(world_api, "sync_popular", fake_sync)
    assert c.post("/api/world/sync", json={"preset": "huge"}).status_code == 422
    assert c.post("/api/world/sync", json={"preset": "quick", "limit": 0}).status_code == 422
    try:
        job = c.post("/api/world/sync", json={"preset": "standard", "limit": 5})
        assert job.status_code == 200, job.text
        job = job.json()
        assert job["kind"] == "world_sync" and job["config"] == {"preset": "standard", "limit": 5}
        assert c.post("/api/world/sync", json={"preset": "quick"}).status_code == 409
        assert any(j["id"] == job["id"] for j in c.get("/api/jobs", params={"kind": "world_sync"}).json())
    finally:
        release.set()
    done = state.jobs.wait(job["id"], 30)
    assert done["status"] == "completed" and done["result"]["added"] == 1
    call = calls[0]
    assert call["index"] is state.world_index and call["preset"] == "standard" and call["limit"] == 5
    assert isinstance(call["lrclib"], LrclibClient) and call["lrclib"].web is state.web
    assert call["state_path"] == store / "world_sync.db"
    last = c.get("/api/world/status").json()["last_sync"]
    assert last["preset"] == "standard" and last["added"] == 1 and last["collected"] == 2 and last["finished_at"]
    # a failed sync is a failed job; afterwards a new one may start again (default body = quick)
    monkeypatch.setattr(world_api, "sync_popular", lambda *a, **k: {"status": "error", "message": "Deezer down"})
    assert c.post("/api/world/sync").status_code == 422          # a JSON body is required (no cross-site POST)
    job = c.post("/api/world/sync", json={}).json()
    done = state.jobs.wait(job["id"], 30)
    assert done["status"] == "failed" and "Deezer down" in done["error"] and done["config"]["preset"] == "quick"


def test_world_sync_refused_when_offline(ctx):
    c, state, _, _ = ctx
    state.web.offline = True
    try:
        assert c.post("/api/world/sync", json={"preset": "quick"}).status_code == 503
        assert c.get("/api/world/status").json()["online_enabled"] is False
    finally:
        state.web.offline = False


# ---------------------------------------------------------------------------- models / ASR choices
def test_models_list_asr_choices(ctx):
    c = ctx[0]
    asr = c.get("/api/models").json()["asr"]
    assert [m["name"] for m in asr["models"]] == list(MODEL_SIZES) and "large-v3-turbo" in MODEL_SIZES
    assert asr["choices"] == list(MODEL_CHOICES) and asr["auto"]["name"] == "auto"
    assert "resolved" in asr["auto"] and asr["active"] == "auto"
    assert asr["active_resolved"] == asr["auto"]["resolved"]
    # downloads are disabled on this server, but the body accepts every choice
    for model in ("auto", "large-v3-turbo"):
        assert c.post("/api/models/asr/download", json={"model": model}).status_code == 403
    assert c.post("/api/models/asr/download", json={"model": "turbo"}).status_code == 422


def test_asr_download_maps_auto_to_the_resolved_model(ctx, monkeypatch):
    fw_utils = pytest.importorskip("faster_whisper.utils")
    from app.api import lab

    c, state, _, _ = ctx
    got: list[str] = []
    monkeypatch.setattr(fw_utils, "download_model", lambda name, **kw: got.append(name) or "/nowhere")
    monkeypatch.setattr(lab, "resolve_model", lambda name, allow_download=False: "large-v3-turbo")
    monkeypatch.setattr(lab, "model_status", lambda name: {"name": name, "cached": True, "installed_package": True})
    monkeypatch.setattr(lab, "asr_available", lambda: True)
    monkeypatch.setattr(state.settings, "allow_model_download", True)
    for requested, expected in (("auto", "large-v3-turbo"), ("small", "small")):
        job = c.post("/api/models/asr/download", json={"model": requested})
        assert job.status_code == 200, job.text
        done = state.jobs.wait(job.json()["id"], 30)
        assert done["status"] == "completed" and done["result"]["name"] == expected
        assert done["config"] == {"model": expected, "requested": requested}
    assert got == ["large-v3-turbo", "small"]


# ---------------------------------------------------------------------------- jobs, cross-process sync, privacy
def test_world_sync_runs_in_its_own_lane(ctx, monkeypatch):
    """Regression: the hours-long sync shared the single job worker and held back downloads / reindexing."""
    from app.api import world as world_api

    c, state, _, _ = ctx
    release = threading.Event()

    def fake_sync(index, lrclib, deezer, preset="quick", limit=None, progress=None, should_stop=None,
                  state_path=None):
        release.wait(20)
        return {"status": "completed", "preset": preset, "collected": 0, "added": 0}

    monkeypatch.setattr(world_api, "sync_popular", fake_sync)
    try:
        sync = c.post("/api/world/sync", json={"preset": "large"}).json()
        other = state.jobs.submit("demo_import", {}, lambda progress, stop: {"ok": True})
        assert state.jobs.wait(other["id"], 10)["status"] == "completed"     # not queued behind the sync
        assert state.jobs.get(sync["id"])["status"] == "running"
    finally:
        release.set()
    assert state.jobs.wait(sync["id"], 30)["status"] == "completed"


def test_world_sync_refused_while_another_process_syncs(ctx, monkeypatch):
    from app.api import world as world_api
    from melodyai.world.sync import sync_lock

    c, _, _, store = ctx
    monkeypatch.setattr(world_api, "sync_popular", lambda *a, **k: pytest.fail("must not start"))
    with sync_lock(store / "world_sync.db"):                 # e.g. `melodyai world-sync` in a terminal
        st = c.get("/api/world/status").json()
        assert st["running_sync"] is not None
        r = c.post("/api/world/sync", json={"preset": "quick"})
        assert r.status_code == 409 and "another process" in r.json()["detail"]
    assert c.get("/api/world/status").json()["running_sync"] is None


def test_cross_site_requests_are_refused(ctx):
    c, _, online, _ = ctx
    before = online.count()
    evil = {"Origin": "https://evil.example"}
    assert c.post("/api/world/sync", json={"preset": "quick"}, headers=evil).status_code == 403
    assert c.post("/api/world/search", json={"q": lines(PARADE, 0, 2)}, headers=evil).status_code == 403
    assert c.get("/api/world/status", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert online.count() == before
    # the dev frontend's origin, this server itself and plain clients (no Origin) are fine
    assert c.get("/api/world/status", headers={"Origin": "http://127.0.0.1:5173"}).status_code == 200
    assert c.get("/api/world/status", headers={"Origin": "http://testserver",
                                               "Sec-Fetch-Site": "same-origin"}).status_code == 200
    assert c.get("/api/world/status").status_code == 200


def test_preview_redirects_to_a_fresh_deezer_url(ctx):
    c, state, _, _ = ctx
    r = c.get("/api/world/preview/7001", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"].startswith("https://cdnt-preview.dzcdn.net/")
    assert c.get("/api/world/preview/7002", follow_redirects=False).status_code == 404     # no such track
    assert c.get("/api/world/preview/666", follow_redirects=False).status_code == 404      # not Deezer's CDN
    assert c.get("/api/world/preview/0", follow_redirects=False).status_code == 422
    state.web.offline = True
    try:
        assert c.get("/api/world/preview/7001", follow_redirects=False).status_code == 503
    finally:
        state.web.offline = False


def test_history_settings_govern_what_the_open_world_keeps(ctx):
    """Without a history nothing derived from recognitions is kept; deleting the history purges cached searches
    with the user's words and the songs learned from recognitions (runs last: it empties the history)."""
    c, state, _, _ = ctx
    c.post("/api/recognize/lyrics", data={"text": lines(PARADE, 0, 2)})
    assert state.world_index.has_key("Velvet Thunder Parade", "Orla Brightwater")
    assert c.put("/api/settings", json={"store_history": False}).status_code == 200
    try:
        assert state.web.cache_private is False and state.world.learn is False
    finally:
        assert c.put("/api/settings", json={"store_history": True}).status_code == 200
    assert state.web.cache_private is True and state.world.learn is True
    out = c.delete("/api/recognitions").json()
    assert out["learned_songs"] >= 1 and "cached_searches" in out
    assert not state.world_index.has_key("Velvet Thunder Parade", "Orla Brightwater")
    assert state.world_index.has_key("Lantern Harbor", "Mira Solenne")                   # synced songs stay
