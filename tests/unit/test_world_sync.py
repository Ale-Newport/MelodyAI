"""Popular-songs sync: Deezer collection (owners, titles, dedup, ordering, artists), LRCLIB lyrics phase
(counts, metadata, resume, miss cache, errors, pacing, cancellation), state DB and the CLI command.

No network: Deezer and LRCLIB are in-memory fakes. Every title, artist and lyric below is invented.
"""

import json
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from melodyai.world import sync
from melodyai.world.deezer import TrackMeta
from melodyai.world.index import WorldLyricsIndex, WorldSong
from melodyai.world.lrclib import LyricsRecord
from melodyai.world.sync import (
    PRESETS,
    TIERS,
    PlaylistQuery,
    SyncPlan,
    SyncState,
    collect,
    collect_tracks,
    preset_list,
    select_playlists,
    sync_popular,
)
from melodyai.world.text import song_key
from melodyai.world.web import WebError

EN = ("Paper lanterns drifting on the river\nWe count the lights until the morning comes\n"
      "Carry me home, carry me home tonight\nThe paper lanterns know the way")
ES = ("Las farolas de papel cruzan el río\nY yo cuento las luces hasta que llega el día\n"
      "Llévame a casa, llévame a casa esta noche\nque las farolas saben el camino")


def tm(tid, title, artist, rank=100, *, artist_id=None, duration=200.0, album="Disco Inventado"):
    return TrackMeta(deezer_id=tid, title=title, title_short=title, artist=artist,
                     artist_id=artist_id if artist_id is not None else abs(hash(artist)) % 10_000 + 1,
                     album=album, duration_s=duration, rank=rank, isrc=f"XX{tid:010d}",
                     cover_url=f"https://cdn.example/cover/{tid}.jpg", preview_url=f"https://cdn.example/p/{tid}.mp3",
                     link=f"https://www.deezer.com/track/{tid}")


def pl(pid, title, user, nb=50):
    return {"id": pid, "title": title, "nb_tracks": nb, "user": user, "user_id": 1, "link": None}


class FakeDeezer:
    """search_playlists / playlist_tracks / chart_tracks / chart_artists / artist_top from dicts; ``fail`` makes
    a method (or a specific query / id) raise WebError."""

    def __init__(self, *, searches=None, playlists=None, charts=None, chart_artists=None, artists=None, fail=()):
        self.searches = searches or {}
        self.playlists = playlists or {}
        self.charts = charts or {}
        self.chart_artist_list = chart_artists or []
        self.artists = artists or {}
        self.fail = set(fail)
        self.calls: list[tuple] = []

    def _check(self, *what):
        self.calls.append(what)
        if what[0] in self.fail or what in self.fail:
            raise WebError("simulated outage", kind="network", host="api.deezer.com")

    def search_playlists(self, query, limit=25):
        self._check("search", query)
        return list(self.searches.get(query, []))[:limit]

    def playlist_tracks(self, playlist_id, limit=500):
        self._check("playlist", playlist_id)
        return list(self.playlists.get(playlist_id, []))[:limit]

    def chart_tracks(self, genre_id=0, limit=100):
        self._check("chart", genre_id)
        return list(self.charts.get(genre_id, []))[:limit]

    def chart_artists(self, genre_id=0, limit=100):
        self._check("chart_artists", genre_id)
        return list(self.chart_artist_list)[:limit]

    def artist_top(self, artist_id, limit=50):
        self._check("artist", artist_id)
        return list(self.artists.get(artist_id, []))[:limit]


class FakeLrclib:
    """``find`` answers from ``lyrics`` (title -> text; None = no lyrics); ``errors`` (title -> list of WebError)
    are raised first. Counts concurrent calls and simulated network requests (``web._host_counts``)."""

    def __init__(self, lyrics=None, errors=None, requests_per_call=1, cached=()):
        self.lyrics = dict(lyrics or {})
        self.errors = {k: list(v) for k, v in (errors or {}).items()}
        self.calls: list[str] = []
        self.active = self.max_active = 0
        self.lock = threading.Lock()
        self.requests_per_call = requests_per_call
        self.cached = set(cached)
        self.web = SimpleNamespace(_host_counts={"lrclib.net": Counter()})

    def find(self, title, artist, duration_s=None, album=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append(title)
            if title not in self.cached:
                self.web._host_counts["lrclib.net"]["requests"] += self.requests_per_call
            if self.errors.get(title):
                raise self.errors[title].pop(0)
            text = self.lyrics.get(title)
            if text is None:
                return None
            return LyricsRecord(id=1000 + len(self.calls), track=title, artist=artist, album=album,
                                duration_s=duration_s, instrumental=False, plain=text, synced=None)
        finally:
            with self.lock:
                self.active -= 1


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


@pytest.fixture()
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(sync, "_monotonic", c.monotonic)
    monkeypatch.setattr(sync, "_sleep", c.sleep)
    return c


# ---------------------------------------------------------------------------- a small world
CHART_A = [tm(1, "Neon Harbor", "Lumen Tide", 900, artist_id=11),
           tm(2, "Velvet Static", "Orla Finch", 950, artist_id=12),
           tm(3, "Glass Orchard (feat. Kip Rowe)", "Mara Sol", 400, artist_id=13)]
CHART_B = [tm(4, "Río de Cobre", "Los Faroles", 300, artist_id=14),
           tm(5, "Glass Orchard", "Mara Sol", 420, artist_id=13),
           tm(6, "Neon Harbor", "Lumen Tide", 900, artist_id=11)]
TIKTOK = [tm(7, "Paper Lanterns (Sped Up)", "Marea Alta", 990, artist_id=15, duration=150),
          tm(8, "Sunday Static", "Orla Finch", 200, artist_id=12)]
DECADES = [tm(9, "Copper Moon - Remastered 2011", "The Wandering Pines", 980, artist_id=16),
           tm(10, "Instrumental Dreams (Instrumental)", "Studio Band", 999, artist_id=17),
           tm(11, "Intro", "Orla Finch", 999, artist_id=12, duration=12),
           tm(12, "Velvet Static", "Orla Finch", 10, artist_id=12),
           tm(13, "Copper Moon", "Wandering Pines", 100, artist_id=16)]
YEARLY = [tm(14, "Long Ago Song", "Old Choir", 50, artist_id=18)]
ARTIST_12 = [tm(15, "Static Bloom", "Orla Finch", 600, artist_id=12),
             tm(2, "Velvet Static", "Orla Finch", 950, artist_id=12)]
ARTIST_99 = [tm(16, "Chart Comet", "Comet Kids", 700, artist_id=99)]

SEARCHES = {
    "Top Arcadia": [pl(101, "Top Arcadia", "Deezer Charts "), pl(901, "Top Arcadia", "someone")],
    "Top Brumaria": [pl(911, "Top Brumaria hits", "Deezer Charts"), pl(102, "Top Brumaria", "Deezer Charts")],
    "tiktok hits": [pl(920, "TikTok Instrumental Hits", "Yannick - Deezer Classical Editor"),
                    pl(921, "TikTok bangers 2026", "anonymous"), pl(103, "TikTok Hits Arcadia", "Pop & Hits Editor")],
    "70s hits": [pl(930, "70s Hits", "Filtr Arcadia"), pl(104, "70s Hits", "Laeti - Deezer Pop Editor")],
    "missing": [pl(940, "Nothing relevant", "Deezer Charts")],
    "Top Hits 2025": [pl(105, "Top Hits 2025", "Deezer Best Of")],
}
PLAYLISTS = {101: CHART_A, 102: CHART_B, 103: TIKTOK, 104: DECADES, 105: YEARLY,
             930: [tm(17, "Label Song", "Label Artist", 999)], 920: [tm(18, "Calm Keys", "Pianist", 999)]}

PLAN = SyncPlan(
    name="test", label="test", approx_songs=10, approx_minutes=1, chart_genres=(0,), countries=("Arcadia", "Brumaria"),
    playlists=(PlaylistQuery("tiktok hits", "tiktok", r"tik ?tok"), PlaylistQuery("70s hits", "editorial", r"\b70s\b"),
               PlaylistQuery("missing", "editorial", r"\b80s\b")),
    best_of_from=2025, tracks_per_playlist=100)


def world(**kw) -> FakeDeezer:
    args = dict(searches=SEARCHES, playlists=PLAYLISTS,
                charts={0: [tm(19, "Global Glow", "Orla Finch", 800, artist_id=12)]},
                artists={12: ARTIST_12, 99: ARTIST_99}, chart_artists=[{"id": 99, "name": "Comet Kids", "position": 1}])
    args.update(kw)
    return FakeDeezer(**args)


def keys(tracks):
    return [song_key(t.title, t.artist) for t in tracks]


# ---------------------------------------------------------------------------- presets
def test_presets_are_consistent():
    assert list(PRESETS) == ["quick", "standard", "large"]
    sizes = [p.approx_songs for p in PRESETS.values()]
    assert sizes == sorted(sizes) and sizes[0] >= 1000
    assert [p["name"] for p in preset_list()] == list(PRESETS)
    assert set(preset_list()[0]) == {"name", "label", "approx_songs", "approx_minutes"}
    for plan in PRESETS.values():
        json.dumps(plan.to_dict())
        assert plan.name and plan.label and plan.countries and plan.chart_genres
        for q in plan.playlists:
            assert q.group in TIERS and q.owner in sync.OWNER_MODES and q.max_playlists >= 1
            re.compile(q.title or "")
        queries = sync._plan_queries(plan, 2026)
        assert len({(q.query, q.title) for q in queries}) == len(queries), "duplicate playlist queries"
    quick = PRESETS["quick"]
    assert {"Spain", "Mexico", "USA", "Argentina", "Colombia", "UK", "Worldwide"} <= set(quick.countries)
    groups = Counter(q.group for q in quick.playlists)
    assert groups["tiktok"] >= 2 and groups["editorial"] >= 15
    text = " ".join(q.query for q in quick.playlists)
    for word in ("70s", "80s", "90s", "2000s", "2010s", "2020s", "reggaeton", "salsa", "bachata", "cumbia",
                 "rock en español", "baladas", "regional mexicano", "pop latino"):
        assert word in text
    # bigger presets are supersets with artist expansion
    assert set(quick.playlists) <= set(PRESETS["standard"].playlists) <= set(PRESETS["large"].playlists)
    assert quick.top_artists == 0 and PRESETS["standard"].top_artists > 0 and PRESETS["large"].best_of_from < 2000


def test_plan_queries_order_and_yearly_names():
    plan = SyncPlan("t", "t", 1, 1, countries=("Worldwide", "Mexico"), country_years=1,
                    playlists=(PlaylistQuery("decade", "editorial"), PlaylistQuery("viral", "tiktok")),
                    best_of_from=2023)
    qs = sync._plan_queries(plan, 2026)
    assert [q.query for q in qs] == ["Top Worldwide", "Top Mexico", "viral", "decade", "Top Worldwide 2025",
                                     "Top México 2025", "Top Hits 2025", "Top Hits 2024", "Top Hits 2023"]
    mx = qs[5]
    assert mx.owner == "charts" and re.search(mx.title, "top mexico 2025") and not re.search(mx.title, "top mexico")


# ---------------------------------------------------------------------------- playlist selection
def test_select_playlists_owner_and_title_rules():
    found = [pl(1, "Top Arcadia", "Deezer Charts "), pl(2, "Top Arcadia 2024", "Deezer Charts"),
             pl(3, "Top Arcadia", "fan account")]
    chart = PlaylistQuery("Top Arcadia", "chart", r"^top arcadia$", 1, "charts")
    assert [p["id"] for p in select_playlists(found, chart)] == [1]
    assert select_playlists(found, chart, exclude={1}) == []

    found = [pl(10, "Viral Songs", "Filtr Arcadia"), pl(11, "Hits Virais", "Laura - Editora Deezer Brasil"),
             pl(12, "Viral", "anonymous"), pl(13, "Viral piano covers", "Deezer Moods Editor"),
             pl(14, "100% Lumen Tide viral", "Deezer Artist Editor"), pl(15, "Viral mini", "Deezer Editor", nb=3),
             pl(16, "Viral Hits Chart", "Topsify")]
    editorial = PlaylistQuery("viral", "tiktok", r"viral|virais", 3)
    # editors first, label curators only as fallback, never anonymous / piano / single-artist / tiny lists
    assert [p["id"] for p in select_playlists(found, editorial)] == [11, 10, 16]
    assert [p["id"] for p in select_playlists(found, PlaylistQuery("viral", "tiktok", r"viral", 1))] == [10]
    official = PlaylistQuery("viral", "tiktok", r"viral|virais", 2, "official")
    assert [p["id"] for p in select_playlists(found, official)] == [10, 11]
    # titles are matched accent- and case-insensitively
    found = [pl(20, "Top México 2025", "Deezer Charts"),
             pl(21, "Pop en español '90s", "Adriana - Deezer LATAM Editor")]
    assert select_playlists(found, PlaylistQuery("x", "yearly", r"^top mexico 2025$", 1, "charts"))[0]["id"] == 20
    assert select_playlists(found, PlaylistQuery("x", "editorial", r"pop en espanol .*\b90s\b"))[0]["id"] == 21


# ---------------------------------------------------------------------------- collection
def test_collect_dedups_filters_and_orders_by_importance():
    dz = world()
    coll = collect(dz, PLAN, year=2026)
    ks = keys(coll.tracks)
    assert len(ks) == len(set(ks))
    # dedup across lists and versions: feat. credit, "- Remastered 2011", "The" prefix, repeated chart entries
    assert ks.count(song_key("Glass Orchard", "Mara Sol")) == 1
    assert ks.count(song_key("Copper Moon", "Wandering Pines")) == 1
    # dropped: instrumental titles, too-short tracks, anonymous / label-fallback / excluded playlists
    titles = {t.title for t in coll.tracks}
    assert not {"Instrumental Dreams (Instrumental)", "Intro", "Label Song", "Calm Keys"} & titles
    assert ("playlist", 901) not in dz.calls and ("playlist", 920) not in dz.calls and ("playlist", 930) not in dz.calls
    # tier 0 (charts, TikTok) first, ordered by best chart position, then Deezer rank
    # (#1s of every list by rank, then the #2s...; "Glass Orchard" keeps the better-ranked chart B version)
    assert [t.title for t in coll.tracks[:7]] == ["Paper Lanterns (Sped Up)", "Neon Harbor", "Global Glow",
                                                  "Río de Cobre", "Velvet Static", "Glass Orchard", "Sunday Static"]
    assert [t.title for t in coll.tracks[7:]] == ["Copper Moon - Remastered 2011", "Long Ago Song"]
    # the better listing's metadata wins (chart version of "Velvet Static" with rank 950, not the rank-10 copy)
    velvet = next(t for t in coll.tracks if t.title == "Velvet Static")
    assert velvet.rank == 950 and velvet.deezer_id == 2
    assert coll.groups[song_key("Velvet Static", "Orla Finch")] == ["chart", "editorial"]
    assert coll.counts() == {"chart": 5, "editorial": 2, "tiktok": 2, "yearly": 1}
    # source report: ok / not_found, with playlist ids; the "Top Brumaria hits" decoy is not a chart
    by = {s["source"]: s for s in coll.sources}
    assert by["playlist:Top Brumaria"]["playlist_id"] == 102 and by["playlist:Top Brumaria"]["new"] == 1
    assert by["search:missing"]["status"] == "not_found"
    assert by["playlist:70s Hits"]["owner"].startswith("Laeti") and by["playlist:70s Hits"]["tracks"] == 3
    assert collect_tracks(world(), PLAN) == coll.tracks


def test_collect_survives_source_failures():
    dz = world(fail={("playlist", 101), ("search", "70s hits"), "chart"})
    coll = collect(dz, PLAN, year=2026)
    status = {s["source"]: s["status"] for s in coll.sources}
    assert status["playlist:Top Arcadia"] == "error" and status["search:70s hits"] == "error"
    assert status["chart:0"] == "error" and status["playlist:Top Brumaria"] == "ok"
    assert "Río de Cobre" in {t.title for t in coll.tracks}
    assert all("message" in s for s in coll.sources if s["status"] == "error")


def test_collect_expands_frequent_and_chart_artists():
    plan = replace(PLAN, top_artists=1, chart_artists=5, artist_tracks=5)
    dz = world()
    coll = collect(dz, plan, year=2026)
    # chart artists first, then Orla Finch (artist 12), the most frequent artist of the lists
    assert [c for c in dz.calls if c[0] == "artist"] == [("artist", 99), ("artist", 12)]
    titles = [t.title for t in coll.tracks]
    assert titles[-2:] == ["Chart Comet", "Static Bloom"]                   # tier 3, by Deezer rank
    assert titles.count("Velvet Static") == 1
    assert coll.groups[song_key("Static Bloom", "Orla Finch")] == ["artist"]


def test_collect_stops_cooperatively_and_reports_progress():
    dz = world()
    seen = []
    coll = collect(dz, PLAN, should_stop=lambda: len(dz.calls) >= 2, progress=lambda p, m: seen.append((p, m)),
                   year=2026)
    assert len(dz.calls) == 2 and {t.title for t in coll.tracks} == {"Global Glow"}
    assert seen and all(0 <= p <= 1 for p, _ in seen) and seen[0][1].startswith("Collecting")


# ---------------------------------------------------------------------------- lyrics phase
LYRICS = {"Global Glow": EN, "Neon Harbor": ES, "Río de Cobre": ES, "Paper Lanterns (Sped Up)": EN,
          "Velvet Static": EN, "Glass Orchard (feat. Kip Rowe)": EN, "Glass Orchard": EN, "Sunday Static": None,
          "Copper Moon - Remastered 2011": EN, "Long Ago Song": None}


@pytest.fixture()
def test_preset(monkeypatch):
    monkeypatch.setitem(PRESETS, "test", PLAN)
    monkeypatch.setattr(sync, "_current_year", lambda: 2026)
    return "test"


def run_sync(index, lrclib, preset, tmp_path, **kw):
    return sync_popular(index, lrclib, world(), preset=preset, state_path=tmp_path / "world_sync.db", **kw)


def test_sync_adds_songs_with_deezer_metadata(tmp_path, test_preset, clock):
    index = WorldLyricsIndex(tmp_path / "world.db")
    lr = FakeLrclib(LYRICS)
    progress = []
    res = run_sync(index, lr, test_preset, tmp_path, progress=lambda p, m: progress.append((p, m)))
    assert res["status"] == "completed"
    assert res["collected"] == 9 and res["to_fetch"] == 9 and res["processed"] == 9
    assert res["added"] == 7 and res["no_lyrics"] == 2 and res["errors"] == 0 and res["skipped_existing"] == 0
    assert res["groups"] == {"chart": 5, "editorial": 2, "tiktok": 2, "yearly": 1}
    assert len(index) == 7 and lr.max_active == 1 and len(lr.calls) == 9
    assert res["lrclib_requests"] == 9 and res["index_songs"] == 7 and res["finished_at"].endswith("+00:00")
    song = index.get_by_key("Neon Harbor", "Lumen Tide")
    assert (song.rank, song.deezer_id, song.isrc, song.album) == (900, 1, "XX0000000001", "Disco Inventado")
    assert song.cover_url.endswith("/1.jpg") and song.preview_url.endswith("/1.mp3") and song.link.endswith("/1")
    assert song.lrclib_id and song.source == "sync:test" and song.language == "es" and "farolas" in song.lyrics
    assert index.get_by_key("Global Glow", "Orla Finch").language == "en"
    # the index finds the synced lyrics
    assert index.search("cuento las luces hasta que llega el dia")[0].song.title in {"Neon Harbor", "Río de Cobre"}
    last = json.loads(index.meta_get("last_sync"))
    assert last["preset"] == "test" and last["added"] == 7 and last["collected"] == 9 and last["songs"] == 7
    # progress: monotonic, ends at 1.0, lyrics messages with an ETA
    ps = [p for p, _ in progress]
    assert ps == sorted(ps) and ps[-1] == 1.0
    assert any(re.match(r"Lyrics 5/9 · \+\d+ · ETA ", m) for _, m in progress)
    state = SyncState(tmp_path / "world_sync.db")
    run = state.last_run()
    assert run["status"] == "completed" and run["done"] == 9 and run["result"]["added"] == 7
    assert state.miss_count() == 2


def test_sync_limit_keeps_the_most_important_songs(tmp_path, test_preset, clock):
    index = WorldLyricsIndex()
    lr = FakeLrclib(LYRICS)
    res = run_sync(index, lr, test_preset, tmp_path, limit=3)
    assert res["collected"] == 3 and lr.calls == ["Paper Lanterns (Sped Up)", "Neon Harbor", "Global Glow"]
    assert res["added"] == 3 and res["groups"] == {"chart": 2, "tiktok": 1}


def test_sync_resumes_after_interruption(tmp_path, test_preset, clock):
    index = WorldLyricsIndex(tmp_path / "world.db")
    lr = FakeLrclib(LYRICS)
    res = run_sync(index, lr, test_preset, tmp_path, should_stop=lambda: len(lr.calls) >= 4)
    assert res["status"] == "cancelled" and res["processed"] == 4 and res["added"] == 4
    assert SyncState(tmp_path / "world_sync.db").last_run()["status"] == "cancelled"
    assert json.loads(index.meta_get("last_sync"))["status"] == "cancelled"
    index.close()
    # a new process: same files, the run continues where it stopped
    index = WorldLyricsIndex(tmp_path / "world.db")
    lr2 = FakeLrclib(LYRICS)
    res2 = run_sync(index, lr2, test_preset, tmp_path)
    assert res2["status"] == "completed" and res2["skipped_existing"] == 4 and res2["to_fetch"] == 5
    assert set(lr.calls).isdisjoint(lr2.calls) and len(lr.calls) + len(lr2.calls) == 9
    assert len(index) == 7


def test_sync_remembers_songs_without_lyrics_for_30_days(tmp_path, test_preset, clock):
    index = WorldLyricsIndex()
    run_sync(index, FakeLrclib(LYRICS), test_preset, tmp_path)
    lr = FakeLrclib(LYRICS)
    res = run_sync(index, lr, test_preset, tmp_path)
    assert lr.calls == [] and res["skipped_recent_miss"] == 2 and res["skipped_existing"] == 7
    assert res["updated"] == 7 and res["added"] == 0
    # 31 days later the misses are retried; one has lyrics by now
    with sqlite3.connect(tmp_path / "world_sync.db") as db:
        db.execute("UPDATE misses SET checked_at = checked_at - 31 * 86400")
    lr = FakeLrclib({**LYRICS, "Sunday Static": EN})
    res = run_sync(index, lr, test_preset, tmp_path)
    assert sorted(lr.calls) == ["Long Ago Song", "Sunday Static"] and res["added"] == 1 and res["no_lyrics"] == 1
    state = SyncState(tmp_path / "world_sync.db")
    assert state.miss_count() == 1 and state.recent_miss(song_key("Long Ago Song", "Old Choir"))
    assert not state.recent_miss(song_key("Sunday Static", "Orla Finch"))


def test_sync_refreshes_metadata_of_indexed_songs_without_lookup(tmp_path, test_preset, clock):
    index = WorldLyricsIndex()
    index.upsert(WorldSong.create("Neon Harbor", "Lumen Tide", lyrics=ES, rank=5, source="lookup", lrclib_id=77))
    lr = FakeLrclib(LYRICS)
    res = run_sync(index, lr, test_preset, tmp_path)
    assert "Neon Harbor" not in lr.calls and res["skipped_existing"] == 1
    song = index.get_by_key("Neon Harbor", "Lumen Tide")
    assert song.rank == 900 and song.deezer_id == 1 and song.cover_url and song.source == "lookup"
    assert song.lrclib_id == 77 and "farolas" in song.lyrics


def test_sync_network_errors_are_counted_and_retried_next_run(tmp_path, test_preset, clock):
    index = WorldLyricsIndex()
    err = WebError("network error", kind="network", host="lrclib.net")
    lr = FakeLrclib(LYRICS, errors={"Neon Harbor": [err]})
    res = run_sync(index, lr, test_preset, tmp_path)
    assert res["status"] == "completed" and res["errors"] == 1 and res["added"] == 6 and res["no_lyrics"] == 2
    lr2 = FakeLrclib(LYRICS)
    res2 = run_sync(index, lr2, test_preset, tmp_path)
    assert lr2.calls == ["Neon Harbor"] and res2["added"] == 1          # errors are not cached as misses


def test_sync_waits_when_rate_limited_then_retries(tmp_path, test_preset, clock):
    index = WorldLyricsIndex()
    limited = WebError("rate limited", kind="rate_limited", status=429, host="lrclib.net", retry_after=42.0)
    lr = FakeLrclib(LYRICS, errors={"Paper Lanterns (Sped Up)": [limited]})
    msgs = []
    res = run_sync(index, lr, test_preset, tmp_path, progress=lambda p, m: msgs.append(m))
    assert res["errors"] == 0 and res["added"] == 7
    assert lr.calls[:3] == ["Paper Lanterns (Sped Up)", "Paper Lanterns (Sped Up)", "Neon Harbor"]
    assert sum(clock.sleeps[:84]) >= 42.0 and any("slow down" in m for m in msgs)


def test_sync_aborts_after_repeated_failures(tmp_path, test_preset, clock, monkeypatch):
    monkeypatch.setattr(sync, "MAX_CONSECUTIVE_ERRORS", 3)
    err = WebError("network error", kind="network", host="lrclib.net")
    lr = FakeLrclib(LYRICS, errors={t: [err] for t in LYRICS})
    res = run_sync(WorldLyricsIndex(), lr, test_preset, tmp_path)
    assert res["status"] == "error" and res["errors"] == 3 and len(lr.calls) == 3
    assert "3 times in a row" in res["message"]


def test_sync_paces_lrclib_requests(tmp_path, test_preset, clock):
    lr = FakeLrclib(LYRICS, requests_per_call=2, cached={"Neon Harbor"})
    res = run_sync(WorldLyricsIndex(), lr, test_preset, tmp_path)
    assert res["lrclib_requests"] == 16
    # 8 network lookups x 2 requests at LRCLIB_COURTESY_RPS; the cached lookup costs nothing
    expected = 2 / sync.LRCLIB_COURTESY_RPS * 7          # the last lookup's pause is never waited for
    assert clock.t == pytest.approx(expected, abs=1e-6)


def test_sync_stop_during_collection_and_deezer_outage(tmp_path, test_preset, clock):
    lr = FakeLrclib(LYRICS)
    res = sync_popular(WorldLyricsIndex(), lr, world(), preset=test_preset, state_path=tmp_path / "s.db",
                       should_stop=lambda: True)
    assert res["status"] == "cancelled" and lr.calls == []
    down = FakeDeezer(fail={"search", "chart", "playlist", "artist", "chart_artists"})
    res = sync_popular(WorldLyricsIndex(), lr, down, preset=test_preset, state_path=tmp_path / "s.db")
    assert res["status"] == "error" and res["collected"] == 0 and "Deezer" in res["message"]
    with pytest.raises(ValueError):
        sync_popular(WorldLyricsIndex(), lr, world(), preset="nope")


def test_sync_state_marks_abandoned_runs_interrupted(tmp_path):
    state = SyncState(tmp_path / "s.db")
    first = state.start_run("quick")
    state.update_run(first, total=10, done=3, result={"added": 2})
    second = state.start_run("quick")
    with sqlite3.connect(tmp_path / "s.db") as db:
        assert db.execute("SELECT status FROM runs WHERE id = ?", (first,)).fetchone()[0] == "interrupted"
    assert state.last_run()["id"] == second and state.last_run()["status"] == "running"
    state.add_miss("a|b", "A", "B", 1, now=1000.0)
    assert state.recent_miss("a|b", now=1000.0 + 29 * 86400) and not state.recent_miss("a|b", now=1000.0 + 31 * 86400)
    state.clear_miss("a|b")
    assert state.miss_count() == 0


def test_default_state_path_is_next_to_the_index(tmp_path, test_preset, clock):
    index = WorldLyricsIndex(tmp_path / "store" / "world.db")
    sync_popular(index, FakeLrclib(LYRICS), world(), preset=test_preset, limit=1)
    assert (tmp_path / "store" / "world_sync.db").exists()


# ---------------------------------------------------------------------------- CLI
def test_cli_world_sync(tmp_path, monkeypatch, capsys):
    from melodyai import cli

    seen = {}

    def fake_sync(index, lrclib, deezer, preset, limit, progress, should_stop, state_path):
        seen.update(preset=preset, limit=limit, state_path=state_path, index_path=index.path,
                    cache=lrclib.web.cache_path, same_web=lrclib.web is deezer.web, stop=should_stop())
        progress(1.0, "done")
        return {"status": "completed", "preset": preset, "collected": 2, "added": 1,
                "sources": [{"source": "chart:0", "group": "chart", "status": "ok", "tracks": 2, "new": 2}]}

    monkeypatch.setattr(sync, "sync_popular", fake_sync)
    cli.main(["world-sync", "--preset", "standard", "--limit", "5", "--store", str(tmp_path / "st")])
    assert seen == {"preset": "standard", "limit": 5, "state_path": tmp_path / "st" / "world_sync.db",
                    "index_path": str(tmp_path / "st" / "world.db"), "cache": tmp_path / "st" / "web_cache.db",
                    "same_web": True, "stop": False}
    out = capsys.readouterr().out
    assert '"added": 1' in out and "chart:0" in out
    with pytest.raises(SystemExit):
        cli.main(["world-sync", "--preset", "huge"])


# ---------------------------------------------------------------------------- one sync per store, across processes
def test_second_sync_on_the_same_store_is_refused(tmp_path, test_preset, clock):
    """Regression: a server sync could start while a CLI sync filled the same store (double LRCLIB rate from one
    IP, concurrent writers) and marked the live run "interrupted"."""
    from melodyai.world.sync import SyncBusy, running_sync, sync_lock

    state_path = tmp_path / "world_sync.db"
    assert running_sync(state_path) is None
    with sync_lock(state_path):                            # held by "another process"
        with pytest.raises(SyncBusy):
            run_sync(WorldLyricsIndex(), FakeLrclib(LYRICS), test_preset, tmp_path)
        assert running_sync(state_path) is not None
    res = run_sync(WorldLyricsIndex(), FakeLrclib(LYRICS), test_preset, tmp_path)     # free again
    assert res["status"] == "completed" and running_sync(state_path) is None
    assert SyncState(state_path).last_run()["result"]["pid"] > 0


def test_live_run_of_an_older_version_without_the_lock_counts_as_running(tmp_path, test_preset, clock):
    from melodyai.world.sync import STALE_RUN_S, SyncBusy, running_sync

    state_path = tmp_path / "world_sync.db"
    with sqlite3.connect(SyncState(state_path).path) as db:      # what a lock-less version writes: no pid
        db.execute("INSERT INTO runs(preset, status, started_at, updated_at, result) VALUES "
                   "('standard', 'running', 1, ?, '{\"added\": 3}')", (__import__("time").time(),))
    assert running_sync(state_path)["preset"] == "standard"
    with pytest.raises(SyncBusy):
        run_sync(WorldLyricsIndex(), FakeLrclib(LYRICS), test_preset, tmp_path)
    with sqlite3.connect(state_path) as db:                      # its heartbeat stops: the process died
        db.execute("UPDATE runs SET updated_at = updated_at - ?", (STALE_RUN_S + 1,))
    assert running_sync(state_path) is None
    assert run_sync(WorldLyricsIndex(), FakeLrclib(LYRICS), test_preset, tmp_path)["status"] == "completed"
    with sqlite3.connect(state_path) as db:
        assert [r[0] for r in db.execute("SELECT status FROM runs ORDER BY id")] == ["interrupted", "completed"]


def test_database_errors_skip_the_song_instead_of_aborting(tmp_path, test_preset, clock, monkeypatch):
    """Regression: an sqlite3 error on one song (e.g. the index locked by another writer) aborted the whole run."""
    index = WorldLyricsIndex()
    real = index.upsert

    def flaky(song):
        if song.title == "Global Glow":
            raise sqlite3.IntegrityError("UNIQUE constraint failed: songs.key")
        return real(song)

    monkeypatch.setattr(index, "upsert", flaky)
    res = run_sync(index, FakeLrclib(LYRICS), test_preset, tmp_path)
    assert res["status"] == "completed" and res["errors"] == 1 and res["added"] == 6


def test_cli_world_sync_reports_a_busy_store(tmp_path, capsys):
    from melodyai import cli
    from melodyai.world.sync import sync_lock

    store = tmp_path / "st"
    with sync_lock(store / "world_sync.db"), pytest.raises(SystemExit) as exc:
        cli.main(["world-sync", "--preset", "quick", "--store", str(store)])
    assert exc.value.code == 1 and "already running" in capsys.readouterr().err


def test_cli_world_search_decides_before_trimming_to_top_k(tmp_path, capsys):
    """Regression: ``--top-k 1`` computed the decision without the runner-up (ambiguous -> confident match)."""
    from melodyai import cli

    store = tmp_path / "st"
    index = WorldLyricsIndex(store / "world.db")
    shared = "Moonlit ferry crossing silver water\nPaper boats are sailing to the shore\n"
    index.upsert(WorldSong.create("Moonlit Ferry", "Ada Quill", lyrics=shared + "Anchors made of daisies\n"
                                  "The harbor cat is dreaming\nWe paint the lighthouse", source="sync:test"))
    index.upsert(WorldSong.create("Ferry Song", "Bruno Vale", lyrics=shared + "Grandpa sells balloons\n"
                                  "The carousel is spinning\nA kite above the market", source="sync:test"))
    index.close()
    q = "moonlit ferry crossing silver water paper boats are sailing to the shore"
    cli.main(["world-search", q, "--offline", "--top-k", "1", "--json", "--store", str(store),
              "--cache", str(tmp_path / "c.db")])
    out = json.loads(capsys.readouterr().out)
    assert len(out["candidates"]) == 1 and out["decision"]["status"] == "uncertain"
