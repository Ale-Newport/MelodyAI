"""Popular-songs sync: fill the world lyrics index with the songs people are most likely to sing.

When people sing from memory they sing songs they heard a lot: current hits, TikTok trends and
classics. This module builds that list from Deezer (editorial / official sources only) and fetches
the lyrics of each song from LRCLIB into :class:`melodyai.world.index.WorldLyricsIndex`.

Collection (:func:`collect` / :func:`collect_tracks`, Deezer only: seconds for quick, minutes for large)
------------------------------------------------------------------------------------------------------
Playlists are resolved by search at run time (editorial playlist ids change; titles are stable), and a
query keeps only playlists whose title matches its pattern and whose owner is trusted:

* charts: Deezer's global and genre charts (``/chart/{genre}/tracks``) and the "Top <Country>"
  playlists of the user "Deezer Charts" (optionally also their yearly "Top <Country> <year>");
* TikTok / viral playlists, decade hits (70s-2020s, English and Spanish), Latin genres (reggaeton,
  pop latino, baladas, rock en español, regional mexicano, salsa, bachata, cumbia...), genre classics
  and Deezer's yearly "Top Hits <year>": playlists owned by Deezer editors (owner name contains
  "Deezer" / "Editor"), or, when none matches, by the major labels' official curators (Filtr, Topsify,
  Digster; a few viral charts take them on equal terms). Playlists of other users are never used;
* (standard / large) the top tracks of the artists that occur most often in the lists above and of
  Deezer's chart artists.

Tracks are deduplicated by :func:`melodyai.world.text.song_key` and ordered by importance: charts and
TikTok first, then editorial lists, yearly lists and artist top tracks. Charts and viral lists are
ordered by popularity in their own market, so their songs are interleaved by best position (every
list's #1, then every #2...; Deezer rank breaks ties): Deezer's global ``rank`` is dominated by its
French audience and would put French rap before the Spanish #1. Other tiers are ordered by rank.
A ``limit`` therefore keeps the most valuable songs.

Lyrics (:func:`sync_popular`)
-----------------------------
For each collected song that is not already indexed and had no lyrics in the last
``MISS_RETRY_DAYS`` days: :meth:`LrclibClient.find` (title, artist, duration, album) -- strictly
sequential, as LRCLIB asks of bulk clients, and paced at ``LRCLIB_COURTESY_RPS`` (below the shared
limiter's rate, so live searches running meanwhile keep a burst reserve) -- then the song is upserted
with its Deezer metadata (rank, cover, preview, link, ISRC, Deezer id) and a guessed language.
Songs already indexed only get their Deezer metadata refreshed (no network).

State lives in ``store/world_sync.db`` (:class:`SyncState`): "no lyrics" misses (retried after 30 days)
and one row per run (counts, status, the pid of the process).

One sync at a time per store, across processes: a run holds an exclusive ``flock`` on
``world_sync.db.lock`` (:func:`sync_lock`; released by the OS if the process dies), so a server sync and a
CLI ``world-sync`` never run side by side (they would double the request rate to LRCLIB from one IP). A
second one gets :class:`SyncBusy`; :func:`running_sync` tells the API what is running. Runs of versions
without the lock are recognised by their recent heartbeat (``updated_at`` within ``STALE_RUN_S``) and
also count as running; ``running`` rows left by dead processes are marked ``interrupted``. Every song is committed as soon as it is processed, so an
interrupted run (Ctrl-C, server restart, ``should_stop``) is resumed by simply running it again: indexed
songs and recent misses are skipped locally, and Deezer responses come from the web cache. A summary
of the last run is stored in the index (``meta_get("last_sync")``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from melodyai.lyrics.normalize import normalize_text
from melodyai.world.deezer import DeezerClient, TrackMeta, storable_preview
from melodyai.world.index import WorldLyricsIndex, WorldSong
from melodyai.world.lrclib import LrclibClient
from melodyai.world.text import guess_language, normalize_title, song_key
from melodyai.world.web import WebError

log = logging.getLogger(__name__)

MISS_RETRY_DAYS = 30            # songs without lyrics on LRCLIB are retried after this many days
LRCLIB_COURTESY_RPS = 1.1       # LRCLIB requests/s of the sync (the shared limiter allows 1.3/s)
MAX_CONSECUTIVE_ERRORS = 8      # abort the lyrics phase when LRCLIB keeps failing (offline, blocked)
RATE_LIMIT_BACKOFF_S = (10.0, 300.0)   # wait bounds after LRCLIB says "rate limited"
MIN_TRACK_S = 30.0              # shorter tracks are intros / skits / jingles
COLLECT_SHARE = 0.08            # share of the progress bar used by the collection phase
VARIOUS_ARTISTS_ID = 5080       # Deezer's "Various Artists"
STALE_RUN_S = 600.0             # a "running" run without a lock owner and no update for this long is dead

# Playlist owners. Matching is done on normalize_text(name) (casefolded, accent-free, no punctuation).
CHARTS_OWNER = "deezer charts"
EDITORIAL_OWNER = re.compile(r"\bdeezer\b|\bedit(?:or|ora|ores|rice|eur|eurs)\b")
LABEL_OWNER = re.compile(r"^(?:filtr|topsify|digster)\b")       # the major labels' official curators
OWNER_MODES = ("charts", "editorial", "official")
# Playlists that are mostly instrumental / functional music, or alternate versions of songs.
EXCLUDE_PLAYLIST = re.compile(r"instrumental|\blo ?fi\b|piano|sleep|dormir|focus|study|estudiar|meditat|ambient|"
                              r"white noise|ruido blanco|\bkids?\b|infantil|ninos|nightcore|\b8d\b|slowed|sped up|"
                              r"karaoke version|remix|^100\b")         # "100% <Artist>": one artist only
EXCLUDE_TRACK = re.compile(r"\binstrumental\b|\bkaraoke\b|\bbacking track\b|\b8d audio\b")

# Importance tiers (lower = synced first). Tier-0 lists are popularity-ordered (charts, viral charts), so
# their songs are ordered by their best position in any list; the others by Deezer rank.
TIERS = {"chart": 0, "tiktok": 0, "editorial": 1, "yearly": 2, "artist": 3}
TIKTOK = r"tik ?tok|viral|virais|virale|trend|tendencia"


@dataclass(frozen=True)
class PlaylistQuery:
    """Deezer playlists found by searching ``query``: those whose normalised title matches the regex ``title``
    and whose owner fits ``owner``: ``"charts"`` = the user "Deezer Charts"; ``"editorial"`` = Deezer editors,
    else (fallback) the labels' curators; ``"official"`` = editors or label curators alike."""

    query: str
    group: str = "editorial"            # importance tier, see TIERS
    title: str | None = None            # regex searched in normalize_text(playlist title); None = any
    max_playlists: int = 1
    owner: str = "editorial"            # see OWNER_MODES

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SyncPlan:
    """What a preset collects. ``approx_*`` are measured estimates shown in the UI."""

    name: str
    label: str
    approx_songs: int                   # songs added to an empty index (lyrics found)
    approx_minutes: int                 # wall time of a first run (LRCLIB-bound)
    chart_genres: tuple[int, ...] = (0,)                 # Deezer chart genre ids (0 = all genres)
    countries: tuple[str, ...] = ()                      # "Top <Country>" playlists of Deezer Charts
    country_years: int = 0                               # also "Top <Country> <year>" for the last N years
    playlists: tuple[PlaylistQuery, ...] = ()
    best_of_from: int | None = None                      # Deezer Best Of "Top Hits <year>" from this year
    tracks_per_playlist: int = 100
    top_artists: int = 0                                 # expand the N most frequent artists of the lists
    chart_artists: int = 0                               # ... and the N most popular chart artists
    artist_tracks: int = 10                              # top tracks per expanded artist

    def to_dict(self) -> dict:
        d = asdict(self)
        d["playlists"] = [p.to_dict() for p in self.playlists]
        return d

    def summary(self) -> dict:
        """``{name, label, approx_songs, approx_minutes}`` for the API / UI."""
        return {"name": self.name, "label": self.label, "approx_songs": self.approx_songs,
                "approx_minutes": self.approx_minutes}


def _pl(query: str, title: str | None = None, n: int = 1, group: str = "editorial",
        owner: str = "editorial") -> PlaylistQuery:
    return PlaylistQuery(query, group, title, n, owner)


# Markets for a Spanish-speaking listener who also listens to English music (verified 2026-11: each has a
# "Top <Country>" playlist by Deezer Charts, Deezer's own spelling; e.g. the Dominican Republic and Uruguay
# have none). Deezer Charts' yearly lists ("Top México 2025") exist only for the last full year and only for
# some markets, so ``country_years`` > 1 finds nothing.
_QUICK_COUNTRIES = ("Worldwide", "Spain", "Mexico", "USA", "Argentina", "Colombia", "UK")
_MORE_COUNTRIES = ("Chile", "Peru", "Venezuela", "Ecuador", "France", "Brazil", "Italy", "Germany", "Portugal")
# Yearly charts use the local name ("Top México 2025", "Top Brasil 2025").
_LOCAL_NAMES = {"Spain": "España", "Mexico": "México", "Brazil": "Brasil", "Italy": "Italia",
                "Germany": "Deutschland", "Peru": "Perú"}
_LARGE_COUNTRIES = ("Bolivia", "Paraguay", "Costa Rica", "Guatemala", "Australia", "Ireland")

_QUICK_PLAYLISTS = (
    # TikTok / viral (tier 0)
    _pl("tiktok hits world", r"tik ?tok", 1, "tiktok"),
    _pl("salidas de tiktok", r"tik ?tok", 1, "tiktok"),
    _pl("tiktok viral hits chart", r"tik ?tok", 1, "tiktok", "official"),
    _pl("top 50 viral españa", r"viral", 1, "tiktok", "official"),
    # decades, English
    _pl("70s hits", r"\b70s\b"), _pl("80s hits", r"\b80s\b"), _pl("90s hits", r"\b90s\b"),
    _pl("2000s hits", r"\b(?:20)?00s\b"), _pl("2010s hits", r"\b(?:20)?10s\b"), _pl("2020s hits", r"\b(?:20)?20s\b"),
    # decades / classics, Spanish
    _pl("pop 80s español", r"\b80s\b"), _pl("pop en español 90s", r"\b90s\b"),
    _pl("éxitos 2000 español", r"\b00s\b"),
    _pl("esenciales pop en español", r"pop en espanol"), _pl("rock en español", r"rock en espanol|rock"),
    _pl("baladas en español", r"baladas? en espanol|baladas"),
    # Latin genres
    _pl("reggaeton classics", r"reggaeton"), _pl("reggaeton hits", r"reggaeton"), _pl("pop latino", r"pop latino"),
    _pl("regional mexicano", r"corrido|banda|norten|regional"), _pl("salsa", r"salsa"),
    _pl("pura bachata", r"bachata"), _pl("cumbia", r"cumbia"),
    # sing-along classics
    _pl("karaoke hits", r"karaoke hits|sing ?along"), _pl("pop essentials", r"pop essentials"),
)

_STANDARD_PLAYLISTS = _QUICK_PLAYLISTS + (
    _pl("tiktok", TIKTOK, 3, "tiktok"), _pl("viral dance", r"viral", 1, "tiktok"),
    _pl("tiktok songs", r"tik ?tok", 2, "tiktok", "official"),
    _pl("80s pop", r"\b80s\b", 2), _pl("90s pop", r"\b90s\b", 2), _pl("00s ballads", r"\b00s\b"),
    _pl("10s pop", r"\b10s\b"), _pl("love ballads", r"ballad"), _pl("power ballads", r"ballad"),
    _pl("rock essentials", r"rock", 2), _pl("classic rock", r"rock", 2), _pl("80s rock", r"\b80s\b"),
    _pl("hip hop classics", r"hip hop|rap|east coast|west coast", 2), _pl("r&b essentials", r"r ?b|rnb"),
    _pl("dance essentials", r"dance|club"), _pl("disco", r"disco"), _pl("country hits", r"country"),
    _pl("baladas viejitas en español", r"balada", 1, owner="official"),
    _pl("reggaeton", r"reggaeton", 2), _pl("dembow", r"dembow"),
    _pl("latin music 00s", r"\b00s\b|2000"), _pl("latin music 10s", r"\b10s\b|2010"),
    _pl("top hits latinoamérica", r"latino|latin"), _pl("pop méxico", r"mexico"), _pl("boleros", r"bolero"),
    _pl("rancheras", r"ranchera|mariachi"), _pl("banda norteña", r"banda|norten"), _pl("vallenato", r"vallenat"),
    _pl("flamenco", r"flamenco"), _pl("pop español", r"pop espanol|espana"), _pl("indie español", r"indie"),
    _pl("duets essentials", r"duet|duo"), _pl("boy band essentials", r"boy ?band"),
    _pl("film hits", r"film|movie|soundtrack|cine", 2),
    _pl("summer hits", r"summer|verano"), _pl("feel good classics", r"feel good|happy"),
)

_LARGE_PLAYLISTS = _STANDARD_PLAYLISTS + tuple(
    _pl(*args) for args in (
        ("70s rock", r"\b70s\b", 2), ("60s hits", r"\b60s\b", 2), ("60s rock", r"\b60s\b"), ("50s", r"\b50s\b"),
        ("70s ballads", r"\b70s\b", 2), ("80s ballads", r"\b80s\b", 2), ("90s ballads", r"\b90s\b", 2),
        ("90s rock", r"\b90s\b"), ("00s party hits", r"\b00s\b", 2), ("10s ballads", r"\b10s\b"),
        ("20s pop", r"\b20s\b", 2), ("hard rock essentials", r"rock"), ("metal essentials", r"metal"),
        ("pop punk essentials", r"punk"), ("brit pop essentials", r"brit"), ("alternative essentials", r"altern"),
        ("new wave essentials", r"wave"), ("synth pop", r"synth"), ("soul classics", r"soul|motown|funk", 2),
        ("rap essentials", r"rap|hip hop", 3), ("k-pop essentials", r"k ?pop", 2), ("j-pop", r"j ?pop"),
        ("reggae", r"reggae"), ("latin love", r"latin|amor|love"), ("salsa hits", r"salsa", 2),
        ("cumbia hits", r"cumbia", 2), ("corridos", r"corrido", 2), ("tango", r"tango"), ("rumba", r"rumba"),
        ("cantautores", r"cantautor|autor"), ("rock en español 80s", r"rock"), ("pop latino 90s", r"\b90s\b|latin"),
        ("música brasileira", r"brasil|mpb", 2), ("sertanejo", r"sertanejo"),
        ("variété française", r"variete|fran", 2),
        ("italian hits", r"ital"), ("eurovision", r"eurovision"),
        ("christmas hits", r"christmas|navidad|noel"), ("anime", r"anime"), ("gospel", r"gospel|worship"),
        ("country classics", r"country", 2), ("blues", r"blues"), ("acoustic hits", r"acoustic|acustic", 2),
        ("party hits", r"party|fiesta", 3), ("workout hits", r"workout|gym"), ("road trip", r"road"),
    ))


# Estimates (measured 2026-11-07): collection gives ~2.0k / 7.5k / 26k songs; a first run's lyrics phase
# costs ~0.95 s per song (1.05 LRCLIB requests per lookup at LRCLIB_COURTESY_RPS); 94 % of the quick
# list had lyrics (fewer expected for artist deep cuts). Later runs only look up new songs.
PRESETS: dict[str, SyncPlan] = {
    "quick": SyncPlan(
        name="quick", label="Quick: current charts, TikTok trends and the classics everybody knows",
        approx_songs=1900, approx_minutes=33, chart_genres=(0, 197), countries=_QUICK_COUNTRIES,
        playlists=_QUICK_PLAYLISTS, tracks_per_playlist=100),
    "standard": SyncPlan(
        name="standard", label="Standard: more countries, genres, yearly hits since 2000 and top artists",
        approx_songs=6950, approx_minutes=130,      # measured 2026-11-07: 6,954 songs in 2 h 08 min
        chart_genres=(0, 197, 132, 116, 152, 113, 165, 85, 106, 144, 84, 169),
        countries=_QUICK_COUNTRIES + _MORE_COUNTRIES, country_years=1, playlists=_STANDARD_PLAYLISTS,
        best_of_from=2000, tracks_per_playlist=150, top_artists=150, chart_artists=100, artist_tracks=10),
    "large": SyncPlan(
        name="large", label="Large: decades of hits from 1970, many genres and the top songs of ~1000 artists",
        approx_songs=24600, approx_minutes=440,     # measured: 24,640 songs, 7 h 20 min from empty
        chart_genres=(0, 197, 132, 116, 152, 113, 165, 85, 106, 144, 84, 169, 464, 466, 153, 173, 75, 2, 16),
        countries=_QUICK_COUNTRIES + _MORE_COUNTRIES + _LARGE_COUNTRIES, country_years=1, playlists=_LARGE_PLAYLISTS,
        best_of_from=1970, tracks_per_playlist=250, top_artists=900, chart_artists=100, artist_tracks=25),
}


def preset_list() -> list[dict]:
    """Presets for the API / UI, smallest first."""
    return [p.summary() for p in PRESETS.values()]


def _current_year() -> int:
    return datetime.now(timezone.utc).year


# ---------------------------------------------------------------------------- collection
@dataclass
class _Entry:
    track: TrackMeta
    tier: int
    order: int                                  # first time the song was seen (stable tie-break)
    groups: set[str] = field(default_factory=set)
    pos: int = 1 << 30                          # best position in a tier-0 (popularity-ordered) list

    def sort_key(self) -> tuple:
        return self.tier, self.pos if self.tier == 0 else 0, -self.track.rank, self.order


@dataclass
class Collection:
    """Result of :func:`collect`: songs in sync order plus a report of every source."""

    tracks: list[TrackMeta]
    sources: list[dict]                         # {source, group, status, tracks, new, playlist_id?, message?}
    groups: dict[str, list[str]]                # song_key -> groups that listed it
    elapsed_s: float = 0.0

    def counts(self) -> dict:
        """Songs per group (a song listed by several groups counts in each)."""
        c: Counter = Counter()
        for gs in self.groups.values():
            c.update(gs)
        return dict(sorted(c.items()))

    def to_dict(self) -> dict:
        return {"collected": len(self.tracks), "elapsed_s": round(self.elapsed_s, 2), "groups": self.counts(),
                "sources": self.sources}


def _usable(tm: TrackMeta) -> bool:
    if not normalize_title(tm.title or "") or not (tm.artist or "").strip():
        return False
    if tm.duration_s is not None and tm.duration_s < MIN_TRACK_S:
        return False
    return not EXCLUDE_TRACK.search(normalize_text(tm.title))


def _owner_tier(user: str, owner: str) -> int | None:
    """0 = preferred owner, 1 = acceptable fallback (label curator), None = never used (anonymous users)."""
    u = normalize_text(user)
    if owner == "charts":
        return 0 if u == CHARTS_OWNER else None
    if EDITORIAL_OWNER.search(u):
        return 0
    if LABEL_OWNER.search(u):
        return 0 if owner == "official" else 1
    return None


def select_playlists(found: list[dict], pq: PlaylistQuery, exclude: set[int] | None = None) -> list[dict]:
    """The playlists of a search result that ``pq`` keeps: title / owner rules, editors before label curators,
    search order otherwise; at most ``pq.max_playlists``; ``exclude`` = playlist ids already used."""
    pattern = re.compile(pq.title) if pq.title else None
    ranked = []
    for pos, p in enumerate(found):
        title = normalize_text(p.get("title") or "")
        tier = _owner_tier(p.get("user") or "", pq.owner)
        if tier is None or not title or (exclude and p.get("id") in exclude) or (p.get("nb_tracks") or 0) < 5:
            continue
        if EXCLUDE_PLAYLIST.search(title) or (pattern is not None and not pattern.search(title)):
            continue
        ranked.append((tier, pos, p))
    ranked.sort(key=lambda x: (x[0], x[1]))
    return [p for _, _, p in ranked[:max(0, pq.max_playlists)]]


def _plan_queries(plan: SyncPlan, year: int) -> list[PlaylistQuery]:
    """Every playlist query of a plan, in priority order."""
    out = [PlaylistQuery(f"Top {c}", "chart", rf"^top {re.escape(normalize_text(c))}$", 1, "charts")
           for c in plan.countries]
    out += [p for p in plan.playlists if TIERS.get(p.group, 1) == 0]
    out += [p for p in plan.playlists if TIERS.get(p.group, 1) != 0]
    for k in range(1, plan.country_years + 1):
        for c in plan.countries:
            local = _LOCAL_NAMES.get(c, c)
            names = "|".join(dict.fromkeys(re.escape(normalize_text(n)) for n in (c, local)))
            out.append(PlaylistQuery(f"Top {local} {year - k}", "yearly", rf"^top (?:{names}) {year - k}$", 1,
                                     "charts"))
    if plan.best_of_from:
        out += [PlaylistQuery(f"Top Hits {y}", "yearly", rf"^top hits {y}$")
                for y in range(year - 1, plan.best_of_from - 1, -1)]
    return out


def collect(deezer: DeezerClient, plan: SyncPlan, should_stop: Callable[[], bool] | None = None,
            progress: Callable[[float, str], None] | None = None, year: int | None = None) -> Collection:
    """Collect the songs of ``plan`` from Deezer (see module docstring), deduplicated and ordered by importance.

    Failures of single sources (network, quota) are reported in ``Collection.sources`` and skipped.
    ``progress`` receives (fraction of the collection, message).
    """
    t0 = time.perf_counter()
    stop = should_stop or (lambda: False)
    year = year or _current_year()
    entries: dict[str, _Entry] = {}
    sources: list[dict] = []
    used_playlists: set[int] = set()

    def add(tracks: list[TrackMeta], group: str) -> tuple[int, int]:
        tier = TIERS.get(group, 1)
        n = new = 0
        for pos, tm in enumerate(tracks):
            if not _usable(tm):
                continue
            n += 1
            key = song_key(tm.title, tm.artist)
            e = entries.get(key)
            if e is None:
                e = entries[key] = _Entry(tm, tier, len(entries), {group})
                new += 1
            else:
                e.groups.add(group)
                if (tier, -tm.rank) < (e.tier, -e.track.rank):     # better listing: keep this version's metadata
                    e.track = tm
                e.tier = min(e.tier, tier)
            if tier == 0:
                e.pos = min(e.pos, pos)
        return n, new

    def run(name: str, group: str, fetch: Callable[[], list[TrackMeta]], **extra) -> None:
        try:
            tracks = fetch()
        except WebError as e:
            log.warning("world sync: source %s failed: %s", name, e)
            sources.append({"source": name, "group": group, "status": "error", "tracks": 0, "new": 0,
                            "message": str(e)[:200], **extra})
            return
        n, new = add(tracks, group)
        sources.append({"source": name, "group": group, "status": "ok", "tracks": n, "new": new, **extra})

    queries = _plan_queries(plan, year)
    expand = bool(plan.top_artists or plan.chart_artists)
    steps = len(plan.chart_genres) + len(queries) + (1 if expand else 0)
    done = 0

    def tick(msg: str, part: float = 1.0) -> None:
        nonlocal done
        done += part
        if progress:
            progress(min(1.0, done / max(1, steps)), msg)

    for gid in plan.chart_genres:
        if stop():
            break
        run(f"chart:{gid}", "chart", lambda gid=gid: deezer.chart_tracks(gid, 100), genre_id=gid)
        tick(f"Collecting: chart {gid} ({len(entries)} songs)")
    for pq in queries:
        if stop():
            break
        try:
            found = deezer.search_playlists(pq.query, 25)
        except WebError as e:
            sources.append({"source": f"search:{pq.query}", "group": pq.group, "status": "error", "tracks": 0,
                            "new": 0, "message": str(e)[:200]})
            tick(f"Collecting: {pq.query}")
            continue
        chosen = select_playlists(found, pq, used_playlists)
        if not chosen:
            sources.append({"source": f"search:{pq.query}", "group": pq.group, "status": "not_found", "tracks": 0,
                            "new": 0})
        for p in chosen:
            if stop():
                break
            used_playlists.add(p["id"])
            run(f"playlist:{p['title']}", pq.group,
                lambda p=p: deezer.playlist_tracks(p["id"], plan.tracks_per_playlist),
                playlist_id=p["id"], owner=p.get("user") or "", query=pq.query)
        tick(f"Collecting: {pq.query} ({len(entries)} songs)")
    if expand and not stop():
        _expand_artists(deezer, plan, entries, run, stop, sources,
                        lambda k, n: tick(f"Collecting: artist {k}/{n} ({len(entries)} songs)", 1.0 / max(1, n)))
    ordered = sorted(entries.items(), key=lambda kv: kv[1].sort_key())
    return Collection(tracks=[e.track for _, e in ordered], sources=sources,
                      groups={k: sorted(e.groups) for k, e in ordered}, elapsed_s=time.perf_counter() - t0)


def _expand_artists(deezer: DeezerClient, plan: SyncPlan, entries: dict[str, _Entry], run, stop,
                    sources: list[dict], tick: Callable[[int, int], None]) -> None:
    """Top tracks of Deezer's chart artists and of the artists listed most often so far."""
    artists: dict[int, str] = {}
    if plan.chart_artists:
        try:
            for a in deezer.chart_artists(0, plan.chart_artists):
                artists.setdefault(a["id"], a["name"])
        except WebError as e:
            sources.append({"source": "chart_artists", "group": "artist", "status": "error", "tracks": 0, "new": 0,
                            "message": str(e)[:200]})
    freq: Counter = Counter()
    best: dict[int, int] = {}
    names: dict[int, str] = {}
    for e in entries.values():
        aid = e.track.artist_id
        if aid and aid != VARIOUS_ARTISTS_ID:
            freq[aid] += 1
            best[aid] = max(best.get(aid, 0), e.track.rank)
            names.setdefault(aid, e.track.artist)
    frequent = sorted(freq, key=lambda a: (-freq[a], -best[a], a))[:plan.top_artists]
    for aid in frequent:
        artists.setdefault(aid, names[aid])
    for k, (aid, name) in enumerate(artists.items(), 1):
        if stop():
            break
        run(f"artist:{name}", "artist", lambda aid=aid: deezer.artist_top(aid, plan.artist_tracks), artist_id=aid)
        tick(k, len(artists))


def collect_tracks(deezer: DeezerClient, plan: SyncPlan,
                   should_stop: Callable[[], bool] | None = None) -> list[TrackMeta]:
    """The songs of ``plan``, deduplicated by song key, most important first (see :func:`collect`)."""
    return collect(deezer, plan, should_stop).tracks


# ---------------------------------------------------------------------------- persistent state
_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS misses(
    key TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    deezer_id INTEGER,
    reason TEXT NOT NULL,
    checked_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs(
    id INTEGER PRIMARY KEY,
    preset TEXT NOT NULL,
    status TEXT NOT NULL,             -- running | completed | cancelled | error | interrupted
    started_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    finished_at REAL,
    total INTEGER NOT NULL DEFAULT 0, -- songs to look up
    done INTEGER NOT NULL DEFAULT 0,
    result TEXT                       -- JSON counts
);
"""


class SyncBusy(RuntimeError):
    """Another process (or thread) is running a popular-songs sync on the same store."""


def _lock_path(state_path: str | Path) -> Path | None:
    p = str(state_path)
    return None if p in (":memory:", "") or p.startswith("file:") else Path(p + ".lock")


@contextmanager
def sync_lock(state_path: str | Path, wait_s: float = 1.0) -> Iterator[None]:
    """Hold the store's sync lock (exclusive ``flock``); :class:`SyncBusy` if another run still holds it after
    ``wait_s`` (a short wait, so that a status probe holding it for an instant never refuses a real sync)."""
    path = _lock_path(state_path)
    try:
        import fcntl
    except ImportError:              # pragma: no cover - no flock (Windows): rely on the run heartbeat only
        fcntl = None
    if path is None or fcntl is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        end = time.monotonic() + max(0.0, wait_s)
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= end:
                    raise SyncBusy("A popular-songs sync is already running on this store (another process)") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _lock_held(state_path: str | Path) -> bool:
    path = _lock_path(state_path)
    if path is None or not path.exists():
        return False
    try:
        with sync_lock(state_path, wait_s=0.0):
            return False
    except SyncBusy:
        return True


def running_sync(state_path: str | Path, now: float | None = None) -> dict | None:
    """The sync running on this store, if any (``{preset, started_at, updated_at, total, done, pid}``): one that
    holds the lock, or a run of an older version (no pid, no lock) with a recent heartbeat. Read-only."""
    held = _lock_held(state_path)
    path = Path(str(state_path))
    run = None
    if path.exists():
        try:
            db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
            try:
                rows = db.execute("SELECT preset, started_at, updated_at, total, done, result FROM runs "
                                  "WHERE status = 'running' ORDER BY id DESC LIMIT 20").fetchall()
            finally:
                db.close()
        except sqlite3.Error:
            rows = []
        now = time.time() if now is None else now
        for preset, started, updated, total, done, result in rows:
            pid = _run_pid(result)
            if (held and pid is not None) or (pid is None and now - float(updated) < STALE_RUN_S):
                run = {"preset": preset, "started_at": started, "updated_at": updated, "total": total,
                       "done": done, "pid": pid}
                break
    if held and run is None:
        run = {"preset": None, "started_at": None, "updated_at": None, "total": 0, "done": 0, "pid": None}
    return run


def _run_pid(result: str | None) -> int | None:
    try:
        pid = json.loads(result).get("pid") if result else None
    except (ValueError, AttributeError):
        return None
    return int(pid) if isinstance(pid, int) else None


class SyncState:
    """``world_sync.db``: songs without lyrics (retried after ``MISS_RETRY_DAYS``) and the history of runs."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.pid = os.getpid()
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        with self._lock, self.conn:
            if self.path != ":memory:":
                self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(_STATE_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # misses
    def recent_miss(self, key: str, now: float | None = None, max_age_days: float = MISS_RETRY_DAYS) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            row = self.conn.execute("SELECT checked_at FROM misses WHERE key = ?", (key,)).fetchone()
        return row is not None and now - float(row[0]) < max_age_days * 86400

    def add_miss(self, key: str, title: str, artist: str, deezer_id: int | None = None, reason: str = "no_lyrics",
                 now: float | None = None) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO misses(key, title, artist, deezer_id, reason, checked_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET reason = excluded.reason, checked_at = excluded.checked_at",
                (key, title, artist, deezer_id, reason, time.time() if now is None else now))

    def clear_miss(self, key: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM misses WHERE key = ?", (key,))

    def miss_count(self) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM misses").fetchone()[0])

    # runs
    def legacy_live_run(self, now: float | None = None) -> dict | None:
        """A ``running`` run of a version without the sync lock (no pid) that updated recently."""
        now = time.time() if now is None else now
        with self._lock:
            rows = self.conn.execute("SELECT id, preset, updated_at, result FROM runs WHERE status = 'running' "
                                     "AND updated_at >= ? ORDER BY id DESC", (now - STALE_RUN_S,)).fetchall()
        for rid, preset, updated, result in rows:
            if _run_pid(result) is None:
                return {"id": rid, "preset": preset, "updated_at": updated}
        return None

    def start_run(self, preset: str, now: float | None = None) -> int:
        """New ``running`` run. Call it holding :func:`sync_lock` (after :meth:`legacy_live_run`): any other
        ``running`` row then belongs to a dead process and is marked ``interrupted``."""
        now = time.time() if now is None else now
        with self._lock, self.conn:
            self.conn.execute("UPDATE runs SET status = 'interrupted', finished_at = updated_at "
                              "WHERE status = 'running'")      # left over by a killed process
            cur = self.conn.execute(
                "INSERT INTO runs(preset, status, started_at, updated_at, result) VALUES (?, ?, ?, ?, ?)",
                (preset, "running", now, now, json.dumps({"pid": self.pid})))
            return int(cur.lastrowid)

    def update_run(self, run_id: int, *, total: int, done: int, result: dict, status: str = "running") -> None:
        now = time.time()
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE runs SET status = ?, updated_at = ?, total = ?, done = ?, result = ?, "
                "finished_at = CASE WHEN ? = 'running' THEN NULL ELSE ? END WHERE id = ?",
                (status, now, total, done, json.dumps({**result, "pid": self.pid}, default=str), status, now,
                 run_id))

    def last_run(self) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT id, preset, status, started_at, updated_at, finished_at, total, done, "
                                    "result FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        keys = ("id", "preset", "status", "started_at", "updated_at", "finished_at", "total", "done")
        out = dict(zip(keys, row[:8]))
        out["result"] = json.loads(row[8]) if row[8] else None
        return out


# ---------------------------------------------------------------------------- lyrics phase
_sleep = time.sleep               # patched by tests
_monotonic = time.monotonic


def _lrclib_requests(lrclib: LrclibClient) -> int:
    """Network requests sent to LRCLIB so far by this client's WebClient (0 when unknown, e.g. a fake).

    Reads WebClient's per-host counter directly: ``stats()`` also scans the whole cache table, too slow to
    call for every song of a large sync.
    """
    web = getattr(lrclib, "web", None)
    counts = getattr(web, "_host_counts", None)
    try:
        if isinstance(counts, dict):
            return int(counts.get("lrclib.net", {}).get("requests", 0))
        return int(web.stats()["hosts"].get("lrclib.net", {}).get("requests", 0)) if web is not None else 0
    except Exception:  # noqa: BLE001 - statistics only
        return 0


def _wait(seconds: float, stop: Callable[[], bool]) -> bool:
    """Sleep ``seconds`` in short slices; False if ``stop()`` became true meanwhile."""
    end = _monotonic() + max(0.0, seconds)
    while True:
        if stop():
            return False
        left = end - _monotonic()
        if left <= 0:
            return True
        _sleep(min(0.5, left))


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds // 3600} h {round(seconds % 3600 / 60):02d} min"


def _fraction(done: int, total: int) -> float:
    """Overall progress during the lyrics phase (the collection used the first ``COLLECT_SHARE``)."""
    return min(1.0, COLLECT_SHARE + (1.0 - COLLECT_SHARE) * done / max(1, total))


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _song(tm: TrackMeta, text: str, lrclib_id: int | None, source: str, now: float) -> WorldSong:
    return WorldSong.create(
        tm.title, tm.artist, album=tm.album, duration_s=tm.duration_s,
        language=guess_language(text) if text else None, lyrics=text, rank=int(tm.rank or 0),
        deezer_id=tm.deezer_id, isrc=tm.isrc, lrclib_id=lrclib_id, cover_url=tm.cover_url, link=tm.link,
        preview_url=storable_preview(tm.preview_url), source=source, added_at=now)


def sync_popular(index: WorldLyricsIndex, lrclib: LrclibClient, deezer: DeezerClient, preset: str = "quick",
                 limit: int | None = None, progress: Callable[[float, str], None] | None = None,
                 should_stop: Callable[[], bool] | None = None, state_path: Path | None = None) -> dict:
    """Collect the songs of ``preset`` and add the lyrics of the missing ones to ``index`` (resumable).

    ``limit`` keeps only the ``limit`` most important collected songs. ``state_path`` defaults to
    ``world_sync.db`` next to the index file (in memory for an in-memory index). Returns
    ``{status, preset, collected, added, skipped_existing, skipped_recent_miss, no_lyrics, errors,
    elapsed_s, ...}`` with ``status`` = completed | cancelled | error; never raises for network errors.
    Raises :class:`SyncBusy` when another sync runs on the same store (see :func:`sync_lock`).
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r} (choose from {', '.join(PRESETS)})")
    plan = PRESETS[preset]
    t0 = time.perf_counter()
    stop = should_stop or (lambda: False)
    report = progress or (lambda p, m: None)
    if state_path is None:
        state_path = Path(index.path).with_name("world_sync.db") if index.path not in (":memory:", "") \
            and not index.path.startswith("file:") else ":memory:"
    with sync_lock(state_path):
        return _sync_locked(index, lrclib, deezer, preset, plan, limit, report, stop, state_path, t0)


def _sync_locked(index: WorldLyricsIndex, lrclib: LrclibClient, deezer: DeezerClient, preset: str, plan: SyncPlan,
                 limit: int | None, report: Callable[[float, str], None], stop: Callable[[], bool],
                 state_path: Path | str, t0: float) -> dict:
    state = SyncState(state_path)
    legacy = state.legacy_live_run()
    if legacy is not None:
        state.close()
        raise SyncBusy(f"A popular-songs sync ({legacy['preset']}) is already running in another process")
    run_id = state.start_run(preset)
    source = f"sync:{preset}"
    res: dict = {"status": "completed", "preset": preset, "limit": limit, "collected": 0, "to_fetch": 0,
                 "processed": 0, "added": 0, "updated": 0, "skipped_existing": 0, "skipped_recent_miss": 0,
                 "no_lyrics": 0, "errors": 0, "lrclib_requests": 0, "collect_s": 0.0, "lyrics_s": 0.0,
                 "elapsed_s": 0.0, "groups": {}, "sources": [], "message": ""}
    try:
        report(0.0, f"Collecting songs ({preset})")
        coll = collect(deezer, plan, stop, lambda p, m: report(COLLECT_SHARE * p, m))
        tracks = coll.tracks[:limit] if limit is not None and limit >= 0 else coll.tracks
        res.update(collected=len(tracks), collect_s=round(coll.elapsed_s, 2), sources=coll.sources,
                   groups=Counter(g for tm in tracks for g in coll.groups.get(song_key(tm.title, tm.artist), ())))
        res["groups"] = dict(sorted(res["groups"].items()))
        if stop():
            res["status"] = "cancelled"
        elif not tracks:
            res["status"] = "error" if any(s["status"] == "error" for s in coll.sources) else "completed"
            res["message"] = "No songs collected" + (" (Deezer unreachable?)" if res["status"] == "error" else "")
        else:
            _lyrics_phase(index, lrclib, state, run_id, tracks, source, res, report, stop)
    except KeyboardInterrupt:
        res["status"] = "cancelled"
        raise
    except Exception as e:
        res["status"], res["message"] = "error", f"Sync failed: {e}"[:300]
        raise
    finally:
        res["elapsed_s"] = round(time.perf_counter() - t0, 2)
        finished = time.time()
        if not res["message"]:
            res["message"] = (f"{res['status'].capitalize()}: +{res['added']} songs ({res['processed']} looked up, "
                              f"{res['no_lyrics']} without lyrics) in {_fmt_duration(res['elapsed_s'])}")
        res["finished_at"] = _iso(finished)
        res["index_songs"] = len(index)
        counts = {k: v for k, v in res.items() if k not in ("sources",)}
        state.update_run(run_id, total=res["to_fetch"], done=res["processed"], result=counts, status=res["status"])
        state.close()
        index.meta_set("last_sync", json.dumps(
            {"preset": preset, "status": res["status"], "finished_at": res["finished_at"],
             "collected": res["collected"], "added": res["added"], "no_lyrics": res["no_lyrics"],
             "errors": res["errors"], "skipped_existing": res["skipped_existing"], "elapsed_s": res["elapsed_s"],
             "songs": res["index_songs"]}))
    report(1.0, res["message"])
    return res


def _lyrics_phase(index: WorldLyricsIndex, lrclib: LrclibClient, state: SyncState, run_id: int,
                  tracks: list[TrackMeta], source: str, res: dict, report, stop) -> None:
    now = time.time()
    todo: list[TrackMeta] = []
    refresh: list[WorldSong] = []
    for tm in tracks:                                           # local checks only: fast
        key = song_key(tm.title, tm.artist)
        if index.has_key(tm.title, tm.artist):
            res["skipped_existing"] += 1
            refresh.append(_song(tm, "", None, source, now))    # Deezer metadata (rank, cover...) only
        elif state.recent_miss(key, now):
            res["skipped_recent_miss"] += 1
        else:
            todo.append(tm)
    if refresh:
        for k in range(0, len(refresh), 500):
            try:
                index.upsert_many(refresh[k:k + 500])
                res["updated"] += len(refresh[k:k + 500])
            except sqlite3.Error as e:                # e.g. the index stayed locked: refreshed next run
                log.warning("world sync: metadata refresh failed: %s", type(e).__name__)
                res["errors"] += 1
    res["to_fetch"] = total = len(todo)
    t_start = _monotonic()
    req0 = before = _lrclib_requests(lrclib)
    next_allowed = t_start
    consecutive = 0
    for i, tm in enumerate(todo):
        if stop() or not _wait(next_allowed - _monotonic(), stop):
            res["status"] = "cancelled"
            break
        key = song_key(tm.title, tm.artist)
        started = _monotonic()
        rec, err = None, None
        for attempt in range(2):
            try:
                rec = lrclib.find(tm.title, tm.artist, tm.duration_s, tm.album)
                err = None
                break
            except WebError as e:
                err = e
                if e.kind != "rate_limited" or attempt == 1:
                    break
                lo, hi = RATE_LIMIT_BACKOFF_S
                wait = min(hi, max(lo, float(e.retry_after or 0.0)))
                log.info("world sync: LRCLIB rate limited, waiting %.0f s", wait)
                report(_fraction(i, total), f"LRCLIB asked to slow down: waiting {_fmt_duration(wait)}")
                if not _wait(wait, stop):
                    break
        after = _lrclib_requests(lrclib)
        next_allowed = started + (after - before) / LRCLIB_COURTESY_RPS     # cache hits cost no wait
        before = after
        if err is not None:
            res["errors"] += 1
            consecutive += 1
            log.warning("world sync: lookup failed (%s): %s", err.kind, err)
            if stop():
                res["status"] = "cancelled"
                break
            if consecutive >= MAX_CONSECUTIVE_ERRORS:
                res["status"] = "error"
                res["message"] = f"Stopped: LRCLIB failed {consecutive} times in a row ({err})"[:300]
                break
        else:
            consecutive = 0
            text = rec.text if rec is not None and rec.has_lyrics else ""
            if text:
                try:
                    if index.upsert(_song(tm, text, rec.id, source, time.time())):
                        res["added"] += 1
                    else:
                        res["updated"] += 1
                    state.clear_miss(key)
                except ValueError as e:                     # invalid title: nothing to retry later
                    log.warning("world sync: skipped %r: %s", tm.title, e)
                    res["errors"] += 1
                except sqlite3.Error as e:                  # e.g. locked by another writer: retried next run
                    log.warning("world sync: could not store %r: %s", tm.title, type(e).__name__)
                    res["errors"] += 1
            else:
                res["no_lyrics"] += 1
                state.add_miss(key, tm.title, tm.artist, tm.deezer_id, "no_lyrics")
        res["processed"] = done = i + 1
        res["lrclib_requests"] = after - req0
        elapsed = _monotonic() - t_start
        eta = elapsed / done * (total - done) if done >= 3 else (total - done) / LRCLIB_COURTESY_RPS
        report(_fraction(done, total),
               f"Lyrics {done}/{total} · +{res['added']} · ETA {_fmt_duration(eta)}")
        if done % 10 == 0:
            state.update_run(run_id, total=total, done=done,
                             result={k: v for k, v in res.items() if k != "sources"})
    res["lyrics_s"] = round(_monotonic() - t_start, 2)
    res["lrclib_requests"] = _lrclib_requests(lrclib) - req0
