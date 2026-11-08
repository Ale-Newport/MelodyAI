"""Deezer public API client (https://developers.deezer.com/api): key-less metadata, popularity and charts.

Used for (a) title search at query time -- a sung chorus usually contains the song title --, (b) metadata
enrichment (album, duration, cover, preview, link, ``rank`` = popularity) and (c) the popular-songs sync
(charts, editorial playlists such as TikTok / viral hits, artist top tracks). Only text queries are sent.

Endpoints: ``/search?q=``, ``/search/playlist``, ``/chart/{genre}/tracks``, ``/chart/{genre}/artists``,
``/playlist/{id}/tracks``, ``/artist/{id}/top``, ``/track/{id}``.

Plain queries only. Deezer documents an advanced syntax (``q=artist:"a" track:"t"``) but a live check
(2026-11-07) showed it is no longer honoured: ``artist:"Queen" track:"Bohemian Rhapsody"`` returned 0 results,
``artist:"Queen"`` returned tracks *titled* "Queen", and ``track:"despacito"`` ranked titles containing the
word "Track" above Luis Fonsi. Plain ``q=despacito`` (or even a sung line containing the title) ranks the right
song first, so ``search_title`` sends the plain phrase and re-ranks by title agreement, and ``find`` sends
``"<primary artist> <title>"`` and verifies the results by title / artist similarity.
Lists are paginated with ``index`` + ``limit`` and a ``next`` URL. Deezer reports errors as HTTP 200 with
``{"error": {"type", "message", "code"}}``: code 4 (quota, 50 requests / 5 s) and 700 (service busy) are
retried once after ~5 s, code 800 (no data) means "not found", anything else raises ``WebError``.
Rate limiting, caching and the User-Agent are handled by ``WebClient``.

Note: ``preview_url`` values are signed CDN links that expire ~15 minutes after they are issued
(``hdnea=exp=<epoch>``): :func:`fresh_preview` drops expired ones, :func:`storable_preview` keeps signed ones
out of the index, and :meth:`DeezerClient.track` gets a fresh one by id when the user presses play.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from urllib.parse import parse_qs, urlsplit

from melodyai.lyrics.normalize import normalize_text
from melodyai.world.lrclib import artist_similarity, clean_title, core_title, primary_artist, title_similarity
from melodyai.world.web import WebClient, WebError, current_deadline

QUOTA_CODES = (4, 700)        # quota exceeded / service busy: wait and retry once
NOT_FOUND_CODE = 800          # "no data"


@dataclass
class TrackMeta:
    deezer_id: int
    title: str
    title_short: str
    artist: str
    artist_id: int | None
    album: str | None
    duration_s: float | None
    rank: int                   # Deezer popularity (0 = unknown); used only as a tie-breaker
    isrc: str | None
    cover_url: str | None
    preview_url: str | None
    link: str | None
    explicit: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> TrackMeta | None:
        if not isinstance(d, dict) or d.get("type", "track") != "track" or not d.get("id") or not d.get("title"):
            return None
        artist = d.get("artist") if isinstance(d.get("artist"), dict) else {}
        album = d.get("album") if isinstance(d.get("album"), dict) else {}
        try:
            deezer_id = int(d["id"])
        except (TypeError, ValueError):
            return None
        return cls(
            deezer_id=deezer_id, title=str(d["title"]), title_short=str(d.get("title_short") or d["title"]),
            artist=str(artist.get("name") or ""), artist_id=_int_or_none(artist.get("id")),
            album=album.get("title") or None, duration_s=float(d["duration"]) if d.get("duration") else None,
            rank=_int_or_none(d.get("rank")) or 0, isrc=d.get("isrc") or None,
            cover_url=album.get("cover_medium") or album.get("cover_big") or album.get("cover") or None,
            preview_url=d.get("preview") or None, link=d.get("link") or None,
            explicit=bool(d.get("explicit_lyrics")),
        )


_PREVIEW_EXP = re.compile(r"(?<![A-Za-z0-9_])exp=(\d{9,11})")


def preview_expires_at(url: str | None) -> int | None:
    """Expiry (epoch seconds) of a signed Deezer preview URL (``...mp3?hdnea=exp=1731000000~acl=...``), else None."""
    m = _PREVIEW_EXP.search(url or "")
    return int(m.group(1)) if m else None


def fresh_preview(url: str | None, now: float | None = None, margin_s: float = 60.0) -> str | None:
    """``url`` unless it is a signed preview that expires within ``margin_s`` (Deezer then answers 403)."""
    if not url:
        return None
    exp = preview_expires_at(url)
    if exp is not None and exp <= (time.time() if now is None else now) + margin_s:
        return None
    return url


def storable_preview(url: str | None) -> str | None:
    """A preview URL worth persisting: signed ones expire ~15 minutes after they are issued (keep ``deezer_id``)."""
    return url if url and preview_expires_at(url) is None else None


def _int_or_none(v) -> int | None:
    try:
        return int(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _phrase(text: str) -> str:
    """Text safe inside an advanced-search ``"..."`` literal."""
    return " ".join((text or "").replace('"', " ").split())


def _next_index(next_url: str | None, default: int) -> int:
    if next_url:
        try:
            return int(parse_qs(urlsplit(next_url).query)["index"][0])
        except (KeyError, IndexError, ValueError):
            pass
    return default


class DeezerClient:
    BASE = "https://api.deezer.com"
    HOST = "api.deezer.com"
    PAGE = 100                       # items per request when paginating
    MAX_PAGES = 50
    SEARCH_TTL_S = 3 * 86400
    CHART_TTL_S = 12 * 3600          # charts change daily
    PLAYLIST_TTL_S = 86400
    ARTIST_TTL_S = 7 * 86400
    PREVIEW_TTL_S = 300              # /track/{id}: its signed preview URL expires ~15 min after it is issued
    MIN_TITLE_SIM = 0.85
    MIN_ARTIST_SIM = 0.7
    TIE_MARGIN = 0.02                # matches this close are ties, broken by popularity

    def __init__(self, web: WebClient) -> None:
        self.web = web
        self.quota_wait_s = 5.0
        self._sleep = time.sleep

    # ------------------------------------------------------------------ transport
    def _get(self, path: str, params: dict | None = None, ttl_s: float = SEARCH_TTL_S,
             private: bool = False) -> dict | None:
        """One API call with Deezer's in-band error handling. None when Deezer has no such data.

        ``private``: the query holds the user's words (see :class:`melodyai.world.web.WebClient`)."""
        url = path if path.startswith("http") else f"{self.BASE}{path}"
        for attempt in range(2):
            data = self.web.get_json(url, params, ttl_s=ttl_s, private=private)
            if data is None:
                return None
            if not isinstance(data, dict):
                raise WebError("unexpected Deezer payload", kind="decode", host=self.HOST)
            err = data.get("error")
            if not err:
                return data
            err = err if isinstance(err, dict) else {"message": str(err)}
            code = _int_or_none(err.get("code")) or 0
            if code == NOT_FOUND_CODE:
                return None
            deadline = current_deadline()
            if code in QUOTA_CODES and attempt == 0 and (
                    deadline is None or time.monotonic() + self.quota_wait_s < deadline):
                self._sleep(self.quota_wait_s)
                continue
            raise WebError(f"Deezer error {code}: {err.get('message', '')}".strip(),
                           kind="rate_limited" if code in QUOTA_CODES else "api", host=self.HOST)
        raise AssertionError("unreachable")  # pragma: no cover

    def _paged(self, path: str, params: dict | None, limit: int, ttl_s: float, private: bool = False) -> list[dict]:
        """Follow ``index`` pagination until ``limit`` items, the end of the list or ``MAX_PAGES``."""
        out: list[dict] = []
        index = 0
        for _ in range(self.MAX_PAGES):
            if len(out) >= limit:
                break
            n = min(self.PAGE, limit - len(out))
            data = self._get(path, {**(params or {}), "index": index or None, "limit": n}, ttl_s, private)
            items = (data or {}).get("data") or []
            if not isinstance(items, list) or not items:
                break
            out.extend(x for x in items if isinstance(x, dict))
            nxt = _next_index(data.get("next"), index + len(items))
            total = _int_or_none(data.get("total"))
            if not data.get("next") or nxt <= index or (total is not None and nxt >= total):
                break
            index = nxt
        return out[:max(0, limit)]

    def _tracks(self, path: str, params: dict | None, limit: int, ttl_s: float,
                private: bool = False) -> list[TrackMeta]:
        return [t for t in (TrackMeta.from_json(d) for d in self._paged(path, params, limit, ttl_s, private)) if t]

    # ------------------------------------------------------------------ search
    def search_tracks(self, query: str, limit: int = 10, private: bool = False) -> list[TrackMeta]:
        """Free-text (or advanced-syntax) track search, Deezer relevance order (``private``: the user's words)."""
        query = " ".join((query or "").split())
        if not query or limit <= 0:
            return []
        return self._tracks("/search", {"q": query}, limit, self.SEARCH_TTL_S, private)

    def track(self, deezer_id: int) -> TrackMeta | None:
        """One track by id, with a *fresh* preview URL (Deezer signs preview URLs for about 15 minutes)."""
        data = self._get(f"/track/{int(deezer_id)}", None, self.PREVIEW_TTL_S)
        return TrackMeta.from_json(data) if isinstance(data, dict) else None

    def search_title(self, phrase: str, limit: int = 10, private: bool = False) -> list[TrackMeta]:
        """Tracks for a candidate title phrase (e.g. a sung chorus line), those whose title agrees first.

        Sends the plain phrase (see module docstring) and stably re-ranks Deezer's relevance order: titles
        equal to the phrase, then titles whose words are all in the phrase (or vice versa), then the rest.
        """
        phrase = _phrase(phrase)
        if not phrase:
            return []
        tracks = self.search_tracks(phrase, limit, private)
        words = normalize_text(phrase)
        ptoks = set(words.split())

        def agreement(tm: TrackMeta) -> int:
            best = 0
            for t in (tm.title, tm.title_short):
                core = core_title(t)
                ttoks = set(core.split())
                if core == words:
                    return 2
                if ttoks and (ttoks <= ptoks or ptoks <= ttoks):
                    best = 1
            return best

        return sorted(tracks, key=lambda tm: -agreement(tm))

    def find(self, title: str, artist: str, limit: int = 5) -> TrackMeta | None:
        """Deezer track for a known title + artist: ``q="<primary artist> <title>"``, then the title alone with a
        wider page; results are verified by title / artist similarity (None if nothing agrees)."""
        t, a = _phrase(clean_title(title)), _phrase(primary_artist(artist))
        if not t or not a:
            return None
        best = self.best_match(self.search_tracks(f"{a} {t}", limit), title, artist)
        if best is None:
            best = self.best_match(self.search_tracks(t, max(limit, 25)), title, artist)
        return best

    def best_match(self, tracks: list[TrackMeta], title: str, artist: str) -> TrackMeta | None:
        """Best title / artist agreement; near-ties (``TIE_MARGIN``) go to the more popular track."""
        scored = []
        for tm in tracks:
            ts = max(title_similarity(title, tm.title), title_similarity(title, tm.title_short))
            as_ = artist_similarity(artist, tm.artist)
            if ts >= self.MIN_TITLE_SIM and as_ >= self.MIN_ARTIST_SIM:
                scored.append((0.6 * ts + 0.4 * as_, tm))
        if not scored:
            return None
        top = max(s for s, _ in scored)
        return max((tm for s, tm in scored if s >= top - self.TIE_MARGIN), key=lambda tm: tm.rank)

    # ------------------------------------------------------------------ charts, playlists, artists
    def chart_tracks(self, genre_id: int = 0, limit: int = 100) -> list[TrackMeta]:
        """Chart for ``genre_id`` (0 = all genres; Deezer localises it by the caller's country)."""
        return self._tracks(f"/chart/{int(genre_id)}/tracks", None, limit, self.CHART_TTL_S)

    def chart_artists(self, genre_id: int = 0, limit: int = 100) -> list[dict]:
        """``[{id, name, position}]`` most popular artists."""
        out = []
        for d in self._paged(f"/chart/{int(genre_id)}/artists", None, limit, self.CHART_TTL_S):
            aid = _int_or_none(d.get("id"))
            if aid and d.get("name"):
                out.append({"id": aid, "name": str(d["name"]), "position": _int_or_none(d.get("position"))})
        return out

    def search_playlists(self, query: str, limit: int = 25) -> list[dict]:
        """``[{id, title, nb_tracks, user, user_id, link}]`` public playlists matching ``query``."""
        query = " ".join((query or "").split())
        if not query or limit <= 0:
            return []
        out = []
        for d in self._paged("/search/playlist", {"q": query}, limit, self.PLAYLIST_TTL_S):
            pid = _int_or_none(d.get("id"))
            if not pid:
                continue
            user = d.get("user") if isinstance(d.get("user"), dict) else {}
            out.append({"id": pid, "title": str(d.get("title") or ""), "nb_tracks": _int_or_none(d.get("nb_tracks")) or 0,
                        "user": str(user.get("name") or ""), "user_id": _int_or_none(user.get("id")),
                        "link": d.get("link") or None})
        return out

    def playlist_tracks(self, playlist_id: int, limit: int = 500) -> list[TrackMeta]:
        """Tracks of a playlist, following pagination up to ``limit``."""
        return self._tracks(f"/playlist/{int(playlist_id)}/tracks", None, limit, self.PLAYLIST_TTL_S)

    def artist_top(self, artist_id: int, limit: int = 50) -> list[TrackMeta]:
        """An artist's most popular tracks."""
        return self._tracks(f"/artist/{int(artist_id)}/top", None, limit, self.ARTIST_TTL_S)
