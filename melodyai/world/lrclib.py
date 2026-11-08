"""LRCLIB client (https://lrclib.net/docs): free, key-less lyrics database (~3M+ records, plain + synced LRC).

Endpoints used (v1, stable):
  GET /api/get?track_name&artist_name[&album_name][&duration]   best record for a track signature, 404 if none.
      LRCLIB only returns a record whose duration is within +/-2 s of ``duration`` when it is given.
      (The docs no longer offer a cached-only variant, so ``/api/get`` is the signature lookup.)
  GET /api/search?q | track_name[&artist_name][&album_name]      up to 20 full records, no pagination.
      ``q`` searches title / artist / album only -- NOT lyric text.

Only titles and artist names leave the machine (never audio). Responses are cached by ``WebClient``
without the bulky ``lyricsfile`` YAML member. ``find`` is the robust lookup used by the open-world
search and the popular-songs sync: exact signature first, then a title+artist search verified by
normalised title / artist similarity and duration.

The title / artist similarity helpers below are shared with ``deezer.py``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from rapidfuzz import fuzz

from melodyai.lyrics.normalize import normalize_text
from melodyai.world.web import WebClient, WebError

# ---------------------------------------------------------------------------- title / artist similarity
_BRACKETS = re.compile(r"\[[^\]]*\]|\{[^}]*\}")
_PARENS = re.compile(r"\([^)]*\)")
_VERSION = (r"remaster(ed)?|live|en vivo|ao vivo|en directo|version|versi[oó]n|edit|mix|remix|acoustic|ac[uú]stic[oa]|"
            r"mono|stereo|radio|single|explicit|clean|deluxe|bonus|demo|original|extended|instrumental|"
            r"sped up|slowed|reverb|from|de la pel[ií]cula|soundtrack|anniversary|mtv|unplugged|session|\d{4}")
_DASH_SUFFIX = re.compile(r"\s[-–—]\s.*\b(" + _VERSION + r")\b.*$", flags=re.IGNORECASE)
_FEAT = re.compile(r"\s(feat\.|ft\.|featuring)\s.*$", flags=re.IGNORECASE)
_ARTIST_SPLIT = re.compile(r"\s*(?:,|;|&|\s/\s|\+|\bfeat(?:\.|\b)|\bft(?:\.|\b)|\bfeaturing\b|\bwith\b|\bcon\b|\bx\b|"
                           r"\by\b|\band\b|\be\b)\s*", flags=re.IGNORECASE)
_PRIMARY_SPLIT = re.compile(r"\s*(?:,|;|\s/\s|\s(?:feat\.?|ft\.?|featuring|with|x|con)\s)\s*", flags=re.IGNORECASE)


def clean_title(title: str) -> str:
    """Title without bracketed parts, ``feat.`` credits and ``- Remastered 2011``-style suffixes (case and
    accents kept: suitable as a search query). Falls back to the input when nothing would remain."""
    t = unicodedata.normalize("NFKC", title or "").strip()
    stripped = _FEAT.sub("", _DASH_SUFFIX.sub("", _PARENS.sub(" ", _BRACKETS.sub(" ", t))))
    stripped = re.sub(r"\s+", " ", stripped).strip(" -–—")
    return stripped if normalize_text(stripped) else t


def core_title(title: str) -> str:
    """Normalised (casefolded, accent-folded, punctuation-free) ``clean_title``."""
    return normalize_text(clean_title(title))


def primary_artist(artist: str) -> str:
    """First credited artist (``"A feat. B"``, ``"A, B"``, ``"A x B"`` -> ``"A"``); duo names with ``&`` are kept.
    Case and accents are kept: suitable as a search query."""
    a = " ".join(unicodedata.normalize("NFKC", artist or "").split())
    return _PRIMARY_SPLIT.split(a, maxsplit=1)[0].strip() or a


def _artist_names(artist: str) -> set[str]:
    full = normalize_text(artist)
    names = {full}
    names.update(normalize_text(p) for p in _ARTIST_SPLIT.split(unicodedata.normalize("NFKC", artist or "")))
    names = {re.sub(r"^the ", "", n) for n in names if n}
    return names


def title_similarity(a: str, b: str) -> float:
    """0-1 similarity of two song titles, tolerant to versions / features / punctuation / accents."""
    ca, cb = core_title(a), core_title(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    return max(fuzz.ratio(ca, cb), fuzz.ratio(normalize_text(a), normalize_text(b))) / 100.0


def artist_similarity(a: str, b: str) -> float:
    """0-1 similarity of two artist credits; any credited artist matching counts (``"A feat. B"`` vs ``"B"``)."""
    na, nb = _artist_names(a), _artist_names(b)
    if not na or not nb:
        return 0.0
    if na & nb:
        return 1.0
    return max(fuzz.ratio(x, y) for x in na for y in nb) / 100.0


# ---------------------------------------------------------------------------- records
_LRC_TIME = re.compile(r"\[\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?\]|<\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?>")
_LRC_META = re.compile(r"^\s*\[[a-zA-Z#]+:[^\]]*\]\s*$")


def strip_synced(lrc: str) -> str:
    """LRC -> plain text: drop ``[mm:ss.xx]`` line tags, ``<mm:ss.xx>`` word tags, ``[ar:..]`` metadata, blanks."""
    lines = []
    for raw in (lrc or "").splitlines():
        if _LRC_META.match(raw):
            continue
        line = _LRC_TIME.sub("", raw).strip()
        if line:
            lines.append(re.sub(r"\s+", " ", line))
    return "\n".join(lines)


@dataclass
class LyricsRecord:
    id: int
    track: str
    artist: str
    album: str | None
    duration_s: float | None
    instrumental: bool
    plain: str | None
    synced: str | None

    @property
    def text(self) -> str:
        """Plain lyrics, else synced lyrics with the time tags stripped, else ``""``."""
        if self.plain and self.plain.strip():
            return self.plain.strip()
        if self.synced and self.synced.strip():
            return strip_synced(self.synced)
        return ""

    @property
    def has_lyrics(self) -> bool:
        return not self.instrumental and bool(self.text)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> LyricsRecord | None:
        if not isinstance(d, dict) or d.get("id") is None:
            return None
        try:
            rid = int(d["id"])
        except (TypeError, ValueError):
            return None
        dur = d.get("duration")
        try:
            duration = float(dur) if dur not in (None, "") else None
        except (TypeError, ValueError):
            duration = None
        return cls(id=rid, track=str(d.get("trackName") or d.get("name") or ""), artist=str(d.get("artistName") or ""),
                   album=d.get("albumName") or None, duration_s=duration, instrumental=bool(d.get("instrumental")),
                   plain=d.get("plainLyrics") or None, synced=d.get("syncedLyrics") or None)


# ---------------------------------------------------------------------------- client
class LrclibClient:
    BASE = "https://lrclib.net"
    GET_TTL_S = 30 * 86400          # lyrics records are stable
    SEARCH_TTL_S = 7 * 86400
    NEGATIVE_TTL_S = 86400          # LRCLIB may fetch missing tracks in the background: retry tomorrow
    DROP_FIELDS = ("lyricsfile",)   # raw Lyricsfile YAML duplicates plain/synced: never stored
    MIN_TITLE_SIM = 0.85
    MIN_ARTIST_SIM = 0.7
    DURATION_TOLERANCE_S = 3.0      # preferred: same recording
    DURATION_FALLBACK_S = 30.0      # accepted only when nothing within tolerance and title+artist agree strongly
    FALLBACK_MIN_SIM = 0.9

    def __init__(self, web: WebClient) -> None:
        self.web = web

    def search(self, *, q: str | None = None, track_name: str | None = None, artist_name: str | None = None,
               album_name: str | None = None, private: bool = False) -> list[LyricsRecord]:
        """``/api/search``: up to 20 records. ``q`` (title/artist/album keywords) takes precedence.

        ``private``: the query holds the user's words (see :class:`melodyai.world.web.WebClient`)."""
        q, track_name = (q or "").strip() or None, (track_name or "").strip() or None
        if not q and not track_name:
            return []
        params = {"q": q} if q else {"track_name": track_name, "artist_name": (artist_name or "").strip() or None,
                                     "album_name": (album_name or "").strip() or None}
        data = self.web.get_json(f"{self.BASE}/api/search", params, ttl_s=self.SEARCH_TTL_S,
                                 negative_ttl_s=self.NEGATIVE_TTL_S, drop_fields=self.DROP_FIELDS, private=private)
        if not isinstance(data, list):
            return []
        return [r for r in (LyricsRecord.from_json(d) for d in data) if r is not None]

    def get(self, track_name: str, artist_name: str, album_name: str | None = None,
            duration_s: float | None = None) -> LyricsRecord | None:
        """``/api/get`` exact signature lookup; None when LRCLIB has no matching record (404)."""
        track_name, artist_name = (track_name or "").strip(), (artist_name or "").strip()
        if not track_name or not artist_name:
            return None
        params = {"track_name": track_name, "artist_name": artist_name, "album_name": (album_name or "").strip() or None}
        if duration_s is not None and 1 <= round(duration_s) <= 3600:   # API rejects other values (400)
            params["duration"] = int(round(duration_s))
        data = self.web.get_json(f"{self.BASE}/api/get", params, ttl_s=self.GET_TTL_S,
                                 negative_ttl_s=self.NEGATIVE_TTL_S, drop_fields=self.DROP_FIELDS)
        return LyricsRecord.from_json(data) if isinstance(data, dict) else None

    def find(self, title: str, artist: str, duration_s: float | None = None,
             album: str | None = None) -> LyricsRecord | None:
        """Robust lookup of the lyrics of ``title`` by ``artist``; None if LRCLIB has no usable lyrics.

        1. ``/api/get`` with the full signature (accepted if non-instrumental, with lyrics, and the title /
           artist really agree);
        2. else ``/api/search`` by clean title + primary artist and pick the best non-instrumental record with lyrics
           by title / artist similarity, preferring durations within +/-3 s.
        Raises ``WebError`` on network / rate-limit failures (HTTP 400 validation errors are skipped).
        """
        title, artist = (title or "").strip(), (artist or "").strip()
        if not title or not artist:
            return None
        try:
            rec = self.get(title, artist, album, duration_s)
        except WebError as e:
            if e.status != 400:
                raise
            rec = None
        if rec is not None and rec.has_lyrics and self._score(rec, title, artist, duration_s) is not None:
            return rec
        records = self.search(track_name=clean_title(title), artist_name=primary_artist(artist))
        return self.best_record(records, title, artist, duration_s)

    def best_record(self, records: list[LyricsRecord], title: str, artist: str,
                    duration_s: float | None = None) -> LyricsRecord | None:
        """Best usable record for ``title`` / ``artist`` among ``records`` (None if none qualifies)."""
        scored = []
        for r in records:
            if not r.has_lyrics:
                continue
            s = self._score(r, title, artist, duration_s)
            if s is not None:
                scored.append((s, r))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return scored[0][1]

    def _score(self, r: LyricsRecord, title: str, artist: str, duration_s: float | None) -> float | None:
        ts, as_ = title_similarity(title, r.track), artist_similarity(artist, r.artist)
        if ts < self.MIN_TITLE_SIM or as_ < self.MIN_ARTIST_SIM:
            return None
        score = 0.6 * ts + 0.4 * as_
        if duration_s is not None and r.duration_s is not None:
            diff = abs(r.duration_s - duration_s)
            if diff <= self.DURATION_TOLERANCE_S:
                score += 1.0          # any in-tolerance record beats every out-of-tolerance one
            elif diff > self.DURATION_FALLBACK_S or min(ts, as_) < self.FALLBACK_MIN_SIM:
                return None
            else:
                score -= diff / 1000.0
        if r.synced:
            score += 0.01             # synced records are usually the curated ones
        return score
