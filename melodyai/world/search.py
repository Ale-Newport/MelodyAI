"""Open-world song search: a (sung, misheard, partial) transcript -> real songs, verified by their lyrics.

Pipeline of :meth:`WorldSearch.search` (every step bounded by one deadline, no step can raise):

1. *Propose* -- the sources (:mod:`melodyai.world.sources`) run concurrently: the local world lyrics index
   (offline) first, alone for a moment (``HEAD_START_S``), then, when online search is enabled and the index's
   answer does not settle the search on its own (:meth:`WorldSearch._index_answer`; a settled answer still checks
   with one cacheable title search that it is not a popular cover of an original the index lacks), Deezer search
   with windows of the sung words and title phrases (plus keyed sources once configured): with the whole deadline,
   or, when the index's answer is probable, ``GRACE_S`` to answer and ``PROBABLE_FETCH_S`` for the lyrics of
   their proposals. LRCLIB title search is a fallback that runs only if the other candidates, once verified,
   matched nothing, and the transcript repeats a phrase. Candidates are merged by
   :func:`melodyai.world.text.song_key`.
2. *Resolve lyrics* -- candidates without lyrics take them from the local index if it has the song, else from
   ``LrclibClient.find`` (only the ``MAX_FETCH`` most promising: title sung in the transcript, proposed by
   several sources, popular; none after ``MIN_FETCH`` once one aligns >= ``FETCH_ENOUGH``). Lookups start as
   soon as a source has answered, not when the slowest source is done, so a slow source does not eat the time
   Deezer's proposals need for their lyrics.
3. *Verify* -- :meth:`WorldLyricsIndex.score_lyrics` aligns the transcript against each candidate's real lyrics
   (fuzzy, phonetic, order-aware; the same scoring the index search uses), so a source's guess only counts with
   lyrics evidence. Candidates under ``min_score`` are dropped.
4. *Enrich + group* -- the best few without Deezer metadata get it (cover, link, rank); a best match that looks
   like an unpopular cover / karaoke also gets the popular recording with that title, which joins the list only
   if its own lyrics align about as well (``ORIGINAL_SCORE_WINDOW``). Versions of one song collapse into one
   entry only with evidence that they *are* one song: most distinct lines of the shorter lyrics are found in the
   other record (``SAME_SONG_OVERLAP``), or the titles are the same and a good part of the lines are shared
   (``SAME_TITLE_OVERLAP``: covers by other artists), or the same title is marked as a version (karaoke, cover,
   live...) and the same lines matched. A shared title or one shared line is not enough (two songs called
   "Farol"). Records that share the lyrics are shown with the song's best alignment (a cover's transcription may
   match the sung words better than the original's) by the most popular one that aligns about as well
   (``COVER_SCORE_WINDOW``) or is clearly more popular than the best-aligned one (``POPULAR_FACTOR``); records
   only marked as versions collapse when they score about the same, and the more popular one wins unless it
   aligns clearly worse (``POPULARITY_OVERRIDE``), keeping its own score.
5. *Rank* -- alignment score + a small popularity tie-break ``0.03 * min(1, log10(1 + rank) / 6)``. A best
   match that is partial, rests on few matched informative words and does not lead clearly is capped first,
   staying first by a hair (:meth:`WorldSearch._evidence_guard`): short generic lines align partly with many
   songs.
6. *Learn* -- verified songs (score >= ``LEARN_MIN_SCORE``) are added to the local index (``source="lookup"``),
   so the next search for them is instant and offline.

Every request of a search (in any worker thread) is bounded by its deadline
(:func:`melodyai.world.web.request_deadline`): once it has passed nothing more is sent, so a search that
answered on time does not leave abandoned requests behind. Deezer preview URLs are signed for ~15 minutes:
expired ones are never returned (``deezer_id`` lets the API resolve a fresh one when the user presses play).
"""

from __future__ import annotations

import contextvars
import logging
import math
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from urllib.parse import quote, quote_plus

from rapidfuzz import fuzz, process

from melodyai.lyrics.align import MATCH_SIM, AlignResult, similarity_matrix
from melodyai.lyrics.normalize import informative_tokens, normalize_text, tokenize
from melodyai.world.deezer import DeezerClient, TrackMeta, fresh_preview, storable_preview
from melodyai.world.index import WorldLyricsIndex, WorldSong, lyric_lines, world_song_id
from melodyai.world.lrclib import (
    LrclibClient,
    artist_similarity,
    core_title,
    primary_artist,
    title_similarity,
)
from melodyai.world.sources import Candidate, Source, default_sources, title_in_query
from melodyai.world.text import FILLERS, guess_language, normalize_title, song_key
from melodyai.world.web import WebClient, request_deadline

log = logging.getLogger(__name__)

POPULARITY_WEIGHT = 0.03
LEARN_MIN_SCORE = 0.5
COVER_SCORE_WINDOW = 0.15     # versions of one song: lyric transcriptions differ a little between records
SAME_SONG_OVERLAP = 0.7       # share of the shorter lyrics' distinct lines also found in the other record
SAME_TITLE_OVERLAP = 0.3      # ... enough when both records have the same title (and >= MIN_SAME_TITLE_LINES)
MIN_SAME_TITLE_LINES = 3
SAME_LINE_SIM = 85.0          # fuzz.ratio of two transcriptions of one lyric line
MIN_SHARED_LINES = 2          # lyrics with fewer distinct lines prove nothing about being the same song
MAX_OVERLAP_LINES = 200       # distinct lines compared per record
COVER_TEXT_SIM = 60.0         # same title marked as a version (karaoke, live...): the same lines matched
SAME_SUNG_MIN = 0.95          # same title, both records hold the sung lines almost exactly ...
SAME_SUNG_SIM = 90.0          # ... and matched the same lines: a cover / translation sung with the original words
MEDLEY_MIN = 0.85             # a medley / mashup that matched the same sung lines about as well is this song
POPULARITY_OVERRIDE = 0.03    # the more popular version wins only if its own alignment is at most this much worse
POPULAR_FACTOR = 2            # "clearly more popular": a Deezer rank more than this many times higher
ORIGINAL_MIN_RANK = 300_000   # a best match less popular than this may be a cover: look for the original
ORIGINAL_SCORE_WINDOW = 0.05  # ... which joins the list only if it aligns about as well as that match
FALLBACK_BELOW = 0.5          # fallback sources (``fallback = True``) run only if no candidate aligned this well
FETCH_ENOUGH = 0.9            # a candidate aligned this well ends the lyric lookups (after MIN_FETCH of them)
PARTIAL_SCORE = 0.9           # an alignment below this is partial ...
LOW_EVIDENCE_WORDS = 2        # ... and one resting on at most this many matched informative words is weak:
LOW_EVIDENCE_LEAD = 0.2       # as the best match it must lead the next song by this much,
LOW_EVIDENCE_CAP = 0.55       # else its score is capped at max(this, the next song's score)
# Decorations that mark a recording as a version of another one (in the title's removed decoration or the artist).
_VERSION_MARK = re.compile(r"\b(?:karaoke|covers?|tribute|in the style of|originally performed|made famous|live|"
                           r"en vivo|ao vivo|en directo|acoustic|acustic[oa]|remix|sped up|slowed|instrumental|"
                           r"version)\b")
_ARTIST_MARK = re.compile(r"\b(?:karaoke|tribute|covers?|in the style of|originally performed|made famous)\b")


def popularity_bonus(rank: int | None) -> float:
    return POPULARITY_WEIGHT * min(1.0, math.log10(1 + max(0, int(rank or 0))) / 6.0)


def distinct_lines(lyrics: str | None) -> list[str]:
    """Normalised distinct lyric lines of >= 2 words (the first ``MAX_OVERLAP_LINES``)."""
    lines = (normalize_text(ln) for ln in lyric_lines(lyrics or ""))
    return list(dict.fromkeys(n for n in lines if len(n.split()) >= 2))[:MAX_OVERLAP_LINES]


def shared_lines(a: list[str], b: list[str]) -> tuple[int, int]:
    """``(hits, n)``: how many of the ``n`` distinct lines of the shorter list have a near-identical line in the
    other one (``n = 0`` when it has fewer than ``MIN_SHARED_LINES``: too short to prove anything)."""
    if len(a) > len(b):
        a, b = b, a
    if len(a) < MIN_SHARED_LINES:
        return 0, 0
    exact = set(b)
    rest = [x for x in a if x not in exact]
    hits = len(a) - len(rest)
    if rest and hits < SAME_SONG_OVERLAP * len(a):
        sims = process.cdist(rest, b, scorer=fuzz.ratio, score_cutoff=SAME_LINE_SIM, workers=1)
        hits += int((sims.max(axis=1) > 0).sum())
    return hits, len(a)


def line_overlap(a: list[str], b: list[str]) -> float:
    """Share of the distinct lines of the shorter list that have a near-identical line in the other one."""
    hits, n = shared_lines(a, b)
    return hits / n if n else 0.0


def lyrics_overlap(a: str | None, b: str | None) -> float:
    """:func:`line_overlap` of two lyrics texts: ~1.0 for two records of one song, low for different songs."""
    return line_overlap(distinct_lines(a), distinct_lines(b))


# words that only name a version when they follow a title: "Hotel California (Spanish Mix)", "X - 2013 Remaster"
_VERSION_TOKENS = frozenset(
    "mix remix remaster remastered version edit radio live en ao vivo directo spanish english espanol spanglish "
    "portuguese portugues french italian acoustic acustico acustica mono stereo extended original single deluxe "
    "demo instrumental karaoke cover sped up slowed reverb".split())


_DECORATION = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]|\s+[-–—]\s+.*$")
_PART_SPLIT = re.compile(r"\s*(?:/|\||\s\+\s|•)\s*")
_MEDLEY_MARK = re.compile(r"\b(?:mash ?up|medley|mega ?mix|enganchados?|popurr[ií]|potpourri|halftime|mix \d+)\b")


def base_title(title: str) -> str:
    """The title without anything in brackets or after " - ": "La Flaca (Acousique)" -> "la flaca"."""
    return normalize_title(_DECORATION.sub("", title or "")) or normalize_title(title)


def medley_marked(title: str) -> bool:
    """A record of several songs: "A / B / C", "X Mashup", "Megamix", "Enganchados", "Halftime Show", "Mix 4"."""
    return bool(_MEDLEY_MARK.search(normalize_text(title))) or len(_PART_SPLIT.split(title or "")) >= 3


def title_version_of(a: str, b: str) -> bool:
    """Titles of one song in two records (their normalised titles differ): one is the other followed only by
    version words ("hotel california spanish mix"), the same once decorations are removed ("La Flaca (Acousique)"),
    or one of the songs a medley's title lists ("A / B / C")."""
    na, nb = normalize_title(a), normalize_title(b)
    short, long_ = sorted((na, nb), key=len)
    if not short or short == long_:
        return False
    if long_.startswith(short + " ") and all(w in _VERSION_TOKENS or w.isdigit()
                                             for w in long_[len(short) + 1:].split()):
        return True
    if base_title(a) and base_title(a) == base_title(b):
        return True
    raw_long = a if na == long_ else b
    return short in {normalize_title(p) for p in _PART_SPLIT.split(raw_long) if p.strip()} - {long_}


def version_marked(title: str, artist: str) -> bool:
    """The recording says it is a version: "(Karaoke Version)", "- En Vivo", "(Acoustic Cover)", a tribute artist."""
    removed = set(_VERSION_MARK.findall(normalize_text(title))) - set(_VERSION_MARK.findall(normalize_title(title)))
    return bool(removed) or bool(_ARTIST_MARK.search(normalize_text(artist)))


def _in_context(ex: ThreadPoolExecutor, fn: Callable, *args, until: float | None = None) -> Future:
    """Submit ``fn`` with the caller's context variables (the request deadline), or with the request deadline
    moved to ``until`` (a ``time.perf_counter()`` instant) when given: work that will be abandoned at that
    point must not keep sending requests after it."""
    if until is None:
        return ex.submit(contextvars.copy_context().run, fn, *args)
    with request_deadline(time.monotonic() + (until - time.perf_counter())):
        return ex.submit(contextvars.copy_context().run, fn, *args)


def song_links(title: str, artist: str, deezer_link: str | None = None) -> dict[str, str]:
    q = f"{title} {artist}".strip()
    links = {"spotify": "https://open.spotify.com/search/" + quote(q, safe=""),
             "youtube": "https://www.youtube.com/results?search_query=" + quote_plus(q)}
    links["deezer"] = deezer_link if deezer_link else "https://www.deezer.com/search/" + quote(q, safe="")
    return links


@dataclass
class WorldCandidate:
    song_id: str                 # "world:" + WorldSong.id
    title: str
    artist: str
    album: str | None
    language: str | None
    duration_s: float | None
    cover_url: str | None
    preview_url: str | None
    rank: int
    links: dict[str, str]
    score: float
    coverage: float
    char_similarity: float
    matched_text: str
    line_range: tuple[int, int]
    sources: list[str]
    alignment_score: float = 0.0
    deezer_id: int | None = None  # resolves a fresh preview (GET /api/world/preview/{deezer_id})

    def to_dict(self) -> dict:
        return {"song_id": self.song_id, "title": self.title, "artist": self.artist, "album": self.album,
                "language": self.language, "duration_s": self.duration_s, "cover_url": self.cover_url,
                "preview_url": self.preview_url, "deezer_id": self.deezer_id, "rank": self.rank,
                "links": dict(self.links),
                "score": round(self.score, 4), "alignment_score": round(self.alignment_score, 4),
                "coverage": round(self.coverage, 4), "char_similarity": round(self.char_similarity, 4),
                "matched_text": self.matched_text, "line_range": list(self.line_range),
                "sources": list(self.sources)}


OFFLINE_MESSAGES = {"setting_off": "online search off", "server_offline": "server offline"}


@dataclass
class WorldResult:
    status: str                  # "ok" | "no_candidates" | "no_query" | "disabled" | "error"
    query_text: str
    online: bool
    candidates: list[WorldCandidate] = field(default_factory=list)
    sources: dict[str, dict] = field(default_factory=dict)
    elapsed_s: float = 0.0
    message: str = ""
    offline_reason: str = ""     # why online sources were not asked: "" | "setting_off" | "server_offline"

    def to_dict(self) -> dict:
        return {"status": self.status, "query_text": self.query_text, "online": self.online,
                "offline_reason": self.offline_reason,
                "elapsed_s": round(self.elapsed_s, 3), "message": self.message,
                "sources": {k: dict(v) for k, v in self.sources.items()},
                "candidates": [c.to_dict() for c in self.candidates]}


@dataclass
class _Entry:
    """All proposals of one song identity (song_key), with the evidence gathered so far."""

    key: str
    cands: list[Candidate] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    lyrics: str | None = None
    lrclib_id: int | None = None
    world_song: WorldSong | None = None
    meta: TrackMeta | None = None
    align: AlignResult | None = None
    index_checked: bool = False
    _lines: list[str] | None = None

    @property
    def first(self) -> Candidate:
        return self.cands[0]

    @property
    def rank(self) -> int:
        return max(self.meta.rank if self.meta else 0, self.world_song.rank if self.world_song else 0)

    def title_artist(self) -> tuple[str, str]:
        if self.meta is not None:
            return self.meta.title, self.meta.artist
        if self.world_song is not None:
            return self.world_song.title, self.world_song.artist
        return self.first.title, self.first.artist

    def full_lyrics(self) -> str:
        return self.lyrics or (self.world_song.lyrics if self.world_song is not None else "") or ""

    def lines(self) -> list[str]:
        if self._lines is None:
            self._lines = distinct_lines(self.full_lyrics())
        return self._lines

    def fetch_priority(self, tokens: list[str]) -> tuple:
        """Most promising first: proposed often / high up (e.g. by several sung lines), title sung, popular."""
        votes = sum(max(0.0, c.prior) for c in self.cands)
        sung = max(title_in_query(c.title, tokens) for c in self.cands)
        return (-(votes + 0.5 * sung + 0.25 * (len(set(self.sources)) - 1)), -self.rank, self.key)


class WorldSearch:
    MAX_FETCH = 7            # LRCLIB lyric lookups per query (candidates proposed without lyrics) ...
    MIN_FETCH = 3            # ... but none after this many once a candidate aligns >= FETCH_ENOUGH
    FETCH_WORKERS = 3
    ENRICH_TOP = 4           # Deezer metadata lookups per query
    ORIGINAL_TOP = 2         # best matches checked for a more popular original recording
    PROPOSE_SHARE = 0.55     # share of the deadline for the proposal phase
    FETCH_SHARE = 0.85       # ... and up to which point lyrics may still be fetched
    HEAD_START_S = 0.5       # the local index answers alone first (at most this long, <= 25 % of the deadline)
    EXIT_SCORE = 0.9         # a near-perfect index match settles the search (see _index_verdict) with this lead
    EXIT_MARGIN = 0.15       # over every other song ...
    EXIT_MIN_SCORE = 0.7     # ... down to this alignment, which needs a lead of EXIT_MIN_SCORE_MARGIN
    EXIT_MIN_SCORE_MARGIN = 0.3
    EXIT_MIN_WORDS = 3       # distinct informative query words of a settling query
    PROBABLE_MIN_SCORE = 0.5  # a partial index match leading by EXIT_MIN_SCORE_MARGIN is probable
    GRACE_S = 1.0            # time left to the online sources when the index answer is only probable
    PROBABLE_FETCH_S = 2.0   # ... and to the lyric lookups of their proposals, from the first one
    PROBABLE_MAX_FETCH = 5   # ... at most this many (a near miss of the index does not end them)
    FALLBACK_MIN_S = 0.5     # a fallback source starts only this long before the proposal phase ends

    def __init__(self, index: WorldLyricsIndex, web: WebClient | None, sources: list[Source] | None = None,
                 *, learn: bool = True, deadline_s: float = 9.0, min_score: float = 0.25) -> None:
        self.index, self.web = index, web
        self.sources: list[Source] = list(sources) if sources is not None else default_sources(index, web)
        self.lrclib = LrclibClient(web) if web is not None else None
        self.deezer = DeezerClient(web) if web is not None else None
        self.learn, self.deadline_s, self.min_score = learn, deadline_s, min_score
        self._learn_lock = threading.Lock()

    # ------------------------------------------------------------------ description
    def describe_sources(self) -> list[dict]:
        out = []
        for s in self.sources:
            try:
                ok = bool(s.available())
            except Exception:  # noqa: BLE001
                ok = False
            out.append({"name": s.name, "label": s.label, "online": s.online, "requires_key": s.requires_key,
                        "available": ok})
        return out

    @property
    def online_capable(self) -> bool:
        return self.web is not None and not self.web.offline

    # ------------------------------------------------------------------ search
    def search(self, text: str, language: str | None = None, top_k: int = 5, online: bool = True) -> WorldResult:
        t0 = time.perf_counter()
        text = (text or "").strip()
        requested, online = bool(online), bool(online) and self.online_capable
        why = "" if online else "setting_off" if not requested else "server_offline"
        tokens = [t for t in tokenize(text) if t not in FILLERS]
        if len(tokens) < 3 or not informative_tokens(tokens):
            return WorldResult("no_query", text, online, elapsed_s=time.perf_counter() - t0,
                               message="not enough words to search", offline_reason=why)
        deadline = t0 + self.deadline_s
        status: dict[str, dict] = {}
        ex = ThreadPoolExecutor(max_workers=8, thread_name_prefix="world-search")
        try:
            # every request of this search, in any worker thread, stops at the deadline
            with request_deadline(time.monotonic() + max(0.0, deadline - time.perf_counter())):
                memo: dict[str, AlignResult | None] = {}
                entries, verdict, end = self._gather(ex, text, tokens, language, online, status, t0, why, memo)
                scored = self._verify(text, entries, memo)
                if online and self.deezer is not None:
                    end = min(deadline, end)
                    self._enrich(ex, scored, end)
                    if verdict != "settled":     # a settled answer is popular and passed _namesake
                        scored = self._add_originals(ex, text, scored, end)
            final = self._rank(self._evidence_guard(tokens, self._group_covers(scored)))
        except Exception as e:  # noqa: BLE001 - the search must never raise into the recogniser
            log.exception("world search failed")
            return WorldResult("error", text, online, sources=status, elapsed_s=time.perf_counter() - t0,
                               message=f"{type(e).__name__}: {e}"[:300], offline_reason=why)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        if self.learn:
            self._learn(final)
        cands = [self._candidate(e, s) for e, s in final[:max(1, top_k)]]
        return WorldResult("ok" if cands else "no_candidates", text, online, cands, status,
                           time.perf_counter() - t0, offline_reason=why)

    # ------------------------------------------------------------------ steps
    def _gather(self, ex: ThreadPoolExecutor, text: str, tokens: list[str], language: str | None, online: bool,
                status: dict[str, dict], t0: float, offline_reason: str = "setting_off", memo: dict | None = None
                ) -> tuple[dict[str, _Entry], str | None, float]:
        """Run the sources and, meanwhile, fetch the lyrics of the most promising candidates proposed without them.

        Returns the candidates, the verdict on the local index's answer (:meth:`_index_verdict`) and the time
        (``time.perf_counter()``) by which the remaining steps (metadata, originals) must be done: ``GRACE_S``
        from now for a settled or probable answer, else the deadline.

        Offline sources (the local index) start first and run alone for up to ``HEAD_START_S``. When their answer
        settles the search, the online sources are not asked (status "skipped", "local index match"): a song the
        index knows costs no wait and at most one cacheable request, :meth:`_namesake` (the settled recording may
        be a popular cover of an original the index lacks; then the answer is only probable). When it is only
        probable, the online sources get ``GRACE_S`` to answer (their requests stop then), and the lyric lookups
        of their proposals ``PROBABLE_FETCH_S`` from the first proposal that needs one (at most
        ``PROBABLE_MAX_FETCH``, and the index's own candidates do not count as the strong match that ends them: a
        song the index does not hold must be verifiable in that time, or a near miss of the index wins). Otherwise
        they start as soon as the offline sources have answered (at the latest after ``HEAD_START_S``) with the
        whole deadline. A source's answer is merged as soon as it arrives; lyric lookups start right away (at most
        ``FETCH_WORKERS`` at a time, ``MAX_FETCH`` in all, best :meth:`_Entry.fetch_priority` first among the
        candidates known at that moment; no new one after ``MIN_FETCH`` once a candidate aligns >= ``FETCH_ENOUGH``:
        LRCLIB is a shared free service). Fallback
        sources (``fallback = True``) start only when the others have answered and their candidates, verified,
        matched nothing (< ``FALLBACK_BELOW``), and only if they would search this text (``skip_reason``). Sources
        still running at ``PROPOSE_SHARE`` of the deadline are reported as timed out, lookups still running at
        ``FETCH_SHARE`` are abandoned.
        """
        futures: dict[Future, Source] = {}
        started: dict[str, float] = {}
        pending: set[Future] = set()
        entries: dict[str, _Entry] = {}
        propose_limit = t0 + self.deadline_s * self.PROPOSE_SHARE
        fetch_limit = t0 + self.deadline_s * self.FETCH_SHARE
        runnable: list[Source] = []
        for s in self.sources:
            try:
                ok = s.available()
            except Exception:  # noqa: BLE001
                ok = False
            if not ok and (s.requires_key or not s.online or online):
                status[s.name] = {"status": "unavailable", "n": 0, "elapsed_s": 0.0,
                                  "message": "needs an API key" if s.requires_key else "not available"}
                continue
            if s.online and not online:
                status[s.name] = {"status": "skipped", "n": 0, "elapsed_s": 0.0,
                                  "message": OFFLINE_MESSAGES.get(offline_reason, "online search off")}
                continue
            runnable.append(s)

        def start(srcs: list[Source]) -> None:
            for s in srcs:
                started[s.name] = time.perf_counter()
                fut = _in_context(ex, s.propose, text, tokens, language, until=propose_limit)
                futures[fut] = s
                pending.add(fut)

        def collect(fut: Future) -> None:
            pending.discard(fut)
            s = futures[fut]
            el = round(time.perf_counter() - started[s.name], 3)
            try:
                cands = fut.result() or []
            except Exception as e:  # noqa: BLE001
                log.info("world source %s failed: %s", s.name, type(e).__name__)
                status[s.name] = {"status": "error", "n": 0, "elapsed_s": el, "message": str(e)[:200]}
                return
            status[s.name] = {"status": "ok", "n": len(cands), "elapsed_s": el, "message": ""}
            for c in cands:
                self._add(entries, c)

        remote = [s for s in runnable if s.online]
        fallback = [s for s in remote if getattr(s, "fallback", False)]
        start([s for s in runnable if not s.online])
        verdict = None
        # probable answer: lookup budget, latest end of the lookups, the index's candidates (they do not end them)
        max_fetch, grace_fetch, index_keys = self.MAX_FETCH, None, set()
        if remote:
            head = min(propose_limit, time.perf_counter() + min(self.HEAD_START_S, 0.25 * self.deadline_s))
            while pending:
                done, _ = wait(pending, timeout=max(0.0, head - time.perf_counter()), return_when=FIRST_COMPLETED)
                if not done:
                    break
                for fut in done:
                    collect(fut)
            verdict, answer = (None, None) if pending else self._index_answer(entries, tokens)
            indexed = set(entries)
            if verdict == "settled" and self.deezer is not None:
                namesake = self._namesake(ex, answer, entries, time.perf_counter() + self.GRACE_S)
                if namesake is not None:
                    verdict = "probable"          # maybe a popular cover: is the more popular namesake the original?
                    self._add(entries, namesake)
            if verdict == "settled":
                for s in remote:
                    status[s.name] = {"status": "skipped", "n": 0, "elapsed_s": 0.0, "message": "local index match"}
                return entries, verdict, time.perf_counter() + self.GRACE_S
            if verdict == "probable":
                grace_fetch = fetch_limit         # the latest end of the lookups of online proposals
                propose_limit = fetch_limit = min(fetch_limit, time.perf_counter() + self.GRACE_S)
                max_fetch, index_keys = self.PROBABLE_MAX_FETCH, indexed
            primary = [s for s in remote if s not in fallback]
            start(primary or fallback)
            if not primary:
                fallback = []
        can_fetch = online and self.lrclib is not None
        fetching: dict[Future, _Entry] = {}
        tried: set[str] = set()
        stats = {"looked_up": 0, "found": 0, "errors": 0, "t0": 0.0}
        enough, fresh = False, False      # a strong match is verified / lyrics arrived since the last check
        while True:
            now = time.perf_counter()
            if pending and now >= propose_limit:
                for fut in pending:
                    s = futures[fut]
                    status[s.name] = {"status": "error", "n": 0, "elapsed_s": round(now - started[s.name], 3),
                                      "message": "timed out"}
                pending.clear()
            if fresh and not enough and stats["looked_up"] >= self.MIN_FETCH:
                fresh = False
                enough = max((a.score for e, a in self._verify(text, entries, memo) if e.key not in index_keys),
                             default=0.0) >= FETCH_ENOUGH
            if grace_fetch is not None and can_fetch and any(
                    self._needs_lyrics(e) for e in entries.values() if e.key not in tried):
                # probable: the first proposal that needs lyrics gets PROBABLE_FETCH_S for its lookups
                fetch_limit = min(grace_fetch, max(fetch_limit, now + self.PROBABLE_FETCH_S))
                grace_fetch = None
            if can_fetch and now < fetch_limit and not enough:
                self._schedule_fetches(ex, entries, tokens, fetching, tried, stats, fetch_limit, max_fetch)
            if fallback and not pending and not fetching:
                # the other sources' candidates are all verified: the fallback runs only if none matched
                best = max((a.score for _, a in self._verify(text, entries, memo)), default=0.0)
                if best < FALLBACK_BELOW and now < propose_limit - self.FALLBACK_MIN_S:
                    ready = []
                    for s in fallback:
                        why = s.skip_reason(tokens) if hasattr(s, "skip_reason") else ""
                        if why:
                            status[s.name] = {"status": "skipped", "n": 0, "elapsed_s": 0.0, "message": why}
                        else:
                            ready.append(s)
                    start(ready)
                else:
                    for s in fallback:
                        status[s.name] = {"status": "skipped", "n": 0, "elapsed_s": 0.0,
                                          "message": "not needed" if best >= FALLBACK_BELOW else "no time left"}
                fallback = []
                continue
            if not pending and not fetching or now >= fetch_limit:
                break
            limit = propose_limit if pending else fetch_limit
            done, _ = wait(pending | set(fetching), timeout=max(0.0, limit - now), return_when=FIRST_COMPLETED)
            for fut in done:
                if fut not in fetching:
                    collect(fut)
                    continue
                e = fetching.pop(fut)
                try:
                    rec = fut.result()
                except Exception:  # noqa: BLE001
                    stats["errors"] += 1
                    continue
                if rec is not None and rec.has_lyrics:
                    e.lyrics, e.lrclib_id = rec.text, rec.id
                    stats["found"] += 1
                    fresh = True
        for s in fallback:
            status[s.name] = {"status": "skipped", "n": 0, "elapsed_s": 0.0, "message": "no time left"}
        missing = sum(1 for e in entries.values() if self._needs_lyrics(e) and e.key not in tried)
        if stats["looked_up"] or (can_fetch and missing):
            stats_msg = (f"{stats['looked_up']} looked up, {missing} skipped"
                         + (f", {stats['errors']} failed" if stats["errors"] else "")
                         + (", timed out" if fetching else ""))
            status["lyrics_fetch"] = {"status": "ok" if not stats["errors"] else "error", "n": stats["found"],
                                      "elapsed_s": round(time.perf_counter() - stats["t0"], 3) if stats["t0"] else 0.0,
                                      "message": stats_msg}
        return entries, verdict, fetch_limit if verdict == "probable" else t0 + self.deadline_s

    def _index_verdict(self, entries: dict[str, _Entry], tokens: list[str]) -> str | None:
        """How far the local index's answer settles the search (see :meth:`_index_answer`)."""
        return self._index_answer(entries, tokens)[0]

    def _index_answer(self, entries: dict[str, _Entry], tokens: list[str]) -> tuple[str | None, _Entry | None]:
        """``(verdict, recording)``: how far the local index's answer settles the search, and the recording that
        would stand for it. The verdict is "settled" (the online sources are not asked), "probable" (they get
        ``GRACE_S`` to add rivals or the original recording) or None (full online search).

        The best verified candidate's *lead* is its alignment score minus that of the best other song (its own
        versions -- karaoke, live, covers, see :meth:`same_song` -- are not rivals). A wrong song rarely leads:
        it scores about as low as many others. The answer settles the search when the less perfect the alignment,
        the clearer the lead (``EXIT_MARGIN`` at ``EXIT_SCORE`` or more, rising linearly to
        ``EXIT_MIN_SCORE_MARGIN`` at ``EXIT_MIN_SCORE``), the query has at least ``EXIT_MIN_WORDS`` distinct
        informative words and the most popular of the versions looks like the original recording (popular, not
        marked as a version). It is probable when it falls short only on the words or the version mark, when
        it aligns near-perfectly (``EXIT_SCORE``) without a clear lead (other songs share the words), or when
        it aligns partly (``PROBABLE_MIN_SCORE``) with a lead of ``EXIT_MIN_SCORE_MARGIN``. An unpopular best
        recording (< ``ORIGINAL_MIN_RANK``) may be a cover whose original only the full online search can find.
        """
        verified = sorted(((e, e.align) for e in entries.values() if e.align is not None),
                          key=lambda x: (-x[1].score, -x[0].rank, x[0].key))
        if not verified or verified[0][1].score < self.PROBABLE_MIN_SCORE:
            return None, None
        top, ta = verified[0]
        versions, lead = [top], 1.0
        for e, a in verified[1:]:
            ev = self.song_evidence(top, ta, e, a)
            if ev == "lines" or (ev == "mark" and ta.score - a.score <= COVER_SCORE_WINDOW):
                versions.append(e)               # the same song: shown as its most popular recording
            elif ev is None:
                lead = min(lead, ta.score - a.score)
        short = min(1.0, max(0.0, self.EXIT_SCORE - ta.score) / (self.EXIT_SCORE - self.EXIT_MIN_SCORE))
        clear = ta.score >= self.EXIT_MIN_SCORE and lead >= (
            self.EXIT_MARGIN + short * (self.EXIT_MIN_SCORE_MARGIN - self.EXIT_MARGIN))
        best = max(versions, key=lambda e: e.rank)
        if best.rank < ORIGINAL_MIN_RANK:
            return None, best
        if clear and len(set(informative_tokens(tokens))) >= self.EXIT_MIN_WORDS and not version_marked(
                *best.title_artist()):
            return "settled", best
        if clear or ta.score >= self.EXIT_SCORE or lead >= self.EXIT_MIN_SCORE_MARGIN:
            return "probable", best
        return None, best

    def _namesake(self, ex: ThreadPoolExecutor, e: _Entry, entries: dict[str, _Entry],
                  until: float) -> Candidate | None:
        """A settled answer's check: the most popular Deezer recording with its title by another artist, if that
        one is more popular, not marked as a version, neither indexed with lyrics nor among the candidates, and
        by an artist the index knows (another of their songs is indexed).

        The index may hold a popular cover (rank >= ``ORIGINAL_MIN_RANK``, no version mark) but not its more
        popular original; only the online search can tell (the grouping then shows the original when the two
        share their lyrics). The known-artist condition keeps out cover channels whose upload Deezer ranks above
        the original (a dev song: an unmarked cover at rank 903k, the indexed original at 436k): the sync collects
        charts, editorial playlists and the top songs of established artists, so the artist of a missing original
        usually has other songs indexed. One title search, cacheable (it carries the indexed title, not the
        user's words); None when nothing qualifies, it fails or it does not answer by ``until``. Another song
        that only shares the title costs the probable path's short wait and is ranked by its own lyrics."""
        title, artist = e.title_artist()
        fut = _in_context(ex, self.deezer.search_title, core_title(title), 10, until=until)
        done, _ = wait([fut], timeout=max(0.0, until - time.perf_counter()))
        try:
            tracks = fut.result() if done else []
        except Exception:  # noqa: BLE001 - the index's answer stands
            tracks = []
        best = None
        for t in sorted(tracks, key=lambda t: -t.rank):
            if (max(title_similarity(title, t.title), title_similarity(title, t.title_short)) < 0.9
                    or artist_similarity(artist, t.artist) >= 0.7 or t.rank <= e.rank
                    or version_marked(t.title, t.artist)):
                continue
            key = song_key(t.title, t.artist)
            ws = self.index.get_by_key(t.title, t.artist)
            if key and key not in entries and (ws is None or not ws.lyrics) and any(
                    artist_similarity(t.artist, m["artist"]) >= 0.7
                    for m in self.index.list_songs(primary_artist(t.artist), limit=10)):
                best = t
                break
        if best is None:
            return None
        return Candidate(title=best.title, artist=best.artist, album=best.album, duration_s=best.duration_s,
                         meta=best, source="deezer", prior=1.0)

    def _needs_lyrics(self, e: _Entry) -> bool:
        """No lyrics evidence yet: not verified by the index, no lyrics proposed, not in the local index."""
        if e.align is not None or e.lyrics or any(c.lyrics for c in e.cands):
            return False
        if e.world_song is None and not e.index_checked:
            e.index_checked = True
            title, artist = e.title_artist()
            e.world_song = self.index.get_by_key(title, artist)
        return e.world_song is None or not e.world_song.lyrics

    def _schedule_fetches(self, ex: ThreadPoolExecutor, entries: dict[str, _Entry], tokens: list[str],
                          fetching: dict[Future, _Entry], tried: set[str], stats: dict, until: float,
                          max_fetch: int | None = None) -> None:
        lrclib = self.lrclib
        max_fetch = self.MAX_FETCH if max_fetch is None else max_fetch

        def fetch(title: str, artist: str, dur: float | None):
            return lrclib.find(title, artist, dur)

        while len(fetching) < self.FETCH_WORKERS and stats["looked_up"] < max_fetch:
            todo = [e for e in entries.values() if e.key not in tried and self._needs_lyrics(e)]
            if not todo:
                return
            e = min(todo, key=lambda x: x.fetch_priority(tokens))
            tried.add(e.key)
            stats["looked_up"] += 1
            stats["t0"] = stats["t0"] or time.perf_counter()
            title, artist = e.title_artist()
            dur = e.meta.duration_s if e.meta else e.first.duration_s
            fetching[_in_context(ex, fetch, title, artist, dur, until=until)] = e

    @staticmethod
    def _add(entries: dict[str, _Entry], c: Candidate) -> None:
        try:
            key = song_key(c.title, c.artist)
        except Exception:  # noqa: BLE001
            return
        if not key or key.startswith("|"):
            return
        e = entries.setdefault(key, _Entry(key))
        e.cands.append(c)
        e.sources.append(c.source)
        if c.world_song is not None and e.world_song is None:
            e.world_song = c.world_song
        if c.meta is not None and (e.meta is None or c.meta.rank > e.meta.rank):
            e.meta = c.meta
        if c.align is not None and (e.align is None or c.align.score > e.align.score):
            e.align = c.align

    def _verify(self, text: str, entries: dict[str, _Entry],
                memo: dict[str, AlignResult | None] | None = None) -> list[tuple[_Entry, AlignResult]]:
        """Each candidate's best alignment over its lyrics texts (``memo``: lyrics -> alignment, kept across the
        calls of one search); candidates under ``min_score`` are dropped."""
        memo = {} if memo is None else memo
        out = []
        for e in entries.values():
            self._needs_lyrics(e)                    # looks the song up in the local index once
            best = e.align
            texts: list[tuple[str, int | None]] = []
            if e.lyrics:
                texts.append((e.lyrics, e.lrclib_id))
            for c in e.cands:
                if c.lyrics:
                    texts.append((c.lyrics, c.lrclib_id))
            if best is None and e.world_song is not None and e.world_song.lyrics:
                texts.append((e.world_song.lyrics, e.world_song.lrclib_id))
            seen = set()
            for lyr, lid in texts:
                h = hash(lyr)
                if h in seen:
                    continue
                seen.add(h)
                if lyr not in memo:
                    memo[lyr] = self.index.score_lyrics(text, lyr)
                r = memo[lyr]
                if r is not None and (best is None or r.score > best.score):
                    best, e.lyrics, e.lrclib_id = r, lyr, lid
            if best is not None and best.score >= self.min_score:
                e.align = best
                out.append((e, best))
        out.sort(key=lambda x: (-x[1].score, -x[0].rank, x[0].key))
        return out

    def _enrich(self, ex: ThreadPoolExecutor, scored: list[tuple[_Entry, AlignResult]], deadline: float) -> None:
        todo = [e for e, _ in scored[: self.ENRICH_TOP]
                if e.meta is None and not (e.world_song is not None and e.world_song.deezer_id)]
        if not todo:
            return
        deezer = self.deezer
        futs = {_in_context(ex, deezer.find, *e.title_artist(), until=deadline): e for e in todo}
        done, _ = wait(set(futs), timeout=max(0.0, deadline - time.perf_counter()))
        for fut in done:
            try:
                tm = fut.result()
            except Exception:  # noqa: BLE001
                continue
            if tm is not None:
                futs[fut].meta = tm

    def _original_of(self, text: str, e: _Entry, keys: set[str], score: float) -> tuple[_Entry, AlignResult] | None:
        """The most popular Deezer recording titled like ``e`` by another artist, verified by its own lyrics: it
        must align about as well as ``e`` (``score``) or share its lyrics (:meth:`song_evidence`), else it is
        another song that shares the title."""
        title, artist = e.title_artist()
        tracks = self.deezer.search_title(core_title(title), 10)
        same = [t for t in tracks if max(title_similarity(title, t.title), title_similarity(title, t.title_short)) >= 0.9
                and artist_similarity(artist, t.artist) < 0.7 and t.rank > max(2 * e.rank, ORIGINAL_MIN_RANK // 3)]
        if not same:
            return None
        best = max(same, key=lambda t: t.rank)
        key = song_key(best.title, best.artist)
        if key in keys:
            return None
        o = _Entry(key, cands=[Candidate(title=best.title, artist=best.artist, album=best.album,
                                         duration_s=best.duration_s, meta=best, source="deezer")],
                   sources=["deezer"], meta=best)
        o.world_song = self.index.get_by_key(best.title, best.artist)
        if o.world_song is not None and o.world_song.lyrics:
            o.lyrics, o.lrclib_id = o.world_song.lyrics, o.world_song.lrclib_id
        else:
            rec = self.lrclib.find(best.title, best.artist, best.duration_s) if self.lrclib is not None else None
            if rec is None or not rec.has_lyrics:
                return None
            o.lyrics, o.lrclib_id = rec.text, rec.id
        r = self.index.score_lyrics(text, o.lyrics)
        if r is None or r.score < self.min_score:
            return None
        if r.score < score - ORIGINAL_SCORE_WINDOW and (
                e.align is None or self.song_evidence(e, e.align, o, r) != "lines"):
            return None                          # another song with that title (the same song shares the lyrics)
        o.align = r
        return o, r

    def _add_originals(self, ex: ThreadPoolExecutor, text: str, scored: list[tuple[_Entry, AlignResult]],
                       deadline: float) -> list[tuple[_Entry, AlignResult]]:
        """Best matches that look like covers / karaoke (unpopular): also verify the popular original."""
        keys = {e.key for e, _ in scored}
        todo = [(e, a) for e, a in scored[: self.ORIGINAL_TOP]
                if a.score >= LEARN_MIN_SCORE and e.rank < ORIGINAL_MIN_RANK]
        if not todo:
            return scored
        futs = [_in_context(ex, self._original_of, text, e, keys, a.score, until=deadline) for e, a in todo]
        done, _ = wait(set(futs), timeout=max(0.0, deadline - time.perf_counter()))
        extra = []
        for fut in futs:
            if fut in done:
                try:
                    r = fut.result()
                except Exception:  # noqa: BLE001
                    continue
                if r is not None and r[0].key not in keys:
                    keys.add(r[0].key)
                    extra.append(r)
        if not extra:
            return scored
        out = scored + extra
        out.sort(key=lambda x: (-x[1].score, -x[0].rank, x[0].key))
        return out

    @staticmethod
    def song_evidence(k: _Entry, ka: AlignResult, e: _Entry, a: AlignResult) -> str | None:
        """Why two records are one song (covers, karaoke, live, re-uploads), if they are: not two songs that share
        a title or a line. A title followed only by version words ("X (Spanish Mix)", :func:`title_version_of`)
        counts as the same title, marked as a version.

        ``"lines"`` -- they share the lyrics: most distinct lines of the shorter one are in the other
        (``SAME_SONG_OVERLAP``, any titles), or the titles are the same and a good part of the lines are shared
        (``SAME_TITLE_OVERLAP``, at least ``MIN_SAME_TITLE_LINES``): two LRCLIB transcriptions of one song often
        share only 30-65 % of their distinct lines (other line breaks, ad-libs, missing parts). Records with
        other titles are compared only when they score about the same (``COVER_SCORE_WINDOW``; comparing lyrics
        costs time). ``"mark"`` -- a medley / mashup (:func:`medley_marked`) matched the same sung lines about as
        well (``MEDLEY_MIN``), or the titles are the same and either one is marked as a version and the same lines
        matched, or both hold the sung lines almost exactly (``SAME_SUNG_MIN``, ``SAME_SUNG_SIM``: a cover that is
        partly translated shares few distinct lines with the original but the sung ones verbatim).
        """
        tk, te = k.title_artist(), e.title_artist()
        same_title = normalize_title(tk[0]) == normalize_title(te[0])
        versioned = not same_title and title_version_of(tk[0], te[0])     # "X" / "X (Spanish Mix)"
        same_title = same_title or versioned
        if same_title or abs(ka.score - a.score) <= COVER_SCORE_WINDOW:
            hits, n = shared_lines(k.lines(), e.lines())
            if n and (hits >= SAME_SONG_OVERLAP * n
                      or (same_title and hits >= max(MIN_SAME_TITLE_LINES, SAME_TITLE_OVERLAP * n))):
                return "lines"
        medley = medley_marked(tk[0]) or medley_marked(te[0])
        sim = fuzz.ratio(ka.matched_text, a.matched_text) if same_title or medley else 0.0
        if medley and not same_title and min(ka.score, a.score) >= MEDLEY_MIN and sim >= SAME_SUNG_SIM:
            return "mark"                     # a medley / mashup holding this song's sung lines
        if same_title and (
                ((versioned or version_marked(*tk) or version_marked(*te)) and sim >= COVER_TEXT_SIM)
                or (min(ka.score, a.score) >= SAME_SUNG_MIN and sim >= SAME_SUNG_SIM)):
            return "mark"
        return None

    @classmethod
    def same_song(cls, k: _Entry, ka: AlignResult, e: _Entry, a: AlignResult) -> bool:
        """Two records of one song (see :meth:`song_evidence`)."""
        return cls.song_evidence(k, ka, e, a) is not None

    @classmethod
    def _group_covers(cls, scored: list[tuple[_Entry, AlignResult]]) -> list[tuple[_Entry, AlignResult]]:
        """Collapse versions of one song (see :meth:`song_evidence`) into one entry, shown as its most popular
        recording. A record joins a group when it is the same song as any of its members (two covers may each
        share most lines with the original but fewer with each other).

        A group joined by records that share the lyrics (``"lines"``) is one song whatever the scores, shown with
        the group's best alignment by its most popular recording among those that align within
        ``COVER_SCORE_WINDOW`` of the best or are clearly more popular than the best-aligned one
        (``POPULAR_FACTOR``): a little-known cover whose transcription matches the sung words better than the
        original's must not hide the original, but between recordings of about the same popularity the one whose
        lyrics hold the sung words is shown (a verse only a remix has). Records only marked as versions
        (``"mark"``) join when they score about the same (``COVER_SCORE_WINDOW``); in such a group the more
        popular recording wins unless it aligns more than ``POPULARITY_OVERRIDE`` worse, and keeps its own
        alignment (the version mark says nothing about the rest of the lyrics).
        """
        groups: list[list[tuple[_Entry, AlignResult]]] = []
        shared: list[bool] = []
        for e, a in scored:
            gi, ev = None, None
            for i, members in enumerate(groups):
                for k, ka in members:
                    ev = cls.song_evidence(k, ka, e, a)
                    if ev == "lines" or (ev == "mark" and abs(ka.score - a.score) <= COVER_SCORE_WINDOW):
                        gi = i
                        break
                if gi is not None:
                    break
            if gi is None:
                groups.append([(e, a)])
                shared.append(False)
            else:
                groups[gi].append((e, a))
                shared[gi] = shared[gi] or ev == "lines"
        kept: list[tuple[_Entry, AlignResult]] = []
        for members, lines in zip(groups, shared):
            best_score = max(a.score for _, a in members)
            singles = [(e, a) for e, a in members if not medley_marked(e.title_artist()[0])]
            if singles and len(singles) < len(members):
                # a medley never stands for the song: its own record does, with the group's evidence
                members = singles if max(a.score for _, a in singles) >= best_score - COVER_SCORE_WINDOW else members
            if lines:
                best, wa = max(members, key=lambda x: x[1].score)
                winner = max(((e, a) for e, a in members if a.score >= wa.score - COVER_SCORE_WINDOW
                              or e.rank > POPULAR_FACTOR * best.rank), key=lambda x: (x[0].rank, x[1].score))[0]
            else:
                winner, wa = members[0]
                for e, a in members[1:]:
                    (p, pa), (q, qa) = ((e, a), (winner, wa)) if e.rank > winner.rank else ((winner, wa), (e, a))
                    winner, wa = (p, pa) if pa.score >= qa.score - POPULARITY_OVERRIDE else (q, qa)
            winner.sources = list(dict.fromkeys(src for e, _ in [(winner, wa)] + members for src in e.sources))
            winner.align = wa
            kept.append((winner, wa))
        kept.sort(key=lambda x: (-x[1].score, -x[0].rank, x[0].key))
        return kept

    @staticmethod
    def matched_words(tokens: list[str], a: AlignResult) -> int:
        """Distinct informative query words found (fuzzy / phonetic, as aligned) in the matched lyric lines."""
        words = list(dict.fromkeys(t for t in informative_tokens(tokens) if t not in FILLERS))
        lyric = list(dict.fromkeys(tokenize(a.matched_text)))
        if not words or not lyric:
            return 0
        return int((similarity_matrix(words, lyric, phonetic_keys=True).max(axis=1) >= MATCH_SIM).sum())

    @classmethod
    def _evidence_guard(cls, tokens: list[str], scored: list[tuple[_Entry, AlignResult]]
                        ) -> list[tuple[_Entry, AlignResult]]:
        """A best match that is partial (< ``PARTIAL_SCORE``), rests on at most ``LOW_EVIDENCE_WORDS`` matched
        informative words and leads the next song by less than ``LOW_EVIDENCE_LEAD`` gets its score capped at
        ``LOW_EVIDENCE_CAP`` or, if higher, just enough to stay first once :meth:`_rank` adds the popularity
        tie-break: the guard lowers the confidence (a lead of 0.001), it does not hand the first place to a song
        with weaker lyric evidence.

        Short, generic lines align partly with many songs, and the alignment score is relative to the query: two
        common words matched out of four look as good as most of a misremembered verse. On the dev
        half of the real-song evaluation, wrong songs that would have been matched (0.64-0.74) were all short
        line queries with 2-3 matched informative words and a lead of 0.10-0.20 over the next song, while every
        right partial match (misremembered verses, 0.64-0.90) led by at least 0.21.
        """
        if not scored:
            return scored
        i, (e, a) = max(enumerate(scored), key=lambda x: (x[1][1].score, -x[0]))
        nxt = max((b.score for j, (_, b) in enumerate(scored) if j != i), default=0.0)
        if (a.score >= PARTIAL_SCORE or a.score - nxt >= LOW_EVIDENCE_LEAD
                or cls.matched_words(tokens, a) > LOW_EVIDENCE_WORDS):
            return scored
        ahead = max((b.score + popularity_bonus(f.rank) for j, (f, b) in enumerate(scored) if j != i), default=0.0)
        capped = min(a.score, max(LOW_EVIDENCE_CAP, nxt, ahead - popularity_bonus(e.rank) + 1e-3))
        if capped >= a.score:
            return scored
        out = list(scored)
        out[i] = (e, AlignResult(capped, a.coverage, a.char_similarity, a.matched_text, a.line_range, a.query_span))
        e.align = out[i][1]
        return out

    @staticmethod
    def _rank(scored: list[tuple[_Entry, AlignResult]]) -> list[tuple[_Entry, float]]:
        ranked = [(e, min(1.0, max(0.0, a.score + popularity_bonus(e.rank)))) for e, a in scored]
        ranked.sort(key=lambda x: (-x[1], -x[0].rank, x[0].key))
        return ranked

    def _candidate(self, e: _Entry, score: float) -> WorldCandidate:
        title, artist = e.title_artist()
        ws, tm, a = e.world_song, e.meta, e.align
        sid = ws.id if ws is not None and ws.key == e.key else world_song_id(title, artist)
        album = (tm.album if tm else None) or (ws.album if ws else None) or e.first.album
        duration = (tm.duration_s if tm else None) or (ws.duration_s if ws else None) or e.first.duration_s
        language = (ws.language if ws else None) or (guess_language(e.lyrics) if e.lyrics else None)
        link = (tm.link if tm else None) or (ws.link if ws else None)
        return WorldCandidate(
            song_id=f"world:{sid}", title=title, artist=artist, album=album, language=language, duration_s=duration,
            cover_url=(tm.cover_url if tm else None) or (ws.cover_url if ws else None),
            preview_url=fresh_preview(tm.preview_url if tm else None) or fresh_preview(ws.preview_url if ws else None),
            rank=e.rank, links=song_links(title, artist, link), score=score, coverage=a.coverage,
            char_similarity=a.char_similarity, matched_text=a.matched_text, line_range=tuple(a.line_range),
            sources=list(dict.fromkeys(e.sources)), alignment_score=a.score,
            deezer_id=(tm.deezer_id if tm else None) or (ws.deezer_id if ws else None))

    def _learn(self, final: list[tuple[_Entry, float]]) -> None:
        songs = []
        for e, _ in final:
            if e.align is None or e.align.score < LEARN_MIN_SCORE or not e.lyrics:
                continue
            ws0 = e.world_song
            if ws0 is not None and ws0.lyrics and ws0.key == e.key and (ws0.deezer_id or e.meta is None):
                continue                      # already indexed, nothing new
            title, artist = e.title_artist()
            tm, ws = e.meta, e.world_song
            try:
                songs.append(WorldSong.create(
                    title, artist, album=(tm.album if tm else None) or (ws.album if ws else None),
                    duration_s=(tm.duration_s if tm else None) or (ws.duration_s if ws else None),
                    language=ws.language if ws else None, lyrics=e.lyrics, rank=e.rank,
                    deezer_id=(tm.deezer_id if tm else None) or (ws.deezer_id if ws else None),
                    isrc=(tm.isrc if tm else None) or (ws.isrc if ws else None),
                    lrclib_id=e.lrclib_id or (ws.lrclib_id if ws else None),
                    cover_url=(tm.cover_url if tm else None) or (ws.cover_url if ws else None),
                    link=(tm.link if tm else None) or (ws.link if ws else None),
                    preview_url=storable_preview(tm.preview_url if tm else None)
                    or storable_preview(ws.preview_url if ws else None),
                    source=ws.source if ws is not None and ws.source else "lookup", added_at=time.time()))
            except Exception:  # noqa: BLE001
                continue
        if not songs:
            return
        with self._learn_lock:
            for s in songs:
                try:
                    self.index.upsert(s)
                except Exception as e:  # noqa: BLE001
                    log.info("could not learn a world song: %s", type(e).__name__)
