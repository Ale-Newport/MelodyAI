"""Candidate sources for the open-world song search: who *might* this transcript be?

A source only proposes :class:`Candidate` songs (title + artist, sometimes with lyrics or Deezer metadata);
nothing it says is trusted. :class:`melodyai.world.search.WorldSearch` fetches the real lyrics of every
candidate and verifies them against the transcript (fuzzy, order-aware alignment) before ranking.

Sources (only text leaves the machine, never audio):

* ``index``  -- the local world lyrics index (popular-songs sync + learned lookups); offline, already verified.
* ``lrclib`` -- LRCLIB title search with phrases of the transcript (a sung chorus usually contains the song
  title). LRCLIB search results include the lyrics, so its candidates are verified without extra requests. A
  *fallback* source (``fallback = True``): it runs only when the other sources' candidates matched nothing, and
  only for a transcript that repeats a phrase (``skip_reason``).
* ``deezer`` -- Deezer track search with overlapping windows of the sung words (Deezer's plain search matches
  lyric text as an exact phrase, see :class:`DeezerTitleSource`), their shorter retries, and the same title
  phrases (for a chorus-like transcript, else as a last resort). Lyrics are fetched later, for the most
  promising only.
* ``genius`` -- Genius API search, which *does* search lyric text (needs ``MELODYAI_GENIUS_TOKEN``; off by default).
* ``claude`` -- asks Claude which songs contain these possibly misheard words (needs the optional ``anthropic``
  package and ``MELODYAI_ANTHROPIC_API_KEY``; off by default). The generic ``ANTHROPIC_API_KEY`` many developers
  export for other tools is only used after an explicit opt-in (``MELODYAI_CLAUDE_SOURCE=1``), so a key in the
  shell never silently sends transcripts to another service. Only titles/artists are requested, never lyrics.

Requests that carry the user's words are sent with ``private=True`` (hashed, short-lived cache entries, see
:class:`melodyai.world.web.WebClient`), and each source's own worker threads inherit the search's request
deadline (:func:`melodyai.world.web.request_deadline`), so a search that timed out stops sending.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from melodyai.lyrics.align import AlignResult, query_segments
from melodyai.lyrics.normalize import informative_tokens, normalize_text
from melodyai.world.deezer import DeezerClient, TrackMeta
from melodyai.world.index import WorldLyricsIndex, WorldSong
from melodyai.world.lrclib import LrclibClient
from melodyai.world.text import FILLERS, is_common_word, normalize_title, title_phrases
from melodyai.world.web import WebClient, WebError, current_deadline, request_deadline

log = logging.getLogger(__name__)

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
TRUTHY = ("1", "true", "yes", "on")


def claude_api_key() -> str:
    """``MELODYAI_ANTHROPIC_API_KEY``; the generic ``ANTHROPIC_API_KEY`` only with ``MELODYAI_CLAUDE_SOURCE=1``."""
    key = os.environ.get("MELODYAI_ANTHROPIC_API_KEY", "").strip()
    if not key and os.environ.get("MELODYAI_CLAUDE_SOURCE", "").strip().lower() in TRUTHY:
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return key


SOURCE_MARGIN_S = 0.25     # a source answers with what it has this long before its deadline


def _in_context(ex: ThreadPoolExecutor, fn: Callable, *args, **kwargs):
    """Submit ``fn`` so that it runs with the caller's context variables (e.g. the request deadline)."""
    return ex.submit(contextvars.copy_context().run, fn, *args, **kwargs)


@contextmanager
def _source_deadline():
    """For a source's own requests: the request deadline moved ``SOURCE_MARGIN_S`` earlier (yielded, a
    ``time.monotonic()`` instant, or None without a deadline). The source answers with what it has by then, and
    none of its requests can be sent after it has answered (see :func:`_results`)."""
    deadline = current_deadline()
    inner = None if deadline is None else deadline - SOURCE_MARGIN_S
    with request_deadline(inner):
        yield inner


def _results(futures: list[Future], until: float | None) -> list[tuple[object, Exception | None]]:
    """``(result, error)`` of each future, waiting at most until ``until`` (``time.monotonic()``): one slow request
    (a stalled connection, a provider's quota pause) must not lose the answers of the others. Futures still
    running count as timed out; their requests stop at ``until`` (:func:`_source_deadline`)."""
    timeout = None if until is None else max(0.0, until - time.monotonic())
    done, _ = wait(futures, timeout=timeout)
    out: list[tuple[object, Exception | None]] = []
    for fut in futures:
        if fut not in done:
            out.append((None, WebError("timed out", kind="deadline")))
            continue
        try:
            out.append((fut.result(), None))
        except Exception as e:  # noqa: BLE001 - one failed request must not lose the others
            out.append((None, e))
    return out


@dataclass
class Candidate:
    """A song proposed by a source, verified later against its real lyrics."""

    title: str
    artist: str
    album: str | None = None
    duration_s: float | None = None
    lyrics: str | None = None
    meta: TrackMeta | None = None
    world_song: WorldSong | None = None
    source: str = ""
    prior: float = 0.0                   # source confidence, only used to choose which lyrics to fetch first
    align: AlignResult | None = None     # already verified by the source (index)
    lrclib_id: int | None = None


@runtime_checkable
class Source(Protocol):
    name: str
    label: str
    online: bool
    requires_key: bool

    def available(self) -> bool: ...

    def propose(self, query_text: str, tokens: list[str], language: str | None) -> list[Candidate]: ...


# ---------------------------------------------------------------------------- query helpers
def _memo(fn: Callable[[str], float] | None) -> Callable[[str], float] | None:
    if fn is None:
        return None
    cache: dict[str, float] = {}

    def wrapped(t: str) -> float:
        if t not in cache:
            cache[t] = float(fn(t))
        return cache[t]

    return wrapped


def query_phrases(tokens: Sequence[str], n: int, idf: Callable[[str], float] | None = None) -> list[str]:
    """Likely title phrases of a transcript (fillers such as "oh", "yeah" removed first)."""
    toks = [t for t in tokens if t not in FILLERS]
    phrases = title_phrases(toks, max_phrases=2 * n, idf=_memo(idf))
    padded = f" {' '.join(toks)} "

    def count(p: str) -> int:
        return padded.count(f" {p} ")

    # a word that is only ever sung inside a longer candidate phrase ("tormenta" in "tormenta de cristal")
    # is a worse title query than that phrase: drop it
    kept = [p for p in phrases
            if not any(q != p and f" {p} " in f" {q} " and count(q) >= count(p) for q in phrases)]
    return kept[:n]


def lyric_windows(query_text: str, size: int = 7, stride: int = 4, max_windows: int = 6, sub: int = 5,
                  idf: Callable[[str], float] | None = None) -> list[tuple[str, list[str]]]:
    """Windows of ``size`` consecutive sung words for a phrase-exact lyric search, each with its retry windows.

    The transcript is one word sequence (fillers removed; windows may straddle its line breaks, which need not
    be the song's). A query of at most ``size + 1`` words is one window; a longer one gets windows every
    ``stride`` words (the last one ends with the query), and the ``max_windows`` most informative (IDF of their
    informative words, :func:`melodyai.lyrics.normalize.informative_tokens`; windows with fewer than two are
    skipped) are kept, in sung order. The retry windows -- the first and the last ``sub`` words -- are meant
    for a window that found nothing: a dropped, added or misheard word breaks a phrase-exact match, and it can
    only sit at one end of the window or in the middle.
    """
    tokens, _ = query_segments(query_text)
    seq = [t for t in tokens if t not in FILLERS]
    if len(seq) < 3:
        return []
    weight = _memo(idf) or (lambda t: 1.0)
    if len(seq) <= size + 1:
        spans = [(0, len(seq))]
    else:
        starts = list(range(0, len(seq) - size + 1, max(1, stride)))
        if starts[-1] != len(seq) - size:
            starts.append(len(seq) - size)
        spans = [(s, s + size) for s in starts]
    scored = []
    for a, b in spans:
        info = list(dict.fromkeys(informative_tokens(seq[a:b])))
        if len(info) >= 2:
            scored.append((sum(weight(t) for t in info), a, b))
    kept = sorted(sorted(scored, key=lambda x: (-x[0], x[1]))[:max(0, max_windows)], key=lambda x: x[1])
    out = []
    for _, a, b in kept:
        w = seq[a:b]
        subs = [" ".join(w[:sub]), " ".join(w[-sub:])] if len(w) > sub else []
        out.append((" ".join(w), subs))
    return out


def repeated_phrase(tokens: Sequence[str]) -> bool:
    """Whether the transcript repeats itself like a chorus (which usually holds the title): a group of 2-4 words,
    not all function words, sung twice without overlap, or a long word (>= 7 letters) sung twice."""
    toks = [t for t in tokens if t not in FILLERS]
    for n in (2, 3, 4):
        first: dict[tuple[str, ...], int] = {}
        for i in range(len(toks) - n + 1):
            g = tuple(toks[i:i + n])
            if all(is_common_word(t) or len(t) <= 1 for t in g):
                continue
            if g in first and i >= first[g] + n:
                return True
            first.setdefault(g, i)
    seen: set[str] = set()
    for t in toks:
        if len(t) >= 7 and t in seen:
            return True
        seen.add(t)
    return False


def title_in_query(title: str, tokens: Sequence[str]) -> float:
    """1.0 if every word of the (normalised) title was sung, else the fraction of its informative words sung."""
    words = normalize_title(title).split()
    if not words:
        return 0.0
    sung = set(tokens)
    if all(w in sung for w in words):
        return 1.0
    inf = [w for w in words if not is_common_word(w)] or words
    return sum(w in sung for w in inf) / len(inf) * 0.8


# ---------------------------------------------------------------------------- sources
class LocalIndexSource:
    name, label, online, requires_key = "index", "Local lyrics index", False, False

    def __init__(self, index: WorldLyricsIndex, top_k: int = 12, max_candidates: int = 150) -> None:
        self.index, self.top_k, self.max_candidates = index, top_k, max_candidates

    def available(self) -> bool:
        return True

    def propose(self, query_text: str, tokens: list[str], language: str | None) -> list[Candidate]:
        matches = self.index.search(query_text, top_k=self.top_k, language=language,
                                    max_candidates=self.max_candidates)
        return [Candidate(title=m.song.title, artist=m.song.artist, album=m.song.album, duration_s=m.song.duration_s,
                          world_song=m.song, source=self.name, prior=m.align.score, align=m.align,
                          lrclib_id=m.song.lrclib_id) for m in matches]


class LrclibTitleSource:
    """LRCLIB title search with title phrases of the transcript; its records include the lyrics.

    A fallback (``fallback = True``): :class:`melodyai.world.search.WorldSearch` runs it only when the other
    sources' candidates, once verified, matched nothing, and it searches only when the transcript repeats a
    phrase (``require_repeat``, :func:`repeated_phrase`): a chorus usually repeats the title. A real-song check
    (dev half of ``configs/eval_world.yaml``, 60 online queries, 2026-11-07) found it proposed anything for 17
    queries and the right song for 3, which Deezer had found too, while it cost 2 of the ~7.6 LRCLIB requests of
    a query (LRCLIB is a free shared service; its budget is better spent on lyric lookups).
    """

    name, label, online, requires_key = "lrclib", "LRCLIB (title search)", True, False
    fallback = True

    def __init__(self, lrclib: LrclibClient, max_phrases: int = 2,
                 idf: Callable[[str], float] | None = None, require_repeat: bool = True) -> None:
        self.lrclib, self.max_phrases, self.idf, self.require_repeat = lrclib, max_phrases, idf, require_repeat

    def available(self) -> bool:
        return not self.lrclib.web.offline

    def skip_reason(self, tokens: list[str]) -> str:
        """Why this source would not search this transcript ("" = it would): reported as its status."""
        return "no repeated phrase" if self.require_repeat and not repeated_phrase(tokens) else ""

    def propose(self, query_text: str, tokens: list[str], language: str | None) -> list[Candidate]:
        if self.skip_reason(tokens):
            return []
        phrases = query_phrases(tokens, self.max_phrases, self.idf)
        if not phrases:
            return []
        out: list[Candidate] = []
        errors: list[Exception] = []
        ex = ThreadPoolExecutor(max_workers=min(3, len(phrases)), thread_name_prefix="lrclib-src")
        try:
            with _source_deadline() as until:
                futures = [_in_context(ex, self.lrclib.search, track_name=p, private=True) for p in phrases]
            for i, (records, err) in enumerate(_results(futures, until)):
                if err is not None:
                    errors.append(err)
                    continue
                for r in records:
                    if r.has_lyrics:
                        out.append(Candidate(title=r.track, artist=r.artist, album=r.album, duration_s=r.duration_s,
                                             lyrics=r.text, source=self.name, lrclib_id=r.id,
                                             prior=title_in_query(r.track, tokens) - 0.05 * i))
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        if errors and not out:
            raise errors[0]
        return out


class DeezerTitleSource:
    """Deezer track search with windows of the sung words and with title phrases.

    Deezer's plain search also matches lyric text, but as an exact phrase: a live check on one long line of
    each of 48 songs (dev half of ``configs/eval_world.yaml``, 2026-11-07) put the right song in the top 5 for
    60 % of 8-word windows, 56 % of 6-word windows and whole lines, 50 % of 5-word windows, 31 % of 4-word
    windows and 2 % of the rarest 3 words; a window across two lines (4 + 4 words) did as well as one inside a
    line (57 %), while 12-word windows across two lines found it 43 % of the time and nothing at all 54 % of
    the time. One dropped, replaced or reordered word gave 0-2 % (nothing at all 67-75 % of the time). So the
    transcript is sent as overlapping windows of ``window`` words (:func:`lyric_windows`), and a window that
    finds nothing is retried with its first and its last ``sub_window`` words (a misheard word sits at one end
    or in the middle): in a second round, most informative windows first, while the query's requests stay
    within ``max_requests`` (Deezer allows 50 per 5 s per IP; text that is not a lyric, where every window finds
    nothing, would otherwise cost three per window). Title phrases are searched for chorus-like transcripts in the
    first round; for any other transcript only as a last resort, when every window and retry found nothing (one
    misheard word in each, or a 4-6-word line): then, within ``max_requests``, and only tracks whose whole title
    was sung are kept (Deezer's answer to two or three common words is a list of unrelated tracks, each a lyric
    lookup to verify). Candidates found by windows get a prior from their position (fetched first; a song found
    by several windows adds up); title-phrase candidates get the share of their title that was sung.
    """

    name, label, online, requires_key = "deezer", "Deezer (sung lines and titles)", True, False

    def __init__(self, deezer: DeezerClient, max_phrases: int = 2, per_query: int = 5, max_windows: int = 6,
                 window: int = 7, stride: int = 4, sub_window: int = 5, max_requests: int = 12,
                 idf: Callable[[str], float] | None = None) -> None:
        self.deezer, self.max_phrases, self.per_query, self.max_windows = deezer, max_phrases, per_query, max_windows
        self.window, self.stride, self.sub_window, self.idf = window, stride, sub_window, idf
        self.max_requests = max_requests

    def available(self) -> bool:
        return not self.deezer.web.offline

    def window_plan(self, query_text: str) -> list[tuple[str, str, list[str]]]:
        """``("window", text, retry windows)`` for the sung words (see :func:`lyric_windows`)."""
        return [("window", q, subs) for q, subs in lyric_windows(query_text, self.window, self.stride,
                                                                 self.max_windows, self.sub_window, self.idf)]

    def queries(self, query_text: str, tokens: list[str]) -> list[tuple[str, str, list[str]]]:
        """Window queries (with their retry windows) and, for a transcript that repeats a phrase like a chorus
        (:func:`repeated_phrase`), ``("title", phrase, [])`` queries; deduplicated. Without a repeated phrase a
        "title" is just a 2-4-word window, which Deezer's phrase search almost never resolves (2 % in the check
        above) at the price of a request."""
        qs = self.window_plan(query_text)
        if repeated_phrase(tokens):
            qs += [("title", q, []) for q in query_phrases(tokens, self.max_phrases, self.idf)]
        seen, out = set(), []
        for kind, q, subs in qs:
            k = normalize_text(q)
            if k and k not in seen:
                seen.add(k)
                out.append((kind, q, subs))
        return out

    def propose(self, query_text: str, tokens: list[str], language: str | None) -> list[Candidate]:
        """All window and title queries at once; then, for the windows that found nothing, their retry windows
        (most informative window first) within ``max_requests``; then, if nothing at all was found and no title
        phrase was searched yet, title phrases within what is left of ``max_requests`` (fully sung titles only).
        Rounds keep the choice of retries independent of which answer happened to arrive first."""
        queries = self.queries(query_text, tokens)
        if not queries:
            return []
        out: list[Candidate] = []
        errors: list[Exception] = []
        weight = _memo(self.idf) or (lambda t: 1.0)

        def info(q: str) -> float:
            return sum(weight(t) for t in dict.fromkeys(informative_tokens(q.split())))

        def add(kind: str, tracks: list[TrackMeta]) -> None:
            for j, tm in enumerate(tracks):
                if kind == "title":
                    prior = max(title_in_query(tm.title, tokens), title_in_query(tm.title_short, tokens)) - 0.04 * j
                elif kind == "retry":
                    prior = max(0.1, 0.7 - 0.12 * j)
                else:
                    prior = max(0.2, 1.0 - 0.15 * j)
                out.append(Candidate(title=tm.title, artist=tm.artist, album=tm.album, duration_s=tm.duration_s,
                                     meta=tm, source=self.name, prior=prior))

        ex = ThreadPoolExecutor(max_workers=min(4, len(queries)), thread_name_prefix="deezer-src")
        try:
            with _source_deadline() as until:
                search = {"window": self.deezer.search_tracks, "title": self.deezer.search_title}
                futures = [_in_context(ex, search[kind], q, self.per_query, private=True) for kind, q, _ in queries]
                empty = []
                for (kind, q, subs), (tracks, err) in zip(queries, _results(futures, until)):
                    if err is not None:
                        errors.append(err)
                    elif tracks:
                        add(kind, tracks)
                    elif subs:
                        empty.append((q, subs))
                retries = [s for _, subs in sorted(empty, key=lambda x: -info(x[0])) for s in subs]
                retries = retries[:max(0, self.max_requests - len(queries))]
                futures = [_in_context(ex, self.deezer.search_tracks, q, self.per_query, private=True)
                           for q in retries]
                for tracks, err in _results(futures, until):
                    if err is not None:
                        errors.append(err)
                    else:
                        add("retry", tracks)
                if not out and not errors and all(kind != "title" for kind, _, _ in queries):
                    # last resort: the title route (a best effort: its failures are not the source's)
                    sent = {normalize_text(q) for _, q, _ in queries} | {normalize_text(q) for q in retries}
                    left = self.max_requests - len(queries) - len(retries)
                    phrases = [p for p in query_phrases(tokens, self.max_phrases, self.idf)
                               if normalize_text(p) not in sent][:max(0, left)]
                    futures = [_in_context(ex, self.deezer.search_title, p, self.per_query, private=True)
                               for p in phrases]
                    for tracks, _ in _results(futures, until):
                        add("title", [tm for tm in tracks or [] if max(title_in_query(tm.title, tokens),
                                                                       title_in_query(tm.title_short, tokens)) >= 1.0])
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        if errors and not out:
            raise errors[0]
        return out


class GeniusSource:
    """Genius API search (lyric full-text). Requires a client access token (``MELODYAI_GENIUS_TOKEN``)."""

    name, label, online, requires_key = "genius", "Genius (full lyrics, needs a token)", True, True
    URL = "https://api.genius.com/search"

    def __init__(self, web: WebClient | None, token: str | None = None, max_results: int = 6) -> None:
        self.web, self.max_results = web, max_results
        self._token = token if token is not None else os.environ.get("MELODYAI_GENIUS_TOKEN", "").strip()

    def available(self) -> bool:
        return bool(self._token) and self.web is not None and not self.web.offline

    def propose(self, query_text: str, tokens: list[str], language: str | None) -> list[Candidate]:
        q = " ".join(t for t in tokens if t not in FILLERS)[:200]
        data = self.web.get_json(self.URL, {"q": q}, ttl_s=7 * 86400, private=True,
                                 headers={"Authorization": f"Bearer {self._token}"})
        hits = ((data or {}).get("response") or {}).get("hits") or []
        out = []
        for i, h in enumerate(hits):
            res = h.get("result") if isinstance(h, dict) else None
            if not isinstance(res, dict) or h.get("type", "song") != "song":
                continue
            title = str(res.get("title") or "").strip()
            artist = str((res.get("primary_artist") or {}).get("name") or res.get("artist_names") or "").strip()
            if title and artist:
                out.append(Candidate(title=title, artist=artist, source=self.name, prior=0.9 - 0.08 * i))
            if len(out) >= self.max_results:
                break
        return out


_CLAUDE_SCHEMA = {
    "type": "object",
    "properties": {
        "songs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "artist": {"type": "string"},
                               "confidence": {"type": "number"}},
                "required": ["title", "artist", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["songs"],
    "additionalProperties": False,
}

_CLAUDE_PROMPT = """Someone sang part of a song from memory and a speech recogniser transcribed it. The words may \
be misheard, misspelled, partial or out of order, and the language may be wrong.

Transcript:
<transcript>
{text}
</transcript>

Which existing, commercially released songs have lyrics containing these words (or words that sound like them)? \
Return up to {n} candidates, most likely first, with the official song title and the main artist, and your \
confidence from 0 to 1. Only return songs you believe are real; return an empty list when unsure. Do not quote lyrics."""


class ClaudeSource:
    """Asks Claude for songs whose lyrics contain the (possibly misheard) words; candidates are verified later."""

    name, label, online, requires_key = "claude", "Claude (AI, needs an API key)", True, True

    def __init__(self, api_key: str | None = None, model: str | None = None, max_results: int = 5,
                 timeout_s: float = 20.0, client=None) -> None:
        key = api_key if api_key is not None else claude_api_key()
        self._key = key.strip()
        self.model = model or os.environ.get("MELODYAI_CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL
        self.max_results, self.timeout_s = max_results, timeout_s
        self._client = client

    def _sdk(self):
        try:
            import anthropic
        except ImportError:
            return None
        return anthropic

    def available(self) -> bool:
        return self._client is not None or (bool(self._key) and self._sdk() is not None)

    def _get_client(self):
        if self._client is None:
            anthropic = self._sdk()
            if anthropic is None:
                raise RuntimeError("the 'anthropic' package is not installed (pip install -e '.[llm]')")
            self._client = anthropic.Anthropic(api_key=self._key, timeout=self.timeout_s, max_retries=1)
        return self._client

    def propose(self, query_text: str, tokens: list[str], language: str | None) -> list[Candidate]:
        text = re.sub(r"\s+\n", "\n", (query_text or "").strip())[:1500]
        if not text:
            return []
        extra = {}
        deadline = current_deadline()
        if deadline is not None:                 # the search's deadline also bounds this request
            left = deadline - time.monotonic()
            if left <= 0.5:
                return []
            extra["timeout"] = min(self.timeout_s, left)
        client = self._get_client()
        anthropic = self._sdk()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": _CLAUDE_PROMPT.format(text=text, n=self.max_results)}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": _CLAUDE_SCHEMA}},
                **extra,
            )
        except Exception as e:
            if anthropic is not None and isinstance(e, (anthropic.RateLimitError, anthropic.APIStatusError,
                                                        anthropic.APIConnectionError)):
                raise RuntimeError(f"Claude API error: {type(e).__name__}") from None
            raise
        if getattr(response, "stop_reason", None) == "refusal":
            return []
        block = next((b for b in response.content if getattr(b, "type", "") == "text"), None)
        if block is None:
            return []
        try:
            songs = json.loads(block.text).get("songs") or []
        except (ValueError, AttributeError):
            return []
        out = []
        for s in songs[: self.max_results]:
            title, artist = str(s.get("title") or "").strip(), str(s.get("artist") or "").strip()
            if title and artist:
                conf = s.get("confidence")
                prior = float(conf) if isinstance(conf, (int, float)) else 0.5
                out.append(Candidate(title=title, artist=artist, source=self.name, prior=max(0.0, min(1.0, prior))))
        return out


def default_sources(index: WorldLyricsIndex, web: WebClient | None) -> list[Source]:
    """Local index + free online sources (LRCLIB, Deezer) + keyed sources (inactive until configured)."""
    sources: list[Source] = [LocalIndexSource(index)]
    if web is not None:
        sources += [LrclibTitleSource(LrclibClient(web), idf=index.idf),
                    DeezerTitleSource(DeezerClient(web), idf=index.idf)]
    sources += [GeniusSource(web), ClaudeSource()]
    return sources
