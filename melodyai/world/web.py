"""Polite, cached HTTP JSON client shared by the open-world sources (LRCLIB, Deezer, Genius).

Design
------
- **Identification**: every request carries ``User-Agent: MelodyAI/<version> (+project URL)`` as LRCLIB
  requires (identified clients get ~2x the anonymous allowance).
- **Rate limiting per host**: a token bucket (burst ``capacity``, steady ``refill_per_s``) plus a cap on
  concurrent requests. Tokens are *reserved* (the bucket may go negative), so waiting callers are served
  in FIFO order without a condition variable. Limiters are shared by every ``WebClient`` in the process
  (the providers' allowances are per IP) unless a client is given its own ``host_limits``.
- **Retry-After**: ``429``/``503`` responses are honoured: the host is blocked for the announced delay (all
  threads back off, not only the one that was throttled), then the request is retried once after
  ``Retry-After`` + up to 30 % jitter if that wait is <= ``max_wait_s``; otherwise ``WebError("rate
  limited")`` is raised at once, without sending anything while the block lasts.
- **Response cache** (SQLite, one row per ``url + sorted params``): zlib-compressed JSON with a TTL;
  ``404`` is cached as a negative entry (shorter TTL); the oldest rows are pruned beyond ``max_cache_mb``.
  Concurrent requests for the same key are collapsed into one network call (single flight). JSON objects
  with a top-level ``"error"`` member (Deezer's HTTP-200 error convention) are returned but never cached.
  Expired rows are deleted at every periodic prune (every ``_PRUNE_EVERY`` writes), not only over budget.
- **Offline mode** serves the cache only (stale entries included) and raises ``WebError`` on a miss.
- **Deadlines**: :func:`request_deadline` sets a deadline (``time.monotonic()``) for the requests made in
  that context (worker threads inherit it when started through ``contextvars.copy_context().run``):
  nothing is sent after it, a request never waits for a rate-limit token past it (nor longer than
  ``max_wait_s``; the token is given back) and the HTTP timeout is cut to the time left. A search that
  hit its deadline therefore stops sending, instead of leaving abandoned requests in the token bucket.

Privacy: request headers (``Authorization``, API keys) are never part of the cache key, the cache rows or
the log lines; logs show host + path and status only, never the query string (httpx's own INFO request
log, which prints full URLs, is silenced). Requests that carry the user's words (``private=True``: sung
or typed lines sent to a title / lyrics search) are cached under a SHA-256 of their key with a short TTL
(``PRIVATE_TTL_S``), are not cached at all while ``cache_private`` is False (the user does not keep a
history) and are removed by :meth:`WebClient.purge_private` (history deleted).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import random
import sqlite3
import threading
import time
import zlib
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx

from melodyai import __version__

log = logging.getLogger(__name__)
# httpx logs every request at INFO with its full URL, query string (the user's words) included.
for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

DEFAULT_USER_AGENT = f"MelodyAI/{__version__} (+https://github.com/Ale-Newport/MelodyAI)"
RETRY_STATUSES = (429, 503)
DEFAULT_RETRY_AFTER_S = {429: 2.0, 503: 1.0}   # used when the server omits Retry-After
RETRY_JITTER = 0.30                            # up to +30 % of the delay (LRCLIB docs)
PRIVATE_TTL_S = 6 * 3600                       # max cache lifetime of requests carrying the user's words
PRIVATE_PREFIX = "h:"                          # cache key of a private request: "h:" + sha256(url?params)
# Free-text searches cached by versions that stored the user's words in plain keys: purged with the history.
LEGACY_QUERY_KEYS = ("https://api.deezer.com/search?", "https://lrclib.net/api/search?",
                     "https://api.genius.com/search?")

_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar("melodyai_web_deadline", default=None)


@contextmanager
def request_deadline(deadline: float | None):
    """Requests made inside this block are bounded by ``deadline`` (``time.monotonic()`` seconds).

    Worker threads inherit it when their task is submitted as ``contextvars.copy_context().run(fn, ...)``.
    """
    token = _DEADLINE.set(deadline)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def current_deadline() -> float | None:
    return _DEADLINE.get()


class WebError(RuntimeError):
    """Network failure, HTTP status >= 400 (except 404), rate limited beyond ``max_wait_s`` or offline miss.

    ``kind`` is one of ``network | http | rate_limited | offline | decode | api | deadline``.
    """

    def __init__(self, message: str, *, kind: str = "http", status: int | None = None, host: str = "",
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.host = host
        self.retry_after = retry_after


@dataclass(frozen=True)
class HostLimit:
    capacity: float          # burst size (tokens)
    refill_per_s: float      # steady request rate
    max_concurrent: int      # simultaneous in-flight requests


HOST_LIMITS: dict[str, HostLimit] = {
    "lrclib.net": HostLimit(capacity=6, refill_per_s=1.3, max_concurrent=3),      # docs: ~90 req/min identified
    "api.deezer.com": HostLimit(capacity=10, refill_per_s=8.0, max_concurrent=6),  # Deezer: 50 req / 5 s
    "api.genius.com": HostLimit(capacity=10, refill_per_s=4.0, max_concurrent=3),
}
DEFAULT_HOST_LIMIT = HostLimit(capacity=10, refill_per_s=5.0, max_concurrent=4)


class _HostLimiter:
    """Token bucket with reservations + a concurrency semaphore + a server-imposed block (Retry-After)."""

    def __init__(self, limit: HostLimit) -> None:
        self.limit = limit
        self.slots = threading.BoundedSemaphore(max(1, int(limit.max_concurrent)))
        self._lock = threading.Lock()
        self._tokens = float(limit.capacity)
        self._last: float | None = None
        self._blocked_until = 0.0

    def reserve(self, now: float, max_block_s: float) -> float:
        """Take one token; return how long the caller must wait before sending.

        Raises ``WebError(kind="rate_limited")`` (without consuming a token) when the host is blocked by a
        Retry-After for longer than ``max_block_s``.
        """
        with self._lock:
            block = self._blocked_until - now
            if block > max_block_s:
                raise WebError("rate limited", kind="rate_limited", retry_after=round(block, 2))
            if self._last is not None:
                elapsed = max(0.0, now - self._last)
                self._tokens = min(float(self.limit.capacity), self._tokens + elapsed * self.limit.refill_per_s)
            self._last = now
            self._tokens -= 1.0
            wait = 0.0 if self._tokens >= 0 else -self._tokens / self.limit.refill_per_s
            return max(wait, block, 0.0)

    def refund(self) -> None:
        """Give back a token taken by :meth:`reserve` for a request that was not sent."""
        with self._lock:
            self._tokens = min(float(self.limit.capacity), self._tokens + 1.0)

    def block(self, now: float, seconds: float) -> None:
        with self._lock:
            self._blocked_until = max(self._blocked_until, now + seconds)


_SHARED_LIMITERS: dict[str, _HostLimiter] = {}
_SHARED_GUARD = threading.Lock()


def _shared_limiter(host: str) -> _HostLimiter:
    with _SHARED_GUARD:
        lim = _SHARED_LIMITERS.get(host)
        if lim is None:
            lim = _SHARED_LIMITERS[host] = _HostLimiter(HOST_LIMITS.get(host, DEFAULT_HOST_LIMIT))
        return lim


def parse_retry_after(value: str | None, now_epoch: float | None = None) -> float | None:
    """Seconds from a ``Retry-After`` header (delta-seconds or HTTP-date); None when absent/invalid."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    return max(0.0, when - (time.time() if now_epoch is None else now_epoch))


def cache_key(url: str, params: dict | None) -> str:
    """``url`` + sorted, non-None query params. Headers are deliberately not part of the key."""
    items = sorted((str(k), str(v)) for k, v in (params or {}).items() if v is not None)
    return url + ("?" + urlencode(items) if items else "")


def _drop(data: Any, fields: tuple[str, ...]) -> Any:
    if not fields:
        return data
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k not in fields}
    if isinstance(data, list):
        return [_drop(x, fields) if isinstance(x, dict) else x for x in data]
    return data


_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    host TEXT NOT NULL,
    status INTEGER NOT NULL,          -- 200 (body) or 404 (negative entry, body NULL)
    body BLOB,                        -- zlib-compressed JSON
    size INTEGER NOT NULL,
    fetched_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_responses_fetched ON responses(fetched_at);
"""
_PRUNE_EVERY = 200      # writes between two exact size checks


class WebClient:
    """Thread-safe JSON GET client with per-host rate limiting and a SQLite response cache."""

    def __init__(self, cache_path: str | Path | None = None, *, user_agent: str = DEFAULT_USER_AGENT,
                 transport: httpx.BaseTransport | None = None, timeout: float = 8.0, offline: bool = False,
                 max_cache_mb: float = 256.0, host_limits: dict[str, HostLimit] | None = None) -> None:
        self.user_agent = user_agent
        self.offline = offline
        self.cache_private = True       # False: requests carrying the user's words are never stored
        self.timeout = timeout
        self.max_cache_bytes = int(max_cache_mb * 1024 * 1024)
        self.cache_path = Path(cache_path) if cache_path not in (None, "", ":memory:") else None   # None: in memory
        # Injectable for tests: monotonic clock for the limiters, sleep, wall clock for cache TTLs.
        self._clock = time.monotonic
        self._sleep = time.sleep
        self._now = time.time
        self._own_limits = dict(host_limits) if host_limits is not None else None
        self._limiters: dict[str, _HostLimiter] = {}
        self._limiters_guard = threading.Lock()
        self._http = httpx.Client(
            timeout=timeout, transport=transport, follow_redirects=True,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            limits=httpx.Limits(max_connections=24, max_keepalive_connections=12),
        )
        self._db_lock = threading.RLock()
        self._db = self._open_db()
        self._inflight: dict[str, list] = {}
        self._inflight_guard = threading.Lock()
        self._stats_lock = threading.Lock()
        self._counts: Counter = Counter()
        self._host_counts: dict[str, Counter] = {}
        self._writes = 0
        self._approx_bytes = self._cache_bytes()
        if self._approx_bytes > self.max_cache_bytes:
            self.prune(expired=not offline)   # offline mode deliberately serves stale rows

    # ------------------------------------------------------------------ cache storage
    def _open_db(self) -> sqlite3.Connection:
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(self.cache_path), check_same_thread=False, timeout=15.0)
            db.execute("PRAGMA auto_vacuum=INCREMENTAL")   # only effective on a new file
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
        else:
            db = sqlite3.connect(":memory:", check_same_thread=False)
        db.executescript(_SCHEMA)
        db.commit()
        return db

    def _cache_get(self, key: str) -> tuple[int, bytes | None, float] | None:
        with self._db_lock:
            row = self._db.execute("SELECT status, body, expires_at FROM responses WHERE key = ?", (key,)).fetchone()
        return (int(row[0]), row[1], float(row[2])) if row else None

    def _cache_put(self, key: str, host: str, status: int, data: Any, ttl_s: float) -> None:
        body = zlib.compress(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 6) \
            if status == 200 else None
        size = len(key) + (len(body) if body else 0) + 64
        now = self._now()
        with self._db_lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses (key, host, status, body, size, fetched_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", (key, host, status, body, size, now, now + max(0.0, ttl_s)))
            self._db.commit()
            self._writes += 1
            self._approx_bytes += size
            due = self._writes % _PRUNE_EVERY == 0 or self._approx_bytes > self.max_cache_bytes
        if due:
            self.prune()

    def _cache_bytes(self) -> int:
        with self._db_lock:
            return int(self._db.execute("SELECT COALESCE(SUM(size), 0) FROM responses").fetchone()[0])

    def prune(self, expired: bool = True) -> int:
        """Drop expired rows (unless ``expired`` is False), then, beyond ``max_cache_mb``, the oldest ones until
        the cache is below 90 % of it."""
        removed = 0
        with self._db_lock:
            if expired:
                removed += self._db.execute("DELETE FROM responses WHERE expires_at < ?", (self._now(),)).rowcount
            total = self._cache_bytes()
            if total > self.max_cache_bytes:
                target = int(self.max_cache_bytes * 0.9)
                while total > target:
                    rows = self._db.execute(
                        "SELECT key, size FROM responses ORDER BY fetched_at ASC LIMIT 256").fetchall()
                    if not rows:
                        break
                    victims = []
                    for key, size in rows:
                        victims.append((key,))
                        total -= int(size)
                        if total <= target:
                            break
                    self._db.executemany("DELETE FROM responses WHERE key = ?", victims)
                    removed += len(victims)
            self._db.commit()
            if removed and self.cache_path is not None:
                self._db.execute("PRAGMA incremental_vacuum").fetchall()
                self._db.commit()
            self._approx_bytes = self._cache_bytes()
        if removed:
            log.debug("web cache pruned %d entries", removed)
        return removed

    def purge_private(self) -> int:
        """Delete every cached request that carried the user's words (and plain-text free-text searches cached
        by older versions). Returns the number of rows removed."""
        end = PRIVATE_PREFIX[:-1] + chr(ord(PRIVATE_PREFIX[-1]) + 1)
        with self._db_lock:
            n = self._db.execute("DELETE FROM responses WHERE key >= ? AND key < ?", (PRIVATE_PREFIX, end)).rowcount
            for prefix in LEGACY_QUERY_KEYS:
                n += self._db.execute("DELETE FROM responses WHERE substr(key, 1, ?) = ?",
                                      (len(prefix), prefix)).rowcount
            self._db.commit()
            if n and self.cache_path is not None:
                self._db.execute("PRAGMA incremental_vacuum").fetchall()
                self._db.commit()
            self._approx_bytes = self._cache_bytes()
        return n

    def clear_cache(self) -> None:
        with self._db_lock:
            self._db.execute("DELETE FROM responses")
            self._db.commit()
            self._approx_bytes = 0

    # ------------------------------------------------------------------ bookkeeping
    def _count(self, host: str, what: str, n: int = 1) -> None:
        with self._stats_lock:
            self._counts[what] += n
            self._host_counts.setdefault(host, Counter())[what] += n

    def stats(self) -> dict:
        """Counters since creation: requests (network), cache_hits (incl. negative_hits), errors, retries, per host."""
        with self._stats_lock:
            out = {k: int(self._counts.get(k, 0)) for k in
                   ("requests", "cache_hits", "negative_hits", "errors", "retries", "rate_limited")}
            out["hosts"] = {h: dict(c) for h, c in sorted(self._host_counts.items())}
        with self._db_lock:
            n = int(self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0])
        out["cache"] = {"entries": n, "bytes": self._cache_bytes(), "max_bytes": self.max_cache_bytes,
                        "path": str(self.cache_path) if self.cache_path else None}
        out["offline"] = self.offline
        return out

    def _limiter(self, host: str) -> _HostLimiter:
        if self._own_limits is None:
            return _shared_limiter(host)
        with self._limiters_guard:
            lim = self._limiters.get(host)
            if lim is None:
                lim = self._limiters[host] = _HostLimiter(self._own_limits.get(host, DEFAULT_HOST_LIMIT))
            return lim

    @contextmanager
    def _single_flight(self, key: str):
        with self._inflight_guard:
            entry = self._inflight.get(key)
            if entry is None:
                entry = self._inflight[key] = [threading.Lock(), 0]
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._inflight_guard:
                entry[1] -= 1
                if entry[1] == 0:
                    self._inflight.pop(key, None)

    # ------------------------------------------------------------------ requests
    def get_json(self, url: str, params: dict | None = None, *, ttl_s: float = 7 * 86400,
                 negative_ttl_s: float = 86400, max_wait_s: float = 4.0, headers: dict | None = None,
                 cache: bool = True, drop_fields: tuple[str, ...] = (), private: bool = False) -> Any | None:
        """GET ``url`` and return the parsed JSON, or None on 404 (negative-cached for ``negative_ttl_s``).

        ``drop_fields`` removes bulky members from a JSON object (or from each object of a JSON array)
        before it is returned and cached. ``headers`` (e.g. ``Authorization``) are sent but never stored.
        ``private``: the request carries the user's words (see the module docstring).
        """
        params = {k: v for k, v in (params or {}).items() if v is not None}
        key = cache_key(url, params)
        host = urlsplit(url).hostname or ""
        if private:
            key = PRIVATE_PREFIX + hashlib.sha256(key.encode("utf-8")).hexdigest()
            ttl_s, negative_ttl_s = min(ttl_s, PRIVATE_TTL_S), min(negative_ttl_s, PRIVATE_TTL_S)
            cache = cache and self.cache_private
        if not cache:
            if self.offline:
                raise WebError("offline: response not cached", kind="offline", host=host)
            return self._fetch(url, params, host, key, ttl_s, negative_ttl_s, max_wait_s, headers, False,
                               drop_fields)
        hit = self._lookup(key, host)
        if hit is not None:
            return hit[0]
        with self._single_flight(key):
            hit = self._lookup(key, host)   # another thread may have fetched it meanwhile
            if hit is not None:
                return hit[0]
            if self.offline:
                raise WebError("offline: response not cached", kind="offline", host=host)
            return self._fetch(url, params, host, key, ttl_s, negative_ttl_s, max_wait_s, headers, True,
                               drop_fields)

    def _lookup(self, key: str, host: str) -> tuple[Any] | None:
        """``(value,)`` on a usable cache entry (offline mode also accepts expired ones), else None."""
        row = self._cache_get(key)
        if row is None:
            return None
        status, body, expires_at = row
        if expires_at < self._now() and not self.offline:
            return None
        if status == 404:
            self._count(host, "cache_hits")
            self._count(host, "negative_hits")
            return (None,)
        try:
            value = json.loads(zlib.decompress(body).decode("utf-8"))
        except (zlib.error, ValueError, TypeError):
            return None   # corrupt row: refetch (and overwrite)
        self._count(host, "cache_hits")
        return (value,)

    def _fetch(self, url: str, params: dict, host: str, key: str, ttl_s: float, negative_ttl_s: float,
               max_wait_s: float, headers: dict | None, store: bool, drop_fields: tuple[str, ...]) -> Any | None:
        limiter = self._limiter(host)
        path = urlsplit(url).path
        deadline = _DEADLINE.get()
        for attempt in range(2):
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0.05:
                self._count(host, "errors")
                raise WebError(f"deadline reached, {host}{path} not requested", kind="deadline", host=host)
            try:
                wait = limiter.reserve(self._clock(), max_wait_s)
            except WebError as e:
                e.host = host
                self._count(host, "rate_limited")
                self._count(host, "errors")
                raise
            if wait > max_wait_s or (left is not None and wait >= left - 0.05):
                limiter.refund()            # not sent: the token goes back to the bucket
                self._count(host, "rate_limited")
                self._count(host, "errors")
                raise WebError("rate limited (local request budget)", kind="rate_limited", host=host,
                               retry_after=round(wait, 2))
            if wait > 0:
                self._sleep(wait)
            timeout = self.timeout if left is None else max(0.1, min(self.timeout, left - wait))
            if not limiter.slots.acquire(timeout=None if left is None else max(0.0, left - wait)):
                self._count(host, "errors")
                raise WebError(f"deadline reached, {host}{path} not requested", kind="deadline", host=host)
            try:
                t0 = time.perf_counter()
                try:
                    resp = self._http.get(url, params=params, headers=headers, timeout=timeout)
                except httpx.HTTPError as e:
                    self._count(host, "requests")
                    self._count(host, "errors")
                    log.debug("GET %s%s failed: %s", host, path, type(e).__name__)
                    raise WebError(f"network error ({type(e).__name__}) for {host}{path}", kind="network",
                                   host=host) from None
            finally:
                limiter.slots.release()
            self._count(host, "requests")
            log.debug("GET %s%s -> %d (%.0f ms)", host, path, resp.status_code, 1000 * (time.perf_counter() - t0))
            status = resp.status_code
            if status in RETRY_STATUSES:
                delay = parse_retry_after(resp.headers.get("Retry-After"), self._now())
                delay = DEFAULT_RETRY_AFTER_S[status] if delay is None else delay
                limiter.block(self._clock(), delay)
                self._count(host, "rate_limited")
                in_time = deadline is None or delay < deadline - time.monotonic() - 0.1
                if attempt == 0 and delay <= max_wait_s and in_time:
                    self._count(host, "retries")
                    self._sleep(delay * (1.0 + random.uniform(0.0, RETRY_JITTER)))
                    continue
                self._count(host, "errors")
                raise WebError("rate limited", kind="rate_limited", status=status, host=host,
                               retry_after=round(delay, 2))
            if status == 404:
                if store:
                    self._cache_put(key, host, 404, None, negative_ttl_s)
                return None
            if status >= 400:
                self._count(host, "errors")
                raise WebError(f"HTTP {status} from {host}{path}", kind="http", status=status, host=host)
            try:
                data = resp.json()
            except ValueError:
                self._count(host, "errors")
                raise WebError(f"invalid JSON from {host}{path}", kind="decode", status=status, host=host) from None
            data = _drop(data, drop_fields)
            if store and not (isinstance(data, dict) and data.get("error")):
                self._cache_put(key, host, 200, data, ttl_s)
            return data
        raise AssertionError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self._http.close()
        with self._db_lock:
            self._db.close()

    def __enter__(self) -> WebClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
