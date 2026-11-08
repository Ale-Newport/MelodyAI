"""Scalable local index of real-world song lyrics, searched by (misheard, partial) sung words.

Storage (one SQLite file, WAL mode, safe to share between threads):

* ``songs`` -- one row per song identity :func:`melodyai.world.text.song_key` (normalised title +
  primary artist), with metadata (Deezer rank = popularity, links, cover...), the lyrics text and its
  normalised lines (zlib, so queries never re-normalise lyrics). Lyrics are read only for the few
  hundred retrieved candidates of a query, never all at once.
* ``fts_lines`` -- contentless FTS5 table (``unicode61 remove_diacritics 2``) with one row per
  distinct lyric line: the normalised words (``norm``) and the phonetic keys of its informative
  words (``phon``, see :func:`melodyai.lyrics.align.phonetic`). Row ids encode the song:
  ``rowid = song_rid << 12 | line``, so a song's lines are deleted with one rowid range.
* ``df(term, n)`` -- song-level document frequency of every word (IDF) and, as ``~key``, of every
  phonetic key (retrieval cost control); maintained on upsert.
* ``meta(k, v)`` -- small key/value store (sync state, schema version).

Search (``search``):

1. *Retrieval*: one FTS5 ``MATCH`` (see :meth:`WorldLyricsIndex._fts_query`): an OR over the query's
   rare informative words (``norm``), their phonetic keys (``phon``, BM25 weight 0.7) and prefixes of
   unknown words, chosen rarest first within a song-frequency budget so very common words never
   force BM25 over a large part of the corpus; common words take part as AND-pairs of phonetic keys
   of nearby query words, and bigram phrases around the rare words lift the very line that was sung.
   Line BM25 is aggregated per song as the sum of its best 3 lines; the best ``max_candidates`` songs
   are kept (an optional ``language`` only demotes songs known to be in another language).
2. *Verification*: :func:`melodyai.lyrics.align.align_lyrics` with ``phonetic=True`` and
   ``chunked=True`` on each candidate (one vocabulary-level similarity matrix for all of them), using
   this corpus' IDF. Results are sorted by alignment score; popularity only breaks exact ties.

Size cap: a lyrics record is stored and aligned only up to ``MAX_LYRIC_TOKENS`` words (no real song comes
close; full-album or spoken-word records on LRCLIB do), because alignment memory grows with
query words x lyric words.

Concurrency: :meth:`WorldLyricsIndex.upsert_many` reads and writes inside one ``BEGIN IMMEDIATE``
transaction, so two processes (a CLI sync and the server) upserting the same song never race between the
"is it there?" read and the insert.

IDF uses the song-level ``df`` table blended with a prior (``(n * idf_corpus + 100 * prior) / (n + 100)``)
so that it is sensible for an empty or small corpus too: the prior gives function words of any
supported language a low weight and longer words a slightly higher one. Function words are also
capped at a low IDF in large corpora (a Portuguese "voce" is rare in an English-heavy corpus but
still carries no information).
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import threading
import time
import zlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

from melodyai.lyrics.align import (
    AlignResult,
    align_lyrics,
    doc_tokens,
    phonetic,
    query_segments,
    similarity_matrix,
    token_weights,
)
from melodyai.lyrics.normalize import informative_tokens, is_stopword, normalize_text, tokenize
from melodyai.world.text import guess_language, is_common_word, normalize_title, song_key

INDEX_VERSION = "1"
LINE_BITS = 12                     # fts rowid = song rid << LINE_BITS | line number
MAX_LINES = (1 << LINE_BITS) - 1
MAX_LYRIC_TOKENS = 4000            # words of a lyrics record that are stored / aligned
TOP_LINES = 3                      # retrieval score of a song = sum of its best TOP_LINES line scores
PHON_WEIGHT = 0.7                  # BM25 weight of the phonetic column
PHON_PREFIX = "~"                  # df rows of phonetic keys (words never contain "~")
MAX_TERMS = 24                     # single words + phonetic keys + prefixes per query
MAX_PAIRS = 16
MAX_PHRASES = 24
MAX_PAIR_COOC = 0.1                # skip AND-pairs of two keys that are both in most songs
LANGUAGE_PENALTY = 0.6             # retrieval score factor for songs known to be in another language
PRIOR_STRENGTH = 100.0             # pseudo-songs of the IDF prior
COMMON_IDF_CAP = 2.5               # max IDF of a function word
_SECTION_LINE = re.compile(          # "[Chorus]", "(x2)", "(Coro)": annotations, not sung words
    r"^\s*(?:\[[^\]]*\]|\((?:x\s*\d+|\d+\s*x|chorus|coro|estribillo|refrain|repeat|repite|bis|hook|bridge|"
    r"puente|intro|outro|pre-?chorus|verse[^)]*|verso[^)]*)\))\s*$", re.IGNORECASE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS songs(
    rid INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    album TEXT,
    duration_s REAL,
    language TEXT,
    lyrics TEXT NOT NULL DEFAULT '',
    norm BLOB,
    rank INTEGER NOT NULL DEFAULT 0,
    deezer_id INTEGER,
    isrc TEXT,
    lrclib_id INTEGER,
    cover_url TEXT,
    link TEXT,
    preview_url TEXT,
    source TEXT NOT NULL DEFAULT '',
    added_at REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_songs_rank ON songs(rank DESC);
CREATE TABLE IF NOT EXISTS df(term TEXT PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT) WITHOUT ROWID;
"""


@dataclass
class WorldSong:
    id: str                      # "w" + first 15 hex chars of sha1(song_key); filled in by upsert
    title: str
    artist: str
    album: str | None = None
    duration_s: float | None = None
    language: str | None = None
    lyrics: str = ""
    rank: int = 0                # Deezer rank (popularity), 0 = unknown
    deezer_id: int | None = None
    isrc: str | None = None
    lrclib_id: int | None = None
    cover_url: str | None = None
    link: str | None = None
    preview_url: str | None = None
    source: str = ""             # "sync:<preset>" | "lookup" | ...
    added_at: float = 0.0

    @classmethod
    def create(cls, title: str, artist: str, **kw) -> WorldSong:
        """New song with its canonical id."""
        return cls(world_song_id(title, artist), title, artist, **kw)

    @property
    def key(self) -> str:
        return song_key(self.title, self.artist)

    def to_meta(self) -> dict:
        d = asdict(self)
        d.pop("lyrics")
        return d


@dataclass
class WorldMatch:
    song: WorldSong
    align: AlignResult

    def to_dict(self) -> dict:
        return {"song": self.song.to_meta(), **self.align.to_dict()}


_FIELDS = [f.name for f in fields(WorldSong)]
_META_FIELDS = [f for f in _FIELDS if f != "lyrics"]


def world_song_id(title: str, artist: str) -> str:
    return "w" + hashlib.sha1(song_key(title, artist).encode("utf-8")).hexdigest()[:15]


def lyric_lines(lyrics: str) -> list[str]:
    """Non-empty lyric lines, without section markers such as "[Chorus]"."""
    return [ln.strip() for ln in (lyrics or "").splitlines() if ln.strip() and not _SECTION_LINE.match(ln)]


def cap_lines(lines: list[str], norm_lines: list[str] | None = None,
              max_tokens: int = MAX_LYRIC_TOKENS) -> int:
    """How many leading lines fit in ``max_tokens`` words (at least one line)."""
    total = 0
    for i, ln in enumerate(lines):
        total += len((norm_lines[i] if norm_lines is not None else normalize_text(ln)).split())
        if total > max_tokens:
            return max(1, i)
    return len(lines)


def _pack(norm_lines: list[str]) -> bytes:
    return zlib.compress("\n".join(norm_lines).encode("utf-8"), 6)


def _unpack(blob: bytes | None, lines: list[str]) -> list[str]:
    """Normalised lines stored with a song (recomputed if missing or inconsistent)."""
    if not lines:
        return []
    if blob:
        out = zlib.decompress(blob).decode("utf-8").split("\n")
        if len(out) == len(lines):
            return out
    return [normalize_text(ln) for ln in lines]


def _df_terms(vocab: set[str]) -> set[str]:
    """Terms counted in ``df``: the words of a song and, prefixed, the phonetic keys of its informative words."""
    return vocab | {PHON_PREFIX + phonetic(t) for t in vocab if not is_stopword(t) and len(t) > 1}


def prior_idf(token: str) -> float:
    """Corpus-free IDF guess: function words low, longer words slightly higher."""
    if is_common_word(token):
        return 1.0
    if len(token) <= 2:
        return 1.5
    return 3.0 + 0.2 * min(len(token) - 3, 6)


def _empty(v) -> bool:
    return v is None or v == ""


class WorldLyricsIndex:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:" and not self.path.startswith("file:"):
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._n = -1                       # cached song count, see _count()
        self._data_version = -1
        self.conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        with self._lock:
            if self.path != ":memory:":
                self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.executescript(_SCHEMA)
            contentless = ", content='', contentless_delete=1" if sqlite3.sqlite_version_info >= (3, 43, 0) else ""
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS fts_lines USING fts5(norm, phon, "
                f"tokenize='unicode61 remove_diacritics 2'{contentless})")
            if self.meta_get("index_version") is None:
                self.meta_set("index_version", INDEX_VERSION)
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------------ writing
    def upsert(self, song: WorldSong) -> bool:
        """Insert a song, or update the song with the same :func:`song_key`. True if it was new.

        On conflict the lyrics are replaced when the new ones are non-empty, the rank becomes the
        maximum of both, metadata of the more popular version wins (missing fields are filled from
        the other one) and the original ``source`` / ``added_at`` are kept. The ``id`` is always
        derived from the key. Raises ``ValueError`` for a song without a title.
        """
        return self.upsert_many([song]) == 1

    def upsert_many(self, songs: Iterable[WorldSong]) -> int:
        """:meth:`upsert` many songs in one transaction (much faster for bulk loads). Returns # new.

        The transaction takes the write lock *before* reading (``BEGIN IMMEDIATE``): another process
        upserting the same song waits (busy timeout) instead of racing into a UNIQUE error or a lost update.
        """
        with self._lock:
            try:
                with self.conn:
                    if not self.conn.in_transaction:
                        self.conn.execute("BEGIN IMMEDIATE")
                    return sum(self._upsert(s) for s in songs)
            except BaseException:
                self._n = -1                   # rolled back: recount on next use
                raise

    def _upsert(self, song: WorldSong) -> bool:
        if not normalize_title(song.title or ""):
            raise ValueError(f"song without a title: {song.title!r} / {song.artist!r}")
        key = song_key(song.title, song.artist)
        lines = lyric_lines(song.lyrics)
        norm_lines = [normalize_text(ln) for ln in lines]
        keep = cap_lines(lines, norm_lines)
        lines, norm_lines = lines[:keep], norm_lines[:keep]
        lyrics = "\n".join(lines)
        now = time.time()
        row = self.conn.execute(f"SELECT rid, norm, {', '.join(_FIELDS)} FROM songs WHERE key = ?",
                                (key,)).fetchone()
        if row is None:
            new = replace(song, id="w" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:15], lyrics=lyrics,
                          language=song.language or guess_language(lyrics), rank=int(song.rank or 0),
                          added_at=song.added_at or now)
            cur = self.conn.execute(
                f"INSERT INTO songs(key, norm, updated_at, {', '.join(_FIELDS)}) "
                f"VALUES (?, ?, ?, {', '.join('?' * len(_FIELDS))})",
                (key, _pack(norm_lines), now, *[getattr(new, f) for f in _FIELDS]))
            self._index_lines(int(cur.lastrowid), norm_lines)
            if self._n >= 0:
                self._n += 1
            return True
        rid, old_blob, old = int(row[0]), row[1], WorldSong(*row[2:])
        prefer_new = int(song.rank or 0) >= old.rank
        merged = {}
        for f in _META_FIELDS:
            o, n = getattr(old, f), getattr(song, f)
            merged[f] = (o if _empty(n) else n) if prefer_new else (n if _empty(o) else o)
        merged.update(id=old.id, rank=max(old.rank, int(song.rank or 0)), source=old.source or song.source,
                      added_at=old.added_at or song.added_at or now)
        relyric = bool(lyrics) and lyrics != old.lyrics
        merged["lyrics"] = lyrics if relyric else old.lyrics
        if relyric and _empty(song.language):
            merged["language"] = guess_language(lyrics) or old.language
        self.conn.execute(f"UPDATE songs SET updated_at = ?, {', '.join(f + ' = ?' for f in _FIELDS)} WHERE rid = ?",
                          (now, *[merged[f] for f in _FIELDS], rid))
        if relyric:
            self._unindex_lines(rid, _unpack(old_blob, lyric_lines(old.lyrics)))
            self.conn.execute("UPDATE songs SET norm = ? WHERE rid = ?", (_pack(norm_lines), rid))
            self._index_lines(rid, norm_lines)
        return False

    def delete_source(self, source: str) -> int:
        """Remove every song whose ``source`` is ``source`` (e.g. ``"lookup"``: learned from recognitions)."""
        with self._lock:
            try:
                with self.conn:
                    if not self.conn.in_transaction:
                        self.conn.execute("BEGIN IMMEDIATE")
                    rows = self.conn.execute("SELECT rid, norm, lyrics FROM songs WHERE source = ?",
                                             (source,)).fetchall()
                    for rid, blob, lyrics in rows:
                        self._unindex_lines(int(rid), _unpack(blob, lyric_lines(lyrics)))
                        self.conn.execute("DELETE FROM songs WHERE rid = ?", (rid,))
            finally:
                self._n = -1
        return len(rows)

    def _index_lines(self, rid: int, norm_lines: list[str]) -> None:
        rows, seen, vocab = [], set(), set()
        for i, norm in enumerate(norm_lines[:MAX_LINES]):
            toks = norm.split()
            if not toks:
                continue
            vocab.update(toks)
            if norm in seen:                       # repeated chorus lines are indexed once
                continue
            seen.add(norm)
            phon = " ".join(dict.fromkeys(phonetic(t) for t in toks if not is_stopword(t) and len(t) > 1))
            rows.append(((rid << LINE_BITS) | i, norm, phon))
        vocab.update(t for norm in norm_lines[MAX_LINES:] for t in norm.split())
        self.conn.executemany("INSERT INTO fts_lines(rowid, norm, phon) VALUES (?, ?, ?)", rows)
        self.conn.executemany("INSERT INTO df(term, n) VALUES (?, 1) ON CONFLICT(term) DO UPDATE SET n = n + 1",
                              [(t,) for t in _df_terms(vocab)])

    def _unindex_lines(self, rid: int, norm_lines: list[str]) -> None:
        self.conn.execute("DELETE FROM fts_lines WHERE rowid BETWEEN ? AND ?",
                          (rid << LINE_BITS, (rid << LINE_BITS) | MAX_LINES))
        terms = [(t,) for t in _df_terms({t for norm in norm_lines for t in norm.split()})]
        self.conn.executemany("UPDATE df SET n = n - 1 WHERE term = ?", terms)
        self.conn.executemany("DELETE FROM df WHERE term = ? AND n <= 0", terms)

    # ------------------------------------------------------------------ reading
    def get(self, song_id: str) -> WorldSong | None:
        sid = song_id.removeprefix("world:")
        with self._lock:
            row = self.conn.execute(f"SELECT {', '.join(_FIELDS)} FROM songs WHERE id = ?", (sid,)).fetchone()
        return WorldSong(*row) if row else None

    def get_by_key(self, title: str, artist: str) -> WorldSong | None:
        with self._lock:
            row = self.conn.execute(f"SELECT {', '.join(_FIELDS)} FROM songs WHERE key = ?",
                                    (song_key(title, artist),)).fetchone()
        return WorldSong(*row) if row else None

    def has_key(self, title: str, artist: str) -> bool:
        with self._lock:
            return self.conn.execute("SELECT 1 FROM songs WHERE key = ?",
                                     (song_key(title, artist),)).fetchone() is not None

    def _count(self) -> int:
        """Number of songs (cached; recounted when another connection committed)."""
        dv = self.conn.execute("PRAGMA data_version").fetchone()[0]
        if self._n < 0 or dv != self._data_version:
            self._n = int(self.conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0])
            self._data_version = dv
        return self._n

    def __len__(self) -> int:
        with self._lock:
            return self._count()

    def stats(self) -> dict:
        with self._lock:
            langs = dict(self.conn.execute(
                "SELECT COALESCE(language, 'unknown'), COUNT(*) FROM songs GROUP BY 1 ORDER BY 2 DESC").fetchall())
            sources = dict(self.conn.execute(
                "SELECT source, COUNT(*) FROM songs GROUP BY 1 ORDER BY 2 DESC").fetchall())
            page = self.conn.execute("PRAGMA page_size").fetchone()[0]
            pages = self.conn.execute("PRAGMA page_count").fetchone()[0]
            return {"songs": self._count(), "languages": langs, "sources": sources, "bytes": int(page * pages)}

    def list_songs(self, q: str | None = None, offset: int = 0, limit: int = 50) -> list[dict]:
        """Song metadata, most popular first; ``q`` keeps songs whose title/artist contain all its words."""
        words = tokenize(q or "")[:8]
        where = " AND ".join("key LIKE ?" for _ in words)
        sql = (f"SELECT {', '.join(_FIELDS)} FROM songs {'WHERE ' + where if where else ''} "
               "ORDER BY rank DESC, title COLLATE NOCASE, rid LIMIT ? OFFSET ?")
        with self._lock:
            rows = self.conn.execute(sql, (*[f"%{w}%" for w in words], max(0, min(int(limit), 10_000)),
                                           max(0, min(int(offset), 1 << 62)))).fetchall()
        return [WorldSong(*r).to_meta() for r in rows]

    def meta_get(self, key: str) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def meta_set(self, key: str, value: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("INSERT INTO meta(k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                              (key, value))

    # ------------------------------------------------------------------ IDF
    def _df_of(self, terms: Iterable[str]) -> dict[str, int]:
        terms = list(dict.fromkeys(terms))
        out: dict[str, int] = {}
        for k in range(0, len(terms), 500):
            chunk = terms[k:k + 500]
            out.update(self.conn.execute(
                f"SELECT term, n FROM df WHERE term IN ({', '.join('?' * len(chunk))})", chunk).fetchall())
        return out

    @staticmethod
    def _idf_value(n: int, df: int, token: str) -> float:
        prior = prior_idf(token)
        if n <= 0:
            return prior
        v = math.log((n + 1) / (df + 0.5)) + 1.0
        if is_common_word(token):
            v = min(v, COMMON_IDF_CAP)
        return (n * v + PRIOR_STRENGTH * prior) / (n + PRIOR_STRENGTH)

    def idf(self, token: str) -> float:
        with self._lock:
            n = self._count()
            return self._idf_value(n, self._df_of([token]).get(token, 0), token)

    def _weights(self, q_tokens: list[str]) -> np.ndarray:
        with self._lock:
            n = self._count()
            df = self._df_of(q_tokens)
        idf = {t: self._idf_value(n, df.get(t, 0), t) for t in set(q_tokens)}
        return token_weights(q_tokens, idf.__getitem__)

    # ------------------------------------------------------------------ search
    def _prefix_df(self, prefix: str) -> int:
        hi = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        return int(self.conn.execute("SELECT COALESCE(SUM(n), 0) FROM df WHERE term >= ? AND term < ?",
                                     (prefix, hi)).fetchone()[0])

    def _fts_query(self, q_tokens: list[str], n: int) -> str | None:
        """FTS5 query for the candidate lines of a query (rare evidence alone, common words in pairs).

        Ranking every line that contains a very common word costs ~1 us per line (BM25 + sort), so
        three kinds of single terms are added rarest first while their summed song frequency stays
        within ``max(2000, 0.15 n)``: words (``norm``), phonetic keys (``phon``; finds misheard
        spellings) and, for words absent from the corpus, a 4-letter prefix (finds other typos).
        Words that are not covered take part in ``("a" AND "b")`` pairs of phonetic keys with a nearby
        query word (next and next-but-one; pairs of two keys that are both in most songs are skipped):
        they match far fewer lines, find a query made only of common words and lift the line where a
        rare word sits next to a common one. Bigram phrases around the rare words (stopwords
        included) lift the very line that was sung; when the query has fewer than two rare words,
        phrases of its other informative bigrams are added too.
        """
        seq = [t for t in q_tokens if len(t) > 1 and not is_stopword(t) and not is_common_word(t)]
        if not seq:
            seq = [t for t in q_tokens if len(t) > 1 and not is_stopword(t)]
        if not seq:
            return None
        key = {t: phonetic(t) for t in seq}
        df = self._df_of(seq + [PHON_PREFIX + k for k in key.values()])
        dfw = lambda t: df.get(t, 0)
        dfk = lambda k: df.get(PHON_PREFIX + k, 0)
        items = [(dfw(t), 0, t) for t in dict.fromkeys(seq)]
        items += [(dfk(k), 1, k) for k in dict.fromkeys(key.values()) if len(k) > 1]
        items += [(self._prefix_df(t[:4]), 2, t[:4]) for t in dict.fromkeys(seq) if dfw(t) == 0 and len(t) >= 6]
        budget, used, chosen = max(2000.0, 0.15 * n), 0, ([], [], [])
        for d, kind, t in sorted(items):            # rarest first
            if sum(map(len, chosen)) >= MAX_TERMS or used + d > budget:
                break
            if d > 0 or kind < 2:
                chosen[kind].append(t)
                used += d
        words, pkeys, prefixes = chosen
        single = set(words)
        covered = single | {t for t in seq if key[t] in set(pkeys)}
        pairs: dict[tuple[str, str], float] = {}
        for i, a in enumerate(seq):
            for b in seq[i + 1:i + 3]:
                ka, kb = key[a], key[b]
                if ka == kb or (a in covered and b in covered) or len(ka) < 2 or len(kb) < 2:
                    continue
                cooc = dfk(ka) * dfk(kb) / max(n, 1) ** 2      # rough share of songs with both keys
                if cooc <= MAX_PAIR_COOC or n < 1000:
                    pairs.setdefault((ka, kb) if ka < kb else (kb, ka), cooc)
        pairs_q = [f'("{a}" AND "{b}")' for (a, b), _ in sorted(pairs.items(), key=lambda kv: (kv[1], kv[0]))]
        if not (words or pkeys or prefixes or pairs_q):
            words = [min(seq, key=lambda t: (dfw(t), t))]
            single = set(words)
        near, other = [], []
        for a, b in zip(q_tokens, q_tokens[1:]):
            if a == b:
                continue
            if a in single or b in single:
                near.append(f'"{a} {b}"')
            elif not (is_stopword(a) or is_stopword(b)) and len(single) < 2:
                other.append((dfw(a) + dfw(b), f'"{a} {b}"'))
        phrases = list(dict.fromkeys(near[:MAX_PHRASES] + [p for _, p in sorted(other)[:MAX_PHRASES // 3]]))
        norm_q = [f'"{t}"' for t in words] + [f'"{t}" *' for t in prefixes] + phrases
        phon_q = [f'"{k}"' for k in pkeys] + pairs_q[:MAX_PAIRS]
        parts = [f"{col} : ({' OR '.join(qs)})" for col, qs in (("norm", norm_q), ("phon", phon_q)) if qs]
        return " OR ".join(parts)

    def _retrieve(self, q_tokens: list[str], language: str | None, max_candidates: int) -> list[int]:
        q = self._fts_query(q_tokens, self._count())
        if q is None:
            return []
        rows = self.conn.execute(
            f"SELECT rowid, bm25(fts_lines, 1.0, {PHON_WEIGHT}) FROM fts_lines WHERE fts_lines MATCH ? "
            "ORDER BY 2 LIMIT ?", (q, 25 * max_candidates)).fetchall()
        agg: dict[int, list] = {}
        for rowid, b in rows:                       # best lines first
            e = agg.setdefault(rowid >> LINE_BITS, [0, 0.0])
            if e[0] < TOP_LINES:
                e[0] += 1
                e[1] -= b
        ranked = sorted(agg.items(), key=lambda kv: (-kv[1][1], kv[0]))
        if language and ranked:
            top = ranked[:2 * max_candidates]
            langs = dict(self.conn.execute(
                f"SELECT rid, language FROM songs WHERE rid IN ({', '.join('?' * len(top))})",
                [r for r, _ in top]).fetchall())
            ranked = sorted(top, key=lambda kv: (
                -kv[1][1] * (LANGUAGE_PENALTY if langs.get(kv[0]) not in (None, language) else 1.0), kv[0]))
        return [rid for rid, _ in ranked[:max_candidates]]

    def search(self, query: str, top_k: int = 10, language: str | None = None,
               max_candidates: int = 150) -> list[WorldMatch]:
        """Songs whose lyrics contain the query's words (fuzzy, order-aware), best alignment first."""
        q_tokens, segments = query_segments(query)
        if len(q_tokens) < 2 or not informative_tokens(q_tokens):
            return []
        with self._lock:
            rids = self._retrieve(q_tokens, language, max(1, int(max_candidates)))
            if not rids:
                return []
            rows = self.conn.execute(f"SELECT norm, {', '.join(_FIELDS)} FROM songs "
                                     f"WHERE rid IN ({', '.join('?' * len(rids))})", rids).fetchall()
        weights = self._weights(q_tokens)
        docs, vocab = [], {}
        for row in rows:
            song = WorldSong(*row[1:])
            lines = lyric_lines(song.lyrics)
            norm_lines = _unpack(row[0], lines)
            toks, token_line = [], []
            for i, norm in enumerate(norm_lines):
                for t in norm.split():
                    toks.append(vocab.setdefault(t, len(vocab)))
                    token_line.append(i)
                if len(toks) >= MAX_LYRIC_TOKENS:          # oversized record (stored before the cap)
                    del toks[MAX_LYRIC_TOKENS:], token_line[MAX_LYRIC_TOKENS:]
                    break
            if toks:
                docs.append((song, lines, norm_lines, token_line, np.array(toks, dtype=np.int64)))
        if not docs:
            return []
        words = list(vocab)
        sim = similarity_matrix(q_tokens, words, phonetic_keys=True)
        out = []
        for song, lines, norm_lines, token_line, idx in docs:
            r = align_lyrics(q_tokens, weights, idx, token_line, lines, phonetic=True, chunked=True,
                             segments=segments, sim=sim[:, idx], norm_lines=norm_lines)
            if r is not None:
                out.append(WorldMatch(song, r))
        out.sort(key=lambda m: (-m.align.score, -m.song.rank, m.song.id))
        return out[:top_k]

    def score_lyrics(self, query: str, lyrics: str) -> AlignResult | None:
        """Verify arbitrary lyrics (e.g. fetched from LRCLIB) against a query, with this corpus' IDF."""
        q_tokens, segments = query_segments(query)
        if len(q_tokens) < 2 or not informative_tokens(q_tokens):
            return None
        lines = lyric_lines(lyrics)
        lines = lines[:cap_lines(lines)]
        toks, token_line = doc_tokens(lines)
        if not toks:
            return None
        if len(toks) > MAX_LYRIC_TOKENS:                   # a single huge line
            toks, token_line = toks[:MAX_LYRIC_TOKENS], token_line[:MAX_LYRIC_TOKENS]
        return align_lyrics(q_tokens, self._weights(q_tokens), toks, token_line, lines,
                            phonetic=True, chunked=True, segments=segments)
