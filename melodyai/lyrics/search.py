"""Approximate lyrics search.

Pipeline
--------
1. **Candidate retrieval** (scales to large catalogs) with SQLite FTS5:
   - word index (unicode61 tokenizer) ranked by BM25 for correctly transcribed words;
   - trigram index for misspelled / mis-transcribed words (substring matches).
   For small catalogs (<= ``exhaustive_limit`` songs) all songs are scored.
2. **Fuzzy local alignment** of the query tokens against each candidate's token
   sequence (:func:`melodyai.lyrics.align.align_lyrics`, shared with the open-world index):
   Smith-Waterman where a token pair scores by RapidFuzz similarity,
   weighted by the IDF of the query token (rare words matter more than "love" or "the").
   This tolerates substituted, missing and extra words while requiring the matched words
   to appear *in order and close together* -- the key protection against two songs
   matching merely because they share a topic.
3. Score = aligned weight / total query weight (coverage in [0, 1]), blended with a
   character-level partial-ratio similarity for robustness to word segmentation errors.

Semantic embeddings are intentionally not used (see docs/ALGORITHMS.md).
"""

from __future__ import annotations

import math
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field

from melodyai.lyrics.align import align_lyrics, doc_tokens, token_weights
from melodyai.lyrics.normalize import informative_tokens, is_stopword, normalize_text, tokenize


@dataclass
class LyricsCandidate:
    song_id: str
    score: float
    coverage: float
    char_similarity: float
    matched_text: str
    line_range: tuple[int, int]
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "song_id": self.song_id, "score": round(self.score, 4), "coverage": round(self.coverage, 4),
            "char_similarity": round(self.char_similarity, 4), "matched_text": self.matched_text,
            "line_range": list(self.line_range), "details": self.details,
        }


@dataclass
class _Doc:
    song_id: str
    language: str
    lines: list[str]
    tokens: list[str]
    token_line: list[int]
    norm_text: str


class LyricsIndex:
    def __init__(self, path: str = ":memory:", exhaustive_limit: int = 500) -> None:
        self.path = path
        self.exhaustive_limit = exhaustive_limit
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS lyric_lines(song_id TEXT, line_no INTEGER, language TEXT, text TEXT, norm TEXT);
            CREATE INDEX IF NOT EXISTS ix_lyric_song ON lyric_lines(song_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_words USING fts5(norm, song_id UNINDEXED, line_no UNINDEXED,
                tokenize='unicode61 remove_diacritics 2');
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_tri USING fts5(norm, song_id UNINDEXED, line_no UNINDEXED,
                tokenize='trigram');
            """
        )
        self.docs: dict[str, _Doc] = {}
        self._df: Counter = Counter()
        self._load()

    # ------------------------------------------------------------------ storage
    def _load(self) -> None:
        rows = self.conn.execute(
            "SELECT song_id, line_no, language, text FROM lyric_lines ORDER BY song_id, line_no").fetchall()
        by_song: dict[str, list] = {}
        for sid, _, lang, text in rows:
            by_song.setdefault(sid, []).append((lang, text))
        for sid, items in by_song.items():
            self._add_doc(sid, items[0][0], [t for _, t in items])

    def _add_doc(self, song_id: str, language: str, lines: list[str]) -> None:
        tokens, token_line = doc_tokens(lines)
        doc = _Doc(song_id, language, lines, tokens, token_line, normalize_text(" ".join(lines)))
        self.docs[song_id] = doc
        self._df.update(set(tokens))

    def add(self, song_id: str, text: str, language: str = "en") -> int:
        """Add/replace the lyrics of one song. Returns number of indexed lines."""
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        with self._lock:
            self.remove(song_id)
            rows = [(song_id, i, language, ln, normalize_text(ln)) for i, ln in enumerate(lines)]
            self.conn.executemany("INSERT INTO lyric_lines VALUES (?,?,?,?,?)", rows)
            self.conn.executemany("INSERT INTO fts_words(norm, song_id, line_no) VALUES (?,?,?)",
                                  [(r[4], song_id, r[1]) for r in rows])
            self.conn.executemany("INSERT INTO fts_tri(norm, song_id, line_no) VALUES (?,?,?)",
                                  [(r[4], song_id, r[1]) for r in rows])
            self.conn.commit()
            if lines:
                self._add_doc(song_id, language, lines)
        return len(lines)

    def remove(self, song_id: str) -> bool:
        with self._lock:
            doc = self.docs.pop(song_id, None)
            if doc:
                self._df.subtract(set(doc.tokens))
            for table in ("lyric_lines", "fts_words", "fts_tri"):
                self.conn.execute(f"DELETE FROM {table} WHERE song_id = ?", (song_id,))
            self.conn.commit()
            return doc is not None

    def __len__(self) -> int:
        return len(self.docs)

    def __contains__(self, song_id: str) -> bool:
        return song_id in self.docs

    # ------------------------------------------------------------------ search
    def idf(self, token: str) -> float:
        n = max(len(self.docs), 1)
        return math.log((n + 1) / (self._df.get(token, 0) + 0.5)) + 1.0

    def _fts_candidates(self, tokens: list[str], limit: int = 100) -> set[str]:
        out: set[str] = set()
        words = [t for t in dict.fromkeys(tokens) if t.isalnum()]
        if words:
            q = " OR ".join(f'"{w}"' for w in words)
            for (sid,) in self.conn.execute(
                    "SELECT song_id FROM fts_words WHERE fts_words MATCH ? ORDER BY bm25(fts_words) LIMIT ?",
                    (q, limit)):
                out.add(sid)
        tri = [t for t in dict.fromkeys(tokens) if len(t) >= 4 and not is_stopword(t)]
        if tri:
            # substrings of length 3..4 tolerate single-character transcription errors
            parts = sorted({t[i:i + 3] for t in tri for i in range(0, len(t) - 2, 2)})
            q = " OR ".join(f'"{p}"' for p in parts)
            for (sid,) in self.conn.execute(
                    "SELECT song_id FROM fts_tri WHERE fts_tri MATCH ? ORDER BY bm25(fts_tri) LIMIT ?", (q, limit)):
                out.add(sid)
        return out

    def search(self, query: str, top_k: int = 10, language: str | None = None) -> list[LyricsCandidate]:
        q_tokens = tokenize(query)
        if len(informative_tokens(q_tokens)) < 1 or len(q_tokens) < 2:
            return []
        with self._lock:
            if len(self.docs) <= self.exhaustive_limit:
                cand_ids = list(self.docs)
            else:
                cand_ids = list(self._fts_candidates(q_tokens))
            docs = [self.docs[s] for s in cand_ids if s in self.docs]
        if language:
            docs = [d for d in docs if d.language == language] or docs
        weights = token_weights(q_tokens, self.idf)
        out = []
        for doc in docs:
            r = align_lyrics(q_tokens, weights, doc.tokens, doc.token_line, doc.lines)
            if r is None:
                continue
            out.append(LyricsCandidate(
                doc.song_id, r.score, r.coverage, r.char_similarity, r.matched_text, r.line_range,
                {"query_tokens": len(q_tokens), "aligned_query_span": list(r.query_span)},
            ))
        out.sort(key=lambda c: -c.score)
        return out[:top_k]
