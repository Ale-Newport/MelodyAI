"""Fuzzy, order-aware alignment of a lyrics query (typed or transcribed from singing) against one song.

The query tokens are aligned to the song's token sequence with Smith-Waterman, where a token pair
scores by its RapidFuzz similarity (>= 0.84 match, >= 0.67 half match, else mismatch) multiplied by
the IDF weight of the query token, and every skipped token costs ``0.4 x`` the mean query weight.
Matched words must therefore appear *in order and close together*: two songs that merely share a
topic ("night", "love", "corazón") do not align. ``coverage`` = aligned weight / total query weight;
``score = 0.8 * coverage + 0.2 * char_similarity`` (character partial ratio between the query and
the matched lines, robust to word segmentation errors).

Options used by the open-world search (both off by default, which reproduces the original
:meth:`melodyai.lyrics.search.LyricsIndex.search` scoring exactly):

* ``phonetic`` -- token similarity = ``max(ratio(a, b), 0.95 * ratio(phonetic(a), phonetic(b)))``
  with a language-agnostic phonetic key (Spanish b/v, s/z/c, ll/y, silent h, qu/k; English ph/f,
  ck/k, silent gh/e, doubled letters...), so misheard or mis-transcribed sung words still match.
* ``chunked`` -- the query is also split into chunks (the transcript's segments/lines when given,
  else balanced windows of <= 12 tokens, min 4) that are aligned independently, and
  ``coverage = max(whole-query coverage, 0.9 x weighted mean of chunk coverages)``. People skip,
  repeat or reorder lines when singing from memory; a single local alignment would only credit the
  longest in-order stretch. Chunks whose informative weight is below ``2 x`` the median informative
  query-token weight are not aligned separately (they count as 0 in the mean), so text made of
  common words cannot gain coverage by matching a common word in a different place per chunk.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz, process

from melodyai.lyrics.normalize import is_stopword, normalize_text, tokenize
from melodyai.matching.dtw import local_alignment

MATCH_SIM = 0.84            # token similarity counted as a full match
PARTIAL_SIM = 0.67          # ... as a half match
PHONETIC_DISCOUNT = 0.95    # phonetic-key similarity never beats an exact spelling
GAP_FACTOR = 0.4            # gap cost = GAP_FACTOR * mean query token weight
CHUNK_MAX, CHUNK_MIN = 12, 4
CHUNK_DISCOUNT = 0.9
CHUNK_MIN_WEIGHT = 2.0      # x median informative query-token weight
MAX_MATCHED_LINES = 8       # matched_text of a chunked alignment
_SEGMENT_SPLIT = re.compile(r"[\n\r.!?;¡¿]+")


@dataclass
class AlignResult:
    score: float
    coverage: float
    char_similarity: float
    matched_text: str                 # matched lyric lines joined with " / "
    line_range: tuple[int, int]       # first and last matched line (inclusive)
    query_span: tuple[int, int]       # aligned query tokens [start, end)

    def to_dict(self) -> dict:
        return {"score": round(self.score, 4), "coverage": round(self.coverage, 4),
                "char_similarity": round(self.char_similarity, 4), "matched_text": self.matched_text,
                "line_range": list(self.line_range), "query_span": list(self.query_span)}


# ---------------------------------------------------------------------- phonetic key
_PHONETIC_RULES: tuple[tuple[re.Pattern, str], ...] = tuple((re.compile(p), r) for p, r in (
    (r"^kn", "n"), (r"^wr", "r"), (r"^gh", "g"), (r"(?<=.)gh", ""),     # English silent letters
    (r"ph", "f"), (r"ck", "k"), (r"x", "ks"),
    (r"qu(?=[eiy])", "k"), (r"qu", "ku"), (r"q", "k"),                  # que/qui -> ke/ki, quando -> kuando
    (r"gu(?=[eiy])", "G"), (r"g(?=[eiy])", "j"),                        # guerra -> gerra, gente -> jente
    (r"ch", "X"), (r"sh", "X"), (r"th", "t"),
    (r"c(?=[eiy])", "s"), (r"c", "k"), (r"z", "s"), (r"v", "b"), (r"w", "u"), (r"nb", "mb"),
    (r"ll(?=[aeiou])", "y"), (r"y(?![aeiou])", "i"), (r"h", ""),         # calle -> caye, hoy -> oi, hola -> ola
    (r"(?<=..)ing$", "in"), (r"(?<=...[aeiou])d$", ""), (r"(?<=...)ado$", "ao"),  # lovin', verdá, enamorao
    (r"(.)\1+", r"\1"),                                                # doubled letters
    (r"(?<=...)e$", ""),                                               # silent / weak final e
))


@lru_cache(maxsize=262144)
def phonetic(token: str) -> str:
    """Language-agnostic phonetic key of one token, for matching misheard sung words.

    Collapses spellings that sound alike in Spanish (vez/bes, hola/ola, calle/caye, queso/keso,
    cielo/sielo, verdad/verdá, enamorado/enamorao) and English (photo/foto, back/bak, night/nite,
    loving/lovin, coffee/cofee, baby/babi) while keeping different-sounding words apart
    (casa/cama, fuego/juego, beso/peso). Never returns an empty string for a non-empty token.
    """
    t = token if token.isascii() and token.isalnum() and token.islower() else normalize_text(token).replace(" ", "")
    if not t or t.isdigit():
        return t
    s = t
    for pattern, repl in _PHONETIC_RULES:
        s = pattern.sub(repl, s)
    s = s.replace("G", "g").replace("X", "x")
    return s or t


# ---------------------------------------------------------------------- tokens and weights
def token_weights(q_tokens: Sequence[str], idf: Callable[[str], float]) -> np.ndarray:
    """IDF weight of each query token; stopwords keep 30 % (they still help to anchor the alignment)."""
    return np.array([idf(t) * (0.3 if is_stopword(t) else 1.0) for t in q_tokens])


def doc_tokens(lines: Sequence[str]) -> tuple[list[str], list[int]]:
    """Tokens of a song's lines and, for each token, the index of its line."""
    tokens: list[str] = []
    token_line: list[int] = []
    for i, line in enumerate(lines):
        for t in tokenize(line):
            tokens.append(t)
            token_line.append(i)
    return tokens, token_line


def query_segments(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Tokens of a query and the token span ``[a, b)`` of each of its lines / sentences.

    Transcripts usually come as one segment per sung phrase (newline or sentence punctuation), which
    are natural chunk boundaries for :func:`align_lyrics` (``chunked=True``).
    """
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for piece in _SEGMENT_SPLIT.split(text or ""):
        toks = tokenize(piece)
        if toks:
            spans.append((len(tokens), len(tokens) + len(toks)))
            tokens.extend(toks)
    return tokens, spans


def similarity_matrix(q_tokens: Sequence[str], tokens: Sequence[str], phonetic_keys: bool = False) -> np.ndarray:
    """Token similarity in [0, 1] (float32), shape (len(q_tokens), len(tokens))."""
    sim = process.cdist(q_tokens, tokens, scorer=fuzz.ratio, dtype=np.float32) / 100.0
    if phonetic_keys:
        psim = process.cdist([phonetic(t) for t in q_tokens], [phonetic(t) for t in tokens],
                             scorer=fuzz.ratio, dtype=np.float32) / 100.0
        sim = np.maximum(sim, PHONETIC_DISCOUNT * psim)
    return sim


# ---------------------------------------------------------------------- alignment
def _sw_start(H: np.ndarray, S: np.ndarray, bi: int, bj: int, gap: float) -> tuple[int, int]:
    """Trace a Smith-Waterman alignment back to its start cell.

    Same rule as :func:`melodyai.matching.dtw.sw_start` (including ``np.isclose``'s tolerance
    ``|a - b| <= 1e-8 + 1e-5 * |b|``) on plain floats, which is ~20x faster for short paths.
    """
    i, j = bi, bj
    while i > 0 and j > 0:
        h = float(H[i, j])
        if h <= 0:
            break
        diag = float(H[i - 1, j - 1]) + float(S[i - 1, j - 1])
        if abs(h - diag) <= 1e-8 + 1e-5 * abs(diag):
            i, j = i - 1, j - 1
            continue
        up = float(H[i - 1, j]) - gap
        if abs(h - up) <= 1e-8 + 1e-5 * abs(up):
            i -= 1
        else:
            j -= 1
    return i, j


@lru_cache(maxsize=256)
def _chunk_spans(n: int, segments: tuple[tuple[int, int], ...] | None) -> tuple[tuple[int, int], ...]:
    """Split ``n`` query tokens into chunks of CHUNK_MIN..CHUNK_MAX tokens (segment boundaries first)."""
    if n < 2 * CHUNK_MIN:
        return ()
    spans: list[tuple[int, int]] = []
    pieces = [(max(0, a), min(n, b)) for a, b in (segments or [(0, n)]) if min(n, b) > max(0, a)]
    merged: list[list[int]] = []
    for a, b in pieces:                     # merge short segments into the next one
        if merged and merged[-1][1] - merged[-1][0] < CHUNK_MIN and merged[-1][1] == a:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < CHUNK_MIN and merged[-2][1] == merged[-1][0]:
        last = merged.pop()
        merged[-1][1] = last[1]
    for a, b in merged:
        size = b - a
        k = max(1, math.ceil(size / CHUNK_MAX))
        if segments is None or (len(merged) == 1 and size >= 2 * CHUNK_MIN):
            k = max(k, 2)
        k = min(k, max(1, size // CHUNK_MIN))
        cuts = [a + round(i * size / k) for i in range(k + 1)]
        spans.extend((cuts[i], cuts[i + 1]) for i in range(k) if cuts[i + 1] > cuts[i])
    return tuple(spans) if len(spans) > 1 else ()


@lru_cache(maxsize=256)
def _informative_mask(q_tokens: tuple[str, ...]) -> np.ndarray:
    mask = np.array([not is_stopword(t) and len(t) > 1 for t in q_tokens], dtype=bool)
    mask.setflags(write=False)
    return mask


def _window_similarity(q_norm: str, lines: Sequence[str], norm_lines: Sequence[str] | None, l0: int, l1: int) -> float:
    if norm_lines is None:
        window = normalize_text(" ".join(lines[max(0, l0 - 1):l1 + 2]))
    else:
        window = " ".join(x for x in norm_lines[max(0, l0 - 1):l1 + 2] if x)
    return fuzz.partial_ratio(q_norm, window) / 100.0 if window else 0.0


def align_lyrics(q_tokens: Sequence[str], weights: np.ndarray, tokens: Sequence[str], token_line: Sequence[int],
                 lines: Sequence[str], *, phonetic: bool = False, chunked: bool = False,
                 segments: Sequence[tuple[int, int]] | None = None, sim: np.ndarray | None = None,
                 norm_lines: Sequence[str] | None = None) -> AlignResult | None:
    """Align query tokens against one song (``tokens``/``token_line`` from :func:`doc_tokens` of ``lines``).

    ``weights`` comes from :func:`token_weights`. ``segments`` (token spans of the query's lines, see
    :func:`query_segments`) guide the chunking. For speed a caller scoring many songs may pass ``sim``,
    a precomputed :func:`similarity_matrix` (e.g. sliced from one vocabulary-level matrix), and
    ``norm_lines`` (``normalize_text`` of each line). Returns ``None`` when nothing aligns.
    """
    if len(tokens) == 0 or len(q_tokens) == 0:
        return None
    if sim is None:
        sim = similarity_matrix(q_tokens, tokens, phonetic)
    # Map token similarity to a match/mismatch score scaled by token weight.
    S = np.where(sim >= MATCH_SIM, 1.0, np.where(sim >= PARTIAL_SIM, 0.5, -0.5)) * weights[:, None]
    gap = GAP_FACTOR * float(np.mean(weights))
    total = float(weights.sum())
    best, bi, bj, H = local_alignment(S, gap)
    if best <= 0:
        return None
    i0, j0 = _sw_start(H, S, bi, bj, gap)
    coverage = min(1.0, best / total)
    l0, l1 = token_line[j0], token_line[max(j0, bj - 1)]
    char_sim = _window_similarity(" ".join(q_tokens), lines, norm_lines, l0, l1)
    whole = AlignResult(0.8 * coverage + 0.2 * char_sim, coverage, char_sim, " / ".join(lines[l0:l1 + 1]),
                        (l0, l1), (int(i0), int(bi)))
    if not chunked or coverage >= CHUNK_DISCOUNT:
        return whole
    spans = _chunk_spans(len(q_tokens), tuple(map(tuple, segments)) if segments is not None else None)
    informative = _informative_mask(tuple(q_tokens))
    if not spans or not informative.any():
        return whole
    inf_w = sorted(weights[informative].tolist())
    floor = CHUNK_MIN_WEIGHT * 0.5 * (inf_w[(len(inf_w) - 1) // 2] + inf_w[len(inf_w) // 2])   # 2 x median
    den = num = 0.0
    aligned = []
    for a, b in spans:
        tot = float(weights[a:b].sum())
        den += tot
        if float(weights[a:b][informative[a:b]].sum()) < floor:
            continue                        # mostly common words: no separate credit
        bc, ci, cj, Hc = local_alignment(S[a:b], gap)
        if bc > 0:
            num += min(bc, tot)
            aligned.append((a, b, tot, ci, cj, Hc))
    chunk_cov = CHUNK_DISCOUNT * num / den if den > 0 else 0.0
    if chunk_cov <= coverage or not aligned:
        return whole
    char_num = 0.0
    parts: list[tuple[int, int, int, int]] = []
    for a, b, tot, ci, cj, Hc in aligned:
        ci0, cj0 = _sw_start(Hc, S[a:b], ci, cj, gap)
        c0, c1 = token_line[cj0], token_line[max(cj0, cj - 1)]
        char_num += tot * _window_similarity(" ".join(q_tokens[a:b]), lines, norm_lines, c0, c1)
        parts.append((c0, c1, a + ci0, a + ci))
    char_sim = max(char_sim, char_num / den)
    shown = sorted({ln for c0, c1, _, _ in parts for ln in range(c0, c1 + 1)})
    return AlignResult(0.8 * chunk_cov + 0.2 * char_sim, chunk_cov, char_sim,
                       " / ".join(lines[ln] for ln in shown[:MAX_MATCHED_LINES]),
                       (shown[0], shown[-1]), (min(p[2] for p in parts), max(p[3] for p in parts)))
