"""Real-song evaluation of the open-world lyrics recognition (:mod:`melodyai.world.search`).

    python -m melodyai.cli evaluate-world --config configs/eval_world.yaml [--conditions index,online,both]
        [--variants chorus,verse,line,misremembered,asr,asr_sung,negative] [--limit N] [--no-tts] [--calibrate]

Ground truth. ``configs/eval_world.yaml`` lists ~100 real songs by title, artist, language and tags *only*
(:func:`load_songs` rejects any other key). Their lyrics are looked up at run time -- the world index record when
the song is already indexed (no request), else ``LrclibClient.find`` through a ``WebClient`` cache
(``<store>/world_eval_cache.db``) -- and live only in memory: no lyric line, query, transcript or matched text is
ever written to a report or logged. Reports hold titles, artists, scores, ranks, word counts, line positions and
timings.

Queries (regenerated every run from a seeded RNG per song, never stored):

* ``chorus`` -- the most repeated block of 2-3 consecutive lines (:func:`chorus_block`): what people remember;
* ``verse`` -- 2-3 consecutive lines from the first half of the song, outside the chorus when possible;
* ``line`` -- one line of 4-14 words;
* ``misremembered`` -- the verse with ~15 % of its words dropped and 1-2 words replaced by a sound-alike spelling
  or a common word (:func:`misremember`), paired with ``verse`` so the effect of the damage is isolated;
* ``asr`` / ``asr_sung`` -- two chorus lines spoken by macOS ``say`` (voices per language), for ``asr_sung``
  slowed down, moved through melodic pitch steps and given a vibrato (:meth:`Speaker.sing`), then transcribed by
  Whisper (``model="auto"``, language auto-detected); the joined segment texts are the query. Only a numeric word
  error rate of the transcript is kept;
* ``negative`` -- ~30 everyday sentences in es/en assembled from a small invented vocabulary
  (:func:`negative_texts`): not lyrics, so the right answer is "no match".

Conditions (always ``learn=False``: the evaluation never writes to the world index):

* ``index`` -- the local world index (``<store>/world.db``) alone, offline;
* ``online`` -- an empty in-memory index + LRCLIB / Deezer at query time: the online fallback alone;
* ``both`` -- world index + online sources, the app's default.

LRCLIB is a free shared service, so online conditions are capped (``max_queries`` per condition, a seeded sample
spread evenly over the variants) and paced to an average LRCLIB request rate (``online.lrclib_rps``). Their
responses are cached in memory only, so latencies are cold-cache latencies.

Scoring. A returned song is the expected one when its title matches (:func:`melodyai.world.lrclib.title_similarity`
>= 0.85 -- covers / versions with the same title count); "artist" additionally needs the expected artist. Decisions
are :func:`melodyai.fusion.fusion.decide` with kind ``"world"`` (the thresholds the app would use): a ``match``
whose top-1 is another song, or any ``match`` on a negative, is a false match. :func:`calibrate_world` picks the
``world`` thresholds that maximise correct matches subject to a false-match rate <= 5 % (and reports the best
thresholds with no false match at all, plus a 2-fold check); they are only written to ``configs/calibration.json``
(``thresholds["world"]``, other entries kept) when asked (``--calibrate``). :func:`source_stats` shows which source
proposed the right songs and what each query cost in LRCLIB / Deezer requests.

Outputs: ``data/reports/<UTC time>-<name>.json`` (config, environment, index size, per-song coverage, per-query
rows, summaries, calibration proposal) and ``docs/WORLD_RESULTS.md`` (tables of metrics only). A hand-written
block between ``<!-- world-notes:start -->`` and ``<!-- world-notes:end -->`` in the Markdown survives rewrites.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from melodyai import __version__
from melodyai.config import ROOT_DIR, data_dir, store_dir
from melodyai.evaluation.metrics import latency_stats, threshold_for_far
from melodyai.fusion.fusion import CALIBRATION_PATH, DEFAULT_THRESHOLDS, decide, load_calibration
from melodyai.lyrics.normalize import informative_tokens, normalize_text, tokenize
from melodyai.world.index import WorldLyricsIndex, WorldSong, lyric_lines
from melodyai.world.lrclib import (
    LrclibClient,
    artist_similarity,
    clean_title,
    primary_artist,
    title_similarity,
)
from melodyai.world.text import FILLERS, guess_language, song_key
from melodyai.world.web import HOST_LIMITS, HostLimit, WebClient, WebError

log = logging.getLogger(__name__)

DEFAULT_CONFIG = ROOT_DIR / "configs" / "eval_world.yaml"
DEFAULT_MARKDOWN = ROOT_DIR / "docs" / "WORLD_RESULTS.md"
VARIANTS = ("chorus", "verse", "line", "misremembered", "asr", "asr_sung", "negative")
TEXT_VARIANTS = ("chorus", "verse", "line", "misremembered")
ASR_VARIANTS = {"asr": "spoken", "asr_sung": "sung"}
CONDITIONS = ("index", "online", "both")
ONLINE_CONDITIONS = ("online", "both")
GROUP_TAGS = ("tiktok", "long_tail")      # summary groups; every other song is a "hit"
TITLE_MATCH = 0.85
ARTIST_MATCH = 0.85
TOP_K = 5
SR = 16000
NOTES_START, NOTES_END = "<!-- world-notes:start -->", "<!-- world-notes:end -->"
# ground-truth lyrics lookups: one at a time, well under LRCLIB's ~90 req/min (a popular-songs sync may share it)
TRUTH_LIMITS = {"lrclib.net": HostLimit(capacity=2, refill_per_s=0.4, max_concurrent=1),
                "api.deezer.com": HOST_LIMITS["api.deezer.com"]}
_EDITORIAL_OWNER = re.compile(r"\bdeezer\b|\bedit(?:or|ora|ores|rice|eur|eurs)\b", re.IGNORECASE)
_TRENDING_PLAYLIST = re.compile(r"tik ?tok|viral", re.IGNORECASE)
_SKIP_PLAYLIST = re.compile(r"instrumental|phonk|techno|house|funk|sped up|slowed|8d|9d|k-?pop", re.IGNORECASE)
_SKIP_TRACK = re.compile(r"\binstrumental\b|\bkaraoke\b|\bsped up\b|\bslowed\b|\b8d\b", re.IGNORECASE)

DEFAULTS: dict = {
    "name": "world",
    "seed": 7,
    "store": None,                 # default: MELODYAI_STORE_DIR or data/store
    "lyrics_cache": None,          # default: <store>/world_eval_cache.db
    "join": " ",                   # query lines joined like a run-on transcript ("\n" keeps line breaks)
    "language_hint": False,        # pass the song's language to the search for typed variants
    "variants": list(VARIANTS),
    "conditions": {"index": {}, "online": {"max_queries": 60}, "both": {"max_queries": 60}},
    "online": {"sources": ["index", "lrclib", "deezer"], "lrclib_rps": 0.4, "deadline_s": 9.0},
    "misremembered": {"drop": 0.15},
    "asr": {"model": "auto", "max_songs": 24, "stretch": 0.75,
            "voices": {"es": ["Paulina", "Mónica"], "en": ["Samantha", "Daniel"]}},
    "negatives": {"n": 30},
    "calibration": {"max_fmr": 0.05, "max_negative_fmr": 0.05, "conditions": None},
    "songs": [],
}


# ---------------------------------------------------------------------------- config
@dataclass
class EvalSong:
    title: str
    artist: str
    language: str
    tags: list[str] = field(default_factory=list)

    @property
    def group(self) -> str:
        return next((t for t in GROUP_TAGS if t in self.tags), "hits")

    @property
    def key(self) -> str:
        return song_key(self.title, self.artist)

    def to_dict(self) -> dict:
        return {"title": self.title, "artist": self.artist, "language": self.language, "tags": list(self.tags),
                "group": self.group}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str | Path = DEFAULT_CONFIG, overrides: dict | None = None) -> dict:
    """The YAML config over :data:`DEFAULTS` (nested dicts merged), then ``overrides``."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping")
    cfg = DEFAULTS
    for layer in (raw, overrides or {}):
        cfg = _merge(cfg, layer)
        if "conditions" in layer:           # the listed conditions replace the defaults (no merge)
            conds = layer["conditions"] or {}
            cfg["conditions"] = ({c: {} for c in conds} if isinstance(conds, list)
                                 else {c: dict(v or {}) for c, v in conds.items()})
    return cfg


def load_songs(cfg: dict) -> list[EvalSong]:
    """Songs of the config: title + artist + language (+ tags) only -- any other key (e.g. lyrics) is an error."""
    songs, seen = [], {}
    for i, s in enumerate(cfg.get("songs") or []):
        if not isinstance(s, dict):
            raise ValueError(f"songs[{i}]: expected a mapping")
        extra = set(s) - {"title", "artist", "language", "tags"}
        if extra:
            raise ValueError(f"songs[{i}]: unexpected keys {sorted(extra)} (title, artist, language and tags only; "
                             "never lyrics)")
        title, artist = str(s.get("title") or "").strip(), str(s.get("artist") or "").strip()
        language = str(s.get("language") or "").strip().lower()
        if not title or not artist or not re.fullmatch(r"[a-z]{2}", language):
            raise ValueError(f"songs[{i}]: needs a title, an artist and a 2-letter language")
        tags = s.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        song = EvalSong(title, artist, language, [str(t).strip().lower() for t in tags if str(t).strip()])
        if song.key in seen:
            raise ValueError(f"songs[{i}]: duplicate of songs[{seen[song.key]}] ({title} / {artist})")
        seen[song.key] = i
        songs.append(song)
    return songs


def select_songs(n: int, limit: int | None, seed: int) -> list[int]:
    """Indices of the songs to evaluate: all, or a seeded sample of ``limit`` (in config order)."""
    if limit is None or limit >= n:
        return list(range(n))
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 1])
    return sorted(int(i) for i in rng.choice(n, max(0, int(limit)), replace=False))


def _rng(seed: int, *parts) -> np.random.Generator:
    """A generator for one (song, variant): independent of which other songs / variants are evaluated."""
    h = int(hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, h])


# ---------------------------------------------------------------------------- ground-truth lyrics
@dataclass
class TruthLyrics:
    """A song's real lyrics (in memory only) and where they came from."""

    lines: list[str] = field(repr=False)
    source: str                    # "index" | "lrclib"
    in_sync: bool                  # already in the world index when the run started
    lrclib_id: int | None = None
    detected_language: str | None = None

    def describe(self) -> dict:
        return {"source": self.source, "in_sync": self.in_sync, "lrclib_id": self.lrclib_id,
                "n_lines": len(self.lines), "detected_language": self.detected_language}


def find_indexed(index: WorldLyricsIndex, title: str, artist: str) -> WorldSong | None:
    """The world index record of a song: exact song key, else a title/artist-similar record."""
    ws = index.get_by_key(title, artist)
    if ws is not None:
        return ws
    words = normalize_text(clean_title(title)).split()
    if not words:
        return None
    for meta in index.list_songs(" ".join(words[:6]), limit=25):
        if title_similarity(title, meta["title"]) >= 0.9 and artist_similarity(artist, meta["artist"]) >= ARTIST_MATCH:
            return index.get(meta["id"])
    return None


def fetch_truth(song: EvalSong, index: WorldLyricsIndex | None, lrclib: LrclibClient | None) -> tuple[
        TruthLyrics | None, str]:
    """``(lyrics, reason)``: index first (free), else LRCLIB; ``(None, why)`` when neither has usable lyrics."""
    ws = find_indexed(index, song.title, song.artist) if index is not None else None
    in_sync = ws is not None and bool(ws.lyrics)
    if in_sync:
        lines = lyric_lines(ws.lyrics)
        return TruthLyrics(lines, "index", True, ws.lrclib_id, guess_language("\n".join(lines))), ""
    if lrclib is None:
        return None, "not indexed (offline)"
    try:
        rec = lrclib.find(song.title, song.artist)
    except WebError as e:
        return None, f"lrclib error ({e.kind})"
    if rec is None or not rec.has_lyrics:
        return None, "no lyrics on LRCLIB"
    lines = lyric_lines(rec.text)
    if not lines:
        return None, "empty lyrics"
    return TruthLyrics(lines, "lrclib", False, rec.id, guess_language("\n".join(lines))), ""


# ---------------------------------------------------------------------------- query variants (pure)
def _content_tokens(line: str) -> list[str]:
    return [t for t in tokenize(line) if t not in FILLERS]


def _informative(line: str) -> int:
    return len(informative_tokens(_content_tokens(line)))


def _non_overlapping(starts: list[int], n: int) -> int:
    count, last = 0, -n
    for s in starts:
        if s >= last + n:
            count, last = count + 1, s
    return count


def chorus_block(lines: Sequence[str]) -> tuple[int, int] | None:
    """``[start, end)`` of the most repeated block of 2-3 consecutive distinct lines (first occurrence).

    Ties go to the longer block, then the more informative one, then the earlier one. A block needs >= 4
    informative words ("na na na" is not a chorus). Without any repeated block: the most repeated informative
    line and the line after it; None when nothing repeats.
    """
    norm = [normalize_text(ln) for ln in lines]
    best_key, best = None, None
    for n in (2, 3):
        pos: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i in range(len(norm) - n + 1):
            block = tuple(norm[i:i + n])
            if all(block) and len(set(block)) == n:
                pos[block].append(i)
        for starts in pos.values():
            count = _non_overlapping(starts, n)
            info = sum(_informative(lines[starts[0] + j]) for j in range(n))
            if count < 2 or info < 4:
                continue
            key = (count, n, info, -starts[0])
            if best_key is None or key > best_key:
                best_key, best = key, (starts[0], starts[0] + n)
    if best is not None:
        return best
    counts = Counter(norm)
    rep = [(counts[ln], i) for i, ln in enumerate(norm)
           if ln and counts[ln] >= 2 and _informative(lines[i]) >= 2 and i + 1 < len(lines)]
    if not rep:
        return None
    _, i = max(rep, key=lambda x: (x[0], -x[1]))
    return i, i + 2


def verse_block(lines: Sequence[str], rng: np.random.Generator, avoid: Iterable[int] = ()) -> tuple[int, int] | None:
    """``[start, end)`` of 2-3 consecutive lines in the first half of the song, avoiding ``avoid`` (chorus lines)
    when possible; every line needs >= 2 words and the block >= 3 informative words."""
    avoid = set(avoid)
    sizes = (3, 2) if rng.random() < 0.5 else (2, 3)
    pools: dict[int, tuple[list[int], list[int]]] = {}
    for n in sizes:
        hi = min(len(lines), max(n, len(lines) // 2))
        starts = [i for i in range(0, hi - n + 1)
                  if all(len(_content_tokens(lines[j])) >= 2 for j in range(i, i + n))
                  and sum(_informative(lines[j]) for j in range(i, i + n)) >= 3]
        pools[n] = ([i for i in starts if not any(j in avoid for j in range(i, i + n))], starts)
    for which in (0, 1):                     # a block outside the chorus (either size) before any block
        for n in sizes:
            pool = pools[n][which]
            if pool:
                i = pool[int(rng.integers(len(pool)))]
                return i, i + n
    return None


def pick_line(lines: Sequence[str], rng: np.random.Generator, min_words: int = 4, max_words: int = 14) -> int | None:
    """Index of one line of ``min_words``-``max_words`` words (each distinct line equally likely)."""
    seen, ok = set(), []
    for i, ln in enumerate(lines):
        norm = normalize_text(ln)
        if norm in seen:
            continue
        seen.add(norm)
        if min_words <= len(_content_tokens(ln)) <= max_words and _informative(ln) >= 2:
            ok.append(i)
    return ok[int(rng.integers(len(ok)))] if ok else None


_SOUND_ALIKE = {
    "es": [(r"v", "b"), (r"b", "v"), (r"ll", "y"), (r"y(?=[aeiou])", "ll"), (r"z", "s"), (r"c(?=[ei])", "s"),
           (r"^h", ""), (r"qu(?=[ei])", "k"), (r"g(?=[ei])", "j"), (r"rr", "r"), (r"s$", ""), (r"d$", "")],
    "en": [(r"ph", "f"), (r"ck", "k"), (r"([b-df-hj-np-tv-z])\1", r"\1"), (r"ee", "ea"), (r"ea", "ee"),
           (r"ight", "ite"), (r"y$", "ie"), (r"c(?=[aou])", "k"), (r"s$", "z"), (r"er$", "a"), (r"th", "d"),
           (r"ing$", "in"), (r"ou", "ow")],
}
_VOWEL_SWAP = {"a": "e", "e": "i", "i": "e", "o": "u", "u": "o"}
COMMON_SWAP = {
    "es": "que el la mi tu yo lo se de en y no amor vida noche tiempo".split(),
    "en": "the a my you i it and to in me so now love baby night time".split(),
}


def sound_alike(word: str, language: str, rng: np.random.Generator) -> str:
    """A misspelling that sounds (about) the same: "vida" -> "bida", "night" -> "nite", "phone" -> "fone"."""
    w = normalize_text(word)
    rules = [(p, r) for p, r in _SOUND_ALIKE.get(language, _SOUND_ALIKE["en"]) if re.search(p, w)]
    if rules:
        p, r = rules[int(rng.integers(len(rules)))]
        out = re.sub(p, r, w, count=1)
        if out and out != w:
            return out
    vowels = [i for i, ch in enumerate(w) if ch in _VOWEL_SWAP]
    if vowels:
        i = vowels[int(rng.integers(len(vowels)))]
        return w[:i] + _VOWEL_SWAP[w[i]] + w[i + 1:]
    return w + w[-1:]


def misremember(lines: Sequence[str], language: str, rng: np.random.Generator, drop: float = 0.15,
                joiner: str = " ") -> tuple[str, dict]:
    """Lines as someone half-remembers them: ~``drop`` of the words left out (at least one when there are >= 5
    words), then 1-2 words replaced by a :func:`sound_alike` spelling or a common word. Returns the text and
    counts (never the words)."""
    words = [(k, w) for k, ln in enumerate(lines) for w in ln.split()]
    n = len(words)
    n_drop = max(1, int(round(drop * n))) if n >= 5 else 0
    keep = np.ones(n, dtype=bool)
    if n_drop:
        keep[rng.choice(n, n_drop, replace=False)] = False
    kept = [[k, w] for (k, w), ok in zip(words, keep) if ok]
    candidates = [i for i, (_, w) in enumerate(kept) if len(normalize_text(w)) >= 3]
    n_rep = min(int(rng.integers(1, 3)), len(candidates))
    kinds: Counter = Counter()
    if n_rep:
        for i in sorted(int(x) for x in rng.choice(candidates, n_rep, replace=False)):
            w = normalize_text(kept[i][1])
            if rng.random() < 0.5:
                kept[i][1] = sound_alike(w, language, rng)
                kinds["sound_alike"] += 1
            else:
                pool = [c for c in COMMON_SWAP.get(language, COMMON_SWAP["en"]) if c != w]
                kept[i][1] = pool[int(rng.integers(len(pool)))]
                kinds["common_word"] += 1
    out_lines = [" ".join(w for k, w in kept if k == j) for j in range(len(lines))]
    text = joiner.join(ln for ln in out_lines if ln)
    return text, {"words": n, "dropped": n_drop, "replaced": n_rep, **kinds}


# invented everyday vocabulary: none of these sentences is a lyric
_NEG_VOCAB = {
    "en": {
        "subj": ["the plumber", "my neighbour", "our accountant", "the delivery driver", "the building manager",
                 "the new intern", "the pharmacy", "the cable company", "my dentist", "the school office"],
        "verb": ["will check", "needs to sign", "forgot to send", "is going to repair", "has to pay",
                 "can pick up", "wants to reschedule", "should update", "already cancelled", "will deliver"],
        "obj": ["the invoice", "the boiler", "two boxes of printer paper", "the spreadsheet", "the parking permit",
                "the electricity bill", "a dozen eggs", "the tax form", "the broken dishwasher", "the spare keys"],
        "when": ["next tuesday", "before noon", "on the third floor", "after the budget meeting", "by friday",
                 "at the post office", "this weekend", "in the garage", "tomorrow at nine", "before the end of the month"],
        "extra": ["please bring the receipt", "the meeting room is booked until four", "the store closes at eight",
                  "turn left after the gas station", "the password is on the yellow folder",
                  "remember to water the tomatoes", "the train to the airport leaves from platform six",
                  "add milk rice and batteries to the list", "the printer on the second floor is out of toner",
                  "call the insurance company about the claim"],
    },
    "es": {
        "subj": ["el fontanero", "mi vecina", "la gestoría", "el repartidor", "el administrador de la finca",
                 "el nuevo becario", "la farmacia", "la compañía del gas", "el dentista", "la secretaría del colegio"],
        "verb": ["revisará", "tiene que firmar", "se olvidó de enviar", "va a reparar", "debe pagar",
                 "puede recoger", "quiere cambiar", "tiene que actualizar", "ya canceló", "entregará"],
        "obj": ["la factura", "la caldera", "dos cajas de folios", "la hoja de cálculo", "el permiso de aparcamiento",
                "el recibo de la luz", "una docena de huevos", "el formulario de hacienda", "el lavavajillas",
                "las llaves del trastero"],
        "when": ["el próximo martes", "antes del mediodía", "en la tercera planta", "después de la reunión",
                 "el viernes", "en correos", "este fin de semana", "en el garaje", "mañana a las nueve",
                 "antes de fin de mes"],
        "extra": ["trae el tique de compra", "la sala de reuniones está reservada hasta las cuatro",
                  "el supermercado cierra a las ocho", "gira a la izquierda después de la gasolinera",
                  "la contraseña está en la carpeta amarilla", "acuérdate de regar los tomates",
                  "el tren al aeropuerto sale del andén seis", "apunta leche arroz y pilas en la lista",
                  "la impresora de la segunda planta no tiene tóner", "llama al seguro por lo del parte"],
    },
}


def negative_texts(n: int, seed: int) -> list[tuple[str, str]]:
    """``n`` ``(language, text)`` everyday sentences (alternating es / en), deterministic for a seed."""
    out = []
    for k in range(max(0, int(n))):
        lang = ("es", "en")[k % 2]
        rng = _rng(seed, "negative", k)
        v = _NEG_VOCAB[lang]
        pick = {part: v[part][int(rng.integers(len(v[part])))] for part in v}
        out.append((lang, f"{pick['subj']} {pick['verb']} {pick['obj']} {pick['when']}. {pick['extra']}"))
    return out


@dataclass
class Query:
    """One evaluation query. ``text`` stays in memory: it is never written to a report or logged."""

    qid: str
    variant: str
    text: str = field(repr=False)
    song: int | None                   # index into the config's songs (None: negative)
    language: str | None               # language passed to the search
    line_range: tuple[int, int] | None = None
    info: dict = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return len(tokenize(self.text))


def text_queries(song_idx: int, song: EvalSong, lines: list[str], variants: Sequence[str], seed: int,
                 joiner: str = " ", drop: float = 0.15, language_hint: bool = False) -> list[Query]:
    """The typed variants of one song (seeded per song and variant; None-safe when a variant has no lines)."""
    out: list[Query] = []
    hint = song.language if language_hint else None
    qid = f"s{song_idx:03d}"
    chorus = chorus_block(lines)
    chorus_norm = {normalize_text(lines[j]) for j in range(*chorus)} if chorus else set()
    avoid = [j for j, ln in enumerate(lines) if normalize_text(ln) in chorus_norm]
    verse = verse_block(lines, _rng(seed, song.key, "verse"), avoid)
    if "chorus" in variants and chorus:
        out.append(Query(f"{qid}-chorus", "chorus", joiner.join(lines[chorus[0]:chorus[1]]), song_idx, hint, chorus))
    if "verse" in variants and verse:
        out.append(Query(f"{qid}-verse", "verse", joiner.join(lines[verse[0]:verse[1]]), song_idx, hint, verse))
    if "line" in variants:
        i = pick_line(lines, _rng(seed, song.key, "line"))
        if i is not None:
            out.append(Query(f"{qid}-line", "line", lines[i], song_idx, hint, (i, i + 1)))
    if "misremembered" in variants and verse:
        text, info = misremember(lines[verse[0]:verse[1]], song.language, _rng(seed, song.key, "misremembered"),
                                 drop=drop, joiner=joiner)
        out.append(Query(f"{qid}-misremembered", "misremembered", text, song_idx, hint, verse, info))
    return out


def asr_lines(lines: list[str], song: EvalSong, seed: int) -> tuple[int, int] | None:
    """The two lines that are "sung" for the ASR variants: the chorus' first two, else two verse lines."""
    block = chorus_block(lines) or verse_block(lines, _rng(seed, song.key, "verse"))
    if block is None:
        return None
    return block[0], min(block[1], block[0] + 2)


def word_error_rate(ref: Sequence[str], hyp: Sequence[str]) -> float:
    """Word-level Levenshtein distance / len(ref)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / len(ref)


# ---------------------------------------------------------------------------- speech synthesis + ASR
class Speaker:
    """macOS ``say`` (+ an optional "singing" transform) -> Whisper: lyric lines -> a transcript, as in the app.

    The text goes to ``say`` through stdin (never on a command line); the audio lives in a temporary directory.
    """

    def __init__(self, voices: dict[str, list[str]] | None = None, model: str = "auto", stretch: float = 0.75) -> None:
        self.voices = voices or DEFAULTS["asr"]["voices"]
        self.model, self.stretch = model, stretch
        self._transcriber = None

    @staticmethod
    def tts_available() -> bool:
        return sys.platform == "darwin" and shutil.which("say") is not None

    def voice(self, language: str, k: int) -> str:
        voices = self.voices.get(language) or self.voices.get("en") or ["Samantha"]
        return voices[k % len(voices)]

    def say(self, text: str, voice: str) -> np.ndarray:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "query.wav"
            subprocess.run(["say", "-v", voice, "-o", str(path), f"--data-format=LEI16@{SR}", "-f", "-"],
                           input=text.encode("utf-8"), check=True, capture_output=True, timeout=120)
            y, _ = sf.read(str(path), dtype="float32")
        return y if y.ndim == 1 else y.mean(axis=1)

    @staticmethod
    def _vibrato(y: np.ndarray, hz: float = 5.5, cents: float = 35.0) -> np.ndarray:
        t = np.arange(len(y)) / SR
        ratio = 2 ** (cents * np.sin(2 * np.pi * hz * t) / 1200)
        pos = np.cumsum(ratio) - ratio[0]
        pos = pos[pos < len(y) - 1]
        return np.interp(pos, np.arange(len(y)), y).astype(np.float32)

    def sing(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Speech -> "sung": slowed down, moved through melodic pitch steps (cross-faded), with a vibrato."""
        import librosa

        y = librosa.effects.time_stretch(y, rate=self.stretch)
        steps = [-4, -2, 0, 3, 5]
        shifted = {s: (y if s == 0 else librosa.effects.pitch_shift(y, sr=SR, n_steps=s)) for s in steps}
        seg, fade = int(0.38 * SR), int(0.03 * SR)
        out, weight = np.zeros_like(y), np.zeros_like(y)
        cur = 2
        for a in range(0, len(y), seg):
            cur = int(np.clip(cur + rng.choice([-2, -1, 0, 1, 2]), 0, len(steps) - 1))
            b = min(len(y), a + seg + fade)
            w = np.ones(b - a)
            k = min(fade, b - a)
            w[:k] = np.linspace(0, 1, k)
            if b < len(y):
                w[-k:] = np.linspace(1, 0, k)
            out[a:b] += w * shifted[steps[cur]][a:b]
            weight[a:b] += w
        out = self._vibrato(out / np.maximum(weight, 1e-3))
        return (0.6 * out / (np.max(np.abs(out)) + 1e-9)).astype(np.float32)

    def transcribe(self, y: np.ndarray):
        if self._transcriber is None:
            from melodyai.lyrics.asr import WhisperTranscriber

            self._transcriber = WhisperTranscriber(self.model)
        return self._transcriber.transcribe(y, SR, language=None)


def asr_query(song_idx: int, song: EvalSong, lines: list[str], variant: str, speaker: Speaker, seed: int,
              k: int) -> Query | None:
    """Synthesise two lines, optionally "sing" them, transcribe them; the joined segment texts are the query."""
    block = asr_lines(lines, song, seed)
    if block is None:
        return None
    text = " ".join(lines[block[0]:block[1]])
    voice = speaker.voice(song.language, k)
    t0 = time.perf_counter()
    y = speaker.say(text, voice)
    if ASR_VARIANTS[variant] == "sung":
        y = speaker.sing(y, _rng(seed, song.key, variant))
    tr = speaker.transcribe(y)
    # like the engine: one line per segment, and no search at all when Whisper heard no speech
    query = "\n".join(s.get("text", "").strip() for s in tr.segments).strip() if tr.speech_detected else ""
    info = {"voice": voice, "style": ASR_VARIANTS[variant], "audio_s": round(len(y) / SR, 2),
            "asr_model": tr.model, "asr_language": tr.language, "speech_detected": bool(tr.speech_detected),
            "asr_s": round(tr.elapsed_s, 2), "synth_s": round(time.perf_counter() - t0 - tr.elapsed_s, 2),
            "wer": round(word_error_rate(_content_tokens(text), _content_tokens(query)), 4),
            "asr_reason": str(getattr(tr, "reason", "") or "")[:80]}     # e.g. "no speech (VAD found 0.8 s)"
    return Query(f"s{song_idx:03d}-{variant}", variant, query, song_idx, tr.language, block, info)


# ---------------------------------------------------------------------------- scoring
def match_song(expected: EvalSong, title: str, artist: str) -> tuple[bool, bool]:
    """``(same song, same song and artist)``: same title (covers / versions count), then the expected artist."""
    same = title_similarity(expected.title, title) >= TITLE_MATCH
    return same, same and artist_similarity(expected.artist, artist) >= ARTIST_MATCH


def score_row(q: Query, song: EvalSong | None, result, decision: dict, condition: str, elapsed_s: float,
              in_sync: bool | None, requests: dict | None = None) -> dict:
    """A report row for one search: titles / artists / scores / ranks only (no query, no matched text)."""
    cands = list(result.candidates)
    rank = artist_rank = None
    top = []
    for i, c in enumerate(cands, 1):
        ok, ok_artist = match_song(song, c.title, c.artist) if song is not None else (False, False)
        if ok and rank is None:
            rank = i
        if ok_artist and artist_rank is None:
            artist_rank = i
        top.append({"title": c.title, "artist": c.artist, "score": round(c.score, 4),
                    "alignment": round(c.alignment_score, 4), "coverage": round(c.coverage, 4),
                    "line_range": list(c.line_range), "sources": list(c.sources), "popularity": c.rank,
                    "correct": ok})
    is_match = decision.get("status") == "match"
    kind = "song" if song is not None else "negative"
    correct = rank == 1
    sources = {k: {"status": v.get("status"), "n": v.get("n", 0), "elapsed_s": v.get("elapsed_s", 0.0),
                   **({"message": str(v["message"])[:120]} if v.get("message") and v.get("status") != "ok" else {})}
               for k, v in (result.sources or {}).items()}
    row = {"qid": q.qid, "condition": condition, "variant": q.variant, "kind": kind, "song": q.song,
           "title": song.title if song else None, "artist": song.artist if song else None,
           "language": song.language if song else q.language, "group": song.group if song else "negative",
           "in_sync": in_sync, "n_words": q.n_words, "line_range": list(q.line_range) if q.line_range else None,
           "status": result.status, "elapsed_s": round(elapsed_s, 4), "rank": rank, "artist_rank": artist_rank,
           "top1_correct": correct, "top1_artist_correct": artist_rank == 1,
           "top_score": round(cands[0].score, 4) if cands else 0.0,
           "second_score": round(cands[1].score, 4) if len(cands) > 1 else 0.0,
           "decision": decision.get("status"), "confidence": decision.get("confidence"),
           "match_correct": is_match and correct and kind == "song",
           "false_match": is_match and (kind == "negative" or not correct),
           "found_by": top[rank - 1]["sources"] if rank else [], "top": top, "sources": sources,
           "requests": requests or {}, "info": dict(q.info)}
    return row


def _mean(xs: Iterable) -> float | None:
    xs = list(xs)
    return float(np.mean(xs)) if xs else None


def song_metrics(rows: list[dict]) -> dict:
    """Retrieval + decision metrics of song queries."""
    if not rows:
        return {"n": 0}
    ranks = [r["rank"] for r in rows]
    lat = latency_stats([r["elapsed_s"] for r in rows])
    return {
        "n": len(rows),
        "top1": _mean(r == 1 for r in ranks),
        "top1_artist": _mean(r["top1_artist_correct"] for r in rows),
        "top5": _mean(r is not None and r <= 5 for r in ranks),
        "mrr": _mean(1.0 / r if r else 0.0 for r in ranks),
        "match_correct": _mean(r["match_correct"] for r in rows),
        "false_match": _mean(r["false_match"] for r in rows),
        "uncertain": _mean(r["decision"] == "uncertain" for r in rows),
        "abstain": _mean(r["decision"] != "match" for r in rows),
        "decision_accuracy": _mean((r["decision"] == "match") == r["top1_correct"] for r in rows),
        "precision": (sum(r["match_correct"] for r in rows) / max(1, sum(r["decision"] == "match" for r in rows))),
        "no_query": _mean(r["status"] == "no_query" for r in rows),
        "p50_ms": lat.get("p50_ms"), "p95_ms": lat.get("p95_ms"),
    }


def negative_metrics(rows: list[dict]) -> dict:
    """Negatives: any match is a false match."""
    if not rows:
        return {"n": 0}
    lat = latency_stats([r["elapsed_s"] for r in rows])
    return {"n": len(rows), "false_match": _mean(r["decision"] == "match" for r in rows),
            "uncertain": _mean(r["decision"] == "uncertain" for r in rows),
            "any_candidate": _mean(bool(r["top"]) for r in rows),
            "max_top_score": max(r["top_score"] for r in rows), "mean_top_score": _mean(r["top_score"] for r in rows),
            "p50_ms": lat.get("p50_ms"), "p95_ms": lat.get("p95_ms")}


def summarize(rows: list[dict], keys: Sequence[str]) -> list[dict]:
    """Metrics per group of ``keys`` (song rows -> :func:`song_metrics`, negatives -> :func:`negative_metrics`)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in keys)].append(r)
    out = []
    for gk in sorted(groups, key=lambda t: tuple(str(x) for x in t)):
        rs = groups[gk]
        song_rows = [r for r in rs if r["kind"] == "song"]
        out.append({**dict(zip(keys, gk)), **(song_metrics(song_rows) if song_rows else negative_metrics(rs))})
    return out


def summaries(rows: list[dict]) -> dict:
    songs = [r for r in rows if r["kind"] == "song"]
    negs = [r for r in rows if r["kind"] == "negative"]
    asr = [r for r in songs if r["variant"] in ASR_VARIANTS]
    return {
        "overall": summarize(songs, ("condition",)),
        "by_variant": summarize(songs, ("condition", "variant")),
        "by_language": summarize(songs, ("condition", "language")),
        "by_in_sync": summarize(songs, ("condition", "in_sync")),
        "by_group": summarize(songs, ("condition", "group")),
        "by_variant_language_sync": summarize(songs, ("condition", "variant", "language", "in_sync")),
        "negatives": summarize(negs, ("condition",)),
        "asr": [{"condition": c, "variant": v, "n": len(rs), "wer_mean": _mean(r["info"].get("wer", 1.0) for r in rs),
                 "speech_detected": _mean(r["info"].get("speech_detected", False) for r in rs),
                 "language_ok": _mean(r["info"].get("asr_language") == r["language"] for r in rs),
                 "asr_s_mean": _mean(r["info"].get("asr_s", 0.0) for r in rs)}
                for (c, v), rs in sorted(_group(asr, ("condition", "variant")).items())],
        "sources": source_stats(rows),
    }


def source_stats(rows: list[dict]) -> list[dict]:
    """Per condition and source: how often it answered / proposed anything / proposed the right song, its time
    and the online requests per query (LRCLIB is a free shared service: this is the cost of a query)."""
    out = []
    for (cond,), rs in sorted(_group(rows, ("condition",)).items()):
        names = sorted({k for r in rs for k in r["sources"]})
        req = {h: _mean(r["requests"].get(h, 0) for r in rs) for h in sorted({h for r in rs for h in r["requests"]})}
        songs = [r for r in rs if r["kind"] == "song"]
        for name in names:
            st = [r["sources"][name] for r in rs if r["sources"].get(name)]     # queries where it was involved
            out.append({"condition": cond, "source": name, "n": len(st),
                        "ok": _mean(x.get("status") == "ok" for x in st),
                        "proposed": _mean((x.get("n") or 0) > 0 for x in st),
                        "mean_n": _mean(x.get("n") or 0 for x in st),
                        "found_correct": sum(name in r["found_by"] for r in songs), "songs": len(songs),
                        "mean_s": _mean(x.get("elapsed_s") or 0.0 for x in st),
                        "errors": sum(x.get("status") == "error" for x in st), "requests_per_query": req})
    return out


def _group(rows: list[dict], keys: Sequence[str]) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        g[tuple(r.get(k) for k in keys)].append(r)
    return g


# ---------------------------------------------------------------------------- calibration
def evaluate_thresholds(rows: list[dict], th: dict) -> dict:
    """Decisions of ``rows`` under thresholds ``th`` ({accept, margin, ...}) from their stored top-2 scores."""
    songs = [r for r in rows if r["kind"] == "song"]
    negs = [r for r in rows if r["kind"] == "negative"]

    def accepted(r: dict) -> bool:
        top = r["top_score"]
        return bool(r["top"]) and bool(top >= th["accept"]) and bool(top - r["second_score"] >= th["margin"])

    correct = sum(accepted(r) and r["top1_correct"] for r in songs)
    wrong = sum(accepted(r) and not r["top1_correct"] for r in songs)
    neg = sum(accepted(r) for r in negs)
    n = len(songs) + len(negs)
    return {"n_songs": len(songs), "n_negatives": len(negs), "correct_matches": int(correct),
            "match_rate": correct / max(1, len(songs)), "false_matches": int(wrong + neg),
            "fmr": (wrong + neg) / max(1, n), "negative_fmr": neg / max(1, len(negs)) if negs else 0.0,
            "precision": correct / max(1, correct + wrong + neg)}


def calibrate_world(rows: list[dict], max_fmr: float = 0.05, max_negative_fmr: float | None = 0.05,
                    accept_grid: Sequence[float] | None = None, margin_grid: Sequence[float] | None = None) -> dict:
    """World thresholds {accept, margin, uncertain} maximising correct matches with false-match rate <= ``max_fmr``.

    A false match is an accepted song query whose top-1 is another song, or an accepted negative; the rate is
    over all queries (songs + negatives), and the rate on negatives alone must also stay <= ``max_negative_fmr``.
    Ties prefer fewer false matches; among equally good thresholds the centre of that plateau is taken (median
    margin, then the middle of its accept range), which generalises better than its edges on small samples.
    ``uncertain`` is set where <= 10 % of the negatives would still be shown as "uncertain" (bounded to
    [0.25, accept]). Also reports the current thresholds' behaviour and a 2-fold check (fit on even-numbered
    songs / negatives, test on odd, and back).
    """
    accept_grid = [float(a) for a in accept_grid] if accept_grid is not None else \
        [round(float(x), 2) for x in np.arange(0.30, 1.0, 0.01)]
    accept_grid.append(1.01)                                  # "never match" is always feasible
    margin_grid = [float(m) for m in margin_grid] if margin_grid is not None else \
        [round(float(x), 2) for x in np.arange(0.0, 0.205, 0.01)]

    def fit(rs: list[dict], fmr: float = max_fmr, neg_fmr: float | None = max_negative_fmr) -> tuple[dict, dict]:
        best_key, plateau = None, []
        for a in accept_grid:
            for m in margin_grid:
                ev = evaluate_thresholds(rs, {"accept": a, "margin": m})
                if ev["fmr"] > fmr or (neg_fmr is not None and ev["negative_fmr"] > neg_fmr):
                    continue
                key = (ev["correct_matches"], -ev["false_matches"])
                if best_key is None or key > best_key:
                    best_key, plateau = key, []
                if key == best_key:
                    plateau.append((a, m, ev))
        # centre of the optimal plateau: the median margin, then the accept closest to the middle of its range
        margins = sorted({m for _, m, _ in plateau})
        m_star = margins[(len(margins) - 1) // 2]
        row = [(a, ev) for a, m, ev in plateau if m == m_star]
        lo, hi = row[0][0], min(row[-1][0], 1.0)
        a_star, ev = min(row, key=lambda x: (abs(x[0] - (lo + hi) / 2), x[0]))
        th = {"accept": round(float(a_star), 4), "margin": round(float(m_star), 4)}
        neg_scores = np.array([r["top_score"] for r in rs if r["kind"] == "negative"], dtype=float)
        unc = threshold_for_far(neg_scores, 0.10) if len(neg_scores) else th["accept"] - 0.15
        th["uncertain"] = round(float(min(th["accept"], max(0.25, unc))), 4)
        return th, ev

    cur = {**DEFAULT_THRESHOLDS["world"], **load_calibration().get("thresholds", {}).get("world", {})}
    th, ev = fit(rows)
    folds = []
    for k in (0, 1):
        def part(r: dict, k: int = k) -> bool:
            idx = r["song"] if r["kind"] == "song" else int(str(r["qid"]).rsplit("-", 1)[-1] or 0)
            return idx % 2 == k

        train = [r for r in rows if part(r)]
        test = [r for r in rows if not part(r)]
        if train and test:
            fth, _ = fit(train)
            folds.append({"fit_on": "even" if k == 0 else "odd", "thresholds": fth,
                          "test": evaluate_thresholds(test, fth)})
    sth, sev = fit(rows, 0.0, 0.0)
    return {"thresholds": th, "fit": ev, "current_thresholds": cur, "current": evaluate_thresholds(rows, cur),
            "max_fmr": max_fmr, "max_negative_fmr": max_negative_fmr, "holdout": folds,
            "strict": {"thresholds": sth, "fit": sev}}


def chosen_world_thresholds(result: dict) -> dict:
    """The thresholds the app should use: the zero-false-match fit when there is one (a confident wrong answer is
    worse than an "uncertain" one that still lists the right song first), else the budgeted fit; ``uncertain``
    never drops below the current value, so that unrelated text is not shown as a possible match."""
    th = dict((result.get("strict") or result)["thresholds"])
    cur = (result.get("current_thresholds") or {}).get("uncertain")
    if cur is not None:
        th["uncertain"] = max(th["uncertain"], cur)
    return {k: th[k] for k in ("accept", "uncertain", "margin")}


def write_world_calibration(result: dict, path: str | Path = CALIBRATION_PATH, source_run: str = "") -> None:
    """Store :func:`chosen_world_thresholds` as ``thresholds["world"]`` in ``path``, keeping every other entry."""
    path = Path(path)
    cal = load_calibration(path)
    cal.setdefault("thresholds", {})["world"] = chosen_world_thresholds(result)
    strict = result.get("strict")
    cal["world_calibration"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "source_run": source_run,
        "note": "Open-world thresholds fitted on real songs (melodyai evaluate-world): "
                + ("the zero-false-match fit" if strict else
                   f"false-match rate <= {result['max_fmr']:.0%} over song queries + non-lyric negatives")
                + "; 'uncertain' kept at least at its previous value.",
        "fit": (strict or result)["fit"],
        "budgeted_fit": {"thresholds": result["thresholds"], **result["fit"]}}
    path.write_text(json.dumps(cal, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------- trending songs (YAML helper)
def pick_trending(deezer, lrclib: LrclibClient, quotas: dict[str, int], *, exclude: Iterable[str] = (),
                  queries: Sequence[str] = ("tiktok", "viral"), max_checks: int = 80,
                  per_playlist: int = 100) -> list[dict]:
    """Recent TikTok / viral songs from Deezer *editorial* playlists, as config entries (title, artist, language).

    Playlists whose title says TikTok / viral and whose owner is a Deezer editor; their tracks are interleaved in
    playlist order (the top of each list first) and kept when LRCLIB has their lyrics in a wanted language
    (``guess_language`` on the lyrics, which are never returned), until ``quotas`` ({"es": 12, ...}) are met.
    ``exclude``: song keys already in the config.
    """
    playlists, seen_pl = [], set()
    for q in queries:
        for p in deezer.search_playlists(q, 50):
            if p["id"] in seen_pl or not _TRENDING_PLAYLIST.search(p["title"]) or _SKIP_PLAYLIST.search(p["title"]):
                continue
            if _EDITORIAL_OWNER.search(p.get("user") or ""):
                seen_pl.add(p["id"])
                playlists.append(p)
    lists = [(p, deezer.playlist_tracks(p["id"], per_playlist)) for p in playlists]
    order = [(p, tracks[i]) for i in range(max((len(t) for _, t in lists), default=0)) for p, tracks in lists
             if i < len(tracks)]
    seen = set(exclude)
    counts: Counter = Counter()
    picked, checks = [], 0
    for p, tm in order:
        if all(counts[lang] >= n for lang, n in quotas.items()) or checks >= max_checks:
            break
        title, artist = clean_title(tm.title_short or tm.title), primary_artist(tm.artist)
        key = song_key(title, artist)
        if key in seen or _SKIP_TRACK.search(tm.title):
            continue
        seen.add(key)
        checks += 1
        try:
            rec = lrclib.find(title, artist)
        except WebError:
            continue
        if rec is None or not rec.has_lyrics:
            continue
        lang = guess_language(rec.text)
        if lang not in quotas or counts[lang] >= quotas[lang]:
            continue
        counts[lang] += 1
        picked.append({"title": title, "artist": artist, "language": lang, "tags": ["tiktok"],
                       "playlist": p["title"], "deezer_rank": tm.rank})
    return picked


# ---------------------------------------------------------------------------- running
def sample_queries(queries: list[Query], max_queries: int | None, seed: int, name: str) -> list[Query]:
    """At most ``max_queries``, spread evenly over the variants (round robin over seeded shuffles)."""
    if max_queries is None or len(queries) <= max_queries:
        return list(queries)
    by_var: dict[str, list[Query]] = defaultdict(list)
    for q in queries:
        by_var[q.variant].append(q)
    pools = []
    for v in sorted(by_var):
        rng = _rng(seed, "sample", name, v)
        qs = by_var[v]
        pools.append([qs[int(i)] for i in rng.permutation(len(qs))])
    out: list[Query] = []
    while len(out) < max_queries and any(pools):
        for pool in pools:
            if pool and len(out) < max_queries:
                out.append(pool.pop(0))
    order = {q.qid: i for i, q in enumerate(queries)}
    return sorted(out, key=lambda q: order[q.qid])


def build_search(condition: str, world_index: WorldLyricsIndex, cfg: dict):
    """``(WorldSearch, online, web)`` for a condition; every search has ``learn=False``."""
    from melodyai.world.search import WorldSearch
    from melodyai.world.sources import LocalIndexSource, default_sources

    deadline = float(cfg["online"].get("deadline_s", 9.0))
    if condition == "index":
        return WorldSearch(world_index, None, [LocalIndexSource(world_index)], learn=False, deadline_s=deadline), \
            False, None
    if condition not in ONLINE_CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; options: {CONDITIONS}")
    index = WorldLyricsIndex(":memory:") if condition == "online" else world_index
    web = WebClient(None)                                     # in-memory cache: cold-cache latencies
    wanted = set(cfg["online"].get("sources") or ["index", "lrclib", "deezer"])
    sources = [s for s in default_sources(index, web) if s.name in wanted]
    return WorldSearch(index, web, sources, learn=False, deadline_s=deadline), True, web


def _host_requests(web: WebClient | None) -> dict[str, int]:
    if web is None:
        return {}
    return {h.split(".")[-2] if h.count(".") else h: int(c.get("requests", 0))
            for h, c in web.stats()["hosts"].items()}


def run_condition(condition: str, search, online: bool, web: WebClient | None, queries: list[Query],
                  songs: list[EvalSong], in_sync: dict[int, bool], cal: dict, lrclib_rps: float = 0.4,
                  progress: Callable[[float, str], None] | None = None,
                  sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    """Search every query (online ones paced to ``lrclib_rps`` LRCLIB requests/s on average) and score it."""
    rows = []
    for k, q in enumerate(queries):
        before = _host_requests(web)
        t0 = time.perf_counter()
        res = search.search(q.text, language=q.language, top_k=TOP_K, online=online)
        el = time.perf_counter() - t0
        after = _host_requests(web)
        used = {h: after.get(h, 0) - before.get(h, 0) for h in after if after.get(h, 0) - before.get(h, 0)}
        song = songs[q.song] if q.song is not None else None
        ranked = [{"song_id": c.song_id, "score": c.score} for c in res.candidates]
        row = score_row(q, song, res, decide(ranked, "world", calibration=cal), condition, el,
                        in_sync.get(q.song) if q.song is not None else None, used)
        rows.append(row)
        if progress:
            progress((k + 1) / max(1, len(queries)), f"{condition} {q.variant} {k + 1}/{len(queries)}")
        if online:
            limited = any("rate limited" in str(s.get("message", "")) for s in res.sources.values())
            wait = used.get("lrclib", 0) / max(1e-6, lrclib_rps) - (time.perf_counter() - t0)
            if limited:
                wait = max(wait, 30.0)
            if wait > 0:
                sleep(wait)
    return rows


def environment(asr_model: str | None = None) -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "melodyai": __version__, "asr_model": asr_model}


def _parse_list(value: str | Sequence[str] | None, allowed: Sequence[str]) -> list[str] | None:
    if value is None:
        return None
    items = [v.strip() for v in (value.split(",") if isinstance(value, str) else value) if v and v.strip()]
    bad = [v for v in items if v not in allowed]
    if bad:
        raise ValueError(f"unknown {bad}; options: {', '.join(allowed)}")
    return items


def run_world_eval(config_path: str | Path = DEFAULT_CONFIG, *, conditions: str | Sequence[str] | None = None,
                   variants: str | Sequence[str] | None = None, limit: int | None = None, tts: bool = True,
                   calibrate: bool = False, seed: int | None = None, online_max: int | None = None,
                   asr_songs: int | None = None, label: str | None = None, store: str | Path | None = None,
                   overrides: dict | None = None, markdown_path: str | Path | None = DEFAULT_MARKDOWN,
                   report_dir: str | Path | None = None,
                   progress: Callable[[float, str], None] | None = None) -> dict:
    """Run the evaluation; returns ``{"report_path", "markdown_path", "report"}``."""
    cfg = load_config(config_path, overrides)
    seed = int(cfg["seed"] if seed is None else seed)
    conds = _parse_list(conditions, CONDITIONS) or [c for c in cfg["conditions"] if c in CONDITIONS]
    wanted = _parse_list(variants, VARIANTS) or [v for v in cfg["variants"] if v in VARIANTS]
    notes = []
    if not tts and any(v in ASR_VARIANTS for v in wanted):
        notes.append("ASR variants skipped (--no-tts)")
        wanted = [v for v in wanted if v not in ASR_VARIANTS]
    if any(v in ASR_VARIANTS for v in wanted) and not Speaker.tts_available():
        notes.append("ASR variants skipped (macOS 'say' not available)")
        wanted = [v for v in wanted if v not in ASR_VARIANTS]
    songs = load_songs(cfg)
    store = Path(store or cfg.get("store") or store_dir())
    progress = progress or (lambda p, m: None)
    t_start = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{cfg.get('name', 'world')}" + (f"-{label}" if label else "")

    world_index = WorldLyricsIndex(store / "world.db")      # read only in practice: every search has learn=False
    index_stats = world_index.stats()
    index_stats["last_sync"] = world_index.meta_get("last_sync")
    truth_web = WebClient(cfg.get("lyrics_cache") or store / "world_eval_cache.db", host_limits=TRUTH_LIMITS)
    lrclib = LrclibClient(truth_web)
    try:
        selected = select_songs(len(songs), limit, seed)
        truths: dict[int, TruthLyrics] = {}
        song_info = []
        for k, i in enumerate(selected):
            tr, why = fetch_truth(songs[i], world_index, lrclib)
            if tr is not None:
                truths[i] = tr
            song_info.append({"song": i, **songs[i].to_dict(), "lyrics": tr.describe() if tr else None,
                              "missing": why or None})
            progress(0.05 * (k + 1) / len(selected), f"lyrics {k + 1}/{len(selected)}")
        in_sync = {i: tr.in_sync for i, tr in truths.items()}

        queries: list[Query] = []
        joiner = "\n" if cfg.get("join") == "\\n" else str(cfg.get("join", " "))
        for i, tr in truths.items():
            queries += text_queries(i, songs[i], tr.lines, wanted, seed, joiner,
                                    float(cfg["misremembered"].get("drop", 0.15)), bool(cfg.get("language_hint")))
        asr_model = None
        asr_vars = [v for v in wanted if v in ASR_VARIANTS]
        if asr_vars:
            acfg = cfg["asr"]
            speaker = Speaker(acfg.get("voices"), acfg.get("model", "auto"), float(acfg.get("stretch", 0.75)))
            n_asr = int(asr_songs if asr_songs is not None else acfg.get("max_songs", 24))
            asr_ids = [i for i in selected if i in truths][:n_asr]
            for k, i in enumerate(asr_ids):
                for v in asr_vars:
                    q = asr_query(i, songs[i], truths[i].lines, v, speaker, seed, k)
                    if q is not None:
                        queries.append(q)
                        asr_model = q.info.get("asr_model") or asr_model
                progress(0.05 + 0.15 * (k + 1) / len(asr_ids), f"asr {k + 1}/{len(asr_ids)}")
        if "negative" in wanted:
            for k, (_lang, text) in enumerate(negative_texts(int(cfg["negatives"].get("n", 30)), seed)):
                queries.append(Query(f"neg-{k:02d}", "negative", text, None, None, None, {}))

        cal = load_calibration()
        rows: list[dict] = []
        plan = {}
        for ci, cond in enumerate(conds):
            ccfg = cfg["conditions"].get(cond) or {}
            cvars = set(_parse_list(ccfg.get("variants"), VARIANTS) or wanted)
            qs = [q for q in queries if q.variant in cvars]
            cap = online_max if online_max is not None and cond in ONLINE_CONDITIONS else ccfg.get("max_queries")
            qs = sample_queries(qs, cap, seed, cond)
            plan[cond] = {"queries": len(qs), "max_queries": cap, "variants": sorted(cvars & set(wanted))}
            search, online, web = build_search(cond, world_index, cfg)
            base, span = 0.2 + 0.8 * ci / len(conds), 0.8 / len(conds)
            rows += run_condition(cond, search, online, web, qs, songs, in_sync, cal,
                                  float(cfg["online"].get("lrclib_rps", 0.4)),
                                  lambda p, m, b=base, s=span: progress(b + s * p, m))
            plan[cond]["web"] = web.stats() if web is not None else None
            if web is not None:
                web.close()

        ccond = cfg["calibration"].get("conditions")
        cal_rows = [r for r in rows if not ccond or r["condition"] in ccond]
        calibration = calibrate_world(cal_rows, float(cfg["calibration"].get("max_fmr", 0.05)),
                                      cfg["calibration"].get("max_negative_fmr", 0.05)) if cal_rows else None
        report = {
            "run_id": run_id, "label": label, "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
            "config": {k: v for k, v in cfg.items() if k != "songs"}, "config_path": str(config_path),
            "conditions": conds, "variants": wanted, "plan": plan, "notes": notes,
            "env": environment(asr_model), "index": index_stats,
            "thresholds": {**DEFAULT_THRESHOLDS["world"], **cal.get("thresholds", {}).get("world", {})},
            "songs_listed": len(songs), "songs_selected": len(selected), "songs_with_lyrics": len(truths),
            "songs": song_info, "truth_web": truth_web.stats(), "rows": rows, "summary": summaries(rows),
            "calibration": calibration, "calibration_written": None, "elapsed_s": round(time.time() - t_start, 1),
        }
        if calibrate and calibration is not None:
            write_world_calibration(calibration, CALIBRATION_PATH, run_id)
            report["calibration_written"] = str(CALIBRATION_PATH.relative_to(ROOT_DIR))
    finally:
        truth_web.close()
        world_index.close()

    out_dir = Path(report_dir) if report_dir else data_dir() / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{run_id}.json"
    report_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    md = None
    if markdown_path:
        md = write_markdown(report, markdown_path)
    progress(1.0, "done")
    return {"report_path": str(report_path), "markdown_path": str(md) if md else None, "report": report}


# ---------------------------------------------------------------------------- Markdown
def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _ms(x: float | None) -> str:
    return "–" if x is None else f"{x:,.0f}"


def _table(headers: Sequence[str], rows: Iterable[Sequence]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


_SONG_COLS = ("n", "Top-1 song", "Top-1 artist", "Top-5", "MRR", "Correct match", "False match", "Abstain",
              "p50 ms", "p95 ms")


def _song_cells(m: dict) -> list:
    return [m["n"], _pct(m.get("top1")), _pct(m.get("top1_artist")), _pct(m.get("top5")),
            "–" if m.get("mrr") is None else f"{m['mrr']:.2f}", _pct(m.get("match_correct")),
            _pct(m.get("false_match")), _pct(m.get("abstain")), _ms(m.get("p50_ms")), _ms(m.get("p95_ms"))]


def render_markdown(report: dict, notes: str = "") -> str:
    """``docs/WORLD_RESULTS.md``: metric tables only (titles, artists and numbers; never lyrics or queries)."""
    s = report["summary"]
    idx = report["index"]
    created = report["created_at"][:10]
    label = report.get("label")
    out = ["# Open-world recognition: real-song results", ""]
    if label == "pilot":
        out += ["> **Pilot run.** Numbers below come from a pilot on a subset of the songs and, "
                "for the `index` condition, a world index that was still being filled by the popular-songs sync. "
                "They are indicative, not final.", ""]
    elif label and (report.get("songs_selected") or 0) < (report.get("songs_listed") or 0):
        out += [f"> **{label.capitalize()} run** on {report['songs_selected']} of {report['songs_listed']} songs.", ""]
    th = report["thresholds"]
    conds = ", ".join(f"`{c}` ({report['plan'][c]['queries']} queries)" for c in report["conditions"])
    out += [f"Generated by `melodyai evaluate-world` on **{created}** (run `{report['run_id']}`, seed {report['seed']}).",
            "",
            f"- World index: **{idx.get('songs', 0):,} songs** "
            f"({', '.join(f'{k} {v:,}' for k, v in list(idx.get('languages', {}).items())[:6])}); "
            f"last sync: {idx.get('last_sync') or 'none recorded (sync in progress)'}.",
            f"- Conditions: {conds} -- `index`: local index only, offline; `online`: empty index + LRCLIB/Deezer; `both`: index + online. "
            "All with `learn=False`.",
            f"- Songs: {report['songs_selected']} of {report['songs_listed']} listed in "
            f"`{Path(report['config_path']).name}`, {report['songs_with_lyrics']} with lyrics on LRCLIB / in the index "
            f"({sum(1 for x in report['songs'] if x['lyrics'] and x['lyrics']['in_sync'])} already in the index).",
            f"- Variants: {', '.join(f'`{v}`' for v in report['variants'])}. Queries are regenerated from the lyrics "
            "at run time and never stored; ASR variants use macOS `say` + Whisper "
            f"(`{report['env'].get('asr_model') or 'n/a'}`).",
            f"- Decisions: `decide(..., \"world\")` with accept {th['accept']}, margin {th['margin']}, "
            f"uncertain {th['uncertain']}. A *correct match* is a `match` whose top-1 has the expected title "
            "(covers count); a *false match* is a `match` on another song or on a non-lyric negative.",
            ""]
    if report.get("notes"):
        out += [f"- Note: {n}" for n in report["notes"]] + [""]
    out += ["## Overall (song queries)", "",
            _table(("Condition", *_SONG_COLS), [[r["condition"], *_song_cells(r)] for r in s["overall"]]), ""]
    out += ["## By query variant", "",
            _table(("Condition", "Variant", *_SONG_COLS),
                   [[r["condition"], r["variant"], *_song_cells(r)]
                    for r in sorted(s["by_variant"], key=lambda r: (r["condition"], VARIANTS.index(r["variant"])))]),
            ""]
    out += ["## By language", "",
            _table(("Condition", "Language", *_SONG_COLS), [[r["condition"], r["language"], *_song_cells(r)]
                                                            for r in s["by_language"]]), ""]
    out += ["## Songs already in the world index vs not", "",
            _table(("Condition", "In index", *_SONG_COLS),
                   [[r["condition"], {True: "yes", False: "no"}.get(r["in_sync"], "?"), *_song_cells(r)]
                    for r in s["by_in_sync"]]), ""]
    out += ["## By song group", "", "`tiktok`: recent TikTok / viral songs picked from Deezer editorial playlists; "
            "`long_tail`: well-known songs unlikely to be in a popular-songs sync (older classics, album tracks); "
            "`hits`: the rest.", "",
            _table(("Condition", "Group", *_SONG_COLS), [[r["condition"], r["group"], *_song_cells(r)]
                                                         for r in s["by_group"]]), ""]
    if s["negatives"]:
        out += ["## Non-lyric negatives", "", "Everyday sentences (es/en) that are not lyrics: any `match` is wrong.", "",
                _table(("Condition", "n", "False match", "Uncertain", "Any candidate", "Mean top score",
                        "Max top score", "p50 ms"),
                       [[r["condition"], r["n"], _pct(r.get("false_match")), _pct(r.get("uncertain")),
                         _pct(r.get("any_candidate")), f"{r.get('mean_top_score') or 0:.2f}",
                         f"{r.get('max_top_score') or 0:.2f}", _ms(r.get("p50_ms"))] for r in s["negatives"]]), ""]
    if s["asr"]:
        out += ["## ASR (synthetic speech / \"singing\" -> Whisper)", "",
                _table(("Condition", "Variant", "n", "Mean WER", "Speech detected", "Language right", "ASR s/clip"),
                       [[r["condition"], r["variant"], r["n"], f"{r['wer_mean']:.2f}", _pct(r["speech_detected"]),
                         _pct(r["language_ok"]), f"{r['asr_s_mean']:.1f}"] for r in s["asr"]]), ""]
    cal = report.get("calibration")
    if cal:
        th, fit, cur = cal["thresholds"], cal["fit"], cal["current"]
        written = report.get("calibration_written")
        out += ["## Threshold calibration", "",
                f"Maximise correct matches subject to false-match rate <= {cal['max_fmr']:.0%} (songs + negatives) "
                f"and <= {cal['max_negative_fmr'] or 0:.0%} on negatives. "
                + (f"The zero-false-match fit was written to `{written}`." if written
                   else "Proposal only (not written; use `--calibrate`)."), "",
                _table(("Thresholds", "accept", "margin", "uncertain", "Correct matches", "Match rate",
                        "False-match rate", "Negatives matched", "Precision"),
                       [["current", cal["current_thresholds"]["accept"], cal["current_thresholds"]["margin"],
                         cal["current_thresholds"]["uncertain"], cur["correct_matches"], _pct(cur["match_rate"]),
                         _pct(cur["fmr"]), _pct(cur["negative_fmr"]), _pct(cur["precision"])],
                        ["fitted", th["accept"], th["margin"], th["uncertain"], fit["correct_matches"],
                         _pct(fit["match_rate"]), _pct(fit["fmr"]), _pct(fit["negative_fmr"]), _pct(fit["precision"])]]
                       + [["fitted, zero false matches", x["thresholds"]["accept"], x["thresholds"]["margin"],
                           x["thresholds"]["uncertain"], x["fit"]["correct_matches"], _pct(x["fit"]["match_rate"]),
                           _pct(x["fit"]["fmr"]), _pct(x["fit"]["negative_fmr"]), _pct(x["fit"]["precision"])]
                          for x in [cal.get("strict")] if x]
                       + [[f"2-fold: fit {f['fit_on']}, test other", f["thresholds"]["accept"],
                           f["thresholds"]["margin"], f["thresholds"]["uncertain"], f["test"]["correct_matches"],
                           _pct(f["test"]["match_rate"]), _pct(f["test"]["fmr"]), _pct(f["test"]["negative_fmr"]),
                           _pct(f["test"]["precision"])] for f in cal["holdout"]]), ""]
    src = s.get("sources") or []
    if any(r["condition"] in ONLINE_CONDITIONS for r in src):
        out += ["## Candidate sources", "",
                "How often each source answered, proposed any candidate, and proposed the expected song (which then "
                "won the top-1), with its time and the online requests per query.", "",
                _table(("Condition", "Source", "Queries", "Answered", "Proposed any", "Mean proposals",
                        "Found the right top-1", "Mean s", "Errors", "Requests / query"),
                       [[r["condition"], r["source"], r["n"], _pct(r["ok"]), _pct(r["proposed"]),
                         f"{r['mean_n'] or 0:.1f}", f"{r['found_correct']} / {r['songs']}", f"{r['mean_s'] or 0:.2f}",
                         r["errors"], ", ".join(f"{h} {v:.1f}" for h, v in r["requests_per_query"].items()) or "–"]
                        for r in src if r["condition"] in ONLINE_CONDITIONS]), ""]
    fails = _failures(report["rows"])
    if fails:
        out += ["## Misses (top-1 not the expected song)", "",
                "Titles and returned titles only; songs already in the index first. `-` = nothing returned.", "",
                _table(("Condition", "Song", "Artist", "In index", "Variant", "Status", "Returned top-1", "Score",
                        "Correct rank"), fails), ""]
    missing = [x for x in report["songs"] if not x["lyrics"]]
    if missing:
        out += ["## Songs without usable lyrics (not evaluated)", "",
                _table(("Song", "Artist", "Reason"), [[x["title"], x["artist"], x["missing"]] for x in missing]), ""]
    out += ["## Notes", "", NOTES_START, notes.strip() or "_No notes yet._", NOTES_END, "",
            "Reproduce: `make evaluate-world` (full config) or `python -m melodyai.cli evaluate-world --help`. "
            f"Raw rows: `data/reports/{report['run_id']}.json` (titles, scores, ranks, timings; no lyrics).", ""]
    return "\n".join(out)


def _failures(rows: list[dict], limit: int = 80) -> list[list]:
    out = []
    for r in rows:
        if r["kind"] != "song" or r["top1_correct"]:
            continue
        t = r["top"][0] if r["top"] else None
        out.append([r["condition"], r["title"], r["artist"], {True: "yes", False: "no"}.get(r["in_sync"], "?"),
                    r["variant"], r["status"],
                    f"{t['title']} -- {t['artist']}" if t else "-", f"{t['score']:.2f}" if t else "-",
                    r["rank"] or "-"])
    out.sort(key=lambda x: (x[0], x[3] == "no", x[1], VARIANTS.index(x[4])))
    return out[:limit]


def write_markdown(report: dict, path: str | Path = DEFAULT_MARKDOWN) -> Path:
    """Write :func:`render_markdown`, keeping an existing hand-written notes block."""
    path = Path(path)
    notes = ""
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if NOTES_START in old and NOTES_END in old:
            notes = old.split(NOTES_START, 1)[1].split(NOTES_END, 1)[0]
            if notes.strip() == "_No notes yet._":
                notes = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report, notes), encoding="utf-8")
    return path
