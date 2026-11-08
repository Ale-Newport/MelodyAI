"""Text helpers for open-world song search: canonical song keys, phonetic keys, language guess, title phrases.

* :func:`normalize_title` / :func:`normalize_artist` / :func:`song_key` make the same recording match
  across sources (Deezer, LRCLIB, user lookups): "Despacito - Remix (feat. X)" by "Luis Fonsi & Daddy
  Yankee" and "Despacito" by "Luis Fonsi" share the key ``despacito|luis fonsi``.
* :func:`phonetic` (defined in :mod:`melodyai.lyrics.align`) maps words that sound alike to the same key.
* :func:`guess_language` uses distinctive function-word ratios (no model, works on short transcripts).
* :func:`title_phrases` proposes likely song titles from a transcript: choruses repeat the title, so
  repeated, informative n-grams come first.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence

from melodyai.lyrics.align import phonetic
from melodyai.lyrics.normalize import is_stopword, normalize_text

__all__ = ["COMMON_WORDS", "FILLERS", "guess_language", "is_common_word", "normalize_artist", "normalize_title",
           "phonetic", "song_key", "title_phrases"]

# Function words per language (accent-folded, apostrophes joined like melodyai.lyrics.normalize).
COMMON_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset(
        "the a an and or but of to in on at by for with from is are was were be been am i you he she it we they "
        "me my your his her its our their this that these those so as if not no do does did dont cant wont im "
        "youre its ill ive id what when where who how why all just can will would could got get gonna wanna "
        "there here up out down now then than too let us them him".split()),
    "es": frozenset(
        "el la los las un una unos unas y o pero de del al a en con por para que es son era fue ser soy eres "
        "estoy esta estas estan yo tu ella nosotros ellos me te se mi mis tus su sus lo le les no si ya muy "
        "mas como cuando donde porque todo toda nada sin hay tan aqui asi nos eso esto esa ese ahora hoy "
        "siempre nunca bien vamos hasta desde entre sobre tambien aunque cada otro otra mucho pues".split()),
    "pt": frozenset(
        "o os as um uma e de do da dos das que em no na nos nas com para pra por nao eu voce ele ela meu "
        "minha seu sua mais tudo isso esse essa quando como ja so sem muito tem estou sou vai vou ate agora "
        "aqui entao mim te me se lhe ao aos foi ser".split()),
    "fr": frozenset(
        "le la les de des du un une et est je tu il elle nous vous ils elles que qui ne pas dans pour sur avec "
        "mon ma mes ton ta tes son sa ses ce cest mais plus tout comme moi toi suis jai au aux on y en si "
        "bien oui quand sans quil cette".split()),
    "it": frozenset(
        "il lo la i gli le di del della dei delle che e un una per con non mi ti si ci io tu lui lei noi voi "
        "sono sei ma come piu tutto anche quando questo questa quello quella nel nella cosa solo ancora "
        "perche ho hai ha mio mia tuo tua alla sulla me te".split()),
}
# Sung vocalisations: never vote for a language, never informative.
FILLERS = frozenset("oh ooh ohh ah aah eh hey uh uhh mm mmm hmm na nah ay ey yeah yea ya woah whoa".split())
_LANG_COUNT = Counter(w for words in COMMON_WORDS.values() for w in words)

_FEAT = r"(?:feat\.?|ft\.?|featuring|with|con|part\.?|prod\.?|duet with|dueto con)"
_VERSION_WORDS = (
    r"remaster\w*|re-?master\w*|live|en vivo|en directo|ao vivo|version|versión|edit|mix|remix|mixed|acoustic|"
    r"acustic[oa]|acústic[oa]|unplugged|demo|mono|stereo|single|radio|extended|original|instrumental|karaoke|"
    r"bonus|deluxe|explicit|clean|sped up|speed up|slowed|reverb|nightcore|tiktok|from|de la pel[ií]cula|"
    r"soundtrack|banda sonora|anniversary|re-?recorded|session|take \d|\d{4}"
)
_LANG_VERSION = re.compile(r"spanish|english|español|espanol|ingl[eé]s|portugu[eê]s|french|fran[cç]ais|italian",
                           re.IGNORECASE)
_BRACKETS = re.compile(r"\[[^\]]*\]|\{[^}]*\}")
_PARENS = re.compile(r"\(([^()]*)\)")
_FEAT_INLINE = re.compile(r"\s(?:feat\.?|ft\.?|featuring)\s.*$", re.IGNORECASE)
_DASH_SUFFIX = re.compile(r"\s[-–—]\s(.*)$")
_VERSION_RE = re.compile(rf"\b(?:{_VERSION_WORDS})\b", re.IGNORECASE)
_FEAT_START = re.compile(rf"^\s*{_FEAT}\s", re.IGNORECASE)
_ARTIST_SPLIT = re.compile(
    r"\s*[,;&+(]\s*|\s+(?:feat\.?|ft\.?|featuring|x|vs\.?|with|and|y|con|e)\s+", re.IGNORECASE)


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


# ---------------------------------------------------------------------- titles and artists
def normalize_title(title: str) -> str:
    """Core title words: casefolded, accent-folded, without featuring / version / remaster decorations.

    "Bohemian Rhapsody - Remastered 2011" -> "bohemian rhapsody"; "Hips Don't Lie (feat. X)" ->
    "hips dont lie"; "Song [Live]" -> "song"; "Song (Sped Up)" -> "song". Parentheticals that are
    part of the name ("(I Can't Get No) Satisfaction") and language versions are kept.
    """
    t = _nfkc(title)
    t = _BRACKETS.sub(" ", t)

    def _paren(m: re.Match) -> str:
        inner = m.group(1)
        if _LANG_VERSION.search(inner):
            return " " + inner + " "
        return " " if _FEAT_START.match(inner) or _VERSION_RE.search(inner) else " " + inner + " "

    t = _PARENS.sub(_paren, t)
    m = _DASH_SUFFIX.search(t)
    if m and (_VERSION_RE.search(m.group(1)) or _FEAT_START.match(m.group(1))) \
            and not _LANG_VERSION.search(m.group(1)):
        t = t[:m.start()]
    t = _FEAT_INLINE.sub("", t)
    out = normalize_text(t)
    return out or normalize_text(title)


def normalize_artist(artist: str) -> str:
    """Primary artist only, casefolded and accent-folded, without a leading "the ".

    Splits at the first ``, ; & + (`` or the words ``feat ft featuring x vs with and y con e`` that
    follows a name of >= 2 characters, so "Wisin y Yandel" and "Wisin & Yandel" both give "wisin"
    while "AC/DC", "Malcolm X" and "X Ambassadors" stay whole. Dots are removed ("N.W.A" -> "nwa").
    """
    a = _nfkc(artist).replace(".", "")
    first = a
    for m in _ARTIST_SPLIT.finditer(a):
        if len(normalize_text(a[:m.start()])) > 1:
            first = a[:m.start()]
            break
    out = normalize_text(first)
    if out.startswith("the ") and len(out) > 4:
        out = out[4:]
    return out or normalize_text(artist)


def song_key(title: str, artist: str) -> str:
    """Canonical identity of a song across sources: ``"<normalize_title>|<normalize_artist>"``."""
    return f"{normalize_title(title)}|{normalize_artist(artist)}"


# ---------------------------------------------------------------------- language
def is_common_word(token: str) -> bool:
    """True for function words of any supported language (en, es, pt, fr, it) and sung fillers."""
    return token in _LANG_COUNT or token in FILLERS or is_stopword(token)


def guess_language(text: str) -> str | None:
    """Guess "es" | "en" | "pt" | "fr" | "it" from function-word frequencies, or ``None`` when unsure.

    Each token votes ``1 / (number of languages that share it)`` for every language whose function-word
    list contains it, so distinctive words ("the", "nao", "che", "je", "pero") weigh most. Fillers and
    immediate repetitions ("na na", "la la la") do not vote.
    """
    tokens = normalize_text(text).split()
    if not tokens:
        return None
    votes = {lang: 0.0 for lang in COMMON_WORDS}
    for k, t in enumerate(tokens):
        n = _LANG_COUNT.get(t)
        if not n or t in FILLERS or (k > 0 and tokens[k - 1] == t):
            continue
        for lang, words in COMMON_WORDS.items():
            if t in words:
                votes[lang] += 1.0 / n
    ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    (best, v1), (_, v2) = ranked[0], ranked[1]
    if v1 < max(1.5, 0.06 * len(tokens)) or v1 < 1.3 * v2:
        return None
    return best


# ---------------------------------------------------------------------- title phrases
_WEAK_EDGE = frozenset(
    "the a an and or but of to in on at by for with from el la los las un una y o pero de del al en con por "
    "para que le les des du et il di da do em".split())
_WEAK_START = frozenset("and or but y o pero e et ma".split())
_LENGTH_PRIOR = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.95, 5: 0.8, 6: 0.65}    # titles are rarely longer than 4 words


def _default_info(token: str) -> float:
    if is_common_word(token) or len(token) <= 1:
        return 0.3
    return 1.0 + 0.1 * min(max(len(token) - 3, 0), 6)


def _occurrences(tokens: Sequence[str], gram: tuple[str, ...]) -> list[int]:
    n, L = len(tokens), len(gram)
    out, last = [], -L
    for i in range(n - L + 1):
        if i >= last + L and tuple(tokens[i:i + L]) == gram:
            out.append(i)
            last = i
    return out


def _contains(big: tuple[str, ...], small: tuple[str, ...]) -> int:
    """Start of ``small`` inside ``big`` (contiguous), or -1."""
    L = len(small)
    return next((k for k in range(len(big) - L + 1) if big[k:k + L] == small), -1)


def _redundant(g: tuple[str, ...], count: int, h: tuple[str, ...], h_count: int) -> bool:
    """Whether candidate ``g`` adds nothing to the already chosen (better) phrase ``h``."""
    if len(g) < len(h) and _contains(h, g) >= 0:
        return count <= h_count                               # sub-phrase seen no more often
    k = _contains(g, h) if len(g) > len(h) else -1
    if k >= 0:                                                # super-phrase adding only common words
        return all(is_common_word(t) for t in g[:k] + g[k + len(h):])
    for m in range(1, min(len(g), len(h))):                   # straddles h (shares informative words)
        for a, b in ((g, h), (h, g)):
            if a[-m:] == b[:m] and any(not is_common_word(t) for t in a[-m:]):
                return count <= h_count
    return False


def title_phrases(tokens: Sequence[str], max_phrases: int = 6,
                  idf: Callable[[str], float] | None = None) -> list[str]:
    """Candidate song-title phrases from a transcript's tokens, best first (deterministic).

    Every n-gram of 1..6 tokens that is not made only of function words is scored by
    ``repetition x informativeness x length prior``: non-overlapping repetitions count most (choruses
    repeat the title), informativeness is the IDF (or a length-based prior) summed over *distinct*
    tokens and divided by ``length ** 0.8`` (long lines must not win just by being long), and phrases
    of 5-6 words are penalised (titles are rarely that long). Phrases ending in an
    article/preposition or starting with a conjunction are penalised. Single tokens qualify only if
    long (>= 7 chars) and repeated, or rare (IDF >= 6). For diversity, a phrase is dropped when a better one makes
    it redundant: it is contained in it and occurs no more often, it extends it with common words
    only ("the paper moon" after "paper moon"), or it straddles it ("dance with the paper").
    """
    toks = [t for t in tokens if t]
    if not toks or max_phrases <= 0:
        return []

    def info(t: str) -> float:
        if idf is None:
            return _default_info(t)
        return float(idf(t)) * (0.3 if is_common_word(t) else 1.0)

    seen: dict[tuple[str, ...], tuple[float, int, int]] = {}
    for L in range(1, 7):
        for i in range(len(toks) - L + 1):
            gram = tuple(toks[i:i + L])
            if gram in seen or all(is_common_word(t) or len(t) <= 1 for t in gram):
                continue
            count = len(_occurrences(toks, gram))
            if L == 1 and not ((len(gram[0]) >= 7 and count >= 2) or (idf is not None and float(idf(gram[0])) >= 6.0)):
                continue                                   # lone words: long and repeated, or rare
            distinct = dict.fromkeys(gram)
            if L > 1 and len(distinct) == 1:
                continue                                   # "na na na", "oh oh"
            score = (1.0 + 0.9 * (count - 1)) * sum(info(t) for t in distinct) / L ** 0.8 * _LENGTH_PRIOR[L]
            if gram[-1] in _WEAK_EDGE:
                score *= 0.5
            if gram[0] in _WEAK_START:
                score *= 0.7
            seen[gram] = (score, count, i)
    ranked = sorted(seen.items(), key=lambda kv: (-kv[1][0], kv[1][2], len(kv[0]), kv[0]))
    chosen: list[tuple[tuple[str, ...], int]] = []
    for gram, (_, count, _) in ranked:
        if any(_redundant(gram, count, g, c) for g, c in chosen):
            continue
        chosen.append((gram, count))
        if len(chosen) >= max_phrases:
            break
    return [" ".join(g) for g, _ in chosen]
