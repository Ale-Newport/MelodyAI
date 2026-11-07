"""Language-aware text normalisation for lyrics matching.

Steps: Unicode NFKC -> case folding -> contraction joining ("don't" -> "dont") ->
accent folding (NFD, drop combining marks: "canción" -> "cancion", "ñ" -> "n") ->
punctuation removal -> whitespace tokenisation. Numbers are kept as tokens.
"""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = {
    "en": set(
        "a an the and or but of to in on at by for with from is are was were be been am i you he she it we they "
        "me my your his her its our their this that these those so as if oh yeah ooh la na do does did not no "
        "all just im youre dont cant ill".split()
    ),
    "es": set(
        "el la los las un una unos unas y o pero de del al a en con por para que es son era fue ser soy eres "
        "yo tu el ella nosotros ellos me te se mi mis tus su sus lo le les no si ya muy mas como oh ay".split()
    ),
}
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_APOS = re.compile(r"(\w)['’`´](\w)")
_SPACE = re.compile(r"\s+")


def fold_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").casefold()
    t = _APOS.sub(r"\1\2", t)
    t = fold_accents(t)
    t = _PUNCT.sub(" ", t.replace("_", " "))
    return _SPACE.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    n = normalize_text(text)
    return n.split() if n else []


def is_stopword(token: str) -> bool:
    return any(token in s for s in STOPWORDS.values())


def informative_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if not is_stopword(t) and len(t) > 1]
