"""Shared lyrics alignment (melodyai.lyrics.align): exact equivalence with the pre-refactor
LyricsIndex.search, phonetic keys, chunked alignment. All lyrics are invented (CC0 demo bank or written here)."""

import numpy as np
import pytest
from rapidfuzz import fuzz, process

from melodyai.demo.lyrics_bank import LYRICS
from melodyai.lyrics.align import (
    AlignResult,
    _sw_start,
    align_lyrics,
    doc_tokens,
    phonetic,
    query_segments,
    similarity_matrix,
    token_weights,
)
from melodyai.lyrics.normalize import informative_tokens, is_stopword, normalize_text, tokenize
from melodyai.lyrics.search import LyricsCandidate, LyricsIndex
from melodyai.matching.dtw import local_alignment, sw_start


# --------------------------------------------------------------------------- oracle
def _old_search(idx: LyricsIndex, query: str, top_k: int = 10, language: str | None = None) -> list:
    """Verbatim copy of LyricsIndex.search before the alignment moved to melodyai/lyrics/align.py."""
    q_tokens = tokenize(query)
    if len(informative_tokens(q_tokens)) < 1 or len(q_tokens) < 2:
        return []
    with idx._lock:
        if len(idx.docs) <= idx.exhaustive_limit:
            cand_ids = list(idx.docs)
        else:
            cand_ids = list(idx._fts_candidates(q_tokens))
        docs = [idx.docs[s] for s in cand_ids if s in idx.docs]
    if language:
        docs = [d for d in docs if d.language == language] or docs
    weights = np.array([idx.idf(t) * (0.3 if is_stopword(t) else 1.0) for t in q_tokens])
    total = float(weights.sum())
    q_norm = " ".join(q_tokens)
    out = []
    for doc in docs:
        if not doc.tokens:
            continue
        sim = process.cdist(q_tokens, doc.tokens, scorer=fuzz.ratio, dtype=np.float32) / 100.0
        S = np.where(sim >= 0.84, 1.0, np.where(sim >= 0.67, 0.5, -0.5)) * weights[:, None]
        gap = 0.4 * float(np.mean(weights))
        best, bi, bj, H = local_alignment(S, gap)
        if best <= 0:
            continue
        i0, j0 = sw_start(H, S, bi, bj, gap)
        coverage = min(1.0, best / total)
        l0, l1 = doc.token_line[j0], doc.token_line[max(j0, bj - 1)]
        window = normalize_text(" ".join(doc.lines[max(0, l0 - 1):l1 + 2]))
        char_sim = fuzz.partial_ratio(q_norm, window) / 100.0 if window else 0.0
        score = 0.8 * coverage + 0.2 * char_sim
        out.append(LyricsCandidate(
            doc.song_id, score, coverage, char_sim, " / ".join(doc.lines[l0:l1 + 1]), (l0, l1),
            {"query_tokens": len(q_tokens), "aligned_query_span": [int(i0), int(bi)]},
        ))
    out.sort(key=lambda c: -c.score)
    return out[:top_k]


def _typo(word: str, rng) -> str:
    if len(word) < 4:
        return word
    i = int(rng.integers(1, len(word) - 1))
    op = rng.integers(0, 3)
    if op == 0:
        return word[:i] + word[i + 1:]
    if op == 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + "aeiou"[int(rng.integers(0, 5))] + word[i + 1:]


def _invented_queries(n: int = 260, seed: int = 21) -> list[tuple[str, str | None]]:
    rng = np.random.default_rng(seed)
    vocab = sorted({t for song in LYRICS for t in tokenize(song["text"])})
    out: list[tuple[str, str | None]] = []
    for k in range(n):
        song = LYRICS[int(rng.integers(0, len(LYRICS)))]
        lines = [ln for ln in song["text"].splitlines() if ln.strip()]
        i = int(rng.integers(0, len(lines) - 1))
        words = (lines[i] + " " + lines[i + 1]).split()
        kind = k % 8
        if kind == 0:
            q = lines[i]
        elif kind == 1:
            a = int(rng.integers(0, max(1, len(words) - 3)))
            q = " ".join(words[a:a + int(rng.integers(2, 7))])
        elif kind == 2:
            q = " ".join(w for w in words if rng.random() > 0.3)
        elif kind == 3:
            q = " ".join(_typo(w, rng) for w in words)
        elif kind == 4:
            q = " ".join(rng.choice(vocab, int(rng.integers(2, 12))))
        elif kind == 5:
            q = lines[i + 1] + " " + lines[i]
        elif kind == 6:
            q = " ".join(words[::-1])
        else:
            q = " ".join(rng.choice(["the", "and", "love", "night", "la", "de", "amor", "noche", "home", "oh"],
                                    int(rng.integers(2, 9))))
        out.append((q, song["language"] if k % 5 == 0 else None))
    return out + [("the and of", None), ("x", None), ("", None), ("luna vuelve pronto", "es")]


@pytest.fixture(scope="module")
def bank_index():
    idx = LyricsIndex(":memory:")
    for i, song in enumerate(LYRICS):
        idx.add(f"L{i}", song["text"], song["language"])
    return idx


@pytest.mark.parametrize("exhaustive_limit", [500, 5])          # all songs scored / FTS retrieval path
def test_refactored_search_is_identical_to_the_original(exhaustive_limit):
    idx = LyricsIndex(":memory:", exhaustive_limit=exhaustive_limit)
    for i, song in enumerate(LYRICS):
        idx.add(f"L{i}", song["text"], song["language"])
    queries = _invented_queries()
    n_results = 0
    for q, lang in queries:
        old, new = _old_search(idx, q, 10, lang), idx.search(q, 10, language=lang)
        assert new == old, q                                      # dataclass equality: every field, exact floats
        n_results += len(new)
    assert n_results > 1000                                       # the comparison is not vacuous


def test_sw_start_matches_dtw_traceback():
    rng = np.random.default_rng(3)
    for _ in range(300):
        n, m = int(rng.integers(1, 12)), int(rng.integers(1, 30))
        w = rng.uniform(0.3, 4.0, n)
        S = rng.choice([1.0, 0.5, -0.5], size=(n, m), p=[0.3, 0.2, 0.5]) * w[:, None]
        gap = 0.4 * float(w.mean())
        best, bi, bj, H = local_alignment(S, gap)
        if best > 0:
            assert _sw_start(H, S, bi, bj, gap) == sw_start(H, S, bi, bj, gap)


# --------------------------------------------------------------------------- helpers
def test_token_weights_and_doc_tokens():
    w = token_weights(["the", "lantern", "slow"], lambda t: 2.0)
    assert w.tolist() == pytest.approx([0.6, 2.0, 2.0])
    toks, tl = doc_tokens(["Turn the lantern,", "", "¡Corazón!"])
    assert toks == ["turn", "the", "lantern", "corazon"] and tl == [0, 0, 0, 2]


def test_query_segments_follow_lines_and_sentences():
    toks, spans = query_segments("Vuelve la lluvia. Sobre la ciudad!\ny en la ventana")
    assert toks == ["vuelve", "la", "lluvia", "sobre", "la", "ciudad", "y", "en", "la", "ventana"]
    assert spans == [(0, 3), (3, 6), (6, 10)]
    assert query_segments("") == ([], [])


# --------------------------------------------------------------------------- phonetic
SOUND_ALIKE = [("vez", "bes"), ("hola", "ola"), ("ahora", "aora"), ("calle", "caye"), ("llorar", "yorar"),
               ("queso", "keso"), ("cielo", "sielo"), ("zapato", "sapato"), ("corazon", "corason"),
               ("quiero", "kiero"), ("gente", "jente"), ("guerra", "guera"), ("carro", "caro"), ("verdad", "verda"),
               ("enamorado", "enamorao"), ("vida", "bida"), ("hoy", "oi"), ("ella", "eya"), ("cuando", "quando"),
               ("photo", "foto"), ("back", "bak"), ("phone", "fone"), ("night", "nite"), ("coffee", "cofee"),
               ("baby", "babi"), ("loving", "lovin"), ("city", "sity"), ("write", "right"), ("kiss", "kis")]
DIFFERENT = [("casa", "cama"), ("luna", "lana"), ("amor", "mar"), ("night", "light"), ("boat", "coat"),
             ("sol", "sal"), ("fuego", "juego"), ("beso", "peso"), ("rio", "mio"), ("dime", "time"),
             ("love", "life"), ("cielo", "celo"), ("vida", "nada"), ("pero", "para"), ("dance", "dice")]


@pytest.mark.parametrize("a,b", SOUND_ALIKE)
def test_phonetic_key_collapses_sound_alike_spellings(a, b):
    assert phonetic(a) == phonetic(b)


@pytest.mark.parametrize("a,b", DIFFERENT)
def test_phonetic_key_keeps_different_words_apart(a, b):
    assert phonetic(a) != phonetic(b)


def test_phonetic_key_edge_cases():
    assert phonetic("Corazón") == phonetic("corason")              # normalises raw input
    assert phonetic("h") == "h" and phonetic("2024") == "2024" and phonetic("") == ""


INVENTED_ES = """Vuelve la lluvia sobre la ciudad
y en la ventana se dibuja tu verdad
calles vacías que cantan sin voz
hoy te busqué en el humo del reloj
Ventana de cristal, no me dejes llorar
ventana de cristal, ya no quiero esperar
la noche llega con su vestido azul
y en cada estrella me hablas tú"""


def _align(query: str, text: str, **kw) -> AlignResult | None:
    q, segs = query_segments(query)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    toks, tl = doc_tokens(lines)
    return align_lyrics(q, token_weights(q, lambda t: 3.0), toks, tl, lines, segments=segs, **kw)


def test_phonetic_alignment_recovers_misheard_words():
    q = "bwelve la yubia sovre la siudad i en la bentana se divuja tu berdad"
    plain, phon = _align(q, INVENTED_ES), _align(q, INVENTED_ES, phonetic=True)
    assert phon.coverage >= 0.85 and phon.line_range == (0, 1)
    assert phon.score > plain.score + 0.1


def test_similarity_matrix_phonetic_never_lowers_similarity():
    a, b = ["yubia", "night", "casa"], ["lluvia", "nite", "cama", "house"]
    plain, phon = similarity_matrix(a, b), similarity_matrix(a, b, phonetic_keys=True)
    assert plain.shape == (3, 4) and (phon >= plain).all()
    assert phon[0, 0] >= 0.95 * 0.99 and phon[1, 1] >= 0.9


# --------------------------------------------------------------------------- chunked
def test_chunked_alignment_handles_reordered_and_skipped_lines():
    reordered = "la noche llega con su vestido azul\nvuelve la lluvia sobre la ciudad"
    whole, chunked = _align(reordered, INVENTED_ES), _align(reordered, INVENTED_ES, chunked=True)
    assert whole.coverage < 0.6 and chunked.coverage >= 0.85
    assert "Vuelve la lluvia" in chunked.matched_text and "vestido azul" in chunked.matched_text
    skipped = "vuelve la lluvia sobre la ciudad\ncalles vacias que cantan sin voz\nventana de cristal no me dejes llorar"
    whole, chunked = _align(skipped, INVENTED_ES), _align(skipped, INVENTED_ES, chunked=True)
    assert chunked.coverage > whole.coverage and chunked.coverage >= 0.85
    in_order = "vuelve la lluvia sobre la ciudad y en la ventana se dibuja tu verdad"
    assert _align(in_order, INVENTED_ES, chunked=True) == _align(in_order, INVENTED_ES)   # nothing to gain


def test_chunked_alignment_does_not_inflate_common_word_text():
    """Text made of common words, in unrelated order, must stay well below a match on every song."""
    df: dict[str, int] = {}
    for song in LYRICS:
        for t in set(tokenize(song["text"])):
            df[t] = df.get(t, 0) + 1
    idf = lambda t: float(np.log((len(LYRICS) + 1) / (df.get(t, 0) + 0.5)) + 1.0)
    queries = [
        "the night is long and i love my home\nbaby i need you tonight\nwe are dancing in the rain",
        "every night i dream of you and the river\nthe stars are falling down on me\ni will wait for you",
        "el amor de mi vida en la noche\nquiero volver a casa con mi corazon\nla luna y el mar son para ti",
        "tell me that you love me tonight\ni keep on walking down the road\nthe sun will rise again",
        "love love night home river rain heart sea\nwater light sun moon\nhome night love",
    ]
    for q in queries:
        qt, segs = query_segments(q)
        w = token_weights(qt, idf)
        for song in LYRICS:
            lines = [ln for ln in song["text"].splitlines() if ln.strip()]
            toks, tl = doc_tokens(lines)
            plain = align_lyrics(qt, w, toks, tl, lines, phonetic=True)
            chunked = align_lyrics(qt, w, toks, tl, lines, phonetic=True, chunked=True, segments=segs)
            if chunked is None:
                continue
            assert chunked.score <= 0.45, (q, song["title"], chunked)
            assert chunked.coverage >= plain.coverage


def test_align_result_to_dict_and_empty_inputs():
    r = _align("vuelve la lluvia sobre la ciudad", INVENTED_ES, phonetic=True, chunked=True)
    d = r.to_dict()
    assert set(d) == {"score", "coverage", "char_similarity", "matched_text", "line_range", "query_span"}
    assert d["line_range"] == [0, 0] and d["query_span"] == [0, 6]
    assert align_lyrics(["a", "b"], np.ones(2), [], [], []) is None
    assert _align("zzzz qqqq", INVENTED_ES) is None
