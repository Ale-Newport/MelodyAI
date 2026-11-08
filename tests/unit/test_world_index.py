"""World lyrics index (melodyai.world.index) and text helpers (melodyai.world.text).

Every lyric here is invented: hand-written below, the CC0 demo bank, or pseudo-words generated
procedurally (seeded) for the retrieval and scale tests. No network access.
"""

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from melodyai.demo.lyrics_bank import LYRICS
from melodyai.lyrics.normalize import tokenize
from melodyai.world.index import (
    WorldLyricsIndex,
    WorldMatch,
    WorldSong,
    lyric_lines,
    prior_idf,
    world_song_id,
)
from melodyai.world.text import (
    guess_language,
    normalize_artist,
    normalize_title,
    phonetic,
    song_key,
    title_phrases,
)

# --------------------------------------------------------------------------- invented songs
PAPER_LANTERNS = """Paper lanterns floating on the bay
carry every word I couldn't say
harbor lights are blinking out of time
and every wave is whispering your rhyme
Hold the ribbon, pull the ribbon tight
we were chasing comets through the night
paper lanterns, paper lanterns glow
take me where the quiet rivers go"""

COPPER_TELESCOPE = """I found a copper telescope inside my father's coat
it showed me every lighthouse and every sinking boat
the city hummed a lullaby of engines and of rain
I counted all the windows on the midnight freight train
Turn the brass, turn the glass
watch the silver planets pass
copper telescope, tell me what you see
is there someone out there looking back at me"""

VENTANA = """Vuelve la lluvia sobre la ciudad
y en la ventana se dibuja tu verdad
calles vacías que cantan sin voz
hoy te busqué en el humo del reloj
Ventana de cristal, no me dejes llorar
ventana de cristal, ya no quiero esperar
la noche llega con su vestido azul
y en cada estrella me hablas tú"""

GOLONDRINAS = """Llueve sobre la calle de las golondrinas
y yo te espero con las manos vacías
la vecina cose sueños en la azotea
mientras la tarde se vuelve de jalea
Quiero volver, quiero volver a tu lado
hecho de lluvia, de viento cansado
golondrinas que vuelan sin avisar
llévame lejos, llévame a tu mar"""

INVENTED = [
    ("Paper Lanterns", "The Velvet Kites", PAPER_LANTERNS, "en", 640_000),
    ("Copper Telescope", "Mira Holloway", COPPER_TELESCOPE, "en", 310_000),
    ("Ventana de Cristal", "Luna Gris", VENTANA, None, 720_000),
    ("Calle de las Golondrinas", "Los Faroles Lentos", GOLONDRINAS, "es", 150_000),
]

_ONSETS = ["b", "c", "d", "f", "g", "l", "m", "n", "p", "r", "s", "t", "v", "ch", "br", "tr", "pl", "gr", "qu",
           "ll", "z", "j", "h", ""]
_VOWELS = ["a", "e", "i", "o", "u", "ia", "ue", "ai", "ou", "ee"]
_CODAS = ["", "", "", "n", "s", "r", "l", "m", "z", "d", "st", "nd"]
_FUNCTION = "the a and of to in my you i me it is on de la el que y en mi tu te no lo se".split()


def pseudo_songs(n: int, seed: int = 7, vocab_size: int = 6000) -> list[WorldSong]:
    """Invented songs of pseudo-words: Zipf-distributed vocabulary, 35 % function words, repeated chorus."""
    rng = np.random.default_rng(seed)
    words: set[str] = set()
    while len(words) < vocab_size:
        w = "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) for _ in range(int(rng.integers(1, 4))))
        w += rng.choice(_CODAS)
        if len(w) >= 3:
            words.add(w)
    vocab = np.array(sorted(words) + _FUNCTION)
    cdf = np.cumsum(1.0 / np.arange(1, vocab_size + 1) ** 1.05)
    cdf /= cdf[-1]
    songs = []
    for s in range(n):
        lens = rng.integers(5, 10, int(rng.integers(10, 18)))
        ids = np.minimum(np.searchsorted(cdf, rng.random(int(lens.sum()))), vocab_size - 1)
        ids = np.where(rng.random(len(ids)) < 0.35, vocab_size + rng.integers(0, len(_FUNCTION), len(ids)), ids)
        cuts = np.concatenate([[0], np.cumsum(lens)])
        lines = [" ".join(vocab[ids[cuts[i]:cuts[i + 1]]]) for i in range(len(lens))]
        nc = int(rng.integers(2, 5))
        chorus, verses = lines[:nc], lines[nc:]
        text = verses[:4] + chorus + verses[4:8] + chorus + verses[8:] + chorus
        songs.append(WorldSong.create(f"Pseudo {s}", f"Generated {s % 997}", lyrics="\n".join(text),
                                      rank=int(rng.integers(0, 1_000_000)), language=("en", "es")[s % 2],
                                      source="sync:test"))
    return songs


_SOUND_ALIKE = [("v", "b"), ("b", "v"), ("ce", "se"), ("ci", "si"), ("z", "s"), ("ll", "y"), ("qu", "k"), ("h", "")]


def sound_alike(word: str, rng) -> str:
    """Re-spell a word the way it sounds (what a transcriber does with an unknown sung word)."""
    for a, b in _SOUND_ALIKE:
        if a in word and rng.random() < 0.7:
            return word.replace(a, b, 1)
    return word


def _song(title, artist, text, lang=None, rank=0, **kw) -> WorldSong:
    return WorldSong.create(title, artist, lyrics=text, language=lang, rank=rank, **kw)


@pytest.fixture(scope="module")
def corpus():
    pseudo = pseudo_songs(2000, seed=11)
    idx = WorldLyricsIndex()
    idx.upsert_many(pseudo)
    for i, song in enumerate(LYRICS):
        idx.upsert(_song(song["title"], f"Demo Artist {i}", song["text"], song["language"], 1000 + i))
    for title, artist, text, lang, rank in INVENTED:
        idx.upsert(_song(title, artist, text, lang, rank, source="sync:quick"))
    return idx, pseudo


def _top(idx, q, **kw) -> WorldMatch | None:
    r = idx.search(q, 5, **kw)
    return r[0] if r else None


# --------------------------------------------------------------------------- text helpers
@pytest.mark.parametrize("raw,expected", [
    ("Bohemian Lanterns - Remastered 2011", "bohemian lanterns"),
    ("Paper Lanterns (feat. Someone Else)", "paper lanterns"),
    ("Paper Lanterns [Live]", "paper lanterns"),
    ("Paper Lanterns (Sped Up)", "paper lanterns"),
    ("Paper Lanterns - Radio Edit", "paper lanterns"),
    ("Paper Lanterns feat. Someone", "paper lanterns"),
    ("Corazón (Acústico)", "corazon"),
    ("(I Never Said) Goodbye", "i never said goodbye"),
    ("Paper Lanterns (Spanish Version)", "paper lanterns spanish version"),
    ("1999", "1999"),
])
def test_normalize_title(raw, expected):
    assert normalize_title(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("The Velvet Kites", "velvet kites"), ("Luna Gris & Mira Holloway", "luna gris"),
    ("Luna Gris y Mira Holloway", "luna gris"), ("Luna Gris feat. Someone", "luna gris"),
    ("Luna Gris x Someone", "luna gris"), ("Luna Gris, Someone", "luna gris"), ("Beyoncé", "beyonce"),
    ("AC/DC", "ac dc"), ("Malcolm X", "malcolm x"), ("X Ambassadors", "x ambassadors"), ("N.W.A", "nwa"),
])
def test_normalize_artist(raw, expected):
    assert normalize_artist(raw) == expected


def test_song_key_and_id_are_shared_by_versions_of_a_song():
    k = song_key("Paper Lanterns", "The Velvet Kites")
    assert k == "paper lanterns|velvet kites"
    assert song_key("Paper Lanterns - Remix (feat. A)", "Velvet Kites & B") == k
    sid = world_song_id("Paper Lanterns", "The Velvet Kites")
    assert sid.startswith("w") and len(sid) == 16 and int(sid[1:], 16) >= 0
    assert WorldSong.create("PAPER LANTERNS [Live]", "the velvet kites").id == sid


@pytest.mark.parametrize("text,lang", [
    (PAPER_LANTERNS, "en"), (VENTANA, "es"), (GOLONDRINAS, "es"),
    ("eu nao sei o que fazer com voce meu amor", "pt"),
    ("je ne sais pas pourquoi tu es partie avec lui", "fr"),
    ("io non so perche sei andata via con lui", "it"),
    ("na na na na hey hey electric tangerine", None),
    ("hola", None), ("", None),
])
def test_guess_language(text, lang):
    assert guess_language(text) == lang


def test_title_phrases_prefer_the_repeated_chorus():
    t = tokenize("I walk along the silver line tonight. Dance with the paper moon, oh dance with the paper moon. "
                 "The city lights are falling down, dance with the paper moon")
    phrases = title_phrases(t)
    assert phrases[0] == "paper moon" and "dance with the paper moon" in phrases[:3]
    assert "the paper moon" not in phrases and "dance with the paper" not in phrases   # redundant variants
    assert len(phrases) == 6 and phrases == title_phrases(list(t))                     # deterministic
    es = title_phrases(tokenize("Baila conmigo bajo la luna de papel, luna de papel, que la noche es nuestra, "
                                "luna de papel, ay luna de papel"), 3)
    assert es[0] == "luna de papel" and len(es) == 3
    assert title_phrases(tokenize("marmalade sky over the harbour, marmalade sky, marmalade sky"))[:2] == [
        "marmalade", "marmalade sky"]


def test_title_phrases_skip_fillers_and_function_words():
    assert title_phrases(tokenize("oh oh oh yeah yeah na na na")) == []
    assert title_phrases([]) == []
    out = title_phrases(tokenize("na na na hey hey electric tangerine, na na na hey hey electric tangerine"))
    assert out[0] == "electric tangerine"
    lone = title_phrases(tokenize("the river runs and the mountain sings"))
    assert all(len(p.split()) > 1 for p in lone)                       # single words must be long and repeated
    rare = title_phrases(tokenize("the river runs and the mountain sings"), idf=lambda t: 8.0 if t == "river" else 2)
    assert "river" in rare


def test_text_phonetic_is_the_alignment_key():
    assert phonetic("lluvia") == phonetic("yubia")


# --------------------------------------------------------------------------- storage
def test_upsert_merges_versions_and_keeps_max_rank():
    idx = WorldLyricsIndex()
    assert idx.upsert(_song("Paper Lanterns", "The Velvet Kites", PAPER_LANTERNS, rank=500, source="sync:quick"))
    first = idx.get_by_key("Paper Lanterns", "The Velvet Kites")
    assert first.language == "en" and first.source == "sync:quick" and first.added_at > 0
    idf_before = idx.idf("lanterns")
    newer = WorldSong("", "Paper Lanterns (Remastered 2011)", "Velvet Kites feat. Somebody", lyrics=COPPER_TELESCOPE,
                      rank=100, deezer_id=42, cover_url="https://example.invalid/c.jpg", source="lookup")
    assert idx.upsert(newer) is False and len(idx) == 1
    s = idx.get(first.id)
    assert s.rank == 500 and s.deezer_id == 42 and s.cover_url.endswith("c.jpg")
    assert s.title == "Paper Lanterns" and s.source == "sync:quick" and s.added_at == first.added_at
    assert s.lyrics == COPPER_TELESCOPE                                      # new non-empty lyrics replace old
    assert _top(idx, "turn the brass turn the glass watch the silver planets").song.id == s.id
    assert idx.search("paper lanterns floating on the bay carry every word") == [] or \
        _top(idx, "paper lanterns floating on the bay carry every word").align.score < 0.4
    assert idx.idf("lanterns") > idf_before                                 # df of removed words went down
    assert idx._df_of(["lanterns", "telescope"]) == {"telescope": 1}
    more_popular = WorldSong("", "Paper Lanterns - Live", "The Velvet Kites", lyrics="", rank=900, album="Kites")
    assert idx.upsert(more_popular) is False
    s = idx.get(first.id)
    assert s.rank == 900 and s.title == "Paper Lanterns - Live" and s.album == "Kites"
    assert s.lyrics == COPPER_TELESCOPE and s.deezer_id == 42                # empty lyrics / fields keep the old


def test_get_has_key_meta_and_list_songs():
    idx = WorldLyricsIndex()
    for title, artist, text, lang, rank in INVENTED:
        idx.upsert(_song(title, artist, text, lang, rank))
    s = idx.get_by_key("ventana de cristal", "LUNA GRIS")
    assert s is not None and idx.get(s.id) == s and idx.get("world:" + s.id) == s
    assert idx.has_key("Ventana de Cristal (En Vivo)", "Luna Gris") and not idx.has_key("Ventana", "Luna Gris")
    assert idx.get("wdoesnotexist000") is None and idx.get_by_key("nope", "nobody") is None
    metas = idx.list_songs()
    assert [m["title"] for m in metas] == ["Ventana de Cristal", "Paper Lanterns", "Copper Telescope",
                                           "Calle de las Golondrinas"]
    assert all("lyrics" not in m and m["id"].startswith("w") for m in metas)
    assert [m["title"] for m in idx.list_songs(q="velvet")] == ["Paper Lanterns"]
    assert [m["title"] for m in idx.list_songs(q="cristal luna")] == ["Ventana de Cristal"]
    assert [m["title"] for m in idx.list_songs(offset=1, limit=2)] == ["Paper Lanterns", "Copper Telescope"]
    assert idx.meta_get("last_sync") is None
    idx.meta_set("last_sync", "{}")
    idx.meta_set("last_sync", '{"added": 3}')
    assert idx.meta_get("last_sync") == '{"added": 3}'
    st = idx.stats()
    assert st["songs"] == 4 and st["languages"] == {"en": 2, "es": 2} and st["sources"] == {"": 4}
    assert st["bytes"] > 0


def test_lyrics_are_cleaned_and_short_queries_return_nothing():
    idx = WorldLyricsIndex()
    idx.upsert(_song("Marker Song", "Someone", "[Chorus]\n  first line here  \n\n(x2)\nsecond line there"))
    assert idx.get_by_key("Marker Song", "Someone").lyrics == "first line here\nsecond line there"
    assert lyric_lines("[Verse 1]\nhello there\n") == ["hello there"]
    assert idx.search("hola") == [] and idx.search("the and of") == [] and idx.search("") == []
    with pytest.raises(ValueError):                                         # whole batch rolled back
        idx.upsert_many([_song("Fine Song", "Someone", "some words here"), WorldSong("", " ", "Nobody")])
    assert len(idx) == 1 and not idx.has_key("Fine Song", "Someone")
    assert idx.score_lyrics("", PAPER_LANTERNS) is None and idx.score_lyrics("paper lanterns float", "") is None


def test_persistence_and_wal(tmp_path):
    path = tmp_path / "world" / "world.db"
    a = WorldLyricsIndex(path)
    for title, artist, text, lang, rank in INVENTED:
        a.upsert(_song(title, artist, text, lang, rank))
    a.meta_set("k", "v")
    assert a.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    b = WorldLyricsIndex(path)
    assert len(b) == 4 and b.meta_get("k") == "v"
    assert _top(b, "turn the brass turn the glass").song.title == "Copper Telescope"
    a.upsert(_song("Another One", "Somebody", "brand new words for a brand new song", "en"))
    assert len(b) == 5                                                       # sees other connections' writes
    a.close()
    b.close()


def test_idf_fallback_without_corpus_and_for_unseen_words():
    empty = WorldLyricsIndex()
    assert empty.idf("lantern") == prior_idf("lantern") and empty.idf("the") < empty.idf("lantern")
    assert empty.idf("voce") < empty.idf("ventana")                         # pt function word, still low
    r = empty.score_lyrics("the and of the in my lantern", "the and of the in my\nnothing else here")
    assert r is None or r.coverage < 0.6                                     # function words alone are weak evidence
    idx = WorldLyricsIndex()
    idx.upsert_many(pseudo_songs(300, seed=5))
    assert idx.idf("zzzunseenword") > idx.idf("the")
    assert idx.idf("che") <= 2.5 and idx.idf("che") < idx.idf("zzzunseenword")


def test_concurrent_searches_and_upserts(tmp_path):
    idx = WorldLyricsIndex(tmp_path / "w.db")
    idx.upsert_many(pseudo_songs(200, seed=9))
    extra = pseudo_songs(40, seed=10)
    for k, s in enumerate(extra):
        s.title, s.id = f"Extra {k}", ""

    def work(k):
        if k % 2:
            return idx.upsert(extra[k // 2])
        return idx.search(extra[0].lyrics.splitlines()[0] + " " + extra[1].lyrics.splitlines()[1], 3)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work, range(80)))
    assert len(idx) == 240


# --------------------------------------------------------------------------- search
def test_search_exact_partial_and_line_skipping(corpus):
    idx, _ = corpus
    cases = {
        "Turn the brass, turn the glass, watch the silver planets pass": "Copper Telescope",
        "every lighthouse and every sinking": "Copper Telescope",
        "hold the ribbon pull the ribbon tight we were chasing comets": "Paper Lanterns",
        "vuelve la lluvia sobre la ciudad\nhoy te busque en el humo del reloj\nla noche llega con su vestido azul":
            "Ventana de Cristal",
        "quiero volver quiero volver a tu lado\ngolondrinas que vuelan sin avisar": "Calle de las Golondrinas",
        "the lighthouse keeper counts the ships at night": "The Lighthouse Keeper",
    }
    for q, title in cases.items():
        m = _top(idx, q)
        assert m is not None and m.song.title == title, q
        assert m.align.score >= 0.7, (q, m.align)
    m = _top(idx, "Turn the brass, turn the glass, watch the silver planets pass")
    assert "Turn the brass" in m.align.matched_text and m.align.line_range == (4, 5)
    d = m.to_dict()
    assert d["song"]["title"] == "Copper Telescope" and "lyrics" not in d["song"] and d["coverage"] > 0.9


def test_search_misheard_and_misspelled(corpus):
    idx, _ = corpus
    cases = {
        "yueve sobre la caye de las golondrinas i yo te espero con las manos basias": "Calle de las Golondrinas",
        "bwelve la yubia sovre la siudad i en la bentana se divuja tu berdad": "Ventana de Cristal",
        "paper lantern floatin on the bay cary evry word i coudnt say": "Paper Lanterns",
        "i found a coper telescop inside my fathers cote": "Copper Telescope",
    }
    for q, title in cases.items():
        m = _top(idx, q)
        assert m is not None and m.song.title == title, q
        assert m.align.score >= 0.6, (q, m.align)


def test_search_language_is_only_a_preference(corpus):
    idx, _ = corpus
    q = "ventana de cristal no me dejes llorar"
    assert _top(idx, q, language="en").song.title == "Ventana de Cristal"
    assert _top(idx, q, language="es").song.title == "Ventana de Cristal"


def test_unrelated_queries_score_low(corpus):
    idx, _ = corpus
    for q in ["the weather is nice today so we will go shopping with my friends",
              "manana tengo que ir al banco a pagar la factura del telefono",
              "please remember to buy milk eggs and bread on the way home",
              "el partido de futbol empieza a las ocho en el estadio"]:
        r = idx.search(q, 3)
        assert not r or r[0].align.score < 0.45, (q, r[0].song.title, r[0].align)


def test_score_lyrics_verifies_arbitrary_lyrics(corpus):
    idx, _ = corpus
    good = idx.score_lyrics("ventana de cristal ya no quiero esperar", VENTANA)
    bad = idx.score_lyrics("ventana de cristal ya no quiero esperar", PAPER_LANTERNS)
    assert good.score > 0.9 and good.line_range == (5, 5)
    assert bad is None or bad.score < 0.3


def test_search_among_many_invented_songs(corpus):
    idx, pseudo = corpus
    rng = np.random.default_rng(17)
    hits = {"exact": 0, "sound_alike": 0, "skip": 0}
    picks = rng.choice(len(pseudo), 30, replace=False)
    for s in picks:
        lines = pseudo[s].lyrics.splitlines()
        k = int(rng.integers(0, len(lines) - 3))
        queries = {"exact": lines[k] + "\n" + lines[k + 1],
                   "sound_alike": " ".join(sound_alike(w, rng) for w in (lines[k] + " " + lines[k + 1]).split()),
                   "skip": lines[k] + "\n" + lines[k + 2] + "\n" + lines[k + 3]}
        for kind, q in queries.items():
            m = _top(idx, q)
            hits[kind] += bool(m and m.song.id == pseudo[s].id)
    assert hits["exact"] >= 28 and hits["skip"] >= 27 and hits["sound_alike"] >= 26, hits


# --------------------------------------------------------------------------- scale
def _scale_run(n_songs: int, tmp_path) -> dict:
    songs = pseudo_songs(n_songs, seed=7)
    idx = WorldLyricsIndex(tmp_path / f"scale{n_songs}.db")
    t0 = time.perf_counter()
    idx.upsert_many(songs)
    build_s = time.perf_counter() - t0
    rng = np.random.default_rng(23)
    idx.search(songs[0].lyrics.splitlines()[0] + " warm up", 1)
    lat, hits = [], 0
    for k, s in enumerate(rng.choice(n_songs, 60, replace=False)):
        lines = songs[s].lyrics.splitlines()
        i = int(rng.integers(0, len(lines) - 3))
        q = [lines[i] + "\n" + lines[i + 1],
             " ".join(sound_alike(w, rng) for w in (lines[i] + " " + lines[i + 1]).split()),
             lines[i] + "\n" + lines[i + 2] + "\n" + lines[i + 3]][k % 3]
        t0 = time.perf_counter()
        r = idx.search(q, 5)
        lat.append(time.perf_counter() - t0)
        hits += bool(r and r[0].song.id == songs[s].id)
    st = idx.stats()
    out = {"songs": st["songs"], "build_s": round(build_s, 1), "p50_ms": round(1000 * float(np.percentile(lat, 50))),
           "p95_ms": round(1000 * float(np.percentile(lat, 95))), "top1": hits / len(lat),
           "mb_per_1k_songs": round(st["bytes"] / 1e6 / n_songs * 1000, 2)}
    print(f"\nworld index scale: {out}")
    idx.close()
    return out


def test_scale_20k_songs(tmp_path):
    """>= 20k invented songs on disk: latency p50/p95 and top-1 (printed with -s). Builds in ~10 s."""
    out = _scale_run(20_000, tmp_path)
    assert out["songs"] == 20_000
    assert out["p95_ms"] < 1500 and out["p50_ms"] < 600                     # generous: ~40 / ~120 ms measured
    assert out["top1"] >= 0.9


@pytest.mark.slow
def test_scale_60k_songs(tmp_path):
    out = _scale_run(60_000, tmp_path)
    assert out["p95_ms"] < 2000 and out["top1"] >= 0.9


def test_fts_schema_is_contentless_when_supported():
    idx = WorldLyricsIndex()
    sql = idx.conn.execute("SELECT sql FROM sqlite_master WHERE name = 'fts_lines'").fetchone()[0]
    assert "remove_diacritics 2" in sql
    if sqlite3.sqlite_version_info >= (3, 43, 0):
        assert "contentless_delete=1" in sql


# --------------------------------------------------------------------------- concurrency, size, forgetting
def test_upserts_of_the_same_songs_from_several_connections_do_not_race(tmp_path):
    """Regression: the "is it there?" read ran before the write transaction, so two writers (a CLI sync and the
    server) upserting the same song raced into ``IntegrityError: UNIQUE constraint failed: songs.key``."""
    songs = [WorldSong.create(f"Invented Song {i}", f"Band {i % 7}",
                              lyrics=f"line one of song {i}\nanother line number {i}") for i in range(150)]
    errors: list[str] = []
    WorldLyricsIndex(tmp_path / "w.db").close()        # the store exists (WAL) before the writers start

    def work(_):
        ix = WorldLyricsIndex(tmp_path / "w.db")       # its own connection, like another process
        for s in songs:
            try:
                ix.upsert(s)
            except Exception as e:  # noqa: BLE001
                errors.append(type(e).__name__)
        ix.close()

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work, range(4)))
    ix = WorldLyricsIndex(tmp_path / "w.db")
    assert errors == [] and len(ix) == 150
    assert ix.conn.execute("SELECT n FROM df WHERE term = 'another'").fetchone()[0] == 150   # no drifted counts


def test_oversized_lyrics_are_capped_for_storage_and_alignment():
    from melodyai.world.index import MAX_LYRIC_TOKENS

    words = [f"w{i % 997}x" for i in range(MAX_LYRIC_TOKENS * 5)]
    huge = "\n".join(" ".join(words[i:i + 8]) for i in range(0, len(words), 8))
    ix = WorldLyricsIndex()
    ix.upsert(WorldSong.create("Endless Recording", "Spoken Word Club", lyrics=huge))
    stored = ix.get_by_key("Endless Recording", "Spoken Word Club").lyrics
    assert len(stored.split()) <= MAX_LYRIC_TOKENS and huge.startswith(stored)
    one_line = " ".join(words)                             # a single enormous line too
    for lyr in (huge, one_line):
        r = ix.score_lyrics(" ".join(words[:12]), lyr)
        assert r is not None and r.score > 0.9


def test_delete_source_forgets_learned_songs_and_their_counts():
    ix = WorldLyricsIndex()
    ix.upsert(WorldSong.create("Paper Lanterns", "Marea Alta", lyrics=PAPER_LANTERNS, source="sync:quick"))
    ix.upsert(WorldSong.create("Learned Tune", "Some Band", lyrics="zebra moonlight saxophone\nzebra again",
                               source="lookup"))
    assert ix.idf("zebra") > 0 and ix._df_of(["zebra"]) == {"zebra": 1}
    assert ix.delete_source("lookup") == 1
    assert len(ix) == 1 and not ix.has_key("Learned Tune", "Some Band") and ix._df_of(["zebra"]) == {}
    assert ix.search("zebra moonlight saxophone") == [] and ix.search("paper lanterns floating on the bay")
    assert ix.list_songs(offset=10**19) == []              # out-of-range offsets are clamped, not an OverflowError
