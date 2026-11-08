"""Open-world candidate sources: lyric windows for Deezer's phrase-exact search and their retries.

No network: Deezer answers come from ``httpx.MockTransport`` over a tiny invented catalog whose search, like the
real one (see :class:`melodyai.world.sources.DeezerTitleSource`), only finds a song when the query is a
contiguous run of its lyric words (or the words of its title). All lyrics, titles and artists are invented.
"""

import time

import httpx

from melodyai.lyrics.normalize import tokenize
from melodyai.world.deezer import DeezerClient
from melodyai.world.sources import DeezerTitleSource, lyric_windows
from melodyai.world.web import HostLimit, WebClient, request_deadline

FAST = {"api.deezer.com": HostLimit(1000, 1000.0, 6), "lrclib.net": HostLimit(1000, 1000.0, 3)}

KITES = ("Seven paper kites above the quarry\nTangled in the wires of the morning train\n"
         "Mother calls our names across the meadow\nNobody answers till the evening rain")
FERRY = ("Ventanas de azufre en el ferry de las once\nLos marineros cuentan monedas de cobre\n"
         "Nadie conoce el idioma del faro")
DB = [(501, "Paper Kites", "Orchard Nine", 700000, KITES), (502, "Ferry de las Once", "Marea Baja", 650000, FERRY)]


def dz(tid, title, artist, rank):
    return {"id": tid, "title": title, "title_short": title, "link": f"https://www.deezer.com/track/{tid}",
            "duration": 200, "rank": rank, "type": "track", "artist": {"id": tid + 1, "name": artist},
            "album": {"id": 7, "title": "Album"}}


def contiguous(needle: list[str], hay: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


class PhraseDeezer:
    """Deezer /search that matches a contiguous run of lyric words or every word of the title."""

    def __init__(self):
        self.queries: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        q = dict(request.url.params).get("q", "")
        self.queries.append(q)
        words = tokenize(q)
        hits = [dz(tid, title, artist, rank) for tid, title, artist, rank, lyrics in DB
                if contiguous(words, tokenize(lyrics)) or (words and set(tokenize(title)) <= set(words))]
        return httpx.Response(200, json={"data": hits, "total": len(hits)})


def source(handler, **kw) -> DeezerTitleSource:
    web = WebClient(transport=httpx.MockTransport(handler), host_limits=FAST)
    return DeezerTitleSource(DeezerClient(web), max_phrases=0, **kw)


# ---------------------------------------------------------------------------- windows
def test_a_short_query_is_one_window_with_its_two_ends_as_retries():
    wins = lyric_windows("oh yeah tangled in the wires of the morning")      # fillers dropped: 7 words
    assert wins == [("tangled in the wires of the morning", ["tangled in the wires of", "the wires of the morning"])]
    # up to size + 1 words stay one window; five words or fewer have no retry windows
    assert [len(w.split()) for w, _ in lyric_windows("seven paper kites above the quarry tangled wires")] == [8]
    assert lyric_windows("seven paper kites above") == [("seven paper kites above", [])]
    assert lyric_windows("oh oh yeah") == [] and lyric_windows("paper kites") == []


def test_a_run_on_query_becomes_overlapping_windows_in_sung_order():
    text = ("seven paper kites above the quarry tangled in the wires of the morning train "
            "mother calls our names across the meadow")
    words = text.split()
    assert len(words) == 21
    wins = lyric_windows(text, size=7, stride=4, max_windows=10)
    # every 4 words, and the last window ends with the query
    assert [w for w, _ in wins] == [" ".join(words[s:s + 7]) for s in (0, 4, 8, 12, 14)]
    for w, subs in wins:
        assert subs == [" ".join(w.split()[:5]), " ".join(w.split()[-5:])]
    # only the most informative windows are kept, still in sung order
    rare = {"quarry": 9.0, "meadow": 9.0}
    kept = lyric_windows(text, size=7, stride=4, max_windows=2, idf=lambda t: rare.get(t, 1.0))
    assert [w for w, _ in kept] == [" ".join(words[0:7]), " ".join(words[14:21])]


def test_windows_made_of_function_words_are_not_sent():
    assert lyric_windows("and the one that was in the way of it all") == []
    text = "and in the way of it all to the quarry kites"
    wins = lyric_windows(text, size=4, stride=2)
    assert [w for w, _ in wins] == ["to the quarry kites"]          # the only window with two informative words


# ---------------------------------------------------------------------------- Deezer source
def test_run_on_text_across_two_lines_is_found_with_short_windows():
    """Regression: run-on transcripts were cut into 12-word pieces that rarely match Deezer's lyric search."""
    handler = PhraseDeezer()
    src = source(handler)
    text = "tangled in the wires of the morning train mother calls our names across the meadow"
    cands = src.propose(text, tokenize(text), None)
    assert cands and cands[0].title == "Paper Kites" and cands[0].prior == 1.0
    assert handler.queries and all(len(q.split()) <= 8 for q in handler.queries)


def test_a_misheard_word_is_worked_around_by_the_retry_windows():
    handler = PhraseDeezer()
    src = source(handler)
    # "cobre" misheard as "pobre": the only window holding it finds nothing; one of its ends is intact
    text = "los marineros cuentan monedas de pobre"
    cands = src.propose(text, tokenize(text), "es")
    assert handler.queries[0] == "los marineros cuentan monedas de pobre"
    assert "los marineros cuentan monedas de" in handler.queries
    assert [(c.title, round(c.prior, 2)) for c in cands] == [("Ferry de las Once", 0.7)]
    # a window that found something is not retried
    handler2 = PhraseDeezer()
    source(handler2).propose("los marineros cuentan monedas de cobre", [], "es")
    assert handler2.queries == ["los marineros cuentan monedas de cobre"]


def test_window_plan_and_title_queries_are_deduplicated():
    src = DeezerTitleSource(None, max_phrases=2)
    text = "paper kites paper kites seven paper kites above the quarry"
    qs = src.queries(text, tokenize(text))
    kinds = [k for k, _, _ in qs]
    assert kinds.count("window") >= 1 and "title" in kinds
    assert len({q for _, q, _ in qs}) == len(qs)
    # no repeated phrase (not chorus-like): a "title" would just be a 2-4-word window, so none is searched
    verse = "seven paper kites above the quarry tangled in the wires"
    assert {k for k, _, _ in src.queries(verse, tokenize(verse))} == {"window"}


def test_one_stalled_window_does_not_lose_the_others():
    """Regression: the Deezer source waited for all its windows, so one stalled request (or a quota pause) lost
    every candidate at the search's proposal deadline. It now answers with what it has shortly before the
    deadline, and none of its requests is sent after it answered."""
    base = PhraseDeezer()
    started: list[float] = []

    def handler(req):
        started.append(time.perf_counter())
        if "quarry" in dict(req.url.params).get("q", ""):
            time.sleep(1.5)
        return base(req)

    src = source(handler)
    text = "seven paper kites above the quarry\nmother calls our names across the meadow"
    with request_deadline(time.monotonic() + 1.0):
        t = time.perf_counter()
        cands = src.propose(text, tokenize(text), None)
        returned = time.perf_counter()
    assert returned - t < 0.9
    assert cands and cands[0].title == "Paper Kites"
    time.sleep(1.0)
    assert started and all(s < returned for s in started)


def test_text_that_is_no_lyric_costs_a_bounded_number_of_requests():
    handler = PhraseDeezer()
    src = source(handler, max_requests=8)
    text = ("the quarterly budget meeting moved to thursday afternoon so please bring the printed invoices "
            "and the spreadsheets for the regional auditors")
    windows = src.window_plan(text)
    assert len(windows) == 5 and all(len(subs) == 2 for _, _, subs in windows)
    assert src.propose(text, tokenize(text), "en") == []
    assert len(handler.queries) == 8                     # 5 windows + 3 of their 10 retry windows


def test_repeated_phrase_tells_a_chorus_from_a_verse():
    from melodyai.world.sources import repeated_phrase

    assert repeated_phrase(tokenize("paper kites over the quarry paper kites over the town"))
    assert repeated_phrase(tokenize("tangled wires again tangled"))                 # a long word, twice
    assert not repeated_phrase(tokenize("seven paper kites above the quarry tangled in the wires"))
    assert repeated_phrase(tokenize("in the night and in the night"))
    assert not repeated_phrase(tokenize("and in the and in the"))                   # function words only
    assert not repeated_phrase(tokenize("kites kites"))                             # one short word


def test_retries_go_to_the_most_informative_empty_windows_whatever_answers_first():
    """Regression: windows took retry requests from the budget as their answers arrived, so whether the one
    intact retry window was ever sent depended on response timing."""
    base = PhraseDeezer()

    def handler(req):
        if dict(req.url.params).get("q", "").endswith("wires of the evening"):
            time.sleep(0.3)                              # the most informative window answers last
        return base(req)

    rare = {"tangled": 9.0, "wires": 9.0}
    src = source(handler, max_requests=5, idf=lambda t: rare.get(t, 1.0))
    # three windows, each broken by a misremembered word; budget for two retry windows
    text = "tangled in the wires of the evening rain mother calls our names across the meadows"
    plan = src.window_plan(text)
    assert [len(w.split()) for _, w, _ in plan] == [7, 7, 7]
    cands = src.propose(text, tokenize(text), None)
    assert base.queries[3:] == plan[0][2]                # both retries of the first (most informative) window
    assert [(c.title, round(c.prior, 2)) for c in cands] == [("Paper Kites", 0.7)]
