# Open-world recognition: real songs by their words

MelodyAI's own catalog holds whatever you import, and out of the box that is 48 *synthetic* demo songs.
The open-world search adds recognition of **real, well-known songs** from the **words** you sing, say or
type. Typed text and Whisper transcripts are searched in a local index of popular songs' lyrics and,
when online search is on, through two free services (LRCLIB and Deezer). Every candidate is checked
against its real lyrics before it is shown. Measured numbers are in [WORLD_RESULTS.md](WORLD_RESULTS.md),
which `make evaluate-world` generates. This page explains the design, the privacy rules and the limits.

> **Scope in one sentence.** Real songs are recognised by their *lyrics*. Humming or whistling a real
> song without words, or an instrumental clip, cannot be identified this way (see [Limitations](#11-limitations)).

## 1. Why

Users sang real songs from memory and played TikTok trend audio, and MelodyAI recognised none of them.
No matcher can find a song that is not among its references, and the reference set was 48 invented
melodies. The options for real songs were:

| Option | Status |
|---|---|
| Index Spotify / TikTok / YouTube audio | Not allowed by their terms ([SCALING.md](SCALING.md#1-constraints-that-are-not-technical)); no scraping and no unofficial APIs are used |
| Licensed recognition provider (ACRCloud: recordings and humming) | Implemented as an optional adapter; needs your own credentials ([SCALING.md](SCALING.md#3-path-a--licensed-provider-acrcloud)) |
| Melodic references of real songs (e.g. Lakh MIDI, MIR-QBSH) | Deliberately not downloaded for this project ([DATASETS.md](DATASETS.md#symbolic-melody-sources)) |
| **Recognise the words** | **Implemented here.** People singing from memory sing words, and the lyrics of popular songs are available from free services without a key |

## 2. Pipeline

```mermaid
flowchart LR
  A[Recording] --> W[Whisper ASR<br/>auto = large-v3-turbo<br/>VAD gate]
  T[Typed text] --> Q[Query text]
  W --> Q
  Q --> S1[Local world index<br/>SQLite FTS5 + phonetic keys]
  Q -.online search on.-> S2[Deezer search<br/>7-word windows + retries,<br/>title phrases]
  Q -.fallback.-> S3[LRCLIB title search]
  Q -.keys configured.-> S4[Genius / Claude]
  S1 & S2 & S3 & S4 --> R[Resolve lyrics<br/>index or LRCLIB]
  R --> V[Verify: fuzzy, phonetic,<br/>order-aware alignment]
  V --> G[Group versions / covers<br/>rank + popularity tie-break]
  G --> D[Decision: match /<br/>uncertain / no match]
  D --> E[Engine resolution<br/>local catalog vs open world]
  G -.score >= 0.5, history on.-> L[Learn into the index]
```

1. **Query text.** Typed text is searched as soon as the request arrives. A recording is transcribed
   by Whisper in *Auto*, *Lyrics* and *Sing + words* modes, and in *Recording/Clip* mode when the local
   fingerprint does not know the clip (a trend clip usually has sung words). The `"auto"` model setting
   uses the best Whisper model already downloaded (`large-v3-turbo` > `large-v3` > `medium` > `small` >
   `base` > `tiny`). A Silero VAD *gate* skips Whisper entirely when it finds too little speech, so hums,
   whistles, noise and instrumental music do not turn into invented words (details in
   [ALGORITHMS.md](ALGORITHMS.md#6-lyrics-melodyailyrics)). The search needs at least 3 words, one of them
   informative; otherwise the status is `no_query`.
2. **Propose.** The search runs in a background thread while the melody engines work, under one
   deadline (9 s by default). A source only *proposes* title and artist; nothing it says is trusted. The
   sources are listed in [section 3](#3-sources-and-what-leaves-the-machine). The local index answers
   first, alone for at most 0.5 s, and its answer decides how much the online sources are asked:
   * *settled* (a near-perfect alignment that clearly leads every other song, at least 3 informative
     query words, a popular recording with no version mark): the online sources are not asked. One
     cacheable Deezer title search for the indexed title checks that the index does not hold a popular
     cover of a more popular original it lacks; if a more popular recording with that title exists, by an
     artist the index knows (another of their songs is indexed: Deezer sometimes ranks a cover channel's
     upload above the original), and the index does not hold it, the answer is only probable.
   * *probable* (near-perfect but other songs share the words, or a partial alignment with a clear lead):
     the online sources get 1 s to answer, and the lyric lookups of their proposals 2 s from the first
     proposal that needs one (at most 5 lookups; the index's own near miss does not end them early).
   * otherwise: the online sources run with the whole deadline.
3. **Resolve lyrics.** A candidate's lyrics come from the local index when the song is there; otherwise
   only the most promising candidates are looked up on LRCLIB by title, artist and duration (at most 7
   per query, none after 3 once a candidate aligns >= 0.9). Promising means proposed by several sung
   windows or high up in Deezer's list, the title was sung, or the song is popular. Lookups start as soon
   as a source has answered.
4. **Verify.** The query is aligned with each candidate's lyrics (`melodyai/lyrics/align.py`). This is the
   Smith-Waterman token alignment of the catalog's lyrics engine, with two additions. *Phonetic*: a
   language-agnostic phonetic key lets misheard words match (Spanish b/v, s/z/c, ll/y, silent h; English
   ph/f, ck/k...). *Chunked*: lines are also aligned one by one, so skipped or reordered lines still
   count. Matched words must appear in order and close together, and candidates under a minimum score
   are dropped.
5. **Group and rank.** Records of one song (different uploads, remasters, covers, live or karaoke
   versions) are merged only with evidence that they are the same song: 70 % of the distinct lyric lines
   of the shorter one are in the other; or the titles are the same and they share at least 30 % and at
   least 3 lines (two LRCLIB transcriptions of one song often share only 30-65 % of their lines); or one
   is marked as a version of the same title (karaoke, live, cover, or a decorated title such as
   "(Spanish Mix)") and the same lines matched; or both records of one title hold the sung lines almost
   verbatim (a partly translated cover); or one is a medley / mashup that matched the same lines (it is
   never shown in place of the song itself). A shared title alone, or one shared line, is not enough. Records that share their lyrics are shown with the group's best
   alignment, by the most popular recording that aligns within 0.15 of it or is more than twice as
   popular as the best-aligned one: a little-known cover does not hide the original, and a recording of
   about the same popularity that lacks the sung lines (a verse only a remix has) does not stand for
   them. Version-marked records keep their own alignment, and the more popular one wins unless it
   aligns clearly worse. Final score = alignment score + a small popularity tie-break
   `0.03 * min(1, log10(1 + deezer_rank) / 6)`. A best match that is partial (< 0.9), rests on at most 2
   matched informative words and leads the next song by less than 0.2 is capped first (at 0.55, or just
   enough to stay first after the popularity tie-break): short generic lines align partly with many songs,
   so it can be shown first but not accepted.
6. **Decide.** `decide(..., "world")` applies the abstention rule used everywhere in MelodyAI:
   `match` if the top score reaches *accept* and its margin over the runner-up reaches *margin*,
   `uncertain` above the *uncertain* threshold, else `no_match`. The thresholds are provisional defaults
   in `DEFAULT_THRESHOLDS["world"]` (`melodyai/fusion/fusion.py`); `evaluate-world --calibrate` writes
   fitted ones to `configs/calibration.json`.
7. **Engine resolution** (`RecognitionEngine.prefer_world`). A local `match` backed by lyrics or
   fingerprint evidence wins. Otherwise a world `match` wins (`resolved_mode = "lyrics_online"`). Then
   comes a local melody-only `match`. Otherwise the better of the two uncertain lists wins. The world
   list is always returned in `engines.world`, so the UI can say "also found online".
8. **Learn.** See [section 6](#6-learning).

## 3. Sources and what leaves the machine

**Audio never leaves the machine through the open-world search.** Only text does: the typed or
transcribed words, and the titles and artists of candidate songs.

| Source | Online | Key | What is sent | Used for |
|---|---|---|---|---|
| `index`: local world lyrics index | no | — | nothing | candidates already verified, offline |
| `deezer`: Deezer search (`api.deezer.com`) | yes | no | overlapping 7-word windows of the transcript (at most 6); for a window that finds nothing, its first and last 5 words; title-like phrases when the transcript repeats a phrase, or as a last resort when every window and retry found nothing. At most 12 such requests per query. Later the title (+ artist) of the best candidates, for metadata, the original recording of a likely cover, and the namesake check of a settled index answer | Deezer's search also matches lyric lines (as an exact phrase); popularity (rank), cover, 30 s preview, link |
| `lrclib`: LRCLIB (`lrclib.net`) | yes | no | title, artist, album and duration of candidates whose lyrics are needed; title-like phrases of the transcript only as a fallback, when no other candidate matched (>= 0.5) and the transcript repeats a phrase | lyrics for verification (LRCLIB searches titles, not lyric text) |
| `genius`: Genius API | yes | `MELODYAI_GENIUS_TOKEN` | the informative words of the query (at most 200 characters); the token goes only in the `Authorization` header | full-text lyrics search. **Off unless the token is set** |
| `claude`: Claude API | yes | `MELODYAI_ANTHROPIC_API_KEY` | the transcript (at most 1,500 characters), asking for titles and artists only, never lyrics | candidates for badly misheard text. **Off unless the key is set** (and the optional `anthropic` package is installed: `pip install -e ".[llm]"`) |

Other traffic:

* **Popular-songs sync** (only when you start it): fixed chart and playlist queries to Deezer, then
  title, artist, album and duration lookups to LRCLIB. Nothing from your recordings is sent.
* **Results page**: album covers come from Deezer's CDN, and only for results that were searched online
  while online search is on. A 30 s Deezer preview streams only when you press play. Spotify and YouTube
  links are search pages built from the title and artist; they are opened only if you click them.
* **Licensed provider (ACRCloud)** is a separate opt-in feature and the only one that sends audio (at most
  12 s at 8 kHz). It needs operator credentials *and* the user's consent in Settings
  ([SCALING.md](SCALING.md)).

Privacy measures:

* **The switch.** *Settings → Online song search* (on by default). When it is off, only the local catalog
  and the local world index are searched and no covers are loaded. Deezer is then contacted only when you
  press a preview's play button, and LRCLIB / Deezer only by a sync you start. `MELODYAI_OFFLINE=1` stops
  all open-world traffic (online song search, sync, previews); the licensed provider and Whisper
  model downloads are governed by their own settings (see [section 9](#9-configuration)).
* **Hashed, short-lived cache.** Requests that carry your words are cached in `data/store/web_cache.db`
  under a SHA-256 of the URL instead of the plain URL, for at most 6 hours. If you do not keep a
  recognition history they are not cached at all. *Delete all history* also purges them
  (`WebClient.purge_private`).
* **No words in logs or URLs of MelodyAI's API.** The web client logs host, path and status only, never
  query strings, and httpx's own request log, which prints full URLs, is silenced. `POST /api/world/search`
  takes the words in a JSON body, so they never appear in URLs or access logs, and `/api/*` refuses
  cross-site requests. Outgoing searches necessarily carry the words in the provider's query string;
  nothing else about you is added.
* **Keys.** Read from the environment only, sent only to their own service in a header, and never part
  of a cache key, a cache row or a log line. The generic `ANTHROPIC_API_KEY` that many developers export
  for other tools is ignored unless `MELODYAI_CLAUDE_SOURCE=1`, so a key in your shell never silently
  sends transcripts to another service. When a keyed source is configured, Settings names it under the
  online-search switch ("Also sent to ...").
* **Identification.** Requests carry `User-Agent: MelodyAI/<version> (+project URL)`, as LRCLIB asks.
  Like any HTTP request, they reveal your IP address to the service.

## 4. The world lyrics index (`melodyai/world/index.py`)

One SQLite file, `data/store/world.db`, in WAL mode and shared safely between the server and a CLI
sync:

* `songs`: one row per song identity (normalised title + primary artist), with the Deezer metadata
  (rank, cover, link, ISRC, ids), a guessed language and the compressed lyrics.
* `fts_lines`: an FTS5 table with one row per distinct lyric line, holding its normalised words and the
  phonetic keys of its informative words.
* `df`: song-level document frequencies (IDF weights for retrieval and alignment).

A search is one FTS5 query (rare query words, their phonetic keys and some word pairs), aggregated per
song over its best 3 lines. The best candidates are then verified by the alignment of
[section 2](#2-pipeline). Lyrics are read only for the retrieved candidates, never all at once. With
24,640 songs an offline search (index only, verification included) took 48 ms p50 and 80 ms p95
([WORLD_RESULTS.md](WORLD_RESULTS.md)).

The index holds copyrighted lyrics fetched at run time. They stay on your machine: `data/store/` is
git-ignored and nothing in the repository (tests, fixtures, docs, reports) contains real lyrics. Do not
redistribute `world.db`.

## 5. Popular-songs sync (`melodyai/world/sync.py`)

The sync fills the index with the songs people are most likely to sing. It reads Deezer's global and
genre charts, the "Top &lt;Country&gt;" playlists of *Deezer Charts* (Spain, Mexico, USA, Argentina,
Colombia, UK and more), TikTok / viral lists, decade hits, Latin genres and classics. It uses only
playlists owned by Deezer's editors or the major labels' official curators. Standard and large also add
the top tracks of the most frequent artists. It then fetches each song's lyrics from LRCLIB, one request
at a time, at about 1.1 requests/s.

| Preset | Collects | Songs added (first run) | Time (first run) |
|---|---|---|---|
| `quick` | charts, TikTok / viral, decades, Latin genres, sing-along classics | ~1,900 (estimate shown in the UI) | ~33 min (estimate) |
| `standard` | + more countries and genres, yearly hits since 2000, top artists | **6,954 measured** (7,494 tracks looked up, 540 without lyrics on LRCLIB) | **2 h 08 min measured** |
| `large` | + hits since 1970, many more genres, the top songs of ~900 artists | **24,640 measured** in total (26,009 tracks collected) | **+5 h 13 min measured** after `standard` (about 7 h 20 min from empty) |

The measured `standard` run (2026-11-07, empty index) collected its 7,494 tracks from Deezer in 93 s,
then spent 2 h 07 min on lyrics: 8,310 LRCLIB requests and 0 errors. Afterwards `world.db` was about
45 MB. The `large` run that followed (2026-11-08) collected 26,009 tracks in 226 s, skipped the 6,954
indexed songs and the 540 recent misses, and looked up the other 18,515: +17,686 songs (827 without
lyrics, 2 errors, 20,234 LRCLIB requests) in 5 h 13 min, for 24,640 songs and about 141 MB on disk. A later
run looks up only the songs it has not seen, so syncing again or moving up a preset is much faster.

How to run it:

```bash
make world-sync PRESET=quick              # or standard / large; LIMIT=N keeps the N most important songs
python -m melodyai.cli world-sync --preset quick [--limit N] [--json]
```

In the web app: *Settings → Online song search → Popular songs*, then pick a preset. The sync runs as a
background job with progress ("Lyrics *done*/*total* · +*added* · about *time* left") and a stop button,
and you can leave the page.

* **Resumable.** Every song is committed as soon as it is processed. Ctrl-C once stops after the current
  song (twice aborts); running the same command again continues where it stopped. Songs without lyrics
  are remembered and retried only after 30 days (`data/store/world_sync.db`).
* **One sync at a time per store**, also across processes: a sync holds a lock on
  `world_sync.db.lock`. While the CLI syncs, the Settings page shows its progress and disables the
  buttons, and `POST /api/world/sync` answers 409. Two syncs would double the load on LRCLIB from your IP.
* **Polite.** Requests are paced, LRCLIB lookups are strictly sequential, and `Retry-After` on 429/503 is
  honoured ("LRCLIB asked us to slow down"). LRCLIB is a free, shared service: prefer `quick` or
  `standard` unless you need the long tail.

## 6. Learning

A song found online and verified with an alignment score of at least 0.5 is added to the local index
(`source = "lookup"`). The next search for it is instant and works offline. Learning follows the history
setting: without *Store recognition history* nothing is learned, and *Delete all history* also removes
the learned songs (`DELETE /api/recognitions` reports `learned_songs`). The CLI `world-search` learns
only with `--learn`, and the evaluation never learns (`learn=False`).

## 7. Settings and results in the web app

* **Online song search** card: the switch with its privacy note, the sources and their state (*local*,
  *active*, *off*, *needs a key*, *unavailable*), the number of songs indexed and the MB on disk, the last
  sync and the preset buttons.
* **Speech recognition** card: *Whisper model* defaults to **auto → best downloaded** (`large-v3-turbo`
  when present). If the selected model is missing, a *Download* button appears when the server allows
  downloads (`MELODYAI_ALLOW_MODEL_DOWNLOAD=true`). Otherwise run `make asr-model MODEL=large-v3-turbo`
  (a ~1.6 GB model file). The *Speech language* can be auto-detected or fixed to English or Spanish.
* **Results**: an online result shows a "Lyrics · online" badge, the matched lines, the sources that
  found it, a 30 s preview, and Deezer / Spotify / YouTube links. Next to a local match, "Also found
  online" lists world candidates; "Possible matches online" lists uncertain ones. When online search was
  off, the page says so and links to Settings.

API: `GET /api/world/status`, `POST /api/world/search`, `POST /api/world/sync`, `GET /api/world/songs`,
`GET /api/world/preview/{deezer_id}` ([API.md](API.md#open-world-search)).

## 8. Command line

```bash
# Words you type (here invented ones); prints titles, artists, scores and sources, never the matched lyrics
python -m melodyai.cli world-search "the lighthouse keeper counts the ships" [--offline] [--language es] \
    [--top-k 5] [--learn] [--json]
python -m melodyai.cli world-sync --preset quick            # = make world-sync PRESET=quick
python -m melodyai.cli asr-model --model large-v3-turbo     # = make asr-model MODEL=large-v3-turbo
python -m melodyai.cli evaluate-world --help                # = make evaluate-world [LIMIT=N] [ARGS=...]
```

`recognize` on the command line searches the local catalog only (its engine has no open-world search); use
`world-search` for words.

`evaluate-world` is the real-song evaluation (`melodyai/evaluation/world_eval.py`):

* `configs/eval_world.yaml` lists 100 songs by title, artist, language and tags only.
* Queries (chorus, verse, single line, misremembered, spoken and "sung" speech synthesis through
  Whisper, plus invented non-lyric negatives) are built from the lyrics at run time, in memory, with a
  seeded RNG.
* It runs the conditions `index` (offline), `online` (empty index + LRCLIB/Deezer) and `both`, always
  with `learn=False`.
* It writes raw rows (titles, scores, ranks, timings; no lyrics) to `data/reports/` and the summary to
  `docs/WORLD_RESULTS.md`. A hand-written notes block in that file survives regeneration.
* `--calibrate` writes the fitted `world` thresholds to `configs/calibration.json`. Use it only after the
  index is complete, because wrong-song scores grow with the index size.

## 9. Configuration

| Variable | Default | Effect |
|---|---|---|
| `MELODYAI_OFFLINE` | off | `1` / `true`: no open-world traffic (online song search, popular-songs sync, previews). The world index and the web cache still answer, online sources show as skipped, and sync and previews return 503. It does **not** cover the licensed provider (ACRCloud sends audio when its credentials are set and the user consented in Settings) or Whisper model downloads (`MELODYAI_ALLOW_MODEL_DOWNLOAD`): for a server without any network access, also leave those unset |
| `MELODYAI_STORE_DIR` | `data/store` | holds `world.db`, `web_cache.db` (capped at 256 MB) and `world_sync.db`; also `--store` on the CLI |
| `MELODYAI_GENIUS_TOKEN` | unset | enables the `genius` source |
| `MELODYAI_ANTHROPIC_API_KEY` | unset | enables the `claude` source (needs `pip install -e ".[llm]"`) |
| `MELODYAI_CLAUDE_SOURCE` | unset | `1`: allow the generic `ANTHROPIC_API_KEY` for the `claude` source |
| `MELODYAI_CLAUDE_MODEL` | `claude-opus-5-5` | model of the `claude` source |
| `MELODYAI_ALLOW_MODEL_DOWNLOAD` | `false` | `true`: Settings can download Whisper models, and `auto` downloads `large-v3-turbo` when nothing is cached |
| `MELODYAI_ASR_COMPUTE_TYPE` | `float32` on Apple Silicon CPUs, else `int8` | CTranslate2 compute type of Whisper |

The keys, `MELODYAI_OFFLINE` and `MELODYAI_ASR_COMPUTE_TYPE` are read from the process environment. Put
them in `.env` (copied into the server's environment at startup, without overriding exported variables)
or export them in the shell that starts the server (`export MELODYAI_GENIUS_TOKEN=...`, then `make dev`).
CLI commands read only the shell environment. Never commit keys.

## 10. Results so far

All numbers come from [WORLD_RESULTS.md](WORLD_RESULTS.md) (run `20261108T061526Z-world-final`, seed 7): the
100 songs of `configs/eval_world.yaml` against the finished 24,640-song index, typed variants (chorus, verse,
single line, misremembered), synthesised speech and a synthetic "sung" version through Whisper
`large-v3-turbo`, and 58 everyday non-lyric sentences.

* **Songs in the index are found.** Typed queries of indexed songs put the right song first 97% of the time
  (TikTok / viral group 99%), in 49 ms p50 offline. Index + online (the default): top-1 92%, confident match
  83%, 0.24 s p50 (the early exit skips the online sources when the index settles the query).
* **No confident wrong answers.** 0 false matches on song queries and 0 of 58 non-lyric sentences, in every
  condition, with the chosen thresholds (accept 0.62, margin 0.07; see [§ thresholds](#11-limitations)).
* **The online fallback alone** (empty index) found the right song first for 70% of the queries (chorus 93%,
  verse 100%, single line 64%, misremembered 64%), 0.6 s p50, about 2 LRCLIB requests per query.
* **Speech.** Synthesised speech: WER 0.04 and 96% top-1 offline. The synthetic "sung" clips remain the weak
  spot (WER ~0.5, 62% top-1): Whisper often decodes sung Spanish as English.
* **Abstention.** Most answers that are not a confident match still list the right song first (top-1 89% vs
  confident match 82% offline): the margin rule abstains on near-ties, which are often other records of the
  same song (versions, medleys) or a single line that really occurs in several songs.

## 11. Limitations

* **Humming and whistling without words cannot identify real songs.** The open world is lyrics-only.
  Recognising a hummed melody needs melodic references of the real songs. Melody datasets such as Lakh
  MIDI were deliberately not downloaded, and extracting melodies from commercial audio is not allowed
  (Spotify/TikTok terms) and not solved ([SCALING.md](SCALING.md#5-what-remains-hard-and-is-not-solved-here)).
  The keyed option is the licensed provider: ACRCloud's humming database and recording fingerprints.
* **TikTok clips are recognised through their sung words.** Whisper over music is weaker than over a
  voice alone, and the evaluation so far used synthetic speech without backing music. Trend audio that
  is instrumental, or only sped-up music without clear vocals, has no words to search. It needs audio
  fingerprinting against a catalog that contains the track: the local fingerprint index knows only
  imported tracks, so for commercial clips that means ACRCloud.
* **Coverage = the index + what LRCLIB and Deezer know.** A song without lyrics on LRCLIB (540 of the
  7,494 tracks of the standard sync) cannot be verified, so it is never returned. LRCLIB does not search
  lyric text. Deezer's search needs an exact phrase, so misremembered lines are weaker online (64% top-1
  online-only, against 89% offline for the same variant). Very short lines (4 to 6 words) can find
  nothing online.
* **The first query needs the internet** unless the song is already in the index (synced or learned).
  Online search of a song that is not indexed takes about 0.5-1 s (p95 a few seconds). Offline, only the
  index is searched.
* **Covers and versions vs the original.** Covers, karaoke tracks and live versions share the words.
  Grouping prefers the most popular recording when the evidence says it is the same song, but it can
  pick the wrong artist: when two recordings of about the same popularity both hold the sung lines, when
  the original's LRCLIB transcription differs too much to be grouped (in the pilot a little-known cover
  scored 1.00 and the original 0.71; same-title records sharing 30 % of their lines are grouped since), or
  when the index holds a popular cover and the original is not found online. The evaluation counts the
  right title as correct.
* **Thresholds favour precision.** They were calibrated on the complete index (`configs/calibration.json`,
  `world_calibration`): the strict fit (accept 0.62, margin 0.07) gave 0 false matches; the looser automatic
  fit (accept 0.52, margin 0) would have accepted 9 % more queries but with 3.6 % confident wrong answers.
  Re-calibrate (`make evaluate-world ARGS=--calibrate`, then review the proposal) if the index grows a lot.
* **Singing is hard for ASR.** Whisper is far less accurate on singing than on speech: wrong words,
  wrong language, or nothing at all when every segment is filtered out. Singing a line slowly and
  clearly, or typing the words you remember, works best.
* **Languages.** The language prior, stop words and phonetic keys are tuned for Spanish and English.
  Portuguese, French and Italian songs are indexed, but they are less tested.
* **Free third-party services.** LRCLIB and Deezer have no service guarantee and may change, throttle or
  disappear; MelodyAI then falls back to the local index. Their data (lyrics transcriptions, metadata)
  contains errors that MelodyAI cannot fix.
* **Not validated yet:** real human singing recordings and real TikTok clips (only synthesised speech so
  far), the full 100-song evaluation on the complete index, and the keyed sources (Genius, Claude)
  against live services.

## 12. Code map

| Module | Role |
|---|---|
| `melodyai/world/web.py` | polite cached HTTP client: per-host rate limits, `Retry-After`, SQLite cache, private (hashed) entries, deadlines, offline mode |
| `melodyai/world/lrclib.py`, `deezer.py` | LRCLIB and Deezer API clients |
| `melodyai/world/text.py` | title / artist normalisation, song keys, language guess, title phrases |
| `melodyai/world/index.py` | world lyrics index (SQLite FTS5 + phonetic keys) |
| `melodyai/world/sources.py` | candidate sources (index, LRCLIB, Deezer, Genius, Claude) |
| `melodyai/world/search.py` | `WorldSearch`: propose, resolve, verify, group, rank, learn |
| `melodyai/world/sync.py` | popular-songs sync, presets, resumable state, cross-process lock |
| `melodyai/lyrics/align.py` | fuzzy, phonetic, chunked lyrics alignment (shared with the catalog's lyrics engine) |
| `melodyai/lyrics/asr.py` | faster-whisper for sung lyrics: `auto` model, VAD gate, language prior, hallucination guards |
| `melodyai/engine.py` | runs the world search next to the local engines and resolves between them |
| `backend/app/api/world.py` | `/api/world/*` endpoints |
| `melodyai/evaluation/world_eval.py`, `configs/eval_world.yaml` | real-song evaluation |
