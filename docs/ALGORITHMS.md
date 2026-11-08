# Algorithms

This document describes what the code does, with the parameters actually used. For the
literature behind each choice see [RESEARCH.md](RESEARCH.md); for measured behaviour see
[RESULTS.md](RESULTS.md).

## 1. Audio front-end (`melodyai/audio`)

1. **Decode** to mono float32 at 16 kHz (`soundfile`; `ffmpeg` pipe fallback for M4A/WebM/AAC).
2. **Validate**: empty, undecodable, < 0.05 s, > limit seconds or > limit MB → `AudioError`.
3. **Preprocess**: remove DC, 2nd-order Butterworth high-pass at 60 Hz (zero-phase), trim
   leading/trailing frames more than 40 dB below the loudest frame (100 ms padding), peak-normalise
   to −1 dBFS. No spectral denoising (it creates "musical noise" that pitch trackers follow).
4. **Quality indicators**: duration, active duration, RMS/peak dBFS, clipping ratio, a rough SNR
   (90th vs 10th percentile frame energy) and a `silent` flag.

## 2. Pitch and notes (`melodyai/pitch`)

| Extractor | Implementation | Frame hop | Range (Hz) by profile |
|---|---|---|---|
| `pyin` (default) | `librosa.pyin`, resolution 0.1 semitone, frame = next power of two ≥ 2.2 periods of fmin | 10 ms | hum/sing 65–1400, whistle 400–4000 |
| `crepe` | `torchcrepe` *tiny*, periodicity median-filtered, voiced if > 0.35 | 10 ms | same (≤ 1997 Hz) |
| `basic_pitch` | Spotify Basic Pitch note events → skyline → frame f0 | 10 ms | same |

**Cleaning**: frames with voicing probability < 0.15 or voiced islands shorter than 50 ms are
dropped (breaths, clicks). **Octave folding**: excursions of ±12/±24 semitones from a running
median that last ≤ 80 ms are folded back (longer ones are genuine leaps).

**Voiced contour** (used by the contour matchers): voiced frames concatenated (rests removed,
because people rarely reproduce rests), block-median downsampled from 100 to 20 frames/s.

**Note segmentation**: inside each voiced run, a 5-frame median filter, then a new note starts when
(a) the pitch departs from the running note median by > 0.8 semitone for ≥ 50 ms (legato) or
(b) the RMS envelope shows a ≥ 6 dB dip within ±80 ms (re-articulated repeated notes). A note's
pitch is the median of its central 60% of frames (ignoring scoops). Notes < 70 ms are merged into
the closest-pitch neighbour.

A query is **informative** if it has ≥ 1.5 s of voiced contour, ≥ 3 notes and ≥ 1 semitone of range;
otherwise no melody matching is attempted (this is what keeps noise or a held tone from matching).

## 3. Representations (`melodyai/features`)

* **Reference contour**: notes rendered at 20 frames/s, each note held until the next onset.
* **Windows**: for lengths L ∈ {6, 9, 13, 18, 25} notes, starting every `max(1, L // 5)` notes;
  each window stores its time range, its median pitch (absolute key) and a 96-point contour with
  the median removed (key invariance) and linear time normalisation (tempo invariance).
  48 demo songs → ~14,000 windows.
* **Chroma**: CQT chroma (C2–C8, 10 frames/s), frames below 5% of the max RMS zeroed.

## 4. Classical matchers (`melodyai/matching/algorithms.py`)

All return candidates sorted by distance with a **relative score** `exp(−distance/τ)` in [0, 1].
Scores are comparable within an algorithm, not across algorithms, and are not probabilities.

### 4.1 Window DTW (`dtw`)
1. Query vector = 96-point median-centred contour.
2. Keep windows whose duration is 0.4–2.6× the query's voiced duration (tempo ratio bound).
3. Banded DTW (band = 15% = 14 points), cost `min(|q−r|, 4 semitones)` (wrong notes cannot dominate),
   additive penalty 0.15 on non-diagonal steps, normalised by path length.
4. Residual key refinement: shift = median(q − w); final distance = min(unshifted, shifted).
5. Per song: best window → matched time range. τ = 1.
6. Above 30,000 candidate windows, a linear-scaling L1 prefilter keeps the closest 30,000 first.

### 4.2 Subsequence DTW (`subseq_dtw`)
Query contour (10 frames/s) linearly rescaled by tempo factors {0.55, 0.75, 1.0, 1.35, 1.8};
for each candidate key (the rounded median pitches of the song's windows) a subsequence DTW
(free start/end in the reference, penalty 0.2, clip 4) against the *whole* song contour; then one
refinement pass with the median residual along the optimal path. Gives the precise alignment
used for visualisation. τ = 1.

### 4.3 Chroma DTW (`chroma_dtw`)
Query and reference CQT chroma; query rescaled by {0.7, 1.0, 1.4}; 12 circular key shifts;
cosine-cost subsequence DTW (penalty 0.05). τ = 0.25. Baseline from the chroma QBH literature.

### 4.4 Note alignment (`note_align`)
Repeated notes merged on both sides, then sequences of (interval, log IOI-ratio). Similarity
`0.8·clip(1 − |Δinterval|/1.5) + 0.2·clip(1 − |Δlog-ratio|)` in [−1, 1]; Smith-Waterman with
gap 0.7; score normalised by the number of query intervals; distance = 1 − similarity. τ = 0.35.
Key- and tempo-invariant by construction; tolerant to missing/extra notes.

### 4.5 Linear scaling (`ls`)
Window vectors compared with a median-aligned mean absolute difference (no warping). Fast
candidate generator / ablation.

### 4.6 Ensemble (`ensemble`, default)
Weighted mean of the relative scores of `dtw`, `note_align` and `subseq_dtw` (missing = 0), each
member contributing its top-20 songs. `ensemble_neural` adds `neural_rerank`.

### Invariance properties (tested)
* **Transposition** (Test A): C–D–E–G and F–G–A–C have identical centred contours
  (`tests/unit/test_dtw_and_features.py::test_A_transposition_invariance`); catalog-level tests
  transpose excerpts by +5/−7 semitones.
* **Tempo** (Test B): uniform tempo changes vanish after time normalisation; accelerando is
  absorbed by DTW (+ residual key search).
* **Mid-song passages** (Test C): windows start every few notes across the whole song.

## 5. Neural matcher (`melodyai/models`, `melodyai/matching/neural.py`)

* **Encoder**: input = 96-point centred contour → channels (pitch/12, Δpitch/4) → Conv1d stem →
  3 residual dilated conv blocks (dilation 1, 2, 4; GroupNorm, GELU) → learned positional
  embedding + 2-layer pre-norm Transformer (4 heads, d=128) → attention pooling → MLP → L2-normalised
  128-d embedding. ≈0.8 M parameters. Shared weights for queries and references.
* **Search**: every window embedded once per model (cached per song on disk); FAISS
  `IndexFlatIP` (exact cosine) or `IndexHNSWFlat` (M=32, efSearch=128, approximate); top-300
  windows → per-song max → candidates (`neural`, `neural_hnsw`).
* **Re-ranking** (`neural_rerank`): the 300 retrieved windows are re-scored with the window-DTW
  distance (same parameters as 4.1).
* Training: see [TRAINING.md](TRAINING.md).

## 6. Lyrics (`melodyai/lyrics`)

* **Normalisation**: NFKC → case-fold → join contractions (`don't`→`dont`) → strip accents
  (`canción`→`cancion`) → remove punctuation → whitespace tokens.
* **Index**: SQLite FTS5 tables with `unicode61 remove_diacritics` (BM25 word retrieval) and
  `trigram` (substring retrieval for misspellings). For ≤ 500 lyric documents every song is scored;
  above that, candidates are the union of the two FTS5 queries.
* **Scoring** (`melodyai/lyrics/align.py`, shared with the open-world search): Smith-Waterman over
  tokens. A query/reference token pair scores +w if RapidFuzz
  ratio ≥ 84, +w/2 if ≥ 67, −w/2 otherwise, where w = IDF of the query token (× 0.3 for stop-words);
  gap penalty 0.4·mean(w). Coverage = aligned weight / total weight. Final score =
  0.8·coverage + 0.2·RapidFuzz partial-ratio of the query against the matched lines (±1 line).
  Words must appear *in order and close together*, which is what separates "the song" from "a song
  about the same topic". Semantic embeddings are intentionally not used.
* **ASR** (`melodyai/lyrics/asr.py`): faster-whisper, `tiny` … `large-v3`, `large-v3-turbo`. The default
  setting `auto` uses the best model already downloaded, in the order `large-v3-turbo`, `large-v3`,
  `medium`, `small`, `base`, `tiny`. `large-v3-turbo` (~1.6 GB) is the model that makes sung lyrics usable;
  `base` mangles most sung lines. It runs on CPU or CUDA (no Apple MPS in CTranslate2): float32 on Apple Silicon, int8
  elsewhere (`MELODYAI_ASR_COMPUTE_TYPE`).
  * **VAD gate**: Silero VAD (threshold 0.5, min silence 400 ms) only decides *whether* Whisper runs.
    Below `MIN_VAD_SPEECH_S` = 1.5 s of detected speech, Whisper is skipped, which keeps hums, whistles,
    noise and instrumental music out. It does not cut the audio, because it under-detects singing.
  * **One encoder pass per clip**: Whisper encodes a padded 30 s window, and for a short query the encoder
    is most of the cost. An `EncoderCache` on each loaded model memoises `encode` by the window's bytes
    (shape, dtype, 128-bit BLAKE2 digest), and the language is detected on exactly the window the decoder
    sees, so language detection and every decoding of a clip <= 30 s share one encoder pass. Every
    decoding also has a token budget (`MAX_TOKENS_PER_S` = 30 tokens per second of audio, at most 400),
    which stops runaway loops early. In an interleaved A/B on the same machine this cut the time per
    sung or spoken clip by about 40 % (4.9 s instead of 8.2 s mean, under the load of a parallel sync).
  * **Language**: when no language is given, it is chosen as `argmax p(lang | audio) × prior(lang)`
    (es, en 1.0; pt 0.35; it, fr 0.3; de 0.2; others 0.05). Sung Spanish is often misdetected.
  * **Decoding**: beam 5, temperature 0 and no temperature fallback (on singing it turned empty
    outputs into fluent invented text), not conditioned on previous text.
  * **Segment filter**: segments are dropped if (no-speech > 0.6 and avg logprob < −0.5), if the avg
    logprob is < −1.2, if the compression ratio is > 2.8 (repetition loops), or if the text is a known Whisper
    hallucination ("thank you", "gracias", "subtítulos…"). Bracketed annotations and music symbols are
    stripped.
  * **Words required**: at least three real words must remain, and vocables such as "la la", "mm" or
    "ooh" do not count. Otherwise the further decodings below are tried, and if none is usable,
    `speech_detected = false` and the lyrics engines abstain instead of inventing text.
  * **Further decodings** (all on the cached encoder output):
    * *Alternative sung language*: when the prior picked a language other than es/en with a posterior
      < 0.9, the clip is also decoded (beam 5) in the likelier of es/en. Of the usable decodings, the one
      with the best `decoding_score` wins: the duration-weighted mean log-prob of its kept segments and of
      the segments it failed on (low log-prob, repetition, loop, hallucination; each counted as at most
      −1.2), so one fluent line cannot beat a complete transcript of the clip. A real pt/fr/it/de
      recording keeps its language.
    * *Greedy retry*: when no decoding is usable although the VAD heard a voice, the clip is decoded once
      more in the same language with beam 1 (never the other language: a failed English chorus re-decoded
      as Spanish became a confident false match). Its transcript needs 3 distinct words, or the first
      decoding must have heard its words (≥ 50 % agreement): greedy decoding of sung vocables returned
      one invented word three times, while a chorus that repeats one word is heard by both decodings.
    * *Relaxed tier*: then a window above the compression-ratio limit is kept with its repetition
      collapsed (a chorus that really repeats a phrase) if it emits at most 5 words per second of audio
      (decoder loops: 6.4–12.4; sung choruses: 1.2–3.8), has at least 3 distinct words and the other
      decoding heard at least 50 % of them (looped hallucinations on reversed speech agreed 0–17 %).
  * **Diagnostics**: `Transcript.passes` lists, without any text, what each decoding produced: language,
    beam, segment counts, the rule that dropped each segment, whether it was usable (strict / relaxed),
    its agreement and whether it was used.

## 7. Fusion and abstention (`melodyai/fusion`)

* **Weighted** (default): Σ_e w_e·r_e·score_e / Σ_e w_e·r_e, where r_e is a reliability heuristic:
  melody r = clip(voiced_s/4, 0.2, 1)·clip(1.4·pitch_confidence, 0.2, 1); lyrics r = 1 for typed
  text, ASR r from mean logprob × language probability (floored at 0.3 in fusion).
* **RRF**: Σ 1/(60 + rank); **CombMNZ**: Σ scores × number of engines that found the song; both
  rescaled to [0, 1] by their maximum attainable value for display.
* **Learned**: logistic regression on per-song features (both engines' score, reciprocal rank,
  reliability, agreement), trained by the benchmark on its dev half only.
* **Abstention**: `match` if top score ≥ accept and margin ≥ margin threshold; `uncertain` if
  top ≥ uncertain threshold; otherwise `no_match` ("No sufficiently reliable match was found").
  Thresholds per algorithm come from `configs/calibration.json`, written by `make evaluate`
  (accept = 5% false-accept rate on out-of-catalog dev queries, uncertain = 25%), else conservative
  defaults. If a Platt model exists for the algorithm, the API also returns
  `calibrated_probability`, which the UI labels as calibrated on synthetic queries.

## 8. Auto mode

1. If only text is provided → lyrics search.
2. Otherwise: profile detection (YIN median f0 over loud frames > 700 Hz → whistle), melody
   analysis; Whisper runs if enabled and the model is available locally.
3. Both engines produced candidates → fusion; only one → that engine; none → `no_match`.
4. Whenever there is query text (typed, or a transcript), the open-world search (section 12) runs in a
   background thread alongside the local engines. Its answer replaces the local one by the rules in
   section 12 (`resolved_mode = "lyrics_online"`).

## 9. Audio fingerprinting (`melodyai/fingerprint`, `clip` mode)

For recordings of the actual track (a TikTok/Reels clip, a song playing nearby) rather than a
person singing it.

* **Peaks**: STFT (n_fft 4096, hop 512 = 32 ms at 16 kHz) → fixed sparse mapping onto a
  log-frequency grid (24 bins/octave, 150 Hz–5 kHz, ~121 bins) → 9×9 local maxima within 45 dB of the
  loudest bin → strongest 18 peaks per second, frequency refined by parabolic interpolation.
* **Hashes** ("quads"): for each anchor, its 5 nearest later peaks within 1.5 s and ±15 semitones;
  every ordered triple of them gives `(Δf₁, Δf₂, Δf₃, Δt₁, Δt₂, Δt₃)` packed into 32 bits (frequency
  differences in 50-cent bins, time differences in 64 ms steps). No absolute frequency → invariant to
  transposition. 10 hashes per anchor, ~146 postings per second of audio.
* **Speed**: query peak times are rescaled by 22 factors between 0.75 and 1.40 (≈3% steps) before
  hashing; resampling-style speed changes also transpose, which the invariance absorbs.
* **Stage 1 vote**: each hit votes for (song, time offset in 64 ms bins, pitch offset in 50-cent bins)
  with weight `log(1 + n_ref / postings(hash))` (IDF); a song's evidence is its best cell over all speeds.
* **Stage 2 verification** (top 10 by evidence): query peaks are mapped into reference coordinates;
  score = fraction landing within ±2 frames / ±1 bin of a reference peak, maximised over speed (±3% in
  0.5% steps around the vote), offset (±4 frames) and pitch (±1 bin, with a +0.02 hysteresis in favour of
  the vote). The transposition reported is the median frequency difference of matched peak pairs.
* **Decision**: `match` if score ≥ 0.50 and margin ≥ 0.10 over the runner-up, `uncertain` ≥ 0.42
  (defaults set from the synthetic benchmark; recalibrate on real data).
* In **Auto** mode fingerprinting runs first (tens of ms); a `match` ends the search, otherwise the
  melody/lyrics engines run as before. Hums must not match recordings: the false-accept rate of hummed
  queries against the fingerprint index is part of the scale benchmark.

## 10. Large-catalog melody search (`melodyai/matching/scalable.py`, `melodyai/indexing/ivfpq.py`)

* **IVF-PQ** (pure NumPy, FAISS-compatible design): spherical k-means coarse quantiser (`nlist`
  lists, capped at N/39); residuals split into `m` = 16 sub-vectors of 8 dims, each encoded with one of
  256 k-means sub-centroids → 16 B per vector. Inner-product search decomposes exactly as
  `q·c + Σⱼ qⱼ·codebookⱼ[codeⱼ]`, so one 16×256 lookup table serves all probed lists. Incremental
  `add` goes to a delta scanned with the same scoring; `consolidate()` merges; `save/load` memory-maps.
* **Stage 1**: query contour → encoder embedding → top 600 windows (nprobe 32) → per-song max → 50
  candidate songs. Reference windows for this stage: lengths 9/13/18 notes with hops 3/5/7 (~75 per song).
* **Stage 2**: window DTW and note alignment (sections 4.1 and 4.4) on the candidates only, features
  loaded through an LRU cache; final score = mean of the two relative scores.

## 11. Licensed provider (`melodyai/providers/acrcloud.py`)

ACRCloud Identification API: `POST https://<host>/v1/identify`, multipart fields `sample`,
`sample_bytes`, `access_key`, `data_type=audio`, `signature_version=1`, `timestamp`,
`signature = base64(HMAC-SHA1(secret, "POST\n/v1/identify\n<key>\naudio\n1\n<timestamp>"))`.
The loudest 12 s of the recording are sent as 8 kHz WAV; `metadata.music` (recordings, score 70–100)
or `metadata.humming` (score 0–1) are parsed into title/artists/album/ISRC/links. Hosts must match
`identify-*.acrcloud.(com|cn)`. Status 1001 = no result. Policy `fallback` (default) only calls the
provider when the local decision is not `match`.

## 12. Open-world lyrics search (`melodyai/world`)

Real songs recognised by their words. The design, privacy rules, popular-songs sync and limits are in
[OPEN_WORLD.md](OPEN_WORLD.md); measured numbers are in [WORLD_RESULTS.md](WORLD_RESULTS.md).

* **Query**: typed text or a Whisper transcript; at least 3 tokens, one of them informative.
* **World lyrics index** (`world/index.py`, SQLite FTS5 `unicode61 remove_diacritics 2`): one row per
  distinct lyric line, with normalised words and the phonetic keys of its informative words. Retrieval
  is one `MATCH` with an OR over the rarest query words (within a song-frequency budget) and their
  phonetic keys (BM25 weight 0.7), plus pairs of phonetic keys for common words. Line BM25 is summed over
  a song's best 3 lines, and the best 150 songs are kept. IDF blends song-level document frequency with
  a prior that gives function words a low weight.
* **Scheduling**: everything runs under one deadline (9 s); a source failure is recorded, never raised.
  The local index answers first, alone for at most 0.5 s. Its answer is *settled* when the best
  alignment leads every other song (not its own versions) by 0.15 at a score of 0.9 or more, rising
  linearly to a lead of 0.3 at 0.7, the query has at least 3 distinct informative words and the song's
  most popular recording has rank ≥ 300,000 and no version mark: the online sources are then not asked,
  except for one cacheable Deezer title search, which makes the answer only probable when it finds a more
  popular recording with that title by another artist that the index does not hold, by an artist with
  other songs in the index (the indexed one may be a popular cover; an unknown artist is more often a
  cover channel that Deezer ranks above the original). The answer is *probable* when it falls short only on the words or the version mark,
  when it aligns ≥ 0.9 without a clear lead, or when it aligns ≥ 0.5 with a lead of 0.3: the online
  sources then get 1 s to answer and the lyric lookups of their proposals 2 s from the first one (at most
  5, and the index's own candidates do not count as the strong match that ends lookups early). Otherwise the online sources run with the whole deadline (proposals until 55 % of it, lookups until
  85 %).
* **Online sources** (only while online search is on): Deezer track search with overlapping 7-word windows
  of the sung words (stride 4, at most 6, the most informative by IDF, in sung order; Deezer's plain search
  matches lyric text, but only as an exact phrase). A window that finds nothing is retried with its first
  and its last 5 words (a misheard word sits at one end or in the middle), most informative window first,
  within 12 requests per query. Title-like phrases (n-grams favoured when repeated, as choruses repeat the
  title) are searched with the windows when the transcript repeats a phrase, else only as a last resort
  when every window and retry found nothing, keeping tracks whose whole title was sung. LRCLIB title search
  is a fallback: it runs only when the other candidates, once verified, matched nothing (< 0.5) and the
  transcript repeats a phrase (else its status says "no repeated phrase"). Optional keyed sources: Genius
  search (`MELODYAI_GENIUS_TOKEN`) and Claude (`MELODYAI_ANTHROPIC_API_KEY`). Lyrics of proposed songs
  come from the index, else from LRCLIB (`/api/get`, then `/api/search`), at most 7 per query, best
  candidates first (proposed by several windows or high up, title sung, popular), none after 3 once one
  aligns ≥ 0.9.
* **Verification**: every candidate, whatever proposed it, is aligned against its real lyrics (from the
  index, else LRCLIB) with the scoring of section 6 plus two options:
  * `phonetic`: token similarity = max(ratio(a, b), 0.95·ratio(phon(a), phon(b))), with a
    language-agnostic phonetic key (b/v, s/z/c, ll/y, silent h, qu/k, ph/f, ck/k, doubled letters…);
  * `chunked`: lines (or windows of 4–12 tokens) are also aligned on their own, and coverage =
    max(whole-query coverage, 0.9 × weighted mean of chunk coverages). Chunks with little informative
    weight are not scored on their own, so text made of common words cannot add up coverage.
* **Grouping**: records are merged only with evidence that they are one song: 70 % of the distinct lines
  of the shorter lyrics are found in the other (fuzz ratio ≥ 85; any titles, compared when the scores are
  within 0.15); or the same normalised title and at least 30 % and 3 shared lines (two LRCLIB
  transcriptions of one song often share only 30–65 %); or the same title, one of them version-marked
  (karaoke, live, cover…, or a title that differs only by a decoration such as "(Acousique)" or
  "(Spanish Mix)") and the same lines matched; or the same title and both hold the sung lines almost
  verbatim (alignment ≥ 0.95, matched lines ≥ 90 % similar: a partly translated cover); or a medley /
  mashup ("A / B / C", "Mashup", "Megamix", "Mix 4", "Halftime") that matched the same lines (≥ 0.85),
  which never stands for the song itself. A group that shares its lyrics is shown with its
  best alignment by the most popular member that aligns within 0.15 of it or is more than twice as popular
  as the best-aligned member. In a group of version-marked records the more popular one wins unless it
  aligns more than 0.03 worse, and keeps its own alignment. An unpopular best match (rank < 300,000) also
  gets the most popular Deezer recording with its title by another artist verified, which joins only if
  it aligns about as well (0.05) or shares the lyrics.
* **Ranking**: alignment score + `0.03 · min(1, log10(1 + deezer_rank) / 6)` (popularity only breaks
  near-ties). A best match that is partial (< 0.9), rests on at most 2 matched informative words and
  leads the next song by less than 0.2 is first capped at 0.55 or, if higher, just enough to stay first
  after the popularity tie-break (a lead of 0.001): short generic lines align partly with many songs.
* **Decision**: section 7's rule with the `world` thresholds (`configs/calibration.json`: accept 0.62,
  margin 0.07, uncertain 0.45, the zero-false-match fit on 100 real songs; `DEFAULT_THRESHOLDS["world"]`
  is the fallback), on at least 5 candidates so that the margin has a runner-up.
* **Resolution against the local engines**: a local `match` backed by lyrics or a fingerprint wins.
  Otherwise a world `match` wins, then a local melody-only `match`. Otherwise the world list wins if its
  decision is at least as good and its top score at least as high. The world list is always returned as
  `engines.world`.
* **Learning**: verified songs scoring ≥ 0.5 are added to the index (`source = "lookup"`), only while the
  recognition history is kept.
