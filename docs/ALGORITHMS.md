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
* **Scoring**: Smith-Waterman over tokens. A query/reference token pair scores +w if RapidFuzz
  ratio ≥ 84, +w/2 if ≥ 67, −w/2 otherwise, where w = IDF of the query token (× 0.3 for stop-words);
  gap penalty 0.4·mean(w). Coverage = aligned weight / total weight. Final score =
  0.8·coverage + 0.2·RapidFuzz partial-ratio of the query against the matched lines (±1 line).
  Words must appear *in order and close together*, which is what separates "the song" from "a song
  about the same topic". Semantic embeddings are intentionally not used.
* **ASR**: faster-whisper (`tiny`…`large-v3`, default `base`, CPU int8), beam 5, temperature 0,
  Silero VAD (min silence 400 ms). Segments are dropped if (no-speech > 0.6 and avg logprob < −0.5),
  avg logprob < −1.2, compression ratio > 2.6, or the text is a known Whisper hallucination
  ("thank you", "gracias", "subtítulos…"). At least two alphabetic words are required; otherwise
  `speech_detected = false` and the lyrics engine abstains instead of inventing text.

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
