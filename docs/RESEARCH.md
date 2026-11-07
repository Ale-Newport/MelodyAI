# MelodyAI: Research Notes

This document reviews the research behind MelodyAI, an open-source system that
identifies songs from humming, whistling, singing or remembered lyrics. It explains
the technology choices and lists where MelodyAI departs from the original research.

Conventions used in this document:

- Facts about external work were checked against the source linked next to them
  in November 2026. A claim marked **(not verified)** could not be confirmed from a
  primary or reliable secondary source. A claim marked **(secondary source)** was
  confirmed only through a paper or page that cites the original.
- Library behaviour (defaults, signatures) was checked against the installed
  package versions in the project virtual environment: librosa 0.11.0,
  faster-whisper 1.2.1, ctranslate2 4.8.5, faiss-cpu 1.15.1, basic-pitch 0.4.0,
  torchcrepe 0.0.24, rapidfuzz 3.14.6, numba 0.68.1.
- Dataset details are in [DATASETS.md](DATASETS.md).

---

## 1. Scientific problem

**Query by humming (QBH)**, also called query by singing/humming (QBSH), retrieves a
piece of music from a short vocal imitation of its melody. The query is usually
monophonic (one pitch at a time), short (about 5 to 30 seconds), and imperfect. The
reference collection may be symbolic (MIDI or note lists), monophonic audio, or full
polyphonic studio recordings.

The task is hard because a hummed query differs from its reference in nearly every
dimension except the melodic shape:

| Source of mismatch | Example | Consequence for matching |
|---|---|---|
| Key / absolute pitch | User hums a fifth lower than the original | Absolute pitch cannot be compared directly; key invariance needed |
| Tempo and rhythm | User sings 30% slower, holds some notes longer | Global and local time warping needed |
| Intonation errors | Intervals off by a semitone, drifting tuning | Matching must tolerate small pitch errors |
| Wrong, missing or extra notes | Skipped passing notes, ornaments | Insertions and deletions must be allowed |
| Octave errors | Pitch tracker reports f0 one octave off | Spurious jumps of about 12 semitones |
| Partial query | Only the chorus, starting mid-song | Subsequence matching with free start and end |
| Timbre and channel | Voice vs instruments, laptop mic, room noise | Features must ignore timbre |
| Polyphony | Reference is a full mix with accompaniment | Melody must be extracted or learned from the mix |
| Out-of-catalogue queries | Song is not in the database | System must be able to abstain |

**Query by lyrics** is a related but separate problem. A user remembers (or sings) a few
words, possibly misheard, misspelled, in another language or with wrong word order. If the
query is sung, automatic speech recognition (ASR) must first turn it into text. ASR models
can also invent words when given humming with no lyrics, so the system has to detect when
there is no speech.

MelodyAI handles melody and lyrics as two independent retrieval channels and fuses their
rankings. It aims to rank the correct song highly when it is in the catalogue and to
abstain when no candidate is convincing.

---

## 2. State of the art

### 2.1 Industrial systems

- **Google "Hum to Search" (2020).** The best documented large-scale QBH system. A neural
  network maps a spectrogram of the hummed query and spectrograms of original studio
  recordings into a shared **melody embedding space**. Retrieval is nearest-neighbour search
  over the embeddings. The model is trained on positive pairs (same melody, for example a
  sung clip and the studio recording) and negative pairs (different melodies). There is no
  intermediate symbolic step such as MIDI or a hand-built melody extractor. Because the
  references are the original recordings, the index can be refreshed with new releases. The post
  reports a database of "over half a million songs".
  ([Google Research blog, 12 Nov 2020](https://research.google/blog/the-machine-learning-behind-hum-to-search/))
  Points from the post that matter for MelodyAI:
  - Training data was augmented by randomly varying pitch and tempo of sung input.
  - The first training set was mostly singing, so the model failed on humming and whistling.
    To fix this they generated **synthetic hums**. The first version used
    [SPICE](https://research.google/blog/spice-self-supervised-pitch-estimation/) to extract
    pitch from sung audio and played it back as simple tones. A later version replaced the
    tone generator with a **neural synthesizer** that sounds like real humming or whistling.
  - Extra positive pairs came from aligning clips of two different singers singing the same
    melody ("mix and match"). Superpositions of training data were also used as augmentation.
  - The model was first trained with **triplet loss**. A second loss was then added, which the
    post says improved precision and recall.
  - The post contrasts the hummed tune with "original (polyphonic) recordings". It does not
    use the word "monophonic". The query is one melodic line, but the reference is a full mix.
  - The post gives **no accuracy figures**.
- Other commercial QBH services exist (for example SoundHound/Midomi), but their methods are
  not publicly documented **(not verified)**. They are not described further here.

### 2.2 Academic state of the art

- **MIREX QBSH** (Music Information Retrieval Evaluation eXchange) ran a Query by
  Singing/Humming task every year from 2006. Queries are matched against monophonic
  ground-truth MIDI files hidden among thousands of "noise" MIDI files from the Essen
  collection. Systems are scored by **top-10 hit rate** and **mean reciprocal rank (MRR)**
  (secondary source: MIREX wiki pages found by search; the wiki's TLS certificate had expired
  when checked, so the pages could not be fetched directly). Top systems were mostly
  pitch-tracking plus DTW / note alignment pipelines.
- **Salamon, Serrà and Gómez (2013)** showed that melody representations extracted
  from a cappella queries can retrieve the original song from a **polyphonic** collection, by
  running the Melodia melody extractor on the references. They released the MTG-QBH query set
  ([IJMIR 2(1):45–58, DOI 10.1007/s13735-012-0026-0](https://doi.org/10.1007/s13735-012-0026-0)).
- **Learned embeddings for QBH (2023).** Amatov et al. (ISMIR 2023,
  [arXiv:2312.01092](https://arxiv.org/abs/2312.01092), code
  [amanteur/CHAD](https://github.com/amanteur/CHAD), MIT) trained a ResNet-18 encoder with
  NT-Xent (contrastive) loss on CREPE pitch or CQT features after vocal separation. They grew
  training data semi-supervised by mining aligned fragments from cover songs, and report
  top-10 hit rates of roughly 0.87–0.97 on MIREX-style benchmarks with CREPE features.
- **Classifier-style QBH on closed sets.** Ranjan and Srivastava (2023,
  [arXiv:2302.04577](https://arxiv.org/abs/2302.04577)) denoise pitch vectors with Total
  Variation regularization and classify queries into 48 MIR-QBSH songs with a fully
  convolutional network, reporting 93% accuracy. This does not scale to an open, growing
  catalogue because each new song needs a new output class.
- **Humming transcription** has become a separate benchmark with
  [HumTrans (2023)](https://arxiv.org/abs/2309.09623), a 56.22-hour humming dataset with
  MIDI labels. Its authors found that existing vocal transcription models are "far from
  satisfactory" on humming.

**Summary.** At web scale, learned embeddings trained on large amounts of paired data
work best (Google). With little data, pitch extraction plus alignment (DTW / note
alignment) is still the reliable, explainable baseline. It is also the only approach that
works with zero training data. MelodyAI implements both and compares them with experiments.

---

## 3. Classical algorithms

### 3.1 Pitch (f0) estimation

| Algorithm | Idea | Notes |
|---|---|---|
| **YIN** (de Cheveigné & Kawahara, 2002) | Cumulative mean normalised difference function; first dip below a threshold gives the period; parabolic interpolation | JASA 111(4):1917–1930, [DOI 10.1121/1.1458024](https://doi.org/10.1121/1.1458024). Simple and fast, but a fixed threshold causes octave errors and voicing mistakes |
| **pYIN** (Mauch & Dixon, 2014) | Replace YIN's single threshold with a Beta-distributed prior over thresholds, giving several f0 candidates with probabilities; HMM + Viterbi decoding picks a smooth pitch track and voicing | ICASSP 2014, pp. 659–663, [DOI 10.1109/ICASSP.2014.6853678](https://doi.org/10.1109/ICASSP.2014.6853678). Gives per-frame **voicing probability**, which QBH needs to segment notes and rests |
| **Melodia** (Salamon & Gómez, 2012) | Salience function, then pitch contours tracked and grouped with auditory streaming cues; contour statistics separate melody from accompaniment | IEEE TASLP 20(6):1759–1770, [DOI 10.1109/TASL.2012.2188515](https://doi.org/10.1109/TASL.2012.2188515). Designed for **polyphonic** audio (predominant melody) |
| **CREPE** (Kim et al., 2018) | Deep CNN on the raw waveform, 360-bin pitch classifier | ICASSP 2018, [arXiv:1802.06182](https://arxiv.org/abs/1802.06182); reports matching or beating pYIN. Code [marl/crepe](https://github.com/marl/crepe) (MIT); PyTorch port [torchcrepe](https://github.com/maxrmorrison/torchcrepe) (MIT) |
| **SPICE** (Gfeller et al., 2020) | Self-supervised: two pitch-shifted CQT slices go through the same encoder, and the output difference must be proportional to the pitch shift; separate confidence head | IEEE/ACM TASLP 28:1118–1128, [arXiv:1910.11664](https://arxiv.org/abs/1910.11664). Used by Google to build synthetic hums |

librosa implements YIN and pYIN. Verified against the installed librosa 0.11.0
([`librosa.pyin`](https://librosa.org/doc/0.11.0/generated/librosa.pyin.html)):
`pyin(y, *, fmin, fmax, sr=22050, frame_length=2048, hop_length=None, ..., resolution=0.1,
max_transition_rate=35.92, switch_prob=0.01, ...)` returns `(f0, voiced_flag, voiced_probs)`.
`hop_length` defaults to `frame_length // 4`. The docstring recommends `fmin ≈ C2 (~65 Hz)`
and `fmax ≈ C7 (~2093 Hz)`. `resolution` is in semitones (0.01 = cents), and
`max_transition_rate` is in octaves per second.

### 3.2 Melody representations

- **Pitch contour / Parsons code.** Ghias et al. (1995) turned the pitch track into a
  string over the alphabet **U / D / S** (up, down, same as the previous note) and matched
  strings with approximate string matching. This is fully key- and tempo-invariant but throws
  away interval size and rhythm, so it needs long queries to be discriminative.
  ([ACM Multimedia 1995, pp. 231–236](https://www.cs.cornell.edu/zeno/papers/humming/humming.html);
  U/D/S detail confirmed via secondary source arXiv:2302.04577.)
- **Note sequences with intervals and durations.** McNab et al. (MELDEX, 1996) transcribed
  sung input into notes and studied string matching requirements for ranked retrieval,
  including an experiment on how accurately people sing well-known tunes
  ([ACM DL 1996, DOI 10.1145/226931.226934](https://doi.org/10.1145/226931.226934);
  [D-Lib 1997 article on MELDEX](https://dlib.org/dlib/may97/meldex/05witten.html)).
  Interval (semitone difference) sequences are key-invariant. Inter-onset-interval *ratios*
  are tempo-invariant.
- **Frame-level pitch vectors.** Jang and colleagues matched frame-level pitch (no note
  segmentation) with dynamic programming. Their hierarchical filtering method first
  eliminates about 80% of unlikely candidates cheaply, then compares the rest in detail
  ([Jang & Lee, ACM MM 2001, pp. 401–410](https://dblp.org/rec/conf/mm/JangL01)). Frame-based
  matching avoids note segmentation errors but costs more.
- **Chroma.** 12-bin pitch-class energy per frame. It ignores octave and timbre, so a hum and
  a full mix of the same melody can look similar. Accompaniment harmony leaks into the
  chroma, though. librosa: [`chroma_cqt`](https://librosa.org/doc/0.11.0/generated/librosa.feature.chroma_cqt.html)
  (defaults `hop_length=512`, `n_chroma=12`, `n_octaves=7`, `bins_per_octave=36`), built on the
  [constant-Q transform `cqt`](https://librosa.org/doc/0.11.0/generated/librosa.cqt.html).
- **MFCC.** Describes timbre, not melody. It is not useful for QBH matching, but can help
  detect voice vs. silence ([`librosa.feature.mfcc`](https://librosa.org/doc/0.11.0/generated/librosa.feature.mfcc.html)).
- **Onsets and tempo.** [`librosa.onset.onset_detect`](https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_detect.html)
  and [`librosa.feature.tempo`](https://librosa.org/doc/0.11.0/generated/librosa.feature.tempo.html)
  (default `start_bpm=120`). Tempo estimation is unreliable on short hummed queries with soft
  onsets. MelodyAI therefore does not depend on it. It handles tempo by alignment instead.

### 3.3 Dynamic time warping (DTW)

DTW finds the minimum-cost monotonic alignment between two sequences. It absorbs global and
local tempo differences.

- **Sakoe & Chiba (1978)** introduced the symmetric DP formulation, **slope constraints** on
  the warping path, and the band-shaped adjustment window now called the **Sakoe-Chiba band**
  ([IEEE TASSP 26(1):43–49, DOI 10.1109/TASSP.1978.1163055](https://doi.org/10.1109/TASSP.1978.1163055)).
- **Müller, *Fundamentals of Music Processing*** (Springer 2015, ISBN 978-3-319-21944-8;
  2nd ed. 2021 with Python/Jupyter notebooks, ISBN 978-3-030-69807-2) is the standard music
  reference. It covers DTW, step size conditions, weights, and **subsequence DTW**. In
  subsequence DTW, the first row of the accumulated cost matrix is initialised with the local
  cost only (free start), and the end position is the argmin of the last row (free end).
  Backtracking from that end gives the start. The normalised last row works as a matching
  function over positions in the long sequence
  ([FMP notebook C7S2 Subsequence DTW](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html),
  [FMP notebook C3S2 DTW basics](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html)).
- **librosa** ([`librosa.sequence.dtw`](https://librosa.org/doc/0.11.0/generated/librosa.sequence.dtw.html),
  verified in 0.11.0) follows Müller's nomenclature. Signature:
  `dtw(X=None, Y=None, *, C=None, metric='euclidean', step_sizes_sigma=None, weights_add=None,
  weights_mul=None, subseq=False, backtrack=True, global_constraints=False, band_rad=0.25,
  return_steps=False)`. `subseq=True` enables subsequence DTW. `global_constraints=True`
  applies a Sakoe-Chiba band of radius `int(band_rad * min(C.shape))`. `step_sizes_sigma`
  and `weights_add` / `weights_mul` set allowed steps and step penalties.
  It returns the accumulated cost matrix `D` and the warping path `wp`.

### 3.4 String and sequence alignment

- **Edit distance / approximate string matching**: used on contour strings and note sequences
  (Ghias; McNab).
- **Smith-Waterman local alignment** (Smith & Waterman 1981,
  [J. Mol. Biol. 147(1):195–197, DOI 10.1016/0022-2836(81)90087-5](https://doi.org/10.1016/0022-2836(81)90087-5))
  finds the best-matching pair of *segments* of two sequences. For lyrics this is the right
  model: the query is a fragment that may sit anywhere in the song text. Ranjan & Arora (2020)
  applied semi-global (bioinformatics-style) alignment to QBH as well (cited in
  arXiv:2302.04577; secondary source).

### 3.5 Text retrieval

- **BM25** (Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*,
  FnTIR 3(4):333–389, 2009, [DOI 10.1561/1500000019](https://doi.org/10.1561/1500000019)) is the
  standard lexical ranking function.
- **SQLite FTS5** ([docs](https://sqlite.org/fts5.html)) has a built-in `bm25()` auxiliary
  function. The returned value is multiplied by −1, so "the better the match, the numerically
  smaller the value", and `ORDER BY rank` uses it by default. The default `unicode61`
  tokenizer is case-insensitive under Unicode 6.1 rules. Its `remove_diacritics` option
  defaults to `1`, and `2` removes diacritics correctly from all Latin characters. A
  `trigram` tokenizer is also available for substring matching.

### 3.6 Rank fusion and calibration

- **CombSUM / CombMNZ** (Fox & Shaw, TREC-2 1993 / TREC-3 1994): sum normalised scores, and
  for CombMNZ multiply by the number of systems that retrieved the item.
- **Reciprocal Rank Fusion (RRF)** (Cormack, Clarke & Büttcher, SIGIR 2009, pp. 758–759,
  [DOI 10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114);
  [author PDF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)):
  `RRF(d) = Σ_r 1 / (k + r(d))` with **k = 60**, "fixed during a pilot investigation". The
  paper reports that the choice of k was "not critical" and that RRF beat Condorcet Fuse and
  CombMNZ on TREC runs. RRF needs no score normalisation and no training data.
- **Calibration.** Guo, Pleiss, Sun & Weinberger, "On Calibration of Modern Neural Networks"
  (ICML 2017, PMLR 70:1321–1330, [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)) showed
  that modern networks are poorly calibrated. They used **Expected Calibration Error (ECE)**
  and reliability diagrams to measure it, and proposed temperature scaling as a fix.
  Classical post-hoc calibrators are **Platt scaling** (sigmoid fit; Platt 1999) and
  **isotonic regression** (Zadrozny & Elkan 2001/2002).

---

## 4. Neural architectures

| Component | Architecture | Use in QBH |
|---|---|---|
| Pitch estimation | CNN on waveform (CREPE), self-supervised CNN on CQT (SPICE) | More robust f0 than pYIN in noise; GPU helps |
| Note transcription | **Basic Pitch**: lightweight CNN predicting onset, multipitch and note posteriorgrams (Bittner et al., ICASSP 2022, [arXiv:2203.09893](https://arxiv.org/abs/2203.09893)) | Direct note events and MIDI from audio |
| Melody embedding | Neural network on spectrograms trained with triplet loss plus an extra loss (Google Hum to Search; architecture not disclosed); ResNet-18 + NT-Xent on CREPE/CQT (CHAD) | End-to-end retrieval by nearest neighbour |
| Source separation | **Demucs / Hybrid Transformer Demucs** (Défossez 2021; Rouard, Massa & Défossez, ICASSP 2023) separates drums, bass, other, vocals | Isolate the vocal melody of polyphonic references before pitch tracking |
| ASR | **Whisper** Transformer ASR model trained on 680,000 h of weakly supervised audio (Radford et al., ICML 2023, [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)) | Turns sung lyrics into text |

### 4.1 Metric and contrastive learning losses

- **Triplet loss** (FaceNet; Schroff, Kalenichenko & Philbin, CVPR 2015,
  [arXiv:1503.03832](https://arxiv.org/abs/1503.03832)): pulls an anchor towards a positive
  and pushes it from a negative by a margin. The paper trains on triplets chosen with
  **online triplet mining**. Used by Google Hum to Search.
- **InfoNCE** (van den Oord, Li & Vinyals 2018, "Representation Learning with Contrastive
  Predictive Coding", [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)): a softmax
  cross-entropy that picks the positive among N−1 negatives. The paper names it "InfoNCE" and
  shows that minimising it maximises a lower bound on mutual information. The bound gets
  tighter as N grows.
- **NT-Xent** (SimCLR; Chen, Kornblith, Norouzi & Hinton 2020,
  [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)): "normalized temperature-scaled cross
  entropy". This is InfoNCE with cosine similarity and a temperature, using the other
  in-batch examples as negatives.

### 4.2 Synthetic training data

With no large set of real hums, the practical option is to synthesise hums from known
melodies. This is what Google did with SPICE plus a neural synthesizer, starting from sung
audio. Augmentations named in the literature: pitch shift, tempo change (Google), plus the
error types listed in Section 1. Synthetic data cannot fully reproduce real human errors.
Google found that a model trained only on singing did not work on humming, so there is a
**domain gap** to watch for.

---

## 5. Main research works (annotated)

| Year | Work | Contribution |
|---|---|---|
| 1978 | Sakoe & Chiba | DTW with slope constraints and adjustment window |
| 1981 | Smith & Waterman | Local sequence alignment |
| 1995 | Ghias, Logan, Chamberlin & Smith, *Query by Humming* | First QBH system: autocorrelation pitch tracking, U/D/S contour, approximate string matching |
| 1996–97 | McNab, Smith, Witten, Henderson, Cunningham; MELDEX | Note transcription of sung queries, matching analysis, study of singing accuracy |
| 2001 | Jang & Lee, hierarchical filtering | Fast candidate filtering, then detailed DP match on frame pitch |
| 2002 | de Cheveigné & Kawahara, YIN | f0 estimator |
| 2006– | MIREX QBSH task | Shared evaluation, MIR-QBSH and IOACAS corpora, top-10 hit rate / MRR |
| 2009 | Cormack et al., RRF | Simple, robust rank fusion |
| 2009 | Robertson & Zaragoza | BM25 reference monograph |
| 2012 | Salamon & Gómez, Melodia | Predominant melody extraction from polyphonic audio |
| 2013 | Salamon, Serrà & Gómez | Melody-based QBH against polyphonic collections; MTG-QBH dataset |
| 2014 | Mauch & Dixon, pYIN | Probabilistic YIN with HMM voicing |
| 2015 | Müller, FMP | DTW, subsequence DTW, audio matching (textbook) |
| 2015 | Schroff et al., FaceNet | Triplet loss for embeddings |
| 2017 | Guo et al. | Calibration of neural networks, ECE |
| 2018 | Kim et al., CREPE | CNN pitch tracker |
| 2018 | van den Oord et al., CPC | InfoNCE |
| 2020 | Gfeller et al., SPICE | Self-supervised pitch |
| 2020 | Google, Hum to Search | Production QBH with melody embeddings and synthetic hums |
| 2020 | Chen et al., SimCLR | NT-Xent contrastive learning |
| 2022 | Bittner et al., Basic Pitch | Lightweight polyphonic note transcription |
| 2022–23 | Radford et al., Whisper | Robust multilingual ASR |
| 2023 | Liu et al., HumTrans | Large humming transcription dataset |
| 2023 | Amatov et al., CHAD | Semi-supervised data collection for neural QBH |

---

## 6. Related repositories

| Repository | What it does | License (verified) | Relevance to MelodyAI |
|---|---|---|---|
| [spotify/basic-pitch](https://github.com/spotify/basic-pitch) | Audio-to-MIDI note transcription with pitch bends; CLI and `predict()` API returning `model_output, midi_data, note_events` | Apache-2.0 | Optional note extractor |
| [ShreyanshGoyal/query_by_humming](https://github.com/ShreyanshGoyal/query_by_humming) | Chroma + DTW QBH (no training) | Apache-2.0 | Design reference for the chroma baseline |
| [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Whisper inference on CTranslate2 | MIT | Optional ASR for sung lyrics |
| [facebookresearch/faiss](https://github.com/facebookresearch/faiss) | Dense vector similarity search | MIT | Neural embedding index (non-macOS; NumPy fallback) |
| [librosa/librosa](https://github.com/librosa/librosa) | Audio analysis (pYIN, CQT, chroma, DTW, onsets, tempo) | ISC (package metadata) | Core signal processing |
| [maxrmorrison/torchcrepe](https://github.com/maxrmorrison/torchcrepe) | PyTorch CREPE (`full` / `tiny`), Viterbi decoding, periodicity output | MIT | Optional third f0 extractor |
| [marl/crepe](https://github.com/marl/crepe) | Original CREPE (TensorFlow) | MIT | Reference |
| [facebookresearch/demucs](https://github.com/facebookresearch/demucs) | Music source separation (htdemucs) | MIT; **archived 1 Jan 2025**, fixes continue in [adefossez/demucs](https://github.com/adefossez/demucs) | Possible future: vocal isolation for polyphonic references |
| [mir-dataset-loaders/mirdata](https://github.com/mir-dataset-loaders/mirdata) | Common loaders for MIR datasets | BSD-3-Clause | Optional dataset access (see DATASETS.md) |
| [amanteur/CHAD](https://github.com/amanteur/CHAD) | Code and metadata for neural QBH data collection | MIT | Neural QBH reference |
| [rapidfuzz/RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) | Fast fuzzy string similarity (`ratio`, `token_set_ratio`, `WRatio`, `cdist`, ...) | MIT | Token similarity in lyrics alignment |

### 6.1 Notes on specific repositories

**Spotify Basic Pitch** ([README](https://github.com/spotify/basic-pitch)):
- Polyphonic and instrument-agnostic, but the README says it "works best on one instrument at
  a time". Stereo is down-mixed to mono, and audio is resampled to 22,050 Hz.
- Backends: TensorFlow, CoreML, TensorFlow Lite and ONNX. Load order is TensorFlow, then
  CoreML, TFLite, ONNX. Default runtime per the README: **CoreML on macOS, TFLite on Linux,
  ONNX on Windows**. `pip install basic-pitch[tf]` adds TensorFlow.
- The README lists Python 3.7–3.11 and notes Apple M1 support only on Python 3.10. The
  project environment runs basic-pitch 0.4.0 on Python 3.11, so this README statement may be
  out of date **(not verified)**.
- Paper: Bittner et al., ICASSP 2022. The README asks users to cite the code version, because
  the library has improved since the paper.

**ShreyanshGoyal/query_by_humming** ([repo](https://github.com/ShreyanshGoyal/query_by_humming)):
- 12-bin chroma per frame, L2-normalised, cosine distance. DTW is vectorised over
  anti-diagonals, and the path cost is normalised by combined sequence length. References
  are ranked by distance.
- Only precomputed chroma is shipped. Raw audio is excluded because the references are
  commercial recordings.
- Evaluation: 4 hums from one person against 12 references (4 songs), 48 alignments in total.
  For the 3 in-catalogue queries, top-1/3/5 accuracy was 3/3. The out-of-catalogue query
  still returned a nearest neighbour at distance 0.0786. That falls inside the 0.0476–0.0792
  range of genuine matches, so **there is no reliable rejection threshold**. The author calls
  it "a proof of concept, not a benchmark".
- Lesson for MelodyAI: chroma + DTW works as a zero-training baseline, but abstention must be
  designed explicitly and tuned on out-of-catalogue queries.

**faster-whisper** ([README](https://github.com/SYSTRAN/faster-whisper);
[`transcribe.py`](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)):
- Reimplementation of Whisper on [CTranslate2](https://opennmt.net/CTranslate2/), claimed
  "up to 4 times faster than openai/whisper for the same accuracy", with 8-bit quantisation on
  CPU and GPU. Requires Python ≥ 3.9. Audio is decoded with PyAV, so FFmpeg is not needed.
- Model names accepted (installed 1.2.1, `faster_whisper.utils.available_models()`):
  `tiny(.en)`, `base(.en)`, `small(.en)`, `medium(.en)`, `large-v1/v2/v3`, `large`,
  `distil-*`, `large-v3-turbo`, `turbo`.
- `WhisperModel(device="auto" | "cpu" | "cuda", compute_type="default")`. On this Mac,
  `ctranslate2.get_supported_compute_types("cpu")` returned `float32`, `int8`, `int8_float32`.
- **No Apple MPS / Metal support:** the CTranslate2
  [hardware support page](https://opennmt.net/CTranslate2/hardware_support.html) lists x86-64
  and ARM64 CPUs, NVIDIA CUDA and AMD ROCm, with no Metal. Verified locally: with ctranslate2
  4.8.5, `get_supported_compute_types("mps")` raises `ValueError: unsupported device mps`.
  On Apple Silicon, faster-whisper therefore runs on the CPU.
- `transcribe()` defaults (1.2.1): `no_speech_threshold=0.6`, `log_prob_threshold=-1.0`,
  `vad_filter=False`, `language_detection_threshold=0.5`, `language_detection_segments=1`,
  `hallucination_silence_threshold=None`. Language is detected automatically when
  `language` is not set, and `info.language` / `info.language_probability` are returned. With
  `vad_filter=True` the Silero VAD removes silence (by default only stretches longer than 2 s;
  tunable through `vad_parameters`).

**FAISS** ([repo](https://github.com/facebookresearch/faiss),
[index wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)):
- `IndexFlatL2` / `IndexFlatIP`: **exact** brute-force search. For cosine similarity,
  L2-normalise the vectors and use inner product.
- `IndexIVFFlat` (`nlist`, `nprobe`) and `IndexIVFPQ`: **approximate**. They need a training
  step (coarse centroids, PQ codebooks) and search only `nprobe` lists.
- `IndexHNSWFlat` (`M`, `efSearch`, `efConstruction`): **approximate** graph search
  (Malkov & Yashunin, [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)), with no training
  step.
- Optional GPU (CUDA, ROCm, cuVS). Citations: Johnson, Douze & Jégou 2019 (IEEE TBD 7(3)),
  Douze et al. 2024 ([arXiv:2401.08281](https://arxiv.org/abs/2401.08281)). The README
  points to conda packages (`faiss-cpu`, `faiss-gpu`). The project uses the `faiss-cpu`
  wheel (1.15.1) from PyPI where possible. On macOS, that wheel and the PyTorch wheel each
  bundle `libomp`, and loading both in one process aborts (or segfaults with
  `KMP_DUPLICATE_LIB_OK`); MelodyAI therefore falls back to NumPy exact search and a NumPy IVF
  index on macOS (see LIMITATIONS.md).

---

## 7. Datasets (summary)

The full survey is in **[DATASETS.md](DATASETS.md)**. Briefly:

- **QBH benchmarks:** MTG-QBH (118 sung queries; reference audio not included), MIR-QBSH
  (4,431 queries / 48 MIDI, secondary source), IOACAS (759 queries / 298 MIDI, secondary
  source), and the MIREX QBSH task protocol.
- **Humming / whistling corpora:** HumTrans (about 56 h, CC BY-NC 4.0) and MLEnd Hums and
  Whistles (6,611 recordings of 8 songs; dataset license not stated).
- **Singing / melody annotation:** vocadito (CC BY 4.0), MedleyDB melody (CC BY-NC-SA 4.0,
  access on request), MDB-stem-synth (CC BY-NC 4.0), MIR-1K (restricted), Orchset
  (CC BY-NC-SA 4.0).
- **Symbolic melody sources:** Lakh MIDI (CC-BY 4.0 for the collection, but files are
  transcriptions of mostly copyrighted songs), Nottingham (no explicit license on the
  original; cleaned version GPL-3.0), Essen (the CCARH Humdrum edition ships a
  **restrictive academic license**).
- **MelodyAI does not redistribute any third-party dataset.** The built-in demo catalogue is
  entirely synthetic and original (Section 9.8).

---

## 8. Licenses and restrictions

### 8.1 Software dependencies

Licenses below are taken from the installed package metadata or the project's repository
page.

| Component | License | Notes |
|---|---|---|
| librosa | ISC | |
| numba | BSD | |
| NumPy / SciPy | BSD-3-Clause **(not verified in this pass)** | |
| faiss-cpu | MIT | |
| basic-pitch | Apache-2.0 | Ships model files inside the package |
| torchcrepe | MIT | Weights converted from original CREPE (MIT) |
| faster-whisper | MIT | Downloads converted Whisper weights from the Hugging Face Hub at runtime. The weight license was **not verified** in this pass |
| ctranslate2 | MIT | |
| rapidfuzz | MIT | |
| pretty_midi, mido | MIT | |
| PyTorch | Package metadata lists a composite expression (Apache-2.0, BSD-2/3-Clause, MIT, ISC, Zlib, BSL-1.0, with third-party components) | |
| FastAPI, SQLAlchemy | MIT | |
| mirdata | BSD-3-Clause | Loader code only, not the data |
| Demucs | MIT | Archived upstream; not a default dependency |
| React, TypeScript, Vite, Tailwind CSS | **not verified in this pass** | Check each package's `LICENSE` before release |

All the core dependencies above are permissive. Optional extras keep the default install small.
Basic Pitch, torchcrepe/PyTorch and faster-whisper are not needed for the DTW baseline.

### 8.2 Content restrictions

- **Melodies and lyrics of commercial songs are copyrighted works.** That includes MIDI
  transcriptions and lyric text. MelodyAI ships **no** commercial audio, MIDI or lyrics.
  Users who index their own collections are responsible for having the rights to do so.
- Most research datasets relevant to QBH are **non-commercial** (CC BY-NC or BY-NC-SA) or
  have custom or academic terms. Models trained on them may inherit those restrictions.
  Treat any such model as research-only unless the terms are reviewed.
- Google Hum to Search is a proprietary service. MelodyAI uses only the published blog
  description, not code, data or models.
- Users' recorded queries are personal data (voice). MelodyAI processes them locally by
  default.

---

## 9. Technologies chosen and justification

### 9.1 Pitch extraction

| Extractor | Status | Why |
|---|---|---|
| **librosa `pyin`** | Default baseline | No extra dependencies; per-frame voicing probability; HMM smoothing reduces octave jumps; well documented and reproducible |
| **Spotify Basic Pitch** | Optional extra (may be missing) | Gives note events directly; runs on CoreML/TFLite/ONNX with no GPU; Apache-2.0 |
| **torchcrepe** | Optional third extractor | Better accuracy in noise according to the CREPE paper; periodicity output acts as confidence; benefits from a GPU (MPS behaviour not verified). Note: CREPE can give high confidence on silence, and torchcrepe provides a silence threshold for this |

Processing chain: frame f0 (Hz) → MIDI pitch as a float (`69 + 12·log2(f/440)`) plus voicing
probability → **note segmentation** (a note starts when voicing is stable and the pitch
stays within a tolerance for a minimum duration; a sustained pitch jump, a voicing drop or a
re-articulation dip in the energy envelope ends it) →
note list (onset, duration, median pitch) → derived **intervals**, **inter-onset ratios** and
**rests**.

### 9.2 Normalisation: key, octave and tempo invariance

- **Key:** subtract a robust centre (the median pitch of voiced frames or notes). Then search
  a small range of residual transpositions (fractions of a semitone to a few semitones)
  during matching. This absorbs centring errors when the query covers only part of the
  melody's range. The **interval representation** is used alongside, because it is
  key-invariant by construction.
- **Octave errors:** excursions of about ±12 (or ±24) semitones from the local median that
  last at most ~80 ms are folded back. This is a known pYIN/YIN failure mode. Longer excursions
  are kept, because genuine octave leaps last whole notes (an earlier version folded all
  ±12 jumps and destroyed real leaps; the benchmark exposed it).
- **Tempo:** notes are resampled to a fixed time grid, so a note sequence becomes a
  piecewise-constant pitch curve. DTW with slope constraints absorbs local tempo changes. For
  global tempo, the query is stretched at several scales (for example 0.6×–1.6×) and the best
  scale is kept. Unconstrained DTW can align anything to anything (pathological warping), and
  slope constraints prevent that, while multi-scale stretching lets the band stay narrow.

### 9.3 Matching

- **Own DTW in numba.** librosa's DTW supports a Sakoe-Chiba band and step weights, but it
  has no early abandoning, it is not specialised for many short-against-long comparisons,
  and it does not combine subsequence mode with transposition search in one pass. MelodyAI
  implements:
  - Global DTW and subsequence DTW (open begin and end, following Müller FMP §7.2.3).
  - Sakoe-Chiba-like band and slope constraints.
  - Step weights that penalise extreme warping (long horizontal/vertical runs).
  - Path backtracking to report *where* in the reference the match lies (time offset).
- **Chroma baseline.** CQT chroma (`librosa.feature.chroma_cqt`) of query and reference
  audio, 12 circular key shifts, 3 tempo scales and MelodyAI's own subsequence DTW with a
  cosine cost. It needs no transcription, so it is a useful comparison. Its weaknesses:
  accompaniment harmony pollutes the chroma and octave/contour direction is lost.
- **Note alignment.** Smith-Waterman local alignment of note-interval sequences (plus
  tempo-invariant inter-onset-ratio features), after merging repeated notes on both sides.
  Tolerates missing/extra notes by construction; very cheap.
- **References as sliding windows.** Each reference (from MIDI, or monophonic audio through
  the pitch pipeline) is stored as a note sequence and indexed as overlapping windows. A
  query can then match any part of the song, such as a chorus in the middle.
- **Window DTW.** Every window is a fixed-length (96-point), key-normalised and
  time-normalised contour vector ("linear scaling"). The query vector is compared with every
  window whose duration is plausible (0.4–2.6× the query) using banded DTW (numba, parallel),
  with a residual key shift. For catalogs above 30,000 windows a cheap linear-scaling L1
  prefilter keeps the closest windows first (Jang-style filter-then-refine). Results are
  grouped per song (best window wins) and returned with the matched time range. Vector
  search is used for the *neural* embeddings: FAISS (`IndexFlatIP` exact, HNSW approximate)
  where it can share the process with PyTorch, otherwise NumPy exact search and a NumPy IVF
  index (exact and approximate are both benchmarked). The classical path does not need it at
  demo scale.
- **Ensemble.** The default recogniser averages the relative scores of window DTW, note
  alignment and subsequence DTW (and optionally neural + DTW re-ranking).

### 9.4 Neural encoder (experimental)

- A small PyTorch encoder (~0.8 M parameters) shared between query and reference: a 1D
  convolution stem with three residual dilated blocks, a 2-layer Transformer encoder,
  attention pooling and L2 normalisation, over the **relative pitch contour** (96 points,
  key-normalised). An optional query-side adapter turns it into a light asymmetric dual
  encoder.
- Trained with **InfoNCE** (in-batch negatives), optionally with **triplet loss**.
- Training pairs: synthetic hums generated from authorised or original melodies, with pitch
  shift, tempo change, vibrato, detuning, wrong or missing notes, duration jitter and noise.
  The splits are **by composition**, so no melody appears in both training and test, which
  would leak. Positive references are the *index-grid* window that best overlaps the sung
  excerpt (a first version paired each query with its exactly aligned window; validation
  top-1 stayed at ~0.33 because at inference excerpts never align with grid windows).
  A share of positives comes from audio rendered and analysed by the real pYIN pipeline.
- Retrieval: exact or approximate vector search over embeddings (FAISS or NumPy, see 9.3),
  with optional DTW re-ranking.
- Device auto-detection: CUDA → MPS → CPU.
- **No assumption that deep learning wins.** With tiny data and synthetic training, DTW
  baselines may well be better. The encoder is reported only next to DTW baselines on the
  same split and metrics (top-1/top-5/top-10 hit rate, MRR, abstention quality).

### 9.5 Lyrics

- **ASR (optional extra):** faster-whisper, `tiny`/`base`/`small` selectable, CPU `int8` by
  default (no MPS; see §6.1), automatic language detection. Segments with high no-speech
  probability or low average log-probability are dropped, using `no_speech_threshold` and
  `log_prob_threshold`. Optional VAD helps too. Together these stop Whisper from turning a
  hum into invented words.
- **Lyrics search works without ASR.** Users can also type the text.
  1. Normalisation: Unicode **NFKC**, case folding, punctuation stripping, accent folding.
  2. Candidate retrieval: **SQLite FTS5 BM25**.
  3. Re-ranking: **word-level local alignment** (Smith-Waterman over tokens, with RapidFuzz
     token similarity as the substitution score, so misspellings partly match) and
     **character n-gram similarity** (robust to word-boundary and spelling errors).
- **No semantic embeddings by default.** Sentence embeddings group songs by *topic*: a query
  about "rain" retrieves other songs about rain. For identifying a specific song, that gives
  confident false positives. Lexical and fuzzy matching is what identification needs.

### 9.6 Fusion and abstention

- Methods: weighted score fusion (after per-channel normalisation), **RRF** (k = 60 as in
  Cormack et al.), **CombMNZ**, and **logistic regression** learned fusion once enough
  labelled queries exist.
- **Abstention:** return "no confident match" when the top score or the margin between the
  first and second results falls below thresholds tuned on **out-of-catalogue** queries.
  The ShreyanshGoyal results show why: without a threshold, an unknown song still returns a
  plausible-looking nearest neighbour.
- **Score semantics:** scores are shown as *relative* scores, not probabilities, unless a
  calibrator (Platt scaling in the current implementation) has been fitted on held-out data. When one is fitted, ECE is
  reported, following Guo et al.

### 9.7 Storage and serving

- SQLite through SQLAlchemy by default (zero setup, and FTS5 is built in). PostgreSQL is
  optional for multi-user deployments.
- Features are stored separately as `.npz` with a **feature version**, so changing the
  extractor invalidates the cached features cleanly. Neural window embeddings are cached on
  disk per song and model; the vector index is rebuilt in memory from them when the catalog
  changes (a concatenation, milliseconds at demo scale).
- Backend: FastAPI. Frontend: React + TypeScript + Vite + Tailwind.

### 9.8 Demo catalogue

- About 48 **original, procedurally generated** melodies, seeded for reproducibility. Each
  has synthesised audio and MIDI, some have original lyrics, and each comes with transformed
  queries and ground truth. No commercial music or lyrics.
- Why: it is legally clean, fully reproducible, and the generator controls exactly which
  distortions each test query contains. The limitation is that synthetic queries are easier
  and more regular than real human hums. Any number measured on the demo set must not be
  presented as real-world accuracy.

---

## 10. Advantages and limitations of approaches

| Approach | Advantages | Limitations |
|---|---|---|
| Contour string (U/D/S) + edit distance | Trivial to compute; key- and tempo-invariant; tiny index | Low discriminative power; needs long queries; sensitive to note segmentation errors |
| Note sequence + DTW / alignment | Explainable; works with zero training data; uses interval sizes and durations; gives the alignment path (time offset) | Depends on note segmentation; costs O(N·M) per comparison, so it needs candidate filtering |
| Frame-level pitch + DTW | No segmentation errors; strong in MIREX QBSH | Slower; needs careful key normalisation and slope constraints |
| Chroma + subsequence DTW | Works on raw audio references, including polyphonic ones, without transcription | Harmony and accompaniment add noise; octave information is lost; weak abstention |
| Melody extraction (Melodia, Demucs + f0) on polyphonic references | Allows QBH against real recordings | Extraction errors on dense mixes; heavy preprocessing; separation models are large |
| Learned embedding (spectrogram) | Very fast retrieval at scale; learns invariances; references can be raw studio audio | Needs very large paired training data; opaque; domain gap between singing, humming and whistling |
| Learned embedding (pitch contour) | Smaller model; trainable on symbolic and synthetic data | Inherits f0 errors; synthetic-to-real gap |
| Lyrics: BM25 + fuzzy alignment | Precise for remembered phrases; explainable; cheap | Fails when words are misheard beyond fuzzy tolerance; needs lyrics, which are often copyrighted |
| Lyrics: ASR | Lets users sing lyrics | Can invent words for hums; language and accent errors; no MPS in faster-whisper |
| Semantic text embeddings | Robust to paraphrase | Retrieves songs on the same topic, not the same song, giving false positives for identification |

---

## 11. Comparison of solutions

| Approach | Input representation | Key / tempo invariance | Needs training data | Works vs polyphonic references | Cost |
|---|---|---|---|---|---|
| Ghias et al. 1995 | U/D/S contour string | Key: yes (by design). Tempo: yes (rhythm discarded) | No | No (MIDI references) | Very low |
| MELDEX (McNab et al.) | Transcribed notes (pitch + duration) | Key: via intervals. Tempo: partial | No | No (symbolic references) | Low |
| Jang-style frame DP (MIREX QBSH)\* | Frame pitch vector | Key: typically mean shift / transposition search. Tempo: DTW / linear scaling | No | No (MIDI references) | Medium–high |
| Salamon et al. 2013 | Melodia pitch contours | Key: by normalisation. Tempo: alignment | No | **Yes** (melody extracted from the mix) | High (extraction on every reference) |
| Chroma + DTW (ShreyanshGoyal; FMP audio matching) | 12-d chroma frames | Key: no (needs 12 circular shifts). Tempo: DTW | No | Partly (harmony pollutes) | Medium |
| Google Hum to Search | Spectrogram → embedding | Learned (pitch and tempo augmentation) | **Yes, very large** | **Yes** | High training cost, very low query cost |
| CHAD (Amatov et al. 2023) | CREPE pitch or CQT → embedding | Learned | Yes (about 300 h mined + crowdsourced) | Yes (after vocal separation) | High training, low query |
| Basic Pitch → notes → DTW | Note events from AMT | Same as note DTW | No (pretrained model) | Limited ("works best on one instrument at a time") | Medium |
| **MelodyAI DTW baseline** | pYIN f0 → voiced contour → normalised contour windows | Key: median + residual search. Tempo: linear scaling + banded DTW, multi-scale subsequence DTW | No | Experimental (Basic Pitch skyline references) | Low–medium (banded DTW over windows; LS prefilter for large catalogs) |
| **MelodyAI note alignment** | Note intervals + IOI ratios | Key and tempo: by construction | No | Same as above | Very low |
| **MelodyAI chroma baseline** | CQT chroma | Key: shift search. Tempo: subsequence DTW | No | Partly | Medium |
| **MelodyAI neural encoder** | Relative pitch contour → embedding | Learned plus normalised input | Yes (synthetic) | No by default | Low query; small training |
| **MelodyAI lyrics** | Normalised tokens | Not applicable | No (BM25 + fuzzy alignment); optional ASR model | Not applicable | Low |

\* The MIREX row describes the typical design of participating systems as reported in the
literature cited above, not one specific published configuration.

---

## 12. Differences between what MelodyAI implements and the original research

1. **Hum to Search (Google) vs MelodyAI neural encoder.**
   - Google embeds *spectrograms* of studio recordings. MelodyAI embeds *relative pitch
     contours* from MIDI or monophonic audio. MelodyAI therefore inherits f0 extraction errors
     and does **not** match against polyphonic recordings by default.
   - Google trained on large real sung/recorded pairs, plus synthetic hums made by running
     SPICE on *sung audio* and a neural synthesizer. MelodyAI's synthetic hums are rendered
     from *symbolic* melodies with rule-based perturbations. That is simpler and legally clean,
     but further from real human humming.
   - Google used triplet loss plus an additional loss. MelodyAI uses InfoNCE (optionally
     triplet). The model is orders of magnitude smaller, and the catalogue is about 48 songs
     instead of over 500,000.
   - Google reports no accuracy figures. MelodyAI does not claim parity with it.
2. **Ghias et al. / MELDEX.** MelodyAI keeps continuous (float) MIDI pitch and durations
   instead of a 3-symbol contour, and uses DTW with transposition search instead of plain
   edit distance. The contour idea survives only in the fixed-length candidate vectors.
3. **Jang's hierarchical filtering / MIREX systems.** The same linear-scaling + DTW idea; the
   filter (L1 on linearly scaled contours) only activates above 30,000 windows. MelodyAI has
   **not** been evaluated on MIR-QBSH or IOACAS. It has no MIREX-comparable numbers.
4. **Salamon et al. 2013 / Melodia.** MelodyAI does not run Melodia or source separation.
   Polyphonic references go through an *experimental* path: Basic Pitch transcription of the
   mix followed by a skyline rule (keep the highest note), which only works when the melody is
   the top voice. It is benchmarked separately on simple synthetic mixes (melody + pad + bass);
   it has not been measured on produced commercial recordings.
5. **Sakoe-Chiba / Müller DTW.** MelodyAI's DTW follows the textbook recursion and
   subsequence initialisation, but adds custom step weights, transposition search inside the
   match, and multi-scale query stretching. These are engineering choices, not reproductions
   of a specific published configuration. librosa's `dtw` is not used; all variants share
   one numba implementation tested against a reference DTW.
6. **pYIN.** Used through librosa as published. MelodyAI's note segmentation and octave
   folding on top are its own heuristics, not part of pYIN.
7. **Basic Pitch.** Used as published: as an optional query f0 extractor (skyline of its
   note events) and, experimentally, to transcribe polyphonic reference mixes.
8. **CREPE / torchcrepe.** Used through the PyTorch port (`tiny` model only), not the
   original TensorFlow release.
9. **ShreyanshGoyal/query_by_humming.** MelodyAI's chroma baseline follows the same idea
   (chroma + cosine + DTW) but uses subsequence DTW (free start and end) with 12 key shifts
   and 3 tempo scales instead of full-sequence DTW, so a query can match part of a song, and
   it adds an abstention threshold.
10. **RRF / CombMNZ.** Implemented as published (RRF k = 60 by default). Learned logistic
    fusion is used only when there is enough labelled data. With the demo set alone it
    risks overfitting.
11. **Calibration (Guo et al.).** Guo et al. calibrate softmax classifiers with temperature
    scaling. MelodyAI's retrieval scores are not softmax outputs, so it uses Platt scaling
    (logistic regression on top score and margin) fitted on the dev half of the synthetic
    benchmark and reports ECE/Brier on the test half. Scores stay *relative* in the UI unless
    that calibrator exists, and even then the estimate is labelled "calibrated on synthetic
    queries". Isotonic regression is not implemented.
12. **Whisper / faster-whisper.** Used as published. MelodyAI adds no fine-tuning. It only
    filters segments by no-speech and log-probability thresholds and keeps lyrics search
    independent of ASR.
13. **Evaluation data.** The research above uses real human queries (MTG-QBH, MIR-QBSH,
    IOACAS, HumTrans). MelodyAI's built-in evaluation uses **synthetic** queries from its own
    demo catalogue. Results on real-query datasets require users to obtain those datasets
    under their own terms (see [DATASETS.md](DATASETS.md)).

---

## 13. References

### QBH systems and evaluation
- Ghias, A., Logan, J., Chamberlin, D., Smith, B. C. (1995). *Query by Humming: Musical
  Information Retrieval in an Audio Database.* ACM Multimedia 1995, pp. 231–236.
  [Electronic proceedings](https://www.cs.cornell.edu/zeno/papers/humming/humming.html);
  DOI 10.1145/217279.215273 (DOI from a secondary bibliographic listing).
- McNab, R. J., Smith, L. A., Witten, I. H., Henderson, C. L., Cunningham, S. J. (1996).
  *Towards the Digital Music Library: Tune Retrieval from Acoustic Input.* ACM DL 1996,
  pp. 11–18. [DOI 10.1145/226931.226934](https://doi.org/10.1145/226931.226934).
- McNab, R. J., Smith, L. A., Bainbridge, D., Witten, I. H. (1997). *The New Zealand Digital
  Library MELody inDEX.* D-Lib Magazine.
  [Link](https://dlib.org/dlib/may97/meldex/05witten.html).
- Jang, J.-S. R., Lee, H.-R. (2001). *Hierarchical Filtering Method for Content-based Music
  Retrieval via Acoustic Input.* ACM Multimedia 2001, pp. 401–410.
  [dblp](https://dblp.org/rec/conf/mm/JangL01).
- MIREX Query by Singing/Humming task.
  [Wiki, 2020 edition](https://www.music-ir.org/mirex/wiki/2020:Query_by_Singing/Humming)
  (TLS certificate expired at time of checking; content confirmed through search snippets).
- Salamon, J., Serrà, J., Gómez, E. (2013). *Tonal Representations for Music Retrieval: From
  Version Identification to Query-by-Humming.* IJMIR 2(1):45–58.
  [DOI 10.1007/s13735-012-0026-0](https://doi.org/10.1007/s13735-012-0026-0).
- Google Research (2020). *The Machine Learning Behind Hum to Search.*
  [Blog](https://research.google/blog/the-machine-learning-behind-hum-to-search/).
- Amatov, A., et al. (2023). *A Semi-Supervised Deep Learning Approach to Dataset Collection
  for Query-by-Humming Task.* ISMIR 2023. [arXiv:2312.01092](https://arxiv.org/abs/2312.01092).
- Ranjan, S., Srivastava, V. (2023). *Incorporating Total Variation Regularization in the
  Design of an Intelligent Query by Humming System.*
  [arXiv:2302.04577](https://arxiv.org/abs/2302.04577).
- Liu, S., Li, X., Li, D., Shan, Y. (2023). *HumTrans: A Novel Open-Source Dataset for Humming
  Melody Transcription and Beyond.* [arXiv:2309.09623](https://arxiv.org/abs/2309.09623).

### Pitch, melody and transcription
- de Cheveigné, A., Kawahara, H. (2002). *YIN, a Fundamental Frequency Estimator for Speech and
  Music.* JASA 111(4):1917–1930. [DOI 10.1121/1.1458024](https://doi.org/10.1121/1.1458024).
- Mauch, M., Dixon, S. (2014). *pYIN: A Fundamental Frequency Estimator Using Probabilistic
  Threshold Distributions.* ICASSP 2014, pp. 659–663.
  [DOI 10.1109/ICASSP.2014.6853678](https://doi.org/10.1109/ICASSP.2014.6853678).
- Salamon, J., Gómez, E. (2012). *Melody Extraction from Polyphonic Music Signals Using Pitch
  Contour Characteristics.* IEEE TASLP 20(6):1759–1770.
  [DOI 10.1109/TASL.2012.2188515](https://doi.org/10.1109/TASL.2012.2188515).
- Kim, J. W., Salamon, J., Li, P., Bello, J. P. (2018). *CREPE: A Convolutional Representation
  for Pitch Estimation.* ICASSP 2018. [arXiv:1802.06182](https://arxiv.org/abs/1802.06182).
- Gfeller, B., Frank, C., Roblek, D., Sharifi, M., Tagliasacchi, M., Velimirović, M. (2020).
  *SPICE: Self-Supervised Pitch Estimation.* IEEE/ACM TASLP 28:1118–1128.
  [arXiv:1910.11664](https://arxiv.org/abs/1910.11664).
- Bittner, R. M., Bosch, J. J., Rubinstein, D., Meseguer-Brocal, G., Ewert, S. (2022). *A
  Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch
  Estimation.* ICASSP 2022. [arXiv:2203.09893](https://arxiv.org/abs/2203.09893).
- Défossez, A. (2021). *Hybrid Spectrogram and Waveform Source Separation.* ISMIR 2021 MDX
  Workshop. Rouard, S., Massa, F., Défossez, A. (2023). *Hybrid Transformers for Music Source
  Separation.* ICASSP 2023. (Both as cited in the
  [Demucs README](https://github.com/facebookresearch/demucs).)

### Alignment and music processing
- Sakoe, H., Chiba, S. (1978). *Dynamic Programming Algorithm Optimization for Spoken Word
  Recognition.* IEEE TASSP 26(1):43–49.
  [DOI 10.1109/TASSP.1978.1163055](https://doi.org/10.1109/TASSP.1978.1163055).
- Müller, M. (2015). *Fundamentals of Music Processing: Audio, Analysis, Algorithms,
  Applications.* Springer. ISBN 978-3-319-21944-8. 2nd ed. (2021), *Using Python and Jupyter
  Notebooks*, ISBN 978-3-030-69807-2. [FMP notebooks](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html).
- Smith, T. F., Waterman, M. S. (1981). *Identification of Common Molecular Subsequences.*
  J. Mol. Biol. 147(1):195–197.
  [DOI 10.1016/0022-2836(81)90087-5](https://doi.org/10.1016/0022-2836(81)90087-5).

### Retrieval, fusion and calibration
- Robertson, S., Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.*
  FnTIR 3(4):333–389. [DOI 10.1561/1500000019](https://doi.org/10.1561/1500000019).
- Fox, E. A., Shaw, J. A. (1994). *Combination of Multiple Searches.* TREC-2 (NIST SP 500-215).
  Shaw, J. A., Fox, E. A. (1995). TREC-3 (NIST SP 500-225), pp. 105–108.
  [TREC-2 text](https://trec.nist.gov/pubs/trec2/papers/txt/23.txt).
- Cormack, G. V., Clarke, C. L. A., Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms
  Condorcet and Individual Rank Learning Methods.* SIGIR 2009, pp. 758–759.
  [DOI 10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114).
- Malkov, Y. A., Yashunin, D. A. (2016). *Efficient and Robust Approximate Nearest Neighbor
  Search Using Hierarchical Navigable Small World Graphs.*
  [arXiv:1603.09320](https://arxiv.org/abs/1603.09320).
- Johnson, J., Douze, M., Jégou, H. (2019). *Billion-scale Similarity Search with GPUs.* IEEE
  Trans. Big Data 7(3):535–547. Douze, M., et al. (2024). *The Faiss Library.*
  [arXiv:2401.08281](https://arxiv.org/abs/2401.08281).
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). *On Calibration of Modern Neural
  Networks.* ICML 2017, PMLR 70:1321–1330. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to
  Regularized Likelihood Methods.* In Advances in Large Margin Classifiers, MIT Press.
  (Bibliographic details from secondary sources.)
- Zadrozny, B., Elkan, C. (2002). *Transforming Classifier Scores into Accurate Multiclass
  Probability Estimates.* KDD 2002.
  [PDF](https://www.cs.cornell.edu/courses/cs678/2007sp/ZadroznyElkan.pdf).

### Representation learning
- van den Oord, A., Li, Y., Vinyals, O. (2018). *Representation Learning with Contrastive
  Predictive Coding.* [arXiv:1807.03748](https://arxiv.org/abs/1807.03748).
- Chen, T., Kornblith, S., Norouzi, M., Hinton, G. (2020). *A Simple Framework for Contrastive
  Learning of Visual Representations.* [arXiv:2002.05709](https://arxiv.org/abs/2002.05709).
- Schroff, F., Kalenichenko, D., Philbin, J. (2015). *FaceNet: A Unified Embedding for Face
  Recognition and Clustering.* CVPR 2015. [arXiv:1503.03832](https://arxiv.org/abs/1503.03832).
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I. (2023). *Robust
  Speech Recognition via Large-Scale Weak Supervision.* ICML 2023.
  [arXiv:2212.04356](https://arxiv.org/abs/2212.04356).

### Software
Repository and documentation links for all software used are listed in Section 6
(Related repositories) and next to each claim above.
