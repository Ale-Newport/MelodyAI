# MelodyAI: Datasets

This document surveys datasets relevant to query by humming / whistling / singing and to
melody extraction, and says how MelodyAI could use each one. The research context is in
[RESEARCH.md](RESEARCH.md).

> **Redistribution policy.**
> - MelodyAI **does not redistribute** any of the datasets listed here: no audio,
>   annotations, MIDI or lyrics.
> - Users who want to use a dataset must **download it themselves** from the official source
>   and accept its license or terms of use. Several of them are non-commercial, restricted, or
>   need an access request.
> - The **built-in demo dataset is synthetic and original**: procedurally generated
>   melodies, synthesised audio, MIDI, original lyrics, transformed queries and ground truth.
>   It contains no third-party or commercial music (see the last section).

Verification notes: facts were checked in November 2026 against the linked source. When only a
paper citing the dataset could be checked, the claim is marked **(secondary source)**. Anything
that could not be confirmed is marked **(not verified)**. A dataset license generally covers the
dataset's own contribution (recordings, annotations). It does **not** clear rights in the
underlying songs, which are often commercial.

---

## Summary table

| Name | Content | Size | Annotations | License | Access | Relevance |
|---|---|---|---|---|---|---|
| [MTG-QBH](https://zenodo.org/records/1290712) | A cappella sung queries (laptop mic) + metadata of target collections | 118 queries, 17 singers, 11–98 s (mean 26.8 s); zip 223.3 MB | Query→song mapping (metadata); **reference audio not included** | Zenodo: **CC BY 4.0**; UPF page terms not verified | Open download | High (real queries), but references must be obtained by the user |
| MIR-QBSH (Roger Jang) | Sung/hummed queries of mostly nursery rhymes + ground-truth MIDI | 4,431 queries (≈8 s, 8 kHz) / 48 MIDI (secondary source) | Query→MIDI; pitch vectors (not verified) | Not verified | Original host not reachable when checked | **Very high**: exactly matches MelodyAI's MIDI-reference setting |
| IOACAS QBH | Sung/hummed queries + monophonic MIDI | 759 queries / 298 MIDI (secondary source) | Query→MIDI | Not verified | Used inside MIREX; public access not verified | High, if obtainable |
| [MIREX QBSH](https://www.music-ir.org/mirex/wiki/2020:Query_by_Singing/Humming) | Evaluation protocol over the corpora above + Essen "noise" MIDI | 2,000 (2006) to 5,000+ (2008 on) noise MIDI (secondary source) | Ground-truth lists; top-10 hit rate, MRR | Task protocol only | Wiki (TLS certificate expired when checked) | High: evaluation protocol to replicate |
| [HumTrans](https://huggingface.co/datasets/dadinghh2/HumTrans) | Humming ("Da-Da-Da") recordings with reference MIDI | ≈56.22 h, 14,614 files, 500 compositions / 1,000 segments, 10 hummers, 44.1 kHz | Note-level MIDI (score-derived); train/valid/test split | **CC BY-NC 4.0** (HF card) | Hugging Face | **Very high**: real hums + MIDI references |
| [MLEnd Hums and Whistles](https://mlenddatasets.github.io/hums_whistles) | Crowd-sourced hums and whistles of 8 well-known songs | 6,611 recordings (4,810 hums, 1,801 whistles), 226 interpreters | Song label, hum/whistle, interpreter demographics, benchmark split | Dataset license **not stated**; `mlend` package is MIT | `pip install mlend` / GitHub | Medium–high (whistling!), but no reference melodies |
| [vocadito](https://zenodo.org/records/5578807) | Solo monophonic singing, 7 languages, many devices | 40 excerpts; 58.5 MB | Frame f0, notes (2 annotators), lyrics, language | **CC BY 4.0** (Zenodo) | Open | High for f0 / note-segmentation tests |
| [MedleyDB Melody](https://zenodo.org/records/2628782) | Multitrack mixes with melody | 108 tracks | melody1/2/3 f0 annotations | **CC BY-NC-SA 4.0** | **Request access** on Zenodo | Medium (polyphonic melody extraction) |
| [MDB-stem-synth](https://zenodo.org/records/1481172) | Resynthesised MedleyDB solo stems with perfect f0 | 230 stems; 1.8 GB | Exact f0 | **CC BY-NC 4.0** | Open | Medium (pitch tracker accuracy) |
| [MIR-1K](https://zenodo.org/records/3532216) | Karaoke clips, voice/accompaniment on separate channels | 1,000 clips (4–13 s, ≈133 min) from 110 songs, 19 amateur singers; 2.6 GB | Pitch contour (semitones, 20 ms hop), unvoiced types, lyrics, vocal segments | Not stated | Zenodo backup **restricted**; uploader says to contact the authors | Medium (singing f0 / lyrics) |
| [Orchset](https://zenodo.org/records/1289786) | Symphonic music excerpts | 64 excerpts; 326.4 MB | Melody f0 | **CC BY-NC-SA 4.0** | Open | Low (orchestral, polyphonic) |
| [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) | Crawled MIDI files | LMD-full 176,581; LMD-matched / aligned 45,129 | Matches to audio (matched/aligned subsets) | **CC-BY 4.0** (collection) | Open download | Medium: scale and distractor tests; song rights caution |
| [Nottingham Music Database (ABC)](https://abc.sourceforge.net/NMD/) | Folk tunes in ABC notation | "over 1000" tunes (≈1,037 across listed categories) | Melody + chords (ABC) | Original: **no license stated**; [cleaned version](https://github.com/jukedeck/nottingham-dataset): **GPL-3.0** | Open download | Medium: melody source / distractors |
| [Essen Folksong Collection](https://github.com/ccarh/essen-folksong-collection) | Folk song melodies in Humdrum **kern | ≈6,255 (CCARH kern README); kernScores lists 8,473 (secondary source) | Symbolic melodies, metadata | **CCARH MuseData license: academic, single-user, restrictive** | GitHub / kernScores | Medium: MIREX noise set; license is restrictive |

---

## QBH query datasets

### MTG-QBH
- **Source:** [UPF page](https://www.upf.edu/web/mtg/mtg-qbh) (returned HTTP 403 to automated
  fetching, so its text was not read directly) and the
  [Zenodo record 10.5281/zenodo.1290712](https://zenodo.org/records/1290712) (v1.0, 13 Nov 2012).
- **Content:** 118 a cappella recordings by 17 subjects (9 female, 8 male), recorded with a basic
  laptop microphone and no post-processing. Durations are 11–98 s (mean 26.8 s). Includes
  three metadata files describing the music collections used in the experiments.
- **References not included:** audio of the target collections is **NOT** included because of
  copyright. Users must obtain the songs themselves. The number of target songs is given as 81
  by Ranjan & Srivastava 2023 (secondary source, not checked against the metadata).
- **License / terms:** the Zenodo record lists **CC BY 4.0** and open access. It asks users to
  cite Salamon, Serrà & Gómez (IJMIR 2013) and to email mtg-info@upf.edu about their use. The
  project brief expected **non-commercial terms** on the UPF page. That could not be verified
  (HTTP 403), and Zenodo shows no NC clause. Users should read the UPF page themselves before
  any commercial use.
- **MelodyAI use:** real-query evaluation. Because the references are polyphonic commercial
  recordings, MelodyAI's default MIDI / monophonic pipeline would need user-provided MIDI
  transcriptions of the targets, or the experimental chroma baseline.

### MIR-QBSH (Roger Jang's corpus)
- **Content (secondary source,** Ranjan & Srivastava 2023, [arXiv:2302.04577](https://arxiv.org/abs/2302.04577),
  and Amatov et al. 2023, [arXiv:2312.01092](https://arxiv.org/abs/2312.01092)**):** 48
  ground-truth monophonic MIDI files and 4,431 WAV queries, each about 8 s at 8 kHz. Recorded
  over about 7 years by students in an "Audio Signal Processing and Recognition" course at
  National Tsing Hua University, mostly nursery rhymes.
- **Access:** historically distributed from Jang's MIR Lab (mirlab.org). The URLs tried
  (`mirlab.org/dataset/public/`, `mirlab.org/dataSet/public/MIR-QBSH.zip`) returned **404** in
  November 2026. **License: not verified.** The ISMIR 2014 literature calls it "freely
  available" (secondary source).
- **MelodyAI use:** the best-matched benchmark. Queries are hums and references are MIDI,
  which is exactly MelodyAI's default reference format. Replicate the MIREX protocol: 48 targets
  plus Essen distractors, scored by top-10 hit rate and MRR.

### IOACAS QBH
- **Content (secondary source,** MIREX QBSH wiki pages found by search**):** the 2009 wiki
  lists two IOACAS sets, 355 queries / 106 MIDI and 404 queries / 192 MIDI, combined in later
  years as **759 queries / 298 monophonic ground-truth MIDI**. Amatov et al. 2023 refer to the
  355 / 106 set as "ThinkIt". MIREX result pages for an "Ioacas Dataset" exist for 2012–2016
  (nema.lis.illinois.edu; TLS certificate expired when checked).
- **License and public access: not verified.**
- **MelodyAI use:** a second QBSH benchmark, if a copy can be obtained legitimately.

### MIREX QBSH task
- Annual task since 2006 ([2020 wiki page](https://www.music-ir.org/mirex/wiki/2020:Query_by_Singing/Humming);
  [2006 page](https://music-ir.org/mirex/wiki/2006:QBSH:_Query-by-Singing/Humming)).
  Secondary source (search snippets; direct fetch failed because of an expired certificate):
  2006 used 48 MIR-QBSH ground-truth MIDI + about 2,000 Essen MIDI as noise. From 2008 the noise
  set was "5000+ Essen collection". The **top-10 hit rate** scores 1 if the ground truth is
  in the top 10. MRR is also reported.
- MIREX distributes a protocol, not a dataset. MelodyAI's evaluation scripts can mirror the
  metrics (top-k hit rate, MRR) on any corpus.

---

## Humming / whistling corpora

### HumTrans (2023)
- **Paper:** Liu, Li, Li, Shan (Tencent), [arXiv:2309.09623](https://arxiv.org/abs/2309.09623).
  **Data:** [huggingface.co/datasets/dadinghh2/HumTrans](https://huggingface.co/datasets/dadinghh2/HumTrans);
  code: [github.com/shansongliu/HumTrans](https://github.com/shansongliu/HumTrans).
- **Content:** ≈56.22 h, 14,614 WAV files at 44.1 kHz. 500 compositions split into 1,000
  segments (4.9–29.9 s, mean 13.9 s). 10 college students (5 male, 5 female, musically
  trained) hummed with the syllables "Da-Da-Da" while listening to the melody through
  headphones. Recordings with DNSMOS below 2.8 were excluded.
- **Annotations:** reference MIDI exported from MuseScore 4 engravings (melody only, no
  chords). Labels come from the score and synchronised performance, not manual transcription.
  An official train/valid/test split is in `train_valid_test_keys.json`. A follow-up
  ("Dynamic HumTrans", 2024) reports onset/offset problems in the labels and proposes
  corrections (secondary source, not verified).
- **License:** **CC BY-NC 4.0** according to the Hugging Face card (the paper states none).
- **Limitations:** trained hummers, guided by headphones, so tempo and pitch are far more
  regular than casual users. Only 10 voices. Non-commercial.
- **MelodyAI use:** (a) evaluate note segmentation on real hums; (b) build a retrieval
  benchmark (hum segment → its MIDI among all 1,000 segments); (c) research-only fine-tuning
  or validation of the neural encoder with real hums. The resulting model would inherit the
  NC terms.

### MLEnd Hums and Whistles
- **Source:** [project page](https://mlenddatasets.github.io/hums_whistles),
  [GitHub MLEndDatasets/HumsAndWhistles](https://github.com/MLEndDatasets/HumsAndWhistles),
  `pip install mlend` (`download_hums_whistles()`, `hums_whistles_load()`). Created at Queen
  Mary University of London.
- **Content (counted from `MLEndHWD_audio_attributes.csv` in the repo):** 6,611 recordings by
  226 interpreters. 4,810 hums and 1,801 whistles of 8 song labels: Potter, Panther, Mamma,
  StarWars, Hakuna, Rain, Frozen, Showman. Demographics CSV and "Benchmark" train/test splits
  are included.
- **License:** the dataset license is **not stated** on the project page or the GitHub repo.
  The `mlend` Python package is MIT on PyPI. The song labels refer to commercial
  compositions **(not verified which exact works)**.
- **Limitations:** only 8 songs; no reference melodies or MIDI; no time alignment.
- **MelodyAI use:** test robustness to **whistling** and to casual non-expert hums. Users would
  have to create reference melodies for the 8 songs themselves, under the songs' rights.

---

## Singing and melody annotation datasets

- **vocadito** ([Zenodo 10.5281/zenodo.5578807](https://zenodo.org/records/5578807), v3,
  Bittner et al.): 40 short solo monophonic singing excerpts, 7 languages, singers with varying
  training, many recording devices. Frame f0, two note annotations (two annotators), lyrics,
  language. **CC BY 4.0** on Zenodo and in the mirdata loader. The mirdata quick-reference
  table lists CC BY-NC-SA 4.0 instead, so the two sources conflict. Use: benchmark pYIN vs
  Basic Pitch vs torchcrepe and MelodyAI's note segmentation, with inter-annotator agreement
  as an upper bound.
- **MedleyDB Melody** ([Zenodo 2628782](https://zenodo.org/records/2628782),
  [MedleyDB site](https://medleydb.weebly.com/)): 108 multitrack mixes with melody1
  (predominant melody, single source), melody2 (predominant, multiple sources) and melody3
  (all melodic lines) f0. **CC BY-NC-SA 4.0**, non-commercial research. **Access by request**
  on Zenodo. Use: evaluate future polyphonic-reference paths (Demucs + f0, chroma baseline).
- **MDB-stem-synth** ([Zenodo 1481172](https://zenodo.org/records/1481172),
  [site](https://synthdatasets.weebly.com/mdb-stem-synth.html)): 230 MedleyDB solo stems
  resynthesised to have perfect f0 labels (Salamon et al., ISMIR 2017). **CC BY-NC 4.0**, open,
  1.8 GB. Use: exact f0 accuracy and octave-error rates of the extractors.
- **MIR-1K** ([Zenodo backup 3532216](https://zenodo.org/records/3532216); Hsu & Jang, IEEE
  TASLP 18(2), 2010): 1,000 clips (4–13 s, ≈133 min) from 110 Chinese karaoke songs, sung by
  8 female and 11 male amateurs. Voice and accompaniment on separate channels. Pitch in
  semitones (40 ms frames, 20 ms hop), unvoiced types, lyrics, vocal/non-vocal segments,
  spoken lyrics. **License not stated.** The Zenodo copy is **restricted**, and its uploader
  says to contact the original authors. The mirlab.org links did not resolve when checked.
  Use: singing f0 evaluation; Mandarin lyrics/ASR tests.
- **Orchset** ([Zenodo 1289786](https://zenodo.org/records/1289786); Bosch, Marxer & Gómez, JNMR
  2016): 64 symphonic excerpts with melody annotation. **CC BY-NC-SA 4.0**, open, 326.4 MB.
  Use: stress test only; orchestral melody is far from MelodyAI's target domain.

## Symbolic melody sources

- **Lakh MIDI Dataset** ([project page](https://colinraffel.com/projects/lmd/); Raffel, PhD
  thesis 2016): LMD-full 176,581 unique MIDI files. LMD-matched 45,129 files matched to audio,
  also available as LMD-aligned. A "clean MIDI" subset exists (count not stated). Distributed
  under **CC-BY 4.0**. The author notes that per-file attribution is not feasible, and that he
  did not transcribe the files. Most files are fan transcriptions of **copyrighted songs**, so the
  collection license does not clear the compositions. Use: scale tests and distractors, which
  needs melody-track selection. Do not ship derived indexes.
- **Nottingham Music Database** ([ABC version](https://abc.sourceforge.net/NMD/), maintained by
  James Allwright; original by Eric Foxley): "over 1000 Folk Tunes" (jigs, reels, hornpipes,
  waltzes, Playford, Morris, carols, the Ashover collection). The page has **no general license
  statement**. It notes that the IPR of the Ashover collection resides with Mick Peat. The
  [jukedeck cleaned version](https://github.com/jukedeck/nottingham-dataset) is **GPL-3.0**.
  The status is **not clearly permissive**. Use locally as references or distractors, or as
  melodies for synthetic-hum training. Do not redistribute.
- **Essen Folksong Collection** ([CCARH GitHub, Humdrum kern](https://github.com/ccarh/essen-folksong-collection)):
  collected by Helmut Schaffrath, kern version by David Huron. The README describes about
  6,255 transcriptions, mostly German (5,370), including 213 children's songs. kernScores lists
  8,473 songs including Chinese material (secondary source). The repository's `license.txt` is
  the **CCARH MuseData license**: academic research only, single-user, no embedding in
  distributed teaching materials, CCARH must be acknowledged. GitHub reports `NOASSERTION`.
  So, contrary to a common assumption, this edition is **not permissively licensed**. The
  melodies themselves are traditional, but the encodings carry CCARH's terms. Use: MIREX-style
  distractor set for local research only.

---

## mirdata coverage

[mirdata](https://github.com/mir-dataset-loaders/mirdata) (BSD-3-Clause; PyPI 1.0.0) provides
download, validation and loading code. It does not provide data hosting or licenses. Loader
modules checked in the GitHub repository (`mirdata/datasets/`):

| Dataset | mirdata loader | License per mirdata |
|---|---|---|
| MedleyDB Melody | `medleydb_melody` (also `medleydb_pitch`) | CC BY-NC-SA 4.0; access request |
| MDB-stem-synth | `mdb_stem_synth` | CC BY-NC 4.0 |
| Orchset | `orchset` | CC BY-NC-SA 4.0 |
| vocadito | `vocadito` | Loader: CC BY 4.0; quick-reference table: CC BY-NC-SA 4.0 |
| iKala (vocal f0 + lyrics) | `ikala` | Custom; "most of the iKala dataset is not available for download" |
| MTG-QBH, MIR-QBSH, IOACAS, HumTrans, MLEnd, MIR-1K, Lakh, Nottingham, Essen | **No loader** | — |

Other loaders that may matter later: `dali` (lyrics/notes aligned to audio), `tonas`,
`cante100`, `dagstuhl_choirset` (F0, notes; CC BY 4.0 per quick reference), `maestro`, `slakh`,
`groove_midi`. MelodyAI could offer mirdata as an **optional** extra for the supported
datasets. The datasets without a loader need small custom readers that point at a user-supplied
local path.

## MTG software and datasets page

The index at [upf.edu/web/mtg/software-datasets](https://www.upf.edu/web/mtg/software-datasets)
returned HTTP 403 to automated fetching, so its contents are **not verified**. MTG-authored
datasets referenced here (MTG-QBH, Orchset, MDB-stem-synth co-authors) were checked through
their Zenodo records.

---

## How MelodyAI can use these datasets (plan)

1. **Default (no external data):** the built-in synthetic demo catalogue (below).
2. **Pitch/notes front-end evaluation:** vocadito, MDB-stem-synth, MIR-1K (f0 RPA/RCA,
   octave-error rate, note F-measure).
3. **Retrieval evaluation with real queries:** MIR-QBSH and IOACAS (MIDI references, MIREX
   protocol). Then HumTrans (hum → MIDI segment). Then MTG-QBH and MLEnd, which need
   user-created references.
4. **Distractors and scale:** Essen, Nottingham or Lakh melodies as non-target references,
   used locally only.
5. **Neural encoder:** train on synthetic hums from original melodies (default). Fine-tuning on
   HumTrans or other NC data is research-only and must be documented with the model.
6. **Data handling:** all dataset paths are user-configured. MelodyAI stores derived features
   only in the user's local data directory and never commits or uploads them.

## Built-in demo dataset (synthetic and original)

- About 48 melodies generated procedurally with fixed random seeds (reproducible): scales,
  rhythms and phrase structure come from the generator, not from existing songs.
- For each melody: MIDI, synthesised audio, and for some songs, **original** lyrics written for
  the project.
- Queries are generated from the melodies with controlled transformations (transposition,
  tempo change, vibrato, detuning, wrong or missing notes, duration jitter, noise, partial
  excerpts), and ground truth is recorded for every query.
- No commercial music, MIDI or lyrics are included. Because the queries are synthetic, scores on
  the demo set measure **pipeline correctness and relative robustness**, not real-world accuracy.
