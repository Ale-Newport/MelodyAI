# Evaluation protocol

```bash
make evaluate          # full benchmark (configs/eval_default.yaml) + regenerates docs/RESULTS.md
make evaluate-quick    # 16 songs, 5 conditions (≈1–2 min)
make report            # rebuild docs/RESULTS.md from the latest run
```

Benchmarks can also be launched from the web UI (Research Lab → Benchmarks), which shows every
table, the per-query inspector and the figures of any stored run.

## What is measured

| Block | Queries | Metrics |
|---|---|---|
| Melody, in-catalog | for each of the 48 demo songs and each condition, one excerpt synthesised and analysed by the real pipeline | Top-1/3/5/10, MRR, mean/p50/p95 matching latency |
| Melody, rejection | 48 hummed excerpts of melodies **not** in the catalog (generator seeds 300000+) + 16 non-melodic signals (noise, near-silence, single tone, glissando) | AUC (correct top-1 vs out-of-catalog score), false-accept rate and accepted-correct rate at the tuned threshold |
| Calibration | in-catalog + out-of-catalog queries | Platt scaling fitted on dev half; ECE (10 bins), Brier, reliability diagram on test half |
| Pitch extractors | 24 songs × {hum_clean, whistle, hum_noisy} re-analysed with pYIN, CREPE-tiny, Basic Pitch | Top-1, MRR, pitch-extraction time |
| Lyrics (text) | per lyric song: exact two-line span, 4-word partial, misremembered (25% content words replaced, one dropped), ASR-like typos (35% of words); 10 original distractor lines + 10 random bags of catalog words | Top-1/MRR per condition, rejection |
| Speech → lyrics | one two-line span per lyric song spoken by macOS `say` voices (en-US/GB/AU, es-MX), transcribed by Whisper `base` with auto-detected and with given language; 10 hum/whistle queries for the hallucination check | song Top-1, language-ID accuracy, WER, real-time factor, rate of hums producing words |
| Fusion | 40 combined queries: short (2.5–4 s) or noisy hum of a lyric song + 4 words of it with ~30% typos | Top-1/MRR for melody only, lyrics only, weighted, RRF, CombMNZ, learned (dev→test) |
| Reference modality | hum_clean / chorus_mid_song / whistle queries of 16 songs vs references from MIDI, monophonic audio (pYIN), polyphonic synthetic mix (Basic Pitch + skyline, and pYIN on the mix) | Top-1/MRR + note-transcription F1 of the reference (mir_eval) |
| Cost | — | indexing time per song, window-matrix size, RSS delta, neural index build time |

### Query conditions (`melodyai/demo/queries.py`)

| Condition | Excerpt | Transformations |
|---|---|---|
| `hum_clean` | 5–10 s anywhere | key ±5, tempo 0.85–1.18×, 15-cent detune, 8% duration jitter, SNR 25–40 dB |
| `hum_transposed` | 5–10 s | key shift 5–11 semitones up or down |
| `hum_tempo` | 5–10 s | tempo 0.60–0.72× or 1.35–1.60× |
| `hum_detuned` | 5–10 s | 40-cent per-note detune, 10% wrong notes (±1–2 semitones) |
| `hum_missing_notes` | 5–10 s | 12% notes dropped (previous note held), 8% notes split in two |
| `hum_noisy` | 5–10 s | pink noise at 10 dB SNR + reverb (RT60 0.4 s) |
| `hum_short` | 2.5–4 s | — |
| `chorus_mid_song` | starts inside a chorus that begins after 1 s | — |
| `whistle` | 5–10 s | whistle timbre in the 78–92 MIDI register (≈ 0.8–1.7 kHz), less vibrato |
| `sing_vowel` | 5–10 s | "ah" vowel timbre |

All hummed/sung renditions are moved by octaves into a vocal register (median MIDI 50–68),
because people sing in their own range, not in the reference's.

## Protocol details

* **Reproducibility**: one seed drives everything. The query set is cached by a hash of
  (catalog size, conditions, seed, extractor, feature version), stored in the run metadata.
* **No tuning on test**: abstention thresholds and Platt models are fitted on songs with even
  indices (and even-numbered out-of-catalog queries) and reported on the odd half. The neural
  model never sees demo-catalog compositions (see [TRAINING.md](TRAINING.md)).
* **Latency** is matching time only, single query, on the machine recorded in `env.json`.
  Pitch extraction time is reported in the pitch-extractor block. End-to-end API latency (decode +
  pitch + matching + explanation) is ~0.5–4 s for 5–15 s recordings on an Apple M-series laptop,
  dominated by pYIN and, in Auto mode, Whisper.
* **Artifacts** per run in `experiments/runs/<UTC>-<name>/`: `config.yaml`, `env.json` (versions,
  hardware, git commit), `metrics.json`, `per_query.jsonl` (every query × algorithm: target, returned
  song, rank, scores, latency), `figures/*.png`.

## What these numbers do *not* say

* Synthetic renditions are cleaner and more regular than people. Ceiling effects on clean
  conditions are expected; the informative columns are the hard conditions (noise, short, detuned),
  rejection and calibration.
* The catalog has 48 songs. Retrieval gets harder as the catalog grows (more near-duplicate
  passages); no large-catalog experiment has been run.
* Polyphonic results use simple synthetic mixes (no vocals, drums or production effects).
* Speech results use text-to-speech, not singing.
* Real-user validation requires recordings of people (e.g. MTG-QBH, MIR-QBSH, or your own with
  consent). The adapters are not part of this release; see [DATASETS.md](DATASETS.md).
