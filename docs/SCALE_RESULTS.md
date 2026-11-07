# Scale benchmark results

> **Auto-generated** by `python -m melodyai.cli report-scale` from `experiments/runs/20261107T145721Z-scale/metrics.json` (run `20261107T145721Z-scale`, 2026-11-07T14:57:21.030983+00:00, arm64 · 12 CPUs · macOS-27.0.1-arm64-arm-64bit, wall time 35 min). Do not edit by hand.

> **Synthetic data.** The 48 demo songs are hidden among procedurally generated distractors rendered with the same simple synthesisers. Real catalogs differ (richer timbres, vocals, but also covers, remixes and near-duplicates). These tables measure how the *system* behaves as the index grows, not real-world accuracy. See `docs/SCALING.md` for the architecture and extrapolations.

## Recordings and TikTok-style clips (audio fingerprint)

8-second clips of the 48 indexed recordings. *top-1* = correct song ranked first; *accepted* = top-1 correct **and** score ≥ 0.5 (the threshold that would answer "match").

| Edit | 48 recordings: top-1 / accepted | 500 recordings: top-1 / accepted | 2,000 recordings: top-1 / accepted |
|---|---|---|---|
| none | 100% / 100% | 100% / 100% | 100% / 100% |
| speed 1.25 | 100% / 98% | 100% / 100% | 100% / 100% |
| speed 1.1 | 100% / 100% | 100% / 100% | 100% / 100% |
| speed 0.8 | 96% / 85% | 85% / 81% | 83% / 77% |
| pitch up 2 | 98% / 92% | 98% / 92% | 96% / 92% |
| stretch 1.15 | 98% / 90% | 96% / 92% | 94% / 90% |
| noise 5db | 98% / 96% | 98% / 96% | 96% / 94% |
| phone | 96% / 83% | 94% / 85% | 92% / 85% |
| reverb | 96% / 75% | 85% / 73% | 77% / 71% |
| overlay | 73% / 6% | 69% / 6% | 67% / 6% |
| **false accepts**: clips of unknown recordings | 6.2% | 10.4% | 10.4% |
| **false accepts**: hummed queries | 0.0% | 0.0% | 0.0% |
| AUC correct vs negatives | 0.972 | 0.972 | 0.973 |
| latency mean / p95 | 52 / 75 ms | 118 / 207 ms | 383 / 768 ms |
| index size | 5 MB (0.61 h audio) | 76 MB (9.24 h audio) | 312 MB (38.02 h audio) |

Storage: **8.22 MB per hour of audio** (133.8 hash postings/s × 16 B + peak maps × 10 B). Edits: speed = resampling (tempo *and* pitch change, as TikTok "sped up"/"slowed"; slowed also adds reverb), pitch_up_2 = phase-vocoder pitch shift, stretch = tempo-only time stretch, phone = 300–3400 Hz band-limit + saturation, overlay = another voice as loud as the music.

## Humming / whistling vs catalog size (two-stage search)

Stage 1: neural window embeddings in an IVF-PQ index (2048 lists, 16 bytes/vector, nprobe 32) → best 50 songs. Stage 2: window DTW + note alignment on those songs only. The same 480 synthetic hum/whistle queries as the main benchmark, plus its out-of-catalog/non-melodic queries for the AUC column.

| Songs | Two-stage top-1 | top-5 | MRR | Stage-1 candidate recall | Neural ANN alone top-1 | Exhaustive note-align top-1 (subset) | Latency stage 1 + 2 | Exhaustive latency | ANN index | AUC vs OOC |
|---|---|---|---|---|---|---|---|---|---|---|
| 48 | 96.2% | 96.7% | 0.964 | 97.5% | 88.5% | 95.8% (n=96) | 9 + 55 ms | 7 ms | 0.00 M vectors, 1 MB | 0.978 |
| 1,000 | 91.0% | 91.7% | 0.914 | 92.3% | 72.3% | 92.7% (n=96) | 6 + 143 ms | 137 ms | 0.09 M vectors, 3 MB | 0.974 |
| 10,000 | 84.8% | 85.2% | 0.850 | 85.6% | 57.7% | 88.5% (n=96) | 8 + 468 ms | 1525 ms | 0.86 M vectors, 22 MB | 0.957 |
| 100,000 | 72.1% | 73.8% | 0.729 | 73.8% | 43.8% | – | 14 + 540 ms | – | 8.58 M vectors, 207 MB | 0.943 |

Stage-1 candidate recall (is the right song among the first *k* songs retrieved by the ANN?) — the ceiling for the two-stage top-1:

| Songs | recall@10 | recall@50 | recall@200 | recall@500 |
|---|---|---|---|---|
| 48 | 95.4% | 97.5% | 97.5% | 97.5% |
| 1,000 | 88.1% | 92.3% | 95.2% | 95.8% |
| 10,000 | 76.9% | 85.6% | 90.4% | 91.9% |
| 100,000 | 61.7% | 73.8% | 81.7% | 84.6% |

Top-1 by condition at the largest size:

| Condition | Top-1 | MRR |
|---|---|---|
| chorus_mid_song | 89.6% | 0.896 |
| hum_clean | 85.4% | 0.854 |
| hum_detuned | 72.9% | 0.781 |
| hum_missing_notes | 52.1% | 0.521 |
| hum_noisy | 37.5% | 0.385 |
| hum_short | 43.8% | 0.448 |
| hum_tempo | 85.4% | 0.854 |
| hum_transposed | 85.4% | 0.854 |
| sing_vowel | 83.3% | 0.844 |
| whistle | 85.4% | 0.854 |

## Linear storage extrapolation (not measured)

Index size grows linearly with the catalog; the per-unit costs below are *measured* in this run, the totals are simple multiplications for a 250-million-track catalog (Spotify's reported order of magnitude, average 3.5 min). Accuracy at that scale is **not** extrapolated: it depends on catalog content and was not measured.

| Index | Measured unit cost | 1 M tracks | 250 M tracks |
|---|---|---|---|
| Fingerprint postings + peak maps | 8.22 MB / audio hour | 480 GB | 119.9 TB |
| Melody IVF-PQ (window embeddings) | 2.1 kB / song (86 windows × 24 B) | 2.1 GB | 0.52 TB |

At those sizes both indexes must be sharded across machines (they are append-only segment files, so sharding by song id is straightforward); see `docs/SCALING.md`.
