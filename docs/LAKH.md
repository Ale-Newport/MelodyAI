# Humming real songs: the Lakh MIDI melody index

The open world ([OPEN_WORLD.md](OPEN_WORLD.md)) recognises real songs by their **words**. A hummed or whistled
query has no words, so MelodyAI also searches its **melody** among the vocal lines of about 25,000 real songs
taken from the [Lakh MIDI dataset](https://colinraffel.com/projects/lmd/).

> **Attribution.** Lakh MIDI Dataset by Colin Raffel, CC-BY 4.0 — *Learning-Based Methods for Comparing
> Sequences, with Applications to Audio-to-MIDI Alignment and Matching*, PhD thesis, Columbia University, 2016.
> Titles and artists come from the Million Song Dataset (Bertin-Mahieux, Ellis, Whitman and Lamere, ISMIR 2011).
> The MIDI files were transcribed by many anonymous authors. The dataset is downloaded at setup time, not
> shipped in this repository.

## Setup

```bash
make lakh            # = melodyai lakh-download (~1.5 GB) + melodyai lakh-build (~30-60 min on an M-series Mac)
```

* `lakh-download` fetches *LMD-matched* (45,129 MIDI files matched to Million Song Dataset tracks, 1.4 GB),
  `match_scores.json` and the MSD `unique_tracks.txt` (titles and artists) into `<store>/lakh/`, and unpacks
  the archive there (4.8 GB on disk; it can be deleted after the build).
* `lakh-build` writes `<store>/world_melodies/` (~380 MB). `--reembed` keeps the extracted melodies and only
  rebuilds the search index (e.g. after training a new encoder); `--limit N` builds a small index.
* Settings → *Online song search* shows the number of songs indexed. The server loads the index in the
  background at startup; recognition works without it.

## What is built

1. **Songs.** The matched MSD tracks are grouped by song (normalised title + primary artist, the same key as the
   lyrics index): 26,225 songs. Per song, the MIDI files with the best match scores are used.
2. **The vocal line of each file.** People hum the voice, but a MIDI file has many tracks. The old rule
   (a track named "melody"/"vocal"…, else the highest pitched one) picked the vocal track in only **25 %** of 666
   karaoke files whose lyric events identify it. A small logistic model over track features (name keywords,
   General MIDI program family, pitch range, polyphony, chords, coverage, density) picks it in **83 %**
   (92 % within its first two; 2-fold cross-validation), see `melodyai/features/melody_track.py`. Files with
   synced lyrics use the lyric-bearing track directly (4,232 melodies). When the model hesitates between two
   tracks, both are indexed (7,169 runner-up melodies). The same model is now used when MIDI files are imported
   into the local catalog.
3. **Index.** 41,231 vocal lines of **25,074 songs** (289 songs are kept out for the evaluation), cut into
   10.1 M overlapping windows, embedded by the neural melody encoder and stored in an IVF-PQ index (243 MB).
4. **Encoder.** `models/melody_encoder_lakh.pt` is the demo encoder's architecture trained on the synthetic
   compositions *plus* 30,000 real vocal lines (`configs/train_lakh.yaml`, 34 min on MPS; the 597 songs used by
   the evaluation were excluded from training). The demo catalog keeps `models/melody_encoder.pt`.

A query is searched in two stages (`melodyai/matching/scalable.py`): the 40 most similar melodies by embedding,
then window DTW + note alignment on those only. Songs found in the lyrics index get their cover, Deezer link and
a small popularity tie-break (+0.05 at most).

## How well it works

Synthetic hums and whistles (excerpts of 7-12 s, rendered with the benchmark's hum / whistle / tempo / detune /
noise conditions and analysed by the real pitch pipeline) of songs **in** the 25k-song index, and of 150 real
songs **not** in it. Songs never seen by the encoder in training.

| Encoder | Top-1 (all Lakh songs) | Top-5 | Top-1, popular songs* | Latency p50 / p95 |
|---|---|---|---|---|
| demo encoder (synthetic training) | 31 % | 49 % | 28 % (42 % with popularity) | 1.2 s / 2.3 s |
| **Lakh encoder + popularity** (used) | **35 %** | **55 %** | **52 %** | 1.1 s / 1.9 s |

\* songs that are also in the popular-songs lyrics index (2,258 of the 25,074).

For comparison, the synthetic benchmark (random generated melodies) reaches 85 % top-1 at 10,000 songs: real pop
melodies resemble each other far more (scales, repeated notes, shared idioms), which is the known hard part of
query-by-humming at scale.

**Decisions.** Correct and wrong answers score alike, so a *confident match* requires a large lead over the second
song (`world_melody` thresholds: accept 0.5, margin 0.3): about 8 % of the hums, with 1 wrong confident answer
in 446 queries. Everything else is shown as *possible songs*, with the right one among the first five about half
of the time. In the app, a melody answer is only used when the words (lyrics engines) found nothing better.

## Limitations

* **Coverage.** Lakh MIDI is mostly English-language pop/rock released before ~2011. Recent hits, TikTok trends and
  most Latin music are not in it: sing the words for those (the lyrics search covers them).
* **Real voices are untested.** All numbers above use synthetic hums; real humming is less precise.
* **Covers in the metadata.** The Million Song Dataset match is sometimes a cover version, so a famous song can be
  listed under another artist (the title is right; the popularity tie-break then misses it).
* **MIDI quality varies.** Some files are arrangements or medleys; the chosen track is the vocal line in about 83 %
  of the cases measured, so some songs are indexed with an instrumental line.
* For a commercial-scale catalogue of hummed songs, a licensed provider (ACRCloud) remains the practical option
  ([SCALING.md](SCALING.md)).
