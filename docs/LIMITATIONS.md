# Limitations and known issues

MelodyAI is a research prototype. This page lists what is verified, what is experimental and
what is not done.

## Verified (tests + benchmark, in this repository's environment)

* Classical melody matching (window DTW, subsequence DTW, chroma DTW, note alignment, ensemble)
  on the synthetic demo catalog, including transposed, tempo-changed, detuned, mid-song, whistled,
  short and noisy queries (`docs/RESULTS.md`).
* Rejection of noise / silence / held tones / unknown melodies with thresholds tuned on a dev split.
* Lyrics search on typed text with misspellings and misremembered words (English and Spanish).
* Whisper `base` transcription of *synthetic speech* (macOS `say`) and its hallucination guard on hums;
  Whisper `large-v3-turbo` on synthesised speech of real songs' lines in the open-world evaluation (WER 0.04).
* Open-world search components (world lyrics index, sources, verification, grouping, sync, API) with unit
  and integration tests that never touch the network (mocked HTTP transports, invented lyrics).
* Catalog import (MIDI, monophonic audio, lyrics TXT/JSON, CSV/JSON manifests), incremental
  indexing, deletion, reindexing, feature-version staleness.
* REST API (FastAPI) and the web UI flows (recognition by upload, explorer, library, lab, settings),
  exercised in Chromium-based browser automation on macOS. Microphone capture code paths were
  **not** exercised by automation (no physical microphone input in the automated browser).
* Neural encoder training on Apple Silicon (MPS) and inference from the exported checkpoint.

## Experimental
* **Open-world recognition of real songs by their words** ([OPEN_WORLD.md](OPEN_WORLD.md)). Evaluated on 100
  real songs ([WORLD_RESULTS.md](WORLD_RESULTS.md)) with typed queries and synthesised speech against a
  24,640-song index: 97% top-1 for typed queries of indexed songs, 92% top-1 with index + online search,
  70% top-1 for the online sources alone, 0 false matches. No real sung recording or real TikTok clip has
  been evaluated (the synthetic "sung" clips are the weakest case, 62% top-1). It depends on free
  third-party services (LRCLIB, Deezer) with no service guarantee.
* **Audio fingerprinting / `clip` mode**: verified on synthetic polyphonic recordings with TikTok-style
  edits (docs/SCALE_RESULTS.md). Not tested on real produced music, where timbral diversity helps but
  covers/remixes/near-duplicates hurt. A voice-over as loud as the music defeats it (6% accepted), heavy
  reverb degrades it (71%), and at the default threshold 10.4% of clips of *unknown* synthetic recordings were
  accepted as a match (AUC 0.97): recalibrate on real data before trusting "match". Search latency grows with
  the index (52 ms at 0.6 h of audio, 383 ms at 38 h): sharding is required for large catalogs.
* **Licensed provider (ACRCloud)**: implemented against the documented API and tested with a mocked HTTP
  transport; **no live round trip was run** (no credentials). Its accuracy, costs and data handling are the
  provider's.
* **Two-stage IVF-PQ melody search**: measured up to 100,000 synthetic songs in a benchmark harness
  (docs/SCALE_RESULTS.md): top-1 falls from 96% (48 songs) to 72% (100,000 songs), limited by the neural first
  stage's recall. The web app's catalog still keeps all melody windows in memory for the
  exhaustive matchers; wiring the two-stage search into the live catalog as its default is future work.

* **Neural embeddings**: trained only on synthetic renditions of procedurally generated
  melodies. Validation is on unseen compositions of the *same generator style*.
* **Polyphonic references** (Basic Pitch + skyline): measured only on simple synthetic mixes
  where the melody is the top voice. Expect poor results on produced songs with vocals, drums or
  accompaniment above the melody. No source separation is included.
* **Calibrated probabilities**: Platt scaling fitted on synthetic queries; not validated on real
  recordings. The UI labels it accordingly and defaults to relative scores.
* **Learned fusion**: trained on 20 combined queries; indicative only.
* **PostgreSQL**: supported through SQLAlchemy URLs but not covered by the test suite.
* **Docker image**: provided as an alternative; not built or run in the development environment.

## Not done / out of scope

* **Indexing Spotify or TikTok audio**: prohibited by their terms (no audio access, no ML use, no
  scraping). Commercial coverage *by audio* is only available through a licensed provider
  (docs/SCALING.md); commercial songs with words are recognised by their lyrics (OPEN_WORLD.md).
* **Humming / whistling real songs without words** works only for the ~25,000 songs of the Lakh MIDI
  index (`make lakh`, [LAKH.md](LAKH.md)): mostly English pop/rock before ~2011, and with modest accuracy
  (synthetic hums: 35 % top-1, 55 % top-5; mostly "possible songs" rather than confident matches). Recent
  hits, TikTok trends and Latin music need their words. The licensed provider's humming database (ACRCloud)
  is the keyed option for wide coverage.
* **Instrumental TikTok / trend audio**: no words to search. It is recognised only if the track is in the
  local fingerprint index (imported by you) or through the licensed provider.
* **Distributed serving** (sharded fingerprint/ANN services, replication, ingestion queues): the storage
  formats are shard-friendly, but everything runs in one process.

* No evaluation on real human query datasets (MTG-QBH, MIR-QBSH, IOACAS, HumTrans). Licences and
  access require the user to obtain them; adapters are future work.
* No large-catalog experiments (thousands of songs). Window DTW scans all windows (with a linear
  scaling prefilter above 30,000 windows), which grows linearly with the catalog.
* No lyrics alignment to the melody (we do not know *where* in the song the words occur in time).
* No real-time streaming recognition; recordings are processed after capture.
* No user accounts or multi-tenant security; the server is meant to run locally.
* GitHub Pages: not configured. A static build cannot run the Python recognition engine; a
  backend must be hosted separately for a public demo.

## Known technical issues

* **FAISS + PyTorch on macOS**: the `faiss-cpu` and `torch` wheels each bundle `libomp`. Loading
  both in one process aborts ("OMP: Error #15"); `KMP_DUPLICATE_LIB_OK=TRUE` turned the abort into a
  segfault in our tests. On macOS the vector backend therefore defaults to NumPy (exact matrix
  product, plus a NumPy IVF index for approximate search). FAISS is used automatically on other
  platforms and can be forced with `MELODYAI_VECTOR_BACKEND=faiss`. At demo scale exact NumPy search
  takes about a millisecond.
* **faster-whisper on Apple Silicon** runs on CPU (CTranslate2 has no MPS backend). `base` runs at a
  real-time factor well below 1 on an M-series CPU; larger models are slower. `large-v3-turbo` took
  6.1 s for a 5.3 s spoken clip in the live check, and 5.6–6.3 s per clip in the open-world pilot; both
  were measured before the encoder cache and the token budget ([ALGORITHMS.md](ALGORITHMS.md#6-lyrics-melodyailyrics)).
  An interleaved A/B on the same machine then measured about 40 % less per clip (4.9 s instead of 8.2 s
  mean on the evaluation's spoken and "sung" clips, with a popular-songs sync running in parallel). A clip
  that no decoding transcribes cleanly is decoded up to three times (the encoder still runs once).
* **Keys and `MELODYAI_OFFLINE` come from the process environment.** `MELODYAI_GENIUS_TOKEN`,
  `MELODYAI_ANTHROPIC_API_KEY`, `MELODYAI_ACRCLOUD_*` and `MELODYAI_OFFLINE` are read with
  `os.environ`. The server copies the `MELODYAI_*` entries of `.env` into its environment at startup;
  the CLI does not, so export them in the shell for CLI commands.
* **pkg_resources**: `pretty_midi` / `resampy` (used by torchcrepe) still import `pkg_resources`,
  so `setuptools<81` is pinned.
* **Browser recording**: echo cancellation / noise suppression / AGC are disabled on purpose
  (they distort pitch). Bluetooth headsets may still apply their own processing. Safari requires
  a user gesture before audio capture and only works on `localhost` or HTTPS.
* **pYIN latency** grows with recording length (~0.1–0.2 s of CPU per second of audio).
* **Repeated notes** in legato humming/whistling are often merged by the note segmenter; the
  contour-based matchers are insensitive to this, the note-alignment matcher merges repeats on
  both sides by design.

## Expected errors in real use

* Very short queries (< 3 s) or mostly-rhythmic passages (repeated notes) carry little information;
  several songs may then score similarly → *uncertain* or *no match*.
* Large intonation drift within a query (singer slowly going flat) reduces DTW similarity; the
  residual key search compensates only a constant offset.
* Octave jumps by the singer in the middle of a phrase are not normalised (only short tracker
  octave errors are).
* Songs sharing a melodic passage (quotations, covers, common motifs) are inherently ambiguous;
  the margin-based abstention then reports *uncertain*.
* Lyrics ASR errors on proper names and on singing (vs speaking) are larger than on the TTS speech
  used in the benchmark. In the open-world evaluation a synthetic "sung" version gave WER ~0.5, and
  sung Spanish lines were often decoded as English.
* Open world: a song without lyrics on LRCLIB is never returned (540 of the 7,494 tracks of the
  `standard` sync had none). Misremembered or very short lines often fail online, because Deezer's
  search needs an exact phrase. A cover or karaoke version can be shown instead of the original.
  Short, generic lines can score high on an unrelated song. The first lookup of a song that is not
  indexed needs the internet.

## Calibration limits

Thresholds in `configs/calibration.json` were tuned for ≤5% false accepts on *synthetic*
out-of-catalog hums. Real recordings have different score distributions; recalibrate with
labelled real queries before trusting the *match* / *no match* boundary, and treat the displayed
score as a relative similarity.

The open-world (`world`) thresholds (accept 0.62, margin 0.07, uncertain 0.45) were calibrated on the
complete 24,640-song index with real songs: the zero-false-match fit was chosen over a looser fit that
accepted 9 % more queries with 3.6 % confident wrong answers. The queries were typed text and synthetic
speech, so real singing may need a recalibration (`make evaluate-world ARGS=--calibrate` writes the
zero-false-match fit; wrong-song scores grow with the index size).
