# Limitations and known issues

MelodyAI is a research prototype. This page lists what is verified, what is experimental and
what is not done.

## Verified (tests + benchmark, in this repository's environment)

* Classical melody matching (window DTW, subsequence DTW, chroma DTW, note alignment, ensemble)
  on the synthetic demo catalog, including transposed, tempo-changed, detuned, mid-song, whistled,
  short and noisy queries (`docs/RESULTS.md`).
* Rejection of noise / silence / held tones / unknown melodies with thresholds tuned on a dev split.
* Lyrics search on typed text with misspellings and misremembered words (English and Spanish).
* Whisper `base` transcription of *synthetic speech* (macOS `say`) and its hallucination guard on hums.
* Catalog import (MIDI, monophonic audio, lyrics TXT/JSON, CSV/JSON manifests), incremental
  indexing, deletion, reindexing, feature-version staleness.
* REST API (FastAPI) and the web UI flows (recognition by upload, explorer, library, lab, settings),
  exercised in Chromium-based browser automation on macOS. Microphone capture code paths were
  **not** exercised by automation (no physical microphone input in the automated browser).
* Neural encoder training on Apple Silicon (MPS) and inference from the exported checkpoint.

## Experimental

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
  real-time factor well below 1 on an M-series CPU; larger models are slower.
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
  used in the benchmark.

## Calibration limits

Thresholds in `configs/calibration.json` were tuned for ≤5% false accepts on *synthetic*
out-of-catalog hums. Real recordings have different score distributions; recalibrate with
labelled real queries before trusting the *match* / *no match* boundary, and treat the displayed
score as a relative similarity.
