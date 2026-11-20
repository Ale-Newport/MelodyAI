# Changelog

All notable changes to MelodyAI are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-11-20

Bug-fix and hardening release. Recognition algorithms, the demo catalog and the benchmark results are unchanged.

### Fixed

- **Microphone recording** (#14, PR #24 by [swpswpswp](https://github.com/swpswpswp)): Stop or Cancel while the
  browser asks for microphone permission no longer sends an empty recording, a stream that arrives after cancelling
  or leaving the page is stopped, duplicate starts are ignored, and a saved microphone that is unplugged falls back
  to the default one.
- **Saved neural matcher** (#15, PR #23 by [swpswpswp](https://github.com/swpswpswp)): a saved neural melody matcher
  is restored after a restart; while a checkpoint loads or after it fails to load, recognition and the Melody
  Explorer use the classical ensemble instead of failing.
- **Library** (#11): song audio stops when you leave the page and no longer overlaps other previews.
- **Add Song** (#12): the dialog starts empty each time (a previous lyrics file was attached to the next song), and
  MIDI files are always sent as `midi`.
- **Microphone selection** (#13): a newly chosen microphone takes effect immediately instead of after a reload.
- **Pitch extractors** (#17): an unavailable extractor is refused in Settings and in requests (422 instead of 500),
  and a saved one that is no longer installed falls back to pYIN.
- **Worker pools** (benchmark, training data, Lakh index, ingest) spawn their processes instead of forking a process
  that already runs numba/OpenMP threads, which could crash a worker on Linux.
- **Online song search** never sends a request after its deadline, even when a connection slot is granted late on
  a busy machine.

### Security

- Bulk imports are limited in total size (`MELODYAI_MAX_BULK_MB`, default 600 MB), and uploads are decoded only
  inside the recognition slots, so concurrent requests cannot exhaust memory (#16).
- Import errors no longer echo the text of a library's own exceptions to the client (CodeQL
  `py/stack-trace-exposure`; no code-scanning alerts remain open).

### Dependencies

- Frontend: TypeScript 7, Recharts 3 (legend order preserved), jsdom 31, and minor updates of Radix UI,
  TanStack Query, Testing Library, autoprefixer and the fonts. `npm audit`: 0 vulnerabilities.
- Docker image: Python 3.12 runtime, Node 26 for building the web app.
- Known and unchanged: the optional `[pitch]` extra (Basic Pitch) keeps `setuptools<81` and, on Linux, pulls
  TensorFlow/Keras packages with published advisories; the dependency audit reports them. The core install and the
  `[ml,asr]` extras are not affected.

## [0.1.0] - 2026-11-19

First public release.

### Added

- **Query by humming, whistling and singing**: pYIN pitch tracking (CREPE and Basic Pitch optional), note
  segmentation, window and subsequence DTW, chroma DTW, note-interval Smith-Waterman alignment, an ensemble, and a
  contrastively trained neural melody encoder with exact or IVF-PQ vector search.
- **Lyrics recognition** from typed, spoken or sung words: SQLite FTS5 plus fuzzy phonetic alignment, local
  speech recognition with faster-whisper.
- **Real songs by their words (open world)**: a local lyrics index of popular songs filled from LRCLIB and Deezer
  (`make world-sync`), online search that sends only words, every candidate verified against its real lyrics.
- **Humming real songs**: an optional melody index of ~25,000 songs from the Lakh MIDI dataset (`make lakh`) and a
  vocal-track classifier for multi-track MIDI.
- **Clip recognition**: pitch- and speed-invariant audio fingerprints for sped-up, slowed, transposed or
  phone-recorded clips; optional opt-in ACRCloud provider for commercial catalogs.
- **Calibrated fusion** that answers "no reliable match" instead of guessing.
- **Web app** (React + TypeScript): recognition with explanations (matched passage, alignment, DTW cost matrix,
  pitch track), Melody Explorer, Library, Research Lab and Settings, in English and Spanish.
- **Reproducible benchmarks** (`make evaluate`, `make evaluate-scale`, `make evaluate-world`) and a training lab.
- **Static live demo** on GitHub Pages, an original CC0 48-song demo catalog, Docker image and `make` workflow.

### Security

- The API refuses cross-site requests and requests whose `Host` header is not a local name (DNS rebinding);
  `MELODYAI_ALLOWED_HOSTS` configures it. `docker compose` publishes the port on `127.0.0.1` only.
- Audio duration and sample rate are checked from the file header before decoding, `ffmpeg` decoding is capped and
  MIDI references longer than 15 minutes are refused, so small crafted files cannot exhaust memory.
- Malformed lyrics or manifest JSON is rejected with a clear error instead of a server error.
- The Docker image runs as an unprivileged user and has a health check.
- Dependencies updated to versions without known advisories (React Router 7, Vite 6.4, Vitest 4, PostCSS 8.5);
  the core install no longer pins an old setuptools.

### Infrastructure

- CI on Linux and macOS (Python 3.11 and 3.12), frontend checks, Docker build and smoke test, CodeQL, dependency
  review, scheduled `pip-audit` / `npm audit`, Dependabot, SHA-pinned actions with read-only tokens, and a tag-driven
  release workflow.

[Unreleased]: https://github.com/Ale-Newport/MelodyAI/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Ale-Newport/MelodyAI/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Ale-Newport/MelodyAI/releases/tag/v0.1.0
