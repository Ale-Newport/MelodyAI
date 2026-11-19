# Changelog

All notable changes to MelodyAI are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Ale-Newport/MelodyAI/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Ale-Newport/MelodyAI/releases/tag/v0.1.0
