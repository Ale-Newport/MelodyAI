# Contributing to MelodyAI

Thanks for helping! MelodyAI values **measured** improvements over impressive-looking ones: a change that makes a
number better should come with the benchmark run that shows it.

New here? Look for issues labelled
[`good first issue`](https://github.com/Ale-Newport/MelodyAI/issues?q=is%3Aopen+label%3A%22good+first+issue%22) or
[`help wanted`](https://github.com/Ale-Newport/MelodyAI/issues?q=is%3Aopen+label%3A%22help+wanted%22), and ask
questions in [Discussions](https://github.com/Ale-Newport/MelodyAI/discussions). Please read the
[Code of Conduct](CODE_OF_CONDUCT.md) first.

## How contributions are merged

* All branches of this repository are protected. Contributors work in a **fork** and open a **pull request**.
* Every pull request needs an approving review from the code owner ([`.github/CODEOWNERS`](.github/CODEOWNERS)) and
  the required CI checks (lint, Python tests, frontend) to pass. New commits pushed after an approval dismiss it.
* Workflows from first-time and external contributors run only after the maintainer approves them.
* Security issues are never reported in public issues: see [SECURITY.md](SECURITY.md).

## Development setup

Requirements: Python 3.11 (3.10–3.12), Node.js 20+, `make`; optionally [uv](https://github.com/astral-sh/uv) and
`ffmpeg`.

```bash
git clone https://github.com/<your-username>/MelodyAI.git
```

```bash
cd MelodyAI && make setup && make demo-data
```

```bash
make test-fast
```

`make setup-min` skips PyTorch, Whisper and Basic Pitch if you only touch the classical engine, the API or docs.

## Workflow

1. Create a branch in your fork; keep the change focused on one thing.
2. Add tests that check behaviour (not just that objects exist). Python tests live in `tests/`, frontend tests next to
   the code (`*.test.tsx`, Vitest).
3. Run `make lint` (ruff + TypeScript) and `make test` (or `make test-fast` plus `make test-frontend`).
4. If you change an algorithm or its parameters: run `make evaluate`, commit the new run under `experiments/runs/`
   and the regenerated `docs/RESULTS.md`, and explain the trade-off in the PR.
5. If you change feature extraction: bump `FEATURE_VERSION` in `melodyai/config.py` so existing catalogs are
   flagged stale.
6. Add a line to the `Unreleased` section of [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
7. Open the pull request and fill in the template.

## Adding a melody matcher

Implement an object with `name`, `label`, `tau` and
`search(query: QueryFeatures, index: MelodyIndex, top_k) -> list[Candidate]` (sorted by distance,
`score` in [0, 1]) and register it in `melodyai/matching/algorithms.py` (`ALGORITHMS`) or via
`RecognitionEngine.register_algorithm`. It then appears in the API, the UI and, once listed in
`configs/eval_default.yaml`, in the benchmark.

## Adding a pitch extractor

Implement `name`, `label`, `available()` and `extract(y, sr, profile) -> PitchTrack` in
`melodyai/pitch/extractors.py` and add it to `EXTRACTORS`.

## Data and copyright rules

* Never commit commercial audio, MIDI transcriptions of commercial songs or copyrighted lyrics.
* Datasets are downloaded by users under their own licences; adapters must not redistribute data.
* Test fixtures must be generated (see `melodyai/demo`) or clearly licensed (CC0 / own work).
* No recordings of people without their consent.
* Never commit secrets. `.env` is git-ignored; document new settings in `.env.example` with empty values.

## Style

Python: ruff (line length 110), type hints, small pure functions in `melodyai/`, I/O at the edges.
TypeScript: strict mode, components in `src/components`, pages in `src/pages`, no decorative charts:
every visualisation must plot data returned by the API.
Commit messages: an imperative summary line ("Add …", "Fix …") and a body that says why.

## Releases

The maintainer bumps the version in `pyproject.toml`, moves the `Unreleased` notes in `CHANGELOG.md` under the new
version and pushes a `vX.Y.Z` tag; the Release workflow builds the package and publishes the GitHub Release.
