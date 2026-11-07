# Contributing

Thanks for helping! MelodyAI values **measured** improvements over impressive-looking ones.

## Workflow

1. `make setup && make demo-data && make test` must pass before you start.
2. Create a branch; keep changes focused.
3. Add tests that check behaviour (not just that objects exist).
4. `make lint` (ruff + TypeScript) and `make test`.
5. If you change an algorithm or its parameters: run `make evaluate`, commit the new run under
   `experiments/runs/` and the regenerated `docs/RESULTS.md`, and explain the trade-off in the PR.
6. If you change feature extraction: bump `FEATURE_VERSION` in `melodyai/config.py` so existing
   catalogs are flagged stale.

## Adding a melody matcher

Implement an object with `name`, `label`, `tau` and
`search(query: QueryFeatures, index: MelodyIndex, top_k) -> list[Candidate]` (sorted by distance,
`score` in [0, 1]) and register it in `melodyai/matching/algorithms.py` (`ALGORITHMS`) or via
`RecognitionEngine.register_algorithm`. It then appears in the API, the UI and — once listed in
`configs/eval_default.yaml` — in the benchmark.

## Adding a pitch extractor

Implement `name`, `label`, `available()` and `extract(y, sr, profile) -> PitchTrack` in
`melodyai/pitch/extractors.py` and add it to `EXTRACTORS`.

## Data and copyright rules

* Never commit commercial audio, MIDI transcriptions of commercial songs or copyrighted lyrics.
* Datasets are downloaded by users under their own licences; adapters must not redistribute data.
* Test fixtures must be generated (see `melodyai/demo`) or clearly licensed (CC0 / own work).
* No recordings of people without their consent.

## Style

Python: ruff (line length 110), type hints, small pure functions in `melodyai/`, I/O at the edges.
TypeScript: strict mode, components in `src/components`, pages in `src/pages`, no decorative charts —
every visualisation must plot data returned by the API.
