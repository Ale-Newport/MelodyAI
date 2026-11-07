# Training the neural melody encoder

```bash
make train                       # configs/train_default.yaml (≈20–30 min on an Apple M-series GPU via MPS)
make train-smoke                 # seconds, CPU; checks the pipeline only
python -m melodyai.cli train --config configs/train_default.yaml --resume   # continue from last.pt
```

The same job can be started from the web UI (Research Lab → Training), which runs it in a
background thread with progress and cancellation.

## Data

| Split | Source | Seeds | Size (default) |
|---|---|---|---|
| train | procedurally generated original compositions | 100000+ | 3000 |
| val | same generator | 200000+ | 150 |
| test (reserved) | same generator | 300000+ | used as out-of-catalog material in the benchmark |
| demo catalog | same generator | 1000+ | 48 (benchmark in-catalog songs) |

**Leakage control (Test I).** Splits are by composition (disjoint seed ranges). On top of that,
`training.data.remove_leaky` drops any training composition that shares an *exact* interval 8-gram
with the validation set, the demo catalog or the reserved test compositions; the counts are written
to `split_report.json` in the run directory. `tests/unit/test_training.py` verifies this.

**Positives.** For a training composition, an excerpt of 6–25 notes starting anywhere is perturbed
with `melodyai.demo.queries.perturb` (key ±6, tempo 0.6–1.6×, duration jitter, 30-cent detuning,
6% wrong notes, 8% dropped and 5% split notes), then passed through a contour-level *pitch-tracker
simulator* (inter-note glides, delayed vibrato, slow drift, frame noise, occasional octave blips,
unvoiced gaps removed, the same octave folding and 100→20 Hz block median as the real pipeline).
The positive reference is the **index-grid window** with the highest overlap with the excerpt —
exactly what the index contains at inference time.

A *pipeline pool* (default 600 compositions) is produced by synthesising hum / whistle / voice /
noisy audio and running the real pYIN pipeline; 25% of positives are drawn from it. It is cached in
`experiments/cache/` and built with a process pool.

**Negatives.** In-batch: one composition per batch slot, so all other references in the batch are
different songs (no false negatives from the same song).

## Objective and optimisation

* `infonce` (default): symmetric cross-entropy over the cosine-similarity matrix / temperature 0.07.
* `triplet`: margin 0.2 with the hardest in-batch negative.
* AdamW (lr 2e-3, weight decay 1e-4), 200 warm-up steps then cosine decay, gradient clipping 1.0,
  batch 384, 100 steps per epoch, up to 30 epochs, early stopping after 6 epochs without improvement
  of validation MRR.
* Seeds: Python, NumPy and torch seeded from the config; the sampler is seeded per epoch on resume.

## Validation metric

All windows of the 150 validation compositions (~43,000 windows) form the index; 300 simulated
queries (2 per composition) are embedded; song-level top-1 / top-5 / MRR use max-similarity
aggregation. The best epoch by MRR is saved as `best.pt`.

## Outputs

```
experiments/training/<name>-<config hash>/
  config.yaml  split_report.json  history.json  result.json
  best.pt  last.pt  tensorboard/
models/<name>.pt            # exported best checkpoint (loaded by the API at start-up)
```

`tensorboard --logdir experiments/training` shows loss, learning rate and validation curves.
`result.json` records parameter count, device, wall time, Python/torch versions and platform.

## Using a checkpoint

* The API loads `models/melody_encoder.pt` at start-up when present
  (`MELODYAI_LOAD_NEURAL_MODEL=false` to disable). Any `models/*.pt` or
  `experiments/training/*/best.pt` can be activated from Research Lab → Models or
  `POST /api/models/neural/activate`.
* This registers `neural`, `neural_hnsw`, `neural_rerank` and `ensemble_neural`.
* The benchmark (`make evaluate`) evaluates the checkpoint configured in `configs/eval_default.yaml`.

## Honest caveats

* All training data is synthetic. The simulator approximates, but does not reproduce, real humming
  (breathiness, scooping habits, rhythm drift, register changes, consonants).
* Validation compositions come from the same generator as the training ones, so validation scores
  measure generalisation to *new compositions of the same style*, not to real music.
* No real-query dataset was used. MTG-QBH or MIR-QBSH (see [DATASETS.md](DATASETS.md)) would be the
  natural next step for fine-tuning and evaluation, under their licences.
