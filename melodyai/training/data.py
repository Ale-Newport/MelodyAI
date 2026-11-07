"""Training data for the neural melody encoder.

Sources
-------
* **Compositions**: procedurally generated ORIGINAL melodies (``melodyai.demo.generator``)
  from seed ranges that are disjoint from the demo catalog and from each other:

      train: 100000 + k     val: 200000 + k     test: 300000 + k     demo catalog: 1000 + k

  Splits are therefore *by composition*. :func:`check_leakage` additionally verifies that
  no exact interval 8-gram is shared between a training composition and any evaluation
  composition (a stricter criterion than distinct seeds).
* **Positives** for a reference window are simulated renditions of the same passage:
  key shift, tempo change, duration jitter, detuning, wrong / missing / repeated notes
  (symbolic perturbations, ``melodyai.demo.queries.perturb``) followed by a *pitch-tracker
  simulator* (glides, vibrato, drift, unvoiced gaps, octave blips, same downsampling
  as the real pipeline). Optionally a pool of fully realistic positives is produced by
  synthesising audio and running the real pYIN pipeline (``build_pipeline_pool``).
* **Negatives** are the other compositions in the batch (one window per composition per
  batch, so no same-song false negatives).

All sampling is driven by explicit seeds for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from melodyai.demo.generator import _window_signatures, generate_melody
from melodyai.demo.queries import QuerySpec, perturb
from melodyai.features.melody import NoteSequence, fold_octave_jumps, notes_to_contour, resample_contour
from melodyai.features.windows import DEFAULT_LENGTHS, hop_for

SPLIT_OFFSETS = {"train": 100_000, "val": 200_000, "test": 300_000, "demo": 1000}
WINDOW_NOTES = (6, 25)

AUG_SPEC = QuerySpec("train", key_shift=(-6, 6), tempo=(0.6, 1.6), detune_cents=30.0, wrong_note_p=0.06,
                     drop_p=0.08, repeat_p=0.05, duration_jitter=0.12, vocal_range=None)


def compositions(split: str, n: int) -> list[NoteSequence]:
    base = SPLIT_OFFSETS[split]
    out = []
    k = 0
    while len(out) < n:
        m = generate_melody(base + k)
        k += 1
        if len(m.notes) >= 30:
            out.append(m.notes)
    return out


def check_leakage(train: list[NoteSequence], others: list[NoteSequence]) -> dict:
    """Count training compositions sharing an exact interval 8-gram with evaluation material."""
    def grams(n):
        s = _window_signatures(n)
        return {tuple(np.round(r).astype(int)) for r in s}

    other = set().union(*(grams(n) for n in others)) if others else set()
    leaked = [i for i, n in enumerate(train) if grams(n) & other]
    return {"n_train": len(train), "n_eval": len(others), "shared_8gram_compositions": len(leaked), "indices": leaked}


def remove_leaky(train: list[NoteSequence], others: list[NoteSequence]) -> tuple[list[NoteSequence], dict]:
    report = check_leakage(train, others)
    bad = set(report["indices"])
    return [n for i, n in enumerate(train) if i not in bad], report


def simulate_tracker(notes: NoteSequence, rng: np.random.Generator, rate: float = 20.0) -> np.ndarray:
    """Approximate the pYIN -> voiced-contour pipeline at contour level (fast)."""
    hi = 100.0
    frames = []
    prev = None
    for p, d in zip(notes.pitches, notes.ioi()):
        n = max(2, int(round(d * hi)))
        seg = np.full(n, p, dtype=np.float64)
        if prev is not None:
            g = min(n, int(rng.integers(2, 8)))
            seg[:g] = prev + (p - prev) * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, g)))
        if n > 20:
            a = rng.uniform(0, 0.35)
            t = np.arange(n) / hi
            seg += a * np.clip((t - 0.15) / 0.2, 0, 1) * np.sin(2 * np.pi * rng.uniform(4.5, 6.5) * t)
        if rng.random() < 0.03:  # octave blip
            k = int(rng.integers(0, max(1, n - 6)))
            seg[k:k + int(rng.integers(2, 6))] += float(rng.choice([-12, 12]))
        if rng.random() < 0.3 and n > 10:  # breath / unvoiced gap removed by the pipeline
            seg = seg[: n - int(rng.integers(2, min(10, n // 2)))]
        frames.append(seg)
        prev = p
    c = np.concatenate(frames)
    c += np.cumsum(rng.normal(0, rng.uniform(0.005, 0.03), len(c)))  # slow drift
    c += rng.normal(0, rng.uniform(0.02, 0.12), len(c))
    c = fold_octave_jumps(c)
    step = int(round(hi / rate))
    m = len(c) // step
    return np.median(c[: m * step].reshape(m, step), axis=1) if m else c


def grid_window(n_notes: int, s: int, e: int, lengths=DEFAULT_LENGTHS) -> tuple[int, int]:
    """Index window (same grid as ``build_windows``) with the highest overlap (IoU) with
    the excerpt [s, e). At inference a hummed excerpt starts and ends anywhere, but only
    grid windows exist in the index, so training positives must be grid windows too."""
    best, best_iou = (0, min(n_notes, lengths[0])), -1.0
    for G in lengths:
        G = min(G, n_notes)
        starts = np.arange(0, max(1, n_notes - G + 1), hop_for(G))
        inter = np.maximum(0, np.minimum(e, starts + G) - np.maximum(s, starts))
        union = np.maximum(e, starts + G) - np.minimum(s, starts)
        iou = inter / union
        k = int(np.argmax(iou))
        if iou[k] > best_iou:
            best, best_iou = (int(starts[k]), int(starts[k] + G)), float(iou[k])
        if G >= n_notes:
            break
    return best


def window_vector(notes: NoteSequence, n_points: int) -> np.ndarray:
    c = notes_to_contour(notes)
    return resample_contour(c - np.median(c), n_points)


def contour_vector(c: np.ndarray, n_points: int) -> np.ndarray:
    return resample_contour(c - np.median(c), n_points)


@dataclass
class PairBatch:
    queries: np.ndarray   # (B, T)
    refs: np.ndarray      # (B, T)
    comp_ids: np.ndarray  # (B,)


class PairSampler:
    """Samples (simulated query, clean reference window) pairs, one per composition."""

    def __init__(self, comps: list[NoteSequence], n_points: int = 96, seed: int = 0,
                 pipeline_pool: dict | None = None, pipeline_fraction: float = 0.0) -> None:
        self.comps = comps
        self.n_points = n_points
        self.rng = np.random.default_rng(seed)
        self.pool = pipeline_pool
        self.pipeline_fraction = pipeline_fraction if pipeline_pool else 0.0

    def sample_excerpt(self, notes: NoteSequence) -> tuple[int, int]:
        L = int(self.rng.integers(WINDOW_NOTES[0], WINDOW_NOTES[1] + 1))
        L = min(L, len(notes))
        s = int(self.rng.integers(0, len(notes) - L + 1))
        return s, s + L

    def sample_window(self, notes: NoteSequence) -> NoteSequence:
        s, e = self.sample_excerpt(notes)
        return notes.slice(s, e)

    def pair(self, ci: int) -> tuple[np.ndarray, np.ndarray]:
        if self.pipeline_fraction and self.rng.random() < self.pipeline_fraction:
            entries = self.pool["by_comp"].get(ci)
            if entries:
                k = entries[int(self.rng.integers(len(entries)))]
                return self.pool["queries"][k], self.pool["refs"][k]
        notes = self.comps[ci]
        s, e = self.sample_excerpt(notes)
        q_notes, _ = perturb(notes.slice(s, e), AUG_SPEC, self.rng)
        gs, ge = grid_window(len(notes), s, e)
        return (contour_vector(simulate_tracker(q_notes, self.rng), self.n_points),
                window_vector(notes.slice(gs, ge), self.n_points))

    def batch(self, size: int) -> PairBatch:
        idx = self.rng.choice(len(self.comps), size=min(size, len(self.comps)), replace=False)
        qs, rs = zip(*(self.pair(int(i)) for i in idx))
        return PairBatch(np.stack(qs).astype(np.float32), np.stack(rs).astype(np.float32), idx)


def _pool_job(args):
    notes, cond, seed, n_points = args
    import warnings

    from melodyai.audio.io import Audio
    from melodyai.demo.queries import CONDITIONS, synth_query
    from melodyai.features.extract import analyze_query

    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    spec = CONDITIONS[cond]
    pn, _ = perturb(notes, spec, rng)
    q = analyze_query(Audio(synth_query(pn, spec, rng), 16000), "whistle" if spec.timbre == "whistle" else "hum",
                      with_chroma=False)
    return contour_vector(q.contour, n_points) if len(q.contour) >= 10 else None


def build_pipeline_pool(comps: list[NoteSequence], per_comp: int, n_points: int, seed: int,
                        max_comps: int | None = None, progress=None, workers: int | None = None) -> dict:
    """Realistic positives: synthesise hum/whistle/voice audio and run the real pYIN pipeline."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    rng = np.random.default_rng(seed)
    sampler = PairSampler(comps, n_points, seed)
    n = min(len(comps), max_comps or len(comps))
    jobs, meta = [], []
    for ci in range(n):
        for _ in range(per_comp):
            s, e = sampler.sample_excerpt(comps[ci])
            gs, ge = grid_window(len(comps[ci]), s, e)
            cond = str(rng.choice(["hum_clean", "hum_detuned", "whistle", "sing_vowel", "hum_noisy"]))
            jobs.append((comps[ci].slice(s, e), cond, int(rng.integers(1 << 31)), n_points))
            meta.append((ci, window_vector(comps[ci].slice(gs, ge), n_points)))
    qs, rs, by = [], [], {}
    with ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2) - 2)) as ex:
        for k, qv in enumerate(ex.map(_pool_job, jobs, chunksize=4)):
            if qv is not None:
                ci, rv = meta[k]
                by.setdefault(ci, []).append(len(qs))
                qs.append(qv)
                rs.append(rv)
            if progress:
                progress((k + 1) / len(jobs), f"pipeline pool {k + 1}/{len(jobs)}")
    return {"queries": np.asarray(qs, np.float32), "refs": np.asarray(rs, np.float32), "by_comp": by}
