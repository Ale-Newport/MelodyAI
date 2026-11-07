"""Scale benchmark: how accuracy, latency and memory change as the catalog grows.

    python -m melodyai.cli evaluate-scale --config configs/eval_scale.yaml

Melody (hum/whistle) part
    The 48 demo songs are hidden among up to N procedurally generated *distractor* melodies
    (generator seeds 500000+, disjoint from the demo, training and test ranges). The same synthetic
    hummed/whistled queries as the main benchmark are matched with the two-stage search
    (neural IVF-PQ candidates -> DTW + note-alignment verification). Songs are added incrementally
    and measured at each checkpoint size. Baselines: neural ANN alone, and exhaustive note
    alignment (only up to ``exhaustive_max`` songs because its cost grows linearly).

Recording (fingerprint) part
    Polyphonic synthetic *recordings* (melody + pad + bass) of the 48 demo songs are hidden among
    distractor recordings. Queries are 8-second clips with TikTok-style edits (sped up / slowed,
    pitch-shifted, time-stretched, noise, phone-speaker band-limiting, reverb, talking-over overlay),
    plus clips of non-indexed recordings and hummed queries that must *not* match.

Everything is synthetic; see docs/SCALING.md for what this does and does not say about real music.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from melodyai.config import ROOT_DIR, experiments_dir
from melodyai.evaluation.metrics import latency_stats, retrieval_metrics, roc_auc

DISTRACTOR_SEED = 500_000
RECALL_AT = (10, 50, 200, 500)   # stage-1 candidate recall curve (songs)
RECORDING_DISTRACTOR_SEED = 600_000


# ----------------------------------------------------------------------------- workers
def _distractor_chunk(args):
    """Worker: generate distractor melodies and their stage-1 window vectors."""
    start, count = args
    from melodyai.demo.generator import generate_melody
    from melodyai.matching.scalable import ann_window_vectors

    out = []
    for k in range(start, start + count):
        m = generate_melody(DISTRACTOR_SEED + k)
        out.append((m.notes.onsets.astype(np.float32), m.notes.durations.astype(np.float32),
                    m.notes.pitches.astype(np.float32), ann_window_vectors(m.notes).astype(np.float32)))
    return out


def _recording_chunk(args):
    """Worker: render distractor recordings and fingerprint them."""
    start, count, seed = args
    import warnings

    warnings.filterwarnings("ignore")
    from melodyai.demo.generator import generate_melody
    from melodyai.evaluation.datasets import render_polyphonic
    from melodyai.fingerprint.index import fingerprint_audio

    out = []
    for k in range(start, start + count):
        mix, _ = render_polyphonic(generate_melody(seed + k))
        p, h = fingerprint_audio(mix, 16000)
        out.append((h, p, len(mix) / 16000))
    return out


# ----------------------------------------------------------------------------- melody
def melody_scale(cfg: dict, progress) -> dict:
    from melodyai.evaluation.runner import catalog_and_ooc, load_query_set
    from melodyai.features.extract import reference_from_notes
    from melodyai.features.melody import NoteSequence
    from melodyai.indexing.ivfpq import IVFPQ
    from melodyai.indexing.melody_index import MelodyIndex
    from melodyai.matching.algorithms import ALGORITHMS
    from melodyai.matching.neural import NeuralEmbedder
    from melodyai.matching.scalable import ScalableMelodySearch, ann_window_vectors

    mc = cfg["melody"]
    base_cfg = yaml.safe_load((ROOT_DIR / mc.get("query_config", "configs/eval_default.yaml")).read_text())
    melodies, ooc = catalog_and_ooc(base_cfg)
    queries, qhash = load_query_set(base_cfg, melodies, ooc, lambda p, m: progress(0.05 * p, f"queries {m}"))
    sizes = sorted(mc["sizes"])
    n_max = sizes[-1]
    emb = NeuralEmbedder(ROOT_DIR / mc.get("checkpoint", "models/melody_encoder.pt"))

    song_ids: list[str] = []
    notes: dict[str, NoteSequence] = {}
    window_song: list[np.ndarray] = []

    def add_song(sid, ns, vecs, ann, next_wid):
        si = len(song_ids)
        song_ids.append(sid)
        notes[sid] = ns
        window_song.append(np.full(len(vecs), si, np.int64))
        E = emb.embed(vecs, is_query=False)
        ann.add(E, np.arange(next_wid, next_wid + len(vecs)))
        return next_wid + len(vecs)

    # Demo songs first, then distractors generated in parallel chunks.
    demo = [(f"demo-{i:02d}", m.notes, ann_window_vectors(m.notes).astype(np.float32)) for i, m in enumerate(melodies)]
    n_dis = n_max - len(demo)
    workers = int(cfg.get("workers", 8))
    chunk = 500
    t_gen = time.perf_counter()
    pool = ProcessPoolExecutor(workers)
    futures = [pool.submit(_distractor_chunk, (s, min(chunk, n_dis - s))) for s in range(0, n_dis, chunk)]

    def distractors():
        k = 0
        for f in futures:
            for on, du, pi, vecs in f.result():
                yield f"x{k:06d}", NoteSequence(on, du, pi), vecs
                k += 1

    stream = iter(demo)
    gen = distractors()
    # Train the coarse quantiser and PQ codebooks on the first ~2000 songs' window embeddings.
    train_songs = []
    while len(train_songs) < min(2000, n_max):
        try:
            train_songs.append(next(stream))
        except StopIteration:
            train_songs.append(next(gen))
    t0 = time.perf_counter()
    T = emb.embed(np.concatenate([v for _, _, v in train_songs]), is_query=False)
    ann = IVFPQ(T.shape[1], nlist=int(mc.get("nlist", 2048)), m=int(mc.get("m", 16)), nprobe=int(mc.get("nprobe", 32)))
    ann.train(T[np.random.default_rng(0).permutation(len(T))[:300_000]])
    train_s = time.perf_counter() - t0

    def loader(sid):
        return reference_from_notes(sid, notes[sid], audio=np.zeros(1))

    results = {"sizes": {}, "train_s": round(train_s, 1), "query_hash": qhash, "nlist": ann.nlist, "m": ann.m,
               "nprobe": ann.nprobe, "n_candidates": mc.get("n_candidates", 50)}
    next_wid = 0
    add_time = 0.0
    pending = list(train_songs)
    in_cat = [q for q in queries if q["target"] is not None]
    neg = [q for q in queries if q["target"] is None]
    for size in sizes:
        t0 = time.perf_counter()
        while len(song_ids) < size:
            sid, ns, vecs = pending.pop(0) if pending else next(gen)
            next_wid = add_song(sid, ns, vecs, ann, next_wid)
        ann.consolidate()
        add_time += time.perf_counter() - t0
        ws = np.concatenate(window_song)
        search = ScalableMelodySearch(emb, ann, ws, song_ids, loader, n_candidates=int(mc.get("n_candidates", 50)),
                                      nprobe=int(mc.get("nprobe", 32)))
        rows = []
        for i, q in enumerate(in_cat + neg):
            qf = q["query"]
            target = f"demo-{q['target']:02d}" if q["target"] is not None else None
            if not qf.informative:
                rows.append({"target": target, "rank2": None, "rank1": None, "in_cands": False, "top": 0.0, "wide_rank": None,
                             "lat1": 0.0, "lat2": 0.0, "condition": q["condition"]})
                continue
            t1 = time.perf_counter()
            wide = search.candidates(qf, n=max(RECALL_AT))
            lat1 = time.perf_counter() - t1
            res = search.search(qf, None, 10, candidates=wide)
            ids2 = [c.song_id for c in res]
            ids_wide = [s for s, _ in wide]
            ids1 = ids_wide[: search.n_candidates]
            rows.append({"target": target, "condition": q["condition"],
                         "rank2": ids2.index(target) + 1 if target in ids2 else None,
                         "rank1": ids1.index(target) + 1 if target in ids1 else None,
                         "in_cands": target in ids1, "top": res[0].score if res else 0.0,
                         "wide_rank": ids_wide.index(target) + 1 if target in ids_wide else None,
                         "lat1": lat1, "lat2": search.last_timings.get("stage2_s", 0.0)})
            if i % 40 == 0:
                progress(0.05 + 0.6 * (sizes.index(size) + i / len(in_cat + neg)) / len(sizes), f"melody N={size}")
        R = [r for r in rows if r["target"]]
        neg_scores = np.array([r["top"] for r in rows if not r["target"]])
        correct = np.array([r["top"] for r in R if r["rank2"] == 1])
        entry = {
            "two_stage": retrieval_metrics([r["rank2"] for r in R]),
            "neural_ann_only": retrieval_metrics([r["rank1"] for r in R]),
            "candidate_recall": float(np.mean([r["in_cands"] for r in R])),
            "candidate_recall_at": {str(k): float(np.mean([r["wide_rank"] is not None and r["wide_rank"] <= k for r in R]))
                                    for k in RECALL_AT},
            "latency_stage1": latency_stats([r["lat1"] for r in R if r["lat1"]]),
            "latency_stage2": latency_stats([r["lat2"] for r in R if r["lat2"]]),
            "auc_correct_vs_ooc": roc_auc(correct, neg_scores),
            "ann_vectors": len(ann), "ann_index_mb": round(ann.nbytes / 1e6, 2),
            "bytes_per_window": round(ann.nbytes / max(1, len(ann)), 1),
            "windows_per_song": round(len(ann) / size, 1),
            "cumulative_add_s": round(add_time, 1),
            "by_condition": {c: retrieval_metrics([r["rank2"] for r in R if r["condition"] == c])
                             for c in sorted({r["condition"] for r in R})},
        }
        # Exhaustive baseline (linear cost) on a subset of queries for the smaller sizes.
        if size <= int(mc.get("exhaustive_max", 10000)):
            sub = MelodyIndex()
            for sid in song_ids:
                sub.refs[sid] = loader(sid)
            alg = ALGORITHMS["note_align"]
            nq = int(mc.get("exhaustive_queries", 96))
            rng = np.random.default_rng(0)
            pick = rng.choice(len(in_cat), size=min(nq, len(in_cat)), replace=False)
            ranks, lats = [], []
            for k in pick:
                q = in_cat[k]
                if not q["query"].informative:
                    ranks.append(None)
                    continue
                t1 = time.perf_counter()
                c = alg.search(q["query"], sub, 10)
                lats.append(time.perf_counter() - t1)
                ids = [x.song_id for x in c]
                tgt = f"demo-{q['target']:02d}"
                ranks.append(ids.index(tgt) + 1 if tgt in ids else None)
            entry["exhaustive_note_align"] = {**retrieval_metrics(ranks), "latency": latency_stats(lats),
                                              "n_queries": len(pick)}
        results["sizes"][str(size)] = entry
    pool.shutdown()
    results["generation_and_index_s"] = round(time.perf_counter() - t_gen, 1)
    return results


# ----------------------------------------------------------------------------- fingerprint
def _edit(y: np.ndarray, cond: str, rng: np.random.Generator, other: np.ndarray) -> np.ndarray:
    import librosa
    from scipy import signal

    from melodyai.audio.synth import add_noise, add_reverb

    sr = 16000
    if cond == "none":
        return y
    if cond.startswith("speed_"):          # TikTok "sped up"/"slowed": resampling changes tempo AND pitch
        f = float(cond.split("_")[1])
        z = librosa.resample(y, orig_sr=sr * f, target_sr=sr)
        return add_reverb(z, sr, 1.2, rng, 0.35) if f < 1 else z  # "slowed + reverb"
    if cond == "pitch_up_2":
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    if cond == "stretch_1.15":
        return librosa.effects.time_stretch(y, rate=1.15)
    if cond == "noise_5db":
        return add_noise(y, 5.0, rng)
    if cond == "phone":                    # phone speaker -> phone mic: band-limit + saturation
        sos = signal.butter(4, [300, 3400], btype="band", fs=sr, output="sos")
        z = signal.sosfilt(sos, y)
        z = np.tanh(3.0 * z / (np.max(np.abs(z)) + 1e-9))
        return add_noise(z.astype(np.float32), 20.0, rng)
    if cond == "reverb":
        return add_reverb(y, sr, 0.8, rng, 0.5)
    if cond == "overlay":                  # someone singing/talking over the music
        o = other[: len(y)]
        o = np.pad(o, (0, len(y) - len(o)))
        return (y + 0.7 * o * (np.std(y) / (np.std(o) + 1e-9))).astype(np.float32)
    raise ValueError(cond)


def fingerprint_scale(cfg: dict, progress) -> dict:
    from melodyai.audio.synth import render
    from melodyai.demo.generator import generate_catalog, generate_melody
    from melodyai.demo.queries import CONDITIONS, perturb, synth_query
    from melodyai.evaluation.datasets import render_polyphonic
    from melodyai.fingerprint.index import FingerprintIndex, fingerprint_audio
    from melodyai.fusion.fusion import DEFAULT_THRESHOLDS

    fc = cfg["fingerprint"]
    sr = 16000
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    demo = generate_catalog(48, 1000)
    mixes = [render_polyphonic(m)[0] for m in demo]
    # Clips that must NOT match: other recordings (not indexed) and hummed queries of demo songs.
    ooc_mix = [render_polyphonic(generate_melody(700_000 + k))[0] for k in range(int(fc.get("ooc", 48)))]
    others = [render(generate_melody(710_000 + k).notes, "voice", sr=sr, seed=k) for k in range(8)]
    hums = []
    for m in demo:
        pn, _ = perturb(m.notes.slice(5, 20), CONDITIONS["hum_clean"], rng)
        hums.append(synth_query(pn, CONDITIONS["hum_clean"], rng))
    clip_s = float(fc.get("clip_seconds", 8.0))

    def clip(y):
        a = int(rng.uniform(0, max(0.1, len(y) / sr - clip_s - 0.5)) * sr)
        return y[a:a + int(clip_s * sr)].astype(np.float32)

    queries = []
    for cond in fc["conditions"]:
        for i, y in enumerate(mixes):
            queries.append((cond, f"demo-{i:02d}", _edit(clip(y), cond, rng, others[i % len(others)])))
    queries += [("out_of_catalog_recording", None, clip(y)) for y in ooc_mix]
    queries += [("hummed_query", None, h.astype(np.float32)) for h in hums]
    progress(0.7, "fingerprint queries ready")

    sizes = sorted(fc["sizes"])
    tmp = tempfile.mkdtemp(prefix="fpscale_")
    idx = FingerprintIndex(tmp, flush_postings=3_000_000)
    t_fp = time.perf_counter()
    audio_s = 0.0
    for i, y in enumerate(mixes):
        p, h = fingerprint_audio(y, sr)
        idx.add(f"demo-{i:02d}", h, len(y) / sr, p)
        audio_s += len(y) / sr
    n_dis = sizes[-1] - len(mixes)
    workers = int(cfg.get("workers", 8))
    chunk = 50
    pool = ProcessPoolExecutor(workers)
    futures = [pool.submit(_recording_chunk, (s, min(chunk, n_dis - s), RECORDING_DISTRACTOR_SEED))
               for s in range(0, n_dis, chunk)]
    dis = (item for f in futures for item in f.result())
    accept = DEFAULT_THRESHOLDS["fingerprint"]["accept"]
    results = {"sizes": {}, "clip_seconds": clip_s, "accept_threshold": accept}
    k = 0
    for size in sizes:
        while len(idx) < size:
            h, p, dur = next(dis)
            idx.add(f"x{k:06d}", h, dur, p)
            audio_s += dur
            k += 1
        idx.flush()
        rows = []
        for cond, target, y in queries:
            t0 = time.perf_counter()
            m = idx.search(y, sr, top_k=5)
            lat = time.perf_counter() - t0
            ids = [x.song_id for x in m]
            rows.append({"cond": cond, "target": target, "rank": ids.index(target) + 1 if target in ids else None,
                         "top": m[0].score if m else 0.0, "lat": lat,
                         "speed": m[0].speed if m else None, "pitch": m[0].pitch_semitones if m else None})
        progress(0.7 + 0.3 * (sizes.index(size) + 1) / len(sizes), f"fingerprint N={size}")
        R = [r for r in rows if r["target"]]
        neg = [r for r in rows if not r["target"]]
        st = idx.stats()
        results["sizes"][str(size)] = {
            "overall": retrieval_metrics([r["rank"] for r in R]),
            "by_condition": {c: {**retrieval_metrics([r["rank"] for r in R if r["cond"] == c]),
                                 "accepted_correct": float(np.mean([r["rank"] == 1 and r["top"] >= accept
                                                                    for r in R if r["cond"] == c]))}
                             for c in fc["conditions"]},
            "false_accept": {c: float(np.mean([r["top"] >= accept for r in neg if r["cond"] == c]))
                             for c in ("out_of_catalog_recording", "hummed_query")},
            "auc_correct_vs_negatives": roc_auc(np.array([r["top"] for r in R if r["rank"] == 1]),
                                                np.array([r["top"] for r in neg])),
            "latency": latency_stats([r["lat"] for r in rows]),
            "index": {**st, "audio_hours": round(audio_s / 3600, 2),
                      "mb_per_audio_hour": round(st["bytes"] / 1e6 / (audio_s / 3600), 2),
                      "postings_per_second": round(st["postings"] / audio_s, 1)},
        }
    results["build_and_eval_s"] = round(time.perf_counter() - t_fp, 1)
    pool.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)
    return results


def run_scale(config_path, progress=None) -> dict:
    from melodyai.evaluation.runner import environment

    cfg = yaml.safe_load(Path(config_path).read_text())
    progress = progress or (lambda p, m: None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{cfg.get('name', 'scale')}"
    run_dir = experiments_dir() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    t0 = time.perf_counter()
    out = {"run_id": run_id, "kind": "scale", "created_at": datetime.now(timezone.utc).isoformat(),
           "env": environment(), "dataset": {"type": "synthetic"}}
    if cfg.get("fingerprint", {}).get("enabled", True):
        out["fingerprint"] = fingerprint_scale(cfg, progress)
    if cfg.get("melody", {}).get("enabled", True):
        out["melody"] = melody_scale(cfg, progress)
    out["elapsed_s"] = round(time.perf_counter() - t0, 1)
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    return {"run_dir": str(run_dir), "run_id": run_id, "metrics": out}
