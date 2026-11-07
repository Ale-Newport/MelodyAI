"""Reproducible benchmark runner.

    python -m melodyai.cli evaluate --config configs/eval_default.yaml

Everything (catalog, queries, out-of-catalog material) is regenerated from seeds; the
run directory ``experiments/runs/<UTC time>-<name>/`` stores the config, environment and
hardware description, metrics JSON, per-query records and figures. ``melodyai report``
turns the latest run into ``docs/RESULTS.md``.

Thresholds for abstention (and Platt calibration) are fitted on a *dev* half of the
songs (even indices) and reported on the *test* half (odd indices).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from melodyai.config import FEATURE_VERSION, ROOT_DIR, experiments_dir
from melodyai.demo.generator import generate_catalog, generate_melody
from melodyai.evaluation.metrics import (
    ece,
    latency_stats,
    rejection_metrics,
    retrieval_metrics,
    threshold_for_far,
)
from melodyai.features.extract import reference_from_notes
from melodyai.indexing.melody_index import MelodyIndex

STAGES = {"setup": 0.02, "queries": 0.25, "melody": 0.30, "extractors": 0.13, "lyrics": 0.05, "asr": 0.08,
          "fusion": 0.07, "polyphonic": 0.08, "write": 0.02}


class _Progress:
    def __init__(self, cb):
        self.cb = cb or (lambda p, m: None)
        self.done = 0.0

    def stage(self, name):
        w = STAGES[name]
        base = self.done
        self.done += w
        return lambda p, m: self.cb(min(1.0, base + w * p), f"{name}: {m}")


class Cached:
    """Memoises an algorithm's results per query object (so ensembles reuse member results)."""

    def __init__(self, alg):
        self.alg, self.name, self.label = alg, alg.name, getattr(alg, "label", alg.name)
        self.tau = getattr(alg, "tau", 1.0)
        self.cache, self.lat = {}, {}

    def search(self, q, index, top_k=10):
        k = id(q)
        if k not in self.cache:
            t0 = time.perf_counter()
            self.cache[k] = self.alg.search(q, index, max(top_k, 20))
            self.lat[k] = time.perf_counter() - t0
        return self.cache[k][:top_k]


def environment() -> dict:
    import psutil

    info = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "memory_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "feature_version": FEATURE_VERSION}
    for mod in ("numpy", "scipy", "librosa", "numba", "torch", "faiss", "faster_whisper", "rapidfuzz", "sklearn"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "?")
        except ImportError:
            info[f"{mod}_version"] = None
    try:
        from melodyai.config import torch_device

        info["torch_device"] = torch_device()
    except Exception:
        pass
    try:
        import subprocess

        info["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                            cwd=ROOT_DIR).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    return info


def build_reference_index(melodies, prefix="demo-") -> tuple[MelodyIndex, dict]:
    import psutil

    rss0 = psutil.Process().memory_info().rss
    idx = MelodyIndex()
    t = []
    for i, m in enumerate(melodies):
        t0 = time.perf_counter()
        idx.add(reference_from_notes(f"{prefix}{i:02d}", m.notes))
        t.append(time.perf_counter() - t0)
    M = idx.matrix()
    return idx, {"songs": len(melodies), "windows": len(M), "total_s": round(sum(t), 3),
                 "mean_ms_per_song": round(1000 * float(np.mean(t)), 2),
                 "window_matrix_mb": round(M.vectors.nbytes / 1e6, 3),
                 "rss_delta_mb": round((psutil.Process().memory_info().rss - rss0) / 1e6, 1)}


def build_algorithms(cfg: dict, index: MelodyIndex, store_cache: Path | None = None) -> tuple[dict, dict]:
    from melodyai.engine import default_algorithms
    from melodyai.matching.ensemble import MelodyEnsemble

    base = default_algorithms()
    notes = {}
    want = list(cfg.get("algorithms", []))
    algs: dict = {}
    for name in ("ls", "dtw", "subseq_dtw", "chroma_dtw", "note_align"):
        if name in want or (name in ("dtw", "note_align", "subseq_dtw") and "ensemble" in want):
            algs[name] = Cached(base[name])
    ckpt = ROOT_DIR / cfg.get("neural_checkpoint", "models/melody_encoder.pt")
    neural_wanted = [a for a in want if a.startswith("neural") or a == "ensemble_neural"]
    if neural_wanted:
        try:
            from melodyai.matching.neural import NeuralEmbedder, NeuralIndex, NeuralMatcher, NeuralRerank

            if not ckpt.exists():
                raise FileNotFoundError(f"{ckpt} not found (run `make train`)")
            emb = NeuralEmbedder(ckpt)
            flat = NeuralIndex(emb, index, None, "exact")
            approx = NeuralIndex(emb, index, None, "approx")
            t0 = time.perf_counter()
            flat._ensure()
            notes["neural_index_build_s"] = round(time.perf_counter() - t0, 3)
            approx._emb = flat._emb
            t0 = time.perf_counter()
            approx._ensure()
            notes["approx_index_build_s"] = round(time.perf_counter() - t0, 3)
            notes["vector_indexes"] = {"exact": flat.index_name, "approx": approx.index_name}
            algs["neural"] = Cached(NeuralMatcher(flat, "neural"))
            algs["neural_approx"] = Cached(NeuralMatcher(approx, "neural_approx"))
            algs["neural_rerank"] = Cached(NeuralRerank(flat, "neural_rerank"))
            notes["neural_model"] = {"checkpoint": str(ckpt.relative_to(ROOT_DIR)), "id": emb.id,
                                     "device": emb.device, "train_config": emb.ckpt.get("train_config"),
                                     "val": emb.ckpt.get("val"), "epoch": emb.ckpt.get("epoch")}
        except Exception as e:  # neural part is optional
            notes["neural_skipped"] = str(e)
    if "ensemble" in want:
        algs["ensemble"] = MelodyEnsemble("ensemble", {k: algs[k] for k in ("dtw", "note_align", "subseq_dtw")})
    if "ensemble_neural" in want and "neural_rerank" in algs:
        algs["ensemble_neural"] = MelodyEnsemble(
            "ensemble_neural", {k: algs[k] for k in ("dtw", "note_align", "subseq_dtw", "neural_rerank")})
    return {k: v for k, v in algs.items() if k in want}, notes | {"_all": algs}


def _rank(cands, target_id):
    for i, c in enumerate(cands):
        if c.song_id == target_id:
            return i + 1
    return None


def evaluate_melody(algs: dict, queries: list[dict], index, all_algs: dict, progress) -> tuple[list, dict]:
    rows = []
    total = len(algs) * len(queries)
    n = 0
    for name, alg in algs.items():
        for q in queries:
            qf = q["query"]
            t0 = time.perf_counter()
            cands = alg.search(qf, index, 10) if qf.informative else []
            lat = time.perf_counter() - t0
            if hasattr(alg, "members"):  # ensemble over cached members: sum member latencies
                lat = sum(m.lat.get(id(qf), 0.0) for m in alg.members.values()) + lat
            elif isinstance(alg, Cached):
                lat = alg.lat.get(id(qf), lat)
            target = f"demo-{q['target']:02d}" if q["target"] is not None else None
            r = _rank(cands, target) if target else None
            rows.append({"algorithm": name, "query": q["id"], "condition": q["condition"], "target": target,
                         "rank": r, "top_song": cands[0].song_id if cands else None,
                         "top_score": cands[0].score if cands else 0.0,
                         "second_score": cands[1].score if len(cands) > 1 else 0.0,
                         "latency_s": lat, "informative": bool(qf.informative),
                         "pitch_s": qf.timings.get("pitch_s"),
                         "song_parity": (q["target"] if q["target"] is not None else int(q["id"].split("/")[1])) % 2})
            n += 1
            if n % 25 == 0:
                progress(n / total, f"{name}")
    return rows, summarize_melody(rows)


def summarize_melody(rows: list[dict]) -> dict:
    out = {}
    for name in sorted({r["algorithm"] for r in rows}):
        R = [r for r in rows if r["algorithm"] == name]
        conds = sorted({r["condition"] for r in R if r["target"]})
        per = {c: retrieval_metrics([r["rank"] for r in R if r["condition"] == c]) for c in conds}
        incat = [r for r in R if r["target"]]
        lat = [r["latency_s"] for r in R if r["informative"]]
        out[name] = {"overall": retrieval_metrics([r["rank"] for r in incat]), "by_condition": per,
                     "latency": latency_stats(lat), "rejection": rejection_analysis(R)}
    return out


def rejection_analysis(R: list[dict]) -> dict:
    """Thresholds tuned on dev half (parity 0), evaluated on test half (parity 1)."""
    from sklearn.linear_model import LogisticRegression

    def split(par):
        incat = [r for r in R if r["target"] and r["song_parity"] == par]
        neg = [r for r in R if not r["target"] and r["song_parity"] == par]
        correct = np.array([r["top_score"] for r in incat if r["rank"] == 1])
        wrong = np.array([r["top_score"] for r in incat if r["rank"] != 1])
        ooc = np.array([r["top_score"] for r in neg])
        X = np.array([[r["top_score"], r["top_score"] - r["second_score"]] for r in incat + neg])
        y = np.array([1 if r["rank"] == 1 else 0 for r in incat] + [0] * len(neg))
        return correct, wrong, ooc, X, y

    c0, w0, o0, X0, y0 = split(0)
    c1, w1, o1, X1, y1 = split(1)
    accept = threshold_for_far(o0, 0.05)
    uncertain = threshold_for_far(o0, 0.25)
    res = {"thresholds_dev": {"accept": round(accept, 4), "uncertain": round(min(uncertain, accept), 4),
                              "margin": 0.0, "fitted_on": "dev half (even song indices)",
                              "target_false_accept_rate": 0.05},
           "test": rejection_metrics(c1, w1, o1, accept),
           "dev": rejection_metrics(c0, w0, o0, accept)}
    if len(set(y0)) == 2 and len(y1):
        lr = LogisticRegression().fit(X0, y0)
        res["platt"] = {"coef": [float(v) for v in lr.coef_[0]], "intercept": float(lr.intercept_[0])}
        res["calibration_test"] = ece(lr.predict_proba(X1)[:, 1], y1)
    return res


def evaluate_extractors(cfg, melodies, ooc, index, all_algs, seed, cache_dir, progress) -> dict:
    from melodyai.evaluation.datasets import build_query_set
    from melodyai.pitch.extractors import EXTRACTORS

    pc = cfg["pitch_extractor_comparison"]
    alg_name = pc.get("algorithm", "ensemble")
    out = {}
    exts = [e for e in pc["extractors"] if EXTRACTORS[e].available()]
    for k, ext in enumerate(exts):
        qs = build_query_set(melodies[: pc["songs"]], ooc, pc["conditions"], 0, 0, seed + 17, ext,
                             cfg.get("workers", 4), cache_dir / f"extractor_{ext}_{pc['songs']}.pkl",
                             progress=lambda p, m, k=k, ext=ext: progress((k + p * 0.8) / len(exts), f"{ext} {m}"))
        alg = all_algs.get(alg_name) or all_algs["dtw"]
        ranks, by_cond = [], {}
        for q in qs:
            c = alg.search(q["query"], index, 10) if q["query"].informative else []
            r = _rank(c, f"demo-{q['target']:02d}")
            ranks.append(r)
            by_cond.setdefault(q["condition"], []).append(r)
        out[ext] = {"overall": retrieval_metrics(ranks),
                    "by_condition": {c: retrieval_metrics(v) for c, v in by_cond.items()},
                    "pitch_time": latency_stats([q["query"].timings.get("pitch_s", 0) for q in qs]),
                    "mean_query_audio_s": float(np.mean([q["meta"]["audio_s"] for q in qs]))}
    return {"algorithm": alg_name, "results": out, "unavailable": [e for e in pc["extractors"] if e not in exts]}


def evaluate_lyrics(seed) -> tuple[dict, object, dict]:
    from melodyai.demo.build import LYRIC_SLOTS
    from melodyai.demo.lyrics_bank import LYRICS
    from melodyai.evaluation.datasets import lyrics_queries
    from melodyai.lyrics.search import LyricsIndex

    idx = LyricsIndex(":memory:")
    ids = {i: f"demo-{slot:02d}" for i, slot in enumerate(LYRIC_SLOTS)}
    for i, l in enumerate(LYRICS):
        idx.add(ids[i], l["text"], l["language"])
    qs = lyrics_queries(LYRICS, seed)
    rows = []
    for k, q in enumerate(qs):
        t0 = time.perf_counter()
        c = idx.search(q["text"], 10)
        lat = time.perf_counter() - t0
        target = ids[q["target"]] if q["target"] is not None else None
        r = next((j + 1 for j, x in enumerate(c) if x.song_id == target), None) if target else None
        rows.append({"algorithm": "lyrics", "query": f"{q['condition']}/{k}", "condition": q["condition"],
                     "target": target, "rank": r, "top_score": c[0].score if c else 0.0,
                     "second_score": c[1].score if len(c) > 1 else 0.0, "latency_s": lat, "informative": True,
                     "song_parity": (q["target"] if q["target"] is not None else k) % 2})
    summary = summarize_melody(rows)["lyrics"]
    return summary, idx, {"ids": ids, "rows": rows}


def evaluate_asr(cfg, lyr_index, ids, melodies, seed, progress) -> dict:
    from melodyai.demo.lyrics_bank import LYRICS
    from melodyai.evaluation.datasets import synthesize_speech, tts_available
    from melodyai.lyrics.asr import WhisperTranscriber, asr_available, model_status

    acfg = cfg.get("asr", {})
    model = acfg.get("model", "base")
    if acfg.get("enabled") is False:
        return {"skipped": "disabled in config"}
    if not (tts_available() and asr_available() and model_status(model)["cached"]):
        return {"skipped": f"requires macOS `say`, faster-whisper and a cached '{model}' model"}
    from melodyai.demo.queries import CONDITIONS, excerpt, perturb, synth_query
    from melodyai.lyrics.normalize import tokenize

    tr = WhisperTranscriber(model)
    rng = np.random.default_rng(seed)
    rows, wers = [], []
    for i, l in enumerate(LYRICS):
        lines = [x for x in l["text"].splitlines() if x.strip()]
        k = int(rng.integers(0, len(lines) - 1))
        text = lines[k] + ", " + lines[k + 1]
        audio = synthesize_speech(text, l["language"], i)
        if audio is None:
            continue
        for lang_mode in ("auto", "given"):
            t = tr.transcribe(audio.samples, audio.sr, None if lang_mode == "auto" else l["language"])
            c = lyr_index.search(t.text, 10) if t.speech_detected else []
            r = next((j + 1 for j, x in enumerate(c) if x.song_id == ids[i]), None)
            ref, hyp = tokenize(text), tokenize(t.text)
            wers.append(_wer(ref, hyp))
            rows.append({"song": ids[i], "language": l["language"], "lang_mode": lang_mode, "rank": r,
                         "detected_language": t.language, "speech_detected": t.speech_detected,
                         "asr_s": t.elapsed_s, "audio_s": len(audio.samples) / audio.sr, "wer": wers[-1]})
        progress((i + 1) / (len(LYRICS) + 10), f"speech {i + 1}/{len(LYRICS)}")
    # Hallucination check: hummed / whistled queries must not produce lyrics.
    hums = []
    for j in range(10):
        m = melodies[j]
        spec = CONDITIONS["whistle" if j % 3 == 2 else "hum_clean"]
        ex, _ = excerpt(m.notes, rng, 5, 8)
        pn, _ = perturb(ex, spec, rng)
        t = tr.transcribe(synth_query(pn, spec, rng), 16000)
        hums.append({"speech_detected": t.speech_detected, "text": t.text})
        progress((len(LYRICS) + j + 1) / (len(LYRICS) + 10), "hum hallucination check")
    out = {"model": model, "tts": "macOS say (synthetic speech)", "rows": rows}
    for mode in ("auto", "given"):
        R = [r for r in rows if r["lang_mode"] == mode]
        out[f"lyrics_from_speech_{mode}_language"] = retrieval_metrics([r["rank"] for r in R])
        out[f"language_id_accuracy_{mode}"] = float(np.mean([r["detected_language"] == r["language"] for r in R]))
    out["mean_wer"] = float(np.mean(wers)) if wers else None
    out["asr_time"] = latency_stats([r["asr_s"] for r in rows])
    out["real_time_factor"] = float(np.mean([r["asr_s"] / r["audio_s"] for r in rows])) if rows else None
    out["hum_false_speech_rate"] = float(np.mean([h["speech_detected"] for h in hums]))
    out["hum_samples"] = hums
    return out


def _wer(ref: list[str], hyp: list[str]) -> float:
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return float(d[len(hyp)] / max(1, len(ref)))


def evaluate_fusion(cfg, melodies, index, all_algs, lyr_index, ids, seed, workers, cache_dir, progress) -> dict:
    """Combined queries: a short ambiguous hum + a few error-prone words of the same song."""
    from sklearn.linear_model import LogisticRegression

    from melodyai.demo.lyrics_bank import LYRICS
    from melodyai.evaluation.datasets import _typo, build_query_set
    from melodyai.fusion.fusion import EngineOutput, fuse, song_features
    from melodyai.lyrics.normalize import tokenize

    slots = {int(v.split("-")[1]): i for i, v in ids.items()}  # song index -> lyrics index
    lyric_songs = [melodies[s] for s in sorted(slots)]
    qs = build_query_set(lyric_songs, [], ["hum_short", "hum_noisy"], 0, 0, seed + 5, "pyin", workers,
                         cache_dir / f"fusion_{len(lyric_songs)}.pkl", progress=lambda p, m: progress(0.6 * p, m))
    rng = np.random.default_rng(seed + 6)
    alg = all_algs.get("ensemble") or all_algs["dtw"]
    variants = {"4_words": 4, "2_words": 2}
    records = []
    for q in qs:
        song_idx = sorted(slots)[q["target"]]
        target = f"demo-{song_idx:02d}"
        l = LYRICS[slots[song_idx]]
        words = tokenize(l["text"])
        mel = alg.search(q["query"], index, 10) if q["query"].informative else []
        for vname, n_words in variants.items():
            s = int(rng.integers(0, len(words) - n_words))
            text = " ".join(_typo(w, rng) if rng.random() < 0.3 else w for w in words[s:s + n_words])
            lyr = lyr_index.search(text, 10)
            records.append({"variant": vname, "target": target, "parity": q["target"] % 2,
                            "condition": q["condition"], "text": text,
                            "melody": [{"song_id": c.song_id, "score": c.score} for c in mel],
                            "lyrics": [{"song_id": c.song_id, "score": c.score} for c in lyr]})
    progress(0.8, "fusing")

    def rank(lst, t):
        return next((i + 1 for i, d in enumerate(lst) if d["song_id"] == t), None)

    def outputs(r):
        return [EngineOutput("melody", r["melody"], 1.0), EngineOutput("lyrics", r["lyrics"], 1.0)]

    out = {"n_queries": len(qs), "melody_algorithm": alg.name, "variants": {}}
    for vname in variants:
        R = [r for r in records if r["variant"] == vname]
        res = {"melody_only": retrieval_metrics([rank(r["melody"], r["target"]) for r in R]),
               "lyrics_only": retrieval_metrics([rank(r["lyrics"], r["target"]) for r in R])}
        for method in cfg["fusion"]["methods"]:
            if method == "learned":
                X, y = [], []
                for r in R:
                    if r["parity"] != 0:
                        continue
                    o = outputs(r)
                    for sid in {d["song_id"] for d in r["melody"] + r["lyrics"]}:
                        X.append(song_features(o, sid))
                        y.append(int(sid == r["target"]))
                model = LogisticRegression(max_iter=1000).fit(np.array(X), np.array(y))
                test = [r for r in R if r["parity"] == 1]
                res["learned"] = retrieval_metrics([rank(fuse(outputs(r), "learned", model=model), r["target"])
                                                    for r in test])
                res["learned"]["note"] = "trained on dev half, evaluated on test half (small data: indicative only)"
            else:
                res[method] = retrieval_metrics([rank(fuse(outputs(r), method), r["target"]) for r in R])
        out["variants"][vname] = res
    out["examples"] = [{"variant": r["variant"], "target": r["target"], "text": r["text"],
                        "condition": r["condition"]} for r in records[:8]]
    return out


def evaluate_polyphonic(cfg, melodies, queries, all_algs, progress) -> dict:
    """Same hummed queries against references derived from different modalities."""
    import mir_eval

    from melodyai.audio.io import Audio
    from melodyai.evaluation.datasets import render_polyphonic
    from melodyai.features.extract import reference_from_audio

    pc = cfg["polyphonic"]
    n = pc["songs"]
    alg_name = pc.get("algorithm", "ensemble")
    modalities = {"midi": [], "monophonic_audio_pyin": [], "polyphonic_basic_pitch_skyline": [],
                  "polyphonic_pyin_on_mix": []}
    for i in range(n):
        m = melodies[i]
        mix, mel = render_polyphonic(m)
        sid = f"demo-{i:02d}"
        modalities["midi"].append(reference_from_notes(sid, m.notes))
        modalities["monophonic_audio_pyin"].append(reference_from_audio(sid, Audio(mel, 16000), "monophonic_audio"))
        modalities["polyphonic_basic_pitch_skyline"].append(
            reference_from_audio(sid, Audio(mix, 16000), "polyphonic_audio", "basic_pitch"))
        modalities["polyphonic_pyin_on_mix"].append(
            reference_from_audio(sid, Audio(mix, 16000), "polyphonic_audio", "pyin"))
        progress((i + 1) / n * 0.7, f"references {i + 1}/{n}")
    qs = [q for q in queries if q["target"] is not None and q["target"] < n
          and q["condition"] in ("hum_clean", "chorus_mid_song", "whistle")]
    from melodyai.engine import default_algorithms

    alg = default_algorithms()[alg_name]
    out = {"algorithm": alg_name, "songs": n, "n_queries": len(qs), "modalities": {}}
    for k, (mod, refs) in enumerate(modalities.items()):
        idx = MelodyIndex()
        for r in refs:
            if len(r.notes) >= 3:
                idx.add(r)
        ranks = [_rank(alg.search(q["query"], idx, 10), f"demo-{q['target']:02d}") if q["query"].informative
                 else None for q in qs]
        f1s = []
        for i, r in enumerate(refs):  # note transcription accuracy vs the true melody
            ref_n, est = melodies[i].notes, r.notes
            if len(est) == 0:
                f1s.append(0.0)
                continue
            iv = lambda ns: np.stack([ns.onsets, ns.onsets + ns.durations], axis=1)
            hz = lambda ns: 440.0 * 2 ** ((ns.pitches - 69) / 12)
            _, _, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
                iv(ref_n), hz(ref_n), iv(est), hz(est), onset_tolerance=0.1, pitch_tolerance=50.0, offset_ratio=None)
            f1s.append(f1)
        out["modalities"][mod] = {"retrieval": retrieval_metrics(ranks), "note_f1_mean": float(np.mean(f1s)),
                                  "note_f1_note": "mir_eval onset tol 100 ms, pitch tol 50 cents, no offsets"}
        progress(0.7 + 0.3 * (k + 1) / len(modalities), mod)
    return out


def make_figures(run_dir: Path, metrics: dict, rows: list[dict]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    fdir = run_dir / "figures"
    fdir.mkdir(exist_ok=True)
    mel = metrics["melody"]
    algs = list(mel)
    conds = sorted({c for a in algs for c in mel[a]["by_condition"]})
    M = np.array([[mel[a]["by_condition"].get(c, {}).get("top1", np.nan) for c in conds] for a in algs])
    fig, ax = plt.subplots(figsize=(1.1 * len(conds) + 3, 0.45 * len(algs) + 2))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(conds)), conds, rotation=40, ha="right")
    ax.set_yticks(range(len(algs)), algs)
    for i in range(len(algs)):
        for j in range(len(conds)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.6 else "k",
                        fontsize=8)
    fig.colorbar(im, ax=ax, label="Top-1 accuracy")
    ax.set_title("Top-1 by condition (synthetic queries)")
    fig.tight_layout()
    fig.savefig(fdir / "top1_by_condition.png", dpi=130)
    plt.close(fig)
    figs.append("figures/top1_by_condition.png")

    main = "ensemble" if "ensemble" in mel else algs[0]
    R = [r for r in rows if r["algorithm"] == main]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(0, 1, 31)
    ax.hist([r["top_score"] for r in R if r["rank"] == 1], bins, alpha=0.6, label="in-catalog, top-1 correct")
    ax.hist([r["top_score"] for r in R if r["target"] and r["rank"] != 1], bins, alpha=0.6, label="in-catalog, wrong")
    ax.hist([r["top_score"] for r in R if not r["target"]], bins, alpha=0.6, label="out-of-catalog / non-melodic")
    th = mel[main]["rejection"]["thresholds_dev"]["accept"]
    ax.axvline(th, color="k", ls="--", lw=1, label=f"accept threshold {th:.2f}")
    ax.set_xlabel(f"top-1 relative score ({main})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fdir / "score_distributions.png", dpi=130)
    plt.close(fig)
    figs.append("figures/score_distributions.png")

    cal = mel[main]["rejection"].get("calibration_test")
    if cal and cal["reliability"]:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.plot([b["confidence"] for b in cal["reliability"]], [b["accuracy"] for b in cal["reliability"]], "o-")
        ax.set_xlabel("predicted P(correct)")
        ax.set_ylabel("observed accuracy")
        ax.set_title(f"Reliability ({main}), ECE={cal['ece']:.3f}")
        fig.tight_layout()
        fig.savefig(fdir / "reliability.png", dpi=130)
        plt.close(fig)
        figs.append("figures/reliability.png")

    fig, ax = plt.subplots(figsize=(6, 0.35 * len(algs) + 1.5))
    lat = [mel[a]["latency"].get("mean_ms", 0) for a in algs]
    ax.barh(algs, lat)
    ax.set_xlabel("mean matching latency per query (ms, excludes pitch extraction)")
    fig.tight_layout()
    fig.savefig(fdir / "latency.png", dpi=130)
    plt.close(fig)
    figs.append("figures/latency.png")
    return figs


def write_calibration(metrics: dict, path: Path) -> None:
    cal = {"generated_at": datetime.now(timezone.utc).isoformat(), "source_run": metrics["run_id"],
           "note": "Thresholds fitted on synthetic dev queries (false-accept target 5% on out-of-catalog queries). "
                   "They are NOT validated on real recordings.", "thresholds": {}, "platt": {}}
    for name, m in metrics["melody"].items():
        rej = m["rejection"]
        cal["thresholds"][f"melody:{name}"] = {k: rej["thresholds_dev"][k] for k in ("accept", "uncertain", "margin")}
        if "platt" in rej:
            cal["platt"][f"melody:{name}"] = rej["platt"]
    if "lyrics" in metrics:
        rej = metrics["lyrics"]["rejection"]
        cal["thresholds"]["lyrics"] = {k: rej["thresholds_dev"][k] for k in ("accept", "uncertain", "margin")}
    path.write_text(json.dumps(cal, indent=1))


def run_evaluation(config_path, progress=None, should_stop=None, run_dir: Path | None = None) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = int(cfg.get("seed", 0))
    P = _Progress(progress)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{cfg.get('name', 'eval')}"
    run_dir = Path(run_dir or experiments_dir() / "runs" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = environment()
    (run_dir / "env.json").write_text(json.dumps(env, indent=1))
    cache_dir = experiments_dir() / "cache"
    workers = int(cfg.get("workers", 4))
    t_start = time.time()

    def check_cancel():
        if should_stop and should_stop():
            raise RuntimeError("cancelled")

    p = P.stage("setup")
    melodies = generate_catalog(cfg["catalog_size"], 1000)
    ooc_all, k = [], 0
    while len(ooc_all) < max(1, cfg.get("out_of_catalog", 0)):
        m = generate_melody(300_000 + k)
        k += 1
        if len(m.notes) >= 30:
            ooc_all.append(m)
    index, index_stats = build_reference_index(melodies)
    p(1.0, "index built")
    qhash = hashlib.sha1(json.dumps([cfg["catalog_size"], cfg["conditions"], cfg.get("out_of_catalog"),
                                     cfg.get("non_melodic"), seed, cfg.get("pitch_extractor"),
                                     cfg.get("queries_per_song", 1), FEATURE_VERSION]).encode()).hexdigest()[:10]
    from melodyai.evaluation.datasets import build_query_set

    queries = build_query_set(melodies, ooc_all, cfg["conditions"], cfg.get("out_of_catalog", 0),
                              cfg.get("non_melodic", 0), seed, cfg.get("pitch_extractor", "pyin"), workers,
                              cache_dir / f"queries_{qhash}.pkl", cfg.get("queries_per_song", 1), P.stage("queries"))
    check_cancel()
    algs, notes = build_algorithms(cfg, index)
    all_algs = notes.pop("_all")
    rows, mel_metrics = evaluate_melody(algs, queries, index, all_algs, P.stage("melody"))
    check_cancel()
    metrics: dict = {"run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
                     "dataset": {"type": "synthetic", "catalog": f"MelodyAI demo generator, {cfg['catalog_size']} songs "
                                 "(seeds 1000+)", "queries": len(queries), "conditions": cfg["conditions"],
                                 "query_hash": qhash, "out_of_catalog_seeds": "300000+"},
                     "indexing": index_stats, "algorithms": {n: getattr(a, "label", n) for n, a in algs.items()},
                     "melody": mel_metrics, "notes": notes}
    pq = run_dir / "per_query.jsonl"
    with pq.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    p = P.stage("extractors")
    if cfg.get("pitch_extractor_comparison", {}).get("enabled"):
        metrics["pitch_extractors"] = evaluate_extractors(cfg, melodies, ooc_all, index, all_algs, seed, cache_dir, p)
    check_cancel()
    p = P.stage("lyrics")
    lyr_index = ids = None
    if cfg.get("lyrics", {}).get("enabled"):
        metrics["lyrics"], lyr_index, lyr_extra = evaluate_lyrics(seed)
        ids = lyr_extra["ids"]
        with pq.open("a") as f:
            for r in lyr_extra["rows"]:
                f.write(json.dumps(r) + "\n")
    p(1.0, "done")
    p = P.stage("asr")
    if lyr_index is not None and cfg.get("asr", {}).get("enabled") is not False:
        metrics["asr"] = evaluate_asr(cfg, lyr_index, ids, melodies, seed, p)
    check_cancel()
    p = P.stage("fusion")
    if lyr_index is not None and cfg.get("fusion", {}).get("enabled") and cfg["catalog_size"] >= 40:
        metrics["fusion"] = evaluate_fusion(cfg, melodies, index, all_algs, lyr_index, ids, seed, workers, cache_dir, p)
    elif lyr_index is not None and cfg.get("fusion", {}).get("enabled"):
        metrics["fusion"] = {"skipped": "fusion benchmark needs the 48-song catalog (lyric songs span 0..38)"}
    check_cancel()
    p = P.stage("polyphonic")
    if cfg.get("polyphonic", {}).get("enabled"):
        metrics["polyphonic"] = evaluate_polyphonic(cfg, melodies, queries, all_algs, p)
    p = P.stage("write")
    metrics["figures"] = make_figures(run_dir, metrics, rows)
    metrics["elapsed_s"] = round(time.time() - t_start, 1)
    metrics["env"] = env
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    if cfg.get("write_calibration"):
        write_calibration(metrics, ROOT_DIR / "configs" / "calibration.json")
        metrics["calibration_written"] = "configs/calibration.json"
    p(1.0, "done")
    return {"run_dir": str(run_dir), "run_id": run_id, "metrics": metrics}
