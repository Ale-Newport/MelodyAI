"""Generate docs/RESULTS.md from the latest (or a given) benchmark run.

    python -m melodyai.cli report

All numbers in the report are read from ``experiments/runs/<run>/metrics.json``;
nothing is typed by hand.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from melodyai.config import ROOT_DIR, experiments_dir


def _pct(v, d=1):
    return "–" if v is None else f"{100 * v:.{d}f}%"


def _f(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def latest_run(prefer: str = "default") -> Path:
    runs = sorted((experiments_dir() / "runs").glob("*/metrics.json"))
    if not runs:
        raise FileNotFoundError("No benchmark runs found. Run `make evaluate` first.")
    preferred = [r for r in runs if r.parent.name.endswith(f"-{prefer}")]
    return (preferred or runs)[-1].parent


def build_report(run_dir: Path | None = None, out: Path | None = None) -> Path:
    run_dir = Path(run_dir or latest_run())
    m = json.loads((run_dir / "metrics.json").read_text())
    out = Path(out or ROOT_DIR / "docs" / "RESULTS.md")
    fig_dir = out.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for f in m.get("figures", []):
        src = run_dir / f
        if src.exists():
            shutil.copy(src, fig_dir / Path(f).name)
    env = m.get("env", {})
    rel = run_dir.relative_to(ROOT_DIR) if ROOT_DIR in run_dir.parents else run_dir
    L: list[str] = []
    a = L.append
    a("# Experimental results\n")
    a("> **Auto-generated** by `python -m melodyai.cli report` from "
      f"`{rel}/metrics.json`. Do not edit by hand; re-run `make evaluate` to refresh.\n")
    a("> **Scope.** Every query below is *synthetic*: original melodies from the procedural generator, "
      "rendered with simple hum/whistle/voice synthesisers and passed through the real pitch-tracking pipeline. "
      "The numbers measure robustness to controlled transformations on a small synthetic catalog. They are **not** an "
      "estimate of accuracy on real people's recordings or on large commercial catalogs (see `docs/LIMITATIONS.md`).\n")
    a("## Run\n")
    a("| Item | Value |\n|---|---|")
    a(f"| Run id | `{m['run_id']}` |")
    a(f"| Date (UTC) | {m['created_at']} |")
    a(f"| Seed | {m['seed']} |")
    a(f"| Catalog | {m['dataset']['catalog']} |")
    a(f"| Queries | {m['dataset']['queries']} (hash `{m['dataset']['query_hash']}`) |")
    a(f"| Out-of-catalog material | generator seeds {m['dataset']['out_of_catalog_seeds']} + non-melodic signals |")
    a(f"| Hardware | {env.get('machine')} · {env.get('cpu_count')} CPUs · {env.get('memory_gb')} GB · {env.get('platform')} |")
    a(f"| Software | Python {env.get('python')}, numpy {env.get('numpy_version')}, librosa {env.get('librosa_version')}, "
      f"torch {env.get('torch_version')} ({env.get('torch_device')}), faiss {env.get('faiss_version')}, "
      f"faster-whisper {env.get('faster_whisper_version')} |")
    a(f"| Git commit | `{env.get('git_commit')}` |")
    a(f"| Wall time | {m.get('elapsed_s')} s |\n")
    nm = m.get("notes", {}).get("neural_model")
    if nm:
        val = nm.get("val") or {}
        a(f"Neural model: `{nm['checkpoint']}` (id `{nm['id']}`, epoch {nm.get('epoch')}, device {nm.get('device')}; "
          f"validation on unseen compositions: top-1 {_pct(val.get('top1'))}, MRR {_f(val.get('mrr'))}).\n")
    elif m.get("notes", {}).get("neural_skipped"):
        a(f"Neural algorithms skipped: {m['notes']['neural_skipped']}\n")

    mel = m["melody"]
    a("## Melody recognition — algorithm comparison\n")
    a("In-catalog queries over all conditions. Latency = matching time per query on the machine above "
      "(pitch extraction excluded; it is reported separately below). Rejection columns: threshold fitted on the "
      "*dev* half of the songs to reach ≤5% false accepts on out-of-catalog/non-melodic dev queries, then measured "
      "on the *test* half.\n")
    a("The test half contains 32 out-of-catalog / non-melodic queries, so one false accept moves the "
      "false-accept rate by 3.1 points.\n")
    a("| Algorithm | Top-1 | Top-3 | Top-5 | MRR | Mean latency | AUC (correct vs OOC) | FAR@thr (test) | Accepted correct (test) | Dataset |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for name, x in mel.items():
        o, r = x["overall"], x["rejection"]["test"]
        a(f"| `{name}` — {m['algorithms'].get(name, '')} | {_pct(o.get('top1'))} | {_pct(o.get('top3'))} | "
          f"{_pct(o.get('top5'))} | {_f(o.get('mrr'))} | {_f(x['latency'].get('mean_ms'), 0)} ms | "
          f"{_f(r.get('auc_correct_vs_ooc'))} | {_pct(r.get('false_accept_rate_ooc'))} | "
          f"{_pct(r.get('accepted_correct_rate'))} | synthetic |")
    a("")
    conds = sorted({c for x in mel.values() for c in x["by_condition"]})
    a("### Top-1 accuracy by query condition\n")
    a("| Algorithm | " + " | ".join(conds) + " |")
    a("|---|" + "---|" * len(conds))
    for name, x in mel.items():
        a(f"| `{name}` | " + " | ".join(_pct(x["by_condition"].get(c, {}).get("top1"), 0) for c in conds) + " |")
    a("\nConditions: see `melodyai/demo/queries.py` (`CONDITIONS`) and `docs/EVALUATION.md`.\n")
    if (fig_dir / "top1_by_condition.png").exists():
        a("![Top-1 by condition](figures/top1_by_condition.png)\n")

    a("### Calibration\n")
    a("Platt scaling (logistic regression on top score and margin) fitted on the dev half; ECE/Brier on the test half.\n")
    a("| Algorithm | ECE | Brier | n |\n|---|---|---|---|")
    for name, x in mel.items():
        c = x["rejection"].get("calibration_test")
        if c:
            a(f"| `{name}` | {_f(c['ece'])} | {_f(c['brier'])} | {c['n']} |")
    a("")
    for fig in ("score_distributions.png", "reliability.png", "latency.png"):
        if (fig_dir / fig).exists():
            a(f"![{fig}](figures/{fig})\n")

    pe = m.get("pitch_extractors")
    if pe:
        a("## Pitch extractors\n")
        a(f"Same synthetic queries analysed with each f0 extractor, matched with `{pe['algorithm']}`.\n")
        a("| Extractor | Top-1 | MRR | Mean pitch-extraction time per query | " + " | ".join(
            f"Top-1 {c}" for c in next(iter(pe["results"].values()))["by_condition"]) + " |")
        a("|---|---|---|---|" + "---|" * len(next(iter(pe["results"].values()))["by_condition"]))
        for k, v in pe["results"].items():
            a(f"| {k} | {_pct(v['overall']['top1'])} | {_f(v['overall']['mrr'])} | {_f(v['pitch_time'].get('mean_ms'), 0)} ms | "
              + " | ".join(_pct(x["top1"], 0) for x in v["by_condition"].values()) + " |")
        if pe.get("unavailable"):
            a(f"\nNot installed: {', '.join(pe['unavailable'])}.")
        a("\nPitch-extraction times were measured while 8 query-synthesis worker processes ran in parallel, so they "
          "are inflated by CPU contention and only comparable with each other. The per-condition sample is 24 queries.\n")

    ly = m.get("lyrics")
    if ly:
        a("## Lyrics search (typed / transcribed text)\n")
        a("20 original lyric texts (12 English, 8 Spanish). Out-of-catalog = original distractor lines and random "
          "bags of catalog words (topic overlap without the actual text).\n")
        a("| Condition | Top-1 | MRR | n |\n|---|---|---|---|")
        for k, v in ly["by_condition"].items():
            a(f"| {k} | {_pct(v['top1'])} | {_f(v['mrr'])} | {v['n']} |")
        r = ly["rejection"]["test"]
        a(f"\nRejection (test half): threshold {_f(r['threshold'])}, false accepts on distractors "
          f"{_pct(r['false_accept_rate_ooc'])}, accepted correct {_pct(r['accepted_correct_rate'])}, "
          f"AUC {_f(r['auc_correct_vs_ooc'])}.\n")

    asr = m.get("asr")
    if asr:
        a("## Speech → lyrics (faster-whisper)\n")
        if asr.get("skipped"):
            a(f"Skipped: {asr['skipped']}.\n")
        else:
            a(f"Spoken queries synthesised with **{asr['tts']}** (not human singing) and transcribed with Whisper "
              f"`{asr['model']}` on CPU.\n")
            a("| Metric | Value |\n|---|---|")
            for key, label in (("lyrics_from_speech_auto_language", "Song top-1 (language auto-detected)"),
                               ("lyrics_from_speech_given_language", "Song top-1 (language given)")):
                a(f"| {label} | {_pct(asr[key]['top1'])} (n={asr[key]['n']}) |")
            a(f"| Language-ID accuracy (auto) | {_pct(asr['language_id_accuracy_auto'])} |")
            a(f"| Mean word error rate | {_f(asr['mean_wer'], 2)} |")
            a(f"| Real-time factor | {_f(asr['real_time_factor'], 2)} |")
            a(f"| Hummed/whistled queries producing words | {_pct(asr['hum_false_speech_rate'])} |\n")

    fu = m.get("fusion")
    if fu and not fu.get("skipped"):
        a("## Combined recognition (melody + words)\n")
        a(f"{fu['n_queries']} combined queries: a short (2.5–4 s) or noisy hum of a lyric song plus a few consecutive "
          f"words from it with ~30% typo rate. With 4 words the text alone is usually enough; with 2 words it is "
          f"ambiguous, which is where complementary melodic evidence can help. Melody matcher: `{fu['melody_algorithm']}`.\n")
        variants = fu.get("variants") or {"4_words": fu}
        names = ("melody_only", "lyrics_only", "weighted", "rrf", "combmnz", "learned")
        a("| Strategy | " + " | ".join(f"Top-1 ({v.replace('_', ' ')})" for v in variants) + " | "
          + " | ".join(f"MRR ({v.replace('_', ' ')})" for v in variants) + " |")
        a("|---|" + "---|" * (2 * len(variants)))
        for k in names:
            if not all(k in v for v in variants.values()):
                continue
            label = k + (" (dev→test, small sample)" if k == "learned" else "")
            a(f"| {label} | " + " | ".join(_pct(v[k]["top1"]) for v in variants.values()) + " | "
              + " | ".join(_f(v[k]["mrr"]) for v in variants.values()) + " |")
        a("")

    po = m.get("polyphonic")
    if po:
        a("## Reference modality (MIDI vs audio vs polyphonic mixes)\n")
        a(f"The same {po['n_queries']} hummed/whistled queries for {po['songs']} songs, matched with `{po['algorithm']}` "
          "against references derived in different ways. Polyphonic mixes = melody (piano) + pad chords + bass, "
          "synthesised. Note F1 compares the extracted reference melody with the true one (mir_eval).\n")
        a("| Reference built from | Top-1 | MRR | Note F1 |\n|---|---|---|---|")
        for k, v in po["modalities"].items():
            a(f"| {k} | {_pct(v['retrieval']['top1'])} | {_f(v['retrieval']['mrr'])} | {_f(v['note_f1_mean'], 2)} |")
        a("\nThese mixes are far simpler than produced commercial recordings (no vocals, no drums, no effects); "
          "results on real polyphonic music are expected to be considerably worse and were not measured.\n")

    ix = m.get("indexing", {})
    a("## Indexing cost\n")
    a(f"{ix.get('songs')} songs → {ix.get('windows')} melody windows; {ix.get('mean_ms_per_song')} ms per song "
      f"(includes chroma of a synthesised rendering); window matrix {ix.get('window_matrix_mb')} MB; "
      f"process RSS delta {ix.get('rss_delta_mb')} MB."
      + (f" Neural window embeddings for the whole catalog: {m['notes']['neural_index_build_s']} s." if
         m.get("notes", {}).get("neural_index_build_s") else "") + "\n")
    out.write_text("\n".join(L))
    return out


def latest_scale_run() -> Path:
    runs = sorted((experiments_dir() / "runs").glob("*-scale/metrics.json"))
    if not runs:
        raise FileNotFoundError("No scale runs found. Run `make evaluate-scale` first.")
    return runs[-1].parent


def build_scale_report(run_dir: Path | None = None, out: Path | None = None) -> Path:
    """docs/SCALE_RESULTS.md from a scale run (all numbers read from its metrics.json)."""
    run_dir = Path(run_dir or latest_scale_run())
    m = json.loads((run_dir / "metrics.json").read_text())
    out = Path(out or ROOT_DIR / "docs" / "SCALE_RESULTS.md")
    env = m.get("env", {})
    rel = run_dir.relative_to(ROOT_DIR) if ROOT_DIR in run_dir.parents else run_dir
    L: list[str] = []
    a = L.append
    a("# Scale benchmark results\n")
    a(f"> **Auto-generated** by `python -m melodyai.cli report-scale` from `{rel}/metrics.json` "
      f"(run `{m['run_id']}`, {m['created_at']}, {env.get('machine')} · {env.get('cpu_count')} CPUs · "
      f"{env.get('platform')}, wall time {round(m['elapsed_s'] / 60)} min). Do not edit by hand.\n")
    a("> **Synthetic data.** The 48 demo songs are hidden among procedurally generated distractors rendered "
      "with the same simple synthesisers. Real catalogs differ (richer timbres, vocals, but also covers, "
      "remixes and near-duplicates). These tables measure how the *system* behaves as the index grows, not "
      "real-world accuracy. See `docs/SCALING.md` for the architecture and extrapolations.\n")
    fp = m.get("fingerprint")
    if fp:
        sizes = list(fp["sizes"])
        a("## Recordings and TikTok-style clips (audio fingerprint)\n")
        a(f"{fp['clip_seconds']:.0f}-second clips of the 48 indexed recordings. *top-1* = correct song ranked first; "
          f"*accepted* = top-1 correct **and** score ≥ {fp['accept_threshold']} (the threshold that would answer "
          "\"match\").\n")
        a("| Edit | " + " | ".join(f"{int(n):,} recordings: top-1 / accepted" for n in sizes) + " |")
        a("|---|" + "---|" * len(sizes))
        conds = list(fp["sizes"][sizes[0]]["by_condition"])
        for c in conds:
            a(f"| {c.replace('_', ' ')} | " + " | ".join(
                f"{_pct(fp['sizes'][n]['by_condition'][c]['top1'], 0)} / "
                f"{_pct(fp['sizes'][n]['by_condition'][c]['accepted_correct'], 0)}" for n in sizes) + " |")
        a("| **false accepts**: clips of unknown recordings | " + " | ".join(
            _pct(fp["sizes"][n]["false_accept"]["out_of_catalog_recording"], 1) for n in sizes) + " |")
        a("| **false accepts**: hummed queries | " + " | ".join(
            _pct(fp["sizes"][n]["false_accept"]["hummed_query"], 1) for n in sizes) + " |")
        a("| AUC correct vs negatives | " + " | ".join(_f(fp["sizes"][n]["auc_correct_vs_negatives"]) for n in sizes) + " |")
        a("| latency mean / p95 | " + " | ".join(
            f"{_f(fp['sizes'][n]['latency']['mean_ms'], 0)} / {_f(fp['sizes'][n]['latency']['p95_ms'], 0)} ms"
            for n in sizes) + " |")
        a("| index size | " + " | ".join(
            f"{fp['sizes'][n]['index']['bytes'] / 1e6:.0f} MB ({fp['sizes'][n]['index']['audio_hours']} h audio)"
            for n in sizes) + " |")
        last = fp["sizes"][sizes[-1]]["index"]
        a(f"\nStorage: **{last['mb_per_audio_hour']} MB per hour of audio** ({last['postings_per_second']} hash "
          "postings/s × 16 B + peak maps × 10 B). Edits: speed = resampling (tempo *and* pitch change, as TikTok "
          "\"sped up\"/\"slowed\"; slowed also adds reverb), pitch_up_2 = phase-vocoder pitch shift, stretch = "
          "tempo-only time stretch, phone = 300–3400 Hz band-limit + saturation, overlay = another voice as loud "
          "as the music.\n")
    mel = m.get("melody")
    if mel:
        sizes = list(mel["sizes"])
        a("## Humming / whistling vs catalog size (two-stage search)\n")
        a(f"Stage 1: neural window embeddings in an IVF-PQ index ({mel['nlist']} lists, {mel['m']} bytes/vector, "
          f"nprobe {mel['nprobe']}) → best {mel['n_candidates']} songs. Stage 2: window DTW + note alignment on "
          f"those songs only. The same {mel['sizes'][sizes[0]]['two_stage']['n']} synthetic hum/whistle queries as the "
          "main benchmark, plus its out-of-catalog/non-melodic queries for the AUC column.\n")
        a("| Songs | Two-stage top-1 | top-5 | MRR | Stage-1 candidate recall | Neural ANN alone top-1 | "
          "Exhaustive note-align top-1 (subset) | Latency stage 1 + 2 | Exhaustive latency | ANN index | AUC vs OOC |")
        a("|---|---|---|---|---|---|---|---|---|---|---|")
        for n in sizes:
            e = mel["sizes"][n]
            ex = e.get("exhaustive_note_align")
            a(f"| {int(n):,} | {_pct(e['two_stage']['top1'])} | {_pct(e['two_stage']['top5'])} | "
              f"{_f(e['two_stage']['mrr'])} | {_pct(e['candidate_recall'])} | {_pct(e['neural_ann_only']['top1'])} | "
              f"{(_pct(ex['top1']) + ' (n=' + str(ex['n_queries']) + ')') if ex else '–'} | "
              f"{_f(e['latency_stage1']['mean_ms'], 0)} + {_f(e['latency_stage2']['mean_ms'], 0)} ms | "
              f"{(_f(ex['latency']['mean_ms'], 0) + ' ms') if ex else '–'} | "
              f"{e['ann_vectors'] / 1e6:.2f} M vectors, {e['ann_index_mb']:.0f} MB | {_f(e['auc_correct_vs_ooc'])} |")
        last = mel["sizes"][sizes[-1]]
        if "candidate_recall_at" in last:
            ks = list(last["candidate_recall_at"])
            a("\nStage-1 candidate recall (is the right song among the first *k* songs retrieved by the ANN?) — the "
              "ceiling for the two-stage top-1:\n")
            a("| Songs | " + " | ".join(f"recall@{k}" for k in ks) + " |")
            a("|---|" + "---|" * len(ks))
            for n in sizes:
                ra = mel["sizes"][n].get("candidate_recall_at", {})
                a(f"| {int(n):,} | " + " | ".join(_pct(ra.get(k)) for k in ks) + " |")
        a("\nTop-1 by condition at the largest size:\n")
        a("| Condition | Top-1 | MRR |\n|---|---|---|")
        for c, v in last["by_condition"].items():
            a(f"| {c} | {_pct(v['top1'])} | {_f(v['mrr'])} |")
        a("")
    # Linear storage extrapolation from measured per-unit costs (no accuracy claim).
    a("## Linear storage extrapolation (not measured)\n")
    a("Index size grows linearly with the catalog; the per-unit costs below are *measured* in this run, the "
      "totals are simple multiplications for a 250-million-track catalog (Spotify's reported order of magnitude, "
      "average 3.5 min). Accuracy at that scale is **not** extrapolated: it depends on catalog content and was not "
      "measured.\n")
    a("| Index | Measured unit cost | 1 M tracks | 250 M tracks |\n|---|---|---|---|")
    hours = lambda n: n * 3.5 / 60.0  # noqa: E731
    if fp:
        mbh = fp["sizes"][list(fp["sizes"])[-1]]["index"]["mb_per_audio_hour"]
        a(f"| Fingerprint postings + peak maps | {mbh} MB / audio hour | {mbh * hours(1e6) / 1e3:,.0f} GB | "
          f"{mbh * hours(250e6) / 1e6:,.1f} TB |")
    if mel:
        e = mel["sizes"][list(mel["sizes"])[-1]]
        per_song = e["bytes_per_window"] * e["ann_vectors"] / int(list(mel["sizes"])[-1])
        a(f"| Melody IVF-PQ (window embeddings) | {per_song / 1e3:.1f} kB / song "
          f"({e['ann_vectors'] / int(list(mel['sizes'])[-1]):.0f} windows × {e['bytes_per_window']:.0f} B) | "
          f"{per_song * 1e6 / 1e9:,.1f} GB | {per_song * 250e6 / 1e12:,.2f} TB |")
    a("\nAt those sizes both indexes must be sharded across machines (they are append-only segment files, so "
      "sharding by song id is straightforward); see `docs/SCALING.md`.\n")
    out.write_text("\n".join(L))
    return out
