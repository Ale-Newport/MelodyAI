"""MelodyAI command line interface.

    melodyai demo-data [--no-import]
    melodyai status
    melodyai recognize FILE [--mode auto|hum|whistle|lyrics|combined] [--algorithm NAME]
    melodyai search-lyrics "some words"
    melodyai import-dir DIR            # DIR/metadata.csv|json referencing files inside DIR
    melodyai ingest DIR --workers 8    # parallel, resumable bulk ingestion (large licensed collections)
    melodyai reindex [--all]
    melodyai evaluate [--config configs/eval_default.yaml]
    melodyai evaluate-scale [--config configs/eval_scale.yaml]
    melodyai train [--config configs/train_default.yaml]
    melodyai report
    melodyai asr-model [--model large-v3-turbo]   # download a Whisper model explicitly
    melodyai world-sync [--preset quick|standard|large] [--limit N] [--store DIR]   # popular songs' lyrics
    melodyai lakh-download / lakh-build [--limit N] [--holdout N]                   # real songs' melodies (humming)
    melodyai world-search "some words" [--offline] [--language es]                  # open-world lyrics search
    melodyai evaluate-world [--conditions index,online,both] [--limit N] [--no-tts] [--calibrate]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)


def _catalog():
    from melodyai.catalog.service import Catalog

    return Catalog()


def _progress(p: float, msg: str) -> None:
    sys.stderr.write(f"\r[{p * 100:5.1f}%] {msg[:60]:<60}")
    sys.stderr.flush()
    if p >= 1:
        sys.stderr.write("\n")


def cmd_demo_data(args) -> None:
    from melodyai.demo.build import build_demo_files, import_demo

    manifest = build_demo_files()
    print(f"Generated {len(manifest['songs'])} demo songs in data/demo")
    if not args.no_import:
        print(json.dumps(import_demo(_catalog(), progress=_progress)))


def cmd_status(args) -> None:
    print(json.dumps(_catalog().status(), indent=2))


def cmd_recognize(args) -> None:
    from melodyai.audio.io import load_audio
    from melodyai.engine import EngineSettings, RecognitionEngine

    cat = _catalog()
    eng = RecognitionEngine(cat.melody_index, cat.lyrics_index,
                            EngineSettings(asr_model=args.asr_model, asr_allow_download=False), fp_index=cat.fp_index)
    audio = load_audio(args.file, max_duration=60) if args.file else None
    res = eng.recognize(audio, args.text, args.mode, args.algorithm, args.language, explain=False)
    meta = cat.songs_by_ids([r["song_id"] for r in res["results"]])
    print(f"status: {res['status']} ({res['message']})  mode: {res['resolved_mode']}")
    for r in res["results"]:
        m = meta.get(r["song_id"], {})
        loc = f" @ {r['match_start']:.1f}-{r['match_end']:.1f}s" if r.get("match_start") is not None else ""
        print(f"  {r['rank']}. {m.get('title')} - {m.get('artist')}  score={r['score']:.3f}{loc}")
    if args.json:
        print(json.dumps(res, indent=1, default=str))


def cmd_search_lyrics(args) -> None:
    cat = _catalog()
    meta = cat.songs_by_ids(list(cat.lyrics_index.docs))
    for c in cat.lyrics_index.search(args.text, 5, args.language):
        print(f"{c.score:.3f}  {meta[c.song_id]['title']}  | {c.matched_text}")


def cmd_import_dir(args) -> None:
    from melodyai.catalog.importers import parse_lyrics, parse_metadata
    from melodyai.catalog.service import DuplicateSongError

    root = Path(args.dir).resolve()
    manifest = next((root / n for n in ("metadata.csv", "metadata.json") if (root / n).exists()), None)
    if manifest is None:
        sys.exit("DIR must contain metadata.csv or metadata.json")
    cat = _catalog()
    for row in parse_metadata(manifest.read_bytes(), manifest.name):
        ref = (root / row["reference_file"]).resolve()
        if root not in ref.parents:
            sys.exit(f"reference_file outside of {root}: {row['reference_file']}")
        lyrics = None
        if row.get("lyrics_file"):
            lp = (root / row["lyrics_file"]).resolve()
            if root in lp.parents:
                lyrics = parse_lyrics(lp.read_bytes(), lp.name)
        try:
            s = cat.add_song(row, ref.read_bytes(), ref.name, lyrics=lyrics)
            print(f"{s['index_status']:>9}  {s['id']}  {s['title']}")
        except DuplicateSongError as e:
            print(f"duplicate  {row['title']} ({e.existing_id})")


def _ingest_job(args):
    """Worker: hash, decode and analyse one file (no catalog access)."""
    import hashlib
    import warnings

    warnings.filterwarnings("ignore")
    from melodyai.catalog.service import compute_song

    path, rtype, track = args
    data = Path(path).read_bytes()
    try:
        comp = compute_song("pending", rtype, data, {"midi_track": track} if track is not None else {})
        return path, hashlib.sha256(data).hexdigest(), comp, None
    except Exception as e:  # report and continue with the other files
        return path, None, None, str(e)


def cmd_ingest(args) -> None:
    """Parallel, resumable bulk ingestion of a large licensed collection.

    DIR/metadata.csv|json as for import-dir; or, without a manifest, every audio/MIDI file in DIR
    (recursively) is imported with its file name as title. Files whose content hash is already in the
    catalog are skipped, so an interrupted run can simply be restarted.
    """
    import hashlib
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from melodyai.catalog.importers import AUDIO_EXTENSIONS, MIDI_EXTENSIONS, parse_lyrics, parse_metadata
    from melodyai.catalog.service import DuplicateSongError

    root = Path(args.dir).resolve()
    manifest = next((root / n for n in ("metadata.csv", "metadata.json") if (root / n).exists()), None)
    if manifest:
        rows = parse_metadata(manifest.read_bytes(), manifest.name)
    else:
        rows = [{"title": p.stem, "reference_file": str(p.relative_to(root)), "source": args.source,
                 "license": args.license} for p in sorted(root.rglob("*")) if p.suffix.lower() in
                AUDIO_EXTENSIONS | MIDI_EXTENSIONS]
    cat = _catalog()
    known = cat.known_hashes()
    jobs, by_path = [], {}
    for row in rows:
        ref = (root / row["reference_file"]).resolve()
        if root not in ref.parents or not ref.is_file():
            print(f"skip (missing/outside dir): {row['reference_file']}")
            continue
        if hashlib.sha256(ref.read_bytes()).hexdigest() in known:
            continue
        rtype = row.get("reference_type") or ("midi" if ref.suffix.lower() in MIDI_EXTENSIONS else args.reference_type)
        row["reference_type"] = rtype
        jobs.append((str(ref), rtype, None))
        by_path[str(ref)] = row
    print(f"{len(rows)} files listed, {len(rows) - len(jobs)} already in the catalog or skipped, {len(jobs)} to ingest")
    done = failed = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futures = [ex.submit(_ingest_job, j) for j in jobs]
        for k, f in enumerate(as_completed(futures)):
            path, digest, comp, err = f.result()
            row = by_path[path]
            if err:
                failed += 1
                print(f"\nerror {row['reference_file']}: {err}")
                continue
            lyrics = None
            if row.get("lyrics_file"):
                lp = (root / row["lyrics_file"]).resolve()
                if root in lp.parents and lp.is_file():
                    lyrics = parse_lyrics(lp.read_bytes(), lp.name)
            try:
                cat.add_song(row, Path(path).read_bytes(), Path(path).name, lyrics=lyrics, precomputed=comp)
                done += 1
            except DuplicateSongError:
                pass
            _progress((k + 1) / max(1, len(futures)), Path(path).name)
    cat.flush_indexes()
    print(json.dumps({"ingested": done, "failed": failed, "catalog": cat.status()["songs"]}))


def cmd_reindex(args) -> None:
    print(_catalog().reindex(only_needed=not args.all, progress=_progress))


def cmd_evaluate(args) -> None:
    from melodyai.evaluation.runner import run_evaluation

    res = run_evaluation(args.config, progress=_progress)
    print(f"Saved results to {res['run_dir']}")


def cmd_evaluate_scale(args) -> None:
    from melodyai.evaluation.scale import run_scale

    res = run_scale(args.config, progress=_progress)
    print(f"Saved results to {res['run_dir']}")


def cmd_train(args) -> None:
    from melodyai.training.train import train_from_config

    res = train_from_config(args.config, progress=_progress, resume=args.resume)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))


def cmd_report(args) -> None:
    from melodyai.evaluation.report import build_report

    print(f"Wrote {build_report()}")


def cmd_report_scale(args) -> None:
    from melodyai.evaluation.report import build_scale_report

    print(f"Wrote {build_scale_report()}")


def cmd_asr_model(args) -> None:
    from faster_whisper.utils import download_model

    from melodyai.lyrics.asr import AUTO, AUTO_DOWNLOAD

    print(download_model(AUTO_DOWNLOAD if args.model == AUTO else args.model))


def cmd_world_sync(args) -> None:
    """Fill the open-world lyrics index with popular songs (Deezer charts / playlists -> LRCLIB lyrics).

    Resumable: run it again to continue. Ctrl-C stops after the current song (twice: abort at once).
    """
    import signal
    import threading

    from melodyai.config import store_dir
    from melodyai.world.deezer import DeezerClient
    from melodyai.world.index import WorldLyricsIndex
    from melodyai.world.lrclib import LrclibClient
    from melodyai.world.sync import SyncBusy, sync_popular
    from melodyai.world.web import WebClient

    store = Path(args.store).expanduser().resolve() if args.store else store_dir()
    web = WebClient(store / "web_cache.db")
    index = WorldLyricsIndex(store / "world.db")
    stop = threading.Event()

    def on_sigint(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        sys.stderr.write("\nStopping after the current song (Ctrl-C again to abort)...\n")

    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        res = sync_popular(index, LrclibClient(web), DeezerClient(web), preset=args.preset, limit=args.limit,
                           progress=_progress, should_stop=stop.is_set, state_path=store / "world_sync.db")
    except SyncBusy as e:
        sys.stderr.write(f"{e}\n")
        raise SystemExit(1) from None
    finally:
        signal.signal(signal.SIGINT, previous)
        index.close()
        web.close()
    if args.json:
        print(json.dumps(res, indent=1, default=str))
        return
    for s in res["sources"]:
        extra = f"  ({s['message']})" if s.get("message") else ""
        print(f"  {s['status']:<9} {s['group']:<9} {s['tracks']:>4} tracks {s['new']:>4} new  {s['source']}{extra}")
    print(json.dumps({k: v for k, v in res.items() if k != "sources"}, indent=1, default=str))


def cmd_world_search(args) -> None:
    """Open-world lyrics search for typed words: titles, artists, scores and sources (lyrics are not printed)."""
    from melodyai.config import store_dir
    from melodyai.engine import WORLD_TOP_K
    from melodyai.fusion.fusion import decide
    from melodyai.world.index import WorldLyricsIndex
    from melodyai.world.search import WorldSearch
    from melodyai.world.web import WebClient

    store = Path(args.store).expanduser().resolve() if args.store else store_dir()
    web = WebClient(Path(args.cache).expanduser() if args.cache else store / "web_cache.db", offline=args.offline)
    index = WorldLyricsIndex(store / "world.db")
    try:
        # decide on at least WORLD_TOP_K candidates (the margin needs the runner-up), then show --top-k
        res = WorldSearch(index, web, learn=args.learn).search(args.text, args.language,
                                                               max(args.top_k, WORLD_TOP_K), online=not args.offline)
        decision = decide([{"song_id": c.song_id, "score": c.score} for c in res.candidates], "world")
        res.candidates = res.candidates[:max(1, args.top_k)]
        if args.json:
            out = res.to_dict()
            for c in out["candidates"]:
                c.pop("matched_text", None)
            print(json.dumps({**out, "decision": decision}, indent=1, ensure_ascii=False))
            return
        print(f"status: {res.status}  decision: {decision['status']}  online: {res.online}  "
              f"index: {len(index):,} songs  {res.elapsed_s:.2f}s" + (f"  ({res.message})" if res.message else ""))
        for name, st in res.sources.items():
            extra = f"  {st['message']}" if st.get("message") else ""
            print(f"  source {name:<12} {st['status']:<11} n={st.get('n', 0):<3} {st.get('elapsed_s', 0):.2f}s{extra}")
        for i, c in enumerate(res.candidates, 1):
            print(f"  {i}. {c.title} - {c.artist}  score={c.score:.3f} (align {c.alignment_score:.3f}, "
                  f"coverage {c.coverage:.2f}, lines {c.line_range[0]}-{c.line_range[1]})  "
                  f"sources={','.join(c.sources)}")
    finally:
        index.close()
        web.close()


def cmd_evaluate_world(args) -> None:
    """Real-song evaluation of the open-world recognition (see melodyai/evaluation/world_eval.py)."""
    from melodyai.evaluation.world_eval import run_world_eval

    res = run_world_eval(args.config, conditions=args.conditions, variants=args.variants, limit=args.limit,
                         tts=not args.no_tts, calibrate=args.calibrate, seed=args.seed, online_max=args.online_max,
                         asr_songs=args.asr_songs, label=args.label, store=args.store,
                         markdown_path=None if args.markdown == "-" else args.markdown, progress=_progress)
    rep = res["report"]
    for r in rep["summary"]["overall"]:
        print(f"  {r['condition']:<7} n={r['n']:<4} top1={r.get('top1') or 0:.2f} top5={r.get('top5') or 0:.2f} "
              f"match_ok={r.get('match_correct') or 0:.2f} false_match={r.get('false_match') or 0:.2f}")
    for r in rep["summary"]["negatives"]:
        print(f"  {r['condition']:<7} negatives n={r['n']:<3} false_match={r.get('false_match') or 0:.2f}")
    cal = rep.get("calibration")
    if cal:
        from melodyai.evaluation.world_eval import chosen_world_thresholds

        print(f"  world thresholds: budgeted fit {cal['thresholds']}, chosen {chosen_world_thresholds(cal)}"
              + (f" (written to {rep['calibration_written']})" if rep.get("calibration_written") else ""))
    print(f"Saved results to {res['report_path']}" + (f" and {res['markdown_path']}" if res["markdown_path"] else ""))


def cmd_lakh_download(args) -> None:
    """Download Lakh MIDI LMD-matched (~1.4 GB) + match scores + Million Song Dataset titles, and unpack it."""
    import shutil
    import subprocess
    import tarfile

    from melodyai.config import store_dir
    from melodyai.world.lakh import DOWNLOADS

    base = (Path(args.store).expanduser().resolve() if args.store else store_dir()) / "lakh"
    dl, raw = base / "download", base / "raw"
    dl.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        target = dl / name
        if target.is_file() and target.stat().st_size > 0 and not args.force:
            print(f"have {name}")
            continue
        print(f"downloading {url}")
        subprocess.run(["curl", "-fL", "--retry", "3", "-C", "-", "-o", str(target), url], check=True)
    if args.force and raw.exists():
        shutil.rmtree(raw)
    if not (raw / "lmd_matched").is_dir():
        print("unpacking lmd_matched.tar.gz")
        raw.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dl / "lmd_matched.tar.gz") as tar:
            tar.extractall(raw, filter="data")
    print(f"Lakh MIDI ready in {base}")


def cmd_lakh_build(args) -> None:
    """Build the world melody index (<store>/world_melodies) from the downloaded Lakh MIDI files."""
    from melodyai.config import models_dir, store_dir
    from melodyai.world.lakh import build_lakh_index

    store = Path(args.store).expanduser().resolve() if args.store else store_dir()
    if args.checkpoint is None and (models_dir() / "melody_encoder_lakh.pt").is_file():
        args.checkpoint = "melody_encoder_lakh.pt"
    meta = build_lakh_index(store / "lakh" / "raw", store / "lakh" / "download", Path(args.out or store / "world_melodies"),
                            workers=args.workers, limit=args.limit, holdout=args.holdout, checkpoint=args.checkpoint,
                            reembed=args.reembed, progress=_progress)
    print(json.dumps(meta, indent=1))


def main(argv: list[str] | None = None) -> None:
    from melodyai.lyrics.asr import MODEL_CHOICES as ASR_MODEL_CHOICES

    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="melodyai")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("demo-data")
    s.add_argument("--no-import", action="store_true")
    s.set_defaults(fn=cmd_demo_data)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    s = sub.add_parser("recognize")
    s.add_argument("file", nargs="?")
    s.add_argument("--text")
    s.add_argument("--mode", default="auto")
    s.add_argument("--algorithm")
    s.add_argument("--language")
    s.add_argument("--asr-model", default="auto", choices=ASR_MODEL_CHOICES,
                   help="Whisper model; auto = best downloaded model")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_recognize)
    s = sub.add_parser("search-lyrics")
    s.add_argument("text")
    s.add_argument("--language")
    s.set_defaults(fn=cmd_search_lyrics)
    s = sub.add_parser("import-dir")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_import_dir)
    s = sub.add_parser("ingest")
    s.add_argument("dir")
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.add_argument("--reference-type", default="polyphonic_audio",
                   help="for audio files without a manifest row (polyphonic_audio for commercial-style mixes)")
    s.add_argument("--source", default=None)
    s.add_argument("--license", default=None)
    s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("reindex")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_reindex)
    s = sub.add_parser("evaluate")
    s.add_argument("--config", default="configs/eval_default.yaml")
    s.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("evaluate-scale")
    s.add_argument("--config", default="configs/eval_scale.yaml")
    s.set_defaults(fn=cmd_evaluate_scale)
    s = sub.add_parser("train")
    s.add_argument("--config", default="configs/train_default.yaml")
    s.add_argument("--resume", action="store_true")
    s.set_defaults(fn=cmd_train)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    sub.add_parser("report-scale").set_defaults(fn=cmd_report_scale)
    s = sub.add_parser("asr-model")
    s.add_argument("--model", default="large-v3-turbo", choices=ASR_MODEL_CHOICES,
                   help="auto = the recommended model (large-v3-turbo, ~1.6 GB)")
    s.set_defaults(fn=cmd_asr_model)
    s = sub.add_parser("world-sync", help="add popular songs (charts, TikTok, classics) to the open-world index")
    s.add_argument("--preset", default="quick", choices=("quick", "standard", "large"))
    s.add_argument("--limit", type=int, default=None, help="keep only the N most important collected songs")
    s.add_argument("--store", default=None, help="store directory (default: MELODYAI_STORE_DIR or data/store)")
    s.add_argument("--json", action="store_true", help="print the full result as JSON")
    s.set_defaults(fn=cmd_world_sync)
    s = sub.add_parser("lakh-download", help="download Lakh MIDI (LMD-matched, CC-BY 4.0, ~1.5 GB) for humming search")
    s.add_argument("--store", default=None, help="store directory (default: MELODYAI_STORE_DIR or data/store)")
    s.add_argument("--force", action="store_true", help="download and unpack again")
    s.set_defaults(fn=cmd_lakh_download)
    s = sub.add_parser("lakh-build", help="build the humming index of real songs from the downloaded Lakh MIDI")
    s.add_argument("--store", default=None, help="store directory (default: MELODYAI_STORE_DIR or data/store)")
    s.add_argument("--out", default=None, help="output folder (default: <store>/world_melodies)")
    s.add_argument("--limit", type=int, default=None, help="only the N most confidently matched songs")
    s.add_argument("--holdout", type=int, default=300, help="songs kept out of the index for the evaluation")
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    s.add_argument("--checkpoint", default=None,
                   help="melody encoder (default: models/melody_encoder_lakh.pt if present, else melody_encoder.pt)")
    s.add_argument("--reembed", action="store_true",
                   help="keep the extracted melodies, rebuild only the search index (e.g. after training an encoder)")
    s.set_defaults(fn=cmd_lakh_build)
    s = sub.add_parser("world-search", help="open-world lyrics search (world index + LRCLIB/Deezer) for typed words")
    s.add_argument("text")
    s.add_argument("--language", default=None, help="es, en, ... (default: any)")
    s.add_argument("--offline", action="store_true", help="local world index only (no online sources)")
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--learn", action="store_true", help="add verified songs to the world index (off by default)")
    s.add_argument("--store", default=None, help="store directory (default: MELODYAI_STORE_DIR or data/store)")
    s.add_argument("--cache", default=None, help="web cache file (default: <store>/web_cache.db)")
    s.add_argument("--json", action="store_true", help="print the result as JSON (without matched lyrics)")
    s.set_defaults(fn=cmd_world_search)
    s = sub.add_parser("evaluate-world", help="real-song evaluation of the open-world recognition")
    s.add_argument("--config", default="configs/eval_world.yaml")
    s.add_argument("--conditions", default=None, help="comma list of index,online,both (default: from the config)")
    s.add_argument("--variants", default=None,
                   help="comma list of chorus,verse,line,misremembered,asr,asr_sung,negative (default: config)")
    s.add_argument("--limit", type=int, default=None, help="evaluate a seeded sample of N songs")
    s.add_argument("--no-tts", action="store_true", help="skip the speech-synthesis + Whisper variants")
    s.add_argument("--calibrate", action="store_true", help="write the fitted world thresholds to calibration.json")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--online-max", type=int, default=None, help="max queries per online condition (LRCLIB budget)")
    s.add_argument("--asr-songs", type=int, default=None, help="songs with ASR variants (default: config)")
    s.add_argument("--label", default=None, help='run label, e.g. "pilot" (shown in the report)')
    s.add_argument("--store", default=None, help="store directory holding world.db (default: data/store)")
    s.add_argument("--markdown", default="docs/WORLD_RESULTS.md", help='summary file ("-" to skip)')
    s.set_defaults(fn=cmd_evaluate_world)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
