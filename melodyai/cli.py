"""MelodyAI command line interface.

    melodyai demo-data [--no-import]
    melodyai status
    melodyai recognize FILE [--mode auto|hum|whistle|lyrics|combined] [--algorithm NAME]
    melodyai search-lyrics "some words"
    melodyai import-dir DIR            # DIR/metadata.csv|json referencing files inside DIR
    melodyai reindex [--all]
    melodyai evaluate [--config configs/eval_default.yaml]
    melodyai train [--config configs/train_default.yaml]
    melodyai report
    melodyai asr-model --model base     # download a Whisper model explicitly
"""

from __future__ import annotations

import argparse
import json
import logging
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
                            EngineSettings(asr_model=args.asr_model, asr_allow_download=False))
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


def cmd_reindex(args) -> None:
    print(_catalog().reindex(only_needed=not args.all, progress=_progress))


def cmd_evaluate(args) -> None:
    from melodyai.evaluation.runner import run_evaluation

    res = run_evaluation(args.config, progress=_progress)
    print(f"Saved results to {res['run_dir']}")


def cmd_train(args) -> None:
    from melodyai.training.train import train_from_config

    res = train_from_config(args.config, progress=_progress, resume=args.resume)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))


def cmd_report(args) -> None:
    from melodyai.evaluation.report import build_report

    print(f"Wrote {build_report()}")


def cmd_asr_model(args) -> None:
    from faster_whisper.utils import download_model

    print(download_model(args.model))


def main(argv: list[str] | None = None) -> None:
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
    s.add_argument("--asr-model", default="base")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_recognize)
    s = sub.add_parser("search-lyrics")
    s.add_argument("text")
    s.add_argument("--language")
    s.set_defaults(fn=cmd_search_lyrics)
    s = sub.add_parser("import-dir")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_import_dir)
    s = sub.add_parser("reindex")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_reindex)
    s = sub.add_parser("evaluate")
    s.add_argument("--config", default="configs/eval_default.yaml")
    s.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("train")
    s.add_argument("--config", default="configs/train_default.yaml")
    s.add_argument("--resume", action="store_true")
    s.set_defaults(fn=cmd_train)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    s = sub.add_parser("asr-model")
    s.add_argument("--model", default="base")
    s.set_defaults(fn=cmd_asr_model)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
