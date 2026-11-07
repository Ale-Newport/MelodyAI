"""Evaluate MelodyAI on YOUR OWN labelled recordings (real people, with their consent).

    python scripts/evaluate_recordings.py RECORDINGS_DIR labels.csv [--algorithms dtw,ensemble] [--mode hum]

labels.csv columns: file,song_id[,mode]   (song_id = id in your catalog, or empty for
"not in catalog" queries, which measure false accepts). Results are written as JSON next
to the CSV. This is the way to obtain real-world numbers; the built-in benchmark is synthetic.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from melodyai.audio.io import load_audio
from melodyai.catalog.service import Catalog
from melodyai.engine import EngineSettings, RecognitionEngine
from melodyai.evaluation.metrics import retrieval_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--algorithms", default="ensemble")
    ap.add_argument("--mode", default="hum")
    args = ap.parse_args()
    cat = Catalog()
    eng = RecognitionEngine(cat.melody_index, cat.lyrics_index, EngineSettings(asr_enabled=True))
    rows = list(csv.DictReader(args.labels.open()))
    out = {"n": len(rows), "algorithms": {}}
    for alg in args.algorithms.split(","):
        ranks, accepted_wrong, accepted_ooc, n_ooc, times = [], 0, 0, 0, []
        per = []
        for r in rows:
            path = (args.recordings / r["file"]).resolve()
            if args.recordings.resolve() not in path.parents:
                raise SystemExit(f"{r['file']} is outside {args.recordings}")
            t0 = time.perf_counter()
            res = eng.recognize(load_audio(path, max_duration=120), mode=r.get("mode") or args.mode,
                                algorithm=alg, explain=False)
            times.append(time.perf_counter() - t0)
            ids = [x["song_id"] for x in res["results"]]
            target = (r.get("song_id") or "").strip() or None
            if target:
                ranks.append(ids.index(target) + 1 if target in ids else None)
                if res["status"] == "match" and ids and ids[0] != target:
                    accepted_wrong += 1
            else:
                n_ooc += 1
                accepted_ooc += res["status"] == "match"
            per.append({"file": r["file"], "target": target, "status": res["status"], "top": ids[:3]})
        out["algorithms"][alg] = {
            "retrieval": retrieval_metrics(ranks), "accepted_wrong_in_catalog": accepted_wrong,
            "false_accept_rate_out_of_catalog": accepted_ooc / n_ooc if n_ooc else None,
            "mean_seconds_per_query": sum(times) / max(1, len(times)), "per_query": per,
        }
        print(alg, json.dumps(out["algorithms"][alg]["retrieval"]))
    dst = args.labels.with_suffix(".results.json")
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
