"""Capture the static demo served by the GitHub Pages build (``frontend/public/demo/``).

The web app normally talks to the local API. The Pages build (``VITE_DEMO=1``) has no server, so it reads the
JSON responses saved here instead. Run it against a local API with the demo catalog imported::

    make backend            # in another terminal
    make demo-site          # = .venv/bin/python scripts/build_demo_site.py

What is captured, and nothing else:

* catalog, model, benchmark and settings responses (the original CC0 demo catalog and the benchmark reports
  that are already in ``experiments/runs``);
* three example recognitions made from *synthetic* queries of demo songs (a hummed chorus, a sped-up noisy clip
  of the recording, a misremembered typed lyric line) and the Explorer analysis of the hummed query;
* the demo songs' audio, transcoded to small MP3s.

Online song search is switched off on the server while the examples run (restored afterwards), so no real
song's lyrics end up in the files, and the example recognitions are removed from the local history again.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from melodyai.catalog.importers import load_midi_melody  # noqa: E402
from melodyai.demo.queries import CONDITIONS, excerpt, perturb, synth_query  # noqa: E402

OUT = ROOT / "frontend" / "public" / "demo"
SR = 16000


def round_floats(x, nd=4):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, list):
        return [round_floats(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: round_floats(v, nd) for k, v in x.items()}
    return x


def without_real_songs(r: dict) -> dict:
    """Drops open-world candidates (real songs from the lyrics / Lakh melody indexes, with their matched lines)."""
    for name in ("world", "world_melody"):
        engine = (r.get("engines") or {}).get(name)
        if engine:
            engine["candidates"] = []
    r["results"] = [x for x in r["results"] if not x["song_id"].startswith("world:")]
    return r


def save(rel: str, data) -> None:
    p = OUT / "api" / f"{rel}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(round_floats(data), separators=(",", ":"), ensure_ascii=False))


def wav_bytes(y: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def catalog_song(song_id: str) -> dict:
    cat = json.loads((ROOT / "data" / "demo" / "catalog.json").read_text())
    return next(s for s in cat["songs"] if s["id"] == song_id)


def hum_query(song_id: str, seed: int) -> np.ndarray:
    """A synthetic hum of the chorus in the middle of a demo song (same simulator as the benchmark)."""
    s = catalog_song(song_id)
    notes = load_midi_melody((ROOT / "data" / "demo" / s["reference_file"]).read_bytes()).notes
    sections = s["sections"] if isinstance(s["sections"], list) else None
    rng = np.random.default_rng(seed)
    spec = CONDITIONS["chorus_mid_song"]
    ex, _ = excerpt(notes, rng, 8.0, 11.0, sections, spec.section)
    pn, _ = perturb(ex, spec, rng)
    return synth_query(pn, spec, rng)


def clip_query(song_id: str, start: float, seconds: float, speed: float, seed: int) -> np.ndarray:
    """A TikTok-style edit of the demo recording: sped up (pitch rises with it) plus background noise."""
    s = catalog_song(song_id)
    y, _ = librosa.load(ROOT / "data" / "demo" / s["audio_file"], sr=SR, offset=start, duration=seconds * speed)
    y = librosa.resample(y, orig_sr=SR, target_sr=int(SR / speed))  # played back at SR -> faster and higher
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, len(y)).astype(np.float32)
    y = y / (np.abs(y).max() + 1e-9) * 0.7
    return y + noise * np.sqrt(np.mean(y**2) / 10 ** (18 / 10))  # 18 dB SNR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--bitrate", default="32k", help="MP3 bitrate of the demo songs")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.api, timeout=300)
    get = lambda path: c.get(path).raise_for_status().json()  # noqa: E731

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "api").mkdir(parents=True)

    # --- catalog, models, status ---------------------------------------------------------------------------
    for rel in ["health", "models", "catalog/status", "catalog/duplicates", "world/status"]:
        save(rel, get(f"/api/{rel}"))
    songs = get("/api/songs?limit=500")
    save("songs", songs)
    user_settings = get("/api/settings")
    save("settings", {**user_settings, "theme": "dark", "ui_language": "en", "microphone_id": None,
                      "retention": "discard", "external_provider_enabled": False, "online_search_enabled": True})

    # --- benchmarks ------------------------------------------------------------------------------------------
    exps = get("/api/experiments")
    save("experiments", exps)
    for run in exps["evaluations"]:
        rid = run["run_id"]
        metrics = get(f"/api/experiments/{rid}")
        save(f"experiments/{rid}", metrics)
        rows = [json.loads(line) for line in (ROOT / "experiments" / "runs" / rid / "per_query.jsonl").open()]
        save(f"experiments/{rid}/queries", rows)
        for fig in metrics.get("figures", []):
            dst = OUT / "figures" / rid / Path(fig).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / "experiments" / "runs" / rid / fig, dst)
    for run in exps.get("scale", []):
        save(f"experiments/{run['run_id']}", get(f"/api/experiments/{run['run_id']}"))

    # --- example recognitions (online search off while they run) ------------------------------------------
    c.put("/api/settings", json={"online_search_enabled": False}).raise_for_status()
    made: list[str] = []
    try:
        def recognize(mode: str, audio: bytes | None = None, text: str | None = None) -> dict:
            files = {"audio": ("query.wav", audio, "audio/wav")} if audio else None
            data = {"mode": mode, **({"text": text} if text else {})}
            r = c.post("/api/recognize", data=data, files=files).raise_for_status().json()
            made.append(r["id"])
            return r

        hum = None
        for song_id, seed in [("demo-07", 11), ("demo-05", 3), ("demo-12", 5), ("demo-18", 7), ("demo-03", 2)]:
            y = hum_query(song_id, seed)
            r = recognize("hum", wav_bytes(y))
            if r["status"] == "match" and r["results"][0]["song_id"] == song_id and r["decision"].get("confidence") == "high":
                hum, hum_audio, hum_song = r, y, song_id
                break
        if hum is None:
            raise SystemExit("No hummed example was recognised with high confidence; try other seeds.")
        clip = recognize("clip", wav_bytes(clip_query("demo-14", 18.0, 8.0, 1.25, seed=4)))
        lyrics = recognize("lyrics", text="he keeps a ledger writen in the lite")
        for name, r in [("hum", hum), ("clip", clip), ("lyrics", lyrics)]:
            print(f"{name:7s} {r['status']:9s} {r['results'][0]['song']['title'] if r['results'] else '-'}")

        examples = {"hum": hum["id"], "whistle": hum["id"], "auto": hum["id"], "combined": hum["id"],
                    "clip": clip["id"], "lyrics": lyrics["id"]}
        save("examples", examples)
        for rid in dict.fromkeys(examples.values()):
            save(f"recognitions/{rid}", without_real_songs(get(f"/api/recognitions/{rid}")))
        history = [h for h in get("/api/recognitions?limit=50") if h["id"] in examples.values()]
        save("recognitions", history)

        analysis = c.post("/api/audio/analyze", data={"profile": "hum", "extractor": "pyin", "song_id": hum_song},
                          files={"audio": ("query.wav", wav_bytes(hum_audio), "audio/wav")}).raise_for_status().json()
        save("audio/analyze", analysis)
    finally:
        c.put("/api/settings", json={"online_search_enabled": user_settings["online_search_enabled"]})
        for rid in made:
            c.delete(f"/api/recognitions/{rid}")

    # --- demo audio as small MP3s ---------------------------------------------------------------------------
    (OUT / "audio").mkdir()
    for s in songs["items"]:
        res = c.get(f"/api/songs/{s['id']}/audio")
        if res.status_code == 404:
            continue
        wav = res.raise_for_status().content
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "22050", "-b:a", args.bitrate,
                        str(OUT / "audio" / f"{s['id']}.mp3")], input=wav, check=True)

    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"Wrote {OUT.relative_to(ROOT)} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
