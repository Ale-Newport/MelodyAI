"""Unified recognition engine (Auto / Hum / Whistle / Lyrics / Combined).

This module is independent of the web server: it can be used from Python, the CLI,
the evaluation harness and the FastAPI backend alike.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass

import numpy as np

from melodyai.audio.io import Audio
from melodyai.features.extract import QueryFeatures, analyze_query
from melodyai.fusion.fusion import EngineOutput, decide, fuse, load_calibration
from melodyai.indexing.melody_index import MelodyIndex
from melodyai.lyrics.asr import WhisperTranscriber, asr_available, model_status
from melodyai.lyrics.search import LyricsIndex
from melodyai.matching.algorithms import ALGORITHMS
from melodyai.matching.ensemble import MelodyEnsemble
from melodyai.matching.explain import explain_alignment

log = logging.getLogger(__name__)

MODES = ("auto", "hum", "whistle", "lyrics", "combined")

NO_MATCH_MESSAGE = "No sufficiently reliable match was found."


@dataclass
class EngineSettings:
    pitch_extractor: str = "pyin"
    melody_algorithm: str = "ensemble"
    fusion_method: str = "weighted"
    asr_enabled: bool = True
    asr_model: str = "base"
    asr_allow_download: bool = False
    top_k: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


def default_algorithms() -> dict:
    algs = dict(ALGORITHMS)
    algs["ensemble"] = MelodyEnsemble(
        "ensemble", {"dtw": ALGORITHMS["dtw"], "note_align": ALGORITHMS["note_align"],
                     "subseq_dtw": ALGORITHMS["subseq_dtw"]},
        label="Ensemble (window DTW + note alignment + subsequence DTW)")
    return algs


def detect_profile(y: np.ndarray, sr: int) -> str:
    """'whistle' if the dominant pitch is in the whistling register (> ~700 Hz), else 'hum'."""
    import librosa

    if len(y) < sr * 0.3:
        return "hum"
    f0 = librosa.yin(y, fmin=80, fmax=min(3500, sr / 2 - 100), sr=sr, frame_length=2048, hop_length=512)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0][: len(f0)]
    loud = rms > 0.3 * (np.max(rms) + 1e-9)
    if not loud.any():
        return "hum"
    return "whistle" if float(np.median(f0[: len(loud)][loud])) > 700 else "hum"


def melody_reliability(q: QueryFeatures) -> float:
    if not q.informative:
        return 0.0
    conf = float(q.quality.get("pitch_confidence", 0.7) or 0.7)
    return float(np.clip(q.voiced_duration / 4.0, 0.2, 1.0) * np.clip(conf * 1.4, 0.2, 1.0))


class RecognitionEngine:
    def __init__(self, melody_index: MelodyIndex, lyrics_index: LyricsIndex,
                 settings: EngineSettings | None = None) -> None:
        self.melody_index = melody_index
        self.lyrics_index = lyrics_index
        self.settings = settings or EngineSettings()
        self.algorithms = default_algorithms()
        self.calibration = load_calibration()

    # -------------------------------------------------------------- registry
    def register_algorithm(self, alg) -> None:
        self.algorithms[alg.name] = alg

    def unregister_algorithm(self, name: str) -> None:
        self.algorithms.pop(name, None)

    def list_algorithms(self) -> list[dict]:
        return [{"name": n, "label": getattr(a, "label", n)} for n, a in self.algorithms.items()]

    def reload_calibration(self) -> None:
        self.calibration = load_calibration()

    def asr_ready(self) -> bool:
        if not (self.settings.asr_enabled and asr_available()):
            return False
        return self.settings.asr_allow_download or model_status(self.settings.asr_model)["cached"]

    # -------------------------------------------------------------- engines
    def run_melody(self, q: QueryFeatures, algorithm: str | None = None, top_k: int = 5) -> dict:
        name = algorithm or self.settings.melody_algorithm
        if name not in self.algorithms:
            raise ValueError(f"Unknown melody algorithm '{name}'")
        if not q.informative:
            return {"algorithm": name, "candidates": [], "status": "insufficient_melody",
                    "reason": "Not enough stable pitched content (need ~1.5 s of voiced melody with >= 3 notes).",
                    "reliability": 0.0, "elapsed_s": 0.0}
        t0 = time.perf_counter()
        cands = self.algorithms[name].search(q, self.melody_index, max(top_k, 10))
        elapsed = time.perf_counter() - t0
        return {"algorithm": name, "candidates": [c.to_dict() for c in cands], "status": "ok",
                "reliability": round(melody_reliability(q), 4), "elapsed_s": round(elapsed, 4)}

    def run_lyrics_text(self, text: str, language: str | None = None, top_k: int = 5) -> dict:
        t0 = time.perf_counter()
        cands = self.lyrics_index.search(text, max(top_k, 10), language)
        return {"query_text": text, "candidates": [c.to_dict() for c in cands],
                "status": "ok" if cands else "no_candidates", "elapsed_s": round(time.perf_counter() - t0, 4)}

    def transcribe(self, audio: Audio, language: str | None = None):
        tr = WhisperTranscriber(self.settings.asr_model, allow_download=self.settings.asr_allow_download)
        return tr.transcribe(audio.samples, audio.sr, language)

    # -------------------------------------------------------------- main API
    def recognize(self, audio: Audio | None = None, text: str | None = None, mode: str = "auto",
                  algorithm: str | None = None, language: str | None = None, top_k: int | None = None,
                  pitch_extractor: str | None = None, fusion_method: str | None = None,
                  explain: bool = True) -> dict:
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'. Options: {MODES}")
        if audio is None and not (text and text.strip()):
            raise ValueError("Provide audio or text")
        top_k = top_k or self.settings.top_k
        t_start = time.perf_counter()
        out: dict = {"mode": mode, "engines": {}, "warnings": []}
        q: QueryFeatures | None = None

        want_melody = audio is not None and mode in ("auto", "hum", "whistle", "combined")
        want_lyrics = mode in ("auto", "lyrics", "combined")

        # ---- melody
        if want_melody:
            profile = mode if mode in ("hum", "whistle") else detect_profile(audio.samples, audio.sr)
            q = analyze_query(audio, profile, pitch_extractor or self.settings.pitch_extractor,
                              with_chroma=(algorithm or self.settings.melody_algorithm) == "chroma_dtw")
            mel = self.run_melody(q, algorithm, top_k)
            mel["profile"] = profile
            mel["timings"] = {k: round(v, 4) for k, v in q.timings.items()}
            out["engines"]["melody"] = mel
            out["query"] = {
                "quality": q.quality,
                "pitch_track": q.track.to_dict(2000) if q.track is not None else None,
                "notes": q.notes.to_dict(),
                "contour": np.round(q.contour, 3).tolist(),
                "contour_rate_hz": 20.0,
            }

        # ---- lyrics
        if want_lyrics:
            lyr: dict | None = None
            if text and text.strip():
                lyr = self.run_lyrics_text(text, language, top_k)
                lyr["source"] = "text"
            elif audio is not None:
                if self.asr_ready():
                    try:
                        tr = self.transcribe(audio, language)
                        lyr = {"source": "asr", "transcript": tr.to_dict(), "reliability": round(tr.reliability, 4)}
                        if tr.speech_detected:
                            lyr.update(self.run_lyrics_text(tr.text, language or tr.language, top_k))
                        else:
                            lyr.update({"candidates": [], "status": "no_speech", "reason": tr.reason})
                    except Exception as e:  # ASR failure must not break melody recognition
                        log.exception("ASR failed")
                        lyr = {"source": "asr", "candidates": [], "status": "error", "reason": str(e)}
                else:
                    lyr = {"source": "asr", "candidates": [], "status": "unavailable",
                           "reason": "Speech recognition model not available (install/download a Whisper "
                                     "model, or type the lyrics)."}
                    if mode == "lyrics":
                        out["warnings"].append(lyr["reason"])
            if lyr is not None:
                lyr.setdefault("reliability", 1.0 if lyr.get("source") == "text" else 0.0)
                out["engines"]["lyrics"] = lyr

        # ---- resolve and fuse
        mel_c = out["engines"].get("melody", {}).get("candidates", [])
        lyr_c = out["engines"].get("lyrics", {}).get("candidates", [])
        alg_name = out["engines"].get("melody", {}).get("algorithm")
        if mel_c and lyr_c:
            resolved = "combined"
            method = fusion_method or self.settings.fusion_method
            outputs = [
                EngineOutput("melody", mel_c, out["engines"]["melody"]["reliability"]),
                EngineOutput("lyrics", lyr_c, max(out["engines"]["lyrics"].get("reliability", 1.0), 0.3)),
            ]
            ranked = fuse(outputs, method)
            decision = decide(ranked, "fused", method, self.calibration)
            out["fusion"] = {"method": method}
        elif mel_c:
            resolved = "whistle" if out["engines"]["melody"].get("profile") == "whistle" else "hum"
            ranked = [{"song_id": c["song_id"], "score": c["score"], "engine_scores": {"melody": c["score"]}}
                      for c in mel_c]
            decision = decide(ranked, "melody", alg_name, self.calibration)
        elif lyr_c:
            resolved = "lyrics"
            ranked = [{"song_id": c["song_id"], "score": c["score"], "engine_scores": {"lyrics": c["score"]}}
                      for c in lyr_c]
            decision = decide(ranked, "lyrics", None, self.calibration)
        else:
            resolved = mode if mode != "auto" else "none"
            ranked = []
            decision = {"status": "no_match", "reason": "No engine produced candidates"}

        mel_by = {c["song_id"]: (i + 1, c) for i, c in enumerate(mel_c)}
        lyr_by = {c["song_id"]: (i + 1, c) for i, c in enumerate(lyr_c)}
        results = []
        for i, r in enumerate(ranked[:top_k]):
            sid = r["song_id"]
            item = {"rank": i + 1, "song_id": sid, "score": round(float(r["score"]), 4), "engines": {}}
            if sid in mel_by:
                rk, c = mel_by[sid]
                item["engines"]["melody"] = {"rank": rk, "score": c["score"], "distance": c["distance"],
                                             "start_time": c["start_time"], "end_time": c["end_time"]}
                item["match_start"], item["match_end"] = c["start_time"], c["end_time"]
            if sid in lyr_by:
                rk, c = lyr_by[sid]
                item["engines"]["lyrics"] = {"rank": rk, "score": c["score"], "matched_text": c["matched_text"],
                                             "line_range": c["line_range"]}
            results.append(item)
        out["resolved_mode"] = resolved
        out["decision"] = decision
        out["status"] = decision["status"]
        out["message"] = NO_MATCH_MESSAGE if decision["status"] == "no_match" else (
            "Possible match, low certainty." if decision["status"] == "uncertain" else "Match found.")
        out["results"] = results
        out["score_kind"] = "calibrated_probability" if "calibrated_probability" in decision else "relative"

        if explain and q is not None and results and results[0]["song_id"] in self.melody_index:
            top = results[0]
            ref = self.melody_index.refs[top["song_id"]]
            try:
                out["alignment"] = explain_alignment(q, ref, top.get("match_start"), top.get("match_end"))
            except Exception:  # visualisation must never break recognition
                log.exception("alignment explanation failed")
        out["elapsed_s"] = round(time.perf_counter() - t_start, 4)
        return out
