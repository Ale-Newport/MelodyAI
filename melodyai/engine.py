"""Unified recognition engine (Auto / Clip / Hum / Whistle / Lyrics / Combined).

This module is independent of the web server: it can be used from Python, the CLI,
the evaluation harness and the FastAPI backend alike.

Local engines: audio fingerprint (recordings of a catalog track), melody (hum / whistle / sing),
lyrics (typed text, or a Whisper transcript of the recording) over the local catalog.

Open-world search (optional, :class:`melodyai.world.search.WorldSearch`): the local catalog only holds
the songs that were imported, but people sing *words* of real songs. Whenever a query text exists --
typed, or transcribed from the recording (Auto / Lyrics / Combined, and Clip / Auto recordings that the
local fingerprint does not know, since a trend clip usually has sung words) -- the world search starts
in a background thread as soon as that text is known, so it overlaps with melody matching and the local
lyrics search, and is joined before the resolution. It searches the local world lyrics index offline and,
when ``online`` (the user setting), LRCLIB / Deezer with the words only (never audio). Its result is
always reported as ``engines.world`` (with its own abstention ``decision``) and wins the resolution as
``resolved_mode = "lyrics_online"`` by the rules of :meth:`RecognitionEngine.prefer_world`:

a. a local ``match`` backed by lyrics or fingerprint evidence wins (the world list stays in ``engines.world``
   so the UI can say "also found online");
b. else a world ``match`` wins;
c. else a local melody-only ``match`` wins;
d. else (both uncertain / no match) the world list wins when its decision is at least as good as the local
   one and its top score is at least the local top score.

Open-world melodies (optional, :class:`melodyai.world.melodies.WorldMelodyIndex`, built from Lakh MIDI): a
hummed or whistled query has no words, so its melody is also searched among the vocal lines of real songs.
The result is reported as ``engines.world_melody`` (decision kind ``"world_melody"``) and is considered after
the answers above, by :meth:`RecognitionEngine.prefer_world_melody`: only when neither a local match nor a
world lyrics match exists, it wins as ``resolved_mode = "melody_online"`` with a ``match``, or with an
``uncertain`` decision when the answer so far is no better.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

from melodyai.audio.io import Audio
from melodyai.features.extract import QueryFeatures, analyze_query
from melodyai.fusion.fusion import EngineOutput, decide, fuse, load_calibration
from melodyai.indexing.melody_index import MelodyIndex
from melodyai.lyrics.asr import WhisperTranscriber, asr_available, model_status, resolve_model
from melodyai.lyrics.search import LyricsIndex
from melodyai.matching.algorithms import ALGORITHMS
from melodyai.matching.ensemble import MelodyEnsemble
from melodyai.matching.explain import explain_alignment

log = logging.getLogger(__name__)

MODES = ("auto", "clip", "hum", "whistle", "lyrics", "combined")

NO_MATCH_MESSAGE = "No sufficiently reliable match was found."

STATUS_RANK = {"no_match": 0, "uncertain": 1, "match": 2}
WORLD_TOP_K = 5              # world candidates requested (at least): the decision needs a runner-up for its margin
WORLD_JOIN_GRACE_S = 5.0     # wait beyond WorldSearch.deadline_s (metadata enrichment, learning) before giving up
WORLD_SONG_SOURCE = "LRCLIB + Deezer"
WORLD_MELODY_SOURCE = "Lakh MIDI (CC-BY 4.0)"


@dataclass
class EngineSettings:
    pitch_extractor: str = "pyin"
    melody_algorithm: str = "ensemble"
    fusion_method: str = "weighted"
    asr_enabled: bool = True
    asr_model: str = "auto"  # "auto" = best cached Whisper model (melodyai.lyrics.asr.AUTO_PREFERENCE)
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


# ------------------------------------------------------------------ open-world search helpers
def world_query_text(transcript) -> str:
    """World-search query of a transcript: one sung phrase per line (Whisper segments become the alignment's
    chunk boundaries, robust to skipped / reordered lines), else the plain transcript text."""
    segments = [str(s.get("text") or "").strip() for s in (getattr(transcript, "segments", None) or [])]
    segments = [s for s in segments if s]
    return "\n".join(segments) if segments else (getattr(transcript, "text", "") or "").strip()


def world_stub(status: str, text: str, online: bool, message: str = "", elapsed_s: float = 0.0,
               offline_reason: str = "") -> dict:
    """A ``WorldResult.to_dict()``-shaped answer when no search ran (disabled) or it failed / timed out."""
    return {"status": status, "query_text": text, "online": online, "offline_reason": offline_reason,
            "elapsed_s": round(elapsed_s, 3), "message": message, "sources": {}, "candidates": []}


def world_song(c: dict) -> dict:
    """The result ``song`` of a world candidate (open-world songs are not in the catalog)."""
    return {"id": c["song_id"], "title": c.get("title"), "artist": c.get("artist"), "album": c.get("album"),
            "language": c.get("language"), "reference_type": "online", "source": WORLD_SONG_SOURCE,
            "duration_s": c.get("duration_s"), "cover_url": c.get("cover_url"), "preview_url": c.get("preview_url"),
            "deezer_id": c.get("deezer_id"), "has_audio": False, "links": dict(c.get("links") or {})}


def world_melody_item(rank: int, c: dict) -> dict:
    """A ``results`` entry for an open-world melody candidate (``rank`` is 1-based)."""
    song = world_song(c)
    song["source"] = WORLD_MELODY_SOURCE
    return {"rank": rank, "song_id": c["song_id"], "score": round(float(c["score"]), 4), "song": song,
            "match_start": c.get("start_time"), "match_end": c.get("end_time"),
            "engines": {"world_melody": {"rank": rank, "score": c["score"], "start_time": c.get("start_time"),
                                         "end_time": c.get("end_time"), "source": WORLD_MELODY_SOURCE}}}


def world_result_item(rank: int, c: dict) -> dict:
    """A ``results`` entry for a world candidate (``rank`` is 1-based)."""
    return {"rank": rank, "song_id": c["song_id"], "score": round(float(c["score"]), 4), "song": world_song(c),
            "engines": {"world": {"rank": rank, "score": c["score"], "coverage": c.get("coverage"),
                                  "matched_text": c.get("matched_text", ""), "line_range": c.get("line_range"),
                                  "sources": list(c.get("sources") or [])}}}


class WorldTask:
    """The open-world search of one recognition, running in a background thread while the local engines work.

    ``result()`` joins it (bounded by the search deadline plus a grace period) and never raises: a missing
    search (``world=None``) reports ``disabled``, a failure or a time-out ``error``.
    """

    def __init__(self, world, text: str, language: str | None, top_k: int, online: bool) -> None:
        self.text, self.online = text, bool(online)
        self._t0 = time.perf_counter()
        self._result: dict | None = None
        self._thread: threading.Thread | None = None
        if world is None:
            self._result = world_stub("disabled", text, False, "open-world search is not configured")
            return
        self.timeout = float(getattr(world, "deadline_s", 9.0) or 9.0) + WORLD_JOIN_GRACE_S
        self._thread = threading.Thread(target=self._run, args=(world, text, language, top_k),
                                        name="melodyai-world-search", daemon=True)
        self._thread.start()

    def _run(self, world, text: str, language: str | None, top_k: int) -> None:
        try:
            self._result = world.search(text, language, top_k, self.online).to_dict()
        except Exception as e:  # WorldSearch catches its own failures; this guard keeps recognition alive anyway
            log.exception("world search failed")
            self._result = world_stub("error", text, self.online, f"{type(e).__name__}: {e}"[:300],
                                      time.perf_counter() - self._t0)

    def result(self) -> dict:
        if self._thread is not None:
            self._thread.join(max(0.0, self.timeout - (time.perf_counter() - self._t0)))
            if self._thread.is_alive():
                log.warning("world search still running after %.1f s; answering without it", self.timeout)
                return world_stub("error", self.text, self.online, "timed out", time.perf_counter() - self._t0)
        if self._result is None:  # pragma: no cover - the thread always sets a result
            return world_stub("error", self.text, self.online, "no result", time.perf_counter() - self._t0)
        return self._result


class RecognitionEngine:
    def __init__(self, melody_index: MelodyIndex, lyrics_index: LyricsIndex,
                 settings: EngineSettings | None = None, fp_index=None, providers: list | None = None,
                 world=None, world_melody=None) -> None:
        self.melody_index = melody_index
        # melodyai.world.melodies.WorldMelodyIndex (optional): hummed melodies of real songs (Lakh MIDI)
        self.world_melody = world_melody
        self.providers = providers or []  # melodyai.providers (licensed external catalogs)
        self.lyrics_index = lyrics_index
        self.fp_index = fp_index  # melodyai.fingerprint.index.FingerprintIndex (optional)
        self.world = world  # melodyai.world.search.WorldSearch (optional): open-world search by the sung words
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
        try:
            model = resolve_model(self.settings.asr_model, self.settings.asr_allow_download)
        except ValueError:  # unknown model name
            return False
        if model is None:
            return False
        return self.settings.asr_allow_download or model_status(model)["cached"]

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
        # "auto" is resolved by the transcriber; Transcript.model reports the concrete model used.
        tr = WhisperTranscriber(self.settings.asr_model, allow_download=self.settings.asr_allow_download)
        return tr.transcribe(audio.samples, audio.sr, language)

    # -------------------------------------------------------------- open-world search
    def start_world(self, text: str, language: str | None, top_k: int, online: bool) -> WorldTask:
        """Start the open-world search for ``text`` in the background (``disabled`` without a WorldSearch)."""
        return WorldTask(self.world, text, language, max(top_k, WORLD_TOP_K), online)

    def lyrics_backed(self, resolved: str, ranked: list[dict], lyr_c: list[dict]) -> bool:
        """Whether the local top result has lyrics or fingerprint evidence (not only a melody match)."""
        if resolved in ("clip", "lyrics"):
            return True
        if resolved == "combined" and ranked and lyr_c and lyr_c[0]["song_id"] == ranked[0]["song_id"]:
            lyr_ranked = [{"song_id": c["song_id"], "score": c["score"]} for c in lyr_c]
            return decide(lyr_ranked, "lyrics", None, self.calibration)["status"] != "no_match"
        return False

    def prefer_world(self, resolved: str, ranked: list[dict], decision: dict, lyr_c: list[dict],
                     world: dict) -> bool:
        """Resolution between the local answer and the open-world list (rules a-d of the module docstring)."""
        candidates = world.get("candidates") or []
        if not candidates:
            return False
        local, remote = decision["status"], world["decision"]["status"]
        if local == "match" and self.lyrics_backed(resolved, ranked, lyr_c):
            return False                                                      # (a)
        if remote == "match":
            return True                                                       # (b)
        if local == "match":
            return False                                                      # (c) melody-only local match
        local_top = float(ranked[0]["score"]) if ranked else 0.0              # (d)
        return STATUS_RANK.get(remote, 0) >= STATUS_RANK.get(local, 0) and float(candidates[0]["score"]) >= local_top

    def run_world_melody(self, q: QueryFeatures, top_k: int) -> dict | None:
        """Search the hummed melody among real songs' vocal lines (None when no index is configured / built)."""
        wm = self.world_melody
        if wm is None or not getattr(wm, "built", False):
            return None
        t0 = time.perf_counter()
        if not q.informative:
            res = {"status": "insufficient_melody", "candidates": []}
        else:
            try:
                res = {"status": "ok", "candidates": [m.to_dict() for m in wm.search(q, max(top_k, WORLD_TOP_K))]}
            except Exception as e:  # noqa: BLE001 - must never break recognition
                log.exception("world melody search failed")
                res = {"status": "error", "candidates": [], "message": f"{type(e).__name__}: {e}"[:300]}
            if res["status"] == "ok" and getattr(wm, "error", ""):
                res = {"status": "error", "candidates": [], "message": wm.error}
        cands = res["candidates"]
        res["decision"] = decide([{"song_id": c["song_id"], "score": c["score"]} for c in cands], "world_melody",
                                 None, self.calibration)
        res["elapsed_s"] = round(time.perf_counter() - t0, 3)
        res["source"] = WORLD_MELODY_SOURCE
        return res

    def prefer_world_melody(self, resolved: str, decision: dict, ranked: list[dict], world_melody: dict) -> bool:
        """Open-world melodies answer only when nothing better did (see the module docstring)."""
        cands = world_melody.get("candidates") or []
        if not cands:
            return False
        local, remote = decision["status"], world_melody["decision"]["status"]
        if local == "match" or remote == "no_match":
            return False
        if remote == "match":
            return True
        top = float(ranked[0]["score"]) if ranked and resolved != "lyrics_online" else 0.0
        return local == "no_match" or (resolved not in ("lyrics_online", "lyrics", "combined", "clip")
                                       and float(cands[0]["score"]) >= top)

    # -------------------------------------------------------------- main API
    def recognize(self, audio: Audio | None = None, text: str | None = None, mode: str = "auto",
                  algorithm: str | None = None, language: str | None = None, top_k: int | None = None,
                  pitch_extractor: str | None = None, fusion_method: str | None = None,
                  explain: bool = True, use_providers: bool = False, provider_policy: str = "fallback",
                  online: bool = True) -> dict:
        """Recognise a recording and/or typed lyrics.

        ``online`` (the user's "online search" setting) lets the open-world search send the typed or
        transcribed words to online sources; the local world index is searched either way.
        """
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'. Options: {MODES}")
        if audio is None and not (text and text.strip()):
            raise ValueError("Provide audio or text")
        top_k = top_k or self.settings.top_k
        t_start = time.perf_counter()
        out: dict = {"mode": mode, "engines": {}, "warnings": []}
        q: QueryFeatures | None = None
        typed = text.strip() if text and text.strip() else None
        # Typed words are known now: search the open world while the local engines work (every mode).
        world_task: WorldTask | None = self.start_world(typed, language, top_k, online) if typed else None

        want_melody = audio is not None and mode in ("auto", "hum", "whistle", "combined")
        want_lyrics = mode in ("auto", "lyrics", "combined")

        # ---- fingerprint (recordings of the actual track: clips, TikTok edits, a song playing nearby).
        # Cheap, so in Auto it runs first; a confident match ends the search.
        fp_c: list[dict] = []
        fp_decision: dict | None = None
        if audio is not None and mode in ("auto", "clip") and self.fp_index is not None and len(self.fp_index):
            t0 = time.perf_counter()
            matches = self.fp_index.search(audio.samples, audio.sr, top_k=max(top_k, 10))
            clip_s = len(audio.samples) / audio.sr
            fp_c = [m.to_dict() for m in matches]
            for c in fp_c:
                c["end_time"] = round(c["start_time"] + clip_s * c["speed"], 3)
            out["engines"]["fingerprint"] = {"candidates": fp_c, "status": "ok" if fp_c else "no_candidates",
                                             "elapsed_s": round(time.perf_counter() - t0, 4)}
            fp_decision = decide([{"song_id": c["song_id"], "score": c["score"]} for c in fp_c], "fingerprint",
                                 None, self.calibration)
        elif mode == "clip":
            out["warnings"].append("The fingerprint index is empty: import audio references (or MIDI with previews).")
        fp_match = fp_decision is not None and fp_decision["status"] == "match"
        if fp_match:
            want_melody = want_lyrics = False
        elif mode == "clip" and audio is not None and (typed or self.asr_ready()):
            # Not a recording the local fingerprint index knows. A clip (e.g. a TikTok trend) usually contains
            # sung words: transcribe it for the local lyrics and the open-world search.
            want_lyrics = True

        # ---- lyrics (before the melody, so the open-world search on a transcript overlaps with melody matching)
        if want_lyrics:
            lyr: dict | None = None
            if typed:
                lyr = self.run_lyrics_text(text, language, top_k)
                lyr["source"] = "text"
            elif audio is not None:
                if self.asr_ready():
                    try:
                        tr = self.transcribe(audio, language)
                        lyr = {"source": "asr", "transcript": tr.to_dict(), "reliability": round(tr.reliability, 4)}
                        if tr.speech_detected:
                            query_text = world_query_text(tr)
                            if world_task is None and query_text:
                                world_task = self.start_world(query_text, language or tr.language, top_k, online)
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

        # ---- melody
        if want_melody:
            profile = mode if mode in ("hum", "whistle") else detect_profile(audio.samples, audio.sr)
            q = analyze_query(audio, profile, pitch_extractor or self.settings.pitch_extractor,
                              with_chroma=(algorithm or self.settings.melody_algorithm) == "chroma_dtw")
            mel = self.run_melody(q, algorithm, top_k)
            mel["profile"] = profile
            world_mel = self.run_world_melody(q, top_k)
            if world_mel is not None:
                out["engines"]["world_melody"] = world_mel
            mel["timings"] = {k: round(v, 4) for k, v in q.timings.items()}
            out["engines"]["melody"] = mel
            out["query"] = {
                "quality": q.quality,
                "pitch_track": q.track.to_dict(2000) if q.track is not None else None,
                "notes": q.notes.to_dict(),
                "contour": np.round(q.contour, 3).tolist(),
                "contour_rate_hz": 20.0,
            }

        # ---- resolve and fuse (local catalog)
        mel_c = out["engines"].get("melody", {}).get("candidates", [])
        lyr_c = out["engines"].get("lyrics", {}).get("candidates", [])
        alg_name = out["engines"].get("melody", {}).get("algorithm")
        use_fp = bool(fp_c) and (fp_match or mode == "clip")
        if use_fp and not fp_match and lyr_c:  # Clip mode, unsure fingerprint: the sung words may do better
            lyr_dec = decide([{"song_id": c["song_id"], "score": c["score"]} for c in lyr_c], "lyrics", None,
                             self.calibration)
            use_fp = STATUS_RANK[lyr_dec["status"]] <= STATUS_RANK[fp_decision["status"]]
        if use_fp:
            resolved = "clip"
            ranked = [{"song_id": c["song_id"], "score": c["score"], "engine_scores": {"fingerprint": c["score"]}}
                      for c in fp_c]
            decision = fp_decision
        elif mel_c and lyr_c:
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
        fp_by = {c["song_id"]: (i + 1, c) for i, c in enumerate(fp_c)} if resolved == "clip" else {}
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
            if sid in fp_by:
                rk, c = fp_by[sid]
                item["engines"]["fingerprint"] = {"rank": rk, **{k: c[k] for k in (
                    "score", "evidence", "speed", "pitch_semitones", "start_time", "end_time")}}
                item["match_start"], item["match_end"] = c["start_time"], c["end_time"]
            results.append(item)

        # ---- open world (joined only now: it ran while the local engines worked)
        if world_task is not None:
            world = world_task.result()
            world["decision"] = decide([{"song_id": c["song_id"], "score": c["score"]}
                                        for c in world.get("candidates") or []], "world", None, self.calibration)
            out["engines"]["world"] = world
            if self.prefer_world(resolved, ranked, decision, lyr_c, world):
                resolved = "lyrics_online"
                decision = dict(world["decision"])
                results = [world_result_item(i + 1, c) for i, c in enumerate(world["candidates"][:top_k])]
                out.pop("fusion", None)

        world_mel = out["engines"].get("world_melody")
        if world_mel and self.prefer_world_melody(resolved, decision, ranked, world_mel):
            resolved = "melody_online"
            decision = dict(world_mel["decision"])
            results = [world_melody_item(i + 1, c) for i, c in enumerate(world_mel["candidates"][:top_k])]
            out.pop("fusion", None)

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
        if use_providers and audio is not None:
            out["external"] = self.run_providers(audio, mode, q, decision["status"], provider_policy)
        out["elapsed_s"] = round(time.perf_counter() - t_start, 4)
        return out

    def run_providers(self, audio: Audio, mode: str, q: QueryFeatures | None, local_status: str,
                      policy: str = "fallback") -> list[dict]:
        """Query licensed external catalogs. ``fallback`` = only if the local catalog had no confident
        match (sends less audio to third parties and costs less); ``always`` = every query."""
        if policy == "fallback" and local_status == "match":
            return []
        if mode == "clip":
            kinds = ["music"]
        elif mode in ("hum", "whistle", "combined"):
            kinds = ["humming"]
        elif mode == "auto":
            kinds = ["music"] + (["humming"] if q is not None and q.informative else [])
        else:
            return []
        results = []
        for p in self.providers:
            for kind in kinds:
                if p.supports(kind):
                    try:
                        results.append(p.identify(audio.samples, audio.sr, kind).to_dict())
                    except Exception as e:  # a provider failure must never break local recognition
                        log.exception("provider %s failed", getattr(p, "name", p))
                        results.append({"provider": getattr(p, "name", "?"), "kind": kind, "status": "error",
                                        "matches": [], "message": str(e)})
        return results
