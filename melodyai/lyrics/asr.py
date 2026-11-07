"""Speech-to-text with faster-whisper (optional dependency).

CTranslate2 (the faster-whisper backend) supports CPU and CUDA but not Apple MPS, so
on Apple Silicon inference runs on CPU with int8 weights. Models are downloaded from
the Hugging Face Hub on first use; :func:`model_status` reports whether a model is
already cached, so the application never downloads anything implicitly.

Guarding against hallucinations on non-speech input (humming, whistling, noise):
- Silero VAD filter (bundled with faster-whisper) drops non-speech regions;
- segments with high ``no_speech_prob`` or very low ``avg_logprob`` are discarded;
- a small list of well-known Whisper hallucination phrases is rejected;
- at least two alphabetic words must remain, otherwise ``speech_detected = False``.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

log = logging.getLogger(__name__)

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3")
HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching", "you", "bye", "gracias",
    "subtitulos realizados por la comunidad de amara org", "gracias por ver el video", "musica", "music",
    "please subscribe", "suscribete",
}


@dataclass
class Transcript:
    text: str
    language: str | None
    language_probability: float
    speech_detected: bool
    reason: str = ""
    segments: list[dict] = field(default_factory=list)
    model: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def reliability(self) -> float:
        """Heuristic [0,1] transcription reliability (NOT a calibrated probability)."""
        if not self.speech_detected or not self.segments:
            return 0.0
        lp = np.mean([s["avg_logprob"] for s in self.segments])
        return float(np.clip(1.0 + lp, 0.0, 1.0) * np.clip(self.language_probability, 0.3, 1.0))


def asr_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def model_status(size: str) -> dict:
    status = {"name": size, "installed_package": asr_available(), "cached": False}
    if not status["installed_package"]:
        return status
    try:
        from faster_whisper.utils import download_model

        download_model(size, local_files_only=True)
        status["cached"] = True
    except Exception:
        status["cached"] = False
    return status


class WhisperTranscriber:
    _models: dict[tuple[str, str], object] = {}
    _lock = threading.Lock()

    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "int8",
                 allow_download: bool = False) -> None:
        if model_size not in MODEL_SIZES:
            raise ValueError(f"Unknown Whisper model size '{model_size}'")
        self.model_size = model_size
        self.device = "cuda" if device == "cuda" else "cpu"
        if device == "auto":
            try:
                import ctranslate2

                self.device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                self.device = "cpu"
        self.compute_type = compute_type
        self.allow_download = allow_download

    def _model(self):
        key = (self.model_size, self.device)
        with self._lock:
            if key not in self._models:
                from faster_whisper import WhisperModel

                if not self.allow_download and not model_status(self.model_size)["cached"]:
                    raise RuntimeError(
                        f"Whisper model '{self.model_size}' is not downloaded. Run "
                        f"`make asr-model MODEL={self.model_size}` or enable downloads in Settings.")
                self._models[key] = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            return self._models[key]

    def transcribe(self, y: np.ndarray, sr: int, language: str | None = None) -> Transcript:
        if sr != 16000:
            from melodyai.audio.io import resample

            y = resample(y, sr, 16000)
        t0 = time.perf_counter()
        model = self._model()
        segments, info = model.transcribe(
            y.astype(np.float32), language=language, beam_size=5, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400}, condition_on_previous_text=False,
            no_speech_threshold=0.6, temperature=0.0,
        )
        kept, dropped = [], []
        for s in segments:
            seg = {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                   "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3),
                   "compression_ratio": round(s.compression_ratio, 3)}
            norm = re.sub(r"[^\w\s]", "", seg["text"].lower()).strip()
            if (s.no_speech_prob > 0.6 and s.avg_logprob < -0.5) or s.avg_logprob < -1.2 \
                    or s.compression_ratio > 2.6 or norm in HALLUCINATIONS:
                dropped.append(seg)
            else:
                kept.append(seg)
        text = " ".join(s["text"] for s in kept).strip()
        words = re.findall(r"[^\W\d_]+", text)
        detected = len(words) >= 2
        reason = "" if detected else ("no speech segments" if not kept else "too few words")
        return Transcript(
            text if detected else "", info.language if detected else None,
            float(info.language_probability), detected, reason, kept, self.model_size,
            time.perf_counter() - t0,
        )
