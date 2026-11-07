"""Audio decoding and validation.

Decoding strategy:
1. ``soundfile`` (libsndfile) handles WAV, FLAC, OGG/Vorbis/Opus and MP3 in memory.
2. If that fails and ``ffmpeg`` is on PATH, it is used as a sandboxed decoder via
   stdin/stdout pipes (M4A/AAC, WebM/Opus from browsers, ...). No file paths from
   the client are ever passed to ffmpeg; only the uploaded bytes.

All audio is returned as mono float32 in [-1, 1] at the requested sample rate.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from melodyai.config import ANALYSIS_SR

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".mp4", ".aac", ".webm"}


class AudioError(ValueError):
    """Raised for invalid, empty, undecodable or out-of-limits audio."""


@dataclass
class Audio:
    samples: np.ndarray  # mono float32
    sr: int
    source_format: str = "unknown"

    @property
    def duration(self) -> float:
        return float(len(self.samples)) / float(self.sr) if self.sr else 0.0


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _decode_soundfile(data: bytes) -> tuple[np.ndarray, int, str]:
    with sf.SoundFile(io.BytesIO(data)) as f:
        y = f.read(dtype="float32", always_2d=True)
        return y, f.samplerate, (f.format or "unknown").lower()


def _decode_ffmpeg(data: bytes, sr: int, timeout: float = 60.0) -> np.ndarray:
    if not ffmpeg_available():
        raise AudioError("Unsupported or corrupt audio (ffmpeg not available for fallback decoding)")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", "pipe:0", "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:  # pragma: no cover
        raise AudioError("Audio decoding timed out") from e
    if proc.returncode != 0 or not proc.stdout:
        raise AudioError("Could not decode audio: unsupported or corrupt file")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq").astype(
        np.float32
    )


def decode_audio(
    data: bytes,
    sr: int = ANALYSIS_SR,
    max_bytes: int | None = 25 * 1024 * 1024,
    max_duration: float | None = None,
    min_duration: float = 0.05,
) -> Audio:
    """Decode an in-memory audio file to mono float32 at ``sr``.

    Raises ``AudioError`` for empty, oversized, undecodable or too short/long input.
    """
    if not data:
        raise AudioError("Empty audio file")
    if max_bytes is not None and len(data) > max_bytes:
        raise AudioError(f"Audio file too large ({len(data) / 1e6:.1f} MB > {max_bytes / 1e6:.0f} MB)")
    fmt = "unknown"
    try:
        y2, orig_sr, fmt = _decode_soundfile(data)
        y = y2.mean(axis=1) if y2.shape[1] > 1 else y2[:, 0]
        y = resample(y, orig_sr, sr)
    except (RuntimeError, sf.LibsndfileError, TypeError, ValueError):
        y = _decode_ffmpeg(data, sr)
        fmt = "ffmpeg"
    y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    audio = Audio(samples=np.clip(y, -1.0, 1.0), sr=sr, source_format=fmt)
    if audio.duration < min_duration:
        raise AudioError("Audio is empty or too short")
    if max_duration is not None and audio.duration > max_duration + 0.5:
        raise AudioError(f"Audio too long ({audio.duration:.1f}s > {max_duration:.0f}s)")
    return audio


def load_audio(path: str | Path, sr: int = ANALYSIS_SR, **kwargs) -> Audio:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(f"Unsupported audio extension: {path.suffix}")
    return decode_audio(path.read_bytes(), sr=sr, **kwargs)


def write_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(y, -1, 1).astype(np.float32), sr, subtype="PCM_16")


def wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
