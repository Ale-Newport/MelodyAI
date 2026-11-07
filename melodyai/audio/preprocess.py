"""Conservative preprocessing and audio quality estimation.

The goal is to *not* destroy melodic information: we only remove DC / rumble with a
gentle high-pass filter, trim leading/trailing silence and apply peak normalisation.
No spectral-subtraction denoising is applied by default because it creates musical
noise artefacts that confuse pitch trackers on short queries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal

from melodyai.audio.io import Audio


@dataclass
class AudioQuality:
    duration_s: float
    active_duration_s: float
    rms_dbfs: float
    peak_dbfs: float
    clipping_ratio: float
    snr_db: float
    silent: bool

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-10)))


def frame_rms(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)


def estimate_quality(y: np.ndarray, sr: int, silence_db: float = -50.0) -> AudioQuality:
    """Simple, explainable quality indicators.

    SNR is estimated as the ratio between the 90th and 10th percentile of frame energy,
    which is a rough proxy (it assumes the quietest frames contain only background).
    """
    frame, hop = int(0.03 * sr), int(0.01 * sr)
    rms = frame_rms(y, frame, hop)
    rms_db = 20 * np.log10(rms + 1e-10)
    active = rms_db > max(silence_db, np.max(rms_db) - 40.0)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    loud = float(np.percentile(rms, 90)) if len(rms) else 0.0
    quiet = float(np.percentile(rms, 10)) if len(rms) else 0.0
    return AudioQuality(
        duration_s=len(y) / sr,
        active_duration_s=float(active.sum() * hop / sr),
        rms_dbfs=_db(float(np.sqrt(np.mean(y**2))) if len(y) else 0.0),
        peak_dbfs=_db(peak),
        clipping_ratio=float(np.mean(np.abs(y) > 0.999)) if len(y) else 0.0,
        snr_db=_db(loud) - _db(quiet),
        silent=bool(peak < 10 ** (silence_db / 20) or not active.any()),
    )


def highpass(y: np.ndarray, sr: int, cutoff: float = 60.0) -> np.ndarray:
    sos = signal.butter(2, cutoff, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y).astype(np.float32) if len(y) > 30 else y


def trim_silence(y: np.ndarray, sr: int, top_db: float = 40.0, pad_s: float = 0.1) -> tuple[np.ndarray, float]:
    """Trim leading/trailing silence relative to the loudest frame. Returns (y, offset_s)."""
    frame, hop = int(0.03 * sr), int(0.01 * sr)
    rms = frame_rms(y, frame, hop)
    if not len(rms):
        return y, 0.0
    db = 20 * np.log10(rms + 1e-10)
    active = np.where(db > db.max() - top_db)[0]
    if not len(active):
        return y, 0.0
    start = max(0, active[0] * hop - int(pad_s * sr))
    end = min(len(y), active[-1] * hop + frame + int(pad_s * sr))
    return y[start:end], start / sr


def normalize_peak(y: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak < 1e-6:
        return y
    return (y * (10 ** (target_dbfs / 20) / peak)).astype(np.float32)


@dataclass
class PreprocessResult:
    audio: Audio
    quality: AudioQuality
    trim_offset_s: float


def preprocess(audio: Audio, hp_cutoff: float = 60.0, trim: bool = True) -> PreprocessResult:
    """Validation-friendly preprocessing pipeline (mono/resampling already done by decode)."""
    y = audio.samples.astype(np.float32)
    y = y - float(np.mean(y)) if len(y) else y
    quality = estimate_quality(y, audio.sr)
    offset = 0.0
    if not quality.silent:
        y = highpass(y, audio.sr, hp_cutoff)
        if trim:
            y, offset = trim_silence(y, audio.sr)
        y = normalize_peak(y)
    return PreprocessResult(Audio(y, audio.sr, audio.source_format), quality, offset)
