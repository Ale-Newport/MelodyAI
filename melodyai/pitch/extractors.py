"""f0 extractor implementations.

- ``pyin``: librosa pYIN (Mauch & Dixon, 2014). Baseline, pure Python/numba, always available.
- ``basic_pitch``: Spotify Basic Pitch (polyphonic note transcription; we derive a frame
  f0 from its note events with a skyline rule). Optional dependency.
- ``crepe``: torchcrepe (Kim et al., 2018) "tiny" model. Optional dependency; weights
  ship with the pip package, no download at runtime.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from functools import lru_cache

import numpy as np

from melodyai.config import PITCH_HOP_S, torch_device
from melodyai.pitch.base import PROFILES, PitchTrack

log = logging.getLogger(__name__)


class PyinExtractor:
    name = "pyin"
    label = "pYIN (librosa)"

    def available(self) -> bool:
        return True

    def extract(self, y: np.ndarray, sr: int, profile: str = "hum") -> PitchTrack:
        import librosa

        fmin, fmax = PROFILES.get(profile, PROFILES["auto"])
        fmax = min(fmax, sr / 2 * 0.95)
        hop = int(round(PITCH_HOP_S * sr))
        # Frame long enough for ~2 periods of fmin, power of two for FFT speed.
        frame = int(2 ** np.ceil(np.log2(max(2.2 * sr / fmin, 512))))
        t0 = time.perf_counter()
        f0, voiced, prob = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame, hop_length=hop,
            center=True, resolution=0.1, max_transition_rate=35.92, switch_prob=0.01,
        )
        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
        voiced = np.asarray(voiced, dtype=bool) & np.isfinite(f0)
        return PitchTrack(
            times, f0.astype(np.float64), np.nan_to_num(prob).astype(np.float64), voiced, self.name, hop / sr,
            {"fmin": fmin, "fmax": fmax, "frame_length": frame, "elapsed_s": time.perf_counter() - t0},
        )


class BasicPitchExtractor:
    name = "basic_pitch"
    label = "Basic Pitch (Spotify)"

    def available(self) -> bool:
        return importlib.util.find_spec("basic_pitch") is not None

    @staticmethod
    @lru_cache(maxsize=1)
    def _model():
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model

        return Model(ICASSP_2022_MODEL_PATH)

    def notes(self, y: np.ndarray, sr: int, profile: str = "reference"):
        """Run Basic Pitch and return its note events (start, end, midi, amplitude)."""
        import tempfile

        import soundfile as sf
        from basic_pitch.inference import predict

        fmin, fmax = PROFILES.get(profile, PROFILES["auto"])
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, y, sr)
            _, _, events = predict(
                tmp.name, self._model(), minimum_frequency=fmin, maximum_frequency=fmax,
                minimum_note_length=80, onset_threshold=0.5, frame_threshold=0.3,
            )
        return sorted([(float(s), float(e), float(p), float(a)) for s, e, p, a, *_ in events])

    def extract(self, y: np.ndarray, sr: int, profile: str = "hum") -> PitchTrack:
        t0 = time.perf_counter()
        events = self.notes(y, sr, profile)
        hop = PITCH_HOP_S
        n = int(np.ceil(len(y) / sr / hop)) + 1
        times = np.arange(n) * hop
        f0 = np.full(n, np.nan)
        conf = np.zeros(n)
        # Skyline: when notes overlap keep the highest one (predominant-melody heuristic).
        for s, e, p, a in events:
            a_i, b_i = int(s / hop), min(n, int(np.ceil(e / hop)))
            hz = 440.0 * 2 ** ((p - 69) / 12)
            seg = slice(a_i, b_i)
            take = ~(f0[seg] > hz)
            f0[seg] = np.where(take, hz, f0[seg])
            conf[seg] = np.where(take, a, conf[seg])
        voiced = np.isfinite(f0)
        return PitchTrack(times, f0, np.clip(conf, 0, 1), voiced, self.name, hop,
                          {"elapsed_s": time.perf_counter() - t0, "n_events": len(events)})


class CrepeExtractor:
    name = "crepe"
    label = "CREPE tiny (torchcrepe)"

    def available(self) -> bool:
        return importlib.util.find_spec("torchcrepe") is not None and importlib.util.find_spec("torch") is not None

    def extract(self, y: np.ndarray, sr: int, profile: str = "hum") -> PitchTrack:
        import torch
        import torchcrepe

        fmin, fmax = PROFILES.get(profile, PROFILES["auto"])
        t0 = time.perf_counter()
        audio = torch.tensor(y, dtype=torch.float32)[None]
        if sr != torchcrepe.SAMPLE_RATE:
            import librosa

            audio = torch.tensor(librosa.resample(y, orig_sr=sr, target_sr=torchcrepe.SAMPLE_RATE))[None]
        hop = int(torchcrepe.SAMPLE_RATE * PITCH_HOP_S)
        device = torch_device()
        device = "cpu" if device == "mps" else device  # some torchcrepe ops are unstable on MPS
        f0, periodicity = torchcrepe.predict(
            audio, torchcrepe.SAMPLE_RATE, hop, fmin, min(fmax, 1997.0), "tiny",
            batch_size=512, device=device, return_periodicity=True,
        )
        periodicity = torchcrepe.filter.median(periodicity, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0 = f0[0].cpu().numpy().astype(np.float64)
        p = periodicity[0].cpu().numpy().astype(np.float64)
        voiced = p > 0.35
        f0[~voiced] = np.nan
        times = np.arange(len(f0)) * PITCH_HOP_S
        return PitchTrack(times, f0, p, voiced, self.name, PITCH_HOP_S, {"elapsed_s": time.perf_counter() - t0})


EXTRACTORS = {e.name: e for e in (PyinExtractor(), BasicPitchExtractor(), CrepeExtractor())}


def get_extractor(name: str = "pyin"):
    if name not in EXTRACTORS:
        raise ValueError(f"Unknown pitch extractor '{name}'. Options: {list(EXTRACTORS)}")
    ext = EXTRACTORS[name]
    if not ext.available():
        raise RuntimeError(f"Pitch extractor '{name}' is not installed (pip install 'melodyai[pitch]')")
    return ext


def list_extractors() -> list[dict]:
    return [{"name": e.name, "label": e.label, "available": e.available()} for e in EXTRACTORS.values()]
