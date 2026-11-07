"""Common interface for f0 extractors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

# Frequency ranges (Hz) tuned per query type. Whistling sits roughly 1-2 octaves
# above sung/hummed melodies, so a dedicated range avoids sub-harmonic errors.
PROFILES: dict[str, tuple[float, float]] = {
    "hum": (65.0, 1400.0),
    "sing": (65.0, 1400.0),
    "whistle": (400.0, 4000.0),
    "auto": (65.0, 2100.0),
    "reference": (50.0, 2100.0),
}


@dataclass
class PitchTrack:
    times: np.ndarray        # frame centres (s)
    f0_hz: np.ndarray        # NaN where unvoiced
    voiced_prob: np.ndarray  # [0, 1]
    voiced: np.ndarray       # bool
    extractor: str
    hop_s: float
    meta: dict = field(default_factory=dict)

    @property
    def midi(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            m = 69.0 + 12.0 * np.log2(self.f0_hz / 440.0)
        m[~self.voiced] = np.nan
        return m

    @property
    def voiced_ratio(self) -> float:
        return float(np.mean(self.voiced)) if len(self.voiced) else 0.0

    @property
    def voiced_duration(self) -> float:
        return float(np.sum(self.voiced) * self.hop_s)

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.voiced_prob[self.voiced])) if self.voiced.any() else 0.0

    def to_dict(self, max_points: int = 4000) -> dict:
        step = max(1, int(np.ceil(len(self.times) / max_points)))
        midi = self.midi[::step]
        return {
            "extractor": self.extractor,
            "hop_s": self.hop_s * step,
            "times": self.times[::step].round(4).tolist(),
            "f0_hz": [None if not np.isfinite(v) else round(float(v), 2) for v in self.f0_hz[::step]],
            "midi": [None if not np.isfinite(v) else round(float(v), 3) for v in midi],
            "voiced_prob": self.voiced_prob[::step].round(3).tolist(),
            "voiced": self.voiced[::step].astype(int).tolist(),
        }


class PitchExtractor(Protocol):
    name: str

    def available(self) -> bool: ...

    def extract(self, y: np.ndarray, sr: int, profile: str = "hum") -> PitchTrack: ...


def clean_track(track: PitchTrack, min_prob: float = 0.15, min_run_s: float = 0.05) -> PitchTrack:
    """Drop low-confidence frames and very short voiced islands (breaths, clicks)."""
    voiced = track.voiced & np.isfinite(track.f0_hz) & (track.voiced_prob >= min_prob)
    min_run = max(1, int(round(min_run_s / track.hop_s)))
    v = voiced.astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], v, [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < min_run:
            voiced[a:b] = False
    f0 = track.f0_hz.copy()
    f0[~voiced] = np.nan
    return PitchTrack(track.times, f0, track.voiced_prob, voiced, track.extractor, track.hop_s, dict(track.meta))
