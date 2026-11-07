"""Symbolic melody representation and contour utilities.

A melody is represented as a monophonic :class:`NoteSequence` (onset, duration,
fractional MIDI pitch). Matching operates on *contours*: pitch sampled on a fixed
frame grid, where rests are removed (sung queries rarely reproduce rests faithfully).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from melodyai.config import CONTOUR_RATE


@dataclass
class NoteSequence:
    onsets: np.ndarray            # seconds
    durations: np.ndarray         # seconds
    pitches: np.ndarray           # fractional MIDI numbers
    confidence: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.onsets = np.asarray(self.onsets, dtype=np.float64)
        self.durations = np.asarray(self.durations, dtype=np.float64)
        self.pitches = np.asarray(self.pitches, dtype=np.float64)
        if self.confidence is None:
            self.confidence = np.ones_like(self.pitches)
        self.confidence = np.asarray(self.confidence, dtype=np.float64)
        order = np.argsort(self.onsets, kind="stable")
        if len(order) and np.any(order != np.arange(len(order))):
            self.onsets, self.durations = self.onsets[order], self.durations[order]
            self.pitches, self.confidence = self.pitches[order], self.confidence[order]

    def __len__(self) -> int:
        return len(self.pitches)

    @property
    def end_time(self) -> float:
        return float(np.max(self.onsets + self.durations)) if len(self) else 0.0

    @classmethod
    def empty(cls) -> NoteSequence:
        return cls(np.zeros(0), np.zeros(0), np.zeros(0))

    @classmethod
    def from_list(cls, notes: list[tuple[float, float, float]]) -> NoteSequence:
        if not notes:
            return cls.empty()
        a = np.asarray(notes, dtype=np.float64)
        return cls(a[:, 0], a[:, 1], a[:, 2])

    def to_dict(self) -> dict:
        return {
            "onsets": self.onsets.round(4).tolist(),
            "durations": self.durations.round(4).tolist(),
            "pitches": self.pitches.round(3).tolist(),
            "confidence": self.confidence.round(3).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> NoteSequence:
        return cls(d["onsets"], d["durations"], d["pitches"], d.get("confidence"))

    def transpose(self, semitones: float) -> NoteSequence:
        return NoteSequence(self.onsets, self.durations, self.pitches + semitones, self.confidence)

    def time_scale(self, factor: float) -> NoteSequence:
        """Stretch time by ``factor`` (>1 = slower)."""
        return NoteSequence(self.onsets * factor, self.durations * factor, self.pitches, self.confidence)

    def slice(self, start: int, end: int) -> NoteSequence:
        sub = NoteSequence(
            self.onsets[start:end].copy(), self.durations[start:end].copy(),
            self.pitches[start:end].copy(), self.confidence[start:end].copy(),
        )
        if len(sub):
            sub.onsets -= sub.onsets[0]
        return sub

    def intervals(self) -> np.ndarray:
        return np.diff(self.pitches)

    def ioi(self) -> np.ndarray:
        """Inter-onset intervals (seconds); the last note uses its duration."""
        if not len(self):
            return np.zeros(0)
        return np.append(np.diff(self.onsets), self.durations[-1])


def notes_to_contour(notes: NoteSequence, rate: float = CONTOUR_RATE, keep_rests: bool = False) -> np.ndarray:
    """Render notes on a fixed frame grid.

    With ``keep_rests=False`` (default) each note contributes ``round(ioi*rate)`` frames,
    i.e. a note is held until the next onset, so rests are absorbed by the previous note.
    With ``keep_rests=True`` rest frames are NaN.
    """
    if not len(notes):
        return np.zeros(0)
    if keep_rests:
        n = int(np.ceil(notes.end_time * rate)) + 1
        out = np.full(n, np.nan)
        for o, d, p in zip(notes.onsets, notes.durations, notes.pitches):
            a, b = int(round(o * rate)), max(int(round((o + d) * rate)), int(round(o * rate)) + 1)
            out[a:b] = p
        return out
    lengths = np.maximum(1, np.round(notes.ioi() * rate).astype(int))
    return np.repeat(notes.pitches, lengths)


def resample_contour(c: np.ndarray, n: int) -> np.ndarray:
    """Linear time normalisation of a contour to ``n`` points (uniform scaling)."""
    if len(c) == 0:
        return np.zeros(n)
    if len(c) == 1:
        return np.full(n, c[0])
    x = np.linspace(0, len(c) - 1, n)
    return np.interp(x, np.arange(len(c)), c)


def center(c: np.ndarray) -> np.ndarray:
    """Key normalisation: subtract the median pitch (robust to outlier notes)."""
    if len(c) == 0:
        return c
    return c - np.median(c)


def fold_octave_jumps(c: np.ndarray, tol: float = 1.0, max_run: int = 8, context: int = 31) -> np.ndarray:
    """Correct short octave errors relative to a running median of the contour.

    Pitch trackers sometimes report f0/2 or 2*f0 for a few frames. Only *short*
    excursions (at most ``max_run`` frames, 80 ms at 100 frames/s) that sit 12 or 24
    +/- ``tol`` semitones away from the local median are folded back: genuine octave
    leaps in a melody last for whole notes and must be preserved.
    """
    if len(c) < 5:
        return c
    from scipy.ndimage import median_filter

    ref = median_filter(c, size=min(len(c) // 2 * 2 + 1, context), mode="nearest")
    d = c - ref
    out = c.copy()
    for k in (12.0, -12.0, 24.0, -24.0):
        mask = np.abs(d - k) < tol
        if not mask.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a <= max_run:
                out[a:b] -= k
    return out


def contour_intervals(c: np.ndarray) -> np.ndarray:
    return np.diff(c) if len(c) > 1 else np.zeros(0)


def pitch_class_chroma(midi: np.ndarray, weights: np.ndarray | None = None, sigma: float = 0.35) -> np.ndarray:
    """Soft 12-bin chroma for each frame of a (fractional) MIDI contour.

    NaN frames produce all-zero chroma vectors.
    """
    midi = np.asarray(midi, dtype=np.float64)
    out = np.zeros((len(midi), 12))
    ok = np.isfinite(midi)
    if not ok.any():
        return out
    pc = np.mod(midi[ok], 12.0)
    bins = np.arange(12)
    d = np.abs(pc[:, None] - bins[None, :])
    d = np.minimum(d, 12 - d)
    ch = np.exp(-0.5 * (d / sigma) ** 2)
    if weights is not None:
        ch *= np.asarray(weights)[ok][:, None]
    out[ok] = ch
    return out


MIDI_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(m: float) -> str:
    r = int(round(m))
    return f"{MIDI_NOTE_NAMES[r % 12]}{r // 12 - 1}"
