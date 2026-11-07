"""Lightweight additive synthesiser for the demo catalog and synthetic queries.

Timbres:
- ``piano``   decaying harmonics, separate attacks per note (reference rendering)
- ``sine``    pure tone
- ``hum``     glottal-like harmonic source through a nasal "mmm" filter, legato, vibrato
- ``voice``   harmonic source with "ah" vowel formants (sung query without words)
- ``whistle`` near-sinusoidal with band-limited breath noise
- ``bass`` / ``pad`` accompaniment timbres for polyphonic references

Everything is deterministic given ``seed``. This is not meant to sound realistic,
but to exercise pitch tracking with plausible spectral structure, legato glides,
vibrato, jitter and breath noise.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from melodyai.features.melody import NoteSequence

FORMANTS = {
    "hum": [(250.0, 60.0, 1.0), (2200.0, 300.0, 0.05)],             # nasal murmur
    "voice": [(730.0, 90.0, 1.0), (1090.0, 110.0, 0.5), (2440.0, 170.0, 0.25)],  # "ah"
}


def midi_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def _formant_gain(freq: np.ndarray, formants) -> np.ndarray:
    g = np.zeros_like(freq)
    for fc, bw, amp in formants:
        g += amp / (1.0 + ((freq - fc) / bw) ** 2)
    return 0.15 + g


def _legato_f0(notes: NoteSequence, sr: int, n: int, glide_s: float, rng, vibrato_cents: float,
               vibrato_hz: float, jitter_cents: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample f0 (Hz) and voicing gate for a legato voice-like rendering."""
    f0 = np.zeros(n)
    gate = np.zeros(n)
    t = np.arange(n) / sr
    for i, (o, d, p) in enumerate(zip(notes.onsets, notes.durations, notes.pitches)):
        a, b = int(o * sr), min(n, int((o + d) * sr))
        if b <= a:
            continue
        f0[a:b] = p
        gate[a:b] = 1.0
        # Glide from the previous note if it is (nearly) contiguous.
        if i > 0 and glide_s > 0:
            prev_end = notes.onsets[i - 1] + notes.durations[i - 1]
            if o - prev_end < 0.03:
                g = min(int(glide_s * sr), b - a)
                f0[a:a + g] = notes.pitches[i - 1] + (p - notes.pitches[i - 1]) * (
                    0.5 - 0.5 * np.cos(np.linspace(0, np.pi, g)))
        # Delayed-onset vibrato inside each note.
        if vibrato_cents > 0 and d > 0.25:
            tt = t[a:b] - o
            depth = vibrato_cents / 100.0 * np.clip((tt - 0.15) / 0.2, 0, 1)
            f0[a:b] += depth * np.sin(2 * np.pi * vibrato_hz * tt + rng.uniform(0, 2 * np.pi))
    if jitter_cents > 0:
        # Slow random drift (band-limited noise), a crude model of pitch instability.
        drift = signal.lfilter([0.002], [1, -0.998], rng.standard_normal(n))
        drift = drift / (np.std(drift) + 1e-9) * jitter_cents / 100.0
        f0 += drift
    hz = midi_to_hz(f0)
    hz[gate == 0] = 0.0
    return hz, gate


def _envelope(gate: np.ndarray, sr: int, attack: float, release: float) -> np.ndarray:
    """Smooth a 0/1 gate into an amplitude envelope (one-pole attack/release)."""
    a_up = np.exp(-1.0 / (attack * sr))
    a_dn = np.exp(-1.0 / (release * sr))
    env = np.empty_like(gate)
    v = 0.0
    for i, g in enumerate(gate):  # small loops (<= 30 s audio); fast enough with numpy scalars
        coef = a_up if g > v else a_dn
        v = coef * v + (1 - coef) * g
        env[i] = v
    return env


try:  # numba speeds up the envelope loop considerably
    from numba import njit

    _envelope = njit(cache=True)(_envelope)  # type: ignore[assignment]
except Exception:  # pragma: no cover
    pass


def _rearticulate(gate: np.ndarray, notes: NoteSequence, sr: int, dip_s: float = 0.04) -> np.ndarray:
    """Insert short dips between contiguous notes so repeated notes remain audible."""
    g = gate.copy()
    for i in range(1, len(notes)):
        o = notes.onsets[i]
        prev_end = notes.onsets[i - 1] + notes.durations[i - 1]
        if o - prev_end < 0.03:
            a = int(max(0, (o - dip_s) * sr))
            g[a:int(o * sr)] = 0.15 if abs(notes.pitches[i] - notes.pitches[i - 1]) < 0.5 else 0.6
    return g


def render(
    notes: NoteSequence,
    timbre: str = "piano",
    sr: int = 16000,
    seed: int = 0,
    tail_s: float = 0.4,
    vibrato_cents: float = 25.0,
    vibrato_hz: float = 5.5,
    jitter_cents: float = 8.0,
    breath: float = 0.02,
    gain: float = 0.5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int((notes.end_time + tail_s) * sr) + 1
    if timbre in ("piano", "sine", "bass", "pad"):
        return _render_discrete(notes, timbre, sr, n, rng, gain)
    if timbre not in ("hum", "voice", "whistle"):
        raise ValueError(f"Unknown timbre {timbre}")
    hz, gate = _legato_f0(notes, sr, n, 0.06 if timbre != "whistle" else 0.04, rng,
                          vibrato_cents if timbre != "whistle" else vibrato_cents * 0.4, vibrato_hz, jitter_cents)
    gate = _rearticulate(gate, notes, sr)
    env = _envelope(gate, sr, 0.03, 0.06)
    phase = 2 * np.pi * np.cumsum(hz) / sr
    y = np.zeros(n)
    if timbre == "whistle":
        y += np.sin(phase) + 0.08 * np.sin(2 * phase)
        noise = rng.standard_normal(n)
        b, a = signal.butter(2, [1500 / (sr / 2), min(0.95, 5000 / (sr / 2))], btype="band")
        y += 6 * breath * signal.lfilter(b, a, noise)
    else:
        formants = FORMANTS[timbre]
        for k in range(1, 40):
            fk = k * hz
            active = (fk > 0) & (fk < sr * 0.45)
            if not active.any():
                break
            w = _formant_gain(fk, formants) * (k ** -1.2) * active
            y += w * np.sin(k * phase)
        y += breath * rng.standard_normal(n)
    y *= env
    peak = np.max(np.abs(y)) + 1e-9
    return (gain * y / peak).astype(np.float32)


def _render_discrete(notes: NoteSequence, timbre: str, sr: int, n: int, rng, gain: float) -> np.ndarray:
    y = np.zeros(n)
    for o, d, p in zip(notes.onsets, notes.durations, notes.pitches):
        a = int(o * sr)
        rel = 0.25 if timbre == "piano" else 0.08
        length = min(n - a, int((d + rel) * sr))
        if length <= 0:
            continue
        t = np.arange(length) / sr
        f = float(midi_to_hz(p))
        if timbre == "sine":
            tone = np.sin(2 * np.pi * f * t)
            env = np.minimum(1, t / 0.01) * np.clip((d + rel - t) / rel, 0, 1)
        elif timbre == "piano":
            tone = sum((0.6 ** (k - 1)) * np.sin(2 * np.pi * k * f * t + rng.uniform(0, 6.28))
                       for k in range(1, 8) if k * f < sr * 0.45)
            env = np.minimum(1, t / 0.005) * np.exp(-t * (1.5 + f / 800)) * np.clip((d + rel - t) / rel, 0, 1)
        elif timbre == "bass":
            tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
            env = np.minimum(1, t / 0.01) * np.exp(-t * 1.2) * np.clip((d + rel - t) / rel, 0, 1)
        else:  # pad
            tone = sum(np.sin(2 * np.pi * f * (1 + det) * t) for det in (-0.003, 0.0, 0.003)) / 3
            env = np.minimum(1, t / 0.12) * np.clip((d + rel - t) / rel, 0, 1)
        y[a:a + length] += tone * env
    peak = np.max(np.abs(y)) + 1e-9
    return (gain * y / peak).astype(np.float32)


def add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator, color: str = "pink") -> np.ndarray:
    """Add noise at a given SNR (computed over the whole signal)."""
    noise = rng.standard_normal(len(y))
    if color == "pink":
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        noise = signal.lfilter(b, a, noise)
    p_sig = np.mean(y**2) + 1e-12
    p_noise = np.mean(noise**2) + 1e-12
    noise *= np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return np.clip(y + noise, -1, 1).astype(np.float32)


def add_reverb(y: np.ndarray, sr: int, rt60: float, rng: np.random.Generator, wet: float = 0.3) -> np.ndarray:
    """Exponentially decaying noise impulse response (moderate room)."""
    length = int(rt60 * sr)
    t = np.arange(length) / sr
    ir = rng.standard_normal(length) * np.exp(-6.9 * t / rt60)
    ir[0] = 0
    ir /= np.sqrt(np.sum(ir**2)) + 1e-9
    wet_sig = signal.fftconvolve(y, ir)[: len(y)]
    out = (1 - wet) * y + wet * wet_sig
    return (out / (np.max(np.abs(out)) + 1e-9) * np.max(np.abs(y))).astype(np.float32)
