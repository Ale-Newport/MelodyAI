"""Synthetic query generation (symbolic transformations + audio rendering).

These simulate how people hum/whistle/sing a remembered passage. They are useful for
regression tests and controlled ablations, but they are NOT a substitute for real
recordings: real queries contain timing drift, articulation and timbre effects that
this simulator only roughly approximates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from melodyai.audio.synth import add_noise, add_reverb, render
from melodyai.features.melody import NoteSequence


@dataclass
class QuerySpec:
    condition: str
    timbre: str = "hum"
    min_s: float = 5.0
    max_s: float = 10.0
    key_shift: tuple[float, float] = (-5, 5)
    tempo: tuple[float, float] = (0.85, 1.18)
    detune_cents: float = 15.0
    wrong_note_p: float = 0.0
    drop_p: float = 0.0
    repeat_p: float = 0.0
    duration_jitter: float = 0.08
    snr_db: float | None = None
    reverb: float | None = None
    register_shift: int = 0     # whistle: +12/+24 octaves above
    vocal_range: tuple[float, float] | None = (50.0, 68.0)  # median MIDI of the sung excerpt
    section: str | None = None  # "chorus" -> excerpt starts inside a chorus in mid-song
    extra: dict = field(default_factory=dict)


CONDITIONS: dict[str, QuerySpec] = {
    "hum_clean": QuerySpec("hum_clean"),
    "hum_transposed": QuerySpec("hum_transposed", key_shift=(5, 11)),
    "hum_tempo": QuerySpec("hum_tempo", tempo=(0.6, 0.72), extra={"alt_tempo": (1.35, 1.6)}),
    "hum_detuned": QuerySpec("hum_detuned", detune_cents=40.0, wrong_note_p=0.1),
    "hum_missing_notes": QuerySpec("hum_missing_notes", drop_p=0.12, repeat_p=0.08),
    "hum_noisy": QuerySpec("hum_noisy", snr_db=10.0, reverb=0.4),
    "hum_short": QuerySpec("hum_short", min_s=2.5, max_s=4.0),
    "chorus_mid_song": QuerySpec("chorus_mid_song", section="chorus"),
    "whistle": QuerySpec("whistle", timbre="whistle", detune_cents=10.0, vocal_range=(78.0, 92.0)),
    "sing_vowel": QuerySpec("sing_vowel", timbre="voice"),
}


def excerpt(notes: NoteSequence, rng: np.random.Generator, min_s: float, max_s: float,
            sections: list[dict] | None = None, section: str | None = None) -> tuple[NoteSequence, float]:
    """Random contiguous excerpt (whole notes) of roughly ``min_s``-``max_s`` seconds.

    Returns the excerpt (re-based to t=0) and its start time in the original.
    """
    target = rng.uniform(min_s, max_s)
    candidates = np.arange(len(notes))
    if section and sections:
        secs = [s for s in sections if s["name"] == section and s["start"] > 1.0]
        if secs:
            s = secs[int(rng.integers(len(secs)))]
            inside = np.flatnonzero((notes.onsets >= s["start"]) & (notes.onsets < s["end"] - 1.0))
            if len(inside):
                candidates = inside
    ends = notes.onsets + notes.durations
    valid = [i for i in candidates if ends[-1] - notes.onsets[i] >= min(target, ends[-1] * 0.5)]
    i0 = int(rng.choice(valid)) if valid else 0
    i1 = i0 + 1
    while i1 < len(notes) and notes.onsets[i1] - notes.onsets[i0] < target:
        i1 += 1
    return notes.slice(i0, i1), float(notes.onsets[i0])


def perturb(notes: NoteSequence, spec: QuerySpec, rng: np.random.Generator) -> tuple[NoteSequence, dict]:
    """Apply key/tempo/intonation/note-level perturbations. Returns notes and the parameters used."""
    lo, hi = spec.key_shift
    shift = float(rng.uniform(lo, hi)) * float(rng.choice([-1, 1]) if lo >= 0 else 1)
    tempo = float(rng.uniform(*spec.tempo))
    if "alt_tempo" in spec.extra and rng.random() < 0.5:
        tempo = float(rng.uniform(*spec.extra["alt_tempo"]))
    on, du, pi = list(notes.onsets), list(notes.durations), list(notes.pitches)
    ioi = list(notes.ioi())
    new: list[tuple[float, float]] = []  # (ioi, pitch)
    for k in range(len(pi)):
        if spec.drop_p and rng.random() < spec.drop_p and 0 < k < len(pi) - 1:
            if new:
                new[-1] = (new[-1][0] + ioi[k], new[-1][1])  # previous note is held longer
            continue
        p = pi[k] + rng.normal(0, spec.detune_cents / 100.0)
        if spec.wrong_note_p and rng.random() < spec.wrong_note_p:
            p += float(rng.choice([-2, -1, 1, 2]))
        d = ioi[k] * float(np.exp(rng.normal(0, spec.duration_jitter)))
        if spec.repeat_p and rng.random() < spec.repeat_p:
            new.append((d / 2, p))
            new.append((d / 2, p))
        else:
            new.append((d, p))
    if spec.vocal_range and new:
        # People hum/sing in their own comfortable register: fold the key shift by
        # octaves so the excerpt's median lands inside ``vocal_range``.
        med = float(np.median([p for _, p in new])) + shift
        lo_v, hi_v = spec.vocal_range
        while med > hi_v:
            shift, med = shift - 12, med - 12
        while med < lo_v:
            shift, med = shift + 12, med + 12
    t = 0.0
    out = []
    for d, p in new:
        d *= 1.0 / tempo  # tempo > 1 -> faster
        out.append((t, d * 0.95, p + shift + spec.register_shift))
        t += d
    params = {"key_shift": round(shift, 3), "tempo_factor": round(tempo, 3)}
    del on, du
    return NoteSequence.from_list(out), params


def synth_query(notes: NoteSequence, spec: QuerySpec, rng: np.random.Generator, sr: int = 16000) -> np.ndarray:
    seed = int(rng.integers(1 << 30))
    y = render(notes, spec.timbre, sr=sr, seed=seed, vibrato_cents=float(rng.uniform(10, 35)),
               jitter_cents=float(rng.uniform(4, 15)), tail_s=0.3)
    lead = np.zeros(int(rng.uniform(0.1, 0.5) * sr), dtype=np.float32)
    y = np.concatenate([lead, y])
    if spec.reverb:
        y = add_reverb(y, sr, spec.reverb, rng)
    snr = spec.snr_db if spec.snr_db is not None else float(rng.uniform(25, 40))
    return add_noise(y, snr, rng)


def non_melodic_query(kind: str, rng: np.random.Generator, sr: int = 16000, seconds: float = 6.0) -> np.ndarray:
    """Inputs with no recognisable melody (Test E): noise, a single held tone, a glissando."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    if kind == "noise":
        return add_noise(np.zeros(n, dtype=np.float32) + 1e-4, -30, rng)
    if kind == "silence":
        return (rng.standard_normal(n) * 1e-5).astype(np.float32)
    if kind == "single_tone":
        f = rng.uniform(150, 300)
        return (0.4 * np.sin(2 * np.pi * f * t) * np.minimum(1, t / 0.05)).astype(np.float32)
    if kind == "glissando":
        f = 150 * 2 ** (t / seconds * 2)
        return (0.4 * np.sin(2 * np.pi * np.cumsum(f) / sr)).astype(np.float32)
    raise ValueError(kind)
