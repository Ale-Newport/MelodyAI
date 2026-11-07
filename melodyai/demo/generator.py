"""Procedural generator of ORIGINAL melodies for the demo catalog and for training.

Every melody is produced deterministically from an integer seed:

1. Draw key (tonic), scale/mode, meter and tempo.
2. Build a verse motif and a chorus motif as constrained scale-degree random walks
   (step-weighted intervals, bounded range) with rhythms drawn from a pattern bank.
3. Assemble phrases with classic variation operators (repeat, sequence/transpose,
   altered cadence), sections A (verse) and B (chorus, higher register), and a
   song form such as A-B-A-B or A-A-B-A-B, so choruses appear mid-song.

The output is a :class:`NoteSequence` (MIDI pitches, seconds). These melodies are
generated for this project and released under CC0; they are not derived from any
existing composition. :func:`generate_catalog` additionally rejects candidates that
share near-identical passages with already accepted melodies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from melodyai.features.melody import NoteSequence

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
}

# Rhythm patterns per bar, in beats (quarter notes) for 4/4 and 3/4, eighths for 6/8.
RHYTHMS = {
    "4/4": [
        [1, 1, 1, 1], [0.5, 0.5, 1, 1, 1], [1.5, 0.5, 1, 1], [2, 1, 1], [1, 1, 2], [0.5, 0.5, 0.5, 0.5, 1, 1],
        [0.75, 0.25, 1, 2], [1, 0.5, 0.5, 2], [3, 1], [0.5, 1, 0.5, 2], [1, 0.5, 0.5, 1, 1], [1.5, 1.5, 1],
        [0.5, 0.5, 1, 0.5, 0.5, 1], [2, 0.5, 0.5, 1],
    ],
    "3/4": [[1, 1, 1], [2, 1], [1, 2], [1.5, 0.5, 1], [0.5, 0.5, 1, 1], [1, 0.5, 0.5, 1], [3]],
    "6/8": [[3, 3], [2, 1, 3], [1, 1, 1, 3], [3, 2, 1], [2, 1, 2, 1], [1, 1, 1, 1, 1, 1], [3, 1, 1, 1]],
}
BAR_BEATS = {"4/4": 4.0, "3/4": 3.0, "6/8": 6.0}

STEP_CHOICES = np.array([0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 7, -7])
STEP_WEIGHTS = np.array([0.12, 0.2, 0.2, 0.11, 0.11, 0.06, 0.06, 0.035, 0.035, 0.02, 0.02, 0.0125, 0.0125])
STEP_WEIGHTS = STEP_WEIGHTS / STEP_WEIGHTS.sum()

ADJ = ["Velvet", "Paper", "Amber", "Hollow", "Silent", "Copper", "Wandering", "Northern", "Glass", "Midnight",
       "Little", "Broken", "Golden", "Distant", "Crimson", "Quiet", "Electric", "Faded", "Hidden", "Wild",
       "Lunar", "Salt", "Winter", "Open"]
NOUN = ["Lantern", "Harbor", "Orchard", "Compass", "Meadow", "Signal", "Ferris Wheel", "Lighthouse", "Kite",
        "River", "Carousel", "Postcard", "Ember", "Station", "Garden", "Tide", "Window", "Satellite", "Canyon",
        "Parade", "Bridge", "Thread", "Mirror", "Train"]
ARTISTS = ["MelodyAI Synth Ensemble", "MelodyAI Folk Machine", "MelodyAI Night Choir", "MelodyAI Pocket Orchestra"]


@dataclass
class GeneratedMelody:
    seed: int
    notes: NoteSequence
    key: int
    scale: str
    meter: str
    tempo: float
    form: str
    sections: list[dict] = field(default_factory=list)  # {"name", "start", "end"} in seconds

    @property
    def duration(self) -> float:
        return self.notes.end_time


def _walk(rng, length: int, start: int, lo: int, hi: int) -> list[int]:
    degs = [start]
    for _ in range(length - 1):
        for _attempt in range(20):
            step = int(rng.choice(STEP_CHOICES, p=STEP_WEIGHTS))
            nd = degs[-1] + step
            if lo <= nd <= hi and not (step == 0 and len(degs) >= 2 and degs[-2] == degs[-1]):
                break
        else:
            nd = degs[-1] + (1 if degs[-1] < (lo + hi) // 2 else -1)
        degs.append(nd)
    return degs


def _deg_to_midi(deg: int, scale: list[int], tonic: int) -> int:
    octave, idx = divmod(deg, len(scale))
    return tonic + 12 * octave + scale[idx]


def _phrase(rng, meter: str, bars: int, start_deg: int, lo: int, hi: int, density: float):
    """One phrase: rhythm (beats) and scale degrees."""
    bank = RHYTHMS[meter]
    rhythm: list[float] = []
    for _ in range(bars):
        # density > 1 favours patterns with more notes
        sizes = np.array([len(p) for p in bank], dtype=float)
        w = sizes ** density
        rhythm += list(bank[int(rng.choice(len(bank), p=w / w.sum()))])
    degs = _walk(rng, len(rhythm), start_deg, lo, hi)
    return rhythm, degs


def generate_melody(seed: int) -> GeneratedMelody:
    rng = np.random.default_rng(seed)
    scale_name = str(rng.choice(list(SCALES)))
    scale = SCALES[scale_name]
    meter = str(rng.choice(["4/4", "4/4", "4/4", "3/4", "6/8"]))
    tempo = float(rng.integers(76, 141))
    tonic = int(rng.integers(53, 64))
    beat_s = 60.0 / tempo * (0.5 if meter == "6/8" else 1.0)
    n = len(scale)

    # Verse (A) in a lower register, chorus (B) higher and denser or sparser.
    a_rhythm1, a_degs1 = _phrase(rng, meter, 2, int(rng.integers(0, 3)), -2, n + 1, 1.0)
    a_rhythm2, a_degs2 = _phrase(rng, meter, 2, a_degs1[-1], -2, n + 2, 1.0)
    b_density = float(rng.choice([0.4, 1.6]))
    b_rhythm1, b_degs1 = _phrase(rng, meter, 2, n + int(rng.integers(-2, 2)), 2, n + n // 2 + 1, b_density)
    b_rhythm2, b_degs2 = _phrase(rng, meter, 2, b_degs1[-1], 1, n + n // 2 + 1, b_density)

    def variant(rhythm, degs, kind):
        degs = list(degs)
        if kind == "sequence":
            shift = int(rng.choice([-2, -1, 1, 2]))
            degs = [d + shift for d in degs]
        elif kind == "cadence":
            degs[-1] = 0 if degs[-1] < n else n
            if len(degs) > 2:
                degs[-2] = degs[-1] + int(rng.choice([1, -1, 2]))
        elif kind == "ornament":
            k = int(rng.integers(1, max(2, len(degs) - 1)))
            degs[k] += int(rng.choice([-1, 1]))
        return list(rhythm), degs

    A = [(a_rhythm1, a_degs1), (a_rhythm2, a_degs2), variant(a_rhythm1, a_degs1, "ornament"),
         variant(a_rhythm2, a_degs2, "cadence")]
    B = [(b_rhythm1, b_degs1), variant(b_rhythm1, b_degs1, "sequence"), (b_rhythm2, b_degs2),
         variant(b_rhythm2, b_degs2, "cadence")]
    form = str(rng.choice(["ABAB", "AABAB", "ABAB", "ABB", "ABABB"]))
    notes: list[tuple[float, float, float]] = []
    sections = []
    t = 0.0
    for sec in form:
        phrases = A if sec == "A" else B
        start = t
        for rhythm, degs in phrases:
            for r, d in zip(rhythm, degs):
                dur = r * beat_s
                articulation = 0.92 if r >= 1 else 0.85
                notes.append((t, dur * articulation, float(_deg_to_midi(d, scale, tonic))))
                t += dur
        sections.append({"name": "verse" if sec == "A" else "chorus", "start": round(start, 3), "end": round(t, 3)})
    seq = NoteSequence.from_list(notes)
    return GeneratedMelody(seed, seq, tonic, scale_name, meter, tempo, form, sections)


def title_for(index: int, seed: int) -> str:
    rng = np.random.default_rng(seed + 7919)
    return f"{ADJ[index % len(ADJ)]} {NOUN[int(rng.integers(len(NOUN)))]}"


def _window_signatures(notes: NoteSequence, length: int = 8) -> np.ndarray:
    """Interval n-grams (merged repeats) used to detect near-duplicate passages."""
    from melodyai.matching.algorithms import merge_repeats

    p, _ = merge_repeats(notes)
    iv = np.diff(p)
    if len(iv) < length:
        return np.zeros((0, length))
    return np.stack([iv[i:i + length] for i in range(len(iv) - length + 1)])


def too_similar(a: np.ndarray, b: np.ndarray, tol: float = 0.5) -> bool:
    """True if any interval 8-gram of ``a`` matches one of ``b`` within ``tol`` semitones."""
    if not len(a) or not len(b):
        return False
    d = np.abs(a[:, None, :] - b[None, :, :]).max(axis=2)
    return bool((d <= tol).any())


def generate_catalog(n: int = 48, base_seed: int = 1000, max_duration: float = 55.0) -> list[GeneratedMelody]:
    """Generate ``n`` mutually distinguishable original melodies."""
    accepted: list[GeneratedMelody] = []
    sigs: list[np.ndarray] = []
    seed = base_seed
    while len(accepted) < n:
        m = generate_melody(seed)
        seed += 1
        if not (20.0 <= m.duration <= max_duration):
            continue
        s = _window_signatures(m.notes)
        if any(too_similar(s, other) for other in sigs):
            continue
        accepted.append(m)
        sigs.append(s)
    return accepted
