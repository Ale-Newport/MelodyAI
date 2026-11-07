"""Note segmentation from a frame-level pitch track.

Algorithm (deliberately simple and explainable):

1. Work on cleaned voiced frames (see :func:`clean_track`), convert to MIDI and
   correct isolated octave errors.
2. Smooth with a short median filter inside each voiced run.
3. Split voiced runs into notes when the pitch departs from the running note
   centre by more than ``split_semitones`` for at least ``min_change_s`` (handles
   legato) or when the amplitude envelope shows a re-articulation dip (handles
   repeated notes, which have no pitch change).
4. Each note gets the median MIDI pitch of its frames; very short notes are merged
   into the neighbour with the closest pitch.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from melodyai.audio.preprocess import frame_rms
from melodyai.features.melody import NoteSequence, fold_octave_jumps
from melodyai.pitch.base import PitchTrack, clean_track


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    v = mask.astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], v, [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def energy_dips(y: np.ndarray | None, sr: int, n_frames: int, hop_s: float, dip_db: float = 6.0) -> np.ndarray:
    """Boolean per pitch frame marking local energy minima that look like re-articulations."""
    out = np.zeros(n_frames, dtype=bool)
    if y is None or len(y) < sr * 0.1:
        return out
    hop = int(round(hop_s * sr))
    rms = frame_rms(y, int(0.03 * sr), hop)
    db = 20 * np.log10(rms + 1e-9)
    db = np.convolve(db, np.ones(3) / 3, mode="same")
    w = 8  # +/- 80 ms neighbourhood
    for i in range(w, len(db) - w):
        if db[i] == db[i - w:i + w + 1].min():
            left, right = db[i - w:i].max(), db[i + 1:i + w + 1].max()
            if min(left, right) - db[i] >= dip_db and i < n_frames:
                out[i] = True
    return out


def segment_notes(
    track: PitchTrack,
    y: np.ndarray | None = None,
    sr: int | None = None,
    split_semitones: float = 0.8,
    min_change_s: float = 0.05,
    min_note_s: float = 0.07,
) -> NoteSequence:
    track = clean_track(track)
    midi = track.midi
    voiced = np.isfinite(midi)
    if voiced.sum() < 3:
        return NoteSequence.empty()
    hop = track.hop_s
    m = midi.copy()
    m[voiced] = fold_octave_jumps(m[voiced])
    dips = energy_dips(y, sr or 16000, len(m), hop) if y is not None else np.zeros(len(m), dtype=bool)
    min_change = max(2, int(round(min_change_s / hop)))

    notes: list[list[float]] = []  # onset, duration, pitch, confidence
    for a, b in _runs(voiced):
        seg = median_filter(m[a:b], size=min(5, (b - a) // 2 * 2 + 1), mode="nearest") if b - a >= 3 else m[a:b]
        start = 0
        i = 1
        while i <= len(seg):
            split = i == len(seg)
            if not split:
                centre = np.median(seg[start:i])
                if dips[a + i] and i - start >= min_change:
                    split = True
                elif abs(seg[i] - centre) > split_semitones:
                    j = min(len(seg), i + min_change)
                    if np.all(np.abs(seg[i:j] - centre) > split_semitones) and j - i >= min(min_change, len(seg) - i):
                        split = True
            if split:
                frames = seg[start:i]
                # Ignore the first/last frames of a note for its pitch (glides/scoops).
                core = frames[len(frames) // 5: len(frames) - len(frames) // 5] if len(frames) >= 5 else frames
                notes.append([
                    float(track.times[a + start]), float((i - start) * hop), float(np.median(core)),
                    float(np.mean(track.voiced_prob[a + start:a + i])),
                ])
                start = i
            i += 1

    # Merge notes that are too short into the neighbour with the closest pitch.
    changed = True
    while changed and len(notes) > 1:
        changed = False
        for k, n in enumerate(notes):
            if n[1] >= min_note_s:
                continue
            cands = [j for j in (k - 1, k + 1) if 0 <= j < len(notes)]
            gaps = [abs(notes[j][2] - n[2]) for j in cands]
            j = cands[int(np.argmin(gaps))]
            tgt = notes[j]
            lo, hi = min(tgt[0], n[0]), max(tgt[0] + tgt[1], n[0] + n[1])
            if hi - lo > tgt[1] + n[1] + 0.15:  # not adjacent (silence between): just drop it
                notes.pop(k)
            else:
                tgt[0], tgt[1] = lo, hi - lo
                notes.pop(k)
            changed = True
            break
    notes = [n for n in notes if n[1] >= min_note_s * 0.5]
    if not notes:
        return NoteSequence.empty()
    a = np.asarray(notes)
    return NoteSequence(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def track_to_contour(track: PitchTrack, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Voiced-only pitch contour resampled to ``rate`` Hz.

    Returns (contour_midi, source_times). Unvoiced frames are removed (concatenated),
    each output frame is the median of the voiced frames that fall in its bin.
    """
    track = clean_track(track)
    midi = track.midi
    voiced = np.isfinite(midi)
    if voiced.sum() < 2:
        return np.zeros(0), np.zeros(0)
    m = fold_octave_jumps(midi[voiced])
    t = track.times[voiced]
    step = max(1, int(round(1.0 / (rate * track.hop_s))))
    n = len(m) // step
    if n == 0:
        return np.array([np.median(m)]), t[:1]
    blocks = m[: n * step].reshape(n, step)
    return np.median(blocks, axis=1), t[: n * step: step]
