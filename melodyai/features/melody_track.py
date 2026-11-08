"""Which track of a multi-track MIDI file carries the sung melody?

People hum the vocal line, so a reference melody taken from the bass, the piano or a pad is useless. In real
MIDI files (Lakh MIDI) the old rule -- a track named "melody"/"lead"/"vocal"/"voice", else the highest median
pitch -- picked the vocal track in only 25 % of 666 karaoke files whose lyric events identify it.

:func:`rank_melody_tracks` scores every pitched track with a logistic model over simple features (name
keywords, General MIDI program family, mean pitch and its distance to the singing range, pitch spread,
polyphony, chords, time coverage, note density, note count). It was fitted on those 666 karaoke files
(label: the track whose note onsets coincide with the lyric events) and picks the vocal track in 83 % of
held-out files (92 % within its first two), see ``docs/LAKH.md``.

:func:`vocal_track_from_lyrics` gives the answer directly when the file has synced lyric events.
"""

from __future__ import annotations

import numpy as np

NAME_KEYS = ("vocal", "voice", "voz", "melod", "lead", "sing", "vox", "canto", "lyric", "solo", "chant", "melo")
VOCAL_PROGRAMS = frozenset({52, 53, 54, 85})                                    # choir / voice oohs / synth voice
LEAD_PROGRAMS = frozenset(set(range(56, 58)) | set(range(64, 80)) | {80, 81, 40, 41})  # brass, reeds, pipes, leads
BASS_PROGRAMS = frozenset(range(32, 40))
PAD_PROGRAMS = frozenset(set(range(48, 52)) | set(range(88, 96)))

# logistic model fitted on 666 Lakh MIDI karaoke files (2-fold CV accuracy 0.83, top-2 0.92)
_COEF = np.array([0.958, -0.523, 0.284, -2.095, -1.123, -0.484, -1.148, 0.309, -0.678, -1.046, -4.956, -5.47,
                  4.136, -0.714, 1.613])
_INTERCEPT = -9.049
MIN_NOTES = 8


def track_features(name: str, program: int, starts: np.ndarray, ends: np.ndarray, pitches: np.ndarray,
                   song_end: float) -> np.ndarray:
    order = np.argsort(starts, kind="stable")
    s, e, p = starts[order], ends[order], pitches[order].astype(float)
    over = float(np.mean(s[1:] < np.maximum.accumulate(e)[:-1] - 0.03)) if len(s) > 1 else 0.0
    chord = float(np.mean(np.diff(s) < 0.02)) if len(s) > 1 else 0.0
    end = max(float(song_end), 1.0)
    cover = min(1.0, float(np.sum(np.minimum(e - s, 2.0))) / end)
    name = (name or "").lower()
    return np.array([any(k in name for k in NAME_KEYS), program in VOCAL_PROGRAMS, program in LEAD_PROGRAMS,
                     program in BASS_PROGRAMS, program in PAD_PROGRAMS, program < 8, 24 <= program < 32,
                     (p.mean() - 66) / 6, abs(p.mean() - 67) / 6, p.std() / 6, over, chord, cover,
                     min(len(s) / end, 8) / 4, np.log(len(s))], dtype=float)


def _arrays(inst):
    return (np.array([n.start for n in inst.notes]), np.array([n.end for n in inst.notes]),
            np.array([n.pitch for n in inst.notes], dtype=float))


def rank_melody_tracks(pm) -> list[tuple[int, float]]:
    """``[(instrument index, logit score)]`` of the pitched tracks of a ``pretty_midi.PrettyMIDI``, best first."""
    end = pm.get_end_time() or 1.0
    out = []
    for i, inst in enumerate(pm.instruments):
        if inst.is_drum or len(inst.notes) < MIN_NOTES:
            continue
        s, e, p = _arrays(inst)
        out.append((i, float(track_features(inst.name, inst.program, s, e, p, end) @ _COEF + _INTERCEPT)))
    return sorted(out, key=lambda x: -x[1])


def vocal_track_from_lyrics(pm, min_lyrics: int = 30, min_hit: float = 0.5, min_lead: float = 0.15) -> int | None:
    """The track whose note onsets coincide (60 ms) with the lyric events, if the file has synced lyrics."""
    if len(pm.lyrics or []) < min_lyrics:
        return None
    lyr = np.array([x.time for x in pm.lyrics])
    hits = []
    for i, inst in enumerate(pm.instruments):
        if inst.is_drum or len(inst.notes) < MIN_NOTES:
            continue
        s = np.array([n.start for n in inst.notes])
        hits.append((float(np.mean(np.min(np.abs(lyr[:, None] - s[None, :]), axis=1) < 0.06)), i))
    if not hits:
        return None
    hits.sort(reverse=True)
    if hits[0][0] < min_hit or (len(hits) > 1 and hits[0][0] - hits[1][0] < min_lead):
        return None
    return hits[0][1]
