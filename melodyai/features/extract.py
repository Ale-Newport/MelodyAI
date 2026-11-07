"""Feature extraction for queries (audio) and references (symbolic or audio)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from melodyai.audio.io import Audio
from melodyai.audio.preprocess import preprocess
from melodyai.config import ANALYSIS_SR, CONTOUR_RATE, FEATURE_VERSION
from melodyai.features.melody import NoteSequence, notes_to_contour
from melodyai.pitch.base import PitchTrack
from melodyai.pitch.extractors import get_extractor
from melodyai.pitch.notes import segment_notes, track_to_contour

CHROMA_RATE = 10.0
REFERENCE_TYPES = ("midi", "monophonic_audio", "vocal_audio", "instrumental_audio", "polyphonic_audio", "symbolic")


def cqt_chroma(y: np.ndarray, sr: int, rate: float = CHROMA_RATE) -> np.ndarray:
    """CQT chroma (T, 12) at ``rate`` frames/s, frames with little energy zeroed."""
    import librosa

    hop = int(2 ** np.round(np.log2(sr / rate)))  # power of two hop needed by CQT
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C2"), n_octaves=6)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0][: ch.shape[1]]
    ch = ch[:, : len(rms)] * (rms > 0.05 * (rms.max() + 1e-9))[None, :]
    # Resample to exactly ``rate`` frames per second.
    t_src = np.arange(ch.shape[1]) * hop / sr
    t_dst = np.arange(0, t_src[-1] if len(t_src) else 0, 1.0 / rate)
    out = np.stack([np.interp(t_dst, t_src, ch[k]) for k in range(12)], axis=1) if len(t_dst) else np.zeros((0, 12))
    return out.astype(np.float32)


@dataclass
class QueryFeatures:
    contour: np.ndarray              # voiced-only MIDI contour at CONTOUR_RATE
    contour_times: np.ndarray        # original query time of each contour frame
    notes: NoteSequence
    track: PitchTrack | None = None
    chroma: np.ndarray | None = None
    quality: dict = field(default_factory=dict)
    profile: str = "hum"
    timings: dict = field(default_factory=dict)

    @property
    def voiced_duration(self) -> float:
        return len(self.contour) / CONTOUR_RATE

    @property
    def informative(self) -> bool:
        """Enough melodic material to attempt matching (>= ~1.5 s voiced and >= 3 distinct notes)."""
        if len(self.contour) < 1.5 * CONTOUR_RATE or len(self.notes) < 3:
            return False
        return float(np.ptp(self.notes.pitches)) >= 1.0

    @classmethod
    def from_notes(cls, notes: NoteSequence, profile: str = "symbolic") -> QueryFeatures:
        """Symbolic query (e.g. typed notes or tests). Rests are absorbed like in references."""
        c = notes_to_contour(notes)
        t = notes.onsets[0] + np.arange(len(c)) / CONTOUR_RATE if len(notes) else np.zeros(0)
        return cls(contour=c, contour_times=t, notes=notes, quality={"symbolic": True}, profile=profile)


def analyze_query(audio: Audio, profile: str = "hum", extractor: str = "pyin", with_chroma: bool = True) -> QueryFeatures:
    """Full query analysis: preprocessing -> pitch -> contour + notes (+ chroma)."""
    timings = {}
    t0 = time.perf_counter()
    pre = preprocess(audio)
    timings["preprocess_s"] = time.perf_counter() - t0
    quality: dict = pre.quality.to_dict()
    y, sr = pre.audio.samples, pre.audio.sr
    if pre.quality.silent or len(y) < 0.2 * sr:
        return QueryFeatures(np.zeros(0), np.zeros(0), NoteSequence.empty(), None, None,
                             {**quality, "voiced_ratio": 0.0, "pitch_confidence": 0.0}, profile, timings)
    pitch_profile = profile if profile in ("hum", "whistle", "sing") else "auto"
    t0 = time.perf_counter()
    track = get_extractor(extractor).extract(y, sr, pitch_profile)
    timings["pitch_s"] = time.perf_counter() - t0
    # Shift times back to the untrimmed recording.
    track.times = track.times + pre.trim_offset_s
    t0 = time.perf_counter()
    contour, ctimes = track_to_contour(track, CONTOUR_RATE)
    notes = segment_notes(track, y, sr)
    if len(notes):
        notes.onsets = notes.onsets + 0.0  # already in track time (offset applied)
    timings["notes_s"] = time.perf_counter() - t0
    chroma = None
    if with_chroma:
        t0 = time.perf_counter()
        chroma = cqt_chroma(y, sr)
        timings["chroma_s"] = time.perf_counter() - t0
    quality.update({
        "voiced_ratio": round(track.voiced_ratio, 4),
        "voiced_duration_s": round(track.voiced_duration, 3),
        "pitch_confidence": round(track.mean_confidence, 4),
        "n_notes": len(notes),
        "pitch_range_semitones": round(float(np.ptp(notes.pitches)), 2) if len(notes) else 0.0,
        "median_f0_hz": round(float(np.nanmedian(track.f0_hz)), 1) if track.voiced.any() else None,
    })
    return QueryFeatures(contour, ctimes, notes, track, chroma, quality, profile, timings)


@dataclass
class ReferenceFeatures:
    song_id: str
    notes: NoteSequence
    contour: np.ndarray
    contour_times: np.ndarray
    chroma: np.ndarray | None
    reference_type: str = "midi"
    feature_version: str = FEATURE_VERSION
    meta: dict = field(default_factory=dict)

    def save(self, path) -> None:
        np.savez_compressed(
            path, onsets=self.notes.onsets, durations=self.notes.durations, pitches=self.notes.pitches,
            confidence=self.notes.confidence, contour=self.contour, contour_times=self.contour_times,
            chroma=self.chroma if self.chroma is not None else np.zeros((0, 12)),
            meta=np.array([self.song_id, self.reference_type, self.feature_version]),
        )

    @classmethod
    def load(cls, path) -> ReferenceFeatures:
        z = np.load(path, allow_pickle=False)
        song_id, rtype, ver = (str(x) for x in z["meta"])
        notes = NoteSequence(z["onsets"], z["durations"], z["pitches"], z["confidence"])
        chroma = z["chroma"] if z["chroma"].shape[0] else None
        return cls(song_id, notes, z["contour"], z["contour_times"], chroma, rtype, ver)


def reference_from_notes(song_id: str, notes: NoteSequence, audio: np.ndarray | None = None,
                         sr: int = ANALYSIS_SR, reference_type: str = "midi") -> ReferenceFeatures:
    contour = notes_to_contour(notes)
    times = notes.onsets[0] + np.arange(len(contour)) / CONTOUR_RATE if len(notes) else np.zeros(0)
    if audio is None and len(notes):
        from melodyai.audio.synth import render

        audio = render(notes, "piano", sr=sr, seed=0)
    chroma = cqt_chroma(audio, sr) if audio is not None and len(audio) > sr * 0.2 else None
    return ReferenceFeatures(song_id, notes, contour, times, chroma, reference_type)


def reference_from_audio(song_id: str, audio: Audio, reference_type: str = "monophonic_audio",
                         extractor: str | None = None) -> ReferenceFeatures:
    """Derive a reference melody from audio.

    - monophonic / vocal audio: pYIN (or the given extractor) + note segmentation.
    - polyphonic / instrumental audio: Basic Pitch transcription + skyline melody when
      available (a heuristic predominant-melody estimate), otherwise pYIN on the mix,
      which is unreliable and flagged in ``meta``.
    """
    y, sr = audio.samples, audio.sr
    meta: dict = {}
    if reference_type in ("polyphonic_audio", "instrumental_audio"):
        from melodyai.pitch.extractors import EXTRACTORS

        bp = EXTRACTORS["basic_pitch"]
        if extractor in (None, "basic_pitch") and bp.available():
            notes = skyline_melody(bp.notes(y, sr, "reference"))
            meta["melody_method"] = "basic_pitch_skyline"
        else:
            track = get_extractor(extractor or "pyin").extract(y, sr, "reference")
            notes = segment_notes(track, y, sr)
            meta["melody_method"] = f"{track.extractor}_on_mix (unreliable for polyphonic audio)"
    else:
        track = get_extractor(extractor or "pyin").extract(y, sr, "reference")
        notes = segment_notes(track, y, sr)
        meta["melody_method"] = f"{track.extractor}_segmentation"
    ref = reference_from_notes(song_id, notes, audio=y, sr=sr, reference_type=reference_type)
    ref.meta = meta
    return ref


def skyline_melody(events: list[tuple[float, float, float, float]], min_amp: float = 0.25) -> NoteSequence:
    """Monophonic melody from polyphonic note events: at each time keep the highest note.

    This is the classic "skyline" heuristic; it works when the melody is the top voice
    and fails when accompaniment plays above it.
    """
    ev = [e for e in events if e[3] >= min_amp]
    if not ev:
        return NoteSequence.empty()
    ev.sort(key=lambda e: (e[0], -e[2]))
    out: list[list[float]] = []
    for s, e, p, _ in ev:
        if out and s < out[-1][0] + out[-1][1] - 0.02:
            last = out[-1]
            if p > last[2]:
                last[1] = max(0.0, s - last[0])  # truncate lower note
                if last[1] < 0.05:
                    out.pop()
                out.append([s, e - s, p])
            continue
        out.append([s, e - s, p])
    out = [n for n in out if n[1] >= 0.06]
    return NoteSequence.from_list([tuple(n) for n in out])
