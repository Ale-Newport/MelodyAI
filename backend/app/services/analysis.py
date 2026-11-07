"""Melody Explorer analysis: every array returned here is computed from the uploaded audio."""

from __future__ import annotations

import numpy as np

from melodyai.audio.io import Audio
from melodyai.config import CONTOUR_RATE
from melodyai.features.extract import analyze_query
from melodyai.features.melody import midi_to_name
from melodyai.matching.explain import explain_alignment


def waveform_envelope(y: np.ndarray, sr: int, points: int = 1200) -> dict:
    n = max(1, len(y) // points)
    m = len(y) // n
    blocks = y[: m * n].reshape(m, n) if m else y.reshape(1, -1)
    return {"min": np.round(blocks.min(axis=1), 4).tolist(), "max": np.round(blocks.max(axis=1), 4).tolist(),
            "duration_s": round(len(y) / sr, 3)}


def spectrogram(y: np.ndarray, sr: int, max_frames: int = 500, fmax: float = 4000.0, bins: int = 160) -> dict:
    """Log-frequency magnitude spectrogram (dB), quantised to 0..255 for transport."""
    import librosa

    hop = max(160, int(np.ceil(len(y) / max_frames / 160)) * 160)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop, center=True))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    edges = np.geomspace(60.0, min(fmax, sr / 2), bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            m = np.zeros_like(freqs, dtype=bool)
            m[int(np.argmin(np.abs(freqs - (lo + hi) / 2)))] = True
        rows.append(S[m].max(axis=0))
    D = librosa.amplitude_to_db(np.asarray(rows), ref=np.max, top_db=80.0)
    Q = np.clip((D + 80.0) / 80.0 * 255.0, 0, 255).astype(np.uint8)
    return {"data": Q.tolist(), "hop_s": hop / sr, "freqs_hz": np.round(np.sqrt(edges[:-1] * edges[1:]), 1).tolist(),
            "db_range": [-80, 0]}


def analyze(audio: Audio, profile: str, extractor: str, melody_index=None, song_id: str | None = None,
            match_fn=None) -> dict:
    q = analyze_query(audio, profile, extractor, with_chroma=True)
    notes = q.notes
    out = {
        "profile": profile, "extractor": extractor, "quality": q.quality, "timings": q.timings,
        "waveform": waveform_envelope(audio.samples, audio.sr), "spectrogram": spectrogram(audio.samples, audio.sr),
        "pitch_track": q.track.to_dict(3000) if q.track is not None else None,
        "notes": [{"onset": round(float(o), 3), "duration": round(float(d), 3), "midi": round(float(p), 2),
                   "name": midi_to_name(p), "confidence": round(float(c), 3)}
                  for o, d, p, c in zip(notes.onsets, notes.durations, notes.pitches, notes.confidence)],
        "intervals": np.round(notes.intervals(), 2).tolist(),
        "contour": np.round(q.contour, 3).tolist(), "contour_rate_hz": CONTOUR_RATE,
        "chroma": {"data": np.round(q.chroma.T, 3).tolist() if q.chroma is not None else [], "rate_hz": 10.0},
        "informative": q.informative,
    }
    target = song_id
    if target is None and match_fn is not None and q.informative:
        best = match_fn(q)
        if best:
            target = best["song_id"]
            out["matched"] = best
    if target and melody_index is not None and target in melody_index:
        ref = melody_index.refs[target]
        t0 = out.get("matched", {}).get("start_time")
        t1 = out.get("matched", {}).get("end_time")
        out["comparison"] = {"song_id": target, "alignment": explain_alignment(q, ref, t0, t1),
                             "reference_notes": ref.notes.to_dict()}
    return out
