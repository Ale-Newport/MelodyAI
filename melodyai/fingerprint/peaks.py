"""Spectral peak extraction on a log-frequency grid.

A log-frequency axis turns a pitch shift into a translation (constant number of bins),
which is what makes the hashes in :mod:`melodyai.fingerprint.hashing` pitch-invariant.
We use an STFT followed by a fixed sparse linear->log mapping (much cheaper than a CQT:
~10x faster on long tracks), then 2-D local maxima with density control.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter

SR = 16000
N_FFT = 4096
HOP = 512                    # 32 ms frames
BINS_PER_OCTAVE = 24         # 50-cent resolution
FMIN = 150.0
FMAX = 5000.0
FRAME_S = HOP / SR


@dataclass
class Peaks:
    t: np.ndarray   # frame index (int32)
    f: np.ndarray   # fractional log-frequency bin (float32)
    a: np.ndarray   # magnitude in dB (float32)
    n_frames: int

    def __len__(self) -> int:
        return len(self.t)


@lru_cache(maxsize=4)
def log_filterbank(sr: int = SR, n_fft: int = N_FFT) -> tuple[np.ndarray, np.ndarray]:
    """Triangular linear->log mapping matrix (n_log, n_lin) and log-bin centre frequencies."""
    n_bins = int(np.floor(BINS_PER_OCTAVE * np.log2(FMAX / FMIN))) + 1
    centres = FMIN * 2.0 ** (np.arange(n_bins) / BINS_PER_OCTAVE)
    lin = np.fft.rfftfreq(n_fft, 1.0 / sr)
    W = np.zeros((n_bins, len(lin)), dtype=np.float32)
    half = 2.0 ** (1.0 / BINS_PER_OCTAVE)
    for k, fc in enumerate(centres):
        lo, hi = fc / half, fc * half
        idx = np.flatnonzero((lin >= lo) & (lin <= hi))
        if not len(idx):  # narrower than one linear bin: use the nearest bin
            idx = np.array([int(np.argmin(np.abs(lin - fc)))])
        w = 1.0 - np.abs(np.log2(lin[idx] / fc)) * BINS_PER_OCTAVE
        W[k, idx] = np.clip(w, 0.05, 1.0)
    return W, centres


def log_spectrogram(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """(n_log_bins, n_frames) magnitude in dB on the log-frequency grid."""
    if sr != SR:
        from melodyai.audio.io import resample

        y = resample(y, sr, SR)
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    _, _, Z = signal.stft(y, fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None,
                          padded=False)
    W, _ = log_filterbank()
    M = W @ np.abs(Z).astype(np.float32)
    return 20.0 * np.log10(M + 1e-6)


def extract_peaks(y: np.ndarray, sr: int = SR, per_second: float = 18.0, nbhd: tuple[int, int] = (9, 9),
                  floor_db: float = 45.0) -> Peaks:
    """Local maxima of the log spectrogram, at most ``per_second`` peaks per second.

    A peak must be the maximum of its (freq x time) neighbourhood and lie within
    ``floor_db`` of the loudest bin of the recording. Frequencies are refined by
    parabolic interpolation (sub-bin accuracy keeps frequency differences stable).
    """
    S = log_spectrogram(y, sr)
    n_bins, n_frames = S.shape
    if n_frames < 3:
        return Peaks(np.zeros(0, np.int32), np.zeros(0, np.float32), np.zeros(0, np.float32), n_frames)
    local = (S == maximum_filter(S, size=nbhd, mode="constant", cval=-1e9)) & (S > S.max() - floor_db)
    # Ignore the spectral edges, where interpolation is undefined.
    local[0, :] = local[-1, :] = False
    fb, tb = np.nonzero(local)
    amp = S[fb, tb]
    # Density control: keep the strongest peaks inside 1-second blocks.
    block = max(1, int(round(1.0 / FRAME_S)))
    keep = np.zeros(len(tb), dtype=bool)
    order = np.lexsort((-amp, tb // block))
    blocks = (tb // block)[order]
    starts = np.flatnonzero(np.r_[True, blocks[1:] != blocks[:-1]])
    quota = max(1, int(round(per_second * block * FRAME_S)))
    for s, e in zip(starts, np.r_[starts[1:], len(order)]):
        keep[order[s:min(e, s + quota)]] = True
    fb, tb, amp = fb[keep], tb[keep], amp[keep]
    # Parabolic interpolation along frequency.
    l, c, r = S[fb - 1, tb], S[fb, tb], S[fb + 1, tb]
    den = l - 2 * c + r
    delta = np.divide(0.5 * (l - r), den, out=np.zeros_like(den), where=np.abs(den) > 1e-9)
    f = fb + np.clip(delta, -0.5, 0.5)
    o = np.lexsort((f, tb))
    return Peaks(tb[o].astype(np.int32), f[o].astype(np.float32), amp[o].astype(np.float32), n_frames)
