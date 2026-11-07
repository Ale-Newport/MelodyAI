"""Scalable fingerprint index (log-structured segments) and two-stage matcher.

Storage model (LSM-like, append-friendly):
* New songs go to an in-memory *delta*. ``flush()`` turns the delta into an immutable segment:
  hash postings sorted by hash (``h`` uint32, ``song`` int32, ``t`` float32, ``f`` float32 =
  16 bytes per posting) and the songs' peak maps sorted by song (``ps`` int32, ``pt`` int32,
  ``pf`` float16 = 10 bytes per peak), written as ``.npy`` and memory-mapped on load.
* Deleting a song adds a tombstone; segments are merged tier-wise (small ones only) when there
  are too many, and fully by ``compact()``, which also drops tombstoned data.

Matching a query clip (two stages):
1. *Candidate generation by hashing.* Peaks are extracted once; for each speed factor ``s`` in
   ``SPEED_GRID`` (0.75–1.40, ~3% steps) query peak times are rescaled and pitch-invariant quad
   hashes computed and looked up. Each hit votes, with an IDF weight, for a cell
   (song, time offset, pitch offset); a song's evidence is its best cell.
2. *Geometric verification.* For the best ``verify_top`` candidates, every query peak is mapped
   into reference coordinates (time ``s * t + offset``, frequency ``f + pitch``) and we measure the
   fraction that lands on an actual reference peak (±2 frames, ±1 bin), refining speed (±3% in
   0.5% steps), offset and pitch locally. Generic chord shapes produce consistent hashes in many
   songs but cannot reproduce the detailed peak layout, so this overlap is the final score.
The winning alignment also says *where* in the song the clip is, how much faster/slower it is
and by how many semitones it is transposed.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from melodyai.fingerprint.hashing import Hashes, constellation_hashes
from melodyai.fingerprint.peaks import BINS_PER_OCTAVE, FRAME_S, Peaks, extract_peaks

FP_VERSION = "fp-v2"
SPEED_GRID = tuple(np.round(np.exp(np.linspace(np.log(0.75), np.log(1.40), 22)), 4).tolist())
OFFSET_BIN = 2.0      # frames
PITCH_BIN = 1.0       # log-frequency bins (half a semitone)
REF_PEAKS_PER_S = 18.0
QUERY_PEAKS_PER_S = 18.0
TOL_T, TOL_F = 2, 1   # verification tolerance (frames, bins)
_ARRAYS = ("h", "song", "t", "f", "ps", "pt", "pf")


@dataclass
class Segment:
    h: np.ndarray
    song: np.ndarray
    t: np.ndarray
    f: np.ndarray
    ps: np.ndarray        # peak-map song slots (sorted)
    pt: np.ndarray        # peak frames
    pf: np.ndarray        # peak log-frequency bins
    name: str | None = None

    def __len__(self) -> int:
        return len(self.h)

    @property
    def nbytes(self) -> int:
        return int(sum(np.asarray(getattr(self, k)).nbytes for k in _ARRAYS))

    @classmethod
    def build(cls, parts: list[tuple[int, Hashes, Peaks | None]], name: str | None = None) -> Segment:
        z = np.zeros(0)
        if not parts:
            return cls(z.astype(np.uint32), z.astype(np.int32), z.astype(np.float32), z.astype(np.float32),
                       z.astype(np.int32), z.astype(np.int32), z.astype(np.float16), name)
        h = np.concatenate([p.h for _, p, _ in parts])
        song = np.concatenate([np.full(len(p), s, np.int32) for s, p, _ in parts])
        t = np.concatenate([p.t for _, p, _ in parts])
        f = np.concatenate([p.f for _, p, _ in parts])
        o = np.argsort(h, kind="stable")
        pk = [(s, pk) for s, _, pk in parts if pk is not None]
        ps = np.concatenate([np.full(len(p), s, np.int32) for s, p in pk]) if pk else z.astype(np.int32)
        pt = np.concatenate([p.t.astype(np.int32) for _, p in pk]) if pk else z.astype(np.int32)
        pf = np.concatenate([p.f.astype(np.float16) for _, p in pk]) if pk else z.astype(np.float16)
        po = np.argsort(ps, kind="stable")
        return cls(h[o], song[o], t[o].astype(np.float32), f[o].astype(np.float32), ps[po], pt[po], pf[po], name)

    @classmethod
    def merge(cls, segs: list[Segment], removed: np.ndarray, name: str) -> Segment:
        cat = {k: np.concatenate([np.asarray(getattr(s, k)) for s in segs]) for k in _ARRAYS}
        keep = ~np.isin(cat["song"], removed)
        o = np.argsort(cat["h"][keep], kind="stable")
        pkeep = ~np.isin(cat["ps"], removed)
        po = np.argsort(cat["ps"][pkeep], kind="stable")
        return cls(*(cat[k][keep][o] for k in ("h", "song", "t", "f")),
                   *(cat[k][pkeep][po] for k in ("ps", "pt", "pf")), name)

    def peaks_of(self, slot: int) -> tuple[np.ndarray, np.ndarray] | None:
        a = np.searchsorted(self.ps, slot, "left")
        b = np.searchsorted(self.ps, slot, "right")
        return (np.asarray(self.pt[a:b]), np.asarray(self.pf[a:b], np.float32)) if b > a else None


@dataclass
class FPMatch:
    song_id: str
    count: float           # IDF-weighted number of consistent hash hits (stage 1)
    score: float           # verified peak-overlap fraction in [0, 1] (stage 2)
    speed: float
    offset_s: float
    pitch_semitones: float
    query_hashes: int
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"song_id": self.song_id, "evidence": round(float(self.count), 2), "score": round(self.score, 4),
                "speed": round(self.speed, 3), "start_time": round(max(0.0, self.offset_s), 3),
                "pitch_semitones": round(self.pitch_semitones, 2), "query_hashes": int(self.query_hashes)}


def fingerprint_audio(y: np.ndarray, sr: int) -> tuple[Peaks, Hashes]:
    peaks = extract_peaks(y, sr, per_second=REF_PEAKS_PER_S)
    return peaks, constellation_hashes(peaks)


class FingerprintIndex:
    def __init__(self, root: Path | str | None = None, flush_postings: int = 2_000_000,
                 max_segments: int = 8, max_postings_per_hash: int = 200, verify_top: int = 10) -> None:
        self.root = Path(root) / FP_VERSION if root else None
        self.flush_postings = flush_postings
        self.max_segments = max_segments
        self.max_postings_per_hash = max_postings_per_hash
        self.verify_top = verify_top
        self.song_ids: list[str] = []
        self.slot: dict[str, int] = {}
        self.duration: dict[str, float] = {}
        self.removed: set[int] = set()
        self.segments: list[Segment] = []
        self._delta: list[tuple[int, Hashes, Peaks | None]] = []
        self._delta_seg: Segment | None = None
        self._lock = threading.RLock()
        if self.root and (self.root / "manifest.json").exists():
            self._load()

    # ------------------------------------------------------------------ bookkeeping
    def __len__(self) -> int:
        return len(self.slot)

    def __contains__(self, song_id: str) -> bool:
        return song_id in self.slot

    @property
    def n_postings(self) -> int:
        return sum(len(s) for s in self.segments) + sum(len(h) for _, h, _ in self._delta)

    def stats(self) -> dict:
        delta_bytes = sum(len(h) * 16 + (len(p) * 10 if p is not None else 0) for _, h, p in self._delta)
        nbytes = sum(s.nbytes for s in self.segments) + delta_bytes
        return {"songs": len(self), "postings": self.n_postings, "segments": len(self.segments),
                "delta_songs": len(self._delta), "tombstones": len(self.removed), "bytes": nbytes,
                "version": FP_VERSION}

    # ------------------------------------------------------------------ mutation
    def add(self, song_id: str, hashes: Hashes, duration_s: float | None = None, peaks: Peaks | None = None) -> None:
        with self._lock:
            if song_id in self.slot:
                self.remove(song_id)
            s = len(self.song_ids)
            self.song_ids.append(song_id)
            self.slot[song_id] = s
            if duration_s is not None:
                self.duration[song_id] = float(duration_s)
            self._delta.append((s, hashes, peaks))
            self._delta_seg = None
            if sum(len(h) for _, h, _ in self._delta) >= self.flush_postings:
                self.flush()

    def remove(self, song_id: str) -> bool:
        with self._lock:
            s = self.slot.pop(song_id, None)
            if s is None:
                return False
            self.removed.add(s)
            self._delta = [x for x in self._delta if x[0] != s]
            self._delta_seg = None
            self._save_manifest()
            return True

    def flush(self) -> None:
        """Persist the delta as a new immutable segment (tier-merging small segments if needed)."""
        with self._lock:
            if self._delta:
                seg = Segment.build(self._delta, self._new_name())
                self._write(seg)
                self.segments.append(seg)
                self._delta, self._delta_seg = [], None
                if len(self.segments) > self.max_segments:
                    self._merge_small()
            self._save_manifest()

    def _merge_small(self) -> None:
        base = max(self.segments, key=len)
        small = [s for s in self.segments if s is not base]
        merged = Segment.merge(small, self._removed_array(), self._new_name())
        self._write(merged)
        self.segments = [base, merged]
        self._delete_files([s.name for s in small])

    def compact(self) -> None:
        """Merge everything into one segment and drop tombstoned data."""
        with self._lock:
            self.flush()
            if not self.segments:
                return
            old = [s.name for s in self.segments]
            merged = Segment.merge(self.segments, self._removed_array(), self._new_name())
            self._write(merged)
            self.segments = [merged]
            self.removed = set()  # slots stay reserved; their data no longer exists
            self._save_manifest()
            self._delete_files(old)

    def _removed_array(self) -> np.ndarray:
        return np.fromiter(self.removed, dtype=np.int32, count=len(self.removed))

    # ------------------------------------------------------------------ persistence
    def _new_name(self) -> str:
        existing = {s.name for s in self.segments}
        i = len(existing)
        while f"seg{i:05d}" in existing:
            i += 1
        return f"seg{i:05d}"

    def _write(self, seg: Segment) -> None:
        if not self.root:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        for k in _ARRAYS:
            np.save(self.root / f"{seg.name}_{k}.npy", np.asarray(getattr(seg, k)))

    def _delete_files(self, names) -> None:
        if not self.root:
            return
        for name in names:
            for k in _ARRAYS:
                (self.root / f"{name}_{k}.npy").unlink(missing_ok=True)

    def _save_manifest(self) -> None:
        if not self.root:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        m = {"version": FP_VERSION, "song_ids": self.song_ids, "slots": self.slot, "removed": sorted(self.removed),
             "duration": self.duration, "segments": [s.name for s in self.segments]}
        tmp = self.root / "manifest.json.tmp"
        tmp.write_text(json.dumps(m))
        tmp.replace(self.root / "manifest.json")

    def _load(self) -> None:
        m = json.loads((self.root / "manifest.json").read_text())
        self.song_ids = m["song_ids"]
        self.slot = {k: int(v) for k, v in m["slots"].items()}
        self.removed = set(m.get("removed", []))
        self.duration = m.get("duration", {})
        for name in m["segments"]:
            arrs = [np.load(self.root / f"{name}_{k}.npy", mmap_mode="r") for k in _ARRAYS]
            self.segments.append(Segment(*arrs, name))

    # ------------------------------------------------------------------ search
    def _all_segments(self) -> list[Segment]:
        with self._lock:
            segs = list(self.segments)
            if self._delta:
                if self._delta_seg is None:
                    self._delta_seg = Segment.build(self._delta)
                segs.append(self._delta_seg)
            return segs

    def _hits(self, seg: Segment, q: Hashes, n_ref: float):
        lo = np.searchsorted(seg.h, q.h, "left")
        hi = np.searchsorted(seg.h, q.h, "right")
        n = hi - lo
        ok = (n > 0) & (n <= self.max_postings_per_hash)
        if not ok.any():
            return None
        qi = np.repeat(np.flatnonzero(ok), n[ok])
        starts = np.repeat(lo[ok], n[ok])
        within = np.arange(len(qi)) - np.repeat(np.cumsum(n[ok]) - n[ok], n[ok])
        pos = starts + within
        # IDF-like weight: hashes shared by many postings (generic chord shapes) carry little evidence.
        w = np.log1p(n_ref / np.repeat(n[ok], n[ok]).astype(np.float64))
        return qi, np.asarray(seg.song[pos]), np.asarray(seg.t[pos]), np.asarray(seg.f[pos]), w

    def _peaks_of(self, segs: list[Segment], slot: int):
        for seg in segs:
            p = seg.peaks_of(slot)
            if p is not None:
                return p
        return None

    def _verify(self, qp: Peaks, ref, s: float, off: float, dp: float) -> tuple[float, float, float, float]:
        """Best peak-overlap fraction around the hash alignment; returns (overlap, speed, offset, pitch)."""
        rt, rf = ref
        t0, t1 = s * float(qp.t.min()) + off - 30, s * float(qp.t.max()) + off + 30
        m = (rt >= t0) & (rt <= t1)
        if m.sum() < 4:
            return 0.0, s, off, dp
        rt, rf = rt[m] - int(np.floor(t0)), rf[m]
        T = int(np.ceil(t1 - t0)) + 1
        F = int(np.ceil(rf.max())) + 3
        grid = np.zeros((T, F), bool)
        grid[rt.clip(0, T - 1), np.rint(rf).astype(int).clip(0, F - 1)] = True
        grid = maximum_filter(grid, size=(2 * TOL_T + 1, 2 * TOL_F + 1))
        qt = qp.t.astype(np.float64)
        qf = qp.f.astype(np.float64)
        # Candidates ordered by distance from the hash estimate; an alternative must be clearly
        # better (+0.02) to replace it, so tolerance plateaus do not bias speed/pitch estimates.
        speeds = s * (1.0 + np.arange(-0.03, 0.0301, 0.005))
        speeds = speeds[np.argsort(np.abs(speeds - s), kind="stable")]
        offs = np.arange(-4, 5, 2) + off - np.floor(t0)
        best = (0.0, s, off, dp)
        for p in (dp, dp - 1, dp + 1):
            fi = np.rint(qf + p).astype(int)
            fin = (fi >= 0) & (fi < F)
            for sp in speeds:
                ti = np.rint(sp * qt[None, :] + offs[:, None]).astype(int)   # (n_offsets, n_peaks)
                valid = fin[None, :] & (ti >= 0) & (ti < T)
                hit = valid & grid[ti.clip(0, T - 1), fi.clip(0, F - 1)[None, :]]
                frac = hit.sum(1) / np.maximum(valid.sum(1), 1)
                k = int(np.argmax(frac))
                if valid[k].sum() >= 0.5 * len(qt) and frac[k] > best[0] + (0.02 if best[0] > 0 else 0.0):
                    best = (float(frac[k]), float(sp), float(offs[k] + np.floor(t0)), float(p))
        if best[0] > 0:
            # Sub-bin transposition: median frequency difference over matched peak pairs.
            frac_, sp, o, p = best
            rt_abs = rt + int(np.floor(t0))
            dt = np.abs(rt_abs[None, :] - (sp * qt[:, None] + o))
            df = rf[None, :] - qf[:, None]
            pair = (dt <= TOL_T) & (np.abs(df - p) <= TOL_F + 0.5)
            if pair.sum() >= 5:
                best = (frac_, sp, o, float(np.median(df[pair])))
        return best

    def search(self, y: np.ndarray | None = None, sr: int = 16000, peaks: Peaks | None = None,
               speed_grid=SPEED_GRID, top_k: int = 10) -> list[FPMatch]:
        if peaks is None:
            peaks = extract_peaks(y, sr, per_second=QUERY_PEAKS_PER_S)
        segs = self._all_segments()
        if not segs or len(peaks) < 4:
            return []
        removed = self._removed_array()
        n_ref = max(1.0, sum(len(sg) for sg in segs) / 2.0**20)
        # All speed variants in one batch, sorted by hash (sorted probes make binary search cache-friendly).
        parts = [(v, constellation_hashes(peaks, time_scale=sp)) for v, sp in enumerate(speed_grid)]
        parts = [(v, q) for v, q in parts if len(q)]
        if not parts:
            return []
        qh = np.concatenate([q.h for _, q in parts])
        qt = np.concatenate([q.t for _, q in parts])
        qf = np.concatenate([q.f for _, q in parts])
        qv = np.concatenate([np.full(len(q), v, np.int64) for v, q in parts])
        n_per_variant = np.bincount(qv, minlength=len(speed_grid))
        o = np.argsort(qh, kind="stable")
        q = Hashes(qh[o], qt[o], qf[o])
        qv = qv[o]
        keys, weights = [], []
        for seg in segs:
            hit = self._hits(seg, q, n_ref)
            if hit is None:
                continue
            qi, song, tr, fr, w = hit
            if len(removed):
                keep = ~np.isin(song, removed)
                qi, song, tr, fr, w = qi[keep], song[keep], tr[keep], fr[keep], w[keep]
            off = np.rint((tr - q.t[qi]) / OFFSET_BIN).astype(np.int64)
            dp = np.rint((fr - q.f[qi]) / PITCH_BIN).astype(np.int64)
            # cell key: song (27 bits) | speed variant (5) | offset (24) | pitch (8)
            keys.append((song.astype(np.int64) << 37) | (qv[qi] << 32) | ((off + 2**23) << 8) | (dp + 128))
            weights.append(w)
        if not keys:
            return []
        uniq, inv = np.unique(np.concatenate(keys), return_inverse=True)
        counts = np.bincount(inv, weights=np.concatenate(weights))
        songs = uniq >> 37
        best = np.lexsort((-counts, songs))                       # best cell first within each song
        first = best[np.r_[True, songs[best][1:] != songs[best][:-1]]]
        first = first[np.argsort(-counts[first])][: max(top_k, self.verify_top)]
        out = []
        for i, c in enumerate(first):
            sid = int(songs[c])
            if sid >= len(self.song_ids) or sid in self.removed or self.slot.get(self.song_ids[sid]) != sid:
                continue
            key, cnt = int(uniq[c]), float(counts[c])
            v = (key >> 32) & 31
            s = float(speed_grid[v])
            off = (((key >> 8) & ((1 << 24) - 1)) - 2**23) * OFFSET_BIN
            dp = ((key & 0xFF) - 128) * PITCH_BIN
            ref = self._peaks_of(segs, sid) if i < self.verify_top else None
            if ref is not None:
                score, s, off, dp = self._verify(peaks, ref, s, off, dp)
            else:  # no stored peak map: fall back to a saturating function of the hash evidence
                score = cnt / (cnt + 400.0)
            out.append(FPMatch(self.song_ids[sid], cnt, score, float(s), off * FRAME_S,
                               -dp * 12.0 / BINS_PER_OCTAVE, int(n_per_variant[v])))
        out.sort(key=lambda m: -m.score)
        return out[:top_k]
