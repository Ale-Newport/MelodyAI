"""Evaluation query sets (all synthetic, all seeded).

Nothing here represents real users: these sets measure robustness to controlled
transformations. See docs/EVALUATION.md for what they can and cannot tell.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from melodyai.audio.io import Audio
from melodyai.audio.synth import render
from melodyai.demo.generator import GeneratedMelody
from melodyai.demo.queries import CONDITIONS, excerpt, non_melodic_query, perturb, synth_query
from melodyai.features.extract import analyze_query
from melodyai.features.melody import NoteSequence

NON_MELODIC_KINDS = ("noise", "silence", "single_tone", "glissando")


def _query_job(args):
    """Worker: synthesise one query and analyse it. Returns (key, QueryFeatures, meta)."""
    key, kind, payload, seed, extractor = args
    rng = np.random.default_rng(seed)
    import warnings

    warnings.filterwarnings("ignore")
    if kind == "non_melodic":
        y = non_melodic_query(payload, rng)
        profile, meta = "hum", {"kind": payload}
    else:
        cond, notes, sections = payload
        spec = CONDITIONS[cond]
        ex, start = excerpt(notes, rng, spec.min_s, spec.max_s, sections, spec.section)
        pn, params = perturb(ex, spec, rng)
        y = synth_query(pn, spec, rng)
        profile = "whistle" if spec.timbre == "whistle" else "hum"
        meta = {"excerpt_start_s": round(start, 3), "excerpt_notes": len(ex), **params}
    q = analyze_query(Audio(y, 16000), profile, extractor, with_chroma=True)
    q.track = None  # not needed for matching; keeps pickles small
    meta["audio_s"] = round(len(y) / 16000, 3)
    return key, q, meta


def build_query_set(catalog: list[GeneratedMelody], ooc: list[GeneratedMelody], conditions: list[str],
                    n_ooc: int, n_nonmel: int, seed: int, extractor: str = "pyin", workers: int = 4,
                    cache_path=None, per_song: int = 1, progress=None) -> list[dict]:
    """Returns a list of {"id", "condition", "target", "query", "meta"} (target None = not in catalog)."""
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    jobs, info = [], {}
    k = 0
    for cond in conditions:
        for i, m in enumerate(catalog):
            for r in range(per_song):
                key = f"{cond}/{i:03d}/{r}"
                jobs.append((key, "melody", (cond, m.notes, m.sections), seed + 7919 * k, extractor))
                info[key] = (cond, i)
                k += 1
    for j in range(n_ooc):
        m = ooc[j % len(ooc)]
        key = f"out_of_catalog/{j:03d}/0"
        jobs.append((key, "melody", ("hum_clean", m.notes, m.sections), seed + 7919 * k, extractor))
        info[key] = ("out_of_catalog", None)
        k += 1
    for j in range(n_nonmel):
        key = f"non_melodic/{j:03d}/0"
        jobs.append((key, "non_melodic", NON_MELODIC_KINDS[j % len(NON_MELODIC_KINDS)], seed + 7919 * k, extractor))
        info[key] = ("non_melodic", None)
        k += 1
    out = []
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for n, (key, q, meta) in enumerate(ex.map(_query_job, jobs, chunksize=4)):
                cond, target = info[key]
                out.append({"id": key, "condition": cond, "target": target, "query": q, "meta": meta})
                if progress:
                    progress((n + 1) / len(jobs), f"queries {n + 1}/{len(jobs)}")
    else:
        for n, job in enumerate(jobs):
            key, q, meta = _query_job(job)
            cond, target = info[key]
            out.append({"id": key, "condition": cond, "target": target, "query": q, "meta": meta})
            if progress:
                progress((n + 1) / len(jobs), f"queries {n + 1}/{len(jobs)}")
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
    return out


# ----------------------------------------------------------------------------- lyrics
DISTRACTORS = [
    "the night is long and i love my home",
    "dancing in the rain until the morning light",
    "my heart is beating like a drum tonight",
    "we were young and wild and free",
    "baby come back to me tonight",
    "the city lights are shining on the water",
    "te quiero tanto mi amor de la noche",
    "bailando bajo la luna hasta el amanecer",
    "mi corazon late por ti cada dia",
    "la ciudad duerme y yo pienso en ti",
]


def _typo(word: str, rng) -> str:
    if len(word) < 4:
        return word
    i = int(rng.integers(1, len(word) - 1))
    op = rng.integers(3)
    if op == 0:
        return word[:i] + word[i + 1:]
    if op == 1:
        return word[:i] + "aeiou"[int(rng.integers(5))] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def lyrics_queries(lyrics: list[dict], seed: int) -> list[dict]:
    """Per lyric song: exact two-line span, 4-word partial, misremembered and typo-ridden
    variants; plus distractors that share vocabulary/themes but no song's actual text."""
    from melodyai.lyrics.normalize import is_stopword, tokenize

    rng = np.random.default_rng(seed)
    vocab = sorted({t for l in lyrics for t in tokenize(l["text"]) if not is_stopword(t)})
    out = []
    for i, l in enumerate(lyrics):
        lines = [x for x in l["text"].splitlines() if x.strip()]
        k = int(rng.integers(0, len(lines) - 1))
        span = tokenize(lines[k] + " " + lines[k + 1])
        out.append({"condition": "exact_lines", "target": i, "text": " ".join(span)})
        s = int(rng.integers(0, max(1, len(span) - 4)))
        out.append({"condition": "partial_4_words", "target": i, "text": " ".join(span[s:s + 4])})
        mis = list(span)
        content = [j for j, t in enumerate(mis) if not is_stopword(t)]
        for j in rng.choice(content, size=max(1, len(content) // 4), replace=False):
            mis[j] = str(rng.choice(vocab))
        if len(mis) > 6:
            mis.pop(int(rng.integers(len(mis))))
        out.append({"condition": "misremembered", "target": i, "text": " ".join(mis)})
        typo = [_typo(t, rng) if rng.random() < 0.35 else t for t in span]
        out.append({"condition": "asr_like_typos", "target": i, "text": " ".join(typo)})
    for d in DISTRACTORS:
        out.append({"condition": "out_of_catalog", "target": None, "text": d})
    for _ in range(10):  # bag of catalog words in random order: topic overlap without the text
        out.append({"condition": "vocabulary_shuffle", "target": None,
                    "text": " ".join(rng.choice(vocab, size=7, replace=False))})
    return out


def tts_available() -> bool:
    return os.path.exists("/usr/bin/say")


VOICES = {"en": ["Samantha", "Daniel", "Karen"], "es": ["Paulina"]}


def synthesize_speech(text: str, language: str, voice_idx: int = 0) -> Audio | None:
    """macOS `say` text-to-speech (synthetic speech, used only for ASR evaluation)."""
    from melodyai.audio.io import decode_audio

    voices = VOICES.get(language, VOICES["en"])
    voice = voices[voice_idx % len(voices)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.aiff")
        r = subprocess.run(["say", "-v", voice, "-o", path, text], capture_output=True, timeout=60)
        if r.returncode != 0 or not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return decode_audio(f.read())


# ----------------------------------------------------------------------------- polyphonic
def accompaniment(m: GeneratedMelody) -> tuple[NoteSequence, NoteSequence]:
    """Simple pad chords + bass derived from the melody's bars (for polyphonic references)."""
    from melodyai.demo.generator import BAR_BEATS, SCALES

    beat_s = 60.0 / m.tempo * (0.5 if m.meter == "6/8" else 1.0)
    bar = BAR_BEATS[m.meter] * beat_s
    scale = SCALES[m.scale]
    pcs = [(m.key + s) % 12 for s in scale]
    pad, bass = [], []
    t = 0.0
    while t < m.notes.end_time:
        sel = (m.notes.onsets >= t) & (m.notes.onsets < t + bar)
        if sel.any():
            counts = np.bincount((m.notes.pitches[sel].astype(int)) % 12, minlength=12)
            root_pc = int(np.argmax(counts))
        else:
            root_pc = m.key % 12
        if root_pc not in pcs:
            root_pc = pcs[0]
        idx = pcs.index(root_pc)
        triad = [pcs[idx], pcs[(idx + 2) % len(pcs)], pcs[(idx + 4) % len(pcs)]]
        for pc in triad:
            pad.append((t, bar * 0.95, 48 + (pc - 48) % 12))
        steps = int(round(BAR_BEATS[m.meter] / (3 if m.meter == "6/8" else 1)))
        for b in range(steps):
            bass.append((t + b * bar / steps, bar / steps * 0.8, 36 + (root_pc - 36) % 12))
        t += bar
    return NoteSequence.from_list(pad), NoteSequence.from_list(bass)


def render_polyphonic(m: GeneratedMelody, sr: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mix, melody_only) audio."""
    pad, bass = accompaniment(m)
    mel = render(m.notes, "piano", sr=sr, seed=m.seed, gain=0.5)
    p = render(pad, "pad", sr=sr, seed=m.seed + 1, gain=0.35)
    b = render(bass, "bass", sr=sr, seed=m.seed + 2, gain=0.4)
    n = max(len(mel), len(p), len(b))
    mix = np.zeros(n, np.float32)
    for x in (mel, p, b):
        mix[: len(x)] += x
    mix /= np.max(np.abs(mix)) + 1e-9
    return 0.9 * mix, mel
