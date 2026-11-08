"""Speech-to-text with faster-whisper (optional dependency), tuned for SUNG lyrics.

CTranslate2 (the faster-whisper backend) supports CPU and CUDA but not Apple MPS, so
on Apple Silicon inference runs on CPU (float32, see :func:`default_compute_type`). Models
are downloaded from the Hugging Face Hub only by an explicit action; :func:`model_status`
reports whether a model is already cached and the ``"auto"`` setting picks the best cached
one (``AUTO_PREFERENCE``).

People who sing from memory are usually Spanish or English speakers singing well-known
songs, and Whisper's language identification is unreliable on singing (sung Spanish is
often detected as Portuguese, Icelandic, French...; once the language token is wrong the
transcript is a translation or gibberish). When no language is given, the language is
chosen as ``argmax p(lang | audio) * LANGUAGE_PRIOR[lang]`` (a soft prior: a clearly
Portuguese or Korean recording still wins). ``large-v3-turbo`` is the model that makes sung
lyrics usable at all (``base`` mangles most sung lines).

One encoder pass per clip: Whisper always encodes a padded 30-s window, and with the 32-layer
encoder / 4-layer decoder of large-v3-turbo the encoder is most of a short query's cost. Language
detection and every decoding of a clip <= 30 s encode the same window, so :class:`EncoderCache`
(installed on each loaded model) memoises ``WhisperModel.encode`` by the window's bytes, and the
language is detected on exactly the window the decoder will see (:func:`detect_language_probs`).
This removes one of the two encoder passes the old code ran and makes a second decoding cheap.

Guarding against hallucinations on non-speech input (humming, whistling, noise, music):
- Silero VAD (bundled with faster-whisper) is a *gate*, not a cutter: if it finds less than
  ``MIN_VAD_SPEECH_S`` of speech, Whisper is not run at all. It is not used to cut the audio
  because it under-detects singing (trained on speech): cutting cost 5 WER points on sung
  clips and 13 on synthesised singing, while gating keeps every hum/whistle/noise out;
- segments with high ``no_speech_prob`` or very low ``avg_logprob`` are discarded, and so
  are repetition loops (compression ratio above ``MAX_COMPRESSION_RATIO``; but see the relaxed
  tier below for choruses that really repeat a phrase). Temperature fallback is NOT used: on
  singing it turned empty outputs into fluent invented text (and sampling is not reproducible);
- bracketed annotations / music symbols ("[Música]", "♪") are stripped and well-known
  Whisper hallucinations in several languages ("Thank you.", "¡Suscríbete al canal!",
  "Subtítulos por la comunidad de Amara.org", "Sous-titrage Radio-Canada"...) are rejected;
- vocables ("mm-hmm", "la la la", "aaah") do not count as words: at least
  ``MIN_LEXICAL_WORDS`` real words must remain, otherwise ``speech_detected = False``.

Sung audio: when the strict filter leaves nothing although the VAD heard a voice, it is
usually right about the words it dropped (a decoder loop, "¡Suscríbete al canal!", a
translation produced under a wrong language token) but the clip still holds lyrics. On the
world evaluation's synthetic singing of real songs, 4 of 39 clips lost every segment: 3 to the
compression-ratio rule (choruses that really repeat a phrase, and one wrong-language loop) and
1 to the hallucination list. So, all on the cached encoder output:
- when the prior picked a language other than es/en with a posterior below
  ``ALTERNATIVE_MAX_POSTERIOR``, the clip is also decoded in the likelier of es/en and the
  more confident usable decoding wins (sung Spanish taken for Portuguese got
  its Spanish back; no pt/fr/it/de clip of the test set changed language);
- decodings are compared by :func:`decoding_score`, which also counts the stretches a decoding
  failed on, so one fluent line cannot beat a complete transcript of the clip;
- when no decoding is usable, a greedy retry in the same language (``RETRY_OPTIONS``; never
  the other language: a failed English chorus re-decoded as Spanish became a confident
  false match); with fewer than ``MIN_DISTINCT_WORDS_RELAXED`` distinct words, its transcript
  also needs the first decoding to have heard its words (sung vocables gave "<word> <word>
  <word>", one invented word; a chorus that repeats one word was heard by both decodings);
- then a relaxed tier keeps windows above ``MAX_COMPRESSION_RATIO`` with their repetition
  collapsed, unless the decoder emitted more than ``MAX_LOOP_WORDS_PER_S`` (a loop), and only
  if the other decoding heard ``MIN_AGREEMENT`` of the same words (looped hallucinations on
  voice-like noise do not agree; a real chorus does).
Every decoding has a token budget (``MAX_TOKENS_PER_S``), which stops runaway loops early.
:attr:`Transcript.passes` records, without any text, what each decoding produced and which
filter rule dropped its segments.

The constants were chosen on a local test set (invented lyric lines in es/en spoken by
macOS TTS voices; "sung-like" tempo/pitch-warped versions with vibrato; macOS singing
voices; the warped versions over music at 10 dB SNR; pt/fr/it/de lines; hums, whistles,
vowel singing, sung vocables, time-reversed speech, instrumental music, noise), comparing word
error rate, false words on non-speech and latency, and checked end to end on the world
evaluation's ASR variants (dev songs only).
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import platform
import re
import sys
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import numpy as np

log = logging.getLogger(__name__)

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
AUTO = "auto"
MODEL_CHOICES = (AUTO, *MODEL_SIZES)
# "auto" = the best model already in the local cache, in this order. large-v3-turbo (pruned 4-layer decoder)
# is as accurate as large-v3 on lyrics and several times faster, so it comes first.
AUTO_PREFERENCE = ("large-v3-turbo", "large-v3", "medium", "small", "base", "tiny")
AUTO_DOWNLOAD = "large-v3-turbo"  # what "auto" asks for when nothing is cached

# ----------------------------------------------------------------------------- decoding
VAD_PARAMETERS = {"threshold": 0.5, "min_silence_duration_ms": 400}
MIN_VAD_SPEECH_S = 1.5            # hums/whistles/noise/music: <= 1.14 s of VAD "speech"; lyric clips >= 4.7 s
DECODE_OPTIONS = {"beam_size": 5, "condition_on_previous_text": False, "no_speech_threshold": 0.6,
                  "temperature": 0.0}
# Soft prior over sung languages (see module docstring); unlisted languages get OTHER_LANGUAGE_PRIOR.
LANGUAGE_PRIOR = {"es": 1.0, "en": 1.0, "pt": 0.35, "it": 0.3, "fr": 0.3, "de": 0.2}
OTHER_LANGUAGE_PRIOR = 0.05

# ----------------------------------------------------------------------------- segment filter
MAX_NO_SPEECH_PROB = 0.6          # ... together with avg_logprob < NO_SPEECH_MAX_LOGPROB
NO_SPEECH_MAX_LOGPROB = -0.5
MIN_AVG_LOGPROB = -1.2
MAX_COMPRESSION_RATIO = 2.8       # loops ("I love you, I love you, ..." x20) score 2.9-50; 2.4-3.2 made no
#                                   difference on the test set, so it sits above the old 2.6 for repeated choruses
MIN_LEXICAL_WORDS = 3            # "All right." on a sung vowel is not a lyrics query (no distinct-word rule here:
#                                   a real chorus can repeat ONE word, and requiring 2 lost such a song)
# Relaxed tier (only when no decoding passed the strict filter): a window above MAX_COMPRESSION_RATIO is kept, its
# repetition collapsed, if it emits no more words per second of audio than people sing (else it is a decoder loop),
# the other decoding of the clip heard at least MIN_AGREEMENT of its distinct words, and it has at least
# MIN_DISTINCT_WORDS_RELAXED distinct words ("Oh, I'm sorry, I'm sorry": a loop both decodings made on reversed speech).
MAX_LOOP_WORDS_PER_S = 5.0        # loops: 6.4-12.4 words/s; sung choruses dropped as repetitive: 1.2-3.8 words/s
MIN_AGREEMENT = 0.5               # hallucination loops on reversed speech agreed 0.0-0.17 with the other decoding
MIN_DISTINCT_WORDS_RELAXED = 3

# ----------------------------------------------------------------------------- further decodings
SUNG_LANGUAGES = tuple(lang for lang, w in LANGUAGE_PRIOR.items() if w >= 1.0)    # ("es", "en")
# When the prior picks another language with a posterior below this, the clip is also decoded in the likelier
# sung language and the decoding with the higher mean log-prob wins.
ALTERNATIVE_MAX_POSTERIOR = 0.9
# When no decoding is usable although the VAD found voice: once more, same language, greedy. Beam 1 rescued as many
# clips as temperature fallback (non-deterministic, 1.6x slower) and more than an "♪" prompt or beam 5 again.
RETRY_OPTIONS = {**DECODE_OPTIONS, "beam_size": 1}
# Token budget of a decoding: lyrics take 3-6 tokens/s, but a looping decoder runs on to Whisper's 448-token limit,
# which made a looped clip 2-3x slower than a normal one (and a loop is dropped anyway). The budget stays high
# enough that a cut loop still exceeds MAX_LOOP_WORDS_PER_S (at 15 tokens/s a cut loop fell to 5.4 words/s).
MAX_TOKENS_PER_S = 30.0
MAX_NEW_TOKENS = 400              # Whisper's limit is 448 tokens including the ~4-token prompt

# Whole-segment hallucinations (normalised: casefold, no accents/punctuation).
HALLUCINATIONS = {
    "thank you", "thank you very much", "thanks", "thanks for watching", "thank you for watching", "you",
    "bye", "bye bye", "gracias", "muchas gracias", "gracias por ver", "gracias por ver el video", "musica",
    "music", "aplausos", "applause", "risas", "laughter", "silencio", "silence", "please subscribe",
    "suscribete", "obrigado", "obrigada", "merci", "grazie", "danke", "see you next time", "ill see you next time",
    "see you in the next video", "nos vemos en el proximo video", "hasta la proxima",
    "subtitulos realizados por la comunidad de amara org",
}
# Substrings that only ever appear in hallucinated subtitle credits / outros.
HALLUCINATION_MARKERS = (
    "amara org", "subtitulos", "subtitulado", "subtitles by", "subtitled by", "captions by", "sous titrage",
    "sous titres", "radio canada", "субтитры", "dimatorzok", "thanks for watching", "thank you for watching",
    "gracias por ver", "suscribete", "subscribe to", "like and subscribe", "продолжение следует", "www ",
    "see you next time", "see you in the next", "for joining us", "nos vemos en el proximo",
)
VOCABLES = {
    "mm", "mmm", "hmm", "hm", "mhm", "uh", "uhm", "um", "ah", "oh", "ooh", "eh", "ha", "hah", "la", "lala", "na",
    "nana", "da", "dada", "du", "doo", "dum", "ba", "pa", "ta", "nah", "woo", "wo", "whoa", "yeah", "ay", "uy",
    "sh", "shh",
}
_ELONGATED = re.compile(r"^(?:[aeiou]+h*|h*m+|m+h+m*)$")      # aaah, ooooh, hmmmm, mmmm
_ANNOTATION = re.compile(r"[\[(（【][^\])）】]*[\])）】]|[♪♫♬]+|\*[^*]*\*")
_NON_WORD = re.compile(r"[^\w\s]|_")
_APOSTROPHE = re.compile(r"['’`´]")


@dataclass
class Transcript:
    text: str
    language: str | None
    language_probability: float
    speech_detected: bool
    reason: str = ""
    segments: list[dict] = field(default_factory=list)
    model: str = ""
    elapsed_s: float = 0.0
    passes: list[dict] = field(default_factory=list)   # per decoding: language, counts, drop rules (no text)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def reliability(self) -> float:
        """Heuristic [0,1] transcription reliability (NOT a calibrated probability)."""
        if not self.speech_detected or not self.segments:
            return 0.0
        lp = np.mean([s["avg_logprob"] for s in self.segments])
        return float(np.clip(1.0 + lp, 0.0, 1.0) * np.clip(self.language_probability, 0.3, 1.0))


def asr_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def model_status(size: str) -> dict:
    """``{"name", "installed_package", "cached"}`` for a size; ``"auto"`` also reports ``"resolved"``."""
    if size == AUTO:
        best = best_cached_model()
        return {"name": AUTO, "installed_package": asr_available(), "cached": best is not None, "resolved": best}
    installed = asr_available()
    return {"name": size, "installed_package": installed, "cached": installed and _cached(size)}


def _cached(size: str) -> bool:
    """True if the model files are in the local Hugging Face cache (never downloads)."""
    try:
        from faster_whisper.utils import download_model

        download_model(size, local_files_only=True)
        return True
    except Exception:
        return False


def best_cached_model() -> str | None:
    """Best Whisper model already downloaded (see ``AUTO_PREFERENCE``), or None."""
    for size in AUTO_PREFERENCE:
        if model_status(size)["cached"]:
            return size
    return None


def resolve_model(name: str, allow_download: bool = False) -> str | None:
    """Concrete model size for a setting value.

    Explicit sizes are returned unchanged (whether they are cached is checked by the caller).
    ``"auto"`` -> the best cached model; when nothing is cached, ``AUTO_DOWNLOAD`` if downloads are
    allowed, else None (ASR unavailable).
    """
    if name == AUTO:
        return best_cached_model() or (AUTO_DOWNLOAD if allow_download else None)
    if name not in MODEL_SIZES:
        raise ValueError(f"Unknown Whisper model size '{name}'. Options: {MODEL_CHOICES}")
    return name


class EncoderCache:
    """Memoises ``WhisperModel.encode`` for byte-identical mel windows (installed on every loaded model).

    Whisper always encodes a padded 30-s window, and for a short query the encoder is most of the work. Language
    detection, the first decoding and an optional second decoding of a clip <= 30 s all encode the same first
    window, so with this cache the encoder runs once per clip instead of two or three times. A hit needs the same
    shape, dtype and 128-bit BLAKE2 digest of the features, so a different recording can never get a stale output.
    A lock guards the small LRU; two threads missing on different audio simply encode in parallel.
    """

    def __init__(self, encode, size: int = 4) -> None:
        self._encode, self._size = encode, size
        self._items: OrderedDict[tuple, object] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(features) -> tuple:
        arr = np.ascontiguousarray(features)
        return arr.shape, arr.dtype.str, hashlib.blake2b(arr.view(np.uint8).data, digest_size=16).digest()

    def __call__(self, features):
        key = self.key(features)
        with self._lock:
            out = self._items.get(key)
            if out is not None:
                self._items.move_to_end(key)
                self.hits += 1
                return out
        out = self._encode(features)
        with self._lock:
            self.misses += 1
            self._items[key] = out
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return out


def default_compute_type(device: str) -> str:
    """CTranslate2 compute type: ``MELODYAI_ASR_COMPUTE_TYPE`` if set, else float32 on Apple Silicon CPUs
    (CTranslate2 uses Accelerate there: large-v3-turbo ran ~1.5-2x faster than int8 on an M2 Max, with
    ~3 GB of RAM instead of ~1 GB), int8 elsewhere."""
    env = os.environ.get("MELODYAI_ASR_COMPUTE_TYPE")
    if env:
        return env
    if device == "cpu" and sys.platform == "darwin" and platform.machine() == "arm64":
        return "float32"
    return "int8"


class WhisperTranscriber:
    _models: dict[tuple[str, str, str], object] = {}
    _lock = threading.Lock()

    def __init__(self, model_size: str = AUTO, device: str = "auto", compute_type: str = "auto",
                 allow_download: bool = False) -> None:
        if model_size not in MODEL_CHOICES:
            raise ValueError(f"Unknown Whisper model size '{model_size}'")
        self.requested = model_size
        # "auto" is resolved per instance, so a model downloaded while the app runs is picked up.
        self.model_size = resolve_model(model_size, allow_download) or AUTO_DOWNLOAD
        self.device = "cuda" if device == "cuda" else "cpu"
        if device == "auto":
            try:
                import ctranslate2

                self.device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                self.device = "cpu"
        self.compute_type = default_compute_type(self.device) if compute_type == "auto" else compute_type
        self.allow_download = allow_download

    def _model(self):
        key = (self.model_size, self.device, self.compute_type)
        with self._lock:
            if key not in self._models:
                from faster_whisper import WhisperModel

                if not self.allow_download and not model_status(self.model_size)["cached"]:
                    raise RuntimeError(
                        f"Whisper model '{self.model_size}' is not downloaded. Run "
                        f"`make asr-model MODEL={self.model_size}` or enable downloads in Settings.")
                model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type,
                                     cpu_threads=min(8, os.cpu_count() or 4))
                model.encode = EncoderCache(model.encode)    # instance attribute: generate_segments() calls it
                self._models[key] = model
            return self._models[key]

    def transcribe(self, y: np.ndarray, sr: int, language: str | None = None) -> Transcript:
        if sr != 16000:
            from melodyai.audio.io import resample

            y = resample(y, sr, 16000)
        y = np.ascontiguousarray(y, dtype=np.float32)
        t0 = time.perf_counter()
        model = self._model()
        speech_s = vad_speech_seconds(y)
        if speech_s < MIN_VAD_SPEECH_S:  # hum / whistle / noise / music: do not even run Whisper
            return Transcript("", None, 0.0, False, f"no speech (VAD found {speech_s:.1f} s)", [],
                              self.model_size, time.perf_counter() - t0)
        language_probability = 1.0 if language else 0.0
        posteriors: dict[str, float] = {}
        if language is None and hasattr(model, "detect_language"):  # faster-whisper >= 1.1
            try:
                posteriors = language_posteriors(detect_language_probs(model, y))
                language = max(posteriors, key=posteriors.get) if posteriors else None
                language_probability = posteriors.get(language, 0.0)
            except Exception:  # fall back to faster-whisper's own detection
                log.debug("language detection failed", exc_info=True)
        clip_s = len(y) / 16000
        # The whole recording is transcribed: VAD is only the gate above (it under-detects singing).
        first = _decode(model, y, language, DECODE_OPTIONS, "first")
        if not language_probability:
            language_probability = first.language_probability
        passes = [first]
        if posteriors and first.language not in SUNG_LANGUAGES and language_probability < ALTERNATIVE_MAX_POSTERIOR:
            # Sung Spanish is often taken for Portuguese / French...: also decode in the likelier of es / en and
            # keep the decoding the model is more confident about (a real Portuguese song keeps Portuguese).
            alt_language = max(SUNG_LANGUAGES, key=lambda lang: posteriors.get(lang, 0.0))
            passes.append(_decode(model, y, alt_language, DECODE_OPTIONS, "alternative"))
        chosen = best_usable(passes, clip_s)
        if chosen is None:
            # The VAD heard a voice but the strict filter left nothing (typically a sung chorus the decoder looped,
            # or a whole-window hallucination): decode again (greedy, same language). The encoder output is
            # cached, so the retry costs one decoder pass.
            retry = _decode(model, y, first.language, RETRY_OPTIONS, "retry")
            passes.append(retry)
            chosen = fallback(first, retry, clip_s)
        if chosen is not None:
            chosen[0].used = True
        used, result = chosen if chosen is not None else (None, first.result(clip_s))
        if used is not None and used.language != language and used.language in posteriors:
            language_probability = posteriors[used.language]
        summaries = [p.summary(clip_s) for p in passes]
        log.debug("ASR passes: %s", summaries)
        return Transcript(
            result.text if result.detected else "", used.language if used is not None else None,
            round(float(language_probability), 4), result.detected, "" if result.detected else first.reason(clip_s),
            result.kept, self.model_size, time.perf_counter() - t0, summaries,
        )


# ----------------------------------------------------------------------------- decoding passes
@dataclass
class _Result:
    kept: list[dict]
    dropped: list[dict]
    text: str
    detected: bool
    relaxed: bool = False


@dataclass
class _Pass:
    """One Whisper decoding of the clip: its raw segments, and the strict / relaxed filter results."""

    name: str
    language: str | None
    language_probability: float
    options: dict
    raw: list[dict]
    elapsed_s: float
    used: bool = False
    agreement: float | None = None    # relaxed tier: agreement of its relaxed text with the other decoding

    def result(self, clip_s: float, relaxed: bool = False) -> _Result:
        kept, dropped = filter_segments(self.raw, clip_s, relaxed=relaxed)
        text = " ".join(s["text"] for s in kept).strip()
        return _Result(kept, dropped, text, is_query(text, MIN_DISTINCT_WORDS_RELAXED if relaxed else 1), relaxed)

    def reason(self, clip_s: float) -> str:
        res = self.result(clip_s)
        if res.detected:
            return ""
        if not self.raw:
            return "no speech segments"
        return "only non-speech or hallucinated segments" if not res.kept else "too few words"

    def summary(self, clip_s: float) -> dict:
        """Diagnostics without any text: language, segment counts, the rule that dropped each segment."""
        strict = self.result(clip_s)
        relaxed = self.result(clip_s, relaxed=True) if not strict.detected else strict
        rules: dict[str, int] = {}
        for seg in self.raw:
            rule = drop_rule(seg, clip_s)
            if rule:
                rules[rule] = rules.get(rule, 0) + 1
        return {"pass": self.name, "language": self.language, "beam_size": self.options.get("beam_size"),
                "segments": len(self.raw), "kept": len(strict.kept), "dropped": rules,
                "usable": strict.detected, "usable_relaxed": relaxed.detected, "agreement": self.agreement,
                "used": self.used, "relaxed": self.used and not strict.detected, "elapsed_s": round(self.elapsed_s, 3)}


def best_usable(passes: list[_Pass], clip_s: float) -> tuple[_Pass, _Result] | None:
    """The decoding with a usable strict transcript and the highest :func:`decoding_score` (None: none usable)."""
    usable = [(p, r) for p in passes for r in [p.result(clip_s)] if r.detected]
    return max(usable, key=lambda pr: decoding_score(pr[0].raw, clip_s)) if usable else None


def fallback(first: _Pass, retry: _Pass, clip_s: float) -> tuple[_Pass, _Result] | None:
    """After a failed first decoding: the retry's strict transcript, if it has ``MIN_DISTINCT_WORDS_RELAXED``
    distinct words or the first decoding heard its words too (``MIN_AGREEMENT``); else a relaxed one (repetitive
    windows kept) that the other decoding agrees with, the first decoding's before the retry's.

    The retry's few-distinct-words case needs the agreement: greedy decoding of sung vocables (the first decoding
    a vocable loop, no word in it) returned one invented word three times, while a real chorus that repeats one
    word (dev song, synthetic singing) was heard as that word by both decodings."""
    for p, other, relaxed in ((retry, first, False), (first, retry, True), (retry, first, True)):
        res = p.result(clip_s, relaxed)
        if not res.detected:
            continue
        if not relaxed and is_query(res.text, MIN_DISTINCT_WORDS_RELAXED):
            return p, res
        p.agreement = round(agreement(res.text, other.raw), 3)
        if p.agreement >= MIN_AGREEMENT:
            return p, res
    return None


def token_budget(clip_s: float) -> int:
    """``max_new_tokens`` for one 30-s window of a clip of ``clip_s`` seconds (see ``MAX_TOKENS_PER_S``)."""
    return min(MAX_NEW_TOKENS, int(MAX_TOKENS_PER_S * min(clip_s, 30.0)) + 24)


def _decode(model, y: np.ndarray, language: str | None, options: dict, name: str) -> _Pass:
    t0 = time.perf_counter()
    options = {"max_new_tokens": token_budget(len(y) / 16000), **options}
    segments, info = model.transcribe(y, language=language, vad_filter=False, **options)
    raw = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
            "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3),
            "compression_ratio": round(s.compression_ratio, 3)} for s in segments]   # consumes the generator
    return _Pass(name, language or info.language, float(info.language_probability), options, raw,
                 time.perf_counter() - t0)


# ----------------------------------------------------------------------------- helpers (pure, unit-tested)
def vad_speech_seconds(y: np.ndarray, sr: int = 16000) -> float:
    """Seconds of speech found by Silero VAD (with its default 400 ms padding around each region)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return sum(c["end"] - c["start"] for c in get_speech_timestamps(y, VadOptions(**VAD_PARAMETERS))) / sr


def detect_language_probs(model, y: np.ndarray) -> list[tuple[str, float]]:
    """``p(language | audio)`` on exactly the window that the first decoding encodes, so that the model's
    :class:`EncoderCache` serves that decoding (``transcribe()`` drops the last mel frame of the clip, whereas
    ``detect_language(audio=...)`` keeps it and so would encode a different window)."""
    extractor = getattr(model, "feature_extractor", None)
    if extractor is None:
        return list(model.detect_language(audio=y)[2])
    features = extractor(y)
    return list(model.detect_language(features=features[:, : features.shape[-1] - 1])[2])


def choose_language(probs: list[tuple[str, float]], prior: dict[str, float] | None = None,
                    other: float = OTHER_LANGUAGE_PRIOR) -> tuple[str | None, float]:
    """Pick ``argmax p(lang) * prior(lang)``; returns (language, posterior probability under the prior)."""
    prior = LANGUAGE_PRIOR if prior is None else prior
    weighted = [(lang, p * prior.get(lang, other)) for lang, p in probs]
    total = sum(w for _, w in weighted)
    if not weighted or total <= 0:
        return None, 0.0
    lang, w = max(weighted, key=lambda lw: lw[1])
    return lang, w / total


def language_posteriors(probs: list[tuple[str, float]], prior: dict[str, float] | None = None,
                        other: float = OTHER_LANGUAGE_PRIOR) -> dict[str, float]:
    """``p(lang | audio) * prior(lang)``, normalised (the argmax is :func:`choose_language`'s choice)."""
    prior = LANGUAGE_PRIOR if prior is None else prior
    weighted = {lang: p * prior.get(lang, other) for lang, p in probs}
    total = sum(weighted.values())
    return {lang: w / total for lang, w in weighted.items()} if total > 0 else {}


def mean_logprob(segments: list[dict]) -> float:
    """Duration-weighted mean ``avg_logprob`` of segments (-inf for none)."""
    if not segments:
        return float("-inf")
    w = np.array([max(0.1, s["end"] - s["start"]) for s in segments])
    return float(np.sum(w * np.array([s["avg_logprob"] for s in segments])) / w.sum())


# Drop rules that say the decoder failed on that stretch of audio (as opposed to: it heard no words there).
FAILED_RULES = ("low_logprob", "repetition", "loop", "hallucination")


def decoding_score(raw: list[dict], clip_s: float | None = None) -> float:
    """Which of several usable decodings of one clip to trust: the duration-weighted mean ``avg_logprob`` of its
    kept segments AND of the segments the strict filter dropped as failures (``FAILED_RULES``), each of these
    counted as at most ``MIN_AVG_LOGPROB``. Segments with no words to lose (no speech, vocables, annotations)
    do not count. Comparing only the kept segments would let a decoding that kept one fluent line and dropped
    the rest as garbage beat a complete transcript of the same audio (-inf when nothing counts)."""
    w, lp = [], []
    for seg in raw:
        rule = drop_rule(seg, clip_s)
        if rule and rule not in FAILED_RULES:
            continue
        end = seg["end"] if clip_s is None else min(seg["end"], clip_s)
        w.append(max(0.1, end - seg["start"]))
        lp.append(seg["avg_logprob"] if not rule else min(seg["avg_logprob"], MIN_AVG_LOGPROB))
    if not w:
        return float("-inf")
    return float(np.dot(w, lp) / np.sum(w))


def is_query(text: str, min_distinct: int = 1) -> bool:
    """Enough words to search lyrics with: ``MIN_LEXICAL_WORDS`` non-vocable words, ``min_distinct`` of them
    distinct."""
    words = lexical_words(text)
    return len(words) >= MIN_LEXICAL_WORDS and len(set(words)) >= min_distinct


def agreement(text: str, other_raw: list[dict]) -> float:
    """Share of the distinct words of ``text`` that another decoding of the same audio also produced (in any of
    its raw segments): two decodings agreeing on the words is evidence that they are really sung."""
    words = set(lexical_words(text))
    if not words:
        return 0.0
    heard = set(lexical_words(" ".join(strip_annotations(s["text"]) for s in other_raw)))
    return len(words & heard) / len(words)


def normalize(text: str) -> str:
    """Casefold, fold accents, punctuation -> spaces (for hallucination / vocable checks)."""
    t = unicodedata.normalize("NFKD", text.casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(_NON_WORD.sub(" ", _APOSTROPHE.sub("", t)).split())  # "I'll" -> "ill"


def strip_annotations(text: str) -> str:
    """Remove "[Música]", "(risas)", "*sighs*" and music symbols that Whisper emits around lyrics."""
    return " ".join(_ANNOTATION.sub(" ", text).split())


def collapse_repeats(text: str, max_repeats: int = 2, max_n: int = 6) -> str:
    """Collapse runs of a repeated n-gram ("a b a b a b") to ``max_repeats`` copies.

    Single words keep one more copy, as lyrics often say "oh oh oh" / "no no no".
    """
    words = text.split()
    for n in range(1, max_n + 1):
        limit = max_repeats + (n == 1)
        out, i = [], 0
        while i < len(words):
            gram = [w.strip(",.!?¡¿;:").casefold() for w in words[i:i + n]]
            reps = 1
            while len(gram) == n and [w.strip(",.!?¡¿;:").casefold()
                                      for w in words[i + reps * n:i + (reps + 1) * n]] == gram:
                reps += 1
            if len(gram) == n and reps > limit:
                out += words[i:i + n * limit]
                i += reps * n
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)


def is_vocable(word: str) -> bool:
    return word in VOCABLES or (len(word) >= 3 and _ELONGATED.match(word) is not None)


def lexical_words(text: str) -> list[str]:
    """Alphabetic words that are not vocables (hums, "la la", "oh", "mm-hmm")."""
    return [w for w in re.findall(r"[^\W\d_]+", normalize(text)) if not is_vocable(w)]


def is_hallucination(text: str) -> bool:
    norm = normalize(text)
    padded = f" {norm} "
    return norm in HALLUCINATIONS or any(m in padded for m in HALLUCINATION_MARKERS)


def collapse_all_repeats(text: str, max_n: int = 6) -> str:
    """Every consecutively repeated n-gram reduced to ONE copy ("thank you thank you thank you" -> "thank you").
    Only for recognising a looped hallucination; transcripts keep :func:`collapse_repeats`."""
    words = text.split()
    for n in range(1, max_n + 1):
        out, i = [], 0
        while i < len(words):
            if i + 2 * n <= len(words) and words[i:i + n] == words[i + n:i + 2 * n]:
                i += n                      # drop this copy, the next one stays
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)


def words_per_second(seg: dict, clip_s: float | None = None) -> float:
    """Words emitted per second of audio by a segment (its end clipped to the clip: a loop may claim 30 s, and a
    segment that starts after the end of the audio is a loop by definition)."""
    words = len(normalize(strip_annotations(seg["text"])).split())
    if clip_s is not None and seg["start"] >= clip_s:
        return float("inf") if words else 0.0
    end = seg["end"] if clip_s is None else min(seg["end"], clip_s)
    return words / max(1.0, end - seg["start"])


def drop_rule(seg: dict, clip_s: float | None = None, relaxed: bool = False) -> str:
    """Why a raw segment is dropped ("" = kept): ``no_speech``, ``low_logprob``, ``repetition`` (compression
    ratio; in the relaxed tier only ``loop``: too many words per second), ``empty``, ``hallucination``,
    ``vocables``."""
    lp, nsp, cr = seg["avg_logprob"], seg["no_speech_prob"], seg["compression_ratio"]
    if nsp > MAX_NO_SPEECH_PROB and lp < NO_SPEECH_MAX_LOGPROB:
        return "no_speech"
    if lp < MIN_AVG_LOGPROB:
        return "low_logprob"
    repetitive = cr > MAX_COMPRESSION_RATIO
    if repetitive:
        if not relaxed:
            return "repetition"
        if words_per_second(seg, clip_s) > MAX_LOOP_WORDS_PER_S:
            return "loop"
    text = collapse_repeats(strip_annotations(seg["text"]))
    if not text:
        return "empty"
    if is_hallucination(text) or (repetitive and is_hallucination(collapse_all_repeats(normalize(text)))):
        return "hallucination"
    if not lexical_words(text):
        return "vocables"
    return ""


def filter_segments(segments: list[dict], clip_s: float | None = None,
                    relaxed: bool = False) -> tuple[list[dict], list[dict]]:
    """Split raw Whisper segments into (kept, dropped); kept texts are cleaned (annotations, repetition).

    ``relaxed`` keeps windows above ``MAX_COMPRESSION_RATIO`` whose word rate is plausible for singing (a repeated
    chorus line) with their repetition collapsed; it is only used when the strict filter leaves nothing.
    """
    kept, dropped = [], []
    for seg in segments:
        if drop_rule(seg, clip_s, relaxed):
            dropped.append(seg)
        else:
            kept.append({**seg, "text": collapse_repeats(strip_annotations(seg["text"]))})
    return kept, dropped
