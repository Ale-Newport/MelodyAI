"""Whisper model selection ("auto" = best cached model), settings validation and the decoding rules for sung
audio (segment filter tiers, second decoding, encoder cache, token budget).

No model is loaded or downloaded: the Hugging Face cache lookup is monkeypatched and Whisper / Silero VAD are fakes
(the real-model test is marked ``asr``). All lyric lines are invented.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from melodyai.lyrics import asr
from melodyai.lyrics.asr import (
    AUTO,
    AUTO_DOWNLOAD,
    AUTO_PREFERENCE,
    MODEL_CHOICES,
    MODEL_SIZES,
    WhisperTranscriber,
    best_cached_model,
    model_status,
    resolve_model,
)


@pytest.fixture()
def cache(monkeypatch):
    """Set of model sizes that pretend to be in the local cache (and an empty in-process model cache, so a
    model loaded by an earlier test cannot answer for a model that is "not downloaded" here)."""
    cached: set[str] = set()
    monkeypatch.setattr(asr, "asr_available", lambda: True)
    monkeypatch.setattr(asr, "_cached", lambda size: size in cached)
    monkeypatch.setattr(WhisperTranscriber, "_models", {})
    return cached


def test_model_lists():
    assert "large-v3-turbo" in MODEL_SIZES
    assert MODEL_CHOICES[0] == AUTO and set(MODEL_CHOICES[1:]) == set(MODEL_SIZES)
    assert set(AUTO_PREFERENCE) == set(MODEL_SIZES)
    assert AUTO_PREFERENCE[0] == "large-v3-turbo" and AUTO_DOWNLOAD == "large-v3-turbo"


@pytest.mark.parametrize("cached,best", [
    ({"base", "large-v3-turbo"}, "large-v3-turbo"),
    ({"base", "large-v3"}, "large-v3"),
    ({"tiny", "small", "medium"}, "medium"),
    ({"tiny", "small"}, "small"),
    ({"tiny", "base"}, "base"),
    ({"tiny"}, "tiny"),
    (set(), None),
])
def test_best_cached_model_preference(cache, cached, best):
    cache.update(cached)
    assert best_cached_model() == best
    assert resolve_model(AUTO) == best


def test_resolve_auto_without_cached_models(cache):
    assert resolve_model(AUTO) is None
    assert resolve_model(AUTO, allow_download=True) == AUTO_DOWNLOAD


def test_resolve_explicit_sizes(cache):
    cache.add("large-v3-turbo")
    for size in MODEL_SIZES:  # explicit choices are honoured even when not cached
        assert resolve_model(size) == size
    with pytest.raises(ValueError):
        resolve_model("large-v2")
    with pytest.raises(ValueError):
        resolve_model("turbo")


def test_model_status_every_name(cache):
    cache.update({"base", "large-v3-turbo"})
    for name in MODEL_CHOICES:
        st = model_status(name)
        assert st["name"] == name and st["installed_package"] is True
        assert st["cached"] is (name in ("auto", "base", "large-v3-turbo"))
    assert model_status(AUTO)["resolved"] == "large-v3-turbo"
    cache.clear()
    assert model_status(AUTO) == {"name": AUTO, "installed_package": True, "cached": False, "resolved": None}


def test_model_status_without_package(monkeypatch):
    monkeypatch.setattr(asr, "asr_available", lambda: False)
    monkeypatch.setattr(asr, "_cached", lambda size: True)
    assert model_status("base")["cached"] is False  # files alone are useless without faster-whisper
    assert model_status(AUTO) == {"name": AUTO, "installed_package": False, "cached": False, "resolved": None}


def test_transcriber_resolves_auto(cache):
    cache.update({"base", "large-v3-turbo"})
    tr = WhisperTranscriber()
    assert tr.requested == AUTO and tr.model_size == "large-v3-turbo"
    assert WhisperTranscriber("base").model_size == "base"
    with pytest.raises(ValueError):
        WhisperTranscriber("huge")


def test_transcriber_auto_with_empty_cache_names_recommended_model(cache):
    pytest.importorskip("faster_whisper")
    tr = WhisperTranscriber(AUTO)
    assert tr.model_size == AUTO_DOWNLOAD
    with pytest.raises(RuntimeError, match="make asr-model MODEL=large-v3-turbo"):
        tr._model()


def test_models_loaded_by_earlier_tests_do_not_leak(monkeypatch, request):
    """Regression: an app built by an earlier test (auto mode, a real cached model) filled the class-level
    model cache, and the "not downloaded" test above then got that model back instead of the error."""
    pytest.importorskip("faster_whisper")
    process_wide: dict = {}
    monkeypatch.setattr(WhisperTranscriber, "_models", process_wide)    # as an earlier test left it
    request.getfixturevalue("cache")
    tr = WhisperTranscriber(AUTO)
    process_wide[(tr.model_size, tr.device, tr.compute_type)] = object()
    with pytest.raises(RuntimeError, match="not downloaded"):
        tr._model()


def test_engine_asr_ready_resolves_auto(cache, monkeypatch, melody_index):
    from melodyai import engine as engine_mod
    from melodyai.engine import EngineSettings, RecognitionEngine
    from melodyai.lyrics.search import LyricsIndex

    monkeypatch.setattr(engine_mod, "asr_available", lambda: True)
    assert EngineSettings().asr_model == AUTO
    eng = RecognitionEngine(melody_index, LyricsIndex(":memory:"), EngineSettings())
    assert eng.asr_ready() is False  # nothing cached, downloads not allowed
    cache.add("base")
    assert eng.asr_ready() is True
    eng.settings.asr_model = "medium"
    assert eng.asr_ready() is False
    eng.settings.asr_allow_download = True
    assert eng.asr_ready() is True
    eng.settings.asr_model = "not-a-model"
    assert eng.asr_ready() is False
    eng.settings.asr_model, eng.settings.asr_enabled = AUTO, False
    assert eng.asr_ready() is False


def test_user_settings_asr_model():
    from app.services.user_settings import UserSettings

    assert UserSettings().asr_model == AUTO
    for name in MODEL_CHOICES:
        assert UserSettings(asr_model=name).asr_model == name
    for bad in ("large-v2", "turbo", "auto ", "Large-v3", "large-v3-turbo2", ""):
        with pytest.raises(ValidationError):
            UserSettings(asr_model=bad)


def test_lab_download_body_accepts_turbo():
    from app.api.lab import AsrDownloadBody

    assert AsrDownloadBody(model="large-v3-turbo").model == "large-v3-turbo"
    with pytest.raises(ValidationError):
        AsrDownloadBody(model="auto")


# ----------------------------------------------------------------------------- decoding helpers
def test_choose_language_soft_prior():
    from melodyai.lyrics.asr import choose_language

    # Sung Spanish is often mis-detected as Portuguese / Icelandic with a low margin -> the prior fixes it.
    assert choose_language([("pt", 0.40), ("es", 0.30), ("gl", 0.1)])[0] == "es"
    assert choose_language([("is", 0.55), ("es", 0.20), ("en", 0.1)])[0] == "es"
    # ...but a clear detection of another language still wins.
    assert choose_language([("pt", 0.95), ("es", 0.02)])[0] == "pt"
    assert choose_language([("ko", 0.97), ("en", 0.01), ("es", 0.01)])[0] == "ko"
    lang, p = choose_language([("en", 0.5), ("es", 0.5)], prior={"en": 1.0, "es": 1.0})
    assert lang in ("en", "es") and p == pytest.approx(0.5)
    assert choose_language([]) == (None, 0.0)


def test_text_cleanup_helpers():
    from melodyai.lyrics.asr import collapse_repeats, is_hallucination, lexical_words, strip_annotations

    assert strip_annotations("[Música] ♪ bailando en la azotea ♪ (aplausos)") == "bailando en la azotea"
    assert collapse_repeats("na na na na na hey") == "na na na hey"
    assert collapse_repeats("oh oh oh tonight") == "oh oh oh tonight"
    assert collapse_repeats("I love you, I love you, I love you, I love you so much") == \
        "I love you, I love you, so much"
    assert collapse_repeats("hold me closer, hold me closer") == "hold me closer, hold me closer"
    assert lexical_words("Mm-hmm. Hmmmm, aaah la la la ooh") == []
    assert lexical_words("Oh, la noche brilla") == ["noche", "brilla"]
    for text in ("Thank you.", "¡Suscríbete al canal!", "Subtítulos realizados por la comunidad de Amara.org",
                 "Sous-titrage Société Radio-Canada", "Субтитры создавал DimaTorzok", "¡Aplausos!",
                 "I'll see you next time.", "Thank you so much for joining us today, and we'll see you next time."):
        assert is_hallucination(text), text
    assert not is_hallucination("thank you for the summer nights")


def _seg(text, lp=-0.3, nsp=0.01, cr=1.2):
    return {"start": 0.0, "end": 2.0, "text": text, "avg_logprob": lp, "no_speech_prob": nsp, "compression_ratio": cr}


def test_filter_segments():
    from melodyai.lyrics.asr import filter_segments

    segs = [
        _seg("♪ Bailando en la azotea hasta que salga el sol ♪"),
        _seg("Thank you."),
        _seg("Mm-hmm."),
        _seg("[Música]"),
        _seg("words with a very low log probability", lp=-1.5),
        _seg("probably silence here", lp=-0.8, nsp=0.9),
        _seg("I'm your host, " * 20, lp=-0.1, cr=15.7),
        _seg("na na na na na na corazón", cr=2.2),
    ]
    kept, dropped = filter_segments(segs)
    assert [k["text"] for k in kept] == ["Bailando en la azotea hasta que salga el sol", "na na na corazón"]
    assert len(dropped) == 6


class _FakeSegment:
    def __init__(self, text, lp=-0.3, nsp=0.01, cr=1.2, start=0.0, end=3.0):
        self.start, self.end, self.text = start, end, text
        self.avg_logprob, self.no_speech_prob, self.compression_ratio = lp, nsp, cr


class _FakeWhisper:
    """``texts``: the segments of every decoding, or (``per_call=True``) one list of segments per decoding, the
    last one repeated. A segment is a text or a (text, {"lp", "nsp", "cr", "start", "end"}) pair."""

    def __init__(self, probs, texts, per_call=False):
        self.probs, self.calls = probs, []
        self.by_language = texts if isinstance(texts, dict) else None   # {language or (language, beam): segments}
        self.script = texts if per_call else [texts]

    def detect_language(self, audio=None, features=None, vad_filter=False, vad_parameters=None):
        self.detected_on = audio if audio is not None else features
        return self.probs[0][0], self.probs[0][1], self.probs

    def transcribe(self, audio, language=None, **kw):
        from types import SimpleNamespace

        if self.by_language is not None:
            segs = self.by_language.get((language, kw.get("beam_size")), self.by_language.get(language, []))
        else:
            segs = self.script[min(len(self.calls), len(self.script) - 1)]
        self.calls.append({"language": language, **kw})
        out = [_FakeSegment(t) if isinstance(t, str) else _FakeSegment(t[0], **t[1]) for t in segs]
        return iter(out), SimpleNamespace(language=language or "en", language_probability=0.9)


@pytest.fixture()
def fake_whisper(cache, monkeypatch):
    """Fake WhisperModel + fake Silero VAD: the VAD reports speech on the samples whose |y| > 0.05."""
    fw = pytest.importorskip("faster_whisper.vad")
    cache.add("large-v3-turbo")

    def fake_vad(audio, options):
        idx = np.flatnonzero(np.abs(audio) > 0.05)
        return [{"start": int(idx[0]), "end": int(idx[-1]) + 1}] if len(idx) else []

    monkeypatch.setattr(fw, "get_speech_timestamps", fake_vad)

    def install(probs, texts, per_call=False):
        model = _FakeWhisper(probs, texts, per_call)
        monkeypatch.setattr(WhisperTranscriber, "_model", lambda self: model)
        return model

    return install


def _audio(speech_s: float, total_s: float = 6.0) -> np.ndarray:
    y = np.zeros(int(total_s * 16000), np.float32)
    y[: int(speech_s * 16000)] = 0.1
    return y


def test_transcribe_applies_language_prior(fake_whisper):
    model = fake_whisper([("pt", 0.5), ("es", 0.4)], ["Guardo tus cartas en un cajón junto al mar", "Thank you."])
    tr = WhisperTranscriber().transcribe(_audio(4.0), 16000)
    # VAD is only a gate: the whole recording is transcribed (VAD under-detects singing)
    assert model.calls[0]["language"] == "es" and model.calls[0]["vad_filter"] is False
    assert len(model.detected_on) == 6 * 16000
    assert tr.speech_detected and tr.language == "es" and tr.model == "large-v3-turbo"
    assert tr.text == "Guardo tus cartas en un cajón junto al mar"
    assert 0.5 < tr.language_probability <= 1.0 and tr.reliability > 0


@pytest.mark.parametrize("speech_s", [0.0, 1.0])
def test_transcribe_skips_whisper_without_enough_vad_speech(fake_whisper, speech_s):
    from melodyai.lyrics.asr import MIN_VAD_SPEECH_S

    assert speech_s < MIN_VAD_SPEECH_S
    model = fake_whisper([("en", 0.9)], ["hallucinated words here"])
    tr = WhisperTranscriber().transcribe(_audio(speech_s), 16000)
    assert not model.calls
    assert not tr.speech_detected and tr.reason.startswith("no speech (VAD") and tr.text == ""


def test_transcribe_given_language_and_vocables(fake_whisper):
    model = fake_whisper([("pt", 0.9)], ["Mm-hmm.", "La la la la."])
    tr = WhisperTranscriber().transcribe(_audio(4.0), 16000, language="en")
    assert model.calls[0]["language"] == "en" and not hasattr(model, "detected_on")
    assert not tr.speech_detected and tr.reason == "only non-speech or hallucinated segments"


def test_transcribe_requires_three_lexical_words(fake_whisper):
    fake_whisper([("en", 0.9)], ["All right."])  # Whisper's answer to a sung "aaah"
    tr = WhisperTranscriber().transcribe(_audio(4.0), 16000)
    assert not tr.speech_detected and tr.reason == "too few words"


# ----------------------------------------------------------------------------- sung audio: filter tiers, second pass
_CHORUS = "Lámparas de papel, lámparas de papel, lámparas de papel en el río"        # invented, repeats a phrase
_LOOP = "Te espero en la estación, " * 12                                           # invented decoder loop


def test_drop_rule_names():
    from melodyai.lyrics.asr import drop_rule

    assert drop_rule(_seg("probably silence here", lp=-0.8, nsp=0.9)) == "no_speech"
    assert drop_rule(_seg("words with a very low log probability", lp=-1.5)) == "low_logprob"
    assert drop_rule(_seg(_CHORUS, cr=3.1)) == "repetition"
    assert drop_rule(_seg("[Música] ♪")) == "empty"
    assert drop_rule(_seg("Thank you.")) == "hallucination"
    assert drop_rule(_seg("La la la la.")) == "vocables"
    assert drop_rule(_seg("Guardo tus cartas en un cajón junto al mar")) == ""


def test_words_per_second_clips_the_segment_end():
    from melodyai.lyrics.asr import words_per_second

    loop = {**_seg(_LOOP), "end": 30.0}                     # a loop often claims the whole 30-s window
    assert words_per_second(loop) == pytest.approx(60 / 30)
    assert words_per_second(loop, clip_s=7.0) == pytest.approx(60 / 7)
    assert words_per_second({**_seg("dos palabras"), "start": 1.0, "end": 1.2}) == 2.0   # at least 1 s


def test_relaxed_filter_keeps_repeated_chorus_but_not_loops():
    from melodyai.lyrics.asr import MAX_LOOP_WORDS_PER_S, drop_rule, filter_segments, words_per_second

    chorus = {**_seg(_CHORUS, cr=3.1), "end": 5.0}           # 13 words in 5 s: someone singing
    loop = {**_seg(_LOOP, lp=-0.1, cr=9.5), "end": 30.0}     # 60 words in a 7-s clip: the decoder looping
    assert words_per_second(chorus, 7.0) < MAX_LOOP_WORDS_PER_S < words_per_second(loop, 7.0)
    assert filter_segments([chorus, loop], 7.0) == ([], [chorus, loop])      # strict: both are repetitive
    kept, dropped = filter_segments([chorus, loop], 7.0, relaxed=True)
    assert [k["text"] for k in kept] == ["Lámparas de papel, lámparas de papel, en el río"] and dropped == [loop]
    assert drop_rule(loop, 7.0, relaxed=True) == "loop"
    # the other rules still apply in the relaxed tier, also to a looped hallucination or vocables
    for text in ("Thank you. Thank you. Thank you. Thank you.", "Aplausos. Aplausos. Aplausos.",
                 "la la la la la la la la la la"):
        seg = {**_seg(text, cr=4.0), "end": 6.0}
        assert drop_rule(seg, 7.0, relaxed=True) in ("hallucination", "vocables"), text
    assert drop_rule({**_seg("Thank you. Thank you.", cr=1.5)}, 7.0, relaxed=True) == ""  # not repetitive: as before


def test_token_budget():
    from melodyai.lyrics.asr import MAX_NEW_TOKENS, MAX_TOKENS_PER_S, token_budget

    assert token_budget(7.0) == int(MAX_TOKENS_PER_S * 7.0) + 24 < 448     # a loop stops long before 448 tokens
    assert token_budget(0.5) >= 24 and token_budget(30.0) == token_budget(300.0) == MAX_NEW_TOKENS <= 440
    # sung lyrics need 3-6 tokens/s; a loop cut at the budget must still look like a loop (> MAX_LOOP_WORDS_PER_S
    # words/s with up to 3 tokens per word, e.g. "I'm your host, I'm your host, ...")
    from melodyai.lyrics.asr import MAX_LOOP_WORDS_PER_S

    assert MAX_TOKENS_PER_S / 3 > MAX_LOOP_WORDS_PER_S


def test_collapse_all_repeats():
    from melodyai.lyrics.asr import collapse_all_repeats

    assert collapse_all_repeats("thank you thank you thank you") == "thank you"
    assert collapse_all_repeats("aplausos aplausos aplausos") == "aplausos"
    assert collapse_all_repeats("la noche brilla") == "la noche brilla"


def test_query_words():
    from melodyai.lyrics.asr import MIN_DISTINCT_WORDS_RELAXED, is_query

    assert not is_query("All right.") and not is_query("la la la oh oh mm")
    assert is_query("Guardo tus cartas en un cajón") and is_query("ven, ven, ven, oh oh")   # a chorus may repeat
    # the relaxed tier (repetitive windows) needs distinct words: a loop on two words is not a chorus
    assert not is_query("Oh, I'm sorry, I'm sorry, I'm sorry", min_distinct=MIN_DISTINCT_WORDS_RELAXED)
    assert is_query("corazón corazón de papel", min_distinct=MIN_DISTINCT_WORDS_RELAXED)


def test_agreement_between_decodings():
    from melodyai.lyrics.asr import agreement

    other = [_seg("♪ Lámparas de papel ♪"), _seg("flotando en el río, oh")]
    assert agreement("lámparas de papel en el río", other) == 1.0
    assert agreement("lámparas de cartón", other) == pytest.approx(2 / 3)     # "de" counts as a word too
    assert agreement("la la la", other) == 0.0 and agreement("nada", []) == 0.0


def test_language_posteriors_and_mean_logprob():
    from melodyai.lyrics.asr import choose_language, language_posteriors, mean_logprob

    probs = [("pt", 0.40), ("es", 0.30), ("gl", 0.1)]
    post = language_posteriors(probs)
    assert max(post, key=post.get) == choose_language(probs)[0] == "es"
    assert sum(post.values()) == pytest.approx(1.0) and post["es"] == pytest.approx(choose_language(probs)[1])
    assert language_posteriors([]) == {}
    segs = [{**_seg("a", lp=-0.2), "end": 3.0}, {**_seg("b", lp=-1.0), "start": 3.0, "end": 4.0}]
    assert mean_logprob(segs) == pytest.approx((3 * -0.2 + 1 * -1.0) / 4) and mean_logprob([]) == float("-inf")


def test_alternative_sung_language_decoding(fake_whisper):
    """Sung Spanish taken for Portuguese: also decode in es (the likelier of es/en) and keep the likelier text."""
    pt = ("Não hagas caso de los cartas junto ao mar", {"lp": -0.9})
    es = ("Guardo tus cartas en un cajón junto al mar", {"lp": -0.3})
    model = fake_whisper([("pt", 0.8), ("en", 0.05), ("es", 0.1)], {"pt": [pt], "es": [es]})
    tr = WhisperTranscriber().transcribe(_audio(4.0), 16000)
    assert [c["language"] for c in model.calls] == ["pt", "es"] and model.calls[1]["beam_size"] == 5
    assert tr.speech_detected and tr.language == "es" and tr.text == es[0]
    assert [(p["pass"], p["used"]) for p in tr.passes] == [("first", False), ("alternative", True)]
    assert 0 < tr.language_probability < 0.5                       # the posterior of "es" under the prior
    # ... but a recording the model is more confident about in Portuguese stays Portuguese
    model = fake_whisper([("pt", 0.8), ("en", 0.05), ("es", 0.1)], {"pt": [(pt[0], {"lp": -0.2})], "es": [es]})
    tr = WhisperTranscriber().transcribe(_audio(4.0), 16000)
    assert len(model.calls) == 2 and tr.language == "pt" and tr.text == pt[0]


def test_alternative_decoding_that_dropped_most_of_the_clip_does_not_win(fake_whisper):
    """Regression: decodings were compared by the log-prob of their KEPT segments only, so an alternative decoding
    that kept one fluent line and dropped the rest as garbage replaced a complete transcript of the clip."""
    pt = [("A casa amarela dorme perto do rio", {"lp": -0.45, "start": 0.0, "end": 3.0}),
          ("Eu canto baixinho quando a noite chega", {"lp": -0.45, "start": 3.0, "end": 6.0}),
          ("E o vento leva os barcos para o mar", {"lp": -0.45, "start": 6.0, "end": 9.0})]
    es = [("Una casa amarilla junto al río", {"lp": -0.30, "start": 0.0, "end": 2.0}),
          ("garbled noises nothing understood here", {"lp": -1.5, "start": 2.0, "end": 6.0}),
          ("more garbled words for the rest", {"lp": -1.6, "start": 6.0, "end": 9.0})]
    model = fake_whisper([("pt", 0.8), ("es", 0.1), ("en", 0.05)], {"pt": pt, "es": es})
    tr = WhisperTranscriber().transcribe(_audio(9.0, 10.0), 16000)
    assert [c["language"] for c in model.calls] == ["pt", "es"]
    assert tr.language == "pt" and tr.text == " ".join(t for t, _ in pt)
    assert [(p["pass"], p["used"]) for p in tr.passes] == [("first", True), ("alternative", False)]


def test_decoding_score():
    from melodyai.lyrics.asr import MIN_AVG_LOGPROB, decoding_score

    def seg(text, lp, start, end, **kw):
        return {**_seg(text, lp=lp, **kw), "start": start, "end": end}

    kept = seg("Guardo tus cartas en un cajón junto al mar", -0.3, 0.0, 2.0)
    assert decoding_score([kept]) == pytest.approx(-0.3) and decoding_score([]) == float("-inf")
    # a stretch the decoder failed on counts with its own log-prob (low) or as MIN_AVG_LOGPROB (loop, credits)
    assert decoding_score([kept, seg("palabras sin sentido aquí", -1.6, 2.0, 4.0)]) == pytest.approx((-0.3 - 1.6) / 2)
    loop = seg(_LOOP, -0.1, 2.0, 30.0, cr=9.5)
    assert decoding_score([kept, loop], clip_s=4.0) == pytest.approx((-0.3 + MIN_AVG_LOGPROB) / 2)
    # no words to lose (vocables, annotations, silence): not counted
    for other in (seg("La la la la.", -0.2, 2.0, 4.0), seg("[Música]", -0.2, 2.0, 4.0),
                  seg("probably silence here", -0.8, 2.0, 4.0, nsp=0.9)):
        assert decoding_score([kept, other]) == pytest.approx(-0.3)


@pytest.mark.parametrize("probs,given", [([("pt", 0.99), ("es", 0.01)], None),     # clearly Portuguese
                                         ([("pt", 0.8), ("es", 0.1)], "pt"),        # language given by the user
                                         ([("es", 0.5), ("en", 0.4)], None)])       # already a sung language
def test_no_alternative_decoding(fake_whisper, probs, given):
    model = fake_whisper(probs, ["Guardo tus cartas en un cajón junto al mar"])
    tr = WhisperTranscriber().transcribe(_audio(4.0), 16000, language=given)
    assert len(model.calls) == 1 and tr.speech_detected and [p["pass"] for p in tr.passes] == ["first"]


def test_encoder_cache_memoises_identical_windows():
    from melodyai.lyrics.asr import EncoderCache

    calls = []

    def encode(features):
        calls.append(features.shape)
        return object()

    cache = EncoderCache(encode, size=2)
    a = np.random.default_rng(0).standard_normal((8, 300)).astype(np.float32)
    out = cache(a)
    assert cache(a.copy()) is out and len(calls) == 1                 # same bytes -> no second encoder pass
    assert cache(np.pad(a, ((0, 0), (0, 5)))[:, :300]) is out          # a non-contiguous view of the same values
    b = a.copy()
    b[3, 7] += 1e-3
    assert cache(b) is not out and len(calls) == 2                    # one different value -> encoded again
    assert cache(a[:, :299]) is not out and len(calls) == 3           # another shape
    assert cache(a) is not out and len(calls) == 4                    # LRU of 2: the first window was evicted
    assert (cache.hits, cache.misses) == (2, 4)


def test_language_detection_uses_the_decoded_window():
    from melodyai.lyrics.asr import detect_language_probs

    class Model:
        def feature_extractor(self, y):
            return np.zeros((4, len(y) // 160 + 1), np.float32)

        def detect_language(self, audio=None, features=None):
            self.audio, self.features = audio, features
            return "es", 0.9, [("es", 0.9), ("en", 0.1)]

    m = Model()
    assert detect_language_probs(m, np.zeros(16000, np.float32)) == [("es", 0.9), ("en", 0.1)]
    assert m.audio is None and m.features.shape == (4, 100)   # the last mel frame is dropped, as transcribe() does


def test_loaded_model_gets_an_encoder_cache(cache, monkeypatch):
    fw = pytest.importorskip("faster_whisper")
    from melodyai.lyrics.asr import EncoderCache

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, features):
            return "encoded"

    monkeypatch.setattr(fw, "WhisperModel", Model)
    cache.add("large-v3-turbo")
    model = WhisperTranscriber()._model()
    assert isinstance(model.encode, EncoderCache) and model.encode(np.zeros((2, 3), np.float32)) == "encoded"


def test_no_second_pass_when_the_first_is_usable(fake_whisper):
    model = fake_whisper([("es", 0.9)], ["Guardo tus cartas en un cajón junto al mar"])
    tr = WhisperTranscriber().transcribe(_audio(4.0), 16000)
    assert len(model.calls) == 1 and tr.speech_detected
    assert [(p["pass"], p["used"], p["relaxed"]) for p in tr.passes] == [("first", True, False)]


def test_second_pass_same_language_greedy(fake_whisper):
    from melodyai.lyrics.asr import DECODE_OPTIONS, RETRY_OPTIONS, token_budget

    model = fake_whisper([("pt", 0.5), ("es", 0.4)],
                         [["¡Suscríbete al canal!"], ["Guardo tus cartas en un cajón junto al mar"]], per_call=True)
    tr = WhisperTranscriber().transcribe(_audio(4.0), 16000)
    assert [c["language"] for c in model.calls] == ["es", "es"]           # the prior's choice, kept for the retry
    assert model.calls[0]["beam_size"] == DECODE_OPTIONS["beam_size"] == 5
    assert model.calls[1]["beam_size"] == RETRY_OPTIONS["beam_size"] == 1
    assert all(c["temperature"] == 0.0 and c["vad_filter"] is False for c in model.calls)
    assert all(c["max_new_tokens"] == token_budget(6.0) for c in model.calls)
    assert tr.speech_detected and tr.language == "es" and tr.text == "Guardo tus cartas en un cajón junto al mar"
    first, retry = tr.passes
    assert first["dropped"] == {"hallucination": 1} and not first["usable"] and not first["used"]
    assert retry["pass"] == "retry" and retry["used"] and retry["usable"] and not retry["relaxed"]
    assert "Guardo" not in str(tr.passes) and "canal" not in str(tr.passes)   # diagnostics carry no text


def test_relaxed_first_pass_when_the_retry_agrees(fake_whisper):
    chorus = (_CHORUS, {"cr": 3.1, "end": 5.0})
    looped = (_CHORUS + ", lámparas de papel, lámparas de papel", {"cr": 4.2, "end": 5.0})   # retry loops too
    model = fake_whisper([("es", 0.9)], [[chorus], [looped]], per_call=True)
    tr = WhisperTranscriber().transcribe(_audio(4.0), 16000)
    assert len(model.calls) == 2 and tr.speech_detected and tr.language == "es"
    assert tr.text == "Lámparas de papel, lámparas de papel, en el río"
    first, retry = tr.passes
    assert first["dropped"] == {"repetition": 1} and first["used"] and first["relaxed"] and first["agreement"] == 1.0
    assert retry["dropped"] == {"repetition": 1} and not retry["used"]


def test_relaxed_tier_needs_three_distinct_words(fake_whisper):
    """Both decodings looping on the same two words (Whisper on voice-like noise) is not a chorus."""
    loop = ("Oh, I'm sorry, I'm sorry, I'm sorry, I'm sorry, I'm sorry", {"cr": 3.5, "end": 6.0})
    model = fake_whisper([("en", 0.9)], [loop])
    tr = WhisperTranscriber().transcribe(_audio(4.0), 16000)
    assert len(model.calls) == 2 and not tr.speech_detected
    assert not any(p["usable_relaxed"] for p in tr.passes)


def test_relaxed_tier_needs_the_other_decoding_to_agree(fake_whisper):
    """A repeated phrase that only one decoding heard is a hallucination loop (reversed speech, vocables...)."""
    loop = ("Nunca vuelvas a la estación, nunca vuelvas a la estación, nunca vuelvas a la estación",
            {"cr": 3.2, "end": 6.0})
    other = ("Mi nombre es el viento del sur", {"lp": -1.4})
    for script in ([[loop], [other]], [[other], [loop]]):
        model = fake_whisper([("es", 0.9)], script, per_call=True)
        tr = WhisperTranscriber().transcribe(_audio(4.0), 16000)
        assert len(model.calls) == 2 and not tr.speech_detected and tr.text == ""
        assert [p["agreement"] for p in tr.passes if p["usable_relaxed"]] == [0.0]


def test_retry_with_few_distinct_words_needs_the_first_decoding_to_agree(fake_whisper):
    """Regression (bench clip of sung vocables): the first decoding was a vocable loop (no word in it), the greedy
    retry returned ONE invented word three times, which passed the strict filter and started a world search."""
    vocables = ("La la la la la la la la la la la la la la la la la la la la", {"lp": -0.1, "cr": 18.0, "end": 5.4})
    word3 = ("Lorumbando, lorumbando, lorumbando.", {"lp": -0.6})
    for first in ([vocables], ["¡Suscríbete al canal!"]):
        model = fake_whisper([("es", 0.9)], [first, [word3]], per_call=True)
        tr = WhisperTranscriber().transcribe(_audio(4.0), 16000)
        assert len(model.calls) == 2 and not tr.speech_detected and tr.text == "" and tr.language is None
        retry = tr.passes[1]
        assert retry["pass"] == "retry" and retry["usable"] and not retry["used"] and retry["agreement"] == 0.0
    # a chorus that repeats ONE word: the first decoding heard it too (and was dropped as a repetition)
    chorus = ("Volveré, volveré, volveré, volveré, volveré, volveré, volveré, volveré", {"cr": 2.9, "end": 6.4})
    fake_whisper([("es", 0.9)], [[chorus], [("Volveré, volveré, volveré", {"lp": -0.9})]], per_call=True)
    tr = WhisperTranscriber().transcribe(_audio(4.0, 7.0), 16000)
    assert tr.speech_detected and tr.text == "Volveré, volveré, volveré" and tr.passes[1]["agreement"] == 1.0
    # a retry with three distinct words needs no agreement (see test_second_pass_same_language_greedy)
    fake_whisper([("es", 0.9)], [[vocables], ["Lorumbando, lorumbando, ven conmigo"]], per_call=True)
    assert WhisperTranscriber().transcribe(_audio(4.0), 16000).speech_detected


def test_retry_preferred_over_relaxed_first_pass(fake_whisper):
    chorus = (_CHORUS, {"cr": 3.1, "end": 5.0})
    fake_whisper([("en", 0.9)], [[chorus], ["Guardo tus cartas en un cajón junto al mar"]], per_call=True)
    tr = WhisperTranscriber().transcribe(_audio(4.0), 16000)
    assert tr.text == "Guardo tus cartas en un cajón junto al mar" and [p["used"] for p in tr.passes] == [False, True]


def test_decoder_loops_never_become_a_transcript(fake_whisper):
    model = fake_whisper([("en", 0.9)], [(_LOOP, {"lp": -0.1, "cr": 9.5, "end": 30.0})])   # every decoding loops
    tr = WhisperTranscriber().transcribe(_audio(4.0), 16000)
    assert len(model.calls) == 2 and not tr.speech_detected and tr.text == "" and tr.language is None
    assert tr.reason == "only non-speech or hallucinated segments"
    assert [p["dropped"] for p in tr.passes] == [{"repetition": 1}, {"repetition": 1}]
    assert not any(p["usable_relaxed"] for p in tr.passes)


def test_default_compute_type(monkeypatch):
    from melodyai.lyrics.asr import default_compute_type

    monkeypatch.delenv("MELODYAI_ASR_COMPUTE_TYPE", raising=False)
    monkeypatch.setattr(asr.sys, "platform", "darwin")
    monkeypatch.setattr(asr.platform, "machine", lambda: "arm64")
    assert default_compute_type("cpu") == "float32"  # Accelerate-backed float32 beats int8 on Apple Silicon
    assert default_compute_type("cuda") == "int8"
    monkeypatch.setattr(asr.sys, "platform", "linux")
    monkeypatch.setattr(asr.platform, "machine", lambda: "x86_64")
    assert default_compute_type("cpu") == "int8"
    monkeypatch.setenv("MELODYAI_ASR_COMPUTE_TYPE", "int8_float32")
    assert default_compute_type("cpu") == "int8_float32"
    assert WhisperTranscriber("base", compute_type="int8").compute_type == "int8"


@pytest.mark.asr
@pytest.mark.slow
def test_real_model_lyrics_and_hum(melodies):
    """End-to-end with the best cached Whisper model (skipped without one or without macOS `say`)."""
    from melodyai.demo.queries import CONDITIONS, excerpt, synth_query
    from melodyai.evaluation.datasets import synthesize_speech, tts_available
    from melodyai.lyrics.asr import normalize

    if not (asr.asr_available() and tts_available() and best_cached_model()):
        pytest.skip("needs faster-whisper, a cached model and macOS `say`")
    tr = WhisperTranscriber(AUTO)
    audio = synthesize_speech("bailando en la azotea hasta que salga el sol, nadie nos dijo que se acaba la canción",
                              "es")
    encoder = tr._model().encode
    misses, hits = encoder.misses, encoder.hits
    out = tr.transcribe(audio.samples, audio.sr)
    assert out.speech_detected and out.language == "es" and out.model == best_cached_model()
    # one encoder pass: language detection and the decoding share the cached window
    assert encoder.misses == misses + 1 and encoder.hits >= hits + 1 and len(out.passes) == 1
    assert "azotea" in normalize(out.text) and "cancion" in normalize(out.text)
    rng = np.random.default_rng(3)
    for cond in ("hum_clean", "whistle"):
        notes, _ = excerpt(melodies[0].notes, rng, 6, 10)
        hum = synth_query(notes, CONDITIONS[cond], rng)
        assert not tr.transcribe(hum, 16000).speech_detected
