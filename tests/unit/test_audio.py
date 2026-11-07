import io
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from melodyai.audio.io import AudioError, decode_audio, wav_bytes
from melodyai.audio.preprocess import estimate_quality, preprocess, trim_silence


def tone(sr=16000, seconds=1.0, f=440.0, channels=1):
    t = np.arange(int(sr * seconds)) / sr
    y = 0.5 * np.sin(2 * np.pi * f * t).astype(np.float32)
    return np.stack([y] * channels, axis=1) if channels > 1 else y


def encode(y, sr, fmt, subtype=None):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


@pytest.mark.parametrize("fmt,subtype", [("WAV", "PCM_16"), ("FLAC", None), ("OGG", "VORBIS"), ("MP3", None)])
def test_decode_formats(fmt, subtype):
    data = encode(tone(), 16000, fmt, subtype)
    a = decode_audio(data)
    assert a.sr == 16000
    assert abs(a.duration - 1.0) < 0.1
    # The decoded signal still contains the 440 Hz tone.
    spec = np.abs(np.fft.rfft(a.samples[:8000]))
    assert abs(np.argmax(spec) * 16000 / 8000 - 440) < 5


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_m4a_via_ffmpeg(tmp_path):
    wav = tmp_path / "a.wav"
    m4a = tmp_path / "a.m4a"
    wav.write_bytes(wav_bytes(tone(), 16000))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav), "-c:a", "aac", str(m4a)], check=True)
    a = decode_audio(m4a.read_bytes())
    assert abs(a.duration - 1.0) < 0.15


@pytest.mark.parametrize("sr", [8000, 22050, 44100, 48000])
def test_resampling_preserves_duration_and_pitch(sr):
    a = decode_audio(encode(tone(sr=sr, seconds=2.0), sr, "WAV", "PCM_16"))
    assert a.sr == 16000
    assert abs(a.duration - 2.0) < 0.01
    spec = np.abs(np.fft.rfft(a.samples[:16000]))
    assert abs(np.argmax(spec) - 440) <= 2


def test_stereo_to_mono():
    a = decode_audio(encode(tone(channels=2), 16000, "WAV", "PCM_16"))
    assert a.samples.ndim == 1


def test_empty_and_invalid_audio():
    with pytest.raises(AudioError):
        decode_audio(b"")
    with pytest.raises(AudioError):
        decode_audio(b"this is definitely not audio" * 100)
    with pytest.raises(AudioError):
        decode_audio(encode(np.zeros(10, dtype=np.float32), 16000, "WAV", "PCM_16"))  # too short


def test_size_and_duration_limits():
    data = encode(tone(seconds=5.0), 16000, "WAV", "PCM_16")
    with pytest.raises(AudioError, match="too long"):
        decode_audio(data, max_duration=2.0)
    with pytest.raises(AudioError, match="too large"):
        decode_audio(data, max_bytes=1000)


def test_silence_is_detected():
    q = estimate_quality(np.zeros(16000, dtype=np.float32), 16000)
    assert q.silent
    q2 = estimate_quality(tone(), 16000)
    assert not q2.silent and q2.active_duration_s > 0.8


def test_trim_and_preprocess():
    y = np.concatenate([np.zeros(16000), tone(), np.zeros(16000)]).astype(np.float32)
    trimmed, offset = trim_silence(y, 16000)
    assert 0.8 < offset < 1.0
    assert len(trimmed) < len(y) * 0.6
    from melodyai.audio.io import Audio

    res = preprocess(Audio(y, 16000))
    assert np.max(np.abs(res.audio.samples)) == pytest.approx(10 ** (-1 / 20), rel=0.02)
