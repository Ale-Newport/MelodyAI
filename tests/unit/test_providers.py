"""ACRCloud adapter: request signing, sample preparation, response parsing, errors, engine policy.

HTTP is mocked (httpx.MockTransport); the live test runs only if real credentials are set.
"""

import base64
import hashlib
import hmac
import io
import os

import httpx
import numpy as np
import pytest
import soundfile as sf

from melodyai.audio.io import Audio
from melodyai.providers.acrcloud import (
    ACRCloudCredentials,
    ACRCloudProvider,
    parse_response,
    prepare_sample,
    sign,
)

CREDS = ACRCloudCredentials("identify-eu-west-1.acrcloud.com", "KEY123", "SECRET456")

MUSIC_BODY = {
    "status": {"msg": "Success", "version": "1.0", "code": 0},
    "metadata": {"music": [{
        "title": "Some Song", "artists": [{"name": "Some Artist"}], "album": {"name": "Some Album"},
        "acrid": "abc", "score": 100, "play_offset_ms": 9040, "duration_ms": 295000, "release_date": "2015-10-23",
        "external_ids": {"isrc": "GBXXX1500214"},
        "external_metadata": {"spotify": {"track": {"id": "4aebBr4JAihzJQR0CiIZJv"}}, "youtube": {"vid": "YQHsXMglC9A"},
                              "deezer": {"track": {"id": "110265034"}}},
    }]},
}
HUMMING_BODY = {
    "status": {"msg": "Success", "version": "1.0", "code": 0},
    "metadata": {"humming": [
        {"title": "B", "artists": [{"name": "Y"}], "acrid": "2", "score": "0.87", "external_metadata": {}},
        {"title": "A", "artists": [{"name": "X"}], "acrid": "1", "score": "0.88",
         "external_metadata": {"spotify": {"track": {"id": "11Oj3hu53ZVOqzcw35elvh"}}}},
    ]},
}


def test_signature_matches_documented_algorithm():
    ts = "1700000000"
    expected = base64.b64encode(hmac.new(b"SECRET456", b"POST\n/v1/identify\nKEY123\naudio\n1\n" + ts.encode(),
                                         hashlib.sha1).digest()).decode()
    assert sign("KEY123", "SECRET456", ts) == expected


def test_sample_is_trimmed_downsampled_wav():
    sr = 16000
    y = np.zeros(sr * 30, dtype=np.float32)
    y[sr * 20: sr * 25] = 0.5 * np.sin(np.arange(sr * 5) * 2 * np.pi * 440 / sr)  # loud part at 20-25 s
    data, seconds = prepare_sample(y, sr)
    z, zsr = sf.read(io.BytesIO(data))
    assert zsr == 8000 and seconds <= 12.0 + 1e-6 and len(z) == pytest.approx(12 * 8000, abs=2)
    assert np.abs(z).max() > 0.3  # the loud segment was kept


def test_parse_music_and_humming():
    r = parse_response(MUSIC_BODY, "music")
    m = r.matches[0]
    assert r.status == "match" and m.title == "Some Song" and m.artists == ["Some Artist"]
    assert m.play_offset_s == pytest.approx(9.04) and m.duration_s == 295.0 and m.isrc == "GBXXX1500214"
    assert m.links["spotify"] == "https://open.spotify.com/track/4aebBr4JAihzJQR0CiIZJv"
    assert m.links["youtube"].endswith("YQHsXMglC9A") and "deezer" in m.links
    h = parse_response(HUMMING_BODY, "humming")
    assert [x.title for x in h.matches] == ["A", "B"] and h.matches[0].score == pytest.approx(0.88)
    assert parse_response({"status": {"code": 1001, "msg": "No result"}}, "music").status == "no_result"
    assert parse_response({"status": {"code": 3001, "msg": "Missing/Invalid Access Key"}}, "music").status == "error"


def mock_provider(handler):
    return ACRCloudProvider(CREDS, CREDS, transport=httpx.MockTransport(handler))


def test_identify_sends_signed_multipart_request():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        body = request.content
        for field in (b"access_key", b"sample_bytes", b"timestamp", b"signature", b"data_type", b"signature_version",
                      b'name="sample"; filename="sample.wav"', b"KEY123"):
            assert field in body
        assert b"SECRET456" not in body  # the secret never leaves the machine
        return httpx.Response(200, json=MUSIC_BODY)

    r = mock_provider(handler).identify(np.random.default_rng(0).normal(size=16000 * 5).astype(np.float32) * 0.1,
                                        16000, "music")
    assert seen["url"] == "https://identify-eu-west-1.acrcloud.com/v1/identify"
    assert r.status == "match" and r.sent_seconds == pytest.approx(5.0, abs=0.01)


def test_identify_handles_http_errors_and_timeouts():
    r = mock_provider(lambda req: httpx.Response(500)).identify(np.zeros(16000, np.float32), 16000)
    assert r.status == "error" and "500" in r.message

    def boom(req):
        raise httpx.ConnectTimeout("timeout")

    assert mock_provider(boom).identify(np.zeros(16000, np.float32), 16000).status == "error"


def test_unconfigured_or_suspicious_host_is_disabled():
    assert not ACRCloudProvider(None, None).configured()
    bad = ACRCloudProvider(ACRCloudCredentials("evil.example.com", "k", "s"), None)
    assert not bad.configured() and bad.describe()["invalid_host"]
    assert bad.identify(np.zeros(16000, np.float32), 16000).status == "disabled"


def test_engine_provider_policy(melody_index):
    """Fallback policy: the provider is only called when the local catalog has no confident match."""
    from melodyai.engine import EngineSettings, RecognitionEngine
    from melodyai.lyrics.search import LyricsIndex

    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=HUMMING_BODY)

    eng = RecognitionEngine(melody_index, LyricsIndex(":memory:"), EngineSettings(asr_enabled=False),
                            providers=[mock_provider(handler)])
    noise = Audio((0.2 * np.random.default_rng(1).normal(size=16000 * 4)).astype(np.float32), 16000)
    r = eng.recognize(noise, mode="hum", use_providers=True)
    assert r["status"] == "no_match" and r["external"][0]["matches"][0]["title"] == "A" and len(calls) == 1
    r = eng.recognize(noise, mode="hum", use_providers=False)
    assert "external" not in r and len(calls) == 1  # no consent -> nothing sent


@pytest.mark.skipif(not os.environ.get("MELODYAI_ACRCLOUD_ACCESS_KEY"), reason="no ACRCloud credentials")
def test_live_acrcloud_roundtrip():
    """Live check with real credentials: a tone is (almost surely) not a catalog song."""
    p = ACRCloudProvider()
    r = p.identify((0.3 * np.sin(np.arange(16000 * 6) * 2 * np.pi * 440 / 16000)).astype(np.float32), 16000)
    assert r.status in ("no_result", "match")
