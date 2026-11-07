"""ACRCloud Identification API client (music fingerprinting and humming).

Protocol (docs.acrcloud.com, Identification API):
  POST https://<host>/v1/identify  (multipart/form-data)
  fields: sample, sample_bytes, access_key, data_type="audio", signature_version="1", timestamp, signature
  signature = base64(HMAC-SHA1(access_secret, "POST\\n/v1/identify\\n" + access_key + "\\n" +
                                data_type + "\\n" + signature_version + "\\n" + timestamp))
  response: {"status": {"code": 0, "msg": "Success"}, "metadata": {"music": [...]} | {"humming": [...]}}
  music score: 70-100; humming score: 0-1 (string in the documented example).

Configuration (environment, never committed):
  MELODYAI_ACRCLOUD_HOST, MELODYAI_ACRCLOUD_ACCESS_KEY, MELODYAI_ACRCLOUD_ACCESS_SECRET
  optional separate humming project: MELODYAI_ACRCLOUD_HUMMING_HOST / _ACCESS_KEY / _ACCESS_SECRET
  (if unset, the main project is used for both; humming must be enabled on that project).

The recording is trimmed to at most 12 s (ACRCloud recommends < 15 s) and sent as 8 kHz mono
16-bit WAV to minimise the data leaving the machine.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import time
from dataclasses import dataclass

import numpy as np

from melodyai.providers.base import ProviderMatch, ProviderResult

HOST_RE = re.compile(r"^identify-[a-z0-9-]+\.acrcloud\.(com|cn)$")
MAX_SECONDS = 12.0
SEND_SR = 8000
STATUS_NO_RESULT = 1001


@dataclass
class ACRCloudCredentials:
    host: str
    access_key: str
    access_secret: str

    @classmethod
    def from_env(cls, prefix: str = "MELODYAI_ACRCLOUD_") -> ACRCloudCredentials | None:
        host = os.environ.get(prefix + "HOST", "").strip()
        key = os.environ.get(prefix + "ACCESS_KEY", "").strip()
        secret = os.environ.get(prefix + "ACCESS_SECRET", "").strip()
        if not (host and key and secret):
            return None
        return cls(host, key, secret)

    def valid_host(self) -> bool:
        return bool(HOST_RE.match(self.host))


def sign(access_key: str, access_secret: str, timestamp: str, data_type: str = "audio",
         signature_version: str = "1", method: str = "POST", uri: str = "/v1/identify") -> str:
    msg = "\n".join([method, uri, access_key, data_type, signature_version, timestamp])
    digest = hmac.new(access_secret.encode("ascii"), msg.encode("ascii"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


def prepare_sample(samples: np.ndarray, sr: int) -> tuple[bytes, float]:
    """Loudest ``MAX_SECONDS`` window, resampled to 8 kHz mono WAV. Returns (bytes, seconds)."""
    from melodyai.audio.io import resample, wav_bytes

    y = np.asarray(samples, dtype=np.float32)
    n = int(MAX_SECONDS * sr)
    if len(y) > n:
        hop = sr // 2
        energy = np.convolve(y.astype(np.float64) ** 2, np.ones(n), mode="valid")[::hop]
        start = int(np.argmax(energy)) * hop
        y = y[start:start + n]
    y = resample(y, sr, SEND_SR)
    return wav_bytes(y, SEND_SR), len(y) / SEND_SR


def _artists(item: dict) -> list[str]:
    return [a.get("name", "") for a in item.get("artists", []) if a.get("name")]


def _links(item: dict) -> dict[str, str]:
    em = item.get("external_metadata") or {}
    out = {}
    sp = (em.get("spotify") or {}).get("track", {}).get("id")
    if sp:
        out["spotify"] = f"https://open.spotify.com/track/{sp}"
    yt = (em.get("youtube") or {}).get("vid")
    if yt:
        out["youtube"] = f"https://www.youtube.com/watch?v={yt}"
    dz = (em.get("deezer") or {}).get("track", {}).get("id")
    if dz:
        out["deezer"] = f"https://www.deezer.com/track/{dz}"
    mb = (em.get("musicbrainz") or {}).get("track", {}).get("id")
    if mb:
        out["musicbrainz"] = f"https://musicbrainz.org/recording/{mb}"
    return out


def parse_response(body: dict, kind: str) -> ProviderResult:
    status = body.get("status") or {}
    code = int(status.get("code", -1))
    if code == STATUS_NO_RESULT:
        return ProviderResult("acrcloud", kind, "no_result", message=status.get("msg", "No result"))
    if code != 0:
        return ProviderResult("acrcloud", kind, "error", message=f"ACRCloud error {code}: {status.get('msg', '')}")
    meta = body.get("metadata") or {}
    items = meta.get("humming" if kind == "humming" else "music") or []
    matches = []
    for it in items:
        try:
            score = float(it.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        album = it.get("album") or {}
        matches.append(ProviderMatch(
            title=it.get("title", "?"), artists=_artists(it), album=album.get("name"),
            release_date=it.get("release_date"), score=score,
            score_kind="humming similarity 0-1 (provider)" if kind == "humming" else "match confidence 70-100 (provider)",
            play_offset_s=(it["play_offset_ms"] / 1000.0) if it.get("play_offset_ms") is not None else None,
            duration_s=(float(it["duration_ms"]) / 1000.0) if it.get("duration_ms") else None,
            isrc=(it.get("external_ids") or {}).get("isrc"), provider_id=it.get("acrid"), links=_links(it),
        ))
    matches.sort(key=lambda m: -m.score)
    return ProviderResult("acrcloud", kind, "match" if matches else "no_result", matches)


class ACRCloudProvider:
    name = "acrcloud"
    label = "ACRCloud (licensed commercial catalog)"

    def __init__(self, music: ACRCloudCredentials | None = None, humming: ACRCloudCredentials | None = None,
                 timeout: float = 15.0, transport=None) -> None:
        self.music = music if music is not None else ACRCloudCredentials.from_env()
        self.humming = humming if humming is not None else (
            ACRCloudCredentials.from_env("MELODYAI_ACRCLOUD_HUMMING_") or self.music)
        self.timeout = timeout
        self.transport = transport  # httpx transport (tests inject a mock)

    def configured(self) -> bool:
        return self.music is not None and self.music.valid_host()

    def supports(self, kind: str) -> bool:
        creds = self.humming if kind == "humming" else self.music
        return creds is not None and creds.valid_host()

    def describe(self) -> dict:
        return {"name": self.name, "label": self.label, "configured": self.configured(),
                "music": self.supports("music"), "humming": self.supports("humming"),
                "host": self.music.host if self.music else None,
                "invalid_host": bool(self.music and not self.music.valid_host()),
                "sends_audio": True, "max_seconds_sent": MAX_SECONDS}

    def identify(self, samples, sr: int, kind: str = "music") -> ProviderResult:
        import httpx

        creds = self.humming if kind == "humming" else self.music
        if creds is None or not creds.valid_host():
            return ProviderResult(self.name, kind, "disabled", message="ACRCloud is not configured")
        data, seconds = prepare_sample(samples, sr)
        ts = str(int(time.time()))
        form = {"access_key": creds.access_key, "sample_bytes": str(len(data)), "timestamp": ts,
                "signature": sign(creds.access_key, creds.access_secret, ts), "data_type": "audio",
                "signature_version": "1"}
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                r = client.post(f"https://{creds.host}/v1/identify", data=form,
                                files={"sample": ("sample.wav", data, "audio/wav")})
            if r.status_code != 200:
                res = ProviderResult(self.name, kind, "error", message=f"HTTP {r.status_code}")
            else:
                res = parse_response(r.json(), kind)
        except (httpx.HTTPError, ValueError) as e:
            res = ProviderResult(self.name, kind, "error", message=f"{type(e).__name__}: {e}")
        res.elapsed_s = round(time.perf_counter() - t0, 3)
        res.sent_seconds = round(seconds, 2)
        return res


def default_providers() -> list:
    return [ACRCloudProvider()]
