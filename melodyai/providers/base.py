"""External recognition providers (licensed commercial catalogs).

MelodyAI cannot legally index Spotify or TikTok audio itself. Providers that hold licensed
catalogs (e.g. ACRCloud) can be queried instead. Their results are kept *separate* from the
local engines: the catalogs differ and their scores have different semantics, so they are
not fused numerically.

Privacy: calling a provider sends (a trimmed copy of) the user's recording to a third party.
It therefore requires (1) credentials configured by the operator and (2) the user's explicit
opt-in in Settings. Nothing is sent otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass
class ProviderMatch:
    title: str
    artists: list[str]
    album: str | None = None
    release_date: str | None = None
    score: float = 0.0                 # provider-native score
    score_kind: str = ""               # documents the provider's scale
    play_offset_s: float | None = None
    duration_s: float | None = None
    isrc: str | None = None
    provider_id: str | None = None
    links: dict[str, str] = field(default_factory=dict)  # spotify / youtube / deezer / musicbrainz URLs

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProviderResult:
    provider: str
    kind: str                          # "music" (recording) | "humming"
    status: str                        # match | no_result | error | disabled
    matches: list[ProviderMatch] = field(default_factory=list)
    message: str = ""
    elapsed_s: float = 0.0
    sent_seconds: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["matches"] = [m.to_dict() for m in self.matches]
        return d


class Provider(Protocol):
    name: str
    label: str

    def configured(self) -> bool: ...

    def supports(self, kind: str) -> bool: ...

    def identify(self, samples, sr: int, kind: str) -> ProviderResult: ...

    def describe(self) -> dict: ...
