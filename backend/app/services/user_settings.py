"""User-editable runtime settings, persisted as JSON in the store directory."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from pydantic import BaseModel, Field


class UserSettings(BaseModel):
    ui_language: str = Field("en", pattern="^(en|es)$")
    theme: str = Field("dark", pattern="^(dark|light|system)$")
    default_mode: str = Field("auto", pattern="^(auto|clip|hum|whistle|lyrics|combined)$")
    melody_algorithm: str = "ensemble"
    pitch_extractor: str = "pyin"
    fusion_method: str = Field("weighted", pattern="^(weighted|rrf|combmnz)$")
    asr_enabled: bool = True
    # Whisper model; "auto" = best cached (large-v3-turbo > large-v3 > medium > small > base > tiny).
    asr_model: str = Field("auto", pattern="^(auto|tiny|base|small|medium|large-v3|large-v3-turbo)$")
    asr_language: str | None = Field(None, pattern="^(en|es)$")
    inference_device: str = Field("auto", pattern="^(auto|cpu|mps|cuda)$")
    min_record_seconds: float = Field(5.0, ge=1.0, le=30.0)
    max_record_seconds: float = Field(20.0, ge=3.0, le=60.0)
    top_k: int = Field(5, ge=1, le=20)
    store_history: bool = True             # store recognition results (never raw audio by default)
    retention: str = Field("discard", pattern="^(discard|keep_with_consent)$")
    microphone_id: str | None = None       # remembered on the client, kept here for portability
    # Licensed external catalog (e.g. ACRCloud). Off by default: enabling it sends a <=12 s copy of
    # each recording to the provider.
    external_provider_enabled: bool = False
    provider_policy: str = Field("fallback", pattern="^(fallback|always)$")
    # Open-world song search: look the typed / transcribed words up on LRCLIB and Deezer (only text is
    # sent, never audio). Off -> only the local indexes (catalog + world lyrics index) are searched.
    online_search_enabled: bool = True


class SettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._value = self._load()

    def _load(self) -> UserSettings:
        try:
            return UserSettings(**json.loads(self.path.read_text()))
        except (OSError, ValueError):
            return UserSettings()

    def get(self) -> UserSettings:
        return self._value

    def update(self, patch: dict) -> UserSettings:
        with self._lock:
            merged = UserSettings(**{**self._value.model_dump(), **patch})
            if merged.min_record_seconds > merged.max_record_seconds:
                raise ValueError("min_record_seconds must be <= max_record_seconds")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(merged.model_dump(), indent=1))
            self._value = merged
            return merged
