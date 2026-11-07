"""Server configuration (environment variables prefixed with ``MELODYAI_``, or ``.env``)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MELODYAI_", env_file=".env", extra="ignore")

    store_dir: str | None = None              # defaults to data/store
    max_upload_mb: float = 20.0               # recognition / analysis uploads
    max_reference_mb: float = 60.0            # catalog imports
    max_query_seconds: float = 60.0
    min_query_seconds: float = 0.5
    max_concurrent_recognitions: int = 2
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    allow_model_download: bool = False        # Whisper model downloads must be enabled explicitly
    serve_frontend: bool = True               # serve frontend/dist if it exists
    load_neural_model: bool = True            # load models/melody_encoder.pt at startup if present


@lru_cache(maxsize=1)
def get_settings() -> ServerSettings:
    return ServerSettings()
