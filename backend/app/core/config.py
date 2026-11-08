"""Server configuration (environment variables prefixed with ``MELODYAI_``, or ``.env``).

``ServerSettings`` reads its own fields from ``.env``. Other modules (licensed provider, open-world sources,
Whisper) read ``MELODYAI_*`` variables with ``os.environ``, so :func:`export_env_file` copies the ``MELODYAI_*``
entries of ``.env`` into the process environment at startup, never overriding a variable that is already set.
``MELODYAI_ENV_FILE`` points to another file; an empty value disables it (the test suite does this).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = ".env"


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


def export_env_file(path: str | Path | None = None) -> list[str]:
    """Copy ``MELODYAI_*`` entries of ``.env`` into ``os.environ`` (existing variables win). Returns the names set."""
    if path is None:
        path = os.environ.get("MELODYAI_ENV_FILE", ENV_FILE)
    if not path or not Path(path).is_file():
        return []
    from dotenv import dotenv_values

    added = []
    for key, value in dotenv_values(path).items():
        if key.startswith("MELODYAI_") and value is not None and key not in os.environ:
            os.environ[key] = value
            added.append(key)
    return added


@lru_cache(maxsize=1)
def get_settings() -> ServerSettings:
    export_env_file()
    env_file = os.environ.get("MELODYAI_ENV_FILE", ENV_FILE) or None
    return ServerSettings(_env_file=env_file)
