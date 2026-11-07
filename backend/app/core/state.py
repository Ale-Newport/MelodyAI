"""Application state: catalog, recognition engine, jobs and settings (one per process)."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from app.core.config import ServerSettings
from app.services.jobs import JobManager
from app.services.user_settings import SettingsStore
from melodyai.catalog.service import Catalog
from melodyai.config import ROOT_DIR, models_dir, store_dir
from melodyai.engine import EngineSettings, RecognitionEngine

log = logging.getLogger(__name__)


def discover_checkpoints() -> list[dict]:
    """Neural checkpoints that may be activated (never arbitrary client paths)."""
    out = []
    for p in sorted(models_dir().glob("*.pt")):
        out.append({"id": f"models/{p.name}", "path": p, "source": "exported"})
    runs = ROOT_DIR / "experiments" / "training"
    for p in sorted(runs.glob("*/best.pt")):
        out.append({"id": f"training/{p.parent.name}/best.pt", "path": p, "source": "training run"})
    return out


class AppState:
    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        root = Path(settings.store_dir) if settings.store_dir else store_dir()
        self.catalog = Catalog(root)
        self.user_settings = SettingsStore(root / "settings.json")
        self.engine = RecognitionEngine(self.catalog.melody_index, self.catalog.lyrics_index, EngineSettings())
        self.apply_user_settings()
        self.jobs = JobManager(self.catalog.Session)
        self.recognition_slots = asyncio.Semaphore(settings.max_concurrent_recognitions)
        self.neural: dict | None = None
        self.neural_error: str | None = None
        self._neural_lock = threading.Lock()
        default = models_dir() / "melody_encoder.pt"
        if settings.load_neural_model and default.exists():
            self.activate_neural(f"models/{default.name}")

    def apply_user_settings(self) -> None:
        us = self.user_settings.get()
        s = self.engine.settings
        s.melody_algorithm = us.melody_algorithm if us.melody_algorithm in self.engine.algorithms else "ensemble"
        s.pitch_extractor = us.pitch_extractor
        s.fusion_method = us.fusion_method
        s.asr_enabled = us.asr_enabled
        s.asr_model = us.asr_model
        s.asr_allow_download = self.settings.allow_model_download
        s.top_k = us.top_k

    def activate_neural(self, checkpoint_id: str) -> dict:
        from melodyai.matching.neural import attach_neural, detach_neural

        match = [c for c in discover_checkpoints() if c["id"] == checkpoint_id]
        if not match:
            raise KeyError(checkpoint_id)
        with self._neural_lock:
            detach_neural(self.engine)
            try:
                info = attach_neural(self.engine, self.catalog, match[0]["path"], self.catalog.root)
                info["checkpoint"] = checkpoint_id
                self.neural, self.neural_error = info, None
            except Exception as e:  # torch missing, incompatible checkpoint...
                log.exception("could not load neural model")
                self.neural, self.neural_error = None, str(e)
                raise
        return self.neural
