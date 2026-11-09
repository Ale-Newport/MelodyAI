"""Application state: catalog, recognition engine, jobs and settings (one per process).

The open-world search (:mod:`melodyai.world`) lives next to the catalog in the store directory:
``world.db`` (world lyrics index, filled by the popular-songs sync and by verified lookups),
``web_cache.db`` (cached LRCLIB / Deezer answers) and ``world_sync.db`` (sync state). Setting
``MELODYAI_OFFLINE=1`` keeps the server off the network: the world index and the web cache still answer,
online sources are reported as skipped.

Privacy of the open-world search follows the history settings: without "store recognition history"
nothing derived from a recognition is kept (searches carrying the user's words are not cached, verified
songs are not learned into the world index), and deleting the history (:meth:`AppState.forget_history`)
also purges the cached searches and the songs learned from recognitions (``source = "lookup"``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path

from app.core.config import ServerSettings
from app.services.jobs import JobManager
from app.services.user_settings import SettingsStore
from melodyai.catalog.service import Catalog
from melodyai.config import ROOT_DIR, models_dir, store_dir
from melodyai.engine import EngineSettings, RecognitionEngine
from melodyai.world.index import WorldLyricsIndex
from melodyai.world.search import WorldSearch
from melodyai.world.web import WebClient

log = logging.getLogger(__name__)

TRUTHY = ("1", "true", "yes", "on")


def offline_mode() -> bool:
    """``MELODYAI_OFFLINE=1|true``: never use the network (serve the web cache only)."""
    return os.environ.get("MELODYAI_OFFLINE", "").strip().lower() in TRUTHY


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
        self.root = root
        self.catalog = Catalog(root)
        self.user_settings = SettingsStore(root / "settings.json")
        from melodyai.providers.acrcloud import default_providers

        self.providers = default_providers()
        # Open-world search: one shared web client (rate limits, cache) and lyrics index per process.
        self.web = WebClient(root / "web_cache.db", offline=offline_mode())
        self.world_index = WorldLyricsIndex(root / "world.db")
        self.world = WorldSearch(self.world_index, self.web)
        # Hummed melodies of real songs (Lakh MIDI, built by `melodyai lakh-build`); loaded on first use.
        from melodyai.world.melodies import WorldMelodyIndex

        self.world_melody = WorldMelodyIndex(root / "world_melodies", meta_lookup=self.world_index.get_by_key)
        self.engine = RecognitionEngine(self.catalog.melody_index, self.catalog.lyrics_index, EngineSettings(),
                                        fp_index=self.catalog.fp_index, providers=self.providers,
                                        world=self.world, world_melody=self.world_melody)
        if self.world_melody.built:
            threading.Thread(target=self.world_melody.warm, name="melodyai-world-melody-load", daemon=True).start()
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
        self.web.cache_private = us.store_history
        self.world.learn = us.store_history
        if not us.store_history:
            self.web.purge_private()

    def forget_history(self) -> dict:
        """What the open world kept of past recognitions: cached searches with the user's words, learned songs."""
        return {"cached_searches": self.web.purge_private(), "learned_songs": self.world_index.delete_source("lookup")}

    def providers_enabled(self) -> bool:
        return self.user_settings.get().external_provider_enabled and any(p.configured() for p in self.providers)

    def online_enabled(self) -> bool:
        """Whether recognitions may query online sources: the user setting and a server that is not offline."""
        return self.user_settings.get().online_search_enabled and self.world.online_capable

    def close(self) -> None:
        """Shutdown: stop background jobs, persist the indexes, close the world index and the web client."""
        syncing = False
        for job in self.jobs.list("world_sync"):
            if job["status"] in ("queued", "running"):
                syncing = syncing or job["status"] == "running"
                try:
                    self.jobs.cancel(job["id"])  # cooperative: the sync stops after the current song
                except Exception:  # noqa: BLE001 - shutting down anyway
                    log.debug("could not cancel job %s", job["id"], exc_info=True)
        self.jobs.shutdown(wait=False, cancel_futures=True)
        self.catalog.flush_indexes()
        if syncing:  # the sync thread still finishes its current song: leave its connections to process exit
            return
        for res in (self.world_index, self.web):
            try:
                res.close()
            except Exception:  # noqa: BLE001
                log.debug("could not close %s", type(res).__name__, exc_info=True)

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
