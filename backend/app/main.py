"""FastAPI application factory.

Run (development):  uvicorn app.main:app --app-dir backend --reload --port 8000
OpenAPI docs:        http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import catalog, lab, recognition
from app.core.config import ServerSettings, get_settings
from app.core.state import AppState
from melodyai import __version__
from melodyai.config import ROOT_DIR

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.melody = AppState(settings)
        from melodyai.matching.dtw import warmup

        warmup()
        yield
        app.state.melody.jobs.pool.shutdown(wait=False, cancel_futures=True)
        app.state.melody.catalog.flush_indexes()

    app = FastAPI(
        title="MelodyAI API", version=__version__, lifespan=lifespan,
        description="Query-by-humming, whistling and lyrics recognition. Scores are relative similarities "
                    "unless explicitly labelled as calibrated probabilities.",
    )
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(recognition.router)
    app.include_router(catalog.router)
    app.include_router(lab.router)

    dist = ROOT_DIR / "frontend" / "dist"
    if settings.serve_frontend and (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            p = (dist / path).resolve()
            if path and dist in p.parents and p.is_file():
                return FileResponse(p)
            return FileResponse(Path(dist / "index.html"))

    return app


app = create_app()
