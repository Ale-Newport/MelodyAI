"""FastAPI application factory.

Run (development):  uvicorn app.main:app --app-dir backend --reload --port 8000
OpenAPI docs:        http://localhost:8000/docs

The API listens on localhost, so any web page the user visits could try to call it. CORS only hides the
answers from other origins, it does not stop the request, so :func:`cross_site_guard` refuses ``/api``
requests that come from another site: an ``Origin`` that is neither this server nor one of
``cors_origins``, or ``Sec-Fetch-Site: cross-site`` without an allowed origin (e.g. an ``<img>`` pointing
at an endpoint). Endpoints that act on the world also take JSON bodies (preflighted cross-origin).
httpx's INFO request log (full URLs, i.e. the user's words in search queries) is silenced.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import catalog, lab, recognition, world
from app.core.config import ServerSettings, get_settings
from app.core.state import AppState
from melodyai import __version__
from melodyai.config import ROOT_DIR

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "httpcore"):        # they log full request URLs (search words) at INFO
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def cross_site_guard(allowed_origins: list[str]):
    """Middleware refusing ``/api`` requests sent by other sites (see the module docstring)."""
    allowed = {o.rstrip("/") for o in allowed_origins}

    async def guard(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            origin = (request.headers.get("origin") or "").rstrip("/")
            own = f"{request.url.scheme}://{request.headers.get('host', '')}"
            ok_origin = origin in allowed or origin == own
            cross = request.headers.get("sec-fetch-site", "") == "cross-site"
            if (origin and not ok_origin) or (cross and not ok_origin):
                return JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
        return await call_next(request)

    return guard


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.melody = AppState(settings)
        from melodyai.matching.dtw import warmup

        warmup()
        yield
        app.state.melody.close()

    app = FastAPI(
        title="MelodyAI API", version=__version__, lifespan=lifespan,
        description="Query-by-humming, whistling and lyrics recognition. Scores are relative similarities "
                    "unless explicitly labelled as calibrated probabilities.",
    )
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
    app.middleware("http")(cross_site_guard(origins))
    app.include_router(recognition.router)
    app.include_router(catalog.router)
    app.include_router(lab.router)
    app.include_router(world.router)

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
