"""Open-world song search endpoints: world lyrics index status, popular-songs sync, direct search, songs, previews.

The world lyrics index (``store/world.db``) holds real songs' lyrics: filled by the popular-songs sync
(Deezer charts / editorial playlists -> lyrics from LRCLIB) and by verified lookups of recognitions.
Only text is ever sent to the online sources (the search words, chart / playlist queries), never audio.

The sync runs as a background job (``kind = "world_sync"``) on its own :class:`JobManager` lane (it is
network-bound and takes hours, so it must not hold back model downloads or reindexing); one at a time per
store, also across processes (a CLI ``world-sync`` holds the same lock, see :func:`melodyai.world.sync.sync_lock`).

Endpoints that make the server talk to third parties take a JSON body (``POST /sync``, ``POST /search``):
a cross-site page cannot send one without a CORS preflight, and the user's words stay out of URLs and
access logs. ``GET /preview/{deezer_id}`` redirects to a fresh 30 s Deezer preview (their URLs are signed for
~15 minutes, so stored ones go stale); it runs only when the user presses play.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.api.deps import get_state
from app.core.state import AppState
from melodyai.engine import WORLD_TOP_K, world_stub
from melodyai.fusion.fusion import decide
from melodyai.world.deezer import DeezerClient
from melodyai.world.lrclib import LrclibClient
from melodyai.world.sync import PRESETS, preset_list, running_sync, sync_popular
from melodyai.world.web import WebError

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/world", tags=["world"])
SYNC_KIND = "world_sync"
ACTIVE = ("queued", "running")
LAST_SYNC_KEYS = ("preset", "status", "finished_at", "added", "collected")
MAX_QUERY_CHARS = 2000
MAX_SONGS_OFFSET = 10_000_000
PREVIEW_HOSTS = (".dzcdn.net",)  # Deezer's CDN: the only redirect target of /preview
_submit_lock = threading.Lock()  # "is a sync active?" + submit must be atomic (sync endpoints run in a thread pool)


def _last_sync(state: AppState) -> dict | None:
    try:
        raw = state.world_index.meta_get("last_sync")
        data = json.loads(raw) if raw else None
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return {k: data.get(k) for k in LAST_SYNC_KEYS}


def _external_sync(state: AppState) -> dict | None:
    """A sync running in another process on this store (e.g. the CLI), else None."""
    try:
        run = running_sync(state.root / "world_sync.db")
    except Exception:  # noqa: BLE001 - status only
        log.debug("could not read the sync state", exc_info=True)
        return None
    if run is None or run.get("pid") == os.getpid():
        return None
    return {k: run.get(k) for k in ("preset", "started_at", "updated_at", "total", "done")}


@router.get("/status", summary="World lyrics index, online sources and popular-songs sync presets")
def status(state: AppState = Depends(get_state)):
    stats = state.world_index.stats()
    return {"songs": stats["songs"], "online_enabled": state.online_enabled(),
            "server_online": state.world.online_capable,
            "sources": state.world.describe_sources(), "last_sync": _last_sync(state), "db_bytes": stats["bytes"],
            "presets": preset_list(), "running_sync": _external_sync(state),
            "melodies": state.world_melody.stats()}


class SyncBody(BaseModel):
    preset: str = Field("quick", pattern="^(" + "|".join(PRESETS) + ")$")
    limit: int | None = Field(None, ge=1, le=100_000)


def _sync_job(state: AppState, preset: str, limit: int | None):
    def run(progress, should_stop):
        res = sync_popular(state.world_index, LrclibClient(state.web), DeezerClient(state.web), preset, limit,
                           progress, should_stop, state_path=state.root / "world_sync.db")
        if res.get("status") == "error":
            raise RuntimeError(res.get("message") or "popular-songs sync failed")
        return res

    return run


@router.post("/sync", summary="Add popular songs' lyrics to the world index (background job)")
def start_sync(body: SyncBody, state: AppState = Depends(get_state)):
    # ``body`` is required JSON: a cross-site "simple" request (no body, text/plain) is rejected with 422.
    if not state.world.online_capable:
        raise HTTPException(503, "This server is offline (MELODYAI_OFFLINE): the sync needs Deezer and LRCLIB")
    with _submit_lock:
        if any(j["status"] in ACTIVE for j in state.jobs.list(SYNC_KIND)):
            raise HTTPException(409, "A popular-songs sync is already running")
        if _external_sync(state) is not None:
            raise HTTPException(409, "A popular-songs sync is already running in another process "
                                     "(e.g. `melodyai world-sync`)")
        return state.jobs.submit(SYNC_KIND, body.model_dump(), _sync_job(state, body.preset, body.limit))


class SearchBody(BaseModel):
    q: str = Field("", max_length=MAX_QUERY_CHARS, description="Words of the song")
    language: str | None = Field(None, pattern="^[a-z]{2}$")
    top_k: int = Field(5, ge=1, le=20)


def _search(state: AppState, q: str, language: str | None, top_k: int, online: bool) -> dict:
    # The abstention decision needs the runner-up: search (and decide) on at least WORLD_TOP_K, then trim.
    try:
        res = state.world.search(q, language, max(top_k, WORLD_TOP_K), online).to_dict()
    except Exception as e:  # noqa: BLE001 - WorldSearch never raises; this endpoint must not 500 either way
        log.exception("world search failed")
        res = world_stub("error", q, online, f"{type(e).__name__}: {e}"[:300])
    cands = res.get("candidates") or []
    res["decision"] = decide([{"song_id": c["song_id"], "score": c["score"]} for c in cands],
                             "world", None, state.engine.calibration)
    res["candidates"] = cands[:top_k]
    return res


@router.post("/search", summary="Search songs by (sung, possibly misheard) words in the open world")
async def search(body: SearchBody, state: AppState = Depends(get_state)):
    online = state.user_settings.get().online_search_enabled
    if not body.q.strip():
        live = online and state.world.online_capable
        why = "" if live else "setting_off" if not online else "server_offline"
        res = world_stub("no_query", "", live, "not enough words to search", offline_reason=why)
        res["decision"] = decide([], "world", None, state.engine.calibration)
        return res
    async with state.recognition_slots:
        return await run_in_threadpool(_search, state, body.q, body.language, body.top_k, online)


@router.get("/songs", summary="Songs of the world lyrics index (metadata only, most popular first)")
def songs(q: str | None = Query(None, max_length=200), offset: int = Query(0, ge=0, le=MAX_SONGS_OFFSET),
          limit: int = Query(50, ge=1, le=200), state: AppState = Depends(get_state)):
    return state.world_index.list_songs(q, offset, limit)


@router.get("/preview/{deezer_id}", summary="Redirect to a fresh 30-second Deezer preview of a track",
            response_class=RedirectResponse, status_code=307)
def preview(deezer_id: int = Path(..., ge=1, le=10**13), state: AppState = Depends(get_state)):
    if not state.world.online_capable:
        raise HTTPException(503, "This server is offline (MELODYAI_OFFLINE): previews come from Deezer")
    try:
        tm = DeezerClient(state.web).track(deezer_id)
    except WebError as e:
        raise HTTPException(502, f"Deezer did not answer ({e.kind})") from None
    url = tm.preview_url if tm is not None else None
    host = urlsplit(url).hostname or "" if url else ""
    if not url or not url.startswith("https://") or not host.endswith(PREVIEW_HOSTS):
        raise HTTPException(404, "No preview for this track")
    return RedirectResponse(url, status_code=307, headers={"Cache-Control": "no-store"})
