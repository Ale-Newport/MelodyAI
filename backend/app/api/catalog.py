"""Songs and catalog management endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.deps import get_state, read_upload, safe_id
from app.core.state import AppState
from melodyai.catalog.importers import ImportErrorMelody, parse_lyrics, parse_metadata
from melodyai.catalog.service import DuplicateSongError, NotFoundError

router = APIRouter(prefix="/api", tags=["catalog"])


class SongPatch(BaseModel):
    title: str | None = Field(None, max_length=300)
    artist: str | None = Field(None, max_length=300)
    album: str | None = Field(None, max_length=300)
    year: int | None = Field(None, ge=0, le=3000)
    language: str | None = Field(None, max_length=16)
    composition_id: str | None = Field(None, max_length=64)
    source: str | None = Field(None, max_length=500)
    license: str | None = Field(None, max_length=200)
    listen_url: str | None = Field(None, max_length=500, pattern=r"^https?://")


class LyricsBody(BaseModel):
    text: str = Field("", max_length=50_000)
    language: str = Field("en", max_length=16)
    source: str | None = Field(None, max_length=500)
    license: str | None = Field(None, max_length=200)


class ReindexBody(BaseModel):
    song_ids: list[str] | None = None
    all: bool = False


def _not_found(e):
    raise HTTPException(404, "Song not found") from e


@router.get("/songs", summary="List / search / filter songs")
def list_songs(q: str | None = None, status: str | None = None, offset: int = 0, limit: int = 100,
               state: AppState = Depends(get_state)):
    items, total = state.catalog.list_songs(q, status, max(0, offset), min(max(1, limit), 500))
    return {"items": items, "total": total}


@router.post("/songs", status_code=201, summary="Add a song (MIDI or audio reference, optional lyrics)")
async def add_song(
    reference: UploadFile = File(..., description="MIDI (.mid) or audio reference"),
    title: str = Form(..., max_length=300),
    artist: str = Form("Unknown", max_length=300),
    album: str | None = Form(None),
    year: int | None = Form(None),
    language: str | None = Form(None),
    reference_type: str | None = Form(None),
    composition_id: str | None = Form(None),
    source: str | None = Form(None),
    license: str | None = Form(None),
    listen_url: str | None = Form(None, pattern=r"^https?://"),
    midi_track: int | None = Form(None),
    lyrics_text: str | None = Form(None, max_length=50_000),
    lyrics_language: str | None = Form(None),
    lyrics_file: UploadFile | None = File(None),
    state: AppState = Depends(get_state),
):
    data = await read_upload(reference, state.settings.max_reference_mb)
    lyrics = None
    try:
        if lyrics_file is not None:
            lyrics = parse_lyrics(await read_upload(lyrics_file, 1), lyrics_file.filename or "lyrics.txt")
        elif lyrics_text and lyrics_text.strip():
            lyrics = {"text": lyrics_text}
        if lyrics is not None and lyrics_language:
            lyrics["language"] = lyrics_language
        meta = {"title": title, "artist": artist, "album": album, "year": year, "language": language,
                "reference_type": reference_type, "composition_id": composition_id, "source": source,
                "license": license, "listen_url": listen_url, "midi_track": midi_track}
        return await run_in_threadpool(state.catalog.add_song, meta, data, Path(reference.filename or "ref").name,
                                       lyrics)
    except DuplicateSongError as e:
        raise HTTPException(409, {"message": str(e), "existing_id": e.existing_id}) from e
    except ImportErrorMelody as e:
        raise HTTPException(422, str(e)) from e


@router.get("/songs/{song_id}")
def get_song(song_id: str, state: AppState = Depends(get_state)):
    try:
        return state.catalog.get_song(safe_id(song_id))
    except NotFoundError as e:
        _not_found(e)


@router.patch("/songs/{song_id}")
def update_song(song_id: str, patch: SongPatch, state: AppState = Depends(get_state)):
    try:
        return state.catalog.update_song(safe_id(song_id), patch.model_dump(exclude_unset=True))
    except NotFoundError as e:
        _not_found(e)


@router.put("/songs/{song_id}/lyrics", summary="Set or clear (empty text) the lyrics of a song")
def set_lyrics(song_id: str, body: LyricsBody, state: AppState = Depends(get_state)):
    try:
        return state.catalog.set_lyrics(safe_id(song_id), body.text, body.language, body.source, body.license)
    except NotFoundError as e:
        _not_found(e)


@router.delete("/songs/{song_id}")
def delete_song(song_id: str, state: AppState = Depends(get_state)):
    try:
        state.catalog.delete_song(safe_id(song_id))
    except NotFoundError as e:
        _not_found(e)
    return {"deleted": song_id}


@router.post("/songs/{song_id}/reindex")
async def reindex_song(song_id: str, state: AppState = Depends(get_state)):
    try:
        return await run_in_threadpool(state.catalog.index_song, safe_id(song_id))
    except NotFoundError as e:
        _not_found(e)


@router.get("/songs/{song_id}/audio", summary="Listen to the reference (audio or synthesised MIDI preview)")
def song_audio(song_id: str, state: AppState = Depends(get_state)):
    p = state.catalog.listen_file(safe_id(song_id))
    if p is None:
        raise HTTPException(404, "No audio available for this song")
    return FileResponse(p, media_type="audio/wav" if p.suffix == ".wav" else None)


@router.get("/songs/{song_id}/melody", summary="Reference melody (notes and contour) used for matching")
def song_melody(song_id: str, state: AppState = Depends(get_state)):
    ref = state.catalog.melody_index.refs.get(safe_id(song_id))
    if ref is None:
        raise HTTPException(404, "Song not indexed")
    return {"song_id": song_id, "reference_type": ref.reference_type, "notes": ref.notes.to_dict(),
            "contour": ref.contour[::2].round(2).tolist(), "contour_rate_hz": 10.0,
            "contour_t0": float(ref.contour_times[0]) if len(ref.contour_times) else 0.0, "meta": ref.meta}


@router.get("/catalog/status")
def catalog_status(state: AppState = Depends(get_state)):
    st = state.catalog.status()
    st["neural_index"] = state.neural
    return st


@router.get("/catalog/duplicates")
def duplicates(state: AppState = Depends(get_state)):
    return state.catalog.find_duplicates()


@router.post("/catalog/import", summary="Bulk import: metadata CSV/JSON + the files it references")
async def bulk_import(manifest: UploadFile = File(...), files: list[UploadFile] = File(...),
                      state: AppState = Depends(get_state)):
    try:
        rows = parse_metadata(await read_upload(manifest, 2), manifest.filename or "metadata.csv")
    except (ImportErrorMelody, ValueError, KeyError) as e:
        raise HTTPException(422, f"Invalid manifest: {e}") from e
    if len(files) > 200:
        raise HTTPException(413, "At most 200 files per import")
    blobs = {}
    for f in files:
        name = Path(f.filename or "").name
        blobs[name] = await read_upload(f, state.settings.max_reference_mb)
    results = []
    for row in rows:
        ref_name = Path(row.get("reference_file") or "").name
        if ref_name not in blobs:
            results.append({"title": row["title"], "status": "error", "error": f"missing file '{ref_name}'"})
            continue
        lyrics = None
        try:
            if row.get("lyrics_file"):
                ln = Path(row["lyrics_file"]).name
                if ln in blobs:
                    lyrics = parse_lyrics(blobs[ln], ln)
            s = await run_in_threadpool(state.catalog.add_song, row, blobs[ref_name], ref_name, lyrics)
            results.append({"title": row["title"], "status": s["index_status"], "id": s["id"],
                            "error": s.get("index_error")})
        except DuplicateSongError as e:
            results.append({"title": row["title"], "status": "duplicate", "id": e.existing_id})
        except ImportErrorMelody as e:
            results.append({"title": row["title"], "status": "error", "error": str(e)})
    state.catalog.flush_indexes()
    return {"results": results}


@router.post("/catalog/index", summary="Reindex songs in the background (pending/stale/error, or all)")
def reindex(body: ReindexBody = Body(default_factory=ReindexBody), state: AppState = Depends(get_state)):
    ids = [safe_id(s) for s in body.song_ids] if body.song_ids else None

    def run(progress, should_stop):
        return state.catalog.reindex(ids, only_needed=not body.all, progress=progress)

    return state.jobs.submit("reindex", body.model_dump(), run)


@router.post("/catalog/demo", summary="Generate and import the original demo catalog (background job)")
def import_demo(state: AppState = Depends(get_state)):
    from melodyai.demo.build import build_demo_files, demo_dir, import_demo

    def run(progress, should_stop):
        if not (demo_dir() / "catalog.json").exists():
            progress(0.0, "generating demo files")
            build_demo_files()
        return import_demo(state.catalog, progress=progress)

    return state.jobs.submit("demo_import", {}, run)
