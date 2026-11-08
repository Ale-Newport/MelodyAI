"""Recognition endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import delete, select

from app.api.deps import get_state, read_upload, safe_id
from app.core.state import AppState
from melodyai.audio.io import AudioError, decode_audio, write_wav
from melodyai.catalog.models import Recognition

router = APIRouter(prefix="/api", tags=["recognition"])


WORLD_PREFIX = "world:"  # open-world songs (melodyai.world): the engine already built their song dict


def _enrich(state: AppState, res: dict) -> dict:
    """Attach catalog metadata to local results; open-world results keep the song dict the engine built."""
    local = [r for r in res["results"] if not str(r["song_id"]).startswith(WORLD_PREFIX)]
    meta = state.catalog.songs_by_ids([r["song_id"] for r in local]) if local else {}
    for r in local:
        m = meta.get(r["song_id"], {})
        r["song"] = {k: m.get(k) for k in ("id", "title", "artist", "album", "year", "language", "reference_type",
                                           "duration_s", "listen_url", "license", "source")}
        r["song"]["has_audio"] = state.catalog.listen_file(r["song_id"]) is not None
    return res


async def _recognize(state: AppState, mode: str, audio: UploadFile | None, text: str | None, algorithm: str | None,
                     language: str | None, top_k: int | None, pitch_extractor: str | None,
                     fusion_method: str | None, retain_audio: bool) -> dict:
    s = state.settings
    us = state.user_settings.get()
    decoded = None
    raw = None
    if audio is not None:
        raw = await read_upload(audio, s.max_upload_mb)
        try:
            decoded = await run_in_threadpool(decode_audio, raw, 16000, int(s.max_upload_mb * 1024 * 1024),
                                              s.max_query_seconds, s.min_query_seconds)
        except AudioError as e:
            raise HTTPException(422, str(e)) from e
    if decoded is None and not (text and text.strip()):
        raise HTTPException(422, "Provide an audio recording or some lyrics text")
    if text and len(text) > 2000:
        raise HTTPException(422, "Text too long (max 2000 characters)")
    if algorithm and algorithm not in state.engine.algorithms:
        raise HTTPException(422, f"Unknown algorithm '{algorithm}'")
    async with state.recognition_slots:
        try:
            res = await run_in_threadpool(
                state.engine.recognize, decoded, text, mode, algorithm, language or us.asr_language,
                top_k, pitch_extractor, fusion_method, True, state.providers_enabled(), us.provider_policy,
                online=us.online_search_enabled)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    res = _enrich(state, res)
    keep_audio = bool(retain_audio and us.retention == "keep_with_consent" and decoded is not None)
    if us.store_history or keep_audio:
        rec = Recognition(mode=mode, resolved_mode=res.get("resolved_mode"), status=res["status"],
                          top_song_id=res["results"][0]["song_id"] if res["results"] else None,
                          result_json="{}", audio_retained=keep_audio)
        with state.catalog.Session() as db:
            db.add(rec)
            db.commit()
            res["id"] = rec.id
            if keep_audio:
                write_wav(state.catalog.root / "recordings" / f"{rec.id}.wav", decoded.samples, decoded.sr)
            rec.result_json = json.dumps(res, default=float)
            db.commit()
    res["audio_retained"] = keep_audio
    return res


def _form_endpoint(mode: str | None):
    async def endpoint(
        audio: UploadFile | None = File(None, description="Recording (wav, mp3, flac, ogg, m4a, webm)"),
        text: str | None = Form(None, description="Lyrics typed by the user"),
        mode_: str | None = Form(None, alias="mode"),
        algorithm: str | None = Form(None),
        language: str | None = Form(None, pattern="^(en|es)$"),
        top_k: int | None = Form(None, ge=1, le=20),
        pitch_extractor: str | None = Form(None),
        fusion_method: str | None = Form(None, pattern="^(weighted|rrf|combmnz)$"),
        retain_audio: bool = Form(False, description="Keep the recording (requires retention consent setting)"),
        state: AppState = Depends(get_state),
    ):
        m = mode or mode_ or state.user_settings.get().default_mode
        return await _recognize(state, m, audio, text, algorithm, language, top_k, pitch_extractor, fusion_method,
                                retain_audio)

    return endpoint


router.add_api_route("/recognize", _form_endpoint(None), methods=["POST"],
                     summary="Recognise a song (mode auto|clip|hum|whistle|lyrics|combined)")
for _m in ("clip", "hum", "whistle", "lyrics", "combined"):
    router.add_api_route(f"/recognize/{_m}", _form_endpoint(_m), methods=["POST"], summary=f"Recognise ({_m} mode)")


@router.get("/recognitions", summary="Recent recognitions (history)")
def list_recognitions(limit: int = 20, state: AppState = Depends(get_state)):
    with state.catalog.Session() as db:
        rows = db.scalars(select(Recognition).order_by(Recognition.created_at.desc()).limit(min(limit, 200))).all()
        out = []
        for r in rows:
            res = json.loads(r.result_json or "{}")
            top = (res.get("results") or [{}])[0]
            out.append({"id": r.id, "created_at": r.created_at.isoformat(), "mode": r.mode,
                        "resolved_mode": r.resolved_mode, "status": r.status, "top_song": top.get("song"),
                        "top_score": top.get("score"), "audio_retained": r.audio_retained})
        return out


@router.get("/recognitions/{rec_id}")
def get_recognition(rec_id: str, state: AppState = Depends(get_state)):
    safe_id(rec_id)
    with state.catalog.Session() as db:
        r = db.get(Recognition, rec_id)
        if r is None:
            raise HTTPException(404, "Recognition not found")
        return json.loads(r.result_json)


@router.delete("/recognitions/{rec_id}")
def delete_recognition(rec_id: str, state: AppState = Depends(get_state)):
    safe_id(rec_id)
    with state.catalog.Session() as db:
        r = db.get(Recognition, rec_id)
        if r is None:
            raise HTTPException(404, "Recognition not found")
        db.delete(r)
        db.commit()
    (state.catalog.root / "recordings" / f"{rec_id}.wav").unlink(missing_ok=True)
    return {"deleted": rec_id}


@router.delete("/recognitions", summary="Delete the whole recognition history and retained recordings")
def clear_recognitions(state: AppState = Depends(get_state)):
    with state.catalog.Session() as db:
        n = db.execute(delete(Recognition)).rowcount
        db.commit()
    rec_dir = state.catalog.root / "recordings"
    if rec_dir.exists():
        for p in rec_dir.glob("*.wav"):
            p.unlink()
    return {"deleted": n, **state.forget_history()}
