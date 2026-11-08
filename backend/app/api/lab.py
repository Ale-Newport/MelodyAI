"""Audio analysis, models, experiments, training jobs, settings and health."""

from __future__ import annotations

import json
import re

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.deps import get_state, read_upload, safe_id
from app.core.state import AppState, discover_checkpoints
from app.services.analysis import analyze
from melodyai import __version__
from melodyai.audio.io import AudioError, decode_audio, ffmpeg_available
from melodyai.config import ROOT_DIR, experiments_dir, torch_device
from melodyai.lyrics.asr import (
    AUTO,
    AUTO_DOWNLOAD,
    AUTO_PREFERENCE,
    MODEL_CHOICES,
    MODEL_SIZES,
    asr_available,
    model_status,
    resolve_model,
)
from melodyai.pitch.extractors import list_extractors

router = APIRouter(prefix="/api", tags=["lab"])
RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[A-Za-z0-9_\-]{1,40}$")
EVAL_CONFIGS = {"quick": "configs/eval_quick.yaml", "default": "configs/eval_default.yaml"}
TRAIN_CONFIGS = {"default": "configs/train_default.yaml", "smoke": "configs/train_smoke.yaml"}
_AUTO_ORDER = {name: i for i, name in enumerate(AUTO_PREFERENCE)}


# ------------------------------------------------------------------ analysis
@router.post("/audio/analyze", summary="Full melodic analysis of a recording (Melody Explorer)")
async def analyze_audio(
    audio: UploadFile = File(...),
    profile: str = Form("hum", pattern="^(hum|whistle|sing)$"),
    extractor: str = Form("pyin"),
    song_id: str | None = Form(None),
    match: bool = Form(True, description="If no song_id is given, compare with the best melody match"),
    state: AppState = Depends(get_state),
):
    raw = await read_upload(audio, state.settings.max_upload_mb)
    try:
        dec = await run_in_threadpool(decode_audio, raw, 16000, None, state.settings.max_query_seconds,
                                      state.settings.min_query_seconds)
    except AudioError as e:
        raise HTTPException(422, str(e)) from e
    if extractor not in {e["name"] for e in list_extractors() if e["available"]}:
        raise HTTPException(422, f"Pitch extractor '{extractor}' not available")
    if song_id:
        safe_id(song_id)

    def match_fn(q):
        alg = state.engine.algorithms[state.engine.settings.melody_algorithm]
        c = alg.search(q, state.catalog.melody_index, 1)
        return c[0].to_dict() if c else None

    async with state.recognition_slots:
        out = await run_in_threadpool(analyze, dec, profile, extractor, state.catalog.melody_index, song_id,
                                      match_fn if match else None)
    if "comparison" in out:
        out["comparison"]["song"] = state.catalog.songs_by_ids([out["comparison"]["song_id"]]).get(
            out["comparison"]["song_id"])
    return out


# ------------------------------------------------------------------ models
def _asr_info(state: AppState) -> dict:
    """Whisper models. ``models`` lists the concrete sizes (incl. large-v3-turbo); the ``"auto"`` choice (best
    cached model) is described apart, with what it resolves to, so clients can offer it as their own option."""
    models = [model_status(m) for m in MODEL_SIZES]
    best = next((m["name"] for m in sorted(models, key=lambda m: _AUTO_ORDER.get(m["name"], 99)) if m["cached"]),
                None)
    auto = {"name": AUTO, "installed_package": asr_available(), "cached": best is not None, "resolved": best,
            "download": None if best else AUTO_DOWNLOAD}
    active = state.engine.settings.asr_model
    return {"package_installed": asr_available(), "models": models, "choices": list(MODEL_CHOICES), "auto": auto,
            "active": active, "active_resolved": best if active == AUTO else active,
            "downloads_allowed": state.settings.allow_model_download,
            "note": "faster-whisper (CTranslate2) runs on CPU or CUDA; Apple MPS is not supported."}


@router.get("/models", summary="Available pitch extractors, matching algorithms, ASR and neural models")
def models(state: AppState = Depends(get_state)):
    return {
        "pitch_extractors": list_extractors(),
        "melody_algorithms": state.engine.list_algorithms(),
        "default_algorithm": state.engine.settings.melody_algorithm,
        "fusion_methods": ["weighted", "rrf", "combmnz"],
        "asr": _asr_info(state),
        "neural": {"active": state.neural, "error": state.neural_error,
                   "checkpoints": [{k: v for k, v in c.items() if k != "path"} for c in discover_checkpoints()]},
        "inference_device": torch_device(),
        "providers": [p.describe() for p in state.providers],
        "providers_enabled": state.providers_enabled(),
    }


class ActivateBody(BaseModel):
    checkpoint: str = Field(..., max_length=200)


@router.post("/models/neural/activate", summary="Load a neural checkpoint for inference")
async def activate_neural(body: ActivateBody, state: AppState = Depends(get_state)):
    try:
        return await run_in_threadpool(state.activate_neural, body.checkpoint)
    except KeyError as e:
        raise HTTPException(404, "Unknown checkpoint") from e
    except Exception as e:
        raise HTTPException(500, f"Could not load checkpoint: {e}") from e


class AsrDownloadBody(BaseModel):
    """A concrete Whisper model size to download."""

    model: str = Field("base", pattern="^(" + "|".join(map(re.escape, MODEL_SIZES)) + ")$")


class AsrDownloadRequest(BaseModel):
    """Download request: a concrete size, or ``"auto"`` (what the "auto" setting resolves to)."""

    model: str = Field("base", pattern="^(" + "|".join(map(re.escape, MODEL_CHOICES)) + ")$")


@router.post("/models/asr/download", summary="Download a Whisper model (must be allowed by the server)")
def download_asr(body: AsrDownloadRequest, state: AppState = Depends(get_state)):
    if not state.settings.allow_model_download:
        raise HTTPException(403, "Model downloads are disabled. Start the server with MELODYAI_ALLOW_MODEL_DOWNLOAD=1 "
                                 "or run `make asr-model`.")
    if not asr_available():
        raise HTTPException(400, "faster-whisper is not installed")
    # "auto" = the best cached model, or the recommended one (large-v3-turbo) when nothing is cached yet.
    target = AsrDownloadBody(model=resolve_model(body.model, allow_download=True) if body.model == AUTO else body.model)

    def run(progress, should_stop):
        from faster_whisper.utils import download_model

        progress(0.05, f"downloading {target.model}")
        download_model(target.model)
        return model_status(target.model)

    return state.jobs.submit("asr_download", {**target.model_dump(), "requested": body.model}, run)


# ------------------------------------------------------------------ experiments
def _run_summary(d) -> dict | None:
    try:
        m = json.loads((d / "metrics.json").read_text())
    except (OSError, ValueError):
        return None
    if m.get("kind") == "scale":
        return {"run_id": m["run_id"], "kind": "scale", "created_at": m.get("created_at"),
                "elapsed_s": m.get("elapsed_s"), "dataset": m.get("dataset"), "algorithms": {}}
    mel = m.get("melody", {})
    return {"run_id": m["run_id"], "created_at": m.get("created_at"), "dataset": m.get("dataset"),
            "elapsed_s": m.get("elapsed_s"),
            "algorithms": {k: {"top1": v["overall"].get("top1"), "top5": v["overall"].get("top5"),
                               "mrr": v["overall"].get("mrr"), "latency_ms": v["latency"].get("mean_ms")}
                           for k, v in mel.items()}}


@router.get("/experiments", summary="Evaluation runs and training runs")
def list_experiments():
    runs, scale = [], []
    for d in sorted((experiments_dir() / "runs").glob("*"), reverse=True):
        s = _run_summary(d)
        if s:
            (scale if s.get("kind") == "scale" else runs).append(s)
    training = []
    for d in sorted((experiments_dir() / "training").glob("*"), reverse=True):
        try:
            r = json.loads((d / "result.json").read_text())
            training.append({"name": d.name, "status": r.get("status"), "best_val_mrr": r.get("best_val_mrr"),
                             "epochs_run": r.get("epochs_run"), "finished_at": r.get("finished_at"),
                             "history": r.get("history"), "device": r.get("device"), "n_params": r.get("n_params"),
                             "train_seconds": r.get("train_seconds")})
        except (OSError, ValueError):
            hist = d / "history.json"
            training.append({"name": d.name, "status": "incomplete",
                             "history": json.loads(hist.read_text()) if hist.exists() else []})
    return {"evaluations": runs, "scale": scale, "training": training}


def _run_dir(run_id: str):
    if not RUN_ID.match(run_id):
        raise HTTPException(400, "Invalid run id")
    d = experiments_dir() / "runs" / run_id
    if not (d / "metrics.json").exists():
        raise HTTPException(404, "Run not found")
    return d


@router.get("/experiments/{run_id}")
def get_experiment(run_id: str):
    return json.loads((_run_dir(run_id) / "metrics.json").read_text())


@router.get("/experiments/{run_id}/queries", summary="Per-query results of a run (filterable)")
def experiment_queries(run_id: str, algorithm: str | None = None, condition: str | None = None,
                       only_errors: bool = False, offset: int = 0, limit: int = 200):
    rows = []
    with (_run_dir(run_id) / "per_query.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if algorithm and r["algorithm"] != algorithm:
                continue
            if condition and r["condition"] != condition:
                continue
            if only_errors and (r["target"] is None or r["rank"] == 1):
                continue
            rows.append(r)
    return {"total": len(rows), "items": rows[offset:offset + min(limit, 1000)]}


@router.get("/experiments/{run_id}/figures/{name}")
def experiment_figure(run_id: str, name: str):
    if not re.match(r"^[a-z0-9_]+\.png$", name):
        raise HTTPException(400, "Invalid figure name")
    p = _run_dir(run_id) / "figures" / name
    if not p.exists():
        raise HTTPException(404, "Figure not found")
    return FileResponse(p, media_type="image/png")


class EvaluateBody(BaseModel):
    config: str = Field("quick", pattern="^(quick|default)$")


@router.post("/experiments/evaluate", summary="Run a benchmark in the background")
def evaluate(body: EvaluateBody = Body(default_factory=EvaluateBody), state: AppState = Depends(get_state)):
    from melodyai.evaluation.runner import run_evaluation

    def run(progress, should_stop):
        res = run_evaluation(ROOT_DIR / EVAL_CONFIGS[body.config], progress, should_stop)
        if (ROOT_DIR / "configs" / "calibration.json").exists():
            state.engine.reload_calibration()
        return {"run_id": res["run_id"], "summary": _run_summary(__import__("pathlib").Path(res["run_dir"]))}

    return state.jobs.submit("evaluation", body.model_dump(), run)


# ------------------------------------------------------------------ training
class TrainBody(BaseModel):
    config: str = Field("default", pattern="^(default|smoke)$")
    epochs: int | None = Field(None, ge=1, le=200)
    loss: str | None = Field(None, pattern="^(infonce|triplet)$")
    lr: float | None = Field(None, gt=0, le=0.1)
    batch_size: int | None = Field(None, ge=8, le=1024)
    resume: bool = False


@router.post("/training/jobs", status_code=201, summary="Start a training job (asynchronous)")
def start_training(body: TrainBody, state: AppState = Depends(get_state)):
    from melodyai.training.train import train_from_config

    overrides = {k: v for k, v in body.model_dump().items() if k not in ("config", "resume") and v is not None}

    def run(progress, should_stop):
        return train_from_config(ROOT_DIR / TRAIN_CONFIGS[body.config], progress, should_stop, body.resume, overrides)

    return state.jobs.submit("training", body.model_dump(), run)


@router.get("/training/jobs")
def list_training_jobs(state: AppState = Depends(get_state)):
    return state.jobs.list("training")


@router.get("/training/jobs/{job_id}")
def get_training_job(job_id: str, state: AppState = Depends(get_state)):
    return get_job(job_id, state)


@router.post("/training/jobs/{job_id}/cancel")
def cancel_training_job(job_id: str, state: AppState = Depends(get_state)):
    return cancel_job(job_id, state)


@router.get("/jobs")
def list_jobs(kind: str | None = None, state: AppState = Depends(get_state)):
    return state.jobs.list(kind)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, state: AppState = Depends(get_state)):
    try:
        return state.jobs.get(safe_id(job_id))
    except KeyError as e:
        raise HTTPException(404, "Job not found") from e


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, state: AppState = Depends(get_state)):
    try:
        return state.jobs.cancel(safe_id(job_id))
    except KeyError as e:
        raise HTTPException(404, "Job not found") from e


# ------------------------------------------------------------------ settings & health
@router.get("/settings")
def get_settings(state: AppState = Depends(get_state)):
    return state.user_settings.get().model_dump()


@router.put("/settings")
def put_settings(patch: dict = Body(...), state: AppState = Depends(get_state)):
    try:
        s = state.user_settings.update(patch)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    state.apply_user_settings()
    return s.model_dump()


@router.get("/health")
def health(state: AppState = Depends(get_state)):
    st = state.catalog.status()
    return {"status": "ok", "version": __version__, "songs": st["songs"], "indexed": st["by_status"].get("indexed", 0),
            "with_lyrics": st["with_lyrics"], "asr_ready": state.engine.asr_ready(),
            "neural_model": (state.neural or {}).get("checkpoint"), "device": torch_device(),
            "fingerprints": len(state.catalog.fp_index), "providers_enabled": state.providers_enabled(),
            "ffmpeg": ffmpeg_available(), "limits": {"max_upload_mb": state.settings.max_upload_mb,
                                                     "max_query_seconds": state.settings.max_query_seconds}}
