"""Contrastive training of the melody encoder.

    python -m melodyai.cli train --config configs/train_default.yaml [--resume]

Features: YAML config, global seeding, composition-level splits with an explicit
leakage check, on-the-fly augmentation, InfoNCE (symmetric) or triplet loss with
in-batch hardest negatives, AdamW + warmup/cosine schedule, TensorBoard logging,
per-epoch retrieval validation, checkpoints (last/best), early stopping, resume,
cancellation hook and export of the best model to ``models/melody_encoder.pt``.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from melodyai.config import ROOT_DIR, experiments_dir, models_dir, torch_device
from melodyai.features.windows import DEFAULT_LENGTHS, build_windows
from melodyai.training.data import (
    PairSampler,
    build_pipeline_pool,
    compositions,
    contour_vector,
    remove_leaky,
    simulate_tracker,
)


@dataclass
class TrainConfig:
    name: str = "melody_encoder"
    seed: int = 1234
    n_train: int = 3000
    n_val: int = 150
    n_points: int = 96
    channels: int = 128
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    asymmetric: bool = False
    loss: str = "infonce"          # infonce | triplet
    temperature: float = 0.07
    margin: float = 0.2
    batch_size: int = 256
    steps_per_epoch: int = 100
    epochs: int = 30
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    patience: int = 6
    pipeline_pool_comps: int = 300  # compositions rendered to audio + real pYIN (0 = disable)
    pipeline_fraction: float = 0.25
    val_queries_per_comp: int = 2
    device: str = "auto"
    export: bool = True
    # Real melodies (a world melody index built from Lakh MIDI, melodyai.world.lakh) added to the synthetic
    # training compositions; keys listed in ``real_exclude`` (one per line) are left out for evaluation.
    real_melodies: str = ""
    real_max: int = 0
    real_exclude: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path) -> TrainConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def load_real_melodies(db: str | Path, max_n: int, exclude_file: str = "", seed: int = 0) -> list:
    """Vocal lines from a world melody index (``melodies.db``), minus the excluded song keys, seeded sample."""
    import sqlite3

    from melodyai.world.melodies import unpack_notes

    path = Path(db)
    if path.is_dir():
        path = path / "melodies.db"
    excluded = set()
    if exclude_file and Path(exclude_file).is_file():
        excluded = {ln.strip() for ln in Path(exclude_file).read_text(encoding="utf-8").splitlines() if ln.strip()}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = [(k, b) for k, b in con.execute("SELECT key, notes FROM melodies") if k not in excluded]
    con.close()
    order = np.random.default_rng(seed).permutation(len(rows))[:max_n]
    return [unpack_notes(rows[i][1]) for i in order]


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def info_nce(q, r, temperature: float):
    import torch
    import torch.nn.functional as F

    logits = q @ r.T / temperature
    target = torch.arange(len(q), device=q.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def triplet_hardest(q, r, margin: float):
    import torch

    sim = q @ r.T
    pos = sim.diag()
    neg = sim - 2.0 * torch.eye(len(q), device=q.device)
    hardest = neg.max(dim=1).values
    return torch.relu(margin - pos + hardest).mean()


def _embed(model, X: np.ndarray, device: str, is_query: bool, bs: int = 1024) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i + bs]).float().to(device), is_query=is_query).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config["dim"]), np.float32)


def build_val_set(comps, n_points: int, per_comp: int, seed: int) -> dict:
    """Reference windows of every validation composition + simulated queries (fixed seed)."""
    rng = np.random.default_rng(seed)
    W, owner = [], []
    for ci, n in enumerate(comps):
        w = build_windows(n, DEFAULT_LENGTHS, n_points)
        W.append(w.vectors)
        owner.append(np.full(len(w), ci))
    sampler = PairSampler(comps, n_points, seed + 1)
    Q, qo = [], []
    for ci, n in enumerate(comps):
        for _ in range(per_comp):
            win = sampler.sample_window(n)
            from melodyai.demo.queries import perturb
            from melodyai.training.data import AUG_SPEC

            qn, _ = perturb(win, AUG_SPEC, rng)
            Q.append(contour_vector(simulate_tracker(qn, rng), n_points))
            qo.append(ci)
    return {"W": np.concatenate(W).astype(np.float32), "W_owner": np.concatenate(owner),
            "Q": np.asarray(Q, np.float32), "Q_owner": np.asarray(qo)}


def retrieval_metrics(Eq: np.ndarray, Ew: np.ndarray, w_owner: np.ndarray, q_owner: np.ndarray) -> dict:
    sims = Eq @ Ew.T
    n_songs = int(w_owner.max()) + 1
    song_sim = np.full((len(Eq), n_songs), -np.inf)
    for s in range(n_songs):
        song_sim[:, s] = sims[:, w_owner == s].max(axis=1)
    order = np.argsort(-song_sim, axis=1)
    ranks = np.array([int(np.flatnonzero(order[i] == q_owner[i])[0]) + 1 for i in range(len(Eq))])
    return {"top1": float(np.mean(ranks == 1)), "top5": float(np.mean(ranks <= 5)), "mrr": float(np.mean(1 / ranks))}


def _env_info(device: str) -> dict:
    import torch

    return {"python": platform.python_version(), "torch": torch.__version__, "device": device,
            "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()}


def train(cfg: TrainConfig, run_dir: Path | None = None, progress=None, should_stop=None,
          resume: bool = False) -> dict:
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from melodyai.models.encoder import MelodyEncoder, save_checkpoint

    device = torch_device(cfg.device)
    seed_everything(cfg.seed)
    cfg_hash = hashlib.sha1(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:8]
    run_dir = Path(run_dir or experiments_dir() / "training" / f"{cfg.name}-{cfg_hash}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    report = progress or (lambda p, m: None)

    # ---- data (composition-level splits + leakage check against val/test/demo)
    report(0.0, "generating compositions")
    train_comps = compositions("train", cfg.n_train)
    val_comps = compositions("val", cfg.n_val)
    from melodyai.demo.generator import generate_catalog

    demo = [m.notes for m in generate_catalog(48)]
    train_comps, leak = remove_leaky(train_comps, val_comps + demo + compositions("test", 100))
    leak.pop("indices")
    n_real = 0
    if cfg.real_melodies and cfg.real_max > 0:
        real = load_real_melodies(cfg.real_melodies, cfg.real_max, cfg.real_exclude, cfg.seed)
        n_real = len(real)
        train_comps = train_comps + real
    (run_dir / "split_report.json").write_text(json.dumps(
        {"train_seed_range": "100000+", "val_seed_range": "200000+", "test_seed_range": "300000+",
         "demo_seed_range": "1000+", "removed_for_leakage": leak, "n_train_after": len(train_comps),
         "real_melodies": n_real, "real_source": cfg.real_melodies, "real_exclude": cfg.real_exclude}, indent=1))

    pool = None
    if cfg.pipeline_pool_comps > 0:
        pool_path = experiments_dir() / "cache" / f"pool_v2_{cfg.seed}_{cfg.pipeline_pool_comps}_{cfg.n_points}.npz"
        if pool_path.exists():
            z = np.load(pool_path, allow_pickle=True)
            pool = {"queries": z["queries"], "refs": z["refs"], "by_comp": z["by_comp"].item()}
        else:
            pool = build_pipeline_pool(train_comps, 1, cfg.n_points, cfg.seed, cfg.pipeline_pool_comps,
                                       progress=lambda p, m: report(0.02 + 0.08 * p, m))
            pool_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(pool_path, queries=pool["queries"], refs=pool["refs"],
                     by_comp=np.array(pool["by_comp"], dtype=object))
    sampler = PairSampler(train_comps, cfg.n_points, cfg.seed, pool, cfg.pipeline_fraction)
    val = build_val_set(val_comps, cfg.n_points, cfg.val_queries_per_comp, cfg.seed + 99)

    # ---- model / optimisation
    model = MelodyEncoder(cfg.n_points, cfg.channels, cfg.dim, cfg.n_layers, cfg.n_heads, cfg.dropout,
                          cfg.asymmetric).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = cfg.epochs * cfg.steps_per_epoch

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    history, start_epoch, best, bad_epochs = [], 0, -1.0, 0
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch, best, bad_epochs, history = ck["epoch"] + 1, ck["best"], ck["bad_epochs"], ck["history"]
        sampler.rng = np.random.default_rng(cfg.seed + start_epoch)
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    n_params = sum(p.numel() for p in model.parameters())
    t_start = time.time()
    status = "completed"

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        losses = []
        t0 = time.time()
        for step in range(cfg.steps_per_epoch):
            if should_stop and should_stop():
                status = "cancelled"
                break
            b = sampler.batch(cfg.batch_size)
            q = model(torch.from_numpy(b.queries).to(device), is_query=True)
            r = model(torch.from_numpy(b.refs).to(device))
            loss = info_nce(q, r, cfg.temperature) if cfg.loss == "infonce" else triplet_hardest(q, r, cfg.margin)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.item()))
            report((epoch + (step + 1) / cfg.steps_per_epoch) / cfg.epochs,
                   f"epoch {epoch + 1}/{cfg.epochs} loss {np.mean(losses):.3f}")
        if status == "cancelled":
            break
        model.eval()
        m = retrieval_metrics(_embed(model, val["Q"], device, True), _embed(model, val["W"], device, False),
                              val["W_owner"], val["Q_owner"])
        rec = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0],
               "val_top1": m["top1"], "val_top5": m["top5"], "val_mrr": m["mrr"], "epoch_s": time.time() - t0}
        history.append(rec)
        for k, v in rec.items():
            if k != "epoch":
                writer.add_scalar(k, v, epoch + 1)
        improved = m["mrr"] > best + 1e-4
        if improved:
            best, bad_epochs = m["mrr"], 0
            save_checkpoint(best_path, model, {"epoch": epoch + 1, "val": m, "train_config": asdict(cfg)})
        else:
            bad_epochs += 1
        torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "best": best, "bad_epochs": bad_epochs, "history": history,
                    "config": model.config}, last_path)
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
        if bad_epochs >= cfg.patience:
            status = "early_stopped"
            break
    writer.close()

    result = {
        "run_dir": str(run_dir.relative_to(ROOT_DIR)) if ROOT_DIR in run_dir.parents else str(run_dir), "status": status, "best_val_mrr": best, "epochs_run": len(history),
        "n_params": n_params, "device": device, "train_seconds": round(time.time() - t_start, 1),
        "n_train_compositions": len(train_comps), "pipeline_pool_size": int(len(pool["queries"])) if pool else 0,
        "history": history, "env": _env_info(device), "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if cfg.export and best_path.exists():
        models_dir().mkdir(parents=True, exist_ok=True)
        dst = models_dir() / f"{cfg.name}.pt"
        shutil.copy(best_path, dst)
        result["exported_model"] = str(dst.relative_to(ROOT_DIR)) if ROOT_DIR in dst.parents else str(dst)
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    return result


def train_from_config(path, progress=None, should_stop=None, resume: bool = False, overrides: dict | None = None):
    cfg = TrainConfig.from_yaml(path)
    for k, v in (overrides or {}).items():
        if k in TrainConfig.__dataclass_fields__:
            setattr(cfg, k, v)
    return train(cfg, progress=progress, should_stop=should_stop, resume=resume)
