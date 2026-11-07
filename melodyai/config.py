"""Global paths, version constants and device detection."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Bump these when the corresponding algorithm changes in a way that makes stored
# artefacts incompatible. Songs indexed with an older version are flagged "stale".
FEATURE_VERSION = "mel-v1"
INDEX_VERSION = "idx-v1"

# Common analysis constants. Queries and references are compared on the same grid.
ANALYSIS_SR = 16000          # sample rate used for all analysis
PITCH_HOP_S = 0.01           # pitch tracker hop (10 ms)
CONTOUR_RATE = 20.0          # contour frame rate (Hz) used for matching


def data_dir() -> Path:
    return Path(os.environ.get("MELODYAI_DATA_DIR", ROOT_DIR / "data")).resolve()


def store_dir() -> Path:
    """Directory holding the catalog database, feature store and indexes."""
    return Path(os.environ.get("MELODYAI_STORE_DIR", data_dir() / "store")).resolve()


def models_dir() -> Path:
    return Path(os.environ.get("MELODYAI_MODELS_DIR", ROOT_DIR / "models")).resolve()


def experiments_dir() -> Path:
    return Path(os.environ.get("MELODYAI_EXPERIMENTS_DIR", ROOT_DIR / "experiments")).resolve()


@lru_cache(maxsize=1)
def torch_device(preferred: str = "auto") -> str:
    """Return the best available torch device ("cuda", "mps" or "cpu").

    Torch is optional; when it is not installed we report "cpu".
    """
    preferred = os.environ.get("MELODYAI_DEVICE", preferred)
    try:
        import torch
    except ImportError:  # pragma: no cover - depends on environment
        return "cpu"
    if preferred != "auto":
        if preferred == "cuda" and torch.cuda.is_available():
            return "cuda"
        if preferred == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
