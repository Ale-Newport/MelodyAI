"""Process pools that are safe to start from a multi-threaded process.

By the time a pool is created, numba, OpenMP (PyTorch, FAISS, BLAS) and Python threads are usually running.
``fork`` (the Linux default before Python 3.14) copies such a process with its locks in whatever state other
threads left them, and the children can deadlock or crash: a pYIN worker forked from the benchmark segfaulted
inside a numba kernel on a Linux CI runner. ``spawn`` starts clean interpreters instead. It is already the
default on macOS and Windows, so every worker function used here is importable and picklable.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def process_pool(max_workers: int | None = None) -> ProcessPoolExecutor:
    """A ``ProcessPoolExecutor`` whose workers are spawned, not forked, on every platform."""
    return ProcessPoolExecutor(max_workers, mp_context=multiprocessing.get_context("spawn"))
