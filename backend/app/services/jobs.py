"""Background jobs (evaluation, training, reindexing, demo import, model download).

CPU-heavy jobs run in a single worker thread (they must not run concurrently on a laptop).
Network-bound kinds (``OWN_LANE``: the popular-songs sync, which spends hours waiting on
LRCLIB's rate limit) get a worker of their own, so they never hold back a model download, a
reindex or an evaluation; each such lane still runs one job at a time. Job state is persisted
in the ``jobs`` table so it survives page reloads, and cancellation is cooperative: the job
function receives ``should_stop()``. Jobs left "running" by a previous server process are
marked "interrupted" at start-up.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from sqlalchemy import select

from melodyai.catalog.models import Job

log = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


def job_to_dict(j: Job) -> dict:
    return {
        "id": j.id, "kind": j.kind, "status": j.status, "progress": round(j.progress or 0.0, 4),
        "message": j.message, "config": json.loads(j.config_json or "{}"),
        "result": json.loads(j.result_json) if j.result_json else None, "error": j.error,
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "started_at": j.started_at.isoformat() if j.started_at else None,
        "finished_at": j.finished_at.isoformat() if j.finished_at else None,
    }


class JobManager:
    OWN_LANE = ("world_sync",)       # I/O-bound kinds with a worker of their own

    def __init__(self, Session) -> None:
        self.Session = Session
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="melodyai-job")
        self.lanes = {kind: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"melodyai-{kind}")
                      for kind in self.OWN_LANE}
        self._cancel: dict[str, threading.Event] = {}
        with self.Session() as db:
            for j in db.scalars(select(Job).where(Job.status.in_(["queued", "running"]))).all():
                j.status, j.error, j.finished_at = "interrupted", "server restarted", _now()
            db.commit()

    def submit(self, kind: str, config: dict, fn: Callable[[Callable, Callable], dict]) -> dict:
        with self.Session() as db:
            j = Job(kind=kind, status="queued", config_json=json.dumps(config))
            db.add(j)
            db.commit()
            jid = j.id
        ev = threading.Event()
        self._cancel[jid] = ev
        self.lanes.get(kind, self.pool).submit(self._run, jid, fn, ev)
        return self.get(jid)

    def shutdown(self, wait: bool = False, cancel_futures: bool = True) -> None:
        for pool in (self.pool, *self.lanes.values()):
            pool.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _update(self, jid: str, **fields) -> None:
        with self.Session() as db:
            j = db.get(Job, jid)
            for k, v in fields.items():
                setattr(j, k, v)
            db.commit()

    def _run(self, jid: str, fn, ev: threading.Event) -> None:
        if ev.is_set():
            self._update(jid, status="cancelled", finished_at=_now())
            return
        self._update(jid, status="running", started_at=_now())
        last = {"t": 0.0}

        def progress(p: float, msg: str) -> None:
            import time

            now = time.time()
            if now - last["t"] > 0.5 or p >= 1.0:  # throttle DB writes
                last["t"] = now
                self._update(jid, progress=float(p), message=msg[:500])

        try:
            result = fn(progress, ev.is_set)
            if ev.is_set() or (isinstance(result, dict) and result.get("status") == "cancelled"):
                self._update(jid, status="cancelled", finished_at=_now(),
                             result_json=json.dumps(result, default=str) if result else None)
            else:
                self._update(jid, status="completed", progress=1.0, finished_at=_now(),
                             result_json=json.dumps(result, default=str))
        except Exception as e:  # report but never crash the worker
            cancelled = ev.is_set() and "cancel" in str(e).lower()
            log.error("job %s failed: %s", jid, traceback.format_exc())
            self._update(jid, status="cancelled" if cancelled else "failed", error=str(e)[:4000],
                         finished_at=_now())
        finally:
            self._cancel.pop(jid, None)

    def cancel(self, jid: str) -> dict:
        ev = self._cancel.get(jid)
        job = self.get(jid)
        if job["status"] in ("completed", "failed", "cancelled", "interrupted"):
            return job
        if ev:
            ev.set()
        if job["status"] == "queued":
            self._update(jid, status="cancelled", finished_at=_now())
        else:
            self._update(jid, message="cancellation requested")
        return self.get(jid)

    def get(self, jid: str) -> dict:
        with self.Session() as db:
            j = db.get(Job, jid)
            if j is None:
                raise KeyError(jid)
            return job_to_dict(j)

    def list(self, kind: str | None = None, limit: int = 50) -> list[dict]:
        with self.Session() as db:
            stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)
            if kind:
                stmt = stmt.where(Job.kind == kind)
            return [job_to_dict(j) for j in db.scalars(stmt).all()]

    def wait(self, jid: str, timeout: float = 60.0) -> dict:
        """Test helper: block until the job is finished."""
        import time

        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.get(jid)
            if j["status"] in ("completed", "failed", "cancelled", "interrupted"):
                return j
            time.sleep(0.05)
        raise TimeoutError(jid)
