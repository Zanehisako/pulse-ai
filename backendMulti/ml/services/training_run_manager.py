"""
In-memory manager for **manual** ("run now") training runs.

The training scripts execute synchronously and can take minutes to hours
(``subprocess.run(..., timeout=7200)`` per script). Running them inside the
HTTP request thread blocks a server worker for the whole duration, ties up
asgiref's bounded thread pool, and — because the client connection is dropped
long before completion — invites duplicate, overlapping batches on retry.

This module dispatches a manual run to a background daemon thread and tracks
its state in memory so the API can return immediately and the UI can poll. A
process-wide lock guarantees **at most one manual run at a time**: a second
request while one is in flight returns the in-flight run instead of starting a
new (overlapping) batch.

State is in-memory only — it is lost on process restart, and is not shared
across multiple server processes. That is acceptable here: the APScheduler that
owns scheduled runs is likewise a single in-process ``BackgroundScheduler``.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Guards every access to ``_runs`` / ``_active_run_id`` below.
_lock = threading.Lock()
# run_id -> run record (insertion-ordered so we can trim the oldest).
_runs: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
# The run_id of the currently-executing run, or None when idle.
_active_run_id: Optional[str] = None
# How many finished runs to retain for status look-ups.
_MAX_HISTORY = 20


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim_locked() -> None:
    """Drop the oldest finished runs beyond ``_MAX_HISTORY``. Caller holds the lock."""
    while len(_runs) > _MAX_HISTORY:
        oldest_id, _ = next(iter(_runs.items()))
        if oldest_id == _active_run_id:
            # Never evict the in-flight run (it is the newest anyway).
            break
        _runs.popitem(last=False)


def is_running() -> bool:
    with _lock:
        return (
            _active_run_id is not None
            and _runs.get(_active_run_id, {}).get("status") == "running"
        )


def get_run(run_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    """
    Return a snapshot of a run record.

    With ``run_id`` → that specific run (or None). Without → the active run if
    one is in flight, otherwise the most recently started run (or None).
    """
    with _lock:
        if run_id is not None:
            record = _runs.get(run_id)
            return dict(record) if record else None
        if _active_run_id is not None and _active_run_id in _runs:
            return dict(_runs[_active_run_id])
        if _runs:
            return dict(next(reversed(_runs.values())))
        return None


def list_runs() -> list[dict[str, Any]]:
    """All retained runs, most recent first."""
    with _lock:
        return [dict(r) for r in reversed(_runs.values())]


def start_run(model_key: Optional[str] = None) -> tuple[dict[str, Any], bool]:
    """
    Start a manual training run in the background.

    Returns ``(run_record, started)``. ``started`` is False when a run was
    already in flight — in that case the existing run is returned unchanged and
    no new batch is launched (idempotent retry guard).
    """
    global _active_run_id

    with _lock:
        if (
            _active_run_id is not None
            and _runs.get(_active_run_id, {}).get("status") == "running"
        ):
            return dict(_runs[_active_run_id]), False

        run_id = uuid.uuid4().hex
        record: dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "model_key": model_key,
            "started_at": _now_iso(),
            "finished_at": None,
            "results": None,
            "error": None,
        }
        _runs[run_id] = record
        _active_run_id = run_id
        _trim_locked()
        snapshot = dict(record)

    thread = threading.Thread(
        target=_execute,
        args=(run_id, model_key),
        name=f"pios-train-run-{run_id[:8]}",
        daemon=True,
    )
    thread.start()
    logger.info("[run-manager] Started manual training run %s (model_key=%s)", run_id, model_key)
    return snapshot, True


def _execute(run_id: str, model_key: Optional[str]) -> None:
    """Background worker: run training and record the outcome."""
    global _active_run_id

    results: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    try:
        # Lazy import to avoid any import-time coupling with the view layer.
        from ml.services.training_scheduler import run_training_now

        results = run_training_now(model_key=model_key)
        status = "failed" if isinstance(results, dict) and results.get("error") else "completed"
    except Exception as exc:  # noqa: BLE001 — surface any failure to the status record
        status = "failed"
        error = str(exc)
        logger.exception("[run-manager] Manual training run %s failed", run_id)
    else:
        logger.info("[run-manager] Manual training run %s finished: %s", run_id, status)

    with _lock:
        record = _runs.get(run_id)
        if record is not None:
            record["status"] = status
            record["results"] = results
            record["error"] = error
            record["finished_at"] = _now_iso()
        if _active_run_id == run_id:
            _active_run_id = None
