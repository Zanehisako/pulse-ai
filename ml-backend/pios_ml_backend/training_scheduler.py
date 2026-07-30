"""
training_scheduler.py
=====================
Background scheduler that triggers ML training scripts based on a JSON
configuration file. Designed to run inside a long-lived host process
so the training utilities do not need to stop.

Uses APScheduler (BackgroundScheduler) with CronTrigger for cron-like
scheduling (daily, weekly, monthly).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime
from datetime import timezone as dt_timezone
from pathlib import Path
from typing import Any, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger("pios.training_scheduler")

_ROOT_DIR = Path(__file__).resolve().parents[1]  # ml-backend/
_TRAINING_SCRIPTS_DIR = _ROOT_DIR / "training_scripts"
_DEFAULT_CONFIG_PATH = _ROOT_DIR / "config" / "training_schedule.json"

# Module-level singleton
_scheduler: BackgroundScheduler | None = None
_scheduler_lock = threading.Lock()

# Optional callback invoked after a successful registry sync so that the
# host process can refresh its in-memory model registry.
_post_sync_hook: Callable[[], None] | None = None

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_schedule_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and validate the training schedule JSON config."""
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        logger.warning(
            "Training schedule config not found at %s — using empty config", path
        )
        return {
            "defaults": {},
            "scripts": {},
            "post_training": {"sync_registry": True},
        }

    with open(path, "r", encoding="utf-8") as fh:
        config: dict[str, Any] = json.load(fh)

    # Ensure required top-level keys
    config.setdefault("defaults", {})
    config.setdefault("scripts", {})
    config.setdefault("post_training", {"sync_registry": True})
    return config


def reload_schedule_config(config_path: Path | None = None) -> dict[str, Any]:
    """Reload config and reschedule all jobs.  Returns the new config."""
    config = load_schedule_config(config_path)
    if _scheduler is not None:
        _apply_schedule(config)
    return config


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _build_child_env() -> dict[str, str]:
    """Build an augmented environment for child training processes."""
    env = os.environ.copy()
    extra_paths = [
        str(_ROOT_DIR),  # ml-backend/
        str(_ROOT_DIR.parent / "backendMulti"),  # backendMulti/
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing] if existing else []))
    env.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
    env["PYTHONUTF8"] = "1"
    return env


def _run_training_script(script_name: str, model_key: str) -> bool:
    """
    Execute a single training script as a subprocess.
    Returns True on success, False on failure.
    """
    script_path = _TRAINING_SCRIPTS_DIR / script_name
    if not script_path.exists():
        logger.error("[scheduler] Training script not found: %s", script_path)
        return False

    logger.info("[scheduler] ▶ Starting training: %s (%s)", model_key, script_name)
    start = datetime.now(dt_timezone.utc)

    try:
        result = subprocess.run(
            [sys.executable, "-u", str(script_path)],
            env=_build_child_env(),
            capture_output=True,
            text=True,
            timeout=7200,  # 2-hour hard timeout per script
        )
        elapsed = (datetime.now(dt_timezone.utc) - start).total_seconds()

        if result.returncode == 0:
            logger.info(
                "[scheduler] ✅ Finished training: %s in %.1fs",
                model_key,
                elapsed,
            )
            if result.stdout:
                # Log last 20 lines of stdout for visibility
                tail = "\n".join(result.stdout.strip().splitlines()[-20:])
                logger.info("[scheduler] stdout tail:\n%s", tail)
            return True
        else:
            logger.error(
                "[scheduler] ❌ Training FAILED: %s (exit code %d) in %.1fs",
                model_key,
                result.returncode,
                elapsed,
            )
            if result.stderr:
                logger.error("[scheduler] stderr:\n%s", result.stderr[-2000:])
            return False

    except subprocess.TimeoutExpired:
        logger.error("[scheduler] ⏰ Training TIMED OUT: %s (>2h)", model_key)
        return False
    except Exception:
        logger.exception("[scheduler] 💥 Unexpected error running: %s", model_key)
        return False


def _run_promotion_gate(model_name: str | None = None) -> bool:
    """
    Run the champion/challenger promotion gate (promote_models.py).

    Compares the newly trained "challenger" against the current "champion"
    using metrics and promotion rules from ``training_schedule.json``.
    The challenger is promoted **only** if it meets or beats the champion.

    Returns True if the script ran successfully, False otherwise.
    """
    promote_script = _ROOT_DIR / "scripts" / "promote_models.py"
    if not promote_script.exists():
        logger.warning("[scheduler] Promotion script not found: %s", promote_script)
        return False

    cmd = [sys.executable, "-u", str(promote_script)]
    if model_name:
        cmd.extend(["--model-name", model_name])

    logger.info(
        "[scheduler] ⚖️  Running promotion gate%s …",
        f" for {model_name}" if model_name else "",
    )
    try:
        result = subprocess.run(
            cmd,
            env=_build_child_env(),
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute timeout for promotion checks
        )
        if result.stdout:
            tail = "\n".join(result.stdout.strip().splitlines()[-30:])
            logger.info("[scheduler] promote stdout:\n%s", tail)
        if result.returncode == 0:
            logger.info("[scheduler] ✅ Promotion gate complete")
            return True
        else:
            logger.error(
                "[scheduler] ❌ Promotion gate failed (exit code %d)",
                result.returncode,
            )
            if result.stderr:
                logger.error("[scheduler] stderr:\n%s", result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        logger.error("[scheduler] ⏰ Promotion gate timed out (>10min)")
        return False
    except Exception:
        logger.exception("[scheduler] 💥 Error during promotion gate")
        return False


def _run_registry_sync() -> None:
    """
    Run the MLflow→Django registry sync (sync_registry.py).

    This script only reads the MLflow model registry and upserts
    ``MLModelConfig`` rows — it does **not** re-run training scripts.
    After a successful sync the optional ``_post_sync_hook`` is called
    so the host process can refresh its in-memory model registry.
    """
    sync_script = _ROOT_DIR / "scripts" / "sync_registry.py"
    if not sync_script.exists():
        logger.warning("[scheduler] Registry sync script not found: %s", sync_script)
        return

    logger.info("[scheduler] 🔄 Running post-training registry sync …")
    try:
        result = subprocess.run(
            [sys.executable, "-u", str(sync_script)],
            env=_build_child_env(),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode == 0:
            logger.info("[scheduler] ✅ Registry sync complete")
            # Notify host process to reload models into memory
            _invoke_post_sync_hook()
        else:
            logger.error(
                "[scheduler] ❌ Registry sync failed (exit code %d)",
                result.returncode,
            )
            if result.stderr:
                logger.error("[scheduler] stderr:\n%s", result.stderr[-2000:])
    except Exception:
        logger.exception("[scheduler] 💥 Error during registry sync")


def _invoke_post_sync_hook() -> None:
    """Call the registered post-sync hook (if any) to refresh in-memory models."""
    hook = _post_sync_hook
    if hook is None:
        return
    try:
        logger.info(
            "[scheduler] 🔃 Invoking post-sync hook to refresh in-memory models …"
        )
        hook()
        logger.info("[scheduler] ✅ Post-sync hook completed")
    except Exception:
        logger.exception("[scheduler] ⚠️  Post-sync hook raised an exception")


def set_post_sync_hook(hook: Callable[[], None] | None) -> None:
    """
    Register (or clear) a callback that runs after every successful
    registry sync.  The Django bridge uses this to call
    ``refresh_runtime_state`` so newly trained models are available
    immediately without a restart.
    """
    global _post_sync_hook
    _post_sync_hook = hook
    logger.info("[scheduler] Post-sync hook %s", "registered" if hook else "cleared")


def _run_drift_monitor(model_name: str | None = None) -> bool:
    """
    Run drift detection (drift_monitor.py) as a subprocess.

    Checks for data/prediction drift in the specified model (or all models
    if *model_name* is ``None``).  Results are logged to MLflow and persisted
    to the Django ``DriftReport`` table.

    Returns True if the script ran successfully, False otherwise.
    """
    drift_script = _ROOT_DIR / "scripts" / "drift_monitor.py"
    if not drift_script.exists():
        logger.warning("[scheduler] Drift monitor script not found: %s", drift_script)
        return False

    cmd = [sys.executable, "-u", str(drift_script)]
    if model_name:
        cmd.extend(["--model-name", model_name])

    logger.info(
        "[scheduler] 🔍 Running drift check%s …",
        f" for {model_name}" if model_name else "",
    )
    try:
        result = subprocess.run(
            cmd,
            env=_build_child_env(),
            capture_output=True,
            text=True,
            timeout=1800,  # 30-minute timeout for drift checks
        )
        if result.stdout:
            tail = "\n".join(result.stdout.strip().splitlines()[-30:])
            logger.info("[scheduler] drift stdout:\n%s", tail)
        if result.returncode == 0:
            logger.info("[scheduler] ✅ Drift check complete")
            return True
        else:
            logger.error(
                "[scheduler] ❌ Drift check failed (exit code %d)",
                result.returncode,
            )
            if result.stderr:
                logger.error("[scheduler] stderr:\n%s", result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        logger.error("[scheduler] ⏰ Drift check timed out (>30min)")
        return False
    except Exception:
        logger.exception("[scheduler] 💥 Error during drift check")
        return False


# ---------------------------------------------------------------------------
# Job wrapper — one per config entry
# ---------------------------------------------------------------------------


def _training_job(
    model_key: str,
    script_name: str,
    sync_after: bool,
    registered_model_name: str | None = None,
    run_drift_after: bool = True,
) -> None:
    """
    Callable executed by APScheduler for each scheduled training.

    Flow:
      1. Run the training script → registers model as ``challenger``
      2. Run the promotion gate  → promotes to ``champion`` only if better
      3. Run the registry sync   → syncs ``champion`` to Django DB
      4. Run drift check         → compare training data distributions
    """
    logger.info(
        "[scheduler] ━━━ Scheduled training triggered: %s @ %s ━━━",
        model_key,
        datetime.now(dt_timezone.utc).isoformat(),
    )
    success = _run_training_script(script_name, model_key)
    if not success:
        return

    # Promotion gate: compare challenger vs champion metrics
    _run_promotion_gate(model_name=registered_model_name)

    if sync_after:
        _run_registry_sync()

    # Drift check: compare training data distributions
    if run_drift_after:
        _run_drift_monitor(model_name=registered_model_name)


def _drift_job(model_name: str | None = None) -> None:
    """Standalone drift monitoring job — scheduled independently of training."""
    logger.info(
        "[scheduler] ━━━ Scheduled drift check triggered%s @ %s ━━━",
        f": {model_name}" if model_name else " (all models)",
        datetime.now(dt_timezone.utc).isoformat(),
    )
    _run_drift_monitor(model_name=model_name)


# ---------------------------------------------------------------------------
# Schedule builder
# ---------------------------------------------------------------------------

_DAY_MAP = {
    "mon": "mon",
    "tue": "tue",
    "wed": "wed",
    "thu": "thu",
    "fri": "fri",
    "sat": "sat",
    "sun": "sun",
}


def _build_cron_trigger(entry: dict[str, Any], defaults: dict[str, Any]) -> CronTrigger:
    """
    Convert a config entry into an APScheduler CronTrigger.

    Supported frequencies:
      - daily   : runs every day at the given time
      - weekly  : runs on ``day_of_week`` at the given time
      - monthly : runs on ``day_of_month`` at the given time
    """
    time_str = entry.get("time") or defaults.get("time", "23:30")
    tz_str = entry.get("timezone") or defaults.get("timezone", "UTC")
    hour, minute = (int(p) for p in time_str.split(":"))
    freq = entry.get("frequency", "daily").lower()

    kwargs: dict[str, Any] = {
        "hour": hour,
        "minute": minute,
        "timezone": tz_str,
    }

    if freq == "weekly":
        day = entry.get("day_of_week", "sun").lower()
        kwargs["day_of_week"] = _DAY_MAP.get(day, day)
    elif freq == "monthly":
        kwargs["day"] = entry.get("day_of_month", 1)
    # daily → no extra constraints needed

    return CronTrigger(**kwargs)


def _apply_schedule(config: dict[str, Any]) -> None:
    """Remove all existing training jobs and re-add them from *config*."""
    global _scheduler
    if _scheduler is None:
        return

    # Remove old training and drift jobs (identified by prefix)
    for job in _scheduler.get_jobs():
        if job.id.startswith("train:") or job.id.startswith("drift:"):
            _scheduler.remove_job(job.id)

    defaults = config.get("defaults", {})
    sync_after = config.get("post_training", {}).get("sync_registry", True)

    for model_key, entry in config.get("scripts", {}).items():
        if not entry.get("enabled", True):
            logger.info("[scheduler] Skipping disabled model: %s", model_key)
            continue

        script_name = entry.get("script")
        if not script_name:
            logger.warning(
                "[scheduler] No 'script' field for model %s — skipping",
                model_key,
            )
            continue

        trigger = _build_cron_trigger(entry, defaults)
        job_id = f"train:{model_key}"

        _scheduler.add_job(
            _training_job,
            trigger=trigger,
            id=job_id,
            name=f"Train {model_key}",
            kwargs={
                "model_key": model_key,
                "script_name": script_name,
                "sync_after": sync_after,
                "registered_model_name": entry.get("registered_model_name"),
                "run_drift_after": config.get("drift_monitoring", {}).get(
                    "run_after_training", True
                ),
            },
            replace_existing=True,
            misfire_grace_time=3600,  # allow 1-hour misfire window
        )
        logger.info(
            "[scheduler] 📅 Scheduled %s — %s @ %s (%s)",
            model_key,
            entry.get("frequency", "daily"),
            entry.get("time") or defaults.get("time", "23:30"),
            entry.get("description", ""),
        )

    # ── Drift monitoring jobs ─────────────────────────────────────────
    drift_cfg = config.get("drift_monitoring", {})
    if drift_cfg.get("enabled", False):
        drift_defaults = {
            "time": drift_cfg.get("time", "06:00"),
            "timezone": defaults.get("timezone", "UTC"),
            "frequency": drift_cfg.get("frequency", "daily"),
        }
        # Per-model drift checks
        per_model = drift_cfg.get("per_model", {})
        if per_model:
            for model_name, model_drift_entry in per_model.items():
                if not model_drift_entry.get("enabled", True):
                    continue
                merged_entry = {**drift_defaults, **model_drift_entry}
                trigger = _build_cron_trigger(merged_entry, defaults)
                job_id = f"drift:{model_name}"
                _scheduler.add_job(
                    _drift_job,
                    trigger=trigger,
                    id=job_id,
                    name=f"Drift check {model_name}",
                    kwargs={"model_name": model_name},
                    replace_existing=True,
                    misfire_grace_time=3600,
                )
                logger.info(
                    "[scheduler] 🔍 Scheduled drift check: %s — %s @ %s",
                    model_name,
                    merged_entry.get("frequency", "daily"),
                    merged_entry.get("time", drift_defaults["time"]),
                )
        else:
            # Single combined drift job for all models
            trigger = _build_cron_trigger(drift_defaults, defaults)
            _scheduler.add_job(
                _drift_job,
                trigger=trigger,
                id="drift:all_models",
                name="Drift check (all models)",
                kwargs={"model_name": None},
                replace_existing=True,
                misfire_grace_time=3600,
            )
            logger.info(
                "[scheduler] 🔍 Scheduled drift check (all models) — %s @ %s",
                drift_defaults.get("frequency", "daily"),
                drift_defaults.get("time", "06:00"),
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_scheduler(
    config_path: Path | None = None,
    post_sync_hook: Callable[[], None] | None = None,
) -> BackgroundScheduler:
    """
    Start the background training scheduler.
    Safe to call multiple times — only the first call creates the scheduler.
    Returns the scheduler instance.
    """
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            logger.info("[scheduler] Already running — skipping start")
            return _scheduler

        if post_sync_hook is not None:
            set_post_sync_hook(post_sync_hook)

        config = load_schedule_config(config_path)
        _scheduler = BackgroundScheduler(
            job_defaults={
                "coalesce": True,  # collapse missed runs into one
                "max_instances": 1,  # never overlap the same job
            },
        )
        _apply_schedule(config)
        _scheduler.start()
        logger.info(
            "[scheduler] 🚀 Training scheduler started with %d job(s)",
            len(_scheduler.get_jobs()),
        )
        return _scheduler


def stop_scheduler() -> None:
    """Gracefully shut down the scheduler."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)
            logger.info("[scheduler] 🛑 Training scheduler stopped")
            _scheduler = None


def get_schedule_status() -> list[dict[str, Any]]:
    """Return a summary of all scheduled training jobs (for health/debug)."""
    if _scheduler is None:
        return []
    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append(
            {
                "id": job.id,
                "name": job.name,
                "next_run": (
                    job.next_run_time.isoformat() if job.next_run_time else None
                ),
                "trigger": str(job.trigger),
            }
        )
    return jobs


def run_training_now(
    model_key: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """
    Manually trigger training immediately (outside the schedule).

    If *model_key* is ``None``, runs **all** enabled scripts.
    Returns a summary dict.
    """
    config = load_schedule_config(config_path)
    sync_after = config.get("post_training", {}).get("sync_registry", True)
    results: dict[str, Any] = {}

    scripts_to_run: dict[str, Any] = {}
    if model_key:
        entry = config.get("scripts", {}).get(model_key)
        if entry and entry.get("enabled", True):
            scripts_to_run[model_key] = entry
        else:
            return {"error": f"Model key '{model_key}' not found or disabled"}
    else:
        for key, entry in config.get("scripts", {}).items():
            if entry.get("enabled", True):
                scripts_to_run[key] = entry

    for key, entry in scripts_to_run.items():
        script_name = entry.get("script")
        if script_name:
            success = _run_training_script(script_name, key)
            results[key] = "success" if success else "failed"
            # Run promotion gate for successfully trained models
            if success:
                model_name = entry.get("registered_model_name")
                promoted = _run_promotion_gate(model_name=model_name)
                results[f"{key}_promotion"] = "done" if promoted else "failed"

    if sync_after and any(v == "success" for v in results.values()):
        _run_registry_sync()
        results["registry_sync"] = "done"

    return results


def run_drift_now(
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Manually trigger drift detection immediately (outside the schedule).

    If *model_name* is ``None``, runs drift check for **all** configured models.
    Returns a summary dict.
    """
    logger.info(
        "[scheduler] Manual drift check triggered%s",
        f" for {model_name}" if model_name else " (all models)",
    )
    success = _run_drift_monitor(model_name=model_name)
    return {
        "model_name": model_name or "all",
        "action": "drift_check",
        "success": success,
    }
