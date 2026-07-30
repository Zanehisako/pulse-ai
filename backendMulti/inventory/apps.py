import logging
import os
import sys
import threading
from pathlib import Path

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class InventoryConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "inventory"

    def ready(self):
        if os.environ.get("DJANGO_SKIP_SIMULATION") == "1":
            return

        argv0 = Path(sys.argv[0]).stem.lower() if sys.argv else ""
        is_daphne = argv0 == "daphne" or any("daphne" in a for a in sys.argv)
        is_runserver = sys.argv[1] == "runserver" if len(sys.argv) > 1 else False

        if not (is_daphne or is_runserver):
            return
        if is_runserver and os.environ.get("RUN_MAIN") != "true":
            return

        threading.Thread(
            target=_start_simulation_scheduler,
            name="pios-simulation-scheduler",
            daemon=True,
        ).start()


def _start_simulation_scheduler() -> None:
    import time

    time.sleep(5)  # let Django finish full startup first
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        from inventory.simulation import (
            get_tick_interval_minutes,
            is_simulation_enabled,
        )

        if not is_simulation_enabled():
            logger.info("Simulation scheduler: disabled via config — not starting")
            return

        interval = get_tick_interval_minutes()

        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(
            _run_prediction_chain,
            trigger="interval",
            minutes=interval,
            id="simulation_tick",
            replace_existing=True,
            misfire_grace_time=60,
        )
        scheduler.start()
        logger.info(
            "Simulation scheduler started — tick every %d minute(s)", interval
        )
    except Exception:
        logger.exception("Failed to start simulation scheduler")


def _run_prediction_chain() -> dict:
    """
    Full periodic chain:
      1. Mutate data (simulation tick)
      2. Run ML predictions
      3. Push results to dashboard via management command

    Returns a summary dict with per-step status so callers (including the
    manual trigger endpoint and the scheduler loop) can detect partial failures
    without relying on exceptions that would otherwise be swallowed.

    Shape::

        {
            "ok": True,           # False if any step errored
            "tick": {"status": "ok", ...} | {"status": "error", "detail": "..."},
            "predictions": {"status": "ok", ...} | {"status": "error", "detail": "..."},
            "dashboard": {"status": "ok"} | {"status": "error", "detail": "..."},
        }
    """
    from django.core import management

    from inventory.simulation import advance_simulation_tick
    from inventory.tasks import run_configured_scheduled_predictions

    summary: dict = {"ok": True}

    # Safety net: ensure alert rules exist before evaluating predictions.
    try:
        from alerts.models import AlertRule

        if not AlertRule.objects.exists():
            management.call_command("generate_alert_rules", verbosity=0)
            logger.info("Safety net: generated alert rules before prediction chain.")
    except Exception as exc:
        logger.debug("Alert rule safety net skipped: %s", exc)

    try:
        tick_result = advance_simulation_tick()
        summary["tick"] = {"status": "ok", **tick_result}
        logger.info("Simulation tick: %s", tick_result)
    except Exception as exc:
        logger.exception("Simulation tick failed")
        summary["tick"] = {"status": "error", "detail": str(exc)}
        summary["ok"] = False

    try:
        pred_result = run_configured_scheduled_predictions()
        summary["predictions"] = {"status": "ok", **pred_result}
        logger.info("Scheduled predictions: %s", pred_result)
    except Exception as exc:
        logger.exception("Scheduled predictions failed")
        summary["predictions"] = {"status": "error", "detail": str(exc)}
        summary["ok"] = False

    try:
        management.call_command("seed_dashboard_from_inventory", "--force")
        summary["dashboard"] = {"status": "ok"}
    except Exception as exc:
        logger.exception("Dashboard sync after prediction tick failed")
        summary["dashboard"] = {"status": "error", "detail": str(exc)}
        summary["ok"] = False

    if not summary["ok"]:
        logger.warning("Prediction chain completed with failures: %s", summary)

    return summary
