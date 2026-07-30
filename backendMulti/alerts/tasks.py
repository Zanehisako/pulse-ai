from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_process_stale_alerts() -> dict[str, int]:
    """
    Periodic job: escalate open/acknowledged alerts that exceed
    their rule's escalation_after_minutes threshold.
    Intended to be invoked by an external scheduler (cron).
    """
    from alerts.services.rule_engine import AlertEngine

    processed = AlertEngine().process_stale_alerts()
    logger.info("process_stale_alerts completed: %d escalated", processed)
    return {"escalated": processed}
