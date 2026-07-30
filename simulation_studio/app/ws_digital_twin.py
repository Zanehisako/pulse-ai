"""
ws_digital_twin.py - Django-backed continuous digital-twin simulation runner.

The studio still uses the existing SimPy simulator for live stepping, but the
digital twin run, snapshots, frames, injected events, actions, branches,
recommendations, and promotions are persisted through the Django digital_twin
app whenever that backend is importable in local development.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .standalone_predictions_config import (
    standalone_predictions_enabled,
    standalone_predictions_status,
)
from .standalone_snapshot import capture_standalone_operational_snapshot
from .forecast_ws import (
    build_forecast_panel,
    resolve_forecast_job_id,
    valid_forecast_job_id,
)
from .twin_forecast_config import forecast_panel_meta

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .simulator_service import (
    discover_dreamerv3_runs,
    public_dreamerv3_run_payload,
)
from .ws_simulation import (
    SIMULATOR_ROOT,
    _center_snapshot,
    _send_json,
    build_init_message,
    build_snapshot,
    get_scenarios_and_strategies,
    initialize_simulation,
)

if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))

from core import BLOOD_TYPES, BloodUnit  # noqa: E402
from engine import step_simulation  # noqa: E402

logger = logging.getLogger("ws_digital_twin")

router = APIRouter(prefix="/api/twin", tags=["digital-twin"])

STUDIO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDIO_ROOT.parent
BACKEND_ROOT = REPO_ROOT / "backendMulti"
DEFAULT_DIGITAL_TWIN_CONFIG = BACKEND_ROOT / "ml" / "config" / "digital_twin.json"

_DJANGO_READY = False
_DJANGO_ERROR: Exception | None = None


def _ensure_django() -> bool:
    global _DJANGO_READY, _DJANGO_ERROR
    if _DJANGO_READY:
        return True
    if _DJANGO_ERROR is not None:
        return False
    try:
        if str(BACKEND_ROOT) not in sys.path:
            sys.path.insert(0, str(BACKEND_ROOT))
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
        import django

        django.setup()
        _DJANGO_READY = True
        return True
    except Exception as exc:  # pragma: no cover - local fallback path
        _DJANGO_ERROR = exc
        logger.warning("Django digital twin backend unavailable: %s", exc)
        return False


def _fallback_public_config() -> dict[str, Any]:
    payload = json.loads(DEFAULT_DIGITAL_TWIN_CONFIG.read_text(encoding="utf-8"))
    return {
        "version": payload.get("version"),
        "runtime": payload.get("runtime", {}),
        "sync": payload.get("sync", {}),
        "events": payload.get("events", []),
        "actions": payload.get("actions", []),
        "branches": payload.get("branches", {}),
        "promotion_targets": {
            "enabled": bool(payload.get("promotion_targets", {}).get("enabled", False))
        },
    }


def _public_config() -> dict[str, Any]:
    if _ensure_django():
        from digital_twin.config import public_config_payload

        return public_config_payload()
    return _fallback_public_config()


def _studio_twin_config(public_cfg: dict[str, Any]) -> dict[str, Any]:
    runtime = public_cfg.get("runtime", {})
    events = []
    for row in public_cfg.get("events", []):
        key = row.get("key") or row.get("id")
        if not key:
            continue
        events.append(
            {
                "key": key,
                "label": row.get("label") or key,
                "description": row.get("description", ""),
                "default_severity": row.get("default_severity", 0.5),
                "duration_hours": row.get("duration_hours", 0),
                "effects": row.get("effects", {}),
            }
        )
    actions = []
    for row in public_cfg.get("actions", []):
        key = row.get("key") or row.get("id")
        if not key:
            continue
        actions.append(
            {
                "key": key,
                "label": row.get("label") or key,
                "description": row.get("description", ""),
                "cost": row.get("cost", 0),
                "promotion_eligible": bool(row.get("promotion_eligible", False)),
                "effects": row.get("effects", {}),
            }
        )
    return {
        "version": public_cfg.get("version"),
        "tick_interval_s": {
            "default": runtime.get("default_tick_interval_s", 2.0),
            "min": runtime.get("min_tick_interval_s", 0.25),
            "max": runtime.get("max_tick_interval_s", 30.0),
        },
        "step_hours": {
            "default": runtime.get("default_step_hours", 6.0),
            "options": runtime.get("step_hour_options", [1.0, 3.0, 6.0, 12.0, 24.0]),
        },
        "budget_cycle_options": runtime.get("budget_cycle_options", []),
        "default_budget_cycle_hours": runtime.get("default_budget_cycle_hours", 168.0),
        "injectable_events": events,
        "actions": actions,
        "branches": public_cfg.get("branches", {}),
        "promotion_targets": public_cfg.get("promotion_targets", {}),
        "sync": public_cfg.get("sync", {}),
        "dynamic_world": {
            "enabled_by_default": bool(public_cfg.get("sync", {}).get("enabled_by_default", True)),
            "location": runtime.get("location", ""),
            "description": "Persisted digital twin with operational snapshots and audited what-if controls.",
        },
    }


def _template_by_key(public_cfg: dict[str, Any], section: str, key: str) -> dict[str, Any] | None:
    for row in public_cfg.get(section, []):
        if (row.get("key") or row.get("id")) == key:
            return dict(row)
    return None


def _twin_run_id_is_persisted(run_id: str | None) -> bool:
    """Return True when run_id matches Django TwinRun UUID primary keys."""
    if not run_id:
        return False
    try:
        uuid.UUID(str(run_id))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _django_twin_db_enabled(
    run_id: str | None,
    *,
    django_persisted: bool,
) -> bool:
    return (
        django_persisted
        and _twin_run_id_is_persisted(run_id)
        and _ensure_django()
    )


def _django_live_snapshot(*, source: str = "live") -> dict[str, Any] | None:
    """Read predictions/model rows from Django (same path as digital_twin.services)."""
    if not _ensure_django():
        return None
    try:
        from digital_twin.services import capture_operational_rows

        rows = capture_operational_rows(source=source)
        captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        return {
            "snapshot_id": f"django-live-{uuid.uuid4().hex[:12]}",
            "run_id": None,
            "source": source,
            "captured_at": captured_at,
            "source_watermarks": rows.watermarks,
            "payload": rows.redacted,
            "freshness": rows.freshness,
            "warnings": list(rows.warnings or []),
        }
    except Exception as exc:
        logger.warning("Django operational snapshot unavailable: %s", exc)
        return None


def _studio_predictions_for_ui() -> bool:
    if not standalone_predictions_enabled():
        return False
    try:
        from .standalone_predictions_config import load_standalone_predictions_config

        cfg = load_standalone_predictions_config()
        studio = cfg.get("studio_twin") or {}
        return bool(studio.get("always_run_predictions_for_ui", True))
    except Exception:
        return standalone_predictions_enabled()


def _use_standalone_snapshots() -> bool:
    """In-memory ML world only when Django is not importable."""
    if not standalone_predictions_enabled():
        return False
    return not _ensure_django()


def _ui_operational_snapshot(*, advance_tick: bool = False) -> dict[str, Any] | None:
    live = _django_live_snapshot()
    if live is not None:
        return live
    if not _studio_predictions_for_ui():
        return None
    return capture_standalone_operational_snapshot(
        run_predictions=True,
        advance_tick=advance_tick,
    )


def _create_standalone_run(start_msg: dict[str, Any]) -> dict[str, Any]:
    snapshot = capture_standalone_operational_snapshot(
        run_predictions=True,
        advance_tick=False,
    )
    run_meta: dict[str, Any] = {
        "mode": "simulation_studio",
        "source_mode": "standalone_live",
        "django_persisted": False,
    }
    try:
        from .standalone_predictions_config import load_standalone_predictions_config

        studio = load_standalone_predictions_config().get("studio_twin") or {}
        persist_run = bool(studio.get("persist_twin_run_to_django", False))
    except Exception:
        persist_run = False

    if persist_run and _ensure_django():
        try:
            from digital_twin.services import create_twin_run, run_payload

            run = create_twin_run(
                {
                    "mode": "simulation_studio",
                    "source_mode": "standalone_live",
                    "scenario_key": str(start_msg.get("scenario_key") or "baseline"),
                    "strategy_key": str(start_msg.get("strategy_key") or "baseline"),
                    "seed": int(start_msg.get("seed") or 100),
                }
            )
            run_meta = run_payload(run)
            run_meta["source_mode"] = "standalone_live"
            run_meta["django_persisted"] = True
        except Exception as exc:
            logger.warning("Standalone twin run DB create failed, using ephemeral id: %s", exc)
            run_meta["run_id"] = str(uuid.uuid4())
    else:
        run_meta["run_id"] = str(uuid.uuid4())

    return {"run": run_meta, "snapshot": snapshot}


def _create_persisted_run(start_msg: dict[str, Any]) -> dict[str, Any] | None:
    if _ensure_django():
        try:
            from digital_twin.services import (
                capture_snapshot,
                create_twin_run,
                run_payload,
                snapshot_payload,
            )

            source_mode = "live" if bool(start_msg.get("enable_live_data")) else "live"
            run = create_twin_run(
                {
                    "mode": "simulation_studio",
                    "source_mode": source_mode,
                    "scenario_key": str(start_msg.get("scenario_key") or "baseline"),
                    "strategy_key": str(start_msg.get("strategy_key") or "baseline"),
                    "seed": int(start_msg.get("seed") or 100),
                }
            )
            snapshot = capture_snapshot(run, actor="simulation_studio")
            run_meta = run_payload(run)
            run_meta["django_persisted"] = True
            ui_snapshot = _django_live_snapshot(source=source_mode) or snapshot_payload(
                snapshot
            )
            return {"run": run_meta, "snapshot": ui_snapshot}
        except Exception as exc:
            logger.warning("Digital twin persistence unavailable for run start: %s", exc)
    if standalone_predictions_enabled():
        return _create_standalone_run(start_msg)
    return None


def _capture_snapshot(
    run_id: str,
    source: str | None = None,
    *,
    django_persisted: bool = False,
) -> dict[str, Any] | None:
    if not run_id:
        return None
    live = _django_live_snapshot(source=source or "live")
    if live is not None:
        return live
    if not _django_twin_db_enabled(run_id, django_persisted=django_persisted):
        return None
    try:
        from digital_twin.models import TwinRun
        from digital_twin.services import capture_snapshot, snapshot_payload

        run = TwinRun.objects.get(run_id=run_id)
        snapshot = capture_snapshot(run, source=source, actor="simulation_studio")
        return snapshot_payload(snapshot)
    except Exception as exc:
        logger.warning("Digital twin snapshot persistence unavailable: %s", exc)
        return None


def _persist_frame(
    run_id: str | None,
    *,
    django_persisted: bool = False,
    tick_number: int,
    simulated_hour: float,
    simulated_state: dict[str, Any],
    active_events: list[dict[str, Any]],
    active_actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _django_twin_db_enabled(run_id, django_persisted=django_persisted):
        return None
    try:
        from digital_twin.models import TwinRun
        from digital_twin.services import create_frame, frame_payload

        run = TwinRun.objects.get(run_id=run_id)
        frame = create_frame(
            run,
            tick_number=tick_number,
            simulated_hour=simulated_hour,
            simulated_state=simulated_state,
            active_events=active_events,
            active_actions=active_actions,
            actor="simulation_studio",
        )
        return frame_payload(frame)
    except Exception as exc:
        logger.warning("Digital twin frame persistence unavailable: %s", exc)
        return None


def _persist_event(
    run_id: str | None,
    *,
    django_persisted: bool = False,
    event_key: str,
    severity: float,
    start_hour: float,
) -> dict[str, Any] | None:
    if not _django_twin_db_enabled(run_id, django_persisted=django_persisted):
        return None
    try:
        from digital_twin.models import TwinRun
        from digital_twin.services import event_payload, inject_event

        run = TwinRun.objects.get(run_id=run_id)
        event = inject_event(
            run,
            event_key=event_key,
            severity=severity,
            start_hour=start_hour,
            source="simulation_studio",
            actor="simulation_studio",
        )
        return event_payload(event)
    except Exception as exc:
        logger.warning("Digital twin event persistence unavailable: %s", exc)
        return None


def _persist_action(
    run_id: str | None,
    *,
    django_persisted: bool = False,
    action_key: str,
    simulated_hour: float,
) -> dict[str, Any] | None:
    if not _django_twin_db_enabled(run_id, django_persisted=django_persisted):
        return None
    try:
        from digital_twin.models import TwinRun
        from digital_twin.services import action_payload, execute_action

        run = TwinRun.objects.get(run_id=run_id)
        action = execute_action(
            run,
            action_key=action_key,
            simulated_hour=simulated_hour,
            source="simulation_studio",
            actor="simulation_studio",
        )
        return action_payload(action)
    except Exception as exc:
        logger.warning("Digital twin action persistence unavailable: %s", exc)
        return None


def _run_branch(
    run_id: str | None,
    msg: dict[str, Any],
    *,
    django_persisted: bool = False,
) -> dict[str, Any] | None:
    if not _django_twin_db_enabled(run_id, django_persisted=django_persisted):
        return None
    try:
        from digital_twin.models import TwinRun
        from digital_twin.services import branch_payload, run_branch

        run = TwinRun.objects.get(run_id=run_id)
        branch = run_branch(
            run,
            branch_key=str(msg.get("branch_key") or "studio-branch"),
            policy_overrides=dict(msg.get("policy_overrides") or {}),
            action_overrides=list(msg.get("action_overrides") or []),
            event_overrides=list(msg.get("event_overrides") or []),
            actor="simulation_studio",
        )
        return branch_payload(branch)
    except Exception as exc:
        logger.warning("Digital twin branch persistence unavailable: %s", exc)
        return None


def _recommendation(
    run_id: str | None,
    *,
    django_persisted: bool = False,
) -> dict[str, Any] | None:
    if not _django_twin_db_enabled(run_id, django_persisted=django_persisted):
        return None
    try:
        from digital_twin.models import TwinRun
        from digital_twin.services import recommend_action, recommendation_payload

        run = TwinRun.objects.get(run_id=run_id)
        recommendation = recommend_action(run, actor="simulation_studio")
        return recommendation_payload(recommendation)
    except Exception as exc:
        logger.warning("Digital twin recommendation persistence unavailable: %s", exc)
        return None


def _promote(action_id: str | None) -> dict[str, Any] | None:
    if not action_id or not _ensure_django():
        return None
    from digital_twin.models import TwinAction
    from digital_twin.services import action_payload, promote_action

    action = TwinAction.objects.get(action_id=action_id)
    promoted = promote_action(action, actor="simulation_studio")
    return action_payload(promoted)


def _stop_persisted_run(
    run_id: str | None,
    *,
    django_persisted: bool = False,
) -> None:
    if not _django_twin_db_enabled(run_id, django_persisted=django_persisted):
        return
    try:
        from digital_twin.models import TwinRun
        from digital_twin.services import stop_run

        stop_run(TwinRun.objects.get(run_id=run_id), actor="simulation_studio")
    except Exception as exc:
        logger.warning("Digital twin stop persistence unavailable: %s", exc)


def _source_status(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not snapshot:
        return {
            "source_mode": "synthetic",
            "status": "degraded",
            "snapshot_id": None,
            "freshness": {},
            "real_signals": {},
        }
    payload = snapshot.get("payload") or {}
    freshness = snapshot.get("freshness") or {}
    source = snapshot.get("source") or "synthetic"
    if source in {"synthetic", "standalone"}:
        status = "synthetic" if source == "synthetic" else "standalone_live"
    elif freshness.get("is_stale"):
        status = "stale_live"
    elif freshness.get("status") == "fresh":
        status = "fresh_live"
    else:
        status = "degraded"
    return {
        "source_mode": source,
        "status": status,
        "snapshot_id": snapshot.get("snapshot_id"),
        "freshness": freshness,
        "watermarks": snapshot.get("source_watermarks", {}),
        "warnings": snapshot.get("warnings", []),
        "real_signals": {
            "alerts": len(payload.get("alerts", [])),
            "predictions": len(payload.get("predictions", [])),
            "drift_reports": len(payload.get("drift_reports", [])),
            "model_configs": len(payload.get("model_configs", [])),
            "blood_supplies": len(payload.get("blood_supplies", [])),
            "donors": len(payload.get("donors", [])),
        },
    }


def _blend_factor(value: Any, severity: float = 1.0) -> float:
    numeric = float(value)
    bounded = max(0.0, min(float(severity), 1.0))
    if numeric >= 1.0:
        return 1.0 + ((numeric - 1.0) * bounded)
    return 1.0 - ((1.0 - numeric) * bounded)


def _apply_effects_to_state(state: Any, effects: dict[str, Any], *, severity: float = 1.0) -> None:
    params = getattr(state, "params", None)
    if params is None:
        return
    for field, raw_value in effects.items():
        if not hasattr(params, field):
            continue
        try:
            current = float(getattr(params, field))
            setattr(params, field, current * _blend_factor(raw_value, severity))
        except (TypeError, ValueError):
            continue


def _apply_snapshot_inventory(state: Any, snapshot: dict[str, Any] | None) -> None:
    payload = (snapshot or {}).get("payload") or {}
    rows = payload.get("blood_supplies") or []
    if not rows:
        return

    units_by_name: dict[str, float] = {}
    for row in rows:
        key = str(row.get("hospital_name") or row.get("hospital_id") or "").strip().lower()
        if not key:
            continue
        units_by_name[key] = units_by_name.get(key, 0.0) + float(row.get("current_stock_units") or 0.0)
    if not units_by_name:
        return

    city_total = sum(units_by_name.values())
    if city_total <= 0:
        return

    centers = list(getattr(state, "centers", []) or [])
    hospitals = [center for center in centers if getattr(center, "ctype", "") == "hospital"]
    target_centers = hospitals or centers
    if not target_centers:
        return

    per_center_total = max(round(city_total / len(target_centers)), 1)
    blood_type = BLOOD_TYPES[0] if BLOOD_TYPES else "O+"
    for center in target_centers:
        existing = center.inventory_by_component()
        existing_total = max(sum(existing.values()), 1)
        center.inventory = []
        center._inventory_counts = {component: 0 for component in existing}
        for component, current_count in existing.items():
            units = max(round(per_center_total * current_count / existing_total), 0)
            for _index in range(units):
                center.add_unit(BloodUnit(state.env, blood_type, component, center.name))


def _simulated_state_from_step(
    payload: dict[str, Any],
    active_events: list[dict[str, Any]],
    active_actions: list[dict[str, Any]],
    latest_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inventory_by_hospital: dict[str, dict[str, float]] = {}
    total_inventory = 0.0
    donor_count = 0
    for center in payload.get("centers", []):
        inventory = center.get("inventory") or {}
        name = str(center.get("name") or "unknown")
        inventory_by_hospital[name] = {
            str(component): float(units or 0) for component, units in inventory.items()
        }
        total_inventory += sum(inventory_by_hospital[name].values())
    for row in payload.get("donor_count_by_center", []):
        donor_count += int(row.get("active_donors") or 0)

    snap_payload = (latest_snapshot or {}).get("payload") or {}
    predictions = list(snap_payload.get("predictions") or [])
    drift_reports = list(snap_payload.get("drift_reports") or [])
    alerts = list(snap_payload.get("alerts") or [])
    model_configs = list(snap_payload.get("model_configs") or [])

    return {
        "simulated_hour": payload.get("hour", 0.0),
        "inventory_by_hospital": inventory_by_hospital,
        "total_inventory_units": round(total_inventory, 2),
        "donor_count": donor_count,
        "active_alert_count": len(active_events) + int(payload.get("recent_shortages", 0) > 0),
        "latest_prediction_count": len(predictions),
        "critical_drift_count": len(
            [
                row
                for row in drift_reports
                if str(row.get("severity", "")).lower() == "critical"
            ]
        ),
        "active_events": active_events,
        "active_actions": active_actions,
        "real_signals": {
            "alerts": alerts[:25],
            "predictions": predictions[:50],
            "drift_reports": drift_reports[:25],
            "model_configs": model_configs[:50],
        },
        "simulator_metrics": {
            "shortage_rate": payload.get("shortage_rate", 0),
            "total_shortage": payload.get("total_shortage", 0),
            "total_donated": payload.get("total_donated", 0),
            "total_transfused": payload.get("total_transfused", 0),
            "total_expired": payload.get("total_expired", 0),
            "budget_remaining": payload.get("budget_remaining", 0),
        },
    }


def _build_twin_init_message(
    state: Any,
    scenario_key: str,
    strategy_key: str,
    seed: int,
    *,
    run_payload: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
    twin_config: dict[str, Any],
    tick_interval_s: float,
    step_hours: float,
    budget_cycle_hours: float,
    enable_live_data: bool,
) -> dict[str, Any]:
    payload = build_init_message(state, scenario_key, strategy_key, seed)
    payload["type"] = "twin_init"
    payload["mode"] = "digital_twin"
    payload["run_id"] = (run_payload or {}).get("run_id")
    payload["config_version"] = twin_config.get("version")
    payload["snapshot"] = snapshot
    payload["source_status"] = _source_status(snapshot)
    payload["tick_interval_s"] = tick_interval_s
    payload["step_hours"] = step_hours
    payload["budget_cycle_hours"] = budget_cycle_hours
    payload["enable_live_data"] = enable_live_data
    payload["injectable_events"] = twin_config.get("injectable_events", [])
    payload["actions"] = twin_config.get("actions", [])
    payload["forecast_panel"] = forecast_panel_meta()
    return payload


def _build_twin_tick(
    state: Any,
    tick_number: int,
    *,
    prev_donated: int,
    prev_shortage: int,
    cycle_number: int,
    cycle_hour: float,
    budget_cycle_hours: float,
    active_events: list[dict[str, Any]],
    active_actions: list[dict[str, Any]],
    run_id: str | None,
    config_version: str | None,
    latest_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = build_snapshot(
        state,
        tick_number,
        prev_donated=prev_donated,
        prev_shortage=prev_shortage,
    )
    payload["type"] = "twin_tick"
    payload["tick"] = tick_number
    payload["run_id"] = run_id
    payload["config_version"] = config_version
    payload["cycle_number"] = cycle_number
    payload["cycle_hour"] = round(cycle_hour, 2)
    payload["budget_cycle_progress"] = round(
        (cycle_hour / budget_cycle_hours * 100.0) if budget_cycle_hours > 0 else 0.0,
        2,
    )
    payload["active_events"] = active_events
    payload["active_actions"] = active_actions
    payload["simulated_state"] = _simulated_state_from_step(
        payload,
        active_events,
        active_actions,
        latest_snapshot,
    )
    payload["source_status"] = _source_status(latest_snapshot)
    payload["data_sources_status"] = payload["source_status"]
    payload["snapshot"] = latest_snapshot
    return payload


@router.get("/predictions-preview")
def get_predictions_preview() -> dict[str, Any]:
    """Lightweight check: scheduled ML jobs + model catalog for the studio UI."""
    snapshot = _ui_operational_snapshot(advance_tick=False)
    if snapshot is None:
        _enabled, status_warnings = standalone_predictions_status()
        warnings = list(status_warnings)
        if not warnings:
            warnings = ["Standalone predictions are disabled."]
        return {
            "enabled": False,
            "predictions": [],
            "model_configs": [],
            "warnings": warnings,
        }
    payload = snapshot.get("payload") or {}
    return {
        "enabled": True,
        "snapshot_id": snapshot.get("snapshot_id"),
        "source": snapshot.get("source"),
        "prediction_count": len(payload.get("predictions") or []),
        "model_config_count": len(payload.get("model_configs") or []),
        "predictions": (payload.get("predictions") or [])[:20],
        "model_configs": (payload.get("model_configs") or [])[:20],
        "warnings": list(snapshot.get("warnings") or []),
    }


@router.get("/setup")
def get_twin_setup_data() -> dict[str, Any]:
    base = get_scenarios_and_strategies()
    dreamerv3_runs = discover_dreamerv3_runs()
    public_cfg = _public_config()
    return {
        **base,
        "dreamerv3_runs": [public_dreamerv3_run_payload(r) for r in dreamerv3_runs],
        "default_dreamerv3_run_key": (
            dreamerv3_runs[0]["key"] if dreamerv3_runs else None
        ),
        "twin_config": _studio_twin_config(public_cfg),
        "forecast_panel": forecast_panel_meta(),
    }


@router.websocket("/ws")
async def twin_websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    public_cfg = _public_config()
    twin_config = _studio_twin_config(public_cfg)

    pause_event = asyncio.Event()
    pause_event.set()

    stop_requested = False
    speed: float = 1.0
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    action_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    run_id: str | None = None
    django_persisted: bool = False
    latest_snapshot: dict[str, Any] | None = None
    forecast_job_id: str | None = None
    forecast_tick_state: dict[str, Any] = {
        "tick_number": 0,
        "simulated_hour": 0.0,
        "simulated_state": {},
        "centers": [],
    }

    async def _listen() -> dict[str, Any] | None:
        try:
            return await websocket.receive_json()
        except (WebSocketDisconnect, RuntimeError):
            return None

    async def _command_listener() -> None:
        nonlocal stop_requested, speed, latest_snapshot, django_persisted, forecast_job_id

        while True:
            msg = await _listen()
            if msg is None:
                stop_requested = True
                pause_event.set()
                return

            command = msg.get("command")

            if command == "pause":
                pause_event.clear()
                await _send_json(websocket, {"type": "paused"})
            elif command == "resume":
                pause_event.set()
                await _send_json(websocket, {"type": "resumed"})
            elif command == "stop":
                stop_requested = True
                pause_event.set()
                await asyncio.to_thread(
                    _stop_persisted_run,
                    run_id,
                    django_persisted=django_persisted,
                )
                await _send_json(websocket, {"type": "stopped"})
                return
            elif command == "set_speed":
                try:
                    speed = max(0.1, float(msg.get("speed", 1.0)))
                except (TypeError, ValueError):
                    speed = 1.0
            elif command == "refresh_snapshot":
                latest_snapshot = await asyncio.to_thread(
                    _capture_snapshot,
                    run_id or "",
                    msg.get("source"),
                    django_persisted=django_persisted,
                )
                await _send_json(
                    websocket,
                    {
                        "type": "data_status",
                        "snapshot": latest_snapshot,
                        "data_sources_status": _source_status(latest_snapshot),
                    },
                )
            elif command == "inject_event":
                event_key = str(msg.get("event_key") or "").strip()
                if not event_key:
                    await _send_json(websocket, {"type": "error", "message": "inject_event requires event_key."})
                    continue
                await event_queue.put(
                    {
                        "event_key": event_key,
                        "severity": float(msg.get("severity", 0.5)),
                    }
                )
            elif command == "execute_action":
                action_key = str(msg.get("action_key") or "").strip()
                if not action_key:
                    await _send_json(websocket, {"type": "error", "message": "execute_action requires action_key."})
                    continue
                await action_queue.put({"action_key": action_key})
            elif command == "run_branch":
                branch = await asyncio.to_thread(
                    _run_branch,
                    run_id,
                    msg,
                    django_persisted=django_persisted,
                )
                await _send_json(websocket, {"type": "branch_result", "branch": branch})
            elif command == "request_recommendation":
                recommendation = await asyncio.to_thread(
                    _recommendation,
                    run_id,
                    django_persisted=django_persisted,
                )
                await _send_json(
                    websocket,
                    {"type": "recommendation_result", "recommendation": recommendation},
                )
            elif command == "promote_action":
                try:
                    promotion = await asyncio.to_thread(_promote, msg.get("action_id"))
                    await _send_json(websocket, {"type": "promotion_result", "action": promotion})
                except Exception as exc:
                    await _send_json(websocket, {"type": "error", "message": str(exc)})
            elif command == "set_forecast_model":
                candidate = valid_forecast_job_id(str(msg.get("job_id") or ""))
                if not candidate:
                    await _send_json(
                        websocket,
                        {"type": "error", "message": "Invalid or disabled forecast job_id."},
                    )
                    continue
                forecast_job_id = candidate
                panel = await asyncio.to_thread(
                    build_forecast_panel,
                    forecast_job_id,
                    tick_number=int(forecast_tick_state.get("tick_number") or 0),
                    simulated_hour=float(forecast_tick_state.get("simulated_hour") or 0),
                    simulated_state=dict(forecast_tick_state.get("simulated_state") or {}),
                    snapshot=latest_snapshot,
                    centers=list(forecast_tick_state.get("centers") or []),
                )
                await _send_json(
                    websocket,
                    {"type": "forecast_panel_updated", "forecast_panel": panel},
                )

    try:
        start_msg: dict[str, Any] | None = None
        while True:
            msg = await _listen()
            if msg is None:
                return
            if msg.get("command") == "start":
                start_msg = msg
                break
            await _send_json(websocket, {"type": "error", "message": "Send a start command to begin the digital twin."})

        scenario_key = str(start_msg.get("scenario_key", "baseline"))
        strategy_key = str(start_msg.get("strategy_key", "baseline"))
        seed = int(start_msg.get("seed", 100))
        tick_cfg = twin_config.get("tick_interval_s", {})
        tick_interval_s = max(
            float(tick_cfg.get("min", 0.25)),
            min(float(start_msg.get("tick_interval_s", tick_cfg.get("default", 2.0))), float(tick_cfg.get("max", 30.0))),
        )
        step_hours = float(start_msg.get("step_hours", twin_config.get("step_hours", {}).get("default", 6.0)))
        budget_cycle_hours = float(
            start_msg.get("budget_cycle_hours", twin_config.get("default_budget_cycle_hours", 168.0))
        )
        dreamerv3_run_key = start_msg.get("dreamerv3_run_key")
        enable_live_data = bool(start_msg.get("enable_live_data", False))
        speed = max(0.1, float(start_msg.get("speed", 1.0)))

        persisted = await asyncio.to_thread(_create_persisted_run, start_msg)
        if persisted is None and standalone_predictions_enabled():
            persisted = await asyncio.to_thread(_create_standalone_run, start_msg)
        run_payload = (persisted or {}).get("run")
        latest_snapshot = (persisted or {}).get("snapshot")
        run_id = (run_payload or {}).get("run_id")
        django_persisted = bool((run_payload or {}).get("django_persisted", False))
        forecast_job_id = resolve_forecast_job_id(start_msg)

        try:
            state = await asyncio.to_thread(
                initialize_simulation,
                scenario_key,
                strategy_key,
                seed,
                None,
                step_hours,
                dreamerv3_run_key,
            )
            state.params.sim_hours = 10_000_000
            _apply_snapshot_inventory(state, latest_snapshot)
        except Exception as exc:
            logger.exception("Digital twin initialisation failed")
            await _send_json(websocket, {"type": "error", "message": f"Init failed: {exc}"})
            return

        init_payload = _build_twin_init_message(
            state,
            scenario_key,
            strategy_key,
            seed,
            run_payload=run_payload,
            snapshot=latest_snapshot,
            twin_config=twin_config,
            tick_interval_s=tick_interval_s,
            step_hours=step_hours,
            budget_cycle_hours=budget_cycle_hours,
            enable_live_data=enable_live_data,
        )
        init_payload["forecast_job_id"] = forecast_job_id
        init_centers = [_center_snapshot(center) for center in state.centers]
        forecast_tick_state["centers"] = init_centers
        forecast_tick_state["simulated_hour"] = float(state.env.now)
        if forecast_job_id:
            init_payload["forecast_live"] = await asyncio.to_thread(
                build_forecast_panel,
                forecast_job_id,
                tick_number=0,
                simulated_hour=float(state.env.now),
                simulated_state={},
                snapshot=latest_snapshot,
                centers=init_centers,
            )
        await _send_json(websocket, init_payload)

        listener_task = asyncio.create_task(_command_listener())

        tick_number = 0
        prev_donated = sum(center.stats["donated"] for center in state.centers)
        prev_shortage = state.total_shortage_units
        cycle_number = 1
        cycle_start_hour = state.env.now
        active_events: list[dict[str, Any]] = []
        active_actions: list[dict[str, Any]] = []
        sync_cfg = twin_config.get("sync", {})
        refresh_interval_s = max(float(sync_cfg.get("refresh_interval_seconds", 30)), 1.0)
        next_snapshot_refresh_tick = 1

        try:
            while not stop_requested:
                await pause_event.wait()
                if stop_requested:
                    break

                while not event_queue.empty():
                    event_request = event_queue.get_nowait()
                    event_key = event_request["event_key"]
                    template = _template_by_key(public_cfg, "events", event_key)
                    if template is None:
                        await _send_json(websocket, {"type": "error", "message": f"Unknown injectable event: {event_key}"})
                        continue
                    severity = max(0.0, min(1.0, float(event_request["severity"])))
                    persisted_event = await asyncio.to_thread(
                        _persist_event,
                        run_id,
                        django_persisted=django_persisted,
                        event_key=event_key,
                        severity=severity,
                        start_hour=state.env.now,
                    )
                    _apply_effects_to_state(state, dict(template.get("effects") or {}), severity=severity)
                    event_payload = {
                        "event_id": (persisted_event or {}).get("event_id"),
                        "event_key": event_key,
                        "name": template.get("label") or event_key,
                        "description": template.get("description", ""),
                        "severity": round(severity, 2),
                        "source": "manual",
                        "fired_at_hour": round(state.env.now, 2),
                        "expires_at_hour": round(state.env.now + float(template.get("duration_hours") or 0), 2),
                    }
                    active_events.append(event_payload)
                    await _send_json(websocket, {"type": "twin_event", "event": event_payload})

                while not action_queue.empty():
                    action_request = action_queue.get_nowait()
                    action_key = action_request["action_key"]
                    template = _template_by_key(public_cfg, "actions", action_key)
                    if template is None:
                        await _send_json(websocket, {"type": "error", "message": f"Unknown action: {action_key}"})
                        continue
                    persisted_action = await asyncio.to_thread(
                        _persist_action,
                        run_id,
                        django_persisted=django_persisted,
                        action_key=action_key,
                        simulated_hour=state.env.now,
                    )
                    _apply_effects_to_state(state, dict(template.get("effects") or {}))
                    action_payload = {
                        "action_id": (persisted_action or {}).get("action_id"),
                        "action_key": action_key,
                        "key": action_key,
                        "name": template.get("label") or action_key,
                        "cost": template.get("cost", 0),
                        "promotion_status": (persisted_action or {}).get("promotion_status"),
                        "intensity": 1.0,
                    }
                    active_actions.append(action_payload)
                    await _send_json(
                        websocket,
                        {
                            "type": "action_result",
                            "result": {
                                "success": True,
                                "message": "Action simulated in the digital twin.",
                                **action_payload,
                            },
                        },
                    )

                if run_id and tick_number >= next_snapshot_refresh_tick:
                    latest_snapshot = await asyncio.to_thread(
                        _capture_snapshot,
                        run_id,
                        None,
                        django_persisted=django_persisted,
                    )
                    ticks_per_refresh = max(round(refresh_interval_s / max(tick_interval_s, 0.1)), 1)
                    next_snapshot_refresh_tick = tick_number + ticks_per_refresh

                try:
                    await asyncio.to_thread(step_simulation, state, step_hours)
                except Exception as exc:
                    logger.exception("Digital twin tick failed")
                    await _send_json(websocket, {"type": "error", "message": f"Tick failed: {exc}"})
                    return

                tick_number += 1
                cycle_hour = state.env.now - cycle_start_hour
                if budget_cycle_hours > 0 and cycle_hour >= budget_cycle_hours:
                    cycle_number += 1
                    cycle_start_hour = state.env.now
                    cycle_hour = 0.0
                    await _send_json(
                        websocket,
                        {
                            "type": "budget_refresh",
                            "cycle": {
                                "cycle_number": cycle_number,
                                "hour": round(state.env.now, 2),
                            },
                        },
                    )

                active_events = [
                    event
                    for event in active_events
                    if float(event.get("expires_at_hour") or 0) > state.env.now
                ]

                tick_payload = _build_twin_tick(
                    state,
                    tick_number,
                    prev_donated=prev_donated,
                    prev_shortage=prev_shortage,
                    cycle_number=cycle_number,
                    cycle_hour=cycle_hour,
                    budget_cycle_hours=budget_cycle_hours,
                    active_events=active_events,
                    active_actions=active_actions,
                    run_id=run_id,
                    config_version=twin_config.get("version"),
                    latest_snapshot=latest_snapshot,
                )
                frame = await asyncio.to_thread(
                    _persist_frame,
                    run_id,
                    django_persisted=django_persisted,
                    tick_number=tick_number,
                    simulated_hour=state.env.now,
                    simulated_state=tick_payload["simulated_state"],
                    active_events=active_events,
                    active_actions=active_actions,
                )
                if frame:
                    tick_payload["frame"] = frame
                    tick_payload["divergence"] = frame.get("divergence", {})

                forecast_tick_state["tick_number"] = tick_number
                forecast_tick_state["simulated_hour"] = float(state.env.now)
                forecast_tick_state["simulated_state"] = dict(
                    tick_payload.get("simulated_state") or {}
                )
                forecast_tick_state["centers"] = list(tick_payload.get("centers") or [])
                forecast_panel = await asyncio.to_thread(
                    build_forecast_panel,
                    forecast_job_id,
                    tick_number=tick_number,
                    simulated_hour=float(state.env.now),
                    simulated_state=forecast_tick_state["simulated_state"],
                    snapshot=latest_snapshot,
                    centers=forecast_tick_state["centers"],
                )
                if forecast_panel:
                    tick_payload["forecast_panel"] = forecast_panel

                await _send_json(websocket, tick_payload)

                prev_donated = sum(center.stats["donated"] for center in state.centers)
                prev_shortage = state.total_shortage_units
                await asyncio.sleep(max(0.05, tick_interval_s / speed))
        finally:
            listener_task.cancel()
            try:
                await listener_task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logger.info("Digital twin WebSocket client disconnected.")
    except Exception as exc:
        logger.exception("Unexpected error in twin_websocket_endpoint")
        await _send_json(websocket, {"type": "error", "message": str(exc)})
