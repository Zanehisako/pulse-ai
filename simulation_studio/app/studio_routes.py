from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter

from .operational_seed import build_operational_seed_plan, load_operational_simulation_config
from .schemas import DataSnapshotRequest, StrategyAssistantRequest, StudioExperimentRequest
from .simulator_service import list_scenarios, list_simulation_lab_policies
from .snapshot_layer import capture_data_snapshot, detect_snapshot_backend, get_django_apps
from .studio_service import (
    default_assistant_universes,
    default_experiment_universes,
    run_strategy_assistant,
    run_studio_experiment,
)

router = APIRouter(prefix="/api/studio", tags=["simulation-lab"])

OPEN_ALERT_STATUSES = {"open", "acknowledged", "escalated"}
INACTIVE_ALERT_STATUSES = {"resolved", "muted"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _date_key(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def _alert_is_active(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").strip().lower()
    if status in INACTIVE_ALERT_STATUSES:
        return False
    return status in OPEN_ALERT_STATUSES or not status


def capture_dashboard_operational_rows():
    try:
        get_django_apps()
        from digital_twin.services import capture_operational_rows

        return capture_operational_rows(source="live")
    except Exception:
        from .standalone_snapshot import capture_standalone_operational_snapshot

        snapshot = capture_standalone_operational_snapshot(
            run_predictions=False,
            advance_tick=False,
        )
        payload = snapshot.get("payload") or {}

        class StandaloneRows:
            normalized = payload
            redacted = payload
            freshness = snapshot.get("freshness") or {}
            warnings = list(snapshot.get("warnings") or [])
            watermarks = snapshot.get("source_watermarks") or {}

        return StandaloneRows()


def _facility_positions(count: int) -> list[tuple[int, int]]:
    base_positions = [
        (22, 24),
        (58, 28),
        (78, 38),
        (28, 58),
        (52, 62),
        (75, 72),
        (15, 72),
        (42, 42),
        (86, 58),
        (34, 78),
        (64, 50),
        (18, 42),
    ]
    if count <= len(base_positions):
        return base_positions[:count]
    positions = list(base_positions)
    for index in range(len(base_positions), count):
        positions.append((12 + (index * 19) % 76, 22 + (index * 23) % 58))
    return positions


def _group_current_supply(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, dict[str, float]], float, float]:
    by_type: dict[str, float] = defaultdict(float)
    by_hospital: dict[str, dict[str, float]] = defaultdict(lambda: {"stock": 0.0, "usage": 0.0})
    total_stock = 0.0
    total_usage = 0.0
    for row in rows:
        hospital_id = str(row.get("hospital_id") or "unknown")
        blood_type = str(row.get("blood_product_type") or "Unknown")
        stock = max(_as_float(row.get("current_stock_units")), 0.0)
        usage = max(_as_float(row.get("usage_today")), 0.0)
        by_type[blood_type] += stock
        by_hospital[hospital_id]["stock"] += stock
        by_hospital[hospital_id]["usage"] += usage
        total_stock += stock
        total_usage += usage
    return dict(by_type), dict(by_hospital), total_stock, total_usage


def _build_facilities(
    hospitals: list[dict[str, Any]],
    supplies: list[dict[str, Any]],
    active_alerts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not hospitals:
        seen: dict[str, dict[str, Any]] = {}
        for row in supplies:
            hospital_id = str(row.get("hospital_id") or "").strip()
            if not hospital_id or hospital_id in seen:
                continue
            seen[hospital_id] = {
                "hospital_id": hospital_id,
                "name": row.get("hospital_name") or hospital_id,
                "wilaya": "",
            }
        hospitals = list(seen.values())

    _, supply_by_hospital, _, _ = _group_current_supply(supplies)
    alert_refs = {
        str(row.get("entity_id") or row.get("source_ref") or "").strip()
        for row in active_alerts
    }
    positions = _facility_positions(len(hospitals))
    facilities: list[dict[str, Any]] = []
    for index, hospital in enumerate(hospitals):
        hospital_id = str(hospital.get("hospital_id") or f"hospital-{index + 1}")
        totals = supply_by_hospital.get(hospital_id, {"stock": 0.0, "usage": 0.0})
        stock = totals["stock"]
        usage = totals["usage"]
        denom = stock + usage
        utilization = round((usage / denom) * 100, 1) if denom else 0.0
        x, y = positions[index]
        facilities.append(
            {
                "id": hospital_id,
                "name": str(hospital.get("name") or hospital_id),
                "type": "hospital",
                "glyph": "H",
                "x": x,
                "y": y,
                "stock": round(stock, 2),
                "usage": round(usage, 2),
                "utilization": utilization,
                "critical": hospital_id in alert_refs,
                "region": hospital.get("wilaya") or "",
            }
        )
    return facilities


def _route_path_between(source: dict[str, Any], target: dict[str, Any], offset: float) -> str:
    x1 = _as_float(source.get("x"), 50.0) * 9.6
    y1 = _as_float(source.get("y"), 50.0) * 6.2
    x2 = _as_float(target.get("x"), 50.0) * 9.6
    y2 = _as_float(target.get("y"), 50.0) * 6.2
    cx1 = x1 + (x2 - x1) * 0.28
    cy1 = y1 - 52 + offset
    cx2 = x1 + (x2 - x1) * 0.72
    cy2 = y2 + 42 - offset
    return (
        f"M {x1:.1f} {y1:.1f} "
        f"C {cx1:.1f} {cy1:.1f} {cx2:.1f} {cy2:.1f} {x2:.1f} {y2:.1f}"
    )


def _route_visual_metadata(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    x1 = _as_float(source.get("x"), 50.0)
    y1 = _as_float(source.get("y"), 50.0)
    x2 = _as_float(target.get("x"), 50.0)
    y2 = _as_float(target.get("y"), 50.0)
    rotation = 0.0
    if x1 != x2 or y1 != y2:
        import math

        rotation = math.degrees(math.atan2(y2 - y1, x2 - x1))
    pressure = _as_float(target.get("utilization"), 0.0)
    return {
        "sourceName": source.get("name") or source.get("id") or "",
        "targetName": target.get("name") or target.get("id") or "",
        "midpoint": {
            "x": round((x1 + x2) / 2.0, 1),
            "y": round((y1 + y2) / 2.0, 1),
        },
        "rotation": f"{rotation:.1f}deg",
        "pressure": round(pressure, 1),
    }


def _build_network_routes(facilities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(facilities) < 2:
        return []
    hub = max(facilities, key=lambda row: _as_float(row.get("stock")))
    palette = ("blue", "green", "purple")
    routes: list[dict[str, Any]] = []
    for index, facility in enumerate(
        row for row in facilities if row.get("id") != hub.get("id")
    ):
        if index >= 8:
            break
        routes.append(
            {
                "sourceFacilityId": hub.get("id"),
                "targetFacilityId": facility.get("id"),
                "color": "red" if facility.get("critical") else palette[index % len(palette)],
                "d": _route_path_between(hub, facility, float((index % 3) * 22)),
                "animated": True,
                "derivedFrom": "operational_facilities",
                **_route_visual_metadata(hub, facility),
            }
        )
    return routes


def _blood_type_rows(by_type: dict[str, float], total_stock: float) -> list[dict[str, Any]]:
    rows = []
    for blood_type, units in sorted(by_type.items(), key=lambda item: item[1], reverse=True):
        pct = round((units / total_stock) * 100, 1) if total_stock else 0.0
        rows.append({"type": blood_type, "pct": pct, "units": round(units, 2)})
    return rows


def _forecast_rows(
    predictions: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    supplies: list[dict[str, Any]],
) -> dict[str, Any]:
    demand_by_date: dict[str, float] = defaultdict(float)
    supply_by_date: dict[str, float] = defaultdict(float)
    for row in predictions:
        key = _date_key(row.get("predicted_for_date") or row.get("created_at"))
        if key:
            demand_by_date[key] += max(_as_float(row.get("predicted_value")), 0.0)
    for row in snapshots:
        key = _date_key(row.get("recorded_at"))
        if key:
            supply_by_date[key] += max(_as_float(row.get("current_stock_units")), 0.0)
    if not supply_by_date:
        for row in supplies:
            key = _date_key(row.get("event_timestamp") or row.get("updated_at"))
            if key:
                supply_by_date[key] += max(_as_float(row.get("current_stock_units")), 0.0)

    labels = sorted(set(demand_by_date) | set(supply_by_date))[-7:]
    return {
        "labels": labels,
        "demand": [round(demand_by_date.get(label, 0.0), 2) for label in labels],
        "supply": [round(supply_by_date.get(label, 0.0), 2) for label in labels],
    }


def _inventory_delta(snapshots: list[dict[str, Any]]) -> float | None:
    totals_by_date: dict[str, float] = defaultdict(float)
    for row in snapshots:
        key = str(row.get("recorded_at") or "")
        if not key:
            continue
        totals_by_date[key] += max(_as_float(row.get("current_stock_units")), 0.0)
    ordered = sorted(totals_by_date.items())
    if len(ordered) < 2:
        return None
    previous = ordered[-2][1]
    current = ordered[-1][1]
    if previous <= 0:
        return None
    return round(((current - previous) / previous) * 100, 1)


def _weather_label(features: list[dict[str, Any]]) -> str:
    if not features:
        return "Weather unavailable"

    latest_by_hospital: dict[str, dict[str, Any]] = {}
    for row in features:
        hospital_id = str(row.get("hospital_id") or "").strip()
        if not hospital_id:
            continue
        timestamp = str(row.get("event_timestamp") or row.get("updated_at") or "")
        previous = latest_by_hospital.get(hospital_id)
        previous_timestamp = str(
            (previous or {}).get("event_timestamp") or (previous or {}).get("updated_at") or ""
        )
        if previous is None or timestamp >= previous_timestamp:
            latest_by_hospital[hospital_id] = row

    latest_rows = list(latest_by_hospital.values())
    temperatures = []
    rain_values = []
    for row in latest_rows:
        if row.get("temperature") not in {None, ""}:
            temperature = _as_float(row.get("temperature"), None)
            if temperature is not None:
                temperatures.append(temperature)
        if row.get("rain_mm") not in {None, ""}:
            rain_values.append(max(_as_float(row.get("rain_mm"), 0.0), 0.0))
    if not temperatures and not rain_values:
        return "Weather unavailable"

    parts: list[str] = []
    if temperatures:
        parts.append(f"{round(sum(temperatures) / len(temperatures))}°C")
    if rain_values:
        avg_rain = sum(rain_values) / len(rain_values)
        if avg_rain >= 10:
            parts.append("Heavy rain")
        elif avg_rain >= 1:
            parts.append("Light rain")
        else:
            parts.append("Clear")
    return " ".join(parts) if parts else "Weather unavailable"


def _alert_rows(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in alerts:
        if not _alert_is_active(row):
            continue
        severity = str(row.get("severity") or "warning").lower()
        rows.append(
            {
                "type": "critical" if severity == "critical" else "warning",
                "icon": "droplet" if row.get("blood_type") else "alert-triangle",
                "title": row.get("title") or severity.title(),
                "text": row.get("message") or row.get("title") or "Operational alert",
                "meta": row.get("blood_type") or row.get("entity_type") or row.get("source_type") or "",
                "time": row.get("opened_at") or row.get("last_evaluated_at") or "",
            }
        )
    return rows[:3]


def _utilization_rows(facilities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {"name": facility["name"], "pct": facility["utilization"]}
        for facility in facilities
        if facility.get("stock", 0) or facility.get("usage", 0)
    ]
    return sorted(rows, key=lambda row: row["pct"], reverse=True)[:5]


def _empty_dashboard_payload(*, status: str, warning: str) -> dict[str, Any]:
    return {
        "source": {
            "status": status,
            "label": "Django data unavailable",
            "warnings": [warning],
            "generatedAt": None,
        },
        "kpis": {
            "totalInventory": 0,
            "inventoryDeltaPct": None,
            "inventoryCaption": "No inventory records",
            "unitsInTransit": 0,
            "deliveryCaption": "No transport records",
            "criticalShortages": 0,
            "shortageCaption": "No active alerts",
            "serviceLevel": None,
            "networkEfficiency": None,
            "efficiencyDeltaPct": None,
        },
        "facilities": [],
        "routes": [],
        "vehicles": [],
        "forecast": {"labels": [], "demand": [], "supply": []},
        "alerts": [],
        "bloodTypes": [],
        "deliveries": {
            "total": 0,
            "onTime": "0",
            "delayed": "0",
            "items": [],
            "caption": "No transport records",
        },
        "utilization": [],
        "weather": "Weather unavailable",
    }


def build_dashboard_payload() -> dict[str, Any]:
    rows = capture_dashboard_operational_rows()
    payload = rows.redacted or rows.normalized or {}
    hospitals = list(payload.get("hospitals") or [])
    supplies = list(payload.get("blood_supplies") or [])
    snapshots = list(payload.get("supply_snapshots") or [])
    features = list(payload.get("hospital_features") or [])
    predictions = list(payload.get("predictions") or [])
    active_alerts = [
        row for row in list(payload.get("alerts") or []) if _alert_is_active(row)
    ]
    by_type, _, total_stock, total_usage = _group_current_supply(supplies)
    facilities = _build_facilities(hospitals, supplies, active_alerts)
    critical_alerts = [
        row
        for row in active_alerts
        if str(row.get("severity") or "").strip().lower() == "critical"
    ]
    triggered_predictions = [
        row for row in predictions if _as_bool(row.get("alert_triggered"))
    ]
    critical_shortages = max(len(critical_alerts), len(triggered_predictions))
    service_level = None
    if facilities:
        service_level = round(max(0.0, 100.0 - (critical_shortages / len(facilities)) * 100.0), 1)
    network_efficiency = None
    if total_stock or total_usage:
        network_efficiency = round((total_stock / max(total_stock + total_usage, 1.0)) * 100.0, 1)
    routes = _build_network_routes(facilities)

    freshness = getattr(rows, "freshness", {}) or {}
    status = str(freshness.get("status") or "fresh")
    if getattr(rows, "warnings", None):
        status = "degraded"

    return {
        "source": {
            "status": status,
            "label": "Django operational data",
            "warnings": list(getattr(rows, "warnings", []) or []),
            "generatedAt": freshness.get("captured_at"),
            "watermarks": getattr(rows, "watermarks", {}) or {},
        },
        "kpis": {
            "totalInventory": round(total_stock, 2),
            "inventoryDeltaPct": _inventory_delta(snapshots),
            "inventoryCaption": "Current Django stock" if supplies else "No inventory records",
            "unitsInTransit": 0,
            "deliveryCaption": "No transport records",
            "criticalShortages": critical_shortages,
            "shortageCaption": f"{len(active_alerts)} active alerts",
            "serviceLevel": service_level,
            "networkEfficiency": network_efficiency,
            "efficiencyDeltaPct": None,
        },
        "facilities": facilities,
        "routes": routes,
        "vehicles": [],
        "forecast": _forecast_rows(predictions, snapshots, supplies),
        "alerts": _alert_rows(active_alerts),
        "alertCount": len(active_alerts),
        "bloodTypes": _blood_type_rows(by_type, total_stock),
        "deliveries": {
            "total": 0,
            "onTime": "0",
            "delayed": "0",
            "items": [],
            "caption": "No transport records",
        },
        "utilization": _utilization_rows(facilities),
        "weather": _weather_label(features),
        "simulationSeed": _simulation_seed_metadata(payload, facilities),
    }


def _simulation_seed_metadata(
    payload: dict[str, Any],
    facilities: list[dict[str, Any]],
) -> dict[str, Any]:
    seed_plan = build_operational_seed_plan(payload, config=load_operational_simulation_config())
    hospital_facilities = [row for row in facilities if row.get("type") == "hospital"]
    return {
        "enabled": bool(seed_plan and seed_plan.enabled),
        "hospitalCount": len(hospital_facilities),
        "centerCount": len(seed_plan.center_configs) if seed_plan else 0,
        "source": seed_plan.source if seed_plan else "unavailable",
    }


@router.get("/setup")
def get_studio_setup() -> dict[str, Any]:
    snapshot_backend = detect_snapshot_backend()
    policy_catalog = list_simulation_lab_policies()
    available_policy_count = sum(
        1 for policy in policy_catalog if policy.get("available", True)
    )
    default_universes = default_experiment_universes()
    return {
        "scenarios": list_scenarios(),
        "timeline_slider": {
            "default_months": 0,
            "min_months": 0,
            "max_months": 24,
        },
        "snapshot_sources": ["django", "synthetic"],
        "snapshot_backend": snapshot_backend,
        "policy_catalog": policy_catalog,
        "policy_selection": {
            "min_policies": 2,
            "default_policies": len(default_universes),
            "max_policies": max(2, available_policy_count),
        },
        "default_universes": [universe.model_dump() for universe in default_universes],
        "assistant_candidates": [
            universe.model_dump() for universe in default_assistant_universes()
        ],
        "features": {
            "data_snapshot_layer": True,
            "parallel_universes": True,
            "agent_mode": True,
            "event_replay": True,
            "strategy_assistant": True,
            "mission_control_contract": True,
            "dashboard_ui": False,
        },
    }


@router.post("/snapshots")
def create_snapshot(payload: DataSnapshotRequest) -> dict[str, Any]:
    return capture_data_snapshot(payload).as_dict()


@router.get("/dashboard-data")
def get_dashboard_data() -> dict[str, Any]:
    try:
        return build_dashboard_payload()
    except Exception as exc:
        return _empty_dashboard_payload(status="degraded", warning=str(exc))


@router.post("/experiments/run")
def run_experiment(payload: StudioExperimentRequest) -> dict[str, Any]:
    return run_studio_experiment(payload)


@router.post("/assistant/strategy")
def strategy_assistant(payload: StrategyAssistantRequest) -> dict[str, Any]:
    return run_strategy_assistant(payload)
