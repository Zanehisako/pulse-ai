from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from django.apps import apps
from django.db import transaction
from django.utils import timezone

from .config import (
    DigitalTwinConfigError,
    get_action_config,
    get_event_config,
    load_digital_twin_config,
)
from .models import (
    TwinAction,
    TwinAuditLog,
    TwinBranch,
    TwinEvent,
    TwinFrame,
    TwinRecommendation,
    TwinRun,
    TwinSnapshot,
)

DECISION_SUPPORT = (
    "Digital twin output is decision support only and must not be treated as final medical judgment."
)


@dataclass(frozen=True)
class SnapshotRows:
    normalized: dict[str, Any]
    redacted: dict[str, Any]
    watermarks: dict[str, Any]
    freshness: dict[str, Any]
    warnings: list[str]


def _actor_name(user: Any) -> str:
    if user is not None and getattr(user, "is_authenticated", False):
        return str(user)
    return "system"


def _iso(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if value is not None and not isinstance(value, (str, int, float, bool, list, dict)):
        return str(value)
    return value


def _model(path: str):
    app_label, model_name = path.split(".", 1)
    return apps.get_model(app_label, model_name)


def _read_fields(obj: Any, mapping: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for output_key, attr_path in mapping.items():
        current = obj
        for part in str(attr_path).split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        row[output_key] = _iso(current)
    return row


def _query_source(source_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    model = _model(str(source_cfg["model"]))
    queryset = model.objects.all()
    filters = source_cfg.get("filters")
    if isinstance(filters, dict) and filters:
        queryset = queryset.filter(**filters)
    order_by = source_cfg.get("order_by")
    if isinstance(order_by, list) and order_by:
        queryset = queryset.order_by(*[str(item) for item in order_by])
    elif isinstance(order_by, str) and order_by:
        queryset = queryset.order_by(order_by)
    select_related = source_cfg.get("select_related")
    if isinstance(select_related, list) and select_related:
        queryset = queryset.select_related(*[str(item) for item in select_related])
    limit = int(source_cfg.get("limit") or 500)
    fields = dict(source_cfg.get("fields") or {})
    return [_read_fields(obj, fields) for obj in queryset[:limit]]


def _watermark(rows: list[dict[str, Any]], *keys: str) -> dict[str, Any]:
    values = [
        row[key]
        for row in rows
        for key in keys
        if row.get(key) not in (None, "")
    ]
    return {
        "count": len(rows),
        "latest": max(values) if values else None,
    }


def _hash_donor(value: Any, salt: str) -> str:
    raw = f"{salt}:{value}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _redact_payload(payload: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    privacy = cfg.get("privacy", {})
    if not bool(privacy.get("redaction_enabled", True)):
        return dict(payload)
    redact_fields = set(str(item) for item in privacy.get("redact_fields", []))
    salt = str(privacy.get("donor_hash_salt") or cfg["version"])
    redacted = dict(payload)
    donors: list[dict[str, Any]] = []
    for donor in payload.get("donors", []):
        row = {
            key: value
            for key, value in donor.items()
            if key not in redact_fields
        }
        donor_id = donor.get("donor_id") or donor.get("id")
        row["donor_ref"] = _hash_donor(donor_id or donor, salt)
        donors.append(row)
    redacted["donors"] = donors
    return redacted


def _synthetic_rows(cfg: dict[str, Any]) -> SnapshotRows:
    now = timezone.now()
    hospitals = [
        {"hospital_id": "SYN-H001", "name": "Synthetic Hospital", "wilaya": "Synthetic"}
    ]
    supplies = [
        {
            "supply_id": f"SYN-{blood}",
            "hospital_id": "SYN-H001",
            "blood_product_type": blood,
            "current_stock_units": 120.0,
            "usage_today": 12.0,
            "event_timestamp": now.isoformat(),
        }
        for blood in ("O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-")
    ]
    normalized = {
        "hospitals": hospitals,
        "blood_supplies": supplies,
        "supply_snapshots": [],
        "hospital_features": [],
        "donors": [],
        "predictions": [],
        "alerts": [],
        "model_configs": [],
        "drift_reports": [],
    }
    return SnapshotRows(
        normalized=normalized,
        redacted=_redact_payload(normalized, cfg),
        watermarks={"synthetic": {"count": 1, "latest": now.isoformat()}},
        freshness={"captured_at": now.isoformat(), "status": "synthetic", "is_stale": False},
        warnings=["Synthetic digital twin snapshot used."],
    )


def capture_operational_rows(*, source: str = "live") -> SnapshotRows:
    cfg = load_digital_twin_config()
    if source == "synthetic":
        return _synthetic_rows(cfg)

    normalized: dict[str, Any] = {}
    watermarks: dict[str, Any] = {}
    warnings: list[str] = []
    for source_name, source_cfg in cfg.get("data_sources", {}).items():
        try:
            rows = _query_source(dict(source_cfg))
            normalized[source_name] = rows
            watermarks[source_name] = _watermark(rows, "updated_at", "event_timestamp", "created_at", "opened_at", "checked_at", "recorded_at")
        except Exception as exc:
            normalized[source_name] = []
            watermarks[source_name] = {"count": 0, "latest": None}
            warnings.append(f"{source_name} snapshot failed: {exc}")

    now = timezone.now()
    stale_after = int(cfg.get("sync", {}).get("stale_after_seconds", 180))
    latest_values = [
        value.get("latest")
        for value in watermarks.values()
        if isinstance(value, dict) and value.get("latest")
    ]
    freshness = {
        "captured_at": now.isoformat(),
        "status": "fresh" if latest_values else "degraded",
        "is_stale": False if latest_values else True,
        "stale_after_seconds": stale_after,
    }
    return SnapshotRows(
        normalized=normalized,
        redacted=_redact_payload(normalized, cfg),
        watermarks=watermarks,
        freshness=freshness,
        warnings=warnings,
    )


def audit(
    *,
    run: TwinRun | None,
    action: str,
    actor: str = "system",
    metadata: dict[str, Any] | None = None,
    source_snapshot_id: str = "",
) -> TwinAuditLog:
    cfg = load_digital_twin_config()
    return TwinAuditLog.objects.create(
        run=run,
        actor=actor,
        action=action,
        config_version=cfg["version"],
        source_snapshot_id=source_snapshot_id,
        metadata=metadata or {},
    )


@transaction.atomic
def create_twin_run(payload: dict[str, Any], *, user: Any = None) -> TwinRun:
    cfg = load_digital_twin_config()
    run = TwinRun.objects.create(
        mode=str(payload.get("mode") or "operational_shadow"),
        status=TwinRun.Status.RUNNING,
        config_version=cfg["version"],
        scenario_key=str(payload.get("scenario_key") or cfg.get("runtime", {}).get("default_scenario_key", "baseline")),
        strategy_key=str(payload.get("strategy_key") or cfg.get("runtime", {}).get("default_strategy_key", "baseline")),
        source_mode=str(payload.get("source_mode") or payload.get("source") or "live"),
        seed=int(payload.get("seed") or cfg.get("runtime", {}).get("default_seed", 100)),
        created_by=_actor_name(user),
    )
    audit(run=run, action="run_created", actor=run.created_by, metadata={"payload": payload})
    return run


@transaction.atomic
def capture_snapshot(run: TwinRun, *, source: str | None = None, actor: str = "system") -> TwinSnapshot:
    selected_source = source or run.source_mode
    rows = capture_operational_rows(source=selected_source)
    snapshot = TwinSnapshot.objects.create(
        run=run,
        source=selected_source,
        source_watermarks=rows.watermarks,
        normalized_payload=rows.normalized,
        redacted_payload=rows.redacted,
        freshness=rows.freshness,
        warnings=rows.warnings,
    )
    audit(
        run=run,
        action="snapshot_captured",
        actor=actor,
        source_snapshot_id=str(snapshot.snapshot_id),
        metadata={"source": selected_source, "warnings": rows.warnings, "watermarks": rows.watermarks},
    )
    return snapshot


def simulated_state_from_snapshot(snapshot: TwinSnapshot, *, simulated_hour: float = 0.0) -> dict[str, Any]:
    payload = snapshot.redacted_payload
    supplies = payload.get("blood_supplies", [])
    inventory_by_hospital: dict[str, dict[str, float]] = {}
    total_inventory = 0.0
    for row in supplies:
        hospital_id = str(row.get("hospital_id") or "unknown")
        blood_type = str(row.get("blood_product_type") or "unknown")
        units = float(row.get("current_stock_units") or 0.0)
        inventory_by_hospital.setdefault(hospital_id, {})[blood_type] = units
        total_inventory += units
    alerts = payload.get("alerts", [])
    predictions = payload.get("predictions", [])
    drift_reports = payload.get("drift_reports", [])
    return {
        "simulated_hour": round(float(simulated_hour), 2),
        "inventory_by_hospital": inventory_by_hospital,
        "total_inventory_units": round(total_inventory, 2),
        "donor_count": len(payload.get("donors", [])),
        "active_alert_count": len(alerts),
        "latest_prediction_count": len(predictions),
        "critical_drift_count": len(
            [row for row in drift_reports if str(row.get("severity", "")).lower() == "critical"]
        ),
        "real_signals": {
            "alerts": alerts[:25],
            "predictions": predictions[:50],
            "drift_reports": drift_reports[:25],
            "model_configs": payload.get("model_configs", [])[:50],
        },
    }


def _latest_snapshot(run: TwinRun) -> TwinSnapshot:
    snapshot = run.snapshots.order_by("-captured_at").first()
    if snapshot is None:
        snapshot = capture_snapshot(run)
    return snapshot


def _latest_frame(run: TwinRun) -> TwinFrame | None:
    return run.frames.order_by("-tick_number").first()


def compute_divergence(snapshot: TwinSnapshot, simulated_state: dict[str, Any]) -> dict[str, Any]:
    real_state = simulated_state_from_snapshot(snapshot, simulated_hour=simulated_state.get("simulated_hour", 0.0))
    real_inventory = float(real_state.get("total_inventory_units", 0.0))
    simulated_inventory = float(simulated_state.get("total_inventory_units", 0.0))
    inventory_delta = simulated_inventory - real_inventory
    denom = max(real_inventory, 1.0)
    inventory_delta_pct = inventory_delta / denom
    donor_delta = int(simulated_state.get("donor_count", 0)) - int(real_state.get("donor_count", 0))
    cfg = load_digital_twin_config()
    thresholds = cfg.get("reconciliation", {}).get("thresholds", {})
    inventory_threshold = float(thresholds.get("inventory_delta_pct", 0.2))
    donor_threshold = int(thresholds.get("donor_count_delta", 25))
    return {
        "inventory_delta_units": round(inventory_delta, 2),
        "inventory_delta_pct": round(inventory_delta_pct, 4),
        "donor_count_delta": donor_delta,
        "alert_count_delta": int(simulated_state.get("active_alert_count", 0)) - int(real_state.get("active_alert_count", 0)),
        "status": "diverged"
        if abs(inventory_delta_pct) > inventory_threshold or abs(donor_delta) > donor_threshold
        else "aligned",
        "thresholds": {
            "inventory_delta_pct": inventory_threshold,
            "donor_count_delta": donor_threshold,
        },
    }


def apply_reconciliation(snapshot: TwinSnapshot, simulated_state: dict[str, Any]) -> dict[str, Any]:
    cfg = load_digital_twin_config()
    rules = cfg.get("reconciliation", {}).get("rules", {})
    if str(rules.get("inventory", "blend")) == "reset_on_divergence":
        divergence = compute_divergence(snapshot, simulated_state)
        if divergence.get("status") == "diverged":
            real_state = simulated_state_from_snapshot(
                snapshot,
                simulated_hour=float(simulated_state.get("simulated_hour", 0.0)),
            )
            simulated_state = {
                **simulated_state,
                "inventory_by_hospital": real_state["inventory_by_hospital"],
                "total_inventory_units": real_state["total_inventory_units"],
                "reconciled": True,
            }
    return simulated_state


@transaction.atomic
def create_frame(
    run: TwinRun,
    *,
    tick_number: int,
    simulated_hour: float,
    snapshot: TwinSnapshot | None = None,
    simulated_state: dict[str, Any] | None = None,
    active_events: list[dict[str, Any]] | None = None,
    active_actions: list[dict[str, Any]] | None = None,
    actor: str = "system",
) -> TwinFrame:
    snapshot = snapshot or _latest_snapshot(run)
    state = simulated_state or simulated_state_from_snapshot(snapshot, simulated_hour=simulated_hour)
    state = apply_reconciliation(snapshot, state)
    divergence = compute_divergence(snapshot, state)
    frame = TwinFrame.objects.create(
        run=run,
        tick_number=tick_number,
        simulated_hour=simulated_hour,
        real_snapshot=snapshot,
        simulated_state=state,
        divergence=divergence,
        active_events=active_events or [],
        active_actions=active_actions or [],
    )
    audit(
        run=run,
        action="frame_created",
        actor=actor,
        source_snapshot_id=str(snapshot.snapshot_id),
        metadata={"tick_number": tick_number, "divergence": divergence},
    )
    return frame


@transaction.atomic
def inject_event(
    run: TwinRun,
    *,
    event_key: str,
    severity: float,
    start_hour: float,
    source: str = "manual",
    payload: dict[str, Any] | None = None,
    actor: str = "system",
) -> TwinEvent:
    cfg = load_digital_twin_config()
    event_cfg = get_event_config(event_key)
    clamped_severity = max(
        float(event_cfg.get("min_severity", 0.0)),
        min(float(severity), float(event_cfg.get("max_severity", 1.0))),
    )
    duration = float(event_cfg["duration_hours"])
    effects = {
        field: round(float(value) * max(clamped_severity, 0.01), 4)
        for field, value in dict(event_cfg.get("effects") or {}).items()
    }
    event = TwinEvent.objects.create(
        run=run,
        event_key=event_key,
        source=source,
        severity=clamped_severity,
        start_hour=start_hour,
        end_hour=start_hour + duration,
        config_version=cfg["version"],
        payload={**event_cfg, **(payload or {})},
        applied_effects=effects,
    )
    audit(run=run, action="event_injected", actor=actor, metadata={"event_id": str(event.event_id), "event_key": event_key, "effects": effects})
    return event


@transaction.atomic
def execute_action(
    run: TwinRun,
    *,
    action_key: str,
    simulated_hour: float,
    source: str = "manual",
    payload: dict[str, Any] | None = None,
    actor: str = "system",
) -> TwinAction:
    cfg = load_digital_twin_config()
    action_cfg = get_action_config(action_key)
    promotion_enabled = bool(cfg.get("promotion_targets", {}).get("enabled", False))
    promotion_status = (
        TwinAction.PromotionStatus.PENDING
        if promotion_enabled and bool(action_cfg.get("promotion_eligible", False))
        else TwinAction.PromotionStatus.DISABLED
    )
    effect = {
        "effects": dict(action_cfg.get("effects") or {}),
        "cost": float(action_cfg.get("cost", 0.0)),
        "simulated_hour": round(float(simulated_hour), 2),
        "decision_support": DECISION_SUPPORT,
    }
    action = TwinAction.objects.create(
        run=run,
        action_key=action_key,
        source=source,
        requested_by=actor,
        simulated_effect=effect,
        audit_metadata={"payload": payload or {}, "config_version": cfg["version"]},
        promotion_status=promotion_status,
    )
    audit(run=run, action="action_simulated", actor=actor, metadata={"action_id": str(action.action_id), "action_key": action_key, "effect": effect})
    return action


@transaction.atomic
def run_branch(
    run: TwinRun,
    *,
    branch_key: str,
    policy_overrides: dict[str, Any] | None = None,
    action_overrides: list[dict[str, Any]] | None = None,
    event_overrides: list[dict[str, Any]] | None = None,
    actor: str = "system",
) -> TwinBranch:
    cfg = load_digital_twin_config()
    if run.branches.count() >= int(cfg.get("branches", {}).get("max_branches", 5)):
        raise DigitalTwinConfigError("Maximum number of digital twin branches reached.")
    parent_frame = _latest_frame(run)
    base_state = parent_frame.simulated_state if parent_frame else simulated_state_from_snapshot(_latest_snapshot(run))
    action_count = len(action_overrides or [])
    event_count = len(event_overrides or [])
    alert_pressure = float(base_state.get("active_alert_count", 0))
    summary = {
        "base_inventory_units": base_state.get("total_inventory_units", 0.0),
        "projected_shortage_pressure": round(max(alert_pressure - action_count + event_count, 0.0), 4),
        "policy_count": len(policy_overrides or {}),
        "decision_support": DECISION_SUPPORT,
    }
    branch = TwinBranch.objects.create(
        run=run,
        branch_key=branch_key,
        parent_frame=parent_frame,
        policy_overrides=policy_overrides or {},
        action_overrides=action_overrides or [],
        event_overrides=event_overrides or [],
        summary_metrics=summary,
        confidence_bands=[
            {"metric": "projected_shortage_pressure", "low": max(summary["projected_shortage_pressure"] - 0.1, 0), "mean": summary["projected_shortage_pressure"], "high": summary["projected_shortage_pressure"] + 0.1}
        ],
    )
    audit(run=run, action="branch_created", actor=actor, metadata={"branch_id": str(branch.branch_id), "branch_key": branch_key})
    return branch


@transaction.atomic
def recommend_action(
    run: TwinRun,
    *,
    branch: TwinBranch | None = None,
    source_tool_ids: list[str] | None = None,
    actor: str = "system",
) -> TwinRecommendation:
    cfg = load_digital_twin_config()
    frame = _latest_frame(run)
    state = frame.simulated_state if frame else simulated_state_from_snapshot(_latest_snapshot(run))
    actions = list(cfg.get("actions", []))
    selected = actions[0] if actions else {"id": "monitor", "label": "Continue monitoring"}
    if float(state.get("active_alert_count", 0)) > 0:
        selected = max(
            actions,
            key=lambda row: float(row.get("recommendation_priority", 0)),
            default=selected,
        )
    recommendation = TwinRecommendation.objects.create(
        run=run,
        branch=branch,
        recommendation_key=str(selected.get("id")),
        rationale=(
            f"Recommended simulated action '{selected.get('label')}' from current twin signals. "
            f"{DECISION_SUPPORT}"
        ),
        expected_impact={
            "active_alert_count": state.get("active_alert_count", 0),
            "total_inventory_units": state.get("total_inventory_units", 0.0),
            "effects": selected.get("effects", {}),
        },
        risk_flags=["operator_approval_required"],
        source_tool_ids=list(source_tool_ids or []),
        source_model_ids=[
            str(row.get("model_id"))
            for row in state.get("real_signals", {}).get("model_configs", [])
            if row.get("model_id")
        ][:10],
    )
    audit(run=run, action="recommendation_created", actor=actor, metadata={"recommendation_id": str(recommendation.recommendation_id)})
    return recommendation


@transaction.atomic
def promote_action(action: TwinAction, *, actor: str = "system") -> TwinAction:
    cfg = load_digital_twin_config()
    if not bool(cfg.get("promotion_targets", {}).get("enabled", False)):
        audit(run=action.run, action="promotion_blocked", actor=actor, metadata={"action_id": str(action.action_id), "reason": "promotion_disabled"})
        raise DigitalTwinConfigError("Digital twin promotion targets are disabled by config.")
    if action.promotion_status != TwinAction.PromotionStatus.PENDING:
        raise DigitalTwinConfigError("This digital twin action is not eligible for promotion.")
    action.status = TwinAction.Status.PROMOTED
    action.promotion_status = TwinAction.PromotionStatus.PROMOTED
    action.promoted_at = timezone.now()
    action.save(update_fields=["status", "promotion_status", "promoted_at"])
    audit(run=action.run, action="action_promoted", actor=actor, metadata={"action_id": str(action.action_id)})
    return action


def stop_run(run: TwinRun, *, actor: str = "system") -> TwinRun:
    run.status = TwinRun.Status.STOPPED
    run.stopped_at = timezone.now()
    run.save(update_fields=["status", "stopped_at", "updated_at"])
    audit(run=run, action="run_stopped", actor=actor)
    return run


def current_state(run: TwinRun) -> dict[str, Any]:
    snapshot = run.snapshots.order_by("-captured_at").first()
    frame = run.frames.order_by("-tick_number").first()
    return {
        "run": run_payload(run),
        "latest_snapshot": snapshot_payload(snapshot) if snapshot else None,
        "latest_frame": frame_payload(frame) if frame else None,
        "decision_support": DECISION_SUPPORT,
    }


def run_payload(run: TwinRun) -> dict[str, Any]:
    return {
        "run_id": str(run.run_id),
        "mode": run.mode,
        "status": run.status,
        "config_version": run.config_version,
        "scenario_key": run.scenario_key,
        "strategy_key": run.strategy_key,
        "source_mode": run.source_mode,
        "seed": run.seed,
        "created_by": run.created_by,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "stopped_at": run.stopped_at.isoformat() if run.stopped_at else None,
    }


def snapshot_payload(snapshot: TwinSnapshot) -> dict[str, Any]:
    return {
        "snapshot_id": str(snapshot.snapshot_id),
        "run_id": str(snapshot.run_id),
        "source": snapshot.source,
        "captured_at": snapshot.captured_at.isoformat(),
        "source_watermarks": snapshot.source_watermarks,
        "payload": snapshot.redacted_payload,
        "freshness": snapshot.freshness,
        "warnings": snapshot.warnings,
    }


def frame_payload(frame: TwinFrame) -> dict[str, Any]:
    return {
        "frame_id": str(frame.frame_id),
        "run_id": str(frame.run_id),
        "tick_number": frame.tick_number,
        "simulated_hour": frame.simulated_hour,
        "real_snapshot_id": str(frame.real_snapshot_id) if frame.real_snapshot_id else None,
        "simulated_state": frame.simulated_state,
        "divergence": frame.divergence,
        "active_events": frame.active_events,
        "active_actions": frame.active_actions,
        "created_at": frame.created_at.isoformat(),
    }


def event_payload(event: TwinEvent) -> dict[str, Any]:
    return {
        "event_id": str(event.event_id),
        "run_id": str(event.run_id),
        "event_key": event.event_key,
        "source": event.source,
        "severity": event.severity,
        "start_hour": event.start_hour,
        "end_hour": event.end_hour,
        "config_version": event.config_version,
        "payload": event.payload,
        "applied_effects": event.applied_effects,
        "status": event.status,
        "created_at": event.created_at.isoformat(),
    }


def action_payload(action: TwinAction) -> dict[str, Any]:
    return {
        "action_id": str(action.action_id),
        "run_id": str(action.run_id),
        "action_key": action.action_key,
        "source": action.source,
        "requested_by": action.requested_by,
        "status": action.status,
        "simulated_effect": action.simulated_effect,
        "promotion_status": action.promotion_status,
        "created_at": action.created_at.isoformat(),
        "promoted_at": action.promoted_at.isoformat() if action.promoted_at else None,
    }


def branch_payload(branch: TwinBranch) -> dict[str, Any]:
    return {
        "branch_id": str(branch.branch_id),
        "run_id": str(branch.run_id),
        "branch_key": branch.branch_key,
        "parent_frame_id": str(branch.parent_frame_id) if branch.parent_frame_id else None,
        "policy_overrides": branch.policy_overrides,
        "action_overrides": branch.action_overrides,
        "event_overrides": branch.event_overrides,
        "summary_metrics": branch.summary_metrics,
        "confidence_bands": branch.confidence_bands,
        "created_at": branch.created_at.isoformat(),
    }


def recommendation_payload(recommendation: TwinRecommendation) -> dict[str, Any]:
    return {
        "recommendation_id": str(recommendation.recommendation_id),
        "run_id": str(recommendation.run_id),
        "branch_id": str(recommendation.branch_id) if recommendation.branch_id else None,
        "recommendation_key": recommendation.recommendation_key,
        "rationale": recommendation.rationale,
        "expected_impact": recommendation.expected_impact,
        "risk_flags": recommendation.risk_flags,
        "source_tool_ids": recommendation.source_tool_ids,
        "source_model_ids": recommendation.source_model_ids,
        "created_at": recommendation.created_at.isoformat(),
        "decision_support": DECISION_SUPPORT,
    }
