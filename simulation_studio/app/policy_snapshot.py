"""Summarize simulator policy_log entries for WebSocket UI snapshots."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

STUDIO_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_ROOT = STUDIO_ROOT / "simulator"
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))

from ppo_shared import (  # noqa: E402
    COMPONENT_ALLOCATION_CONTROL_MAP,
    HOSPITAL_ROUTE_CONTROL_MAP,
)
from scenarios import ACTION_CATALOG  # noqa: E402

_TOP_LEVEL_LIMIT = 5
_DECISION_KINDS = frozenset({"decision", "continuous_decision", "activation"})


def _display_name(action_key: str) -> str:
    action = ACTION_CATALOG.get(action_key)
    if action is not None:
        return action.name
    hospital = HOSPITAL_ROUTE_CONTROL_MAP.get(action_key)
    if hospital is not None:
        return f"Route priority: {hospital}"
    component = COMPONENT_ALLOCATION_CONTROL_MAP.get(action_key)
    if component is not None:
        return f"Allocation: {component}"
    return action_key.replace("_", " ").title()


def _headline_continuous(entry: dict[str, Any]) -> str:
    upserted = entry.get("upserted") or []
    deactivated = entry.get("deactivated") or []
    routing = entry.get("routing_updates") or []
    allocation = entry.get("allocation_updates") or []
    parts: list[str] = []
    if upserted:
        changed = ", ".join(
            f"{_display_name(item.get('action_key', ''))} {float(item.get('level', 0)):.2f}"
            for item in upserted[:4]
        )
        parts.append(f"Changed: {changed}")
    if routing:
        route_bits = ", ".join(
            f"{_display_name(item.get('route_key', ''))} {float(item.get('level', 0)):.2f}"
            for item in routing[:3]
        )
        parts.append(f"Routing: {route_bits}")
    if allocation:
        alloc_bits = ", ".join(
            f"{_display_name(item.get('allocation_key', ''))} {float(item.get('level', 0)):.2f}"
            for item in allocation[:3]
        )
        parts.append(f"Allocation: {alloc_bits}")
    if deactivated:
        off_keys = ", ".join(_display_name(key) for key in deactivated[:4])
        parts.append(f"Off: {off_keys}")
    if not parts:
        return "Held previous controls"
    return " · ".join(parts)


def _headline_discrete(entry: dict[str, Any]) -> str:
    selected = entry.get("selected_action")
    if not selected:
        return "No action selected"
    prob = None
    policy = entry.get("policy") or {}
    if isinstance(policy, dict) and selected in policy:
        prob = float(policy[selected])
    name = _display_name(str(selected))
    applied = entry.get("activated")
    if prob is not None:
        return f"Picked: {name} (p={prob:.2f}) · Applied: {'yes' if applied else 'no'}"
    return f"Picked: {name} · Applied: {'yes' if applied else 'no'}"


def _collect_changes(entry: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for item in entry.get("upserted") or []:
        key = str(item.get("action_key", ""))
        level = float(item.get("level", 0))
        changes.append(
            {
                "key": key,
                "name": _display_name(key),
                "level": round(level, 4),
                "direction": "up" if level > 0 else "neutral",
            }
        )
    for key in entry.get("deactivated") or []:
        key_str = str(key)
        changes.append(
            {
                "key": key_str,
                "name": _display_name(key_str),
                "level": 0.0,
                "direction": "off",
            }
        )
    for item in entry.get("routing_updates") or []:
        key = str(item.get("route_key", ""))
        level = float(item.get("level", 0))
        changes.append(
            {
                "key": key,
                "name": _display_name(key),
                "level": round(level, 4),
                "direction": "route",
            }
        )
    for item in entry.get("allocation_updates") or []:
        key = str(item.get("allocation_key", ""))
        level = float(item.get("level", 0))
        changes.append(
            {
                "key": key,
                "name": _display_name(key),
                "level": round(level, 4),
                "direction": "allocation",
            }
        )
    return changes


def _top_levels_from_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    kind = entry.get("kind")
    rows: list[tuple[str, float]] = []
    if kind == "continuous_decision":
        levels = entry.get("levels") or {}
        if isinstance(levels, dict):
            rows = [(str(k), float(v)) for k, v in levels.items()]
    elif kind == "decision":
        policy = entry.get("policy") or {}
        if isinstance(policy, dict):
            rows = [(str(k), float(v)) for k, v in policy.items()]
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    top: list[dict[str, Any]] = []
    for key, level in rows[:_TOP_LEVEL_LIMIT]:
        top.append(
            {
                "key": key,
                "name": _display_name(key),
                "level": round(level, 4),
            }
        )
    return top


def summarize_policy_events(
    policy_log: list[dict[str, Any]],
    since_index: int = 0,
    *,
    runtime_controller: str | None = None,
) -> dict[str, Any] | None:
    """
    Build a UI-safe summary from policy_log entries since ``since_index``.

    Returns None when there is no controller or no decision entries in the slice.
    """
    if since_index < 0:
        since_index = 0
    slice_entries = policy_log[since_index:]
    decision_entries = [
        entry
        for entry in slice_entries
        if entry.get("kind") in _DECISION_KINDS
    ]
    if not decision_entries and not runtime_controller:
        return None

    primary = decision_entries[-1] if decision_entries else None
    if primary is None:
        return {
            "controller": runtime_controller,
            "kind": "idle",
            "hour": round(float(policy_log[-1]["time"]), 2) if policy_log else 0.0,
            "headline": "Held previous controls",
            "changes": [],
            "top_levels": [],
            "discrete_pick": None,
            "has_changes": False,
        }

    kind = str(primary.get("kind", ""))
    hour = round(float(primary.get("time", 0.0)), 2)
    controller = str(primary.get("controller") or runtime_controller or "")
    changes = _collect_changes(primary) if kind == "continuous_decision" else []
    if kind == "decision":
        selected = primary.get("selected_action")
        discrete_pick = (
            {
                "key": str(selected),
                "name": _display_name(str(selected)),
                "probability": float((primary.get("policy") or {}).get(selected, 0.0))
                if selected
                else None,
                "activated": bool(primary.get("activated")),
            }
            if selected
            else None
        )
        headline = _headline_discrete(primary)
        top_levels = _top_levels_from_entry(primary)
        has_changes = bool(primary.get("activated"))
    elif kind == "continuous_decision":
        discrete_pick = None
        headline = _headline_continuous(primary)
        top_levels = _top_levels_from_entry(primary)
        has_changes = bool(changes)
    elif kind == "activation":
        action_key = str(primary.get("action", ""))
        discrete_pick = None
        headline = f"Activated {_display_name(action_key)}"
        top_levels = []
        has_changes = True
        changes = [
            {
                "key": action_key,
                "name": _display_name(action_key),
                "level": 1.0,
                "direction": "up",
            }
        ]
    else:
        return None

    return {
        "controller": controller or runtime_controller,
        "kind": kind,
        "hour": hour,
        "headline": headline,
        "changes": changes,
        "top_levels": top_levels,
        "discrete_pick": discrete_pick,
        "has_changes": has_changes,
    }