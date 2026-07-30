from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from django.conf import settings


SUPPORTED_SOURCE_ADAPTERS = {"django_model", "django_query"}
class DigitalTwinConfigError(RuntimeError):
    pass


def digital_twin_config_path() -> Path:
    raw = os.getenv("PIOS_DIGITAL_TWIN_CONFIG_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(settings.PIOS_DIGITAL_TWIN_CONFIG_PATH).expanduser().resolve()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DigitalTwinConfigError(f"{label} must be an object.")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise DigitalTwinConfigError(f"{label} must be a list.")
    return value


def _require_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise DigitalTwinConfigError(f"{label} must be non-empty.")
    return text


def _validate_unique_rows(rows: list[Any], *, id_field: str, label: str) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        item = _require_mapping(row, f"{label}[{index}]")
        item_id = _require_text(item.get(id_field), f"{label}[{index}].{id_field}")
        if item_id in seen:
            raise DigitalTwinConfigError(f"Duplicate {label} id: {item_id}")
        seen.add(item_id)


def _validate_data_sources(payload: dict[str, Any]) -> None:
    data_sources = _require_mapping(payload.get("data_sources"), "data_sources")
    for source_name, source in data_sources.items():
        source_cfg = _require_mapping(source, f"data_sources.{source_name}")
        adapter = _require_text(source_cfg.get("adapter"), f"data_sources.{source_name}.adapter")
        if adapter not in SUPPORTED_SOURCE_ADAPTERS:
            raise DigitalTwinConfigError(
                f"data_sources.{source_name}.adapter is unsupported: {adapter}"
            )
        _require_text(source_cfg.get("model"), f"data_sources.{source_name}.model")
        fields = _require_mapping(source_cfg.get("fields"), f"data_sources.{source_name}.fields")
        if not fields:
            raise DigitalTwinConfigError(f"data_sources.{source_name}.fields cannot be empty.")
        limit = source_cfg.get("limit")
        if limit is not None and int(limit) <= 0:
            raise DigitalTwinConfigError(f"data_sources.{source_name}.limit must be positive.")


def _supported_effect_fields(payload: dict[str, Any]) -> set[str]:
    runtime = _require_mapping(payload.get("runtime"), "runtime")
    fields = _require_list(
        runtime.get("supported_effect_fields"),
        "runtime.supported_effect_fields",
    )
    supported = {str(item).strip() for item in fields if str(item).strip()}
    if not supported:
        raise DigitalTwinConfigError("runtime.supported_effect_fields cannot be empty.")
    return supported


def _validate_effects(row: dict[str, Any], label: str, supported_fields: set[str]) -> None:
    effects = _require_mapping(row.get("effects"), f"{label}.effects")
    for field_name, value in effects.items():
        if field_name not in supported_fields:
            raise DigitalTwinConfigError(f"{label}.effects has unsupported field: {field_name}")
        try:
            float(value)
        except (TypeError, ValueError) as exc:
            raise DigitalTwinConfigError(f"{label}.effects.{field_name} must be numeric.") from exc


def validate_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    root = _require_mapping(payload, "digital_twin")
    _require_text(root.get("version"), "version")
    supported_effect_fields = _supported_effect_fields(root)
    _require_mapping(root.get("sync"), "sync")
    _require_mapping(root.get("retention"), "retention")
    _validate_data_sources(root)

    privacy = _require_mapping(root.get("privacy"), "privacy")
    _require_list(privacy.get("redact_fields"), "privacy.redact_fields")
    _require_text(privacy.get("donor_hash_salt"), "privacy.donor_hash_salt")

    state_schema = _require_mapping(root.get("state_schema"), "state_schema")
    for entity, required_fields in state_schema.items():
        fields = _require_list(required_fields, f"state_schema.{entity}")
        if not fields:
            raise DigitalTwinConfigError(f"state_schema.{entity} cannot be empty.")

    reconciliation = _require_mapping(root.get("reconciliation"), "reconciliation")
    _require_mapping(reconciliation.get("thresholds"), "reconciliation.thresholds")
    _require_mapping(reconciliation.get("rules"), "reconciliation.rules")

    events = _require_list(root.get("events"), "events")
    _validate_unique_rows(events, id_field="id", label="events")
    for index, row in enumerate(events):
        event = _require_mapping(row, f"events[{index}]")
        _require_text(event.get("label"), f"events[{index}].label")
        if float(event.get("duration_hours", 0)) <= 0:
            raise DigitalTwinConfigError(f"events[{index}].duration_hours must be positive.")
        _validate_effects(event, f"events[{index}]", supported_effect_fields)

    actions = _require_list(root.get("actions"), "actions")
    _validate_unique_rows(actions, id_field="id", label="actions")
    for index, row in enumerate(actions):
        action = _require_mapping(row, f"actions[{index}]")
        _require_text(action.get("label"), f"actions[{index}].label")
        _require_list(action.get("permissions"), f"actions[{index}].permissions")
        _validate_effects(action, f"actions[{index}]", supported_effect_fields)

    branches = _require_mapping(root.get("branches"), "branches")
    _require_list(branches.get("allowed_policies"), "branches.allowed_policies")
    if int(branches.get("max_branches", 0)) <= 0:
        raise DigitalTwinConfigError("branches.max_branches must be positive.")
    if int(branches.get("replications", 0)) <= 0:
        raise DigitalTwinConfigError("branches.replications must be positive.")

    promotion_targets = _require_mapping(root.get("promotion_targets"), "promotion_targets")
    if bool(promotion_targets.get("enabled", False)):
        targets = _require_list(promotion_targets.get("targets"), "promotion_targets.targets")
        if not targets:
            raise DigitalTwinConfigError("promotion_targets.targets cannot be empty when enabled.")
    return root


@lru_cache(maxsize=1)
def load_digital_twin_config() -> dict[str, Any]:
    path = digital_twin_config_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DigitalTwinConfigError(f"Digital twin config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DigitalTwinConfigError(f"Invalid digital twin config JSON: {path}") from exc
    return validate_config_payload(payload)


def validate_digital_twin_config() -> dict[str, Any]:
    return load_digital_twin_config()


def get_event_config(event_key: str) -> dict[str, Any]:
    normalized = str(event_key or "").strip()
    for row in load_digital_twin_config().get("events", []):
        if row.get("id") == normalized:
            return dict(row)
    raise DigitalTwinConfigError(f"Unknown digital twin event: {event_key}")


def get_action_config(action_key: str) -> dict[str, Any]:
    normalized = str(action_key or "").strip()
    for row in load_digital_twin_config().get("actions", []):
        if row.get("id") == normalized:
            return dict(row)
    raise DigitalTwinConfigError(f"Unknown digital twin action: {action_key}")


def public_config_payload() -> dict[str, Any]:
    cfg = load_digital_twin_config()
    return {
        "version": cfg["version"],
        "runtime": cfg.get("runtime", {}),
        "sync": cfg.get("sync", {}),
        "retention": cfg.get("retention", {}),
        "privacy": {
            "redaction_enabled": bool(cfg.get("privacy", {}).get("redaction_enabled", True)),
            "redact_fields": list(cfg.get("privacy", {}).get("redact_fields", [])),
        },
        "events": cfg.get("events", []),
        "actions": [
            {
                "id": row.get("id"),
                "label": row.get("label"),
                "description": row.get("description", ""),
                "permissions": row.get("permissions", []),
                "promotion_eligible": bool(row.get("promotion_eligible", False)),
                "cost": row.get("cost", 0),
                "effects": row.get("effects", {}),
            }
            for row in cfg.get("actions", [])
        ],
        "branches": cfg.get("branches", {}),
        "promotion_targets": {
            "enabled": bool(cfg.get("promotion_targets", {}).get("enabled", False)),
        },
        "state_schema": cfg.get("state_schema", {}),
    }
