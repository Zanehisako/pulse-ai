from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_DIR = Path(
    os.getenv(
        "PIOS_ML_TRAINING_CONFIG_DIR",
        str(Path(__file__).resolve().parents[1] / "config"),
    )
)

DEFAULT_TOOL_REGISTRY_POLICY_FILE = "tool_registry_policy.json"




def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def _text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required.")
    return text


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _positive_number(value: Any, label: str) -> float:
    number = _number(value, label)
    if number <= 0:
        raise ValueError(f"{label} must be positive.")
    return number


def _required_string_list(value: Any, label: str) -> list[str]:
    rows = _list(value, label)
    values = [str(item).strip() for item in rows if str(item).strip()]
    if not values:
        raise ValueError(f"{label} must contain at least one value.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates.")
    return values


def _string_list(value: Any, label: str) -> list[str]:
    rows = _list(value, label)
    values = [str(item).strip() for item in rows if str(item).strip()]
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates.")
    return values


def load_tool_registry_policy(config_dir: Path | None = None) -> dict[str, Any]:
    path = (config_dir or DEFAULT_CONFIG_DIR) / DEFAULT_TOOL_REGISTRY_POLICY_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read tool registry policy: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON tool registry policy: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("tool registry policy root must be an object.")
    return payload


def validate_tool_registry_config(
    config: dict[str, Any],
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = _mapping(config, "tool config")
    policy_row = policy or load_tool_registry_policy()
    required_fields = _required_string_list(
        policy_row.get("required_fields", []),
        "tool registry policy.required_fields",
    )
    for field in required_fields:
        if field not in row:
            raise ValueError(f"{row.get('id', '<unknown>')}: missing required field {field}.")

    tool_id = _text(row.get("id"), "id")
    _text(row.get("name"), f"{tool_id}.name")
    _text(row.get("description"), f"{tool_id}.description")
    if not isinstance(row.get("enabled"), bool):
        raise ValueError(f"{tool_id}.enabled must be boolean.")
    _mapping(row.get("input_schema"), f"{tool_id}.input_schema")
    _mapping(row.get("output_schema"), f"{tool_id}.output_schema")
    _text(row.get("entrypoint"), f"{tool_id}.entrypoint")
    _positive_number(row.get("timeout_seconds"), f"{tool_id}.timeout_seconds")

    supported_types = set(_required_string_list(
        policy_row.get("supported_types", []),
        "tool registry policy.supported_types",
    ))
    tool_type = _text(row.get("type"), f"{tool_id}.type")
    if tool_type not in supported_types:
        raise ValueError(f"{tool_id}.type is not supported: {tool_type}.")

    supported_adapters = set(_required_string_list(
        policy_row.get("supported_adapters", []),
        "tool registry policy.supported_adapters",
    ))
    adapter = _text(row.get("adapter"), f"{tool_id}.adapter")
    if adapter not in supported_adapters:
        raise ValueError(f"{tool_id}.adapter is not supported: {adapter}.")

    supported_permissions = set(_required_string_list(
        policy_row.get("supported_permissions", []),
        "tool registry policy.supported_permissions",
    ))
    permissions = _required_string_list(row.get("permissions"), f"{tool_id}.permissions")
    unsupported = sorted(set(permissions) - supported_permissions)
    if unsupported:
        raise ValueError(f"{tool_id}.permissions contains unsupported values: {unsupported}.")
    return row


def validate_tool_registry_configs(
    configs: list[dict[str, Any]],
    *,
    policy: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    policy_row = policy or load_tool_registry_policy()
    validated = [
        validate_tool_registry_config(config, policy=policy_row)
        for config in configs
    ]
    ids = [str(row["id"]) for row in validated]
    duplicates = sorted({tool_id for tool_id in ids if ids.count(tool_id) > 1})
    if duplicates:
        raise ValueError(f"tool ids must be unique: {duplicates}.")
    return validated








def load_model_architecture_config(
    filename: str,
    *,
    validator: Any,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    path = (config_dir or DEFAULT_CONFIG_DIR) / filename
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read model config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON model config: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("model config root must be an object.")
    return validator(payload)


