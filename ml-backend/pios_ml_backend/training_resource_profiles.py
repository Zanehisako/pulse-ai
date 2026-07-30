from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, MutableMapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_PROFILE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "training_resource_profiles.json"
)
DEFAULT_SELECTION_ENV_VAR = "PIOS_TRAINING_PROFILE"


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return value


def _string_mapping(value: Any, label: str) -> dict[str, str]:
    row = _mapping(value, label)
    return {str(key): str(raw_value) for key, raw_value in row.items()}


def _string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return [str(item).strip() for item in value if str(item).strip()]


def _int_list(value: Any, label: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    rows: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{label} values must be integers.")
        try:
            rows.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} values must be integers.") from exc
    return rows


def load_training_resource_profiles(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_PROFILE_CONFIG_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read training resource profiles: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON training resource profiles: {path}") from exc
    root = _mapping(payload, "training resource profiles")
    selection_env_var = str(root.get("selection_env_var") or DEFAULT_SELECTION_ENV_VAR).strip()
    if not selection_env_var:
        raise ValueError("training resource profiles.selection_env_var is required.")
    default_profile = str(root.get("default_profile") or "default").strip()
    if not default_profile:
        raise ValueError("training resource profiles.default_profile is required.")
    profiles = _mapping(root.get("profiles"), "training resource profiles.profiles")
    if default_profile not in profiles:
        raise ValueError(f"default training resource profile is missing: {default_profile}.")
    for name, raw_profile in profiles.items():
        label = f"training resource profile {name}"
        profile = _mapping(raw_profile, label)
        if not isinstance(profile.get("enabled", True), bool):
            raise ValueError(f"{label}.enabled must be boolean.")
        _string_mapping(profile.get("environment", {}), f"{label}.environment")
        _int_list(profile.get("retry_on_exit_codes", []), f"{label}.retry_on_exit_codes")
        _string_list(profile.get("retry_profiles", []), f"{label}.retry_profiles")
        _mapping(profile.get("training_overrides", {}), f"{label}.training_overrides")
    return {
        **root,
        "selection_env_var": selection_env_var,
        "default_profile": default_profile,
        "profiles": profiles,
    }


def selected_training_profile_name(
    *,
    config_path: str | Path | None = None,
    env: MutableMapping[str, str] | None = None,
) -> str:
    config = load_training_resource_profiles(config_path)
    source_env = env if env is not None else os.environ
    selected = str(source_env.get(config["selection_env_var"], "")).strip()
    return selected or str(config["default_profile"])


def resolve_training_resource_profile(
    profile_name: str | None = None,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_training_resource_profiles(config_path)
    name = str(profile_name or selected_training_profile_name(config_path=config_path)).strip()
    profiles = config["profiles"]
    if name not in profiles:
        raise ValueError(f"Unknown training resource profile: {name}.")
    resolving: set[str] = set()

    def resolve_one(current_name: str) -> dict[str, Any]:
        if current_name in resolving:
            raise ValueError(f"Training resource profile inheritance cycle: {current_name}.")
        if current_name not in profiles:
            raise ValueError(f"Unknown inherited training resource profile: {current_name}.")
        resolving.add(current_name)
        current = dict(_mapping(profiles[current_name], f"training resource profile {current_name}"))
        parent_name = str(current.pop("inherits", "") or "").strip()
        if parent_name:
            parent = resolve_one(parent_name)
            current = deep_merge_dicts(parent, current)
        resolving.remove(current_name)
        current["name"] = current_name
        return current

    profile = resolve_one(name)
    if profile.get("enabled", True) is not True:
        raise ValueError(f"Training resource profile is disabled: {name}.")
    profile["environment"] = _string_mapping(
        profile.get("environment", {}),
        f"training resource profile {name}.environment",
    )
    profile["retry_on_exit_codes"] = _int_list(
        profile.get("retry_on_exit_codes", []),
        f"training resource profile {name}.retry_on_exit_codes",
    )
    profile["retry_profiles"] = _string_list(
        profile.get("retry_profiles", []),
        f"training resource profile {name}.retry_profiles",
    )
    profile["training_overrides"] = _mapping(
        profile.get("training_overrides", {}),
        f"training resource profile {name}.training_overrides",
    )
    return profile


def profile_environment(
    profile_name: str | None = None,
    *,
    config_path: str | Path | None = None,
) -> dict[str, str]:
    profile = resolve_training_resource_profile(profile_name, config_path=config_path)
    return dict(profile.get("environment", {}))


def apply_profile_environment(
    env: MutableMapping[str, str],
    profile_name: str | None = None,
    *,
    config_path: str | Path | None = None,
) -> MutableMapping[str, str]:
    config = load_training_resource_profiles(config_path)
    resolved_name = str(profile_name or selected_training_profile_name(
        config_path=config_path,
        env=env,
    )).strip()
    profile = resolve_training_resource_profile(resolved_name, config_path=config_path)
    env[config["selection_env_var"]] = str(profile["name"])
    for key, value in profile_environment(str(profile["name"]), config_path=config_path).items():
        env[str(key)] = str(value)
    return env


def apply_selected_profile_environment(
    env: MutableMapping[str, str] | None = None,
    *,
    config_path: str | Path | None = None,
) -> MutableMapping[str, str]:
    target_env = env if env is not None else os.environ
    return apply_profile_environment(target_env, config_path=config_path)


def _profile_training_overrides_for_model(
    profile: dict[str, Any],
    *,
    model_id: str,
    registered_model_name: str,
) -> dict[str, Any]:
    overrides = profile.get("training_overrides", {})
    if not isinstance(overrides, dict):
        return {}
    merged: dict[str, Any] = {}
    for key in ("*", model_id, registered_model_name):
        row = overrides.get(key)
        if isinstance(row, dict):
            merged = deep_merge_dicts(merged, row)
    return merged


def apply_profile_to_model_config(
    config: dict[str, Any],
    profile_name: str | None = None,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    row = dict(config)
    profile = resolve_training_resource_profile(profile_name, config_path=config_path)
    model_id = str(row.get("id") or "").strip()
    registered_model_name = str(row.get("registered_model_name") or "").strip()
    overrides = _profile_training_overrides_for_model(
        profile,
        model_id=model_id,
        registered_model_name=registered_model_name,
    )
    if overrides:
        row["training"] = deep_merge_dicts(
            dict(row.get("training", {})),
            overrides,
        )
    if isinstance(row.get("training"), dict):
        resource = dict(row["training"].get("resource", {}))
        resource["profile_name"] = str(profile["name"])
        row["training"] = {**row["training"], "resource": resource}
    return row


def nested_bool(config: dict[str, Any], path: Sequence[str], default: bool) -> bool:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def run_subprocess_with_profile_retries(
    args: Sequence[str],
    *,
    env: MutableMapping[str, str],
    profile_name: str | None = None,
    config_path: str | Path | None = None,
    cwd: str | Path | None = None,
    on_retry: Callable[[subprocess.CompletedProcess[Any], str], None] | None = None,
) -> subprocess.CompletedProcess[Any]:
    active_profile_name = str(
        profile_name or selected_training_profile_name(config_path=config_path, env=env)
    )
    attempted_profiles: set[str] = set()
    while True:
        attempted_profiles.add(active_profile_name)
        attempt_env = dict(env)
        apply_profile_environment(
            attempt_env,
            active_profile_name,
            config_path=config_path,
        )
        result = subprocess.run(
            list(args),
            env=attempt_env,
            cwd=str(cwd) if cwd is not None else None,
        )
        profile = resolve_training_resource_profile(
            active_profile_name,
            config_path=config_path,
        )
        retry_codes = set(profile.get("retry_on_exit_codes", []))
        retry_profiles = [
            name for name in profile.get("retry_profiles", [])
            if name not in attempted_profiles
        ]
        if result.returncode not in retry_codes or not retry_profiles:
            return result
        active_profile_name = retry_profiles[0]
        if on_retry is not None:
            on_retry(result, active_profile_name)
