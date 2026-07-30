from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("center_loader")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "operational_simulation.json"


def load_operational_simulation_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {"enabled": True, "fallback_to_default_centers": True}
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


_DAY_ORDER = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]


def _parse_time(value: str) -> float | None:
    """Parse 'HH:MM' into decimal hours."""
    value = str(value or "").strip()
    if not value:
        return None
    parts = value.split(":")
    try:
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        return hours + minutes / 60.0
    except (ValueError, IndexError):
        return None


def _parse_hours_window(window: str) -> tuple[float, float] | None:
    """Parse 'HH:MM-HH:MM' into (open, close) decimal hours."""
    window = str(window or "").strip().lower()
    if window in {"fermé", "ferme", "closed", ""}:
        return None
    match = re.search(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})", window)
    if not match:
        return None
    start = _parse_time(match.group(1))
    end = _parse_time(match.group(2))
    if start is None or end is None:
        return None
    return (start, end)


def _first_open_hours(horaires: Any) -> tuple[float, float] | None:
    if not horaires:
        return None
    if isinstance(horaires, dict):
        for day in _DAY_ORDER:
            window = horaires.get(day)
            if window:
                parsed = _parse_hours_window(window)
                if parsed:
                    return parsed
        for window in horaires.values():
            parsed = _parse_hours_window(window)
            if parsed:
                return parsed
    elif isinstance(horaires, (list, tuple)) and len(horaires) >= 2:
        start = _parse_time(horaires[0])
        end = _parse_time(horaires[1])
        if start is not None and end is not None:
            return (start, end)
    return None


def _default_hours_for_role(role: str, config: dict[str, Any]) -> tuple[float, float]:
    defaults = config.get("django_default_hours") or {}
    if role == "hospital":
        return tuple(defaults.get("hospital", [0, 24]))  # type: ignore[return-value]
    if role == "mobile":
        return tuple(defaults.get("mobile", [9, 18]))  # type: ignore[return-value]
    return tuple(defaults.get("blood_bank", [7.25, 19.75]))  # type: ignore[return-value]


def _capacity_from_daily(
    daily: Any,
    role: str,
    config: dict[str, Any],
) -> dict[str, int]:
    try:
        daily_value = max(int(daily), 0)
    except (TypeError, ValueError):
        daily_value = 0

    divisors = config.get("django_capacity_divisors") or {
        "nurses": 30,
        "lab": 60,
        "processing": 90,
    }

    if daily_value > 0:
        return {
            "nurses": max(1, round(daily_value / max(int(divisors.get("nurses", 30)), 1))),
            "lab": max(1, round(daily_value / max(int(divisors.get("lab", 60)), 1))),
            "processing": max(
                1, round(daily_value / max(int(divisors.get("processing", 90)), 1))
            ),
        }

    defaults = config.get("django_default_capacity") or {
        "hospital": {"nurses": 4, "lab": 3, "processing": 2},
        "blood_bank": {"nurses": 4, "lab": 2, "processing": 2},
        "mobile": {"nurses": 2, "lab": 1, "processing": 1},
    }
    return dict(defaults.get(role, defaults.get("blood_bank", {"nurses": 2, "lab": 1, "processing": 1})))


def _map_db_type(db_type: str, config: dict[str, Any]) -> str:
    type_map = config.get("django_center_type_map") or {
        "fixe": "blood_bank",
        "mobile": "mobile",
        "hopital": "hospital",
        "clinique": "hospital",
    }
    normalized = str(db_type or "").strip().lower()
    return type_map.get(normalized) or type_map.get("*", "blood_bank")


def _center_to_config(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    db_type = str(row.get("type") or "")
    role = _map_db_type(db_type, config)
    hours = _first_open_hours(row.get("horaires")) or _default_hours_for_role(role, config)
    lat = row.get("latitude")
    lon = row.get("longitude")
    try:
        lat = float(lat)
    except (TypeError, ValueError):
        lat = 0.0
    try:
        lon = float(lon)
    except (TypeError, ValueError):
        lon = 0.0

    return {
        "name": str(row.get("nom") or row.get("code_centre") or "Centre"),
        "type": role,
        "original_type": db_type,
        "external_id": str(row.get("code_centre") or row.get("id") or ""),
        "region": str(row.get("region") or row.get("ville") or ""),
        "lat": lat,
        "lon": lon,
        "capacity": _capacity_from_daily(row.get("capacite_journaliere"), role, config),
        "hours": hours,
    }


def load_django_centre_configs(
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Load active CentreDon records from Django and convert them to simulation
    centre configs.

    Returns (configs, source). If Django is unavailable or no active centres
    exist, returns ([], None).
    """
    config = config or load_operational_simulation_config()
    if not config.get("use_django_centers", True):
        return [], None

    try:
        from .snapshot_layer import get_django_apps

        apps = get_django_apps()
        CentreDon = apps.get_model("centres", "CentreDon")  # noqa: N806
    except Exception as exc:
        logger.info("Django centre model unavailable: %s", exc)
        return [], None

    try:
        queryset = CentreDon.objects.filter(statut="actif").order_by("code_centre")
        max_centers = int(config.get("max_django_centers") or 50)
        rows = list(queryset.values()[:max_centers])
    except Exception as exc:
        logger.warning("Failed to query CentreDon: %s", exc)
        return [], None

    if not rows:
        return [], None

    configs = [_center_to_config(row, config) for row in rows]
    return configs, "django_centres"


def _import_build_operational_seed_plan() -> Any:
    from .operational_seed import build_operational_seed_plan

    return build_operational_seed_plan


def resolve_center_configs(
    config: dict[str, Any] | None = None,
    operational_payload: dict[str, Any] | None = None,
    _build_operational_seed_plan: Any = None,
) -> tuple[list[dict[str, Any]] | None, str]:
    """
    Decide which centre configs to use for a simulation.

    Priority:
      1. Operational dashboard payload (hospitals + blood supplies) if enabled.
      2. Django CentreDon records if enabled.
      3. Hardcoded CENTER_CONFIGS fallback if allowed.

    Returns (center_configs, source). source values: 'operational_rows',
    'django_centres', 'default_centers', or 'unavailable'.
    """
    config = config or load_operational_simulation_config()

    if config.get("enabled", True) and operational_payload:
        build_fn = _build_operational_seed_plan or _import_build_operational_seed_plan()
        seed_plan = build_fn(operational_payload, config=config)
        if seed_plan and seed_plan.enabled:
            return seed_plan.center_configs, "operational_rows"

    django_configs, django_source = load_django_centre_configs(config)
    if django_configs:
        return django_configs, django_source or "django_centres"

    if config.get("fallback_to_default_centers", True):
        return None, "default_centers"

    return None, "unavailable"
