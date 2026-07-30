from __future__ import annotations

from typing import Any

from .operational_world import OperationalWorld, get_operational_world


def supply_entities(world: OperationalWorld | None = None) -> list[dict[str, Any]]:
    active = world or get_operational_world()
    return list(active.blood_supplies)


def donor_entities(world: OperationalWorld | None = None) -> list[dict[str, Any]]:
    active = world or get_operational_world()
    return list(active.donors)


def resolve_standalone_entities(
    entity_source_path: str,
    *,
    entity_source_map: dict[str, str],
    world: OperationalWorld | None = None,
) -> list[Any]:
    from .standalone_predictions_config import import_callable_path

    resolver_path = entity_source_map.get(entity_source_path)
    if not resolver_path:
        raise RuntimeError(f"No standalone entity resolver for: {entity_source_path}")
    resolver = import_callable_path(resolver_path)
    return list(resolver(world or get_operational_world()))