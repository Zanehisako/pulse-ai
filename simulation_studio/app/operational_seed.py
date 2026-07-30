from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import CENTER_CONFIGS, BloodUnit, DonationCenter  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "operational_simulation.json"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_operational_simulation_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {"enabled": False}
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _group_supplies_by_hospital(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        hospital_id = str(row.get("hospital_id") or "").strip()
        if not hospital_id:
            continue
        grouped.setdefault(hospital_id, []).append(row)
    return grouped


def _hospital_usage_total(rows: list[dict[str, Any]]) -> float:
    return sum(max(_as_float(row.get("usage_today")), 0.0) for row in rows)


def _slot_lat_lon(index: int, total: int, bbox: dict[str, float]) -> tuple[float, float]:
    if total <= 1:
        lat = (bbox["north"] + bbox["south"]) / 2.0
        lon = (bbox["east"] + bbox["west"]) / 2.0
        return lat, lon
    ratio = index / max(total - 1, 1)
    lat = bbox["south"] + (bbox["north"] - bbox["south"]) * (0.18 + ratio * 0.64)
    lon = bbox["west"] + (bbox["east"] - bbox["west"]) * (0.12 + ratio * 0.76)
    return round(lat, 6), round(lon, 6)


def _scaled_hospital_capacity(
    usage_total: float,
    *,
    base_capacity: dict[str, int],
    config: dict[str, Any],
) -> dict[str, int]:
    if not config.get("demand_weight_from_usage", True):
        return dict(base_capacity)
    baseline = max(_as_float(config.get("usage_baseline_per_day"), 8.0), 1.0)
    weight = usage_total / baseline if usage_total > 0 else 1.0
    floor = _as_float(config.get("usage_demand_weight_floor"), 0.5)
    cap = _as_float(config.get("usage_demand_weight_cap"), 2.5)
    weight = max(floor, min(cap, weight))
    return {
        key: max(1, int(round(value * weight)))
        for key, value in base_capacity.items()
    }


def _blood_bank_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    if not config.get("include_blood_banks", True):
        return []
    allowed = {
        str(name).strip()
        for name in config.get("blood_bank_center_names") or []
        if str(name).strip()
    }
    banks: list[dict[str, Any]] = []
    for entry in CENTER_CONFIGS:
        if entry.get("type") != "blood_bank":
            continue
        if allowed and entry.get("name") not in allowed:
            continue
        banks.append(dict(entry))
    return banks


@dataclass
class OperationalSeedPlan:
    center_configs: list[dict[str, Any]]
    supplies: list[dict[str, Any]]
    hospital_count: int
    source: str

    @property
    def enabled(self) -> bool:
        return self.hospital_count > 0 and bool(self.center_configs)


def build_operational_seed_plan(
    payload: dict[str, Any] | None,
    *,
    config: dict[str, Any] | None = None,
) -> OperationalSeedPlan | None:
    config = config or load_operational_simulation_config()
    if not config.get("enabled", True):
        return None

    payload = payload or {}
    hospitals = list(payload.get("hospitals") or [])
    supplies = list(payload.get("blood_supplies") or [])

    if not hospitals and supplies:
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

    if not hospitals:
        return None

    max_hospitals = int(config.get("max_hospitals") or 12)
    hospitals = hospitals[:max_hospitals]
    supplies_by_hospital = _group_supplies_by_hospital(supplies)
    bbox = config.get("geo_bbox") or {}
    base_capacity = dict(config.get("hospital_capacity") or {})
    hours = tuple(config.get("hospital_hours") or [0, 24])

    hospital_configs: list[dict[str, Any]] = []
    for index, hospital in enumerate(hospitals):
        hospital_id = str(hospital.get("hospital_id") or f"hospital-{index + 1}")
        hospital_rows = supplies_by_hospital.get(hospital_id, [])
        usage_total = _hospital_usage_total(hospital_rows)
        lat, lon = _slot_lat_lon(index, len(hospitals), bbox)
        hospital_configs.append(
            {
                "name": str(hospital.get("name") or hospital_id),
                "type": "hospital",
                "external_id": hospital_id,
                "region": str(hospital.get("wilaya") or ""),
                "lat": lat,
                "lon": lon,
                "capacity": _scaled_hospital_capacity(
                    usage_total,
                    base_capacity=base_capacity,
                    config=config,
                ),
                "hours": hours,
                "initial_usage_today": round(usage_total, 2),
            }
        )

    center_configs = hospital_configs + _blood_bank_configs(config)
    if len(hospital_configs) < int(config.get("min_hospitals") or 1):
        return None

    return OperationalSeedPlan(
        center_configs=center_configs,
        supplies=supplies,
        hospital_count=len(hospital_configs),
        source="operational_rows",
    )


def apply_strategy_to_center_config(config: dict[str, Any], strategy: Any) -> dict[str, Any]:
    cfg = dict(config)
    cfg["capacity"] = dict(config.get("capacity") or {})
    cfg["original_type"] = config.get("original_type") or config.get("type")
    cfg["capacity"]["nurses"] = int(cfg["capacity"].get("nurses", 1)) + int(
        getattr(strategy, "extra_nurses", 0) or 0
    )
    cfg["capacity"]["lab"] = int(cfg["capacity"].get("lab", 1)) + int(
        getattr(strategy, "extra_lab_staff", 0) or 0
    )
    cfg["capacity"]["processing"] = int(cfg["capacity"].get("processing", 1)) + int(
        getattr(strategy, "extra_processing_staff", 0) or 0
    )
    if getattr(strategy, "hours_extension_h", 0) and cfg.get("type") != "hospital":
        start, end = cfg.get("hours") or (0, 24)
        extension = float(strategy.hours_extension_h)
        cfg["hours"] = (max(float(start) - extension, 0), min(float(end) + extension, 24))
    return cfg


def build_simulation_centers(
    env: Any,
    strategy: Any,
    *,
    center_configs: list[dict[str, Any]] | None = None,
) -> list[DonationCenter]:
    centers: list[DonationCenter] = []
    configs = center_configs or list(CENTER_CONFIGS)
    for config in configs:
        centers.append(DonationCenter(env, apply_strategy_to_center_config(config, strategy)))
    if center_configs is not None:
        attach_external_ids(centers, center_configs)

    include_mobile = center_configs is None or bool(
        load_operational_simulation_config().get("include_mobile_from_strategy", True)
    )
    if include_mobile and int(getattr(strategy, "mobile_units", 0) or 0) > 0:
        for idx in range(int(strategy.mobile_units)):
            start_h, end_h = 9.0, 18.0
            if getattr(strategy, "hours_extension_h", 0):
                start_h = max(start_h - float(strategy.hours_extension_h), 0)
                end_h = min(end_h + float(strategy.hours_extension_h), 24)
            extra_cfg = {
                "name": f"Mobile Surge Unit #{idx + 1}",
                "type": "mobile",
                "lat": [46.793, 46.817, 46.846][idx % 3],
                "lon": [-71.262, -71.229, -71.287][idx % 3],
                "capacity": {
                    "nurses": 2 + int(getattr(strategy, "extra_nurses", 0) or 0),
                    "lab": 1 + int(getattr(strategy, "extra_lab_staff", 0) or 0),
                    "processing": 1 + int(getattr(strategy, "extra_processing_staff", 0) or 0),
                },
                "hours": (start_h, end_h),
            }
            centers.append(DonationCenter(env, extra_cfg))
    return centers


def _split_units(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {key: 0 for key in weights}
    keys = list(weights.keys())
    raw = {key: total * float(weights[key]) for key in keys}
    allocated = {key: int(raw[key]) for key in keys}
    remainder = total - sum(allocated.values())
    order = sorted(keys, key=lambda key: raw[key] - allocated[key], reverse=True)
    index = 0
    while remainder > 0 and order:
        allocated[order[index % len(order)]] += 1
        remainder -= 1
        index += 1
    return allocated


def prestock_centers_from_operational(
    centers: list[DonationCenter],
    env: Any,
    supplies: list[dict[str, Any]],
    *,
    config: dict[str, Any] | None = None,
) -> bool:
    config = config or load_operational_simulation_config()
    component_split = dict(config.get("component_split") or {})
    if not component_split:
        component_split = {"RBC": 0.58, "PLATELETS": 0.27, "PLASMA": 0.15}

    supplies_by_hospital = _group_supplies_by_hospital(supplies)
    seeded_any = False

    for center in centers:
        if center.ctype != "hospital":
            continue
        external_id = getattr(center, "external_id", None)
        if not external_id:
            continue
        rows = supplies_by_hospital.get(str(external_id), [])
        center.inventory = []
        center._inventory_counts = {"RBC": 0, "PLATELETS": 0, "PLASMA": 0}

        for row in rows:
            blood_type = str(row.get("blood_product_type") or "O+").strip() or "O+"
            stock_units = max(int(round(_as_float(row.get("current_stock_units")))), 0)
            if stock_units <= 0:
                continue
            component_units = _split_units(stock_units, component_split)
            for component, units in component_units.items():
                for _ in range(units):
                    unit = BloodUnit(env, blood_type, component, center.name)
                    shelf_life = unit.expiry - env.now
                    age_fraction = 0.12 if component == "RBC" else 0.08
                    unit.collection_time = -random.uniform(0, max(shelf_life * age_fraction, 0.1))
                    unit.expiry = unit.collection_time + shelf_life
                    center.add_unit(unit)
            seeded_any = True

        usage_total = _hospital_usage_total(rows)
        if usage_total > 0:
            center.stats["orders"] = max(int(round(usage_total * 0.25)), 0)

    if not seeded_any:
        return False

    for center in centers:
        if center.ctype != "blood_bank":
            continue
        center.inventory = []
        center._inventory_counts = {"RBC": 0, "PLATELETS": 0, "PLASMA": 0}

    from engine import prestock_centers  # noqa: E402

    bank_centers = [center for center in centers if center.ctype == "blood_bank"]
    if bank_centers:
        prestock_centers(bank_centers, env, inventory_days=2.0)
    return True


def attach_external_ids(centers: list[DonationCenter], configs: list[dict[str, Any]]) -> None:
    config_by_name = {str(cfg.get("name")): cfg for cfg in configs}
    for center in centers:
        cfg = config_by_name.get(center.name, {})
        external_id = cfg.get("external_id")
        if external_id:
            center.external_id = str(external_id)
        initial_usage = cfg.get("initial_usage_today")
        if initial_usage is not None:
            center.initial_usage_today = float(initial_usage)