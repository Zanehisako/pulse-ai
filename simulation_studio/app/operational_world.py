from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .standalone_predictions_config import load_standalone_predictions_config, ml_config_root

BLOOD_TYPES = ("O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-")

# ABO/Rh demand mix, mirrored from simulator/core.py BLOOD_DISTRIBUTION so the
# lightweight forecast path avoids importing the heavy simulator package
# (networkx/osmnx/simpy). Shares sum to 1.0.
BLOOD_DISTRIBUTION = {
    "O+": 0.44,
    "A+": 0.30,
    "B+": 0.12,
    "AB+": 0.05,
    "O-": 0.04,
    "A-": 0.03,
    "B-": 0.015,
    "AB-": 0.005,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class OperationalWorld:
    hospitals: list[dict[str, Any]] = field(default_factory=list)
    blood_supplies: list[dict[str, Any]] = field(default_factory=list)
    supply_snapshots: list[dict[str, Any]] = field(default_factory=list)
    hospital_features: list[dict[str, Any]] = field(default_factory=list)
    donors: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    model_configs: list[dict[str, Any]] = field(default_factory=list)
    drift_reports: list[dict[str, Any]] = field(default_factory=list)

    def normalized_payload(self) -> dict[str, Any]:
        return {
            "hospitals": deepcopy(self.hospitals),
            "blood_supplies": deepcopy(self.blood_supplies),
            "supply_snapshots": deepcopy(self.supply_snapshots),
            "hospital_features": deepcopy(self.hospital_features),
            "donors": deepcopy(self.donors),
            "predictions": deepcopy(self.predictions),
            "alerts": deepcopy(self.alerts),
            "model_configs": deepcopy(self.model_configs),
            "drift_reports": deepcopy(self.drift_reports),
        }


_WORLD: OperationalWorld | None = None


def get_operational_world() -> OperationalWorld:
    global _WORLD
    if _WORLD is None:
        _WORLD = seed_operational_world()
    return _WORLD


def reset_operational_world(world: OperationalWorld | None = None) -> OperationalWorld:
    global _WORLD
    _WORLD = world or seed_operational_world()
    return _WORLD


def _load_simulation_tick_config() -> dict[str, Any]:
    payload = load_standalone_predictions_config()
    root = ml_config_root(payload)
    sim_path = root / str(payload.get("simulation_config_file") or "simulation_config.json")
    return json.loads(sim_path.read_text(encoding="utf-8"))


def seed_operational_world() -> OperationalWorld:
    now = utc_now_iso()
    hospitals = [
        {"hospital_id": "SYN-H001", "name": "Synthetic Hospital A", "wilaya": "North"},
        {"hospital_id": "SYN-H002", "name": "Synthetic Hospital B", "wilaya": "South"},
    ]
    supplies: list[dict[str, Any]] = []
    for hospital in hospitals:
        for index, blood in enumerate(BLOOD_TYPES):
            supplies.append(
                {
                    "supply_id": f"{hospital['hospital_id']}-{blood}",
                    "hospital_id": hospital["hospital_id"],
                    "hospital_name": hospital["name"],
                    "hospital": {
                        "hospital_id": hospital["hospital_id"],
                        "name": hospital["name"],
                    },
                    "blood_product_type": blood,
                    "current_stock_units": 80.0 + index * 6,
                    "usage_today": 8.0 + index * 0.5,
                    "lead_time_days": 2,
                    "days_since_last_restock": 3,
                    "stockout_count_90d": 0,
                    "scheduled_surgeries_next7d": 12,
                    "event_timestamp": now,
                    "updated_at": now,
                }
            )

    hospital_features = [
        {
            "hospital_id": hospital["hospital_id"],
            "event_timestamp": now,
            "temperature": 18.0,
            "rain_mm": 1.2,
            "holiday": 0,
            "disaster": 0,
            "scheduled_surgeries": 24,
            "trauma_cases": 6,
            "current_inventory": sum(
                row["current_stock_units"]
                for row in supplies
                if row["hospital_id"] == hospital["hospital_id"]
            ),
            "updated_at": now,
        }
        for hospital in hospitals
    ]

    donors: list[dict[str, Any]] = []
    for index in range(32):
        donors.append(
            {
                "donor_id": f"DON-{index + 1:04d}",
                "name": f"Donor {index + 1}",
                "wilaya": "Synthetic",
                "event_timestamp": now,
                "city_id": f"C{index % 5}",
                "lat": 46.81 + (index % 7) * 0.01,
                "lon": -71.21 - (index % 5) * 0.01,
                "availability": 1 if index % 4 else 0,
                "blood_group": BLOOD_TYPES[index % len(BLOOD_TYPES)],
                "recency_days": 30 + index,
                "frequency_365": min(index % 5, 4),
                "days_until_eligible": max(0, 14 - (index % 20)),
                "cluster_id": index % 6,
                "updated_at": now,
            }
        )

    return OperationalWorld(
        hospitals=hospitals,
        blood_supplies=supplies,
        hospital_features=hospital_features,
        donors=donors,
    )


def advance_operational_tick(world: OperationalWorld | None = None) -> dict[str, Any]:
    world = world or get_operational_world()
    cfg = _load_simulation_tick_config()
    rng = random.Random()
    now_dt = datetime.now(timezone.utc)
    now = now_dt.replace(microsecond=0).isoformat()

    bs_cfg = cfg["blood_supply"]
    variance_pct = float(bs_cfg["consumption_variance_pct"])
    restock_prob = float(bs_cfg["restock_probability"])
    restock_min = float(bs_cfg["restock_amount_min"])
    restock_max = float(bs_cfg["restock_amount_max"])
    min_floor = float(bs_cfg["min_stock_floor"])
    retention_days = int(bs_cfg.get("snapshot_retention_days", 90))

    restocked = stockouts = 0
    for supply in world.blood_supplies:
        usage_delta = float(supply["usage_today"]) * variance_pct
        varied_usage = max(
            0.0,
            float(supply["usage_today"]) + rng.uniform(-usage_delta, usage_delta),
        )
        new_stock = float(supply["current_stock_units"]) - varied_usage
        new_days_restock = int(supply.get("days_since_last_restock", 0)) + 1

        if rng.random() < restock_prob:
            new_stock += rng.uniform(restock_min, restock_max)
            new_days_restock = 0
            restocked += 1

        if new_stock <= min_floor and float(supply["current_stock_units"]) > min_floor:
            supply["stockout_count_90d"] = int(supply.get("stockout_count_90d", 0)) + 1
            stockouts += 1

        supply["current_stock_units"] = round(max(min_floor, new_stock), 2)
        supply["usage_today"] = round(varied_usage, 2)
        supply["days_since_last_restock"] = new_days_restock
        supply["event_timestamp"] = now
        supply["updated_at"] = now

        world.supply_snapshots.append(
            {
                "hospital_id": supply["hospital_id"],
                "blood_product_type": supply["blood_product_type"],
                "current_stock_units": supply["current_stock_units"],
                "usage_today": supply["usage_today"],
                "days_since_last_restock": supply["days_since_last_restock"],
                "recorded_at": now,
            }
        )

    cutoff = now_dt - timedelta(days=retention_days)
    world.supply_snapshots = [
        row
        for row in world.supply_snapshots
        if datetime.fromisoformat(str(row["recorded_at"]).replace("Z", "+00:00")) >= cutoff
    ]

    stock_by_hospital: dict[str, float] = {}
    for supply in world.blood_supplies:
        stock_by_hospital[supply["hospital_id"]] = stock_by_hospital.get(
            supply["hospital_id"], 0.0
        ) + float(supply["current_stock_units"])

    hf_cfg = cfg["hospital_features"]
    temp_drift = float(hf_cfg["temperature_drift_max"])
    rain_max = float(hf_cfg["rain_mm_max"])
    surgery_variance = float(hf_cfg["scheduled_surgeries_variance_pct"])
    trauma_variance = float(hf_cfg["trauma_cases_variance_pct"])

    for feature in world.hospital_features:
        feature["temperature"] = round(
            max(
                -30.0,
                min(
                    35.0,
                    float(feature["temperature"]) + rng.uniform(-temp_drift, temp_drift),
                ),
            ),
            1,
        )
        feature["rain_mm"] = round(max(0.0, rng.gauss(0.0, rain_max / 3.0)), 1)
        surgery_delta = max(0, round(float(feature["scheduled_surgeries"]) * surgery_variance))
        feature["scheduled_surgeries"] = max(
            0,
            int(feature["scheduled_surgeries"])
            + rng.randint(-surgery_delta, surgery_delta),
        )
        trauma_delta = max(0, round(float(feature["trauma_cases"]) * trauma_variance))
        feature["trauma_cases"] = max(
            0,
            int(feature["trauma_cases"]) + rng.randint(-trauma_delta, trauma_delta),
        )
        total = stock_by_hospital.get(str(feature["hospital_id"]))
        if total is not None:
            feature["current_inventory"] = max(0, round(total))
        feature["event_timestamp"] = now
        feature["updated_at"] = now

    donor_cfg = cfg["donors"]
    recency_increment = int(donor_cfg["recency_increment"])
    churn_rate = float(donor_cfg["availability_churn_rate"])
    recovery_rate = float(donor_cfg["availability_recovery_rate"])
    churned = recovered = 0

    for donor in world.donors:
        donor["recency_days"] = int(donor.get("recency_days", 0)) + recency_increment
        donor["days_until_eligible"] = max(
            0, int(donor.get("days_until_eligible", 0)) - recency_increment
        )
        if int(donor.get("availability", 0)) == 1 and rng.random() < churn_rate:
            donor["availability"] = 0
            churned += 1
        elif int(donor.get("availability", 0)) == 0 and rng.random() < recovery_rate:
            donor["availability"] = 1
            recovered += 1
        donor["event_timestamp"] = now
        donor["updated_at"] = now

    return {
        "blood_supply": {
            "updated": len(world.blood_supplies),
            "restocked": restocked,
            "stockouts": stockouts,
        },
        "hospital_features": {"updated": len(world.hospital_features)},
        "donors": {
            "updated": len(world.donors),
            "churned": churned,
            "recovered": recovered,
        },
    }