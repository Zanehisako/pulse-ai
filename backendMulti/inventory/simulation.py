"""
Simulation engine for advancing the demo world state between prediction runs.

Each call to `advance_simulation_tick()` mutates BloodSupply, HospitalSupplyFeature,
and Donor rows so that consecutive prediction runs produce different ML outputs and
trigger different alerts — making the dashboard feel live without real hospital data.

All parameters are read from ml/config/simulation_config.json (configurable via
the PIOS_SIMULATION_CONFIG env var / settings.PIOS_SIMULATION_CONFIG_PATH).
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)


# ── Config ──────────────────────────────────────────────────────────────────

def _load_simulation_config() -> dict[str, Any]:
    path = Path(settings.PIOS_SIMULATION_CONFIG_PATH)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Simulation config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid simulation config JSON: {path}") from exc
    _validate_simulation_config(payload)
    return payload


def _validate_simulation_config(cfg: dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise RuntimeError("Simulation config root must be an object.")
    for top_key in ("enabled", "tick_interval_minutes", "blood_supply", "hospital_features", "donors"):
        if top_key not in cfg:
            raise RuntimeError(f"Simulation config missing required field: '{top_key}'.")

    bs = cfg["blood_supply"]
    for key in ("consumption_variance_pct", "restock_probability", "restock_amount_min",
                "restock_amount_max", "min_stock_floor"):
        if key not in bs:
            raise RuntimeError(f"Simulation config blood_supply missing '{key}'.")

    hf = cfg["hospital_features"]
    for key in ("temperature_drift_max", "rain_mm_max",
                "scheduled_surgeries_variance_pct", "trauma_cases_variance_pct"):
        if key not in hf:
            raise RuntimeError(f"Simulation config hospital_features missing '{key}'.")

    donors = cfg["donors"]
    for key in ("recency_increment", "availability_churn_rate", "availability_recovery_rate"):
        if key not in donors:
            raise RuntimeError(f"Simulation config donors missing '{key}'.")


def get_tick_interval_minutes() -> int:
    """Return tick interval in minutes.

    Priority: PIOS_SIMULATION_TICK_MINUTES env var → simulation_config.json → 10.
    """
    env_val = os.environ.get("PIOS_SIMULATION_TICK_MINUTES", "").strip()
    if env_val:
        return int(env_val)
    return int(_load_simulation_config().get("tick_interval_minutes", 10))


def is_simulation_enabled() -> bool:
    """Return whether simulation is enabled.

    Priority: PIOS_SIMULATION_ENABLED env var ("1"/"0") → simulation_config.json → False.
    """
    env_val = os.environ.get("PIOS_SIMULATION_ENABLED", "").strip()
    if env_val != "":
        return env_val == "1"
    return bool(_load_simulation_config().get("enabled", False))


# ── Blood supply tick ────────────────────────────────────────────────────────

def _tick_blood_supply(cfg: dict[str, Any], rng: random.Random) -> dict[str, int]:
    from django.utils import timezone
    from inventory.models import BloodSupply
 
    bs_cfg = cfg["blood_supply"]
    variance_pct  = float(bs_cfg["consumption_variance_pct"])
    restock_prob  = float(bs_cfg["restock_probability"])
    restock_min   = float(bs_cfg["restock_amount_min"])
    restock_max   = float(bs_cfg["restock_amount_max"])
    min_floor     = float(bs_cfg["min_stock_floor"])
    snapshot_retention_days = int(bs_cfg.get("snapshot_retention_days", 90))
 
    supplies = list(BloodSupply.objects.select_related("hospital").all())
    if not supplies:
        return {"updated": 0, "restocked": 0, "stockouts": 0}
 
    now = timezone.now()  # FIX 3: capture tick timestamp
    updated = restocked = stockouts = 0
 
    for supply in supplies:
        usage_delta   = supply.usage_today * variance_pct
        varied_usage  = max(0.0, supply.usage_today + rng.uniform(-usage_delta, usage_delta))
        new_stock     = supply.current_stock_units - varied_usage
        new_days_restock = supply.days_since_last_restock + 1
 
        if rng.random() < restock_prob:
            new_stock += rng.uniform(restock_min, restock_max)
            new_days_restock = 0
            restocked += 1
 
        new_stock = max(min_floor, new_stock)
 
        if new_stock <= min_floor and supply.current_stock_units > min_floor:
            supply.stockout_count_90d += 1
            stockouts += 1
 
        supply.current_stock_units   = round(new_stock, 2)
        supply.usage_today           = round(varied_usage, 2)
        supply.days_since_last_restock = new_days_restock
        supply.event_timestamp       = now  # FIX 3: advance timestamp each tick
        updated += 1
 
    BloodSupply.objects.bulk_update(
        supplies,
        [
            "current_stock_units",
            "usage_today",
            "days_since_last_restock",
            "stockout_count_90d",
            "event_timestamp",  # FIX 3
        ],
    )

    # Append a snapshot row for each supply so trend history accumulates
    from inventory.models import BloodSupplySnapshot
    BloodSupplySnapshot.objects.bulk_create([
        BloodSupplySnapshot(
            hospital=s.hospital,
            blood_product_type=s.blood_product_type,
            current_stock_units=s.current_stock_units,
            usage_today=s.usage_today,
            days_since_last_restock=s.days_since_last_restock,
            recorded_at=now,
        )
        for s in supplies
    ])

    # Prune snapshots older than snapshot_retention_days to cap table growth
    from datetime import timedelta
    BloodSupplySnapshot.objects.filter(
        recorded_at__lt=now - timedelta(days=snapshot_retention_days)
    ).delete()

    return {"updated": updated, "restocked": restocked, "stockouts": stockouts}


# ── Hospital feature tick ────────────────────────────────────────────────────

def _tick_hospital_features(cfg: dict[str, Any], rng: random.Random) -> dict[str, int]:
    from django.db.models import Sum
    from inventory.models import BloodSupply, HospitalSupplyFeature

    hf_cfg = cfg["hospital_features"]
    temp_drift = float(hf_cfg["temperature_drift_max"])
    rain_max = float(hf_cfg["rain_mm_max"])
    surgery_variance = float(hf_cfg["scheduled_surgeries_variance_pct"])
    trauma_variance = float(hf_cfg["trauma_cases_variance_pct"])

    features = list(HospitalSupplyFeature.objects.all())
    if not features:
        return {"updated": 0}

    # Sync current_inventory from actual stock after the supply tick
    stock_by_hospital: dict[str, float] = dict(
        BloodSupply.objects
        .values("hospital__hospital_id")
        .annotate(total=Sum("current_stock_units"))
        .values_list("hospital__hospital_id", "total")
    )

    for feature in features:
        # Temperature: random walk clamped to realistic Quebec range
        feature.temperature = round(
            max(-30.0, min(35.0, feature.temperature + rng.uniform(-temp_drift, temp_drift))), 1
        )

        # Rain: Gaussian weighted towards 0, always non-negative
        feature.rain_mm = round(max(0.0, rng.gauss(0.0, rain_max / 3.0)), 1)

        # Scheduled surgeries: integer variance, floor 0
        surgery_delta = max(0, round(feature.scheduled_surgeries * surgery_variance))
        feature.scheduled_surgeries = max(
            0, feature.scheduled_surgeries + rng.randint(-surgery_delta, surgery_delta)
        )

        # Trauma cases: integer variance, floor 0
        trauma_delta = max(0, round(feature.trauma_cases * trauma_variance))
        feature.trauma_cases = max(
            0, feature.trauma_cases + rng.randint(-trauma_delta, trauma_delta)
        )

        # Mirror updated stock total into current_inventory
        total = stock_by_hospital.get(feature.hospital_id)
        if total is not None:
            feature.current_inventory = max(0, round(total))

    HospitalSupplyFeature.objects.bulk_update(
        features,
        ["temperature", "rain_mm", "scheduled_surgeries", "trauma_cases", "current_inventory"],
    )
    return {"updated": len(features)}


# ── Donor tick ───────────────────────────────────────────────────────────────

def _tick_donors(cfg: dict[str, Any], rng: random.Random) -> dict[str, int]:
    from inventory.models import Donor

    donor_cfg = cfg["donors"]
    recency_increment = int(donor_cfg["recency_increment"])
    churn_rate = float(donor_cfg["availability_churn_rate"])
    recovery_rate = float(donor_cfg["availability_recovery_rate"])

    donors = list(Donor.objects.all())
    if not donors:
        return {"updated": 0, "churned": 0, "recovered": 0}

    churned = recovered = 0

    for donor in donors:
        donor.recency_days += recency_increment
        donor.days_until_eligible = max(0, donor.days_until_eligible - recency_increment)

        if donor.availability == 1 and rng.random() < churn_rate:
            donor.availability = 0
            churned += 1
        elif donor.availability == 0 and rng.random() < recovery_rate:
            donor.availability = 1
            recovered += 1

    Donor.objects.bulk_update(donors, ["recency_days", "days_until_eligible", "availability"])
    return {"updated": len(donors), "churned": churned, "recovered": recovered}


# ── Public entry point ───────────────────────────────────────────────────────

def advance_simulation_tick() -> dict[str, Any]:
    """
    Advance the simulated world state by one tick.

    Mutates BloodSupply, HospitalSupplyFeature, and Donor rows so that the
    next prediction run produces different ML outputs from the last run.
    Uses a fresh unseeded Random instance so each tick yields distinct results.

    Returns a summary dict suitable for logging and the trigger API endpoint.
    """
    cfg = _load_simulation_config()

    if not bool(cfg.get("enabled", False)):
        logger.info("Simulation tick skipped: disabled in config.")
        return {"status": "disabled"}

    rng = random.Random()  # deliberately unseeded — different output every tick

    supply_summary = _tick_blood_supply(cfg, rng)
    feature_summary = _tick_hospital_features(cfg, rng)
    donor_summary = _tick_donors(cfg, rng)

    summary: dict[str, Any] = {
        "status": "ok",
        "blood_supply": supply_summary,
        "hospital_features": feature_summary,
        "donors": donor_summary,
    }
    logger.info("Simulation tick complete: %s", summary)
    return summary
