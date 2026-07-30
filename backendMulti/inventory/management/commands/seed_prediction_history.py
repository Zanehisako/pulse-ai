"""
Seed deterministic prediction history by combining each supply row's runway
profile with Quebec-specific seasonal pressure modifiers and stable per-row
jitter for trend-sensitive alert demos.

Configuration (model name, history window, alert threshold) is read from
ml/config/scheduled_predictions.json::prediction_history_seed so it stays
aligned with the live prediction pipeline without code changes.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
import random

from django.conf import settings
from django.core.management.base import BaseCommand

from inventory.models import BloodSupply, PredictionResult
from inventory.seed_data import (
    HOSPITALS_BY_ID,
    HOSPITAL_SIZE_PROFILES,
    SEED,
    quebec_seasonality_modifier,
    seeded_random,
)

logger = logging.getLogger(__name__)

SIZE_HISTORY_FACTORS = {
    "large": 1.12,
    "urban": 1.0,
    "regional": 0.9,
    "remote": 0.82,
}

BLOOD_TYPE_HISTORY_FACTORS = {
    "O+": 1.03,
    "A+": 1.0,
    "B+": 0.94,
    "O-": 0.9,
    "A-": 0.87,
    "B-": 0.82,
    "AB+": 0.88,
    "AB-": 0.77,
}

# Defaults used only when the config block is missing or malformed.
_DEFAULT_MODEL_NAME = "stockout_days_predictor"
_DEFAULT_HISTORY_DAYS = 60
_DEFAULT_ALERT_THRESHOLD = 3.0


def _load_history_seed_config() -> dict:
    """Load the prediction_history_seed block from scheduled_predictions.json."""
    config_path = Path(settings.ML_CONFIG_DIR) / "scheduled_predictions.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Could not read prediction_history_seed config from %s: %s — using defaults",
            config_path,
            exc,
        )
        return {}
    block = payload.get("prediction_history_seed")
    if not isinstance(block, dict):
        return {}
    return block


class Command(BaseCommand):
    help = "Seed deterministic prediction history for dynamic thresholds and trend detection."

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true")

    def handle(self, *args, **options):
        random.seed(SEED)

        cfg = _load_history_seed_config()
        model_name = str(cfg.get("model_name") or _DEFAULT_MODEL_NAME)
        history_days = int(cfg.get("history_days") or _DEFAULT_HISTORY_DAYS)
        alert_threshold = float(cfg.get("alert_threshold") or _DEFAULT_ALERT_THRESHOLD)

        reset = options["reset"]
        supplies = list(BloodSupply.objects.select_related("hospital").all())
        today = date.today()
        target_dates = [today - timedelta(days=days_ago) for days_ago in range(1, history_days + 1)]
        target_date_set = set(target_dates)
        supply_ids = [supply.supply_id for supply in supplies]

        if reset:
            # Wipe all PredictionResult rows — not just those matching current
            # supply_ids — to remove orphaned rows left from previous reseeds
            # (old supply_ids no longer in BloodSupply after a --reset run).
            PredictionResult.objects.all().delete()

        existing_history: dict[str, set[date]] = {}
        if supply_ids:
            existing_rows = PredictionResult.objects.filter(
                entity_id__in=supply_ids,
                model_name=model_name,
                predicted_for_date__in=target_dates,
            ).values_list("entity_id", "predicted_for_date")
            for entity_id, predicted_for_date in existing_rows:
                existing_history.setdefault(entity_id, set()).add(predicted_for_date)

        rows_to_create: list[PredictionResult] = []

        for supply in supplies:
            existing_dates = existing_history.get(supply.supply_id, set())
            if existing_dates >= target_date_set:
                continue

            hospital_seed = HOSPITALS_BY_ID.get(supply.hospital.hospital_id)
            size_tier = hospital_seed.size_tier if hospital_seed is not None else "urban"
            size_profile = HOSPITAL_SIZE_PROFILES[size_tier]

            base_runway = supply.current_stock_units / max(supply.usage_today, 0.8)
            type_factor = BLOOD_TYPE_HISTORY_FACTORS.get(supply.blood_product_type, 0.9)
            size_factor = SIZE_HISTORY_FACTORS[size_tier]

            for days_ago in range(1, history_days + 1):
                past_date = today - timedelta(days=days_ago)
                if past_date in existing_dates:
                    continue

                rng = seeded_random("prediction-history", supply.supply_id, past_date.isoformat())
                trend_factor = 1.0 + (days_ago * 0.0025)
                volatility_adjustment = 1.0 - (size_profile["volatility"] * 0.35)
                seasonal_modifier = quebec_seasonality_modifier(past_date)
                jitter = rng.uniform(-0.85, 0.85)

                value = round(
                    base_runway
                    * type_factor
                    * size_factor
                    * trend_factor
                    * volatility_adjustment
                    * seasonal_modifier
                    + jitter,
                    2,
                )
                value = max(1.0, value)

                rows_to_create.append(
                    PredictionResult(
                        entity_id=supply.supply_id,
                        model_name=model_name,
                        predicted_for_date=past_date,
                        entity_type="supply",
                        hospital_id=supply.hospital.hospital_id,
                        blood_type=supply.blood_product_type,
                        model_version="champion",
                        predicted_value=value,
                        alert_triggered=value <= alert_threshold,
                        alert_threshold_used=alert_threshold,
                    )
                )

        if rows_to_create:
            PredictionResult.objects.bulk_create(rows_to_create, ignore_conflicts=True)

        self.stdout.write(
            self.style.SUCCESS(
                f"Seeded {len(rows_to_create)} prediction history row(s) "
                f"(model={model_name}, days={history_days}, threshold={alert_threshold})."
            )
        )
