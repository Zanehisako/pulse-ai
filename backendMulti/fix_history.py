import django
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
django.setup()
from inventory.models import PredictionResult, BloodSupply
from datetime import date, timedelta
import random

# Realistic baseline stockout days per blood type
# O- and B- are rarer so naturally lower
BASELINE = {
    "O+": 18, "O-": 8, "A+": 20, "A-": 10,
    "B+": 15, "B-": 7, "AB+": 22, "AB-": 9,
}

supplies = list(BloodSupply.objects.select_related("hospital").all())
today = date.today()
created = 0

for supply in supplies:
    baseline = BASELINE.get(supply.blood_product_type, 15)
    for days_ago in range(1, 61):  # 60 days of realistic history
        past_date = today - timedelta(days=days_ago)
        # Realistic variance: ±20% around baseline, slight downward trend near present
        trend_factor = 1.0 - (days_ago * 0.002)  # very mild decline toward today
        value = round(baseline * trend_factor + random.uniform(-2, 2), 2)
        value = max(1.0, value)  # never negative

        _, was_created = PredictionResult.objects.get_or_create(
            entity_id=supply.supply_id,
            model_name="stockout_days_predictor",
            predicted_for_date=past_date,
            defaults={
                "entity_type": "supply",
                "hospital_id": supply.hospital.hospital_id,
                "blood_type": supply.blood_product_type,
                "model_version": "champion",
                "predicted_value": value,
                "alert_triggered": value <= 3.0,
                "alert_threshold_used": 3.0,
            }
        )
        if was_created:
            created += 1

print("Created:", created)

# Verify thresholds now make sense
from alerts.services.threshold_engine import ThresholdEngine
for bt in ["O+", "O-", "A+", "B-"]:
    t = ThresholdEngine().compute("H001", bt, "zscore", fallback=999)
    print(f"H001/{bt} zscore threshold: {t:.2f} days")