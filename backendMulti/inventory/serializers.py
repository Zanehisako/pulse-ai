from rest_framework import serializers

from inventory.models import (
    BloodSupply,
    BloodSupplySnapshot,
    Donor,
    Hospital,
    HospitalSupplyFeature,
    PredictionResult,
)


class HospitalSerializer(serializers.ModelSerializer):
    class Meta:
        model = Hospital
        fields = ["id", "hospital_id", "name", "wilaya"]


class BloodSupplySerializer(serializers.ModelSerializer):
    hospital_id = serializers.CharField(source="hospital.hospital_id", read_only=True)

    class Meta:
        model = BloodSupply
        fields = [
            "id", "supply_id", "hospital_id", "blood_product_type",
            "current_stock_units", "usage_today", "lead_time_days",
            "days_since_last_restock", "stockout_count_90d",
            "scheduled_surgeries_next7d", "event_timestamp", "updated_at",
        ]


class HospitalSupplyFeatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = HospitalSupplyFeature
        fields = [
            "id", "hospital_id", "event_timestamp", "temperature",
            "rain_mm", "holiday", "disaster", "scheduled_surgeries",
            "trauma_cases", "current_inventory", "updated_at",
        ]


class DonorSerializer(serializers.ModelSerializer):
    donation_count = serializers.IntegerField(source="frequency_365", read_only=True)

    class Meta:
        model = Donor
        fields = [
            "id", "donor_id", "name", "wilaya", "event_timestamp",
            "city_id", "lat", "lon", "availability", "blood_group",
            "recency_days", "frequency_365", "donation_count",
            "days_until_eligible", "cluster_id", "updated_at",
        ]


class PredictionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionResult
        fields = [
            "entity_id", "entity_type", "hospital_id", "blood_type",
            "model_name", "model_version", "predicted_value", "confidence",
            "alert_triggered", "alert_threshold_used",
            "predicted_for_date", "created_at",
        ]


class BloodSupplySnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = BloodSupplySnapshot
        fields = ["recorded_at", "current_stock_units", "usage_today"]
