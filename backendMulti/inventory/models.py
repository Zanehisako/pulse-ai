from django.db import models
 
 
class Hospital(models.Model):
    hospital_id = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=255)
    wilaya = models.CharField(max_length=120, blank=True, default="")
 
    def __str__(self) -> str:
        return f"{self.hospital_id} - {self.name}"
 
 
class BloodSupply(models.Model):
    supply_id = models.CharField(max_length=20, unique=True)
    hospital = models.ForeignKey(Hospital, on_delete=models.CASCADE, related_name="supplies")
    blood_product_type = models.CharField(max_length=5)
    current_stock_units = models.FloatField()
    usage_today = models.FloatField(default=0)
    lead_time_days = models.IntegerField(default=3)
    days_since_last_restock = models.IntegerField(default=0)
    stockout_count_90d = models.IntegerField(default=0)
    scheduled_surgeries_next7d = models.IntegerField(default=0)
    event_timestamp = models.DateTimeField()
    updated_at = models.DateTimeField(auto_now=True)
 
    class Meta:
        indexes = [
            models.Index(fields=["supply_id", "event_timestamp"]),
        ]
 
    def __str__(self) -> str:
        return self.supply_id
 
 
class HospitalSupplyFeature(models.Model):
    hospital_id = models.CharField(max_length=20, db_index=True)
    event_timestamp = models.DateTimeField()
    temperature = models.FloatField()
    rain_mm = models.FloatField()
    holiday = models.IntegerField()
    disaster = models.IntegerField()
    scheduled_surgeries = models.IntegerField()
    trauma_cases = models.IntegerField()
    # FIX 1: was IntegerField — ML model trained on float values, rounding loses precision
    current_inventory = models.FloatField()
    updated_at = models.DateTimeField(auto_now=True)
 
    class Meta:
        indexes = [
            models.Index(fields=["hospital_id", "event_timestamp"]),
        ]
 
 
class Donor(models.Model):
    donor_id = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100, default="")
    wilaya = models.CharField(max_length=50, default="")
    event_timestamp = models.DateTimeField()
    city_id = models.CharField(max_length=20)
    lat = models.FloatField()
    lon = models.FloatField()
    availability = models.IntegerField()
    blood_group = models.CharField(max_length=5)
    recency_days = models.IntegerField()
    frequency_365 = models.IntegerField()
    days_until_eligible = models.IntegerField()
    cluster_id = models.IntegerField()
    updated_at = models.DateTimeField(auto_now=True)
 
    class Meta:
        indexes = [
            models.Index(fields=["donor_id", "event_timestamp"]),
        ]
 
 
class BloodSupplySnapshot(models.Model):
    hospital = models.ForeignKey(Hospital, on_delete=models.CASCADE, related_name="snapshots")
    blood_product_type = models.CharField(max_length=10)
    current_stock_units = models.FloatField()
    usage_today = models.FloatField()
    days_since_last_restock = models.IntegerField()
    recorded_at = models.DateTimeField(db_index=True)

    class Meta:
        ordering = ["-recorded_at"]
        indexes = [
            models.Index(fields=["hospital", "blood_product_type", "recorded_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.hospital.hospital_id}/{self.blood_product_type} @ {self.recorded_at}"


class StockForecast(models.Model):
    hospital = models.ForeignKey(Hospital, on_delete=models.CASCADE, related_name="forecasts")
    blood_product_type = models.CharField(max_length=10)
    forecast_date = models.DateField()
    horizon = models.CharField(max_length=5)  # 't1', 't7', 't30'
    predicted_units = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["hospital", "blood_product_type", "horizon"]),
            models.Index(fields=["hospital", "blood_product_type", "forecast_date"]),
        ]
        unique_together = ["hospital", "blood_product_type", "forecast_date", "horizon"]

    def __str__(self) -> str:
        return f"{self.hospital.hospital_id}/{self.blood_product_type} {self.horizon} @ {self.forecast_date}"


class PredictionResult(models.Model):
    entity_id = models.CharField(max_length=50)
    entity_type = models.CharField(max_length=30)
    hospital_id = models.CharField(max_length=20, blank=True, default="", db_index=True)
    blood_type = models.CharField(max_length=16, blank=True, default="", db_index=True)
    model_name = models.CharField(max_length=128)
    model_version = models.CharField(max_length=20, default="champion")
    predicted_value = models.FloatField()
    confidence = models.FloatField(null=True, blank=True)
    predicted_for_date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)
    alert_triggered = models.BooleanField(default=False)
    alert_threshold_used = models.FloatField(null=True, blank=True)
 
    class Meta:
        indexes = [
            models.Index(fields=["entity_id", "model_name", "predicted_for_date"]),
            models.Index(fields=["hospital_id", "blood_type", "predicted_for_date"]),
        ]
        unique_together = ["entity_id", "model_name", "predicted_for_date"]
