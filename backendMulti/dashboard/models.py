from django.db import models

# Create your models here.
class ModelResponse(models.Model):
    class Region(models.TextChoices):
        QUEBEC = "quebec", "Quebec"
        MONTREAL = "montreal", "Montreal"

    class Product(models.TextChoices):
        BLOOD = "blood", "Blood"

    class Period(models.TextChoices):
        H24 = "24h", "24H"
        D2 = "2d", "2D"
        W1 = "1w", "1W"

    class AlertLevel(models.TextChoices):
        LOW = "low", "Low"
        MEDIUM = "medium", "Medium"
        HIGH = "high", "High"

    typeModel = models.CharField(max_length=100)
    jsonResponse = models.JSONField()

    region = models.CharField(
        max_length=32, choices=Region.choices, default=Region.QUEBEC, db_index=True
    )
    product = models.CharField(
        max_length=32, choices=Product.choices, default=Product.BLOOD, db_index=True
    )
    period = models.CharField(
        max_length=8, choices=Period.choices, default=Period.H24, db_index=True
    )
    alert_level = models.CharField(
        max_length=16, choices=AlertLevel.choices, default=AlertLevel.LOW, db_index=True
    )

    startDate = models.DateTimeField(auto_now_add=True)
    endDate = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["region", "product", "period", "alert_level", "startDate"]),
        ]

    def __str__(self):
        return f"{self.typeModel} ({self.id}) [{self.region}/{self.product}/{self.period}/{self.alert_level}]"