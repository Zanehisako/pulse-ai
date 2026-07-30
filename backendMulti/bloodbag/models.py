from django.db import models


class BloodBag(models.Model):
    STATUS_AVAILABLE = "available"
    STATUS_RESERVED = "reserved"
    STATUS_USED = "used"
    STATUS_EXPIRED = "expired"
    STATUS_DAMAGED = "damaged"

    STATUS_CHOICES = [
        (STATUS_AVAILABLE, "Available"),
        (STATUS_RESERVED, "Reserved"),
        (STATUS_USED, "Used"),
        (STATUS_EXPIRED, "Expired"),
        (STATUS_DAMAGED, "Damaged"),
    ]

    barcode = models.CharField(max_length=100, unique=True)
    blood_type = models.CharField(max_length=20)
    component = models.CharField(max_length=50)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_AVAILABLE,
    )
    collection_date = models.DateTimeField()
    expiry_date = models.DateTimeField()
    donor_id = models.CharField(max_length=100, blank=True, null=True)
    volume_ml = models.FloatField(default=450)
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()

    class Meta:
        ordering = ["expiry_date", "barcode"]

    def __str__(self) -> str:
        return self.barcode
