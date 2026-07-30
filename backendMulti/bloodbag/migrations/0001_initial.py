from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="BloodBag",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("barcode", models.CharField(max_length=100, unique=True)),
                ("blood_type", models.CharField(max_length=20)),
                ("component", models.CharField(max_length=50)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("available", "Available"),
                            ("reserved", "Reserved"),
                            ("used", "Used"),
                            ("expired", "Expired"),
                            ("damaged", "Damaged"),
                        ],
                        default="available",
                        max_length=20,
                    ),
                ),
                ("collection_date", models.DateTimeField()),
                ("expiry_date", models.DateTimeField()),
                ("donor_id", models.CharField(blank=True, max_length=100, null=True)),
                ("volume_ml", models.FloatField(default=450)),
                ("notes", models.TextField(blank=True, null=True)),
                ("created_at", models.DateTimeField()),
                ("updated_at", models.DateTimeField()),
            ],
            options={
                "ordering": ["expiry_date", "barcode"],
            },
        ),
    ]
