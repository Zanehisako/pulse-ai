from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        ("dashboard", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="modelresponse",
            name="alert_level",
            field=models.CharField(
                choices=[("low", "Low"), ("medium", "Medium"), ("high", "High")],
                db_index=True,
                default="low",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="modelresponse",
            name="extraFilters",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="modelresponse",
            name="period",
            field=models.CharField(
                choices=[("24h", "24H"), ("2d", "2D"), ("1w", "1W")],
                db_index=True,
                default="24h",
                max_length=8,
            ),
        ),
        migrations.AddField(
            model_name="modelresponse",
            name="product",
            field=models.CharField(
                choices=[("blood", "Blood")],
                db_index=True,
                default="blood",
                max_length=32,
            ),
        ),
        migrations.AddField(
            model_name="modelresponse",
            name="region",
            field=models.CharField(
                choices=[("quebec", "Quebec"), ("montreal", "Montreal")],
                db_index=True,
                default="quebec",
                max_length=32,
            ),
        ),
        migrations.AddIndex(
            model_name="modelresponse",
            index=models.Index(
                fields=["region", "product", "period", "alert_level", "startDate"],
                name="dashboard_mo_region__b9cbf1_idx",
            ),
        ),
    ]