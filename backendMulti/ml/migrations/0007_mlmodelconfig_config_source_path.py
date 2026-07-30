from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ml", "0006_alter_mlmodelconfig_model_type"),
    ]

    operations = [
        migrations.AddField(
            model_name="mlmodelconfig",
            name="config_source_path",
            field=models.CharField(
                blank=True,
                db_index=True,
                default="",
                max_length=1024,
            ),
        ),
    ]
