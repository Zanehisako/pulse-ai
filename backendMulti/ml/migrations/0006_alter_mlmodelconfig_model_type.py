from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ml", "0005_add_drift_report"),
    ]

    operations = [
        migrations.AlterField(
            model_name="mlmodelconfig",
            name="model_type",
            field=models.CharField(
                choices=[
                    ("sklearn", "Scikit-learn"),
                    ("xgboost", "XGBoost"),
                    ("custom", "Custom Linear"),
                    ("online_agent", "Online Agent"),
                    ("pytorch", "PyTorch"),
                    ("mlflow", "MLflow Registry"),
                    ("sota_joblib", "SOTA Joblib Artifact"),
                ],
                default="sklearn",
                max_length=64,
            ),
        ),
    ]
