from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("authApp", "0001_initial"),
        ("notifications", "0003_remove_notification_is_read"),
    ]

    operations = [
        migrations.AddField(
            model_name="notification",
            name="is_read",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="notification",
            name="recipient",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="notifications",
                to="authApp.user",
            ),
        ),
        migrations.AlterField(
            model_name="notification",
            name="type",
            field=models.CharField(
                choices=[
                    ("info", "Info"),
                    ("success", "Success"),
                    ("warning", "Warning"),
                    ("blood_donation", "Blood Donation"),
                    ("workspace_share", "Workspace Share"),
                ],
                default="info",
                max_length=20,
            ),
        ),
    ]
