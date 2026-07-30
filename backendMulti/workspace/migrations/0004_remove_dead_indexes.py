# Generated — cleanup: removed dead models (CentreDon, Don, Donneur, Stock)
# that were originally created here. Only the index removals remain.

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('workspace', '0003_alter_dashboardpersonnel_options_and_more'),
    ]

    operations = [
        migrations.RemoveIndex(
            model_name='dashboardpersonnel',
            name='workspace_d_id_util_f64222_idx',
        ),
        migrations.RemoveIndex(
            model_name='dashboardpersonnel',
            name='workspace_d_est_fav_43c4cd_idx',
        ),
    ]
