from django.db import migrations, models
import django.db.models.deletion
import uuid

class Migration(migrations.Migration):

    dependencies = [
        ('authApp', '0001_initial'),  # Ou ta dernière migration users
    ]

    operations = [
        # Table DASHBOARD_PERSONNEL
        migrations.CreateModel(
            name='DashboardPersonnel',
            fields=[
                ('id_dashboard', models.AutoField(primary_key=True, serialize=False)),
                ('nom', models.CharField(max_length=255)),
                ('description', models.TextField(blank=True, null=True)),
                ('config', models.JSONField()),
                ('est_favori', models.BooleanField(default=False)),
                ('est_template', models.BooleanField(default=False)),
                ('type_template', models.CharField(choices=[('admin', 'Admin'), ('manager', 'Manager'), ('technicien', 'Technicien')], max_length=50, null=True, blank=True)),
                ('auto_refresh_ms', models.IntegerField(default=0)),
                ('is_draft', models.BooleanField(default=False)),
                ('date_creation', models.DateTimeField(auto_now_add=True)),
                ('date_modification', models.DateTimeField(auto_now=True)),
                ('id_utilisateur', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='authApp.user')),
            ],
            options={
                'verbose_name': 'Dashboard Personnel',
                'unique_together': {('id_utilisateur', 'nom')},
            },
        ),

        # Table WIDGETS_LAYOUT
        migrations.CreateModel(
            name='WidgetsLayout',
            fields=[
                ('id_layout', models.AutoField(primary_key=True, serialize=False)),
                ('widget_id', models.CharField(max_length=100)),
                ('position_x', models.IntegerField(default=0)),
                ('position_y', models.IntegerField(default=0)),
                ('width', models.IntegerField(default=3)),
                ('height', models.IntegerField(default=2)),
                ('ordre_z', models.IntegerField(default=0)),
                ('date_modification', models.DateTimeField(auto_now=True)),
                ('id_dashboard', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='workspace.dashboardpersonnel')),
            ],
            options={
                'verbose_name': 'Widget Layout',
                'unique_together': {('id_dashboard', 'widget_id')},
            },
        ),

        # Table PARTAGE_DASHBOARD
        migrations.CreateModel(
            name='PartageDashboard',
            fields=[
                ('id_partage', models.AutoField(primary_key=True, serialize=False)),
                ('permission', models.CharField(choices=[('view', 'View'), ('edit', 'Edit'), ('share', 'Share')], default='view', max_length=20)),
                ('date_partage', models.DateTimeField(auto_now_add=True)),
                ('id_dashboard', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='workspace.dashboardpersonnel')),
                ('id_utilisateur_dest', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='dashboards_shared_with', to='authApp.user')),
            ],
            options={
                'verbose_name': 'Partage Dashboard',
                'unique_together': {('id_dashboard', 'id_utilisateur_dest')},
            },
        ),

        # Index sur permission
        migrations.AddIndex(
            model_name='dashboardpersonnel',
            index=models.Index(fields=['id_utilisateur'], name='dashboard_user_idx'),
        ),
        migrations.AddIndex(
            model_name='dashboardpersonnel',
            index=models.Index(fields=['est_favori'], name='dashboard_favori_idx'),
        ),
    ]