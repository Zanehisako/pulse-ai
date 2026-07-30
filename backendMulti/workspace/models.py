from django.db import models
from authApp.models import User

# ─────────────────────────────────────────────
# 🎯 DASHBOARD (EXISTANT - NE PAS TOUCHER)
# ─────────────────────────────────────────────

class DashboardPersonnel(models.Model):
    TEMPLATE_CHOICES = (
        ('admin', 'Admin'),
        ('manager', 'Manager'),
        ('technicien', 'Technicien'),
    )

    id_dashboard = models.AutoField(primary_key=True)
    id_utilisateur = models.ForeignKey(User, on_delete=models.CASCADE)
    nom = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    config = models.JSONField(default=dict)
    est_favori = models.BooleanField(default=False)
    est_template = models.BooleanField(default=False)
    type_template = models.CharField(max_length=50, choices=TEMPLATE_CHOICES, blank=True, null=True)
    auto_refresh_ms = models.IntegerField(default=0)
    is_draft = models.BooleanField(default=False)
    date_creation = models.DateTimeField(auto_now_add=True)
    date_modification = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('id_utilisateur', 'nom')

    def __str__(self):
        return f"{self.nom} - {self.id_utilisateur.email}"


class WidgetsLayout(models.Model):
    id_layout = models.AutoField(primary_key=True)
    id_dashboard = models.ForeignKey(
        DashboardPersonnel,
        on_delete=models.CASCADE,
        related_name='widgets_layout'
    )
    widget_id = models.CharField(max_length=100)
    position_x = models.IntegerField(default=0)
    position_y = models.IntegerField(default=0)
    width = models.IntegerField(default=3)
    height = models.IntegerField(default=2)
    ordre_z = models.IntegerField(default=0)

    # 🔥 IMPORTANT : config dynamique
    config = models.JSONField(default=dict)

    date_modification = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('id_dashboard', 'widget_id')

    def __str__(self):
        return f"Widget {self.widget_id} - Dashboard {self.id_dashboard.nom}"


class PartageDashboard(models.Model):
    PERMISSION_CHOICES = (
        ('view', 'View'),
        ('edit', 'Edit'),
        ('share', 'Share'),
    )

    id_partage = models.AutoField(primary_key=True)
    id_dashboard = models.ForeignKey(
        DashboardPersonnel,
        on_delete=models.CASCADE,
        related_name='partages'
    )
    id_utilisateur_dest = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='dashboards_shared_with'
    )
    permission = models.CharField(max_length=20, choices=PERMISSION_CHOICES, default='view')
    date_partage = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('id_dashboard', 'id_utilisateur_dest')

    def __str__(self):
        return f"{self.id_dashboard.nom} partagé à {self.id_utilisateur_dest.email}"


