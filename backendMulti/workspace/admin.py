from django.contrib import admin
from .models import DashboardPersonnel, WidgetsLayout, PartageDashboard

@admin.register(DashboardPersonnel)
class DashboardPersonnelAdmin(admin.ModelAdmin):
    list_display = ('nom', 'id_utilisateur', 'est_favori', 'date_creation')
    list_filter = ('est_favori', 'est_template', 'date_creation')
    search_fields = ('nom', 'id_utilisateur__username')

@admin.register(WidgetsLayout)
class WidgetsLayoutAdmin(admin.ModelAdmin):
    list_display = ('widget_id', 'id_dashboard', 'position_x', 'position_y')

@admin.register(PartageDashboard)
class PartageDashboardAdmin(admin.ModelAdmin):
    list_display = ('id_dashboard', 'id_utilisateur_dest', 'permission', 'date_partage')
    list_filter = ('permission', 'date_partage')