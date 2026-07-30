from rest_framework import serializers

from authApp.models import User
from .models import DashboardPersonnel, WidgetsLayout, PartageDashboard


class WorkspaceUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "email", "name"]


class WidgetsLayoutSerializer(serializers.ModelSerializer):
    class Meta:
        model = WidgetsLayout
        fields = "__all__"


class PartageDashboardSerializer(serializers.ModelSerializer):
    utilisateur_dest = WorkspaceUserSerializer(
        source="id_utilisateur_dest",
        read_only=True,
    )

    class Meta:
        model = PartageDashboard
        fields = [
            "id_partage",
            "id_utilisateur_dest",
            "utilisateur_dest",
            "permission",
            "date_partage",
        ]


class DashboardPersonnelSerializer(serializers.ModelSerializer):
    widgets_layout = WidgetsLayoutSerializer(many=True, read_only=True)
    partages = PartageDashboardSerializer(many=True, read_only=True)
    est_owner = serializers.SerializerMethodField()
    utilisateur = WorkspaceUserSerializer(source="id_utilisateur", read_only=True)
    owner_name = serializers.CharField(source="id_utilisateur.name", read_only=True)

    class Meta:
        model = DashboardPersonnel
        fields = "__all__"
        read_only_fields = ["id_utilisateur"]

    def get_est_owner(self, obj):
        request = self.context.get("request")
        if not request:
            return False

        user = request.user
        if hasattr(user, "django_user"):
            user = user.django_user

        return obj.id_utilisateur == user
