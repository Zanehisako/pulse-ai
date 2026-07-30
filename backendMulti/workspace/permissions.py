# permissions.py - Adapté pour utiliser KeycloakUser
from rest_framework.permissions import BasePermission
from .models import DashboardPersonnel, PartageDashboard
from authApp.models import User


def _resolve_user(request) -> User:
    """
    Retourne un User Django depuis request.user.
    Gère les cas:
    - KeycloakUser (avec .django_user)
    - User Django direct
    """
    u = request.user

    # Si c'est un KeycloakUser, utiliser son django_user
    if hasattr(u, 'django_user'):
        return u.django_user

    # Si c'est déjà un User Django
    if isinstance(u, User):
        return u

    # Fallback: essayer d'en créer un depuis les données disponibles
    if hasattr(u, 'email'):
        email = u.email
        name = getattr(u, 'name', email)
        user, _ = User.objects.get_or_create(
            email=email,
            defaults={'name': name, 'is_active': True},
        )
        return user

    return None


class IsOwner(BasePermission):
    """Vérifie que l'utilisateur est le propriétaire du dashboard."""
    def has_object_permission(self, request, view, obj):
        user = _resolve_user(request)
        return user and obj.id_utilisateur == user


class CanViewDashboard(BasePermission):
    """Vérifie que l'utilisateur peut voir le dashboard (propriétaire ou partagé)."""
    def has_object_permission(self, request, view, obj):
        user = _resolve_user(request)
        if not user:
            return False
        # Propriétaire
        if obj.id_utilisateur == user:
            return True
        # Vérifié via partage
        return PartageDashboard.objects.filter(
            id_dashboard=obj.id_dashboard,
            id_utilisateur_dest=user,
            permission__in=['view', 'edit', 'share']
        ).exists()


class CanEditDashboard(BasePermission):
    """Vérifie que l'utilisateur peut modifier le dashboard."""
    def has_object_permission(self, request, view, obj):
        user = _resolve_user(request)
        if not user:
            return False
        # Propriétaire
        if obj.id_utilisateur == user:
            return True
        # Vérifié via partage avec permission edit ou share
        return PartageDashboard.objects.filter(
            id_dashboard=obj.id_dashboard,
            id_utilisateur_dest=user,
            permission__in=['edit', 'share']
        ).exists()


class CanShareDashboard(BasePermission):
    """Vérifie que l'utilisateur peut partager le dashboard."""
    def has_object_permission(self, request, view, obj):
        user = _resolve_user(request)
        if not user:
            return False
        # Propriétaire
        if obj.id_utilisateur == user:
            return True
        # Vérifié via partage avec permission share
        return PartageDashboard.objects.filter(
            id_dashboard=obj.id_dashboard,
            id_utilisateur_dest=user,
            permission='share'
        ).exists()
