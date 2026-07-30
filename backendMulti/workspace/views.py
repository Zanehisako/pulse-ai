from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from .services.query_service import execute_dynamic_query
from .services.widget_sources import get_widget_sources_catalog
from backendMulti.services.keycloak_auth import sync_keycloak_user_by_email

from django.db.models import Q

from .models import DashboardPersonnel, WidgetsLayout, PartageDashboard
from .serializers import (
    DashboardPersonnelSerializer,
    WidgetsLayoutSerializer,
    PartageDashboardSerializer
)
from .permissions import CanEditDashboard, CanShareDashboard
from authApp.models import User
from notifications.models import NotificationType
from notifications.services import publish_notification


# ============================================================
# USER HELPER
# ============================================================

def get_django_user(request) -> User:
    user_obj = request.user

    if hasattr(user_obj, 'django_user'):
        return user_obj.django_user

    if isinstance(user_obj, User):
        return user_obj

    if hasattr(user_obj, 'email'):
        user, _ = User.objects.get_or_create(
            email=user_obj.email,
            defaults={'name': getattr(user_obj, 'name', user_obj.email)}
        )
        return user

    raise Exception('Unable to resolve user')

# ============================================================
# HELPER : extraire config d'un widget depuis le payload
# ============================================================
 
def _extract_widget_config(widget: dict) -> dict:
    """
    Extrait et retourne le dict config à stocker dans WidgetsLayout.config.
    On y persiste type, title, table, group_by, metric, field
    pour pouvoir reconstruire le widget au chargement.
    """
    raw_config = widget.get('config', {})
    return {
        # Informations graphiques
        'type': widget.get('type') or raw_config.get('type') or 'bar',
        'title': widget.get('title') or raw_config.get('title') or '',
        # Informations SQL
        'table': raw_config.get('table') or raw_config.get('model', ''),
        'group_by': raw_config.get('group_by') or '',
        'metric': raw_config.get('metric') or 'count',
        'field': raw_config.get('field') or 'id',
    }
# ============================================================
# DASHBOARD VIEWSET
# ============================================================

class DashboardPersonnelViewSet(viewsets.ModelViewSet):
    serializer_class = DashboardPersonnelSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = get_django_user(self.request)
        return DashboardPersonnel.objects.filter(
            Q(id_utilisateur=user) |
            Q(partages__id_utilisateur_dest=user)
        ).distinct().prefetch_related('widgets_layout', 'partages')

    # ================= CREATE =================

    def create(self, request, *args, **kwargs):
        user = get_django_user(request)
        data = request.data.copy()

        data.setdefault('nom', 'Nouveau Dashboard')
        data.setdefault('description', '')
        data.setdefault('config', {})

        widgets = data.get('config', {}).get('widgets', [])

        config = data['config'].copy()
        config.pop('widgets', None)
        data['config'] = config

        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        dashboard = serializer.save(id_utilisateur=user)

        for idx, widget in enumerate(widgets):
            widget_config = _extract_widget_config(widget)  # 🔥 AJOUT

            WidgetsLayout.objects.create(
                id_dashboard=dashboard,
                widget_id=widget.get('id', f'widget-{idx}'),
                position_x=widget.get('position_x', 0),
                position_y=widget.get('position_y', 0),
                width=widget.get('width', 3),
                height=widget.get('height', 2),
                ordre_z=widget.get('ordre_z', idx),
                config=widget_config  # 🔥 AJOUT CRITIQUE

            )

        return Response(
            DashboardPersonnelSerializer(dashboard, context=self.get_serializer_context()).data,
            status=status.HTTP_201_CREATED
        )

    # ================= UPDATE =================

    def partial_update(self, request, *args, **kwargs):
        dashboard = self.get_object()

        perm = CanEditDashboard()
        if not perm.has_object_permission(request, self, dashboard):
            return Response({'error': 'Permission denied'}, status=403)

        widgets_data = request.data.get('config', {}).get('widgets', [])

        existing_widgets = {w.widget_id: w for w in dashboard.widgets_layout.all()}
        incoming_ids = []

        for idx, widget in enumerate(widgets_data):
            widget_id = widget.get('id', f'widget-{idx}')
            incoming_ids.append(widget_id)
            widget_config = _extract_widget_config(widget)


            if widget_id in existing_widgets:
                w = existing_widgets[widget_id]
                w.position_x = widget.get('position_x', w.position_x)
                w.position_y = widget.get('position_y', w.position_y)
                w.width = widget.get('width', w.width)
                w.height = widget.get('height', w.height)
                w.ordre_z = widget.get('ordre_z', idx)
                w.config     = widget_config   # ✅ mise à jour config
                w.save()
            else:
                WidgetsLayout.objects.create(
                    id_dashboard=dashboard,
                    widget_id=widget_id,
                    position_x=widget.get('position_x', 0),
                    position_y=widget.get('position_y', 0),
                    width=widget.get('width', 3),
                    height=widget.get('height', 2),
                    ordre_z=widget.get('ordre_z', idx),
                    config=widget_config,      # ✅ config au create

                )

        for wid, obj in existing_widgets.items():
            if wid not in incoming_ids:
                obj.delete()

        if 'nom' in request.data:
            dashboard.nom = request.data['nom']

        if 'description' in request.data:
            dashboard.description = request.data['description']

        if 'auto_refresh_ms' in request.data:
            dashboard.auto_refresh_ms = request.data['auto_refresh_ms']

        if 'is_draft' in request.data:
            dashboard.is_draft = request.data['is_draft']

        if 'est_template' in request.data:
            dashboard.est_template = request.data['est_template']

        if 'type_template' in request.data:
            dashboard.type_template = request.data['type_template']

        if 'config' in request.data:
            config = request.data['config'].copy()
            config.pop('widgets', None)
            dashboard.config = config

        dashboard.save()

        return Response(
            DashboardPersonnelSerializer(dashboard, context=self.get_serializer_context()).data
        )

    # ================= ACTIONS =================

    @action(detail=False, methods=["post"], url_path="query_data")
    def query_data(self, request):
        config = request.data

        try:
            data = execute_dynamic_query(config)
            return Response(data)
        except Exception as e:
            return Response({"error": str(e)}, status=400)

    @action(detail=False, methods=["get"], url_path="metadata/widget-sources")
    def widget_sources(self, request):
        return Response(get_widget_sources_catalog())

    # ✅ FAVORITE
    @action(detail=True, methods=['patch'])
    def favorite(self, request, pk=None):
        dashboard = self.get_object()
        dashboard.est_favori = not dashboard.est_favori
        dashboard.save()
        return Response(
            DashboardPersonnelSerializer(
                dashboard,
                context=self.get_serializer_context(),
            ).data
        )

    # ✅ SHARED LIST
    @action(detail=False, methods=['get'])
    def shared(self, request):
        user = get_django_user(request)
        dashboards = DashboardPersonnel.objects.filter(
            partages__id_utilisateur_dest=user
        ).distinct()
        serializer = self.get_serializer(dashboards, many=True)
        return Response(serializer.data)

    # ✅ TEMPLATES
    @action(detail=False, methods=['get'])
    def templates(self, request):
        dashboards = DashboardPersonnel.objects.filter(est_template=True)
        serializer = self.get_serializer(dashboards, many=True)
        return Response(serializer.data)

    # ✅ AUTO REFRESH
    @action(detail=True, methods=['patch'])
    def auto_refresh(self, request, pk=None):
        dashboard = self.get_object()

        perm = CanEditDashboard()
        if not perm.has_object_permission(request, self, dashboard):
            return Response({'error': 'Permission denied'}, status=403)

        ms = request.data.get('auto_refresh_ms', 0)
        dashboard.auto_refresh_ms = ms
        dashboard.save()
        return Response(
            DashboardPersonnelSerializer(dashboard, context=self.get_serializer_context()).data
        )

    # ✅ SHARE
    @action(detail=True, methods=['post'])
    def share(self, request, pk=None):
        dashboard = self.get_object()
        shared_by_user = get_django_user(request)

        perm = CanShareDashboard()
        if not perm.has_object_permission(request, self, dashboard):
            return Response({'error': 'Permission denied'}, status=403)

        user_email = request.data.get('user_email')
        user_id = request.data.get('id_utilisateur_dest')
        permission = request.data.get('permission', 'view')
        allowed_permissions = {
            choice[0] for choice in PartageDashboard.PERMISSION_CHOICES
        }

        if not user_email and not user_id:
            return Response({'error': 'User required'}, status=400)
        if permission not in allowed_permissions:
            return Response({'error': 'Invalid permission'}, status=400)

        try:
            if user_id:
                user_dest = User.objects.get(id=user_id)
            else:
                user_dest = User.objects.filter(email__iexact=user_email).first()
                if user_dest is None:
                    user_dest = sync_keycloak_user_by_email(user_email)
                if user_dest is None:
                    raise User.DoesNotExist
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=404)

        partage, created = PartageDashboard.objects.get_or_create(
            id_dashboard=dashboard,
            id_utilisateur_dest=user_dest,
            defaults={'permission': permission}
        )

        if not created:
            partage.permission = permission
            partage.save()
        else:
            shared_by_name = (
                shared_by_user.name
                or shared_by_user.email
                or "A colleague"
            )
            permission_message = {
                "view": "shared this workspace with you",
                "edit": "shared this workspace with edit access",
                "share": "shared this workspace with sharing access",
            }.get(permission, "shared this workspace with you")

            publish_notification(
                title="Workspace shared with you",
                body=f'{shared_by_name} {permission_message}: "{dashboard.nom}"',
                notif_type=NotificationType.WORKSPACE_SHARE,
                extra_data={
                    "kind": "workspace_share",
                    "workspace_id": dashboard.id_dashboard,
                    "owner_name": shared_by_name,
                    "owner_email": shared_by_user.email,
                    "permission": permission,
                },
                sent_by=shared_by_user.email,
                recipient=user_dest,
            )

        serializer = PartageDashboardSerializer(partage)
        return Response(serializer.data, status=201 if created else 200)

    # ✅ SHARE LIST
    @action(detail=True, methods=['get'])
    def share_list(self, request, pk=None):
        dashboard = self.get_object()
        partages = PartageDashboard.objects.filter(id_dashboard=dashboard)
        serializer = PartageDashboardSerializer(partages, many=True)
        return Response(serializer.data)

    # ✅ UNSHARE — accepte id_partage dans le body (cohérent avec workspaceAPI.ts)
    @action(detail=True, methods=['delete'])
    def unshare(self, request, pk=None):
        dashboard = self.get_object()

        perm = CanShareDashboard()
        if not perm.has_object_permission(request, self, dashboard):
            return Response({'error': 'Permission denied'}, status=403)

        partage_id = request.data.get('id_partage')
        if not partage_id:
            return Response({'error': 'id_partage required'}, status=400)

        try:
            partage = PartageDashboard.objects.get(
                id_partage=partage_id,
                id_dashboard=dashboard
            )
            partage.delete()
            return Response({'status': 'deleted'})
        except PartageDashboard.DoesNotExist:
            return Response({'error': 'Not found'}, status=404)

    # ✅ DUPLICATE
    @action(detail=True, methods=['post'])
    def duplicate(self, request, pk=None):
        dashboard = self.get_object()
        user = get_django_user(request)

        nom = request.data.get('nom', f"Copie de {dashboard.nom}")

        new_dashboard = DashboardPersonnel.objects.create(
            id_utilisateur=user,
            nom=nom,
            description=dashboard.description,
            config=dashboard.config,
            est_template=dashboard.est_template,
            type_template=dashboard.type_template,
            auto_refresh_ms=dashboard.auto_refresh_ms,
            is_draft=dashboard.is_draft,
        )

        for w in dashboard.widgets_layout.all():
            WidgetsLayout.objects.create(
                id_dashboard=new_dashboard,
                widget_id=w.widget_id,
                position_x=w.position_x,
                position_y=w.position_y,
                width=w.width,
                height=w.height,
                ordre_z=w.ordre_z,
                config=w.config,   # ✅ copie la config complète

            )

        serializer = self.get_serializer(new_dashboard)
        return Response(serializer.data, status=201)


# ============================================================
# WIDGETS VIEWSET
# ============================================================

class WidgetsLayoutViewSet(viewsets.ModelViewSet):
    serializer_class = WidgetsLayoutSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = get_django_user(self.request)
        return WidgetsLayout.objects.filter(id_dashboard__id_utilisateur=user)

    def create(self, request, *args, **kwargs):
        dashboard_id = request.data.get('id_dashboard')

        try:
            dashboard = DashboardPersonnel.objects.get(id_dashboard=dashboard_id)
        except DashboardPersonnel.DoesNotExist:
            return Response({'error': 'Dashboard not found'}, status=404)

        perm = CanEditDashboard()
        if not perm.has_object_permission(request, self, dashboard):
            return Response({'error': 'Permission denied'}, status=403)

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(id_dashboard=dashboard)

        return Response(serializer.data, status=201)


class WorkspaceWidgetSourcesView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(get_widget_sources_catalog())
