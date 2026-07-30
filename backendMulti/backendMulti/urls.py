from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from django.views.static import serve
from inventory.views import DonorListView
from ml.api.views import OrchestratorChatView

from backendMulti.api_docs import (
    DocsAutoAuthTokenView,
    PiosSwaggerView,
    normalized_api_docs_token_route,
)
import backendMulti.schema  # registers KeycloakAuthenticationScheme with drf-spectacular

urlpatterns = [
    path("v1/auth/", include("authApp.urls")),
    path("testUrl/", include("testUrl.urls")),
    path("notifications/", include("notifications.urls")),
    path("alerts/", include("alerts.urls")),
    path("admin/", admin.site.urls),
    path('api/', include('communications.urls')),
    path('api/workspace/', include('workspace.urls')),
    path("api/donors/", DonorListView.as_view(), name="donors-list"),
    path("dashboard/", include(("dashboard.urls", "dashboard"), namespace="dashboard")),
    path('centres/', include('centres.urls')),
    path("api/digital-twin/", include("digital_twin.urls")),
    path("api/predictions/", include("inventory.urls")),
    path("api/inventory/", include("inventory.urls")),
    path("api/bloodbags/", include("bloodbag.urls")),

    path("api/ml/", include(("ml.urls", "ml"), namespace="ml")),
    path("ml/", include(("ml.urls", "ml-legacy"), namespace="ml-legacy")),
    path("chat/", OrchestratorChatView.as_view(), name="root-chat"),
    path("assets/<path:path>", serve, {"document_root": settings.BASE_DIR.parent / "chatapp" / "dist" / "assets"}),
]

try:
    from drf_spectacular.views import SpectacularAPIView
    from rest_framework.permissions import AllowAny
    urlpatterns += [
        path(
            "api/schema/",
            SpectacularAPIView.as_view(
                authentication_classes=[],
                permission_classes=[AllowAny],
            ),
            name="schema",
        ),
        path(
            normalized_api_docs_token_route(),
            DocsAutoAuthTokenView.as_view(),
            name="swagger-auto-auth-token",
        ),
        path(
            "api/docs/",
            PiosSwaggerView.as_view(
                url_name="schema",
                authentication_classes=[],
                permission_classes=[AllowAny],
            ),
            name="swagger-ui",
        ),
    ]
except ImportError:
    pass
