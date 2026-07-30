from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    DashboardPersonnelViewSet,
    WidgetsLayoutViewSet,
    WorkspaceWidgetSourcesView,
)
 
router = DefaultRouter()
# backendMulti/workspace/urls.py
router.register(r'workspaces', DashboardPersonnelViewSet, basename='workspace')
router.register(r'widgets-layout', WidgetsLayoutViewSet, basename='widgets-layout')
urlpatterns = [
    path('', include(router.urls)),
    path("query_data/", DashboardPersonnelViewSet.as_view({"post": "query_data"})),
    path("metadata/widget-sources/", WorkspaceWidgetSourcesView.as_view()),
]
