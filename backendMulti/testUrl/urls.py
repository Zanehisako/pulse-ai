from django.urls import path

from testUrl.algiers_views import AlgiersClusterSummaryView, AlgiersGridCellView, AlgiersClusterSummaryView, DataVersionView, QuebecLabsByBoundsView, ReloadDataView
from testUrl.views import LaboratoiresView, TiledLaboratoiresView

urlpatterns = [
    # Mobile App Urls
    path('api/labs/quebec-index/', AlgiersClusterSummaryView.as_view()),
    path('api/labs/quebec-cell/',  AlgiersGridCellView.as_view()),
    path('api/labs/quebec-clusters/', AlgiersClusterSummaryView.as_view()),
    path('api/labs/quebec-bounds/',    QuebecLabsByBoundsView.as_view()),
    path('api/labs/version/',         DataVersionView.as_view()),
    path('api/labs/reload/', ReloadDataView.as_view()),
    # Web App Urls
    path('api/laboratoires/tiled/', TiledLaboratoiresView.as_view(), name='laboratoires-tiled'),

    # share App Urls
    path('api/laboratoires/', LaboratoiresView.as_view(), name='laboratoires'),
]