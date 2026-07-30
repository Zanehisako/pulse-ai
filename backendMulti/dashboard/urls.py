from django.urls import path

from . import views

urlpatterns = [
    # path('api/donneur-ideal/',),
    path("add/", views.add_dashboard_row, name="add_model_response"),

    # WEB
    path("add-web/", views.add_dashboard_row_web, name="add_model_response_web"),
    path("latest/", views.latest_dashboard_data, name="latest_dashboard_data"),
]