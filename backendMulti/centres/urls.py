# testUrl/urls.py

from django.urls import path
from .views import CentreDonListView, CentreDonDetailView

urlpatterns = [
  
    path('api/centres/',      CentreDonListView.as_view()),
    path('api/centres/<int:pk>/', CentreDonDetailView.as_view()),
]