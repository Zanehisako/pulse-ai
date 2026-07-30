from django.urls import path
from .views import UploadMediaView, GetMediaView, DeleteMediaView

urlpatterns = [
    path('upload/', UploadMediaView.as_view()),
    path('<str:file_id>/', GetMediaView.as_view()),
    path('<str:file_id>/delete/', DeleteMediaView.as_view()),
]