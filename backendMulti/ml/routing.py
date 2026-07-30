from django.urls import re_path
from ml.consumers import OrchestratorWebsocketConsumer

websocket_urlpatterns = [
    re_path(r"^ws/ml/orchestrator/$", OrchestratorWebsocketConsumer.as_asgi()),
]
