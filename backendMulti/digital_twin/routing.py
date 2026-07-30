from django.urls import re_path

from .consumers import DigitalTwinConsumer


websocket_urlpatterns = [
    re_path(r"ws/digital-twin/(?P<run_id>[0-9a-f-]+)/$", DigitalTwinConsumer.as_asgi()),
]
