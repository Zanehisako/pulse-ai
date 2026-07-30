"""
ASGI config for backendMulti project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")


django_asgi_app = get_asgi_application()


from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from alerts.routing import websocket_urlpatterns as alerts_ws
from notifications.routing import websocket_urlpatterns as notifications_ws
from dashboard.routing import websocket_urlpatterns as dashboard_ws
from digital_twin.routing import websocket_urlpatterns as digital_twin_ws
from backendMulti.middleware.ws_auth import TokenAuthMiddleware

_inner = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AuthMiddlewareStack(
        TokenAuthMiddleware(
            URLRouter(
                alerts_ws + notifications_ws + dashboard_ws + digital_twin_ws
            )
        )
    ),
})


application = _inner
