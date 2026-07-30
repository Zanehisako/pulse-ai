import json

from channels.generic.websocket import AsyncWebsocketConsumer
from django.contrib.auth.models import AnonymousUser

from .services import get_notification_group


class NotificationConsumer(AsyncWebsocketConsumer):
    """
    User-scoped notification consumer.
    Authenticated users join their personal notification group.
    Anonymous users do not receive targeted notifications.
    """

    async def connect(self):
        user = self.scope.get("user")
        self.group_name = None
        self.broadcast_group_name = get_notification_group()

        await self.channel_layer.group_add(self.broadcast_group_name, self.channel_name)

        if user and not isinstance(user, AnonymousUser) and getattr(user, "is_authenticated", False):
            self.group_name = get_notification_group(user)
            await self.channel_layer.group_add(self.group_name, self.channel_name)

        await self.accept()
        await self.send(
            text_data=json.dumps(
                {
                    "type": "connection",
                    "status": "connected",
                    "message": "WebSocket connected successfully",
                    "authenticated": bool(
                        user
                        and not isinstance(user, AnonymousUser)
                        and getattr(user, "is_authenticated", False)
                    ),
                }
            )
        )

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.broadcast_group_name, self.channel_name)
        if self.group_name:
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            if data.get("type") == "ping":
                await self.send(text_data=json.dumps({"type": "pong"}))
        except Exception:
            return

    async def send_notification(self, event):
        await self.send(text_data=json.dumps(event["data"]))
