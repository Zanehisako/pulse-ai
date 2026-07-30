import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from dashboard.models import ModelResponse

class DashboardConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.group_name = "dashboard_updates"

        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

        await self.accept()

        # Send all active rows (endDate is None) immediately
        rows = await sync_to_async(list)(
            ModelResponse.objects.filter(endDate__isnull=True)
        )

        for row in rows:
            data = {
                "id": row.id,
                "typeModel": row.typeModel,
                "jsonResponse": row.jsonResponse,
                "region": row.region,
                "product": row.product,
                "period": row.period,
                "alert_level": row.alert_level,
                "startDate": row.startDate.isoformat(),
                "endDate": row.endDate.isoformat() if row.endDate else None
            }
            await self.send(text_data=json.dumps(data))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def dashboard_update(self, event):
        print(f"[Consumer] Received event: {event}")
        await self.send(text_data=json.dumps(event["data"]))