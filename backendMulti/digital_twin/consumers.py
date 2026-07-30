from __future__ import annotations

import json

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from .config import DigitalTwinConfigError
from .models import TwinRun
from .services import (
    action_payload,
    capture_snapshot,
    current_state,
    event_payload,
    execute_action,
    inject_event,
    stop_run,
)


class DigitalTwinConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.run_id = self.scope["url_route"]["kwargs"]["run_id"]
        self.group_name = f"digital_twin_{self.run_id}"
        user = self.scope.get("user")
        if not user or not user.is_authenticated:
            await self.close(code=4003)
            return
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        try:
            state = await sync_to_async(self._current_state)()
            await self.send(text_data=json.dumps({"type": "twin_state", "data": state}))
        except Exception as exc:
            await self.send(text_data=json.dumps({"type": "error", "message": str(exc)}))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    def _run(self) -> TwinRun:
        return TwinRun.objects.get(run_id=self.run_id)

    def _actor(self) -> str:
        user = self.scope.get("user")
        return str(user) if user and user.is_authenticated else "system"

    def _current_state(self) -> dict:
        return current_state(self._run())

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            command = data.get("command")
            if command == "ping":
                await self.send(text_data=json.dumps({"type": "pong"}))
            elif command == "refresh_snapshot":
                payload = await sync_to_async(self._refresh_snapshot)(data)
                await self._broadcast({"type": "twin_snapshot", "data": payload})
            elif command == "inject_event":
                payload = await sync_to_async(self._inject_event)(data)
                await self._broadcast({"type": "twin_event", "data": payload})
            elif command == "execute_action":
                payload = await sync_to_async(self._execute_action)(data)
                await self._broadcast({"type": "twin_action", "data": payload})
            elif command == "stop":
                payload = await sync_to_async(self._stop)()
                await self._broadcast({"type": "stopped", "data": payload})
            else:
                await self.send(text_data=json.dumps({"type": "error", "message": f"Unsupported command: {command}"}))
        except (DigitalTwinConfigError, TwinRun.DoesNotExist) as exc:
            await self.send(text_data=json.dumps({"type": "error", "message": str(exc)}))
        except Exception as exc:
            await self.send(text_data=json.dumps({"type": "error", "message": str(exc)}))

    def _refresh_snapshot(self, data: dict) -> dict:
        snapshot = capture_snapshot(
            self._run(),
            source=data.get("source"),
            actor=self._actor(),
        )
        return {
            "snapshot_id": str(snapshot.snapshot_id),
            "source": snapshot.source,
            "captured_at": snapshot.captured_at.isoformat(),
            "freshness": snapshot.freshness,
            "warnings": snapshot.warnings,
        }

    def _inject_event(self, data: dict) -> dict:
        event = inject_event(
            self._run(),
            event_key=str(data.get("event_key") or ""),
            severity=float(data.get("severity", 0.5)),
            start_hour=float(data.get("start_hour", 0.0)),
            source=str(data.get("source") or "manual"),
            payload=dict(data.get("payload") or {}),
            actor=self._actor(),
        )
        return event_payload(event)

    def _execute_action(self, data: dict) -> dict:
        action = execute_action(
            self._run(),
            action_key=str(data.get("action_key") or ""),
            simulated_hour=float(data.get("simulated_hour", 0.0)),
            source=str(data.get("source") or "manual"),
            payload=dict(data.get("payload") or {}),
            actor=self._actor(),
        )
        return action_payload(action)

    def _stop(self) -> dict:
        run = stop_run(self._run(), actor=self._actor())
        return {"run_id": str(run.run_id), "status": run.status}

    async def _broadcast(self, payload: dict) -> None:
        await self.channel_layer.group_send(
            self.group_name,
            {"type": "twin.message", "data": payload},
        )

    async def twin_message(self, event):
        await self.send(text_data=json.dumps(event["data"]))
