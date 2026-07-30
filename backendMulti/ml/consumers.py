"""
WebSocket Consumers for PIOS ML Orchestrator.
Provides real-time bidirectional WebSocket streaming for agent reasoning, log records,
tool execution events, and LLM tokens.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from asgiref.sync import async_to_sync
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from ml.core.startup import get_orchestrator, is_ready, refresh_runtime_state


class WebsocketLogHandler(logging.Handler):
    """
    Custom Logging Handler that intercepts Python log records and publishes them
    live as WebSocket JSON frames.
    """

    def __init__(self, consumer: "OrchestratorWebsocketConsumer", loop: Any) -> None:
        super().__init__()
        self.consumer = consumer
        self.loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            payload = {
                "type": "log",
                "level": record.levelname,
                "logger": record.name,
                "message": msg,
                "timestamp": time.strftime("%H:%M:%S", time.localtime(record.created)),
            }
            async_to_sync(self.consumer.send_json)(payload)
        except Exception:
            pass


class OrchestratorWebsocketConsumer(AsyncJsonWebsocketConsumer):
    """
    Async WebSocket consumer for real-time orchestrator execution streaming.
    Listens for {"action": "predict", "query": "..."} messages and streams
    live logs, reasoning, tool steps, and tokens back to the client.
    """

    async def connect(self) -> None:
        await self.accept()
        await self.send_json({
            "type": "connection",
            "status": "connected",
            "message": "Connected to PulseAI Orchestrator WebSocket stream."
        })

    async def disconnect(self, close_code: int) -> None:
        pass

    async def receive_json(self, content: Dict[str, Any], **kwargs: Any) -> None:
        action = content.get("action")
        if action == "ping":
            await self.send_json({"type": "pong", "timestamp": time.time()})
            return

        if action == "predict":
            query = content.get("query", "")
            features = content.get("features", {})
            if not query:
                await self.send_json({"type": "error", "message": "Query parameter is required."})
                return

            # Execute prediction in background thread to avoid blocking ASGI loop
            import threading
            threading.Thread(
                target=self._run_orchestrator,
                args=(query, features),
                daemon=True
            ).start()

    def _run_orchestrator(self, query: str, features: Dict[str, Any]) -> None:
        if not is_ready():
            async_to_sync(self.send_json)({
                "type": "error",
                "message": "ML Orchestrator models are still loading."
            })
            return

        refresh_runtime_state("ws-predict", force=False, warmup=False)
        orchestrator = get_orchestrator()

        # Attach custom log handler to catch live python logs during orchestrator run
        root_logger = logging.getLogger()
        log_handler = WebsocketLogHandler(self, None)
        log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        log_handler.setFormatter(formatter)
        root_logger.addHandler(log_handler)

        def event_callback(event: Dict[str, Any]) -> None:
            async_to_sync(self.send_json)(event)

        try:
            async_to_sync(self.send_json)({
                "type": "progress",
                "phase": "received",
                "message": f"Starting orchestrator for query: '{query}'"
            })

            result = orchestrator.run(
                query=query,
                provided_features=features,
                event_callback=event_callback,
            )

            async_to_sync(self.send_json)({
                "type": "final",
                "phase": "completed",
                "success": bool(result.get("success")),
                "markdown": str(result.get("natural_language_response") or ""),
                "tool_summaries": result.get("tool_summaries", []),
                "result": result,
            })
        except Exception as exc:
            async_to_sync(self.send_json)({
                "type": "error",
                "phase": "error",
                "message": f"Orchestrator execution error: {exc}"
            })
        finally:
            root_logger.removeHandler(log_handler)
