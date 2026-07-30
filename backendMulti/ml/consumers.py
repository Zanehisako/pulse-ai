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

        # Attach log handler to root and ML loggers to capture all live execution logs
        loggers_to_attach = [
            logging.getLogger(),
            logging.getLogger("ml"),
            logging.getLogger("ml.orchestrator"),
            logging.getLogger("ml.core"),
        ]
        log_handler = WebsocketLogHandler(self, None)
        log_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")
        log_handler.setFormatter(formatter)

        for lgr in loggers_to_attach:
            lgr.addHandler(log_handler)

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
            logging.error("Orchestrator unhandled exception: %s", exc, exc_info=True)
            async_to_sync(self.send_json)({
                "type": "error",
                "phase": "error",
                "message": f"Orchestrator execution error: {exc}"
            })
        finally:
            for lgr in loggers_to_attach:
                lgr.removeHandler(log_handler)
