from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ml.core.model_config_sync import load_model_config_payload

logger = logging.getLogger(__name__)

ReloadCallback = Callable[[Path], dict[str, Any]]


def load_hot_reload_options(path: str | Path) -> dict[str, Any]:
    payload = load_model_config_payload(path)
    runtime = payload.get("runtime")
    runtime_config = runtime if isinstance(runtime, dict) else {}
    hot_reload = runtime_config.get("config_hot_reload")
    options = hot_reload if isinstance(hot_reload, dict) else {}
    return {
        "enabled": bool(options.get("enabled", False)),
        "poll_interval_seconds": float(options.get("poll_interval_seconds", 1.0)),
        "debounce_seconds": float(options.get("debounce_seconds", 0.5)),
    }


class ModelConfigHotReloadWatcher:
    def __init__(
        self,
        *,
        path: str | Path,
        poll_interval_seconds: float,
        debounce_seconds: float,
        reload_callback: ReloadCallback,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self.debounce_seconds = max(0.0, float(debounce_seconds))
        self.reload_callback = reload_callback
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._processed_signature: str | None = None
        self._pending_signature: str | None = None
        self._pending_since: float | None = None
        self.last_signature: str | None = None
        self.last_file_reload_at: float | None = None
        self.last_file_error: str | None = None
        self.last_reload_result: dict[str, Any] | None = None

    def _file_signature(self) -> str:
        stat = self.path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"

    def prime(self) -> None:
        signature = self._file_signature()
        with self._lock:
            self.last_signature = signature
            self._processed_signature = signature
            self._pending_signature = None
            self._pending_since = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self.run_forever,
                name="pios-model-config-hot-reload",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def run_forever(self) -> None:
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(self.poll_interval_seconds)

    def poll_once(self, *, now: float | None = None) -> dict[str, Any]:
        current_time = time.monotonic() if now is None else float(now)
        try:
            signature = self._file_signature()
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.last_file_error = error
            logger.warning("Model config hot reload stat failed: %s", error)
            return {"changed": False, "applied": False, "error": error}

        with self._lock:
            self.last_signature = signature
            if self._processed_signature is None:
                self._processed_signature = signature
                return {"changed": False, "applied": False, "baseline": True}
            if signature == self._processed_signature:
                self._pending_signature = None
                self._pending_since = None
                return {"changed": False, "applied": False}
            if signature != self._pending_signature:
                self._pending_signature = signature
                self._pending_since = current_time
                return {"changed": True, "applied": False, "debounced": True}
            pending_since = self._pending_since

        if pending_since is not None and current_time - pending_since < self.debounce_seconds:
            return {"changed": True, "applied": False, "debounced": True}

        try:
            result = self.reload_callback(self.path)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self.last_file_error = error
                self._processed_signature = signature
                self._pending_signature = None
                self._pending_since = None
            logger.warning("Model config hot reload rejected file: %s", error)
            return {"changed": True, "applied": False, "error": error}

        with self._lock:
            self.last_file_error = None
            self.last_file_reload_at = time.time()
            self.last_reload_result = result
            self._processed_signature = signature
            self._pending_signature = None
            self._pending_since = None
        logger.info("Model config hot reload applied: %s", result)
        return {"changed": True, "applied": True, "result": result}

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            return {
                "watched_path": str(self.path),
                "running": running,
                "last_file_reload_at": self.last_file_reload_at,
                "last_file_error": self.last_file_error,
                "poll_interval_seconds": self.poll_interval_seconds,
                "debounce_seconds": self.debounce_seconds,
                "last_signature": self.last_signature,
                "last_reload_result": self.last_reload_result,
            }
