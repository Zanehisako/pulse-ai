"""
Lightweight HTTP JSON client using only stdlib (no requests/httpx needed).
"""
from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def _get_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def http_json_request(
    *,
    method: str,
    url: str,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    method_upper = (method or "GET").strip().upper()
    query_params = {}
    if isinstance(params, dict):
        query_params = {k: v for k, v in params.items() if v is not None}
    full_url = url
    if query_params:
        separator = "&" if "?" in url else "?"
        full_url = f"{url}{separator}{urllib.parse.urlencode(query_params, doseq=True)}"

    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    body_bytes = None
    if body is not None:
        request_headers.setdefault("Content-Type", "application/json")
        body_bytes = json.dumps(body, ensure_ascii=True).encode("utf-8")

    request_obj = urllib.request.Request(
        full_url,
        data=body_bytes,
        headers=request_headers,
        method=method_upper,
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=timeout_s, context=_get_ssl_context()) as response:
            raw_text = response.read().decode("utf-8", errors="replace")
            status_code = int(response.getcode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc

    try:
        payload = json.loads(raw_text) if raw_text else {}
    except json.JSONDecodeError:
        payload = {"text": raw_text}
    return {"status_code": status_code, "url": full_url, "data": payload}


def http_stream_sse_request(
    *,
    method: str = "POST",
    url: str,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 30.0,
):
    """
    Streams Server-Sent Events (SSE) from an HTTP endpoint.
    Yields data string payloads extracted from 'data: <payload>' lines.
    Stops upon encountering '[DONE]' or stream end.
    """
    method_upper = (method or "POST").strip().upper()
    query_params = {}
    if isinstance(params, dict):
        query_params = {k: v for k, v in params.items() if v is not None}
    full_url = url
    if query_params:
        separator = "&" if "?" in url else "?"
        full_url = f"{url}{separator}{urllib.parse.urlencode(query_params, doseq=True)}"

    request_headers = {"Accept": "text/event-stream, application/json"}
    if headers:
        request_headers.update(headers)

    body_bytes = None
    if body is not None:
        request_headers.setdefault("Content-Type", "application/json")
        body_bytes = json.dumps(body, ensure_ascii=True).encode("utf-8")

    request_obj = urllib.request.Request(
        full_url,
        data=body_bytes,
        headers=request_headers,
        method=method_upper,
    )
    try:
        response = urllib.request.urlopen(request_obj, timeout=timeout_s, context=_get_ssl_context())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc

    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("data:"):
                data_payload = line[5:].strip()
                if data_payload == "[DONE]":
                    break
                if data_payload:
                    yield data_payload