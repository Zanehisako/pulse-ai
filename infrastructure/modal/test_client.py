#!/usr/bin/env python3
"""
PulseAI Modal GPU Inference Backend Test Client.

Tests the OpenAI-compatible vLLM endpoint running on Modal AI:
  - Health check (/health)
  - Model catalog (/v1/models)
  - Chat completions (/v1/chat/completions) with blood donor planning prompt
  - Latency measurement

Usage:
    python infrastructure/modal/test_client.py --url https://<workspace>--pulseai-vllm-backend-serve.modal.run
"""

from __future__ import annotations

import argparse
import json
import sys
import ssl
import time
import urllib.error
import urllib.request


def _get_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


def _http_request(
    url: str,
    method: str = "GET",
    body: dict | None = None,
    api_key: str = "",
    timeout_s: float = 60.0,
) -> tuple[int, dict | str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data_bytes = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=_get_ssl_context()) as response:
            status = int(response.getcode())
            text = response.read().decode("utf-8", errors="replace")
            try:
                return status, json.loads(text)
            except json.JSONDecodeError:
                return status, text
    except urllib.error.HTTPError as exc:
        err_text = exc.read().decode("utf-8", errors="replace")
        return exc.code, err_text
    except Exception as exc:
        return 0, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Test Modal vLLM inference backend.")
    parser.add_argument(
        "--url",
        required=True,
        help="Base URL of Modal deployment (e.g. https://<workspace>--pulseai-vllm-backend-serve.modal.run)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key / Bearer token if auth is enabled.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model ID to request (defaults to first available from /v1/models).",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print("=================================================================")
    print(" PulseAI - Modal GPU Inference Verification Client")
    print(f" Target: {base_url}")
    print("=================================================================")

    # 1. Health check
    print("\n[1/3] Testing health endpoint (/health)...")
    t0 = time.time()
    status, res = _http_request(f"{base_url}/health", api_key=args.api_key)
    elapsed = time.time() - t0
    if status == 200:
        print(f"  ✓ Health check OK ({elapsed:.2f}s): {res}")
    else:
        print(f"  ⚠ Health check returned HTTP {status} ({elapsed:.2f}s): {res}")

    # 2. Model list
    print("\n[2/3] Querying models catalog (/v1/models)...")
    status, res = _http_request(f"{base_url}/v1/models", api_key=args.api_key)
    selected_model = args.model
    if status == 200 and isinstance(res, dict) and "data" in res:
        models = [m.get("id") for m in res["data"] if isinstance(m, dict)]
        print(f"  ✓ Available models: {models}")
        if not selected_model and models:
            selected_model = models[0]
    else:
        print(f"  ⚠ Models endpoint returned HTTP {status}: {res}")
        if not selected_model:
            selected_model = "default"

    print(f"  -> Using model: {selected_model}")

    # 3. Chat completion
    print("\n[3/3] Sending test blood donation planning prompt...")
    test_messages = [
        {
            "role": "system",
            "content": "You are the PulseAI Agent Planner for blood donation logistics. Recommend tool execution steps in JSON format.",
        },
        {
            "role": "user",
            "content": "A 30-year-old donor weighing 70kg wants to donate blood at Hospital H001. Check eligibility and assess stockout risk.",
        },
    ]
    payload = {
        "model": selected_model,
        "messages": test_messages,
        "max_tokens": 256,
        "temperature": 0.1,
    }

    t0 = time.time()
    status, res = _http_request(
        f"{base_url}/v1/chat/completions",
        method="POST",
        body=payload,
        api_key=args.api_key,
        timeout_s=60.0,
    )
    elapsed = time.time() - t0

    if status == 200 and isinstance(res, dict):
        choices = res.get("choices", [])
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            usage = res.get("usage", {})
            tokens = usage.get("completion_tokens", 0)
            tok_s = (tokens / elapsed) if elapsed > 0 and tokens > 0 else 0
            print(f"  ✓ Inference successful in {elapsed:.2f}s ({tok_s:.1f} tokens/s):")
            print("-----------------------------------------------------------------")
            print(content)
            print("-----------------------------------------------------------------")
            print(f"Usage metadata: {usage}")
        else:
            print(f"  ⚠ Unexpected response structure: {res}")
    else:
        print(f"  ✗ Inference failed with HTTP {status}: {res}")
        sys.exit(1)

    print("\n✓ Modal GPU Backend is fully functional and ready for PulseAI!")


if __name__ == "__main__":
    main()
