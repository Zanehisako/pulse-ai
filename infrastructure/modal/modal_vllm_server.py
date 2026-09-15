"""
Modal AI Serverless GPU Inference Backend using vLLM for PulseAI.

Deploys a high-throughput, OpenAI-compatible LLM inference server on NVIDIA GPUs
(e.g., A10G, L4, A100) to serve PulseAI's Agent Planner & Orchestrator.

Usage:
    # 1. Ephemeral testing / live development:
    modal serve infrastructure/modal/modal_vllm_server.py

    # 2. Production deployment:
    modal deploy infrastructure/modal/modal_vllm_server.py

    # 3. Check configuration & status:
    modal run infrastructure/modal/modal_vllm_server.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import modal

CONFIG_FILE = Path(__file__).parent / "config.json"


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: Failed to parse {CONFIG_FILE}: {exc}")
    return {}


_cfg = _load_config()

# Configuration parameters with environment variable override support
APP_NAME = os.getenv("MODAL_APP_NAME", _cfg.get("app_name", "pulseai-vllm-backend"))
MODEL_ID = os.getenv("MODAL_MODEL_ID", _cfg.get("model_id", "Qwen/Qwen2.5-7B-Instruct"))
GPU_SPEC = os.getenv("MODAL_GPU", _cfg.get("gpu", "A10G"))
MAX_MODEL_LEN = int(os.getenv("MODAL_MAX_MODEL_LEN", _cfg.get("max_model_len", 4096)))
GPU_MEM_UTIL = float(os.getenv("MODAL_GPU_MEMORY_UTILIZATION", _cfg.get("gpu_memory_utilization", 0.90)))
TIMEOUT_SECONDS = int(os.getenv("MODAL_TIMEOUT_SECONDS", _cfg.get("timeout_seconds", 600)))
SCALEDOWN_WINDOW = int(os.getenv("MODAL_SCALEDOWN_WINDOW", _cfg.get("scaledown_window_seconds", 300)))
VOLUME_NAME = os.getenv("MODAL_VOLUME_NAME", _cfg.get("volume_name", "pulseai-model-cache"))
VOLUME_MOUNT_PATH = _cfg.get("volume_mount_path", "/root/.cache/huggingface")
PORT = int(_cfg.get("port", 8000))
HF_SECRET_NAME = _cfg.get("hf_token_secret_name", "huggingface-secret")
AUTH_TOKEN_ENV = _cfg.get("auth_token_env", "MODAL_API_KEY")

# Modal App and Cache Volume
app = modal.App(APP_NAME)
model_cache_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Container image with vLLM and HuggingFace dependencies
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.3",
        "huggingface_hub",
        "fastapi>=0.115.0",
        "pydantic>=2.9.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "VLLM_USAGE_SOURCE": "pulseai-modal",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_USE_V1": "0",
    })
)


secrets = []
if os.getenv("HF_TOKEN"):
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))
elif _cfg.get("use_hf_secret", False):
    secrets.append(modal.Secret.from_name(HF_SECRET_NAME))


STARTUP_TIMEOUT = int(os.getenv("MODAL_STARTUP_TIMEOUT", _cfg.get("startup_timeout_seconds", 1200)))
ENFORCE_EAGER = bool(str(os.getenv("MODAL_ENFORCE_EAGER", _cfg.get("enforce_eager", True))).lower() in ("1", "true", "yes"))


@app.function(
    image=vllm_image,
    gpu=GPU_SPEC,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT_SECONDS,
    volumes={VOLUME_MOUNT_PATH: model_cache_volume},
    secrets=secrets,
)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT)
def serve():
    """
    Spins up vLLM's native OpenAI-compatible API server on port 8000.
    Exposes:
      - POST /v1/chat/completions (with SSE streaming and function calling)
      - POST /v1/completions
      - GET /v1/models
      - GET /health
    """
    cmd = [
        "vllm", "serve",
        MODEL_ID,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEM_UTIL),
        "--trust-remote-code",
    ]

    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")

    # Optional Bearer Token Authentication
    auth_key = os.getenv(AUTH_TOKEN_ENV, "").strip()
    if auth_key:
        cmd.extend(["--api-key", auth_key])

    print(f"Starting vLLM server: {' '.join(cmd)}")
    subprocess.Popen(cmd)


@app.local_entrypoint()
def main():
    """
    Diagnostic entrypoint: prints active configuration and instructions.
    """
    print("=================================================================")
    print(" PulseAI - Modal AI GPU Inference Backend (vLLM)")
    print("=================================================================")
    print(f"App Name:             {APP_NAME}")
    print(f"Model ID:             {MODEL_ID}")
    print(f"GPU:                  {GPU_SPEC}")
    print(f"Max Model Length:     {MAX_MODEL_LEN}")
    print(f"GPU Mem Utilization:  {GPU_MEM_UTIL}")
    print(f"Scaledown Window:     {SCALEDOWN_WINDOW}s")
    print(f"Cache Volume:         {VOLUME_NAME}")
    print(f"Internal Port:        {PORT}")
    print("-----------------------------------------------------------------")
    print("Commands:")
    print("  Deploy: modal deploy infrastructure/modal/modal_vllm_server.py")
    print("  Serve:  modal serve infrastructure/modal/modal_vllm_server.py")
    print("=================================================================")
