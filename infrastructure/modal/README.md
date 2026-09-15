# Modal AI GPU Inference Backend for PulseAI

This directory contains the production-ready Modal AI serverless deployment script and configuration for hosting PulseAI's LLM Agent Planner and GPU inference backend using **vLLM**.

---

## 1. Architecture Overview

```
 [PulseAI Agent Orchestrator] (Django / backendMulti)
                │
                ▼ (OpenAI-compatible HTTP/SSE)
 [Modal AI Serverless API Endpoint]
                │
                ▼
 [vLLM on Cloud GPU (e.g. NVIDIA A10G / L4 / A100)]
                │
                ├──► PagedAttention Engine
                ├──► HF Model Weights (Persistent Modal Volume Cache)
                └──► /v1/chat/completions (SSE Streaming & Function Calling)
```

### Why Modal + vLLM for PulseAI?
- **Zero Local Hardware Limits**: Offloads heavy 7B-12B parameter LLMs (like Qwen2.5-7B-Instruct or Salesforce/xLAM-7b-fc-r) to dedicated cloud GPUs.
- **Serverless Autoscaling**: Scales to zero when idle (no ongoing idle GPU costs).
- **Fast Cold Starts**: Hugging Face model weights are stored in a persistent `modal.Volume` cache (`pulseai-model-cache`), avoiding re-downloading model weights on every cold start.
- **High Throughput**: vLLM's PagedAttention and continuous batching deliver 3-5x the tokens/sec of naive PyTorch/HuggingFace servers.
- **OpenAI-Compatible**: Exposes standard `/v1/chat/completions` and `/health` endpoints compatible with PulseAI's config-driven tool architecture.

---

## 2. Prerequisites & Setup

### A. Install Modal CLI
```bash
pip install modal
```

### B. Authenticate with Modal
If you haven't authenticated yet:
```bash
modal setup
```
This opens a browser window to link your Modal account API token.

### C. (Optional) Hugging Face Secret
If using gated models (like Llama 3):
```bash
modal secret create huggingface-secret HF_TOKEN=hf_yourTokenHere
```

---

## 3. Configuration (`config.json`)

All deployment parameters are strictly config-driven in `infrastructure/modal/config.json`:

```json
{
  "app_name": "pulseai-vllm-backend",
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "gpu": "A10G",
  "max_model_len": 4096,
  "gpu_memory_utilization": 0.90,
  "timeout_seconds": 600,
  "scaledown_window_seconds": 300,
  "volume_name": "pulseai-model-cache",
  "port": 8000,
  "auth_enabled": false,
  "auth_token_env": "MODAL_API_KEY"
}
```

Every parameter can also be overridden at runtime via environment variables:
- `MODAL_MODEL_ID`: e.g. `Qwen/Qwen2.5-7B-Instruct` or `Salesforce/xLAM-7b-fc-r`
- `MODAL_GPU`: e.g. `A10G`, `L4`, `A100`, `H100`, or `T4`
- `MODAL_SCALEDOWN_WINDOW`: idle time in seconds before container stops (default: 300)

---

## 4. Deployment

### Option A: Ephemeral Dev Server (Hot-Reloading)
To test and view live logs in your terminal:
```bash
modal serve infrastructure/modal/modal_vllm_server.py
```
This outputs a temporary dev URL such as:
`https://<your-username>--pulseai-vllm-backend-serve-dev.modal.run`

### Option B: Production Deployment
To deploy permanently to your Modal workspace:
```bash
modal deploy infrastructure/modal/modal_vllm_server.py
```
This outputs your production URL:
`https://<your-username>--pulseai-vllm-backend-serve.modal.run`

---

## 5. Verify the Deployment

Run the test client with your Modal URL:
```bash
python infrastructure/modal/test_client.py --url https://<your-username>--pulseai-vllm-backend-serve.modal.run
```

Expected output:
```text
[1/3] Testing health endpoint (/health)...
  ✓ Health check OK (0.12s)
[2/3] Querying models catalog (/v1/models)...
  ✓ Available models: ['Qwen/Qwen2.5-7B-Instruct']
[3/3] Sending test blood donation planning prompt...
  ✓ Inference successful in 0.85s (48.2 tokens/s)
```

---

## 6. Integrating with PulseAI Backend

Once your Modal endpoint is deployed, configure PulseAI's backend in `.env`:

```env
# Modal AI GPU Inference Backend
PIOS_ORCH_LLM_API_URL=https://<your-username>--pulseai-vllm-backend-serve.modal.run
PIOS_ORCH_LLM_API_MODEL=Qwen/Qwen2.5-7B-Instruct
PIOS_ORCH_LLM_API_TIMEOUT_S=30.0
PIOS_ORCH_PREFER_REMOTE_LLM=1
```

Or configure it in `backendMulti/ml/config/config.json` under `"orchestrator.remote_llm"`.

PulseAI will automatically:
1. Route natural language planning (`_generate_plan`) to the Modal GPU backend.
2. Route response summarization (`_summarize_results`) to the Modal GPU backend.
3. Stream tokens in real-time to the web frontend and WebSocket connections.
4. Gracefully fall back to local heuristic routing if the network or endpoint is unavailable.
