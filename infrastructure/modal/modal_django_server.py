"""
Modal AI Serverless Deployment for PulseAI Django Backend & DynamicXLAMOrchestrator.

Hosts the entire Django backend, Tool Registry, ML models, and clinical pipelines
serverless on Modal AI, completely removing any reliance on local machines.

Usage:
    # 1. Test the remote orchestrator and tool execution in the cloud:
    modal run infrastructure/modal/modal_django_server.py

    # 2. Ephemeral interactive development / live testing:
    modal serve infrastructure/modal/modal_django_server.py

    # 3. Production deployment (provides a persistent public HTTPS URL):
    modal deploy infrastructure/modal/modal_django_server.py
"""

from __future__ import annotations

import json
import os
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
_django_cfg = _cfg.get("django_backend", {})

# Configurable parameters with environment variable override support
APP_NAME = os.getenv("DJANGO_MODAL_APP_NAME", _django_cfg.get("app_name", "pulseai-django-backend"))
CPU_COUNT = float(os.getenv("MODAL_DJANGO_CPU", _django_cfg.get("cpu", 2.0)))
MEMORY_MB = int(os.getenv("MODAL_DJANGO_MEMORY_MB", _django_cfg.get("memory_mb", 4096)))
TIMEOUT_SECONDS = int(os.getenv("MODAL_DJANGO_TIMEOUT_S", _django_cfg.get("timeout_seconds", 600)))
SCALEDOWN_WINDOW = int(os.getenv("MODAL_DJANGO_SCALEDOWN_WINDOW", _django_cfg.get("scaledown_window_seconds", 300)))
VLLM_API_URL = os.getenv("PIOS_ORCH_LLM_API_URL", _django_cfg.get("vllm_api_url", "https://yassinetakiko--pulseai-vllm-backend-serve.modal.run"))
DB_ENGINE = os.getenv("DB_ENGINE", _django_cfg.get("db_engine", "django.db.backends.sqlite3"))
PREFER_REMOTE_LLM = "1" if _django_cfg.get("prefer_remote_llm", True) else "0"
DISABLE_LOCAL_LLM = "1" if _django_cfg.get("disable_local_llm", True) else "0"

app = modal.App(APP_NAME)
django_volume = modal.Volume.from_name("pulseai-django-volume", create_if_missing=True)

# Build serverless container image with Django ML dependencies directly in cloud datacenter
django_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1", "curl", "git")
    .pip_install(
        "torch>=2.0.0",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "Django>=5.0",
        "djangorestframework>=3.15.0",
        "django-cors-headers>=4.0",
        "django-filter>=24.0",
        "drf-spectacular>=0.27.0",
        "channels>=4.0",
        "daphne>=4.0",
        "python-dotenv>=1.0.0",
        "python-keycloak>=7.1.1",
        "PyJWT>=2.10.0",
        "cryptography>=43.0.0",
        "pymongo>=4.6.0",
        "psycopg2-binary>=2.9.9",
        "numpy>=1.26.0,<2.3",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "xgboost>=2.0.0",
        "catboost>=1.2",
        "lightgbm>=4.0.0",
        "huggingface-hub>=0.23.0",
        "requests>=2.31.0",
        "mlflow-skinny==2.22.5",
        "APScheduler>=3.10.0",
        "asgiref>=3.8.0",
        "cloudpickle",
        "dill>=0.3.8",
        "psutil>=5.9.0",
        "tqdm>=4.66.0",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/Zanehisako/pulse-ai.git /root/pulse-ai",
        "cd /root/pulse-ai/backendMulti && DB_ENGINE=django.db.backends.sqlite3 PIOS_DISABLE_AUTH=1 python manage.py migrate",
    )
    .env({
        "DJANGO_SETTINGS_MODULE": "backendMulti.settings",
        "DJANGO_DEBUG": "1",
        "DJANGO_ALLOWED_HOSTS": "*",
        "DB_ENGINE": DB_ENGINE,
        "PIOS_DISABLE_AUTH": "1",
        "PIOS_MLFLOW_SUBPROCESS": "0",
        "PIOS_XLAM_DISABLE_LLM": DISABLE_LOCAL_LLM,
        "PIOS_ORCH_PREFER_REMOTE_LLM": PREFER_REMOTE_LLM,
        "PIOS_ORCH_LLM_API_URL": VLLM_API_URL,
        "PYTHONPATH": "/root/pulse-ai/backendMulti:/root/pulse-ai/ml-backend:/root/pulse-ai",
    })
)


def _ensure_model_artifacts():
    """Unpack model artifacts from the persistent Modal volume if needed."""
    import tarfile
    from pathlib import Path
    tar_path = Path("/root/volume/donor_models.tar.gz")
    dest_path = Path("/root/pulse-ai/mlruns/1/3cb18023d4244b7dbae9f073774c03c4")
    if tar_path.exists() and not dest_path.exists():
        print(f"Extracting {tar_path} into /root/pulse-ai...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall("/root/pulse-ai")
        print("✓ Model artifacts extracted successfully.")


@app.function(
    image=django_image,
    volumes={"/root/volume": django_volume},
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_WINDOW,
)
@modal.asgi_app()
def serve():
    """Exposes the full Django ASGI application via an autoscaling HTTPS endpoint."""
    import os
    import sys
    os.chdir("/root/pulse-ai/backendMulti")
    sys.path.insert(0, "/root/pulse-ai/backendMulti")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
    os.environ.setdefault("PIOS_DISABLE_AUTH", "1")
    os.environ.setdefault("PIOS_MLFLOW_SUBPROCESS", "0")
    os.environ.setdefault("DB_ENGINE", DB_ENGINE)

    _ensure_model_artifacts()

    import django
    django.setup()

    # Pre-initialize the tool registry on cold boot
    try:
        from ml.core.startup import initialize
        initialize()
    except Exception as exc:
        print(f"Notice: Background ML registry initialization deferred: {exc}")

    from backendMulti.asgi import application
    return application


@app.function(
    image=django_image,
    volumes={"/root/volume": django_volume},
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
)
def test_orchestrator():
    """Remote test runner to verify orchestrator and tool execution on Modal."""
    import os
    import sys
    os.chdir("/root/pulse-ai/backendMulti")
    sys.path.insert(0, "/root/pulse-ai/backendMulti")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
    os.environ.setdefault("PIOS_DISABLE_AUTH", "1")
    os.environ.setdefault("PIOS_MLFLOW_SUBPROCESS", "0")
    os.environ.setdefault("DB_ENGINE", DB_ENGINE)

    _ensure_model_artifacts()

    import django
    django.setup()

    from ml.core.startup import get_orchestrator, get_registry
    reg = get_registry()
    loaded_models = len(reg.loaded())
    print(f"✓ ModelRegistry initialized with {loaded_models} loaded models.")

    orch = get_orchestrator()
    status = orch.status()
    print(f"✓ Orchestrator status: LLM ready={status.get('llm_ready')}, mode={status.get('planner_mode')}")

    test_query = "Is a 30 year old donor weighing 70kg eligible to donate blood?"
    print(f"Executing test query: '{test_query}'...")
    result = orch.run(
        query=test_query,
        provided_features={"age": 30, "weight": 70},
    )

    tools_used = [r.get("tool") for r in result.get("execution_results", [])]
    print(f"✓ Orchestration success: {result.get('success')}")
    print(f"✓ Tools executed: {tools_used}")

    return {
        "success": result.get("success", False),
        "loaded_models_count": loaded_models,
        "tools_used": tools_used,
        "llm_ready": status.get("llm_ready", False),
    }


@app.local_entrypoint()
def main():
    """CLI test entrypoint: modal run infrastructure/modal/modal_django_server.py"""
    print("Testing PulseAI Django DynamicXLAMOrchestrator on Modal AI...")
    result = test_orchestrator.remote()
    print("\nModal Test Results:")
    print(json.dumps(result, indent=2))
