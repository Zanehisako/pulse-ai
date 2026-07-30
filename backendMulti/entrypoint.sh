#!/bin/sh
set -e

echo "[backendmulti] Waiting for PostgreSQL at ${DB_HOST:-postgres}:${DB_PORT:-5432}..."
python - <<'PY'
import os, time, psycopg2

host     = os.getenv("DB_HOST", "postgres")
port     = int(os.getenv("DB_PORT", "5432"))
name     = os.getenv("DB_NAME", "pios")
user     = os.getenv("DB_USER", "admin")
password = os.getenv("DB_PASSWORD", "admin")

for attempt in range(1, 61):
    try:
        psycopg2.connect(host=host, port=port, dbname=name, user=user, password=password).close()
        print("[backendmulti] PostgreSQL is ready.")
        break
    except Exception as exc:
        if attempt == 60:
            raise SystemExit(f"[backendmulti] PostgreSQL not reachable after 60 attempts: {exc}")
        print(f"[backendmulti] DB not ready (attempt {attempt}/60): {exc}")
        time.sleep(2)
PY

echo "[backendmulti] Applying migrations..."
python manage.py migrate --noinput --run-syncdb

echo "[backendmulti] Syncing ML config into DB..."
python manage.py import_ml_config

echo "[backendmulti] Generating SOTA model stats for dashboard cards..."
export PIOS_MODELS_DIR="${PIOS_MODELS_DIR:-/app/ml/models}"
export PIOS_SOTA_CATALOG_PATH="${PIOS_SOTA_CATALOG_PATH:-/ml-backend/config/sota_model_catalog.json}"
export PIOS_SOTA_DATASETS_DIR="${PIOS_SOTA_DATASETS_DIR:-/ml-backend/datasets_featues_labels_seperated}"
if python manage.py generate_sota_model_stats; then
    echo "[backendmulti] SOTA model stats are ready."
else
    echo "[backendmulti] SOTA model stats generation failed — dashboard will use existing stats if present."
fi

# ── SEED PIPELINE ──────────────────────────────────────────────────────────────
echo "[backendmulti] Checking MLflow availability..."
if python - <<'PY'
import os, sys, urllib.request
uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:8889")
try:
    urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=5)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
then
    echo "[backendmulti] MLflow is reachable — predictions enabled."
    echo "[backendmulti] Preloading configured prediction models..."
    if python manage.py preload_prediction_models; then
        echo "[backendmulti] Prediction models are ready."
        SEED_FLAGS=""
    else
        echo "[backendmulti] Prediction model preload failed — skipping live predictions only (synthetic history still seeds)."
        SEED_FLAGS="--skip-predict-live"
    fi
else
    echo "[backendmulti] MLflow not reachable — skipping live predictions only (synthetic history still seeds)."
    SEED_FLAGS="--skip-predict-live"
fi

echo "[backendmulti] Running seed_all (force reseed for clean state)..."
python manage.py seed_all --force $SEED_FLAGS

# ── SERVER ────────────────────────────────────────────────────────────────────
echo "[backendmulti] Starting Daphne ASGI server..."
exec daphne -b 0.0.0.0 -p 8000 backendMulti.asgi:application
