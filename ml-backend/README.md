# PIOS ML Backend

This backend includes a local Feast Feature Store + MLflow workflow for bloodbank
demand prediction, a dynamic notebook-model training pipeline, an **automated training
scheduler** that retrains models on a cron schedule, a **champion / challenger A/B
testing gate** that only promotes newly trained models to production when their metrics
beat the current champion, and a **drift monitoring system** that detects when
production data distributions shift away from training data.

## What was added

- Feast repo: `ml-backend/feature_repo`
- Pipeline module: `ml-backend/pios_ml_backend/mlops/feast_mlflow_pipeline.py`
- Notebook pipeline module: `ml-backend/pios_ml_backend/mlops/notebook_models_pipeline.py`
- CLI entrypoint: `ml-backend/pios_ml_backend/main.py`
- Training scheduler: `ml-backend/pios_ml_backend/training_scheduler.py`
- Drift detection engine: `ml-backend/pios_ml_backend/drift_detection.py`
- Drift monitor script: `ml-backend/scripts/drift_monitor.py`

## Install

From `ml-backend/`:

```bash
python -m pip install -e .
```

## Official DreamerV3 On Apple Silicon

The official DreamerV3 runner at `ml-backend/simulator/official_dreamerv3.py`
now auto-selects JAX Metal when it finds `jax-metal` in the Dreamer runtime. On macOS,
the runner also enables the JAX compilation cache under the simulator cache directory and
uses backend-aware environment defaults:

- `--jax-platform auto` prefers Metal on Apple Silicon and explains CPU fallback reasons.
- `--envs 0` auto-tunes training environments. On Metal, it uses up to 4 parallel envs.
- `--eval-envs 0` keeps evaluation lightweight with 1 eval environment by default.
- `--jax-profiler auto` disables the JAX profiler on Metal by default because the profiler path is unstable there.
- `--metrics-write-mode buffered` batches `metrics.jsonl` and `scores.jsonl` writes until shutdown or `Ctrl+C`, which avoids constant disk writes during long training runs.
- Full DreamerV3 checkpoints are currently disabled on Metal because the current `jax-metal` PJRT plugin crashes during checkpoint save/load buffer layout inspection.
- The runner defaults to an offline simulator graph context so training startup does not block on live OpenStreetMap requests. Use `--online-city-graph` if you explicitly want the live fetch path.

Example:

```bash
python ml-backend/simulator/official_dreamerv3.py \
  --script train \
  --size size1m \
  --steps 50000 \
  --jax-platform auto \
  --envs 0 \
  --eval-envs 0
```

If the runner reports a CPU fallback on Apple Silicon, install `jax-metal` into the package
location referenced by `PIOS_DREAMERV3_PKGS` and rerun the same command.

The official DreamerV3 runtime itself is not published as a standalone `embodied` package on PyPI.
Use a checkout of the official repository instead:

```bash
git clone https://github.com/danijar/dreamerv3.git ml-backend/simulator/cache/dreamerv3-official
```

On Apple Silicon, install the runtime deps into the Python environment used by
Simulation Studio without the official CUDA extra:

```bash
cd backendMulti
python -m pip install -e ".[dreamerv3]"
```

Or install the same pinned set directly into an existing venv:

```bash
backendMulti/.venv/bin/python -m pip install \
  'jax==0.4.33' 'jaxlib==0.4.33' \
  'elements>=3.19.1' 'portal>=3.5.0' 'ruamel.yaml' \
  'chex' 'ninjax>=3.5.1' 'optax' 'granular>=0.20.3' 'scope>=0.4.4' \
  einops colored_traceback jaxtyping
```

Linux/CUDA environments can still use the official checkout's
`requirements.txt` if they are intentionally setting up the CUDA JAX runtime.

After that, either set `PIOS_DREAMERV3_SRC` to the checkout root, or place the checkout in one of
the repo-local cache locations above so the simulator can discover it automatically.

## DreamerV4 Native PyTorch On Apple Silicon

The native DreamerV4 training backend at `ml-backend/simulator/dreamerv4_agent.py`
now auto-detects Apple Silicon and enables Metal GPU acceleration through PyTorch's
MPS backend, matching the same class of optimisations the DreamerV3 runner uses via
`jax-metal`:

| Optimisation | DreamerV3 (JAX Metal) | DreamerV4 (PyTorch MPS) |
|---|---|---|
| Compiled kernels | JAX XLA compilation cache (`XDG_CACHE_HOME`) | `torch.compile(backend="aot_eager")` on all models |
| Async CPU ↔ GPU transfers | JAX async dispatch | `tensor.to(device, non_blocking=True)` |
| Dynamic GPU memory | `XLA_PYTHON_CLIENT_PREALLOCATE=false` | `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0` |
| Operation fallback | `ENABLE_PJRT_COMPATIBILITY=1` | `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Efficient grad zeroing | N/A (JAX is functional) | `optimizer.zero_grad(set_to_none=True)` |
| GPU sync at boundaries | `jax.block_until_ready` | `torch.mps.synchronize()` before timing / checkpoints |
| Memory release after ckpt | Managed by XLA | `torch.mps.empty_cache()` after checkpoints |
| Pre-allocated GPU buffers | JAX donated buffers | Reusable `_first_mask_buf` tensor on device |

When `--device auto` (the default) selects `mps`, the trainer prints:

```
[DreamerV4] Apple Metal GPU detected — enabling MPS optimisations (torch.compile/aot_eager, async transfers, dynamic memory)
[DreamerV4]   metal_gpu      = ENABLED (MPS + torch.compile/aot_eager)
```

Example — train on Metal GPU (auto-detected):

```bash
python ml-backend/simulator/train_dreamerv4.py native \
  --steps 50000 \
  --logdir dreamerv4_runs/native
```

Explicitly select Metal or force CPU:

```bash
# Force Metal GPU
python ml-backend/simulator/train_dreamerv4.py native --device mps --steps 50000

# Force CPU (disable GPU)
python ml-backend/simulator/train_dreamerv4.py native --device cpu --steps 50000
```

Unlike the DreamerV3 JAX runner, the DreamerV4 native backend has no dependency on
`jax`, `jax-metal`, or any external Dreamer clone — only PyTorch >= 2.0 is required.
Checkpoints save and load correctly on MPS (no PJRT buffer-layout issues).

## DreamerV4 CUDA Notebook Workflow

There is now a CUDA-first notebook at
`ml-backend/notebooks/dreamerv4_cuda_training.ipynb`.

It is backed by `ml-backend/simulator/dreamerv4_notebook.py`, which wraps the native
trainer for notebook use and keeps the checkpoint format identical to the current
simulator runtime:

- training still produces `dreamerv4_agent.pt`
- the notebook publishes that checkpoint into `ml-backend/simulator/dreamerv4_runs/native/ckpt/`
- the existing `dreamerv4` strategy can load the published model without code changes
- `PIOS_DREAMERV4_CHECKPOINT` can point to either the exported `.pt` file or the checkpoint directory

Typical workflow:

```bash
jupyter lab ml-backend/notebooks/dreamerv4_cuda_training.ipynb
```

Inside the notebook:

1. Verify CUDA availability and the active GPU.
2. Adjust the `DreamerV4NotebookConfig`.
3. Run `train_from_notebook(config)`.
4. Validate with `evaluate_checkpoint(...)`.

If you want to move the trained model elsewhere, copy the single
`dreamerv4_agent.pt` file into another simulator `ckpt/` directory and the
current runtime loader will pick it up there as well.

## DreamerV4 Colab Workflow

For Google Colab there is now a separate self-contained notebook:

`ml-backend/notebooks/dreamerv4_colab_training.ipynb`

It is designed for the Colab case specifically:

- does not clone or import this repository at runtime
- defines the simulator, DreamerV4 architecture, training loop, evaluation path, and checkpoint export directly inside the notebook
- installs only the simulator-side DreamerV4 dependencies it needs
- checks that a CUDA runtime is available before training
- downloads the generated drop-in `dreamerv4_agent.pt` from Colab at the end

The easiest path is to open that notebook in Colab and run it top to bottom.
When training finishes, download the generated `dreamerv4_agent.pt` and place it in
the simulator checkpoint directory you already use locally.

## Canonical config sources

Runtime model loading is DB-only:

- Runtime model catalog: `ml_model_config` row keyed by `runtime_models` (or `PIOS_MODEL_CONFIG_KEY`).
- Orchestrator catalog: `ml_model_config` row keyed by `orchestrator_models` (or `PIOS_ORCHESTRATOR_MODEL_CONFIG_KEY`).
- If runtime catalog is empty, startup/manual reload can auto-seed DB entries from local `models/*.pkl`.

JSON files are kept only for notebook training inputs:

- `ml-backend/notebooks/ml_models/config.json`
  - Notebook model catalog used by `mlops-train-notebooks`.
- `ml-backend/notebooks/ml_models/training_spec.json`
  - Training defaults/spec consumed by the notebook training pipeline.

## Active project scope

The maintained runtime/training flow is intentionally centered on:

- `ml-backend/pios_ml_backend`
- `ml-backend/feature_repo`
- `ml-backend/notebooks/ml_models`
- `ml-backend/datasets`
- `ml-backend/models`

Generated artifacts (MLflow DBs/runs, Feast online/registry DBs, generated feature definitions)
are recreated at runtime and are not considered source-of-truth.

---

## MLflow Tracking — Docker & Local

All scripts (training, promotion, sync, drift) use the **same** MLflow tracking
URI so that every run, metric, and artifact ends up in one place:

```python
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8889"))
```

### Docker (recommended — production)

The Docker Compose stack at `infrastructure/docker/docker-compose.local.yaml`
runs an MLflow server container backed by PostgreSQL:

```
┌────────────┐      ┌──────────────┐      ┌───────────┐
│ Training   │─────▶│ MLflow       │─────▶│ PostgreSQL│
│ Scripts    │ HTTP │ Server       │  SQL │ (pios DB) │
│ Drift Mon. │:8889 │ :8889        │      │ :5432     │
│ Promotion  │      │              │      │           │
└────────────┘      │ Artifacts:   │      └───────────┘
                    │ /mlflow/     │
                    │  artifacts/  │
                    └──────────────┘
```

Start the stack:

```bash
cd infrastructure/docker
docker compose -f docker-compose.local.yaml up -d postgres mlflow
```

Open the **MLflow UI** at **http://localhost:8889** — all experiments (training,
drift monitoring, notebooks) appear here automatically.

### Local development (no Docker)

If you are running without Docker, you have two options:

**Option A — File-based store (simplest):**

```bash
export MLFLOW_TRACKING_URI=file:///absolute/path/to/ml-backend/training_scripts/mlruns
```

Then start the MLflow UI:

```bash
cd ml-backend/training_scripts
mlflow ui --port 5000
# Open http://localhost:5000
```

**Option B — Local SQLite store:**

```bash
export MLFLOW_TRACKING_URI=sqlite:///$(pwd)/ml-backend/mlflow.db
```

Then start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///ml-backend/mlflow.db --port 5000
```

> **Important:** Every script defaults to `http://localhost:8889`. If the Docker
> MLflow server is not running and you haven't set `MLFLOW_TRACKING_URI`, scripts
> will fail to connect. Always either start the Docker stack or export the env var.

---

## Automated Training Scheduler

The backend includes a background training scheduler powered by
[APScheduler](https://apscheduler.readthedocs.io/). It runs inside a
long-lived host process used by the retained training utilities and Django
integration, so models are retrained automatically on a configurable cron schedule.

### How it works

```
 23:30 UTC ─ APScheduler CronTrigger fires
     │
     ├─ 1. TRAIN ─ run training script (subprocess)
     │      → trains model, registers in MLflow as "challenger"
     │
     ├─ 2. PROMOTE ─ run promotion gate (promote_models.py)
     │      → compares challenger vs current champion metrics
     │      → promotes ONLY if challenger is better (A/B gate)
     │      → if worse: keeps old champion, discards challenger alias
     │
     ├─ 3. SYNC ─ run sync_registry.py (subprocess)
     │      → reads MLflow registry, upserts Django MLModelConfig
     │      → only the "champion" version is synced to production
     │
     ├─ 4. REFRESH ─ post-sync hook (in-process)
     │      → Django: refresh_runtime_state() → ModelRegistry.refresh()
     │      → host runtime refreshes in-memory model state after sync
     │      → new models served on next request — zero downtime
     │
     └─ 5. DRIFT CHECK ─ run drift_monitor.py (subprocess, optional)
            → compares training data vs current data distributions
            → logs per-feature metrics to MLflow
            → persists DriftReport to Django DB
```

### Configuration — `config/training_schedule.json`

All scheduling, promotion rules, drift monitoring, and model mappings live in a
single JSON file at `ml-backend/config/training_schedule.json`:

```json
{
  "defaults": {
    "time": "23:30",
    "timezone": "UTC"
  },

  "scripts": {
    "blood_shortage_predictor": {
      "script": "blood_shortage_predictor.py",
      "registered_model_name": "hospital_shortage_predictor",
      "frequency": "daily",
      "time": "23:30",
      "enabled": true,
      "description": "XGBoost blood shortage prediction model"
    },
    "ideal_donor": {
      "script": "ideal_donor.py",
      "registered_model_name": "donor_propensity_model",
      "frequency": "daily",
      "time": "23:35",
      "enabled": true,
      "description": "Ideal donor scoring model"
    },
    "stockout_lstm_ensemble": {
      "script": "train_stockout_lstm.py",
      "registered_model_name": "stockout_days_predictor",
      "frequency": "weekly",
      "day_of_week": "sun",
      "time": "02:00",
      "enabled": true,
      "description": "LSTM stacking ensemble (heavy — runs weekly)"
    }
  },

  "post_training": {
    "sync_registry": true
  },

  "drift_monitoring": {
    "enabled": true,
    "frequency": "daily",
    "time": "06:00",
    "run_after_training": true,
    "thresholds": {
      "psi": 0.2,
      "ks_pvalue": 0.05,
      "psi_warning": 0.1
    },
    "per_model": {
      "hospital_shortage_predictor": {
        "enabled": true,
        "frequency": "daily",
        "time": "06:00"
      },
      "donor_propensity_model": {
        "enabled": true,
        "frequency": "daily",
        "time": "06:05"
      },
      "stockout_days_predictor": {
        "enabled": true,
        "frequency": "weekly",
        "day_of_week": "mon",
        "time": "06:10"
      }
    }
  },

  "promotion_rules": {
    "default": {
      "primary_metric": "root_mean_squared_error",
      "direction": "minimize",
      "min_improvement_pct": 0.0,
      "fallback_metrics": ["mean_absolute_error"]
    },
    "hospital_shortage_predictor": {
      "primary_metric": "root_mean_squared_error",
      "direction": "minimize",
      "min_improvement_pct": 0.0,
      "fallback_metrics": ["mean_absolute_error", "r2_score"]
    },
    "donor_propensity_model": {
      "primary_metric": "root_mean_squared_error",
      "direction": "minimize",
      "min_improvement_pct": 0.0,
      "fallback_metrics": ["mean_absolute_error", "r2_score"]
    },
    "stockout_days_predictor": {
      "primary_metric": "rmse",
      "direction": "minimize",
      "min_improvement_pct": 0.0,
      "fallback_metrics": ["mae", "r2"]
    }
  }
}
```

#### Script entry fields

| Field | Required | Description |
|---|---|---|
| `script` | yes | Filename inside `ml-backend/training_scripts/` |
| `registered_model_name` | yes | The MLflow registered model name (used by promotion gate) |
| `frequency` | yes | `daily`, `weekly`, or `monthly` |
| `time` | no | `"HH:MM"` 24-hour format (falls back to `defaults.time`) |
| `day_of_week` | weekly only | `mon` / `tue` / `wed` / `thu` / `fri` / `sat` / `sun` |
| `day_of_month` | monthly only | `1`–`28` |
| `enabled` | no | `true` / `false` (default `true`) |
| `description` | no | Human-readable note |

#### Promotion rule fields

| Field | Default | Description |
|---|---|---|
| `primary_metric` | `root_mean_squared_error` | MLflow metric name to compare |
| `direction` | `minimize` | `minimize` for error metrics (RMSE, MAE, loss) or `maximize` for quality metrics (R², AUC, accuracy) |
| `min_improvement_pct` | `0.0` | Minimum % improvement required (`0` = just match, `2` = must be 2% better) |
| `fallback_metrics` | `[]` | Metrics to try when the primary is unavailable on one or both runs |

Per-model rules override the `default` entry. Any field not specified in a
per-model override inherits from `default`.

#### Drift monitoring fields

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Master switch for scheduled drift checks |
| `frequency` | `daily` | How often the standalone drift jobs run |
| `time` | `06:00` | When the drift check fires (HH:MM) |
| `run_after_training` | `true` | Also run a drift check after each training + promotion cycle |
| `thresholds.psi` | `0.2` | PSI above this → critical drift |
| `thresholds.ks_pvalue` | `0.05` | KS p-value below this → drift detected |
| `thresholds.psi_warning` | `0.1` | PSI between warning and critical → warning severity |
| `per_model` | `{}` | Per-model schedule overrides (same fields as script entries) |

---

## Champion / Challenger A/B Testing

Every training script registers its newly trained model with the **`challenger`**
alias in MLflow (not `champion`). A separate promotion gate then decides whether
the challenger deserves to replace the current champion.

### Promotion gate logic (`scripts/promote_models.py`)

For each registered model that has a `challenger` alias:

1. **No existing champion** → auto-promote the challenger.
2. **Challenger is the same version as champion** → clean up alias, skip.
3. **Both exist** → fetch MLflow run metrics for both versions:
   - Compare the `primary_metric` configured in `promotion_rules`.
   - If the challenger is better (respecting `direction` and `min_improvement_pct`) → **promote**: move `champion` alias to the challenger version, remove `challenger` alias.
   - If the challenger is worse → **reject**: keep the old champion, remove `challenger` alias. The old model stays in production.
4. **No comparable metrics found** → auto-promote (benefit of the doubt).

### Running the promotion gate manually

```bash
# Check ALL registered models
python ml-backend/scripts/promote_models.py

# Check a single model
python ml-backend/scripts/promote_models.py --model-name hospital_shortage_predictor
```

The script outputs a JSON summary line prefixed with `[promote] RESULTS_JSON:` for
machine-readable consumption.

### Example: requiring 2% improvement

To require that a challenger must be at least 2% better than the current champion
before it gets promoted, set `min_improvement_pct` in `training_schedule.json`:

```json
"hospital_shortage_predictor": {
  "primary_metric": "root_mean_squared_error",
  "direction": "minimize",
  "min_improvement_pct": 2.0,
  "fallback_metrics": ["mean_absolute_error"]
}
```

With `direction: "minimize"`, the challenger's RMSE must be ≤ 98% of the champion's
RMSE. With `direction: "maximize"`, the challenger's metric must be ≥ 102% of the
champion's.

---

## Drift Monitoring

Drift detection checks whether the data distributions your models see in
production have shifted compared to the data they were trained on. When drift
is detected it's a signal that the model may need retraining.

### What it checks

| Test | Applied to | Metric | Threshold |
|---|---|---|---|
| **Kolmogorov–Smirnov** | Numerical features | KS statistic + p-value | p < 0.05 |
| **Population Stability Index (PSI)** | Numerical features | PSI score | > 0.2 critical, > 0.1 warning |
| **Chi-squared** | Categorical features | χ² statistic + p-value | p < 0.05 |
| **Jensen–Shannon divergence** | Any distribution | JS divergence | > 0.1 |
| **Prediction drift** | Model outputs | KS on predictions | p < 0.05 |

The drift engine at `pios_ml_backend/drift_detection.py` is a pure-Python
statistics module using only `scipy`, `numpy`, and `pandas`. It has **no**
Django or MLflow dependencies — the callers handle integration.

### Where results are stored

1. **MLflow** (primary) — Each drift check creates a run in the
   `model_drift_monitoring` experiment. Open the MLflow UI
   (http://localhost:8889 with Docker, or wherever you started `mlflow ui`) and
   navigate to that experiment to see:
   - Per-feature metrics: `drift.<feature>.ks_statistic`, `drift.<feature>.psi`,
     `drift.<feature>.drift_detected`, etc.
   - Overall metrics: `drift.overall_score`, `drift.overall_drift_detected`,
     `drift.features_drifted_count`
   - A `drift_report.json` artifact with the full per-feature breakdown
   - Tags: `model_name`, `check_type`, `production_mode`
   - Chart `drift.overall_score` over time per model tag to visualize drift trends

2. **Django DB** — A `DriftReport` row is persisted for each check, queryable
   via the REST API endpoints (see below). This feeds dashboards.

### MLflow tracking URI

The drift monitor uses the **same** tracking URI as every other script:

```python
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8889"))
```

This means:

- **Docker running** → data goes to PostgreSQL via the MLflow server → visible
  at http://localhost:8889.
- **No Docker, env var set** → data goes wherever `MLFLOW_TRACKING_URI` points
  (file-based `mlruns/`, SQLite, remote server, etc.).
- **No Docker, no env var** → scripts fail to connect. You must either start
  Docker or set the env var.

### Running drift checks

```bash
# Check all models
python scripts/drift_monitor.py

# Check a single model
python scripts/drift_monitor.py --model-name hospital_shortage_predictor

# Use PredictionLog data as "current" (production mode)
python scripts/drift_monitor.py --production-mode

# Override thresholds
python scripts/drift_monitor.py --threshold-psi 0.15 --threshold-ks 0.01

# Limit reference window (last N days of data)
python scripts/drift_monitor.py --reference-window 90
```

### Scheduled drift checks

Drift monitoring is configured in `config/training_schedule.json` under the
`drift_monitoring` section (see [Configuration](#configuration--configtraining_schedulejson)
above for the full example).

When `run_after_training` is `true`, a drift check automatically runs after
every training + promotion + sync cycle. The `per_model` section schedules
independent drift-only jobs (useful for monitoring between training runs).

### Drift Monitoring API Endpoints

**Django** (under `/api/ml/drift/`):

| Endpoint | Method | Description |
|---|---|---|
| `/api/ml/drift/status/` | `GET` | Latest drift result per model (one row each) |
| `/api/ml/drift/reports/` | `GET` | List all drift reports (supports `?limit=`, `?offset=`, `?drift_only=true`) |
| `/api/ml/drift/reports/<model_id>/` | `GET` | Drift reports for a specific model |
| `/api/ml/drift/run/` | `POST` | Manually trigger drift detection |

The retained drift API surface is exposed through Django under `/api/ml/drift/`.
Historical FastAPI endpoints have been removed from `ml-backend`.

### Drift API Examples

```bash
# Check drift status for all models
curl http://localhost:8000/api/ml/drift/status/

# Get detailed drift reports for one model
curl http://localhost:8000/api/ml/drift/reports/hospital_shortage_predictor/

# Get only reports where drift was detected
curl "http://localhost:8000/api/ml/drift/reports/?drift_only=true"

# Manually trigger drift check for one model
curl -X POST http://localhost:8000/api/ml/drift/run/ \
  -H "Content-Type: application/json" \
  -d '{"model_name": "donor_propensity_model"}'

# Trigger drift check for all models
curl -X POST http://localhost:8000/api/ml/drift/run/
```

### Severity levels

| Severity | Meaning | Trigger |
|---|---|---|
| `none` | No drift detected | All tests pass |
| `warning` | Moderate shift | PSI 0.1–0.2 or borderline p-values |
| `critical` | Significant drift | PSI > 0.2 or KS p < 0.05 for multiple features |

The `drift_score` field (0.0–1.0) represents the fraction of features that
exhibited drift. A score of 0.3 means 30% of features shifted significantly.

### Drift Detection Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  pios_ml_backend/drift_detection.py              │
│                  (pure Python — scipy, numpy, pandas)            │
│                                                                  │
│  compute_ks_test()     compute_psi()      compute_chi2_test()   │
│  compute_js_divergence()                                         │
│  detect_feature_drift()   detect_prediction_drift()             │
│  build_drift_report()  ← main entry point                       │
└───────────────────────────┬──────────────────────────────────────┘
                            │ imported by
          ┌─────────────────┼─────────────────────┐
          ▼                 ▼                     ▼
  scripts/              training_             Django views
  drift_monitor.py      scheduler.py          (manual trigger
  (CLI + MLflow +       (subprocess            + dashboard
   Django DB)            after training         queries)
                         or on cron)
          │                                       │
          ▼                                       ▼
  ┌───────────────┐                    ┌──────────────────┐
  │ MLflow Server │                    │ Django DriftReport│
  │ (PostgreSQL   │                    │ table             │
  │  + artifacts) │                    │ (ml_drift_report) │
  └───────────────┘                    └──────────────────┘
```

### Django DriftReport Model

The `DriftReport` model stores drift results for the REST API and dashboards:

| Field | Type | Description |
|---|---|---|
| `model_id` | CharField | Model name (e.g., `hospital_shortage_predictor`) |
| `mlflow_run_id` | CharField | MLflow run ID for cross-reference |
| `drift_detected` | BooleanField | Whether any feature drifted significantly |
| `drift_score` | FloatField | Fraction of features drifted (0.0–1.0) |
| `severity` | CharField | `none`, `warning`, or `critical` |
| `features_checked` | IntegerField | Number of features tested |
| `features_drifted` | IntegerField | Number that showed drift |
| `feature_details` | JSONField | Full per-feature breakdown (KS, PSI, chi², severity per feature) |
| `reference_size` | IntegerField | Number of rows in the reference dataset |
| `current_size` | IntegerField | Number of rows in the current dataset |
| `check_type` | CharField | `scheduled`, `manual`, or `post_training` |
| `checked_at` | DateTimeField | When the check ran |

Migration: `ml/migrations/0005_add_drift_report.py`

---

## Scheduler Django API Endpoints

All endpoints are under `/api/ml/scheduler/`:

| Endpoint | Method | Description |
|---|---|---|
| `/api/ml/scheduler/status/` | `GET` | List all scheduled jobs with next run times |
| `/api/ml/scheduler/reload/` | `POST` | Hot-reload `training_schedule.json` without restart |
| `/api/ml/scheduler/run/` | `POST` | Manually trigger training now |
| `/api/ml/scheduler/config/` | `GET` | Read the current schedule configuration |
| `/api/ml/scheduler/config/` | `PUT` | Update configuration (merge) and auto-reload |

### Scheduler API Examples

```bash
# Check what's scheduled and when
curl http://localhost:8000/api/ml/scheduler/status/

# Manually train all enabled models right now
curl -X POST http://localhost:8000/api/ml/scheduler/run/

# Manually train a single model
curl -X POST http://localhost:8000/api/ml/scheduler/run/ \
  -H "Content-Type: application/json" \
  -d '{"model_key": "blood_shortage_predictor"}'

# Hot-reload the config after editing training_schedule.json
curl -X POST http://localhost:8000/api/ml/scheduler/reload/

# Read current config
curl http://localhost:8000/api/ml/scheduler/config/

# Disable a model via API (merges into existing config)
curl -X PUT http://localhost:8000/api/ml/scheduler/config/ \
  -H "Content-Type: application/json" \
  -d '{
    "scripts": {
      "stockout_lstm_ensemble": { "enabled": false }
    }
  }'
```

The `/api/ml/health/` endpoint also includes a `training_scheduler` section
showing all scheduled jobs.

The retained scheduler endpoints are exposed through Django under
`/api/ml/scheduler/`.

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/train_models.py` | Original all-in-one: runs all training scripts + syncs registry (still works standalone) |
| `scripts/sync_registry.py` | Sync-only: reads MLflow registry → upserts Django `MLModelConfig` (no training) |
| `scripts/promote_models.py` | A/B gate: compares challenger vs champion metrics, promotes only if better |
| `scripts/drift_monitor.py` | Drift detection: compares reference vs current data distributions, logs to MLflow + Django |

All scripts use the same MLflow tracking URI:

```python
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8889"))
```

---

## Overall Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  config/training_schedule.json                                     │
│    ├─ scripts (what to train, when, how often)                     │
│    ├─ promotion_rules (A/B metric comparison per model)            │
│    ├─ drift_monitoring (drift check schedule + thresholds)         │
│    └─ post_training (sync_registry toggle)                         │
│                                                                    │
│  pios_ml_backend/training_scheduler.py  ← core scheduler engine    │
│    • APScheduler BackgroundScheduler + CronTrigger                  │
│    • Runs training scripts as isolated subprocesses                 │
│    • Runs promotion gate after each successful training             │
│    • Runs registry sync after promotion                             │
│    • Runs drift checks (after training or on independent schedule)  │
│    • Calls post-sync hook to refresh in-memory models               │
│                                                                    │
│  pios_ml_backend/drift_detection.py  ← pure stats engine           │
│    • KS, PSI, Chi-squared, Jensen-Shannon tests                     │
│    • No Django/MLflow dependency — imported by callers              │
│                                                                    │
│  ┌──────────────────────────────┐  ┌────────────────────────────┐  │
│  │ Retained training utilities  │  │ Django (ml/ app)           │  │
│  │  scheduler + CLI workflows   │  │  /api/ml/scheduler/*       │  │
│  │  drift + MLflow integration  │  │  /api/ml/drift/*           │  │
│  │  post_sync_hook support      │  │  /api/ml/predict/*         │  │
│  │                              │  │  post_sync_hook =          │  │
│  │                              │  │  refresh_runtime_state()   │  │
│  └──────────────────────────────┘  └────────────────────────────┘  │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Docker (infrastructure/docker/docker-compose.local.yaml)    │   │
│  │  postgres:5432   ← Django DB + MLflow backend store         │   │
│  │  mlflow:8889     ← MLflow server (UI + artifact serving)    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

The core scheduler module has no Django dependency — it uses `subprocess` and
`Path` calculations only. The Django bridge at
`backendMulti/ml/services/training_scheduler.py` adds `ml-backend/` to
`sys.path` and re-exports the scheduler functions with a Django-specific
post-sync hook that calls `refresh_runtime_state()`.

---

## Training and MLOps utilities

The training utilities include the dynamic xLAM orchestrator logic from
`notebooks/xlma-orchestrator.ipynb` for model workflows and orchestration support.

- `POST /predict/nl`: natural language orchestration + tool/function calling
- `POST /models/{model_id}/predict`: same orchestrator, but tool scope forced to one model
- `POST /predict/model/{model_slug}`: static alias endpoint per model, still orchestrator-backed
- `PUT /config/runtime`: update full runtime model catalog in Postgres (hot reload)
- `POST /config/runtime/reload`: force immediate reload of runtime catalog
- `GET /config/runtime`: inspect current runtime-config source/listener state
- `GET /orchestrator/models`: list downloadable orchestrator models with metadata and local availability
- `PUT /orchestrator/models`: replace orchestrator model catalog (models + selected id) in Postgres
- `PUT /orchestrator/models/selected`: set selected orchestrator model id (persisted + hot applied)
- `POST /orchestrator/models/{model_id}/download`: trigger download/warmup for a catalog model id
- `GET /scheduler/*`: training scheduler status, config
- `POST /scheduler/*`: reload config, trigger training
- `GET /drift/*`: drift monitoring status, reports
- `POST /drift/run`: trigger drift detection

The orchestrator auto-checks available RAM/VRAM and selects a smaller GGUF variant when
resources are low.
At server startup, it checks/downloads/loads the xLAM GGUF once (background by default), so
query requests do not trigger repeated model checks.

### Run locally

From `ml-backend/`:

```bash
python -m pios_ml_backend.main run
```

### Build and run container

```bash
docker build -t pios-ml-api .
docker run --rm -p 8000:8000 pios-ml-api
```

### Test endpoints

```bash
# Health
curl http://localhost:8000/health

# API root helper
curl http://localhost:8000/

# List loaded models + endpoint paths
curl http://localhost:8000/models

# Runtime config listener/status
curl http://localhost:8000/config/runtime

# Replace runtime model catalog (hot applies in-process via LISTEN/NOTIFY)
curl -X PUT http://localhost:8000/config/runtime \
  -H "Content-Type: application/json" \
  -d '{
    "models": [
      {
        "id": "donor_prediction",
        "description": "Predict donor eligibility",
        "file_path": "models/donor_v1.pkl",
        "features": ["recency_days", "donation_count_last_12m", "age", "bmi", "sex"]
      }
    ]
  }'

# Orchestrator status (selected model variant, RAM/VRAM profile, fallback/LLM mode)
curl http://localhost:8000/orchestrator/status

# List available orchestrator models to download (metadata + local state)
curl http://localhost:8000/orchestrator/models

# Persist and switch selected orchestrator model (hot applied)
curl -X PUT http://localhost:8000/orchestrator/models/selected \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "xlam_7b_q4_k_m",
    "download": true,
    "wait": false
  }'

# Replace full orchestrator model catalog in Postgres
curl -X PUT http://localhost:8000/orchestrator/models \
  -H "Content-Type: application/json" \
  -d '{
    "selected_model_id": "xlam_7b_q4_k_m",
    "models": [
      {
        "id": "xlam_7b_q4_k_m",
        "name": "xLAM 7B Q4_K_M",
        "description": "Balanced profile",
        "repo_id": "bartowski/xLAM-7b-fc-r-GGUF",
        "filename": "xLAM-7b-fc-r-Q4_K_M.gguf",
        "size_mb": 4200,
        "min_ram_gb": 8,
        "min_vram_mb": 5000,
        "n_ctx": 4096,
        "n_batch": 256
      }
    ]
  }'

# Download/warmup by model id
curl -X POST "http://localhost:8000/orchestrator/models/xlam_7b_q4_k_m/download?wait=false"

# Start model warmup/download in background (non-blocking)
curl -X POST http://localhost:8000/orchestrator/warmup

# Blocking warmup (waits until download/load completes or fails)
curl -X POST "http://localhost:8000/orchestrator/warmup?wait=true"

# Natural-language orchestration endpoint
curl -X POST http://localhost:8000/predict/nl \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare two donors: first age 35 bmi 24 recency 30, second age 42 bmi 28 recency 15. Who is more likely to donate?"
  }'

# Alternative (no JSON body)
curl -X POST "http://localhost:8000/predict/nl?query=Check%20donor%20age%2035%20bmi%2024%20recency%2045"

# Per-model endpoint (still runs through orchestrator planner/executor)
curl -X POST http://localhost:8000/models/donor_v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Evaluate donor risk with these values",
    "features": {
      "age": 35,
      "sex": "M",
      "bmi": 24,
      "recency_days": 45,
      "donation_count_last_12m": 3
    }
  }'

# Static per-model endpoint alias (same behavior as above)
curl -X POST http://localhost:8000/predict/model/donor_v1 \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 35,
      "sex": "M",
      "bmi": 24,
      "recency_days": 45,
      "donation_count_last_12m": 3
    }
  }'
```

Swagger docs are available at:

- `http://localhost:8000/docs`

### Orchestrator environment variables

- `PIOS_XLAM_REPO_ID`: Hugging Face GGUF repo (default: `bartowski/xLAM-7b-fc-r-GGUF`)
- `PIOS_XLAM_FILENAME`: force a specific GGUF filename instead of auto-selection
- `PIOS_XLAM_MODEL_DIR`: local folder to store GGUF files (auto-download target)
- `PIOS_XLAM_MODEL_PATH`: local GGUF file path, or a subfolder path under `PIOS_XLAM_MODEL_DIR`
- `PIOS_XLAM_N_CTX`: override context size
- `PIOS_XLAM_N_BATCH`: override batch size
- `PIOS_XLAM_N_GPU_LAYERS`: override GPU layer count
- `PIOS_XLAM_DISABLE_LLM=1`: disable xLAM load (forces fallback planner)
- `PIOS_XLAM_ALLOW_FALLBACK=0`: fail if xLAM cannot load instead of fallback
- `PIOS_XLAM_INIT_MODE`: `background` (default) or `blocking`
- `PIOS_XLAM_DISABLE_PROGRESS=1`: hide HF download progress bars (default shows progress bars)
- `PIOS_XLAM_PLANNER_TOOLS`: max tools included in LLM planning prompt (default `4`)
- `PIOS_XLAM_PLANNER_FEATURES`: max feature hints per tool in planning prompt (default `8`)
- `PIOS_CONFIG_DB_BACKEND`: config DB backend (`sqlite` default, or `postgres`)
- `PIOS_CONFIG_DB_PATH`: SQLite path (default `ml-backend/data/config.db`)
- In docker-compose mode, set `PIOS_CONFIG_DB_BACKEND=postgres`.
- `PIOS_MODEL_CONFIG_KEY`: config key used for runtime model catalog (default `runtime_models`)
- `PIOS_MODEL_CONFIG_CHANNEL`: Postgres notify channel for hot reload (default `pios_model_config_changed`, postgres backend only)
- `PIOS_MODEL_CONFIG_POLL_SECONDS`: fallback poll interval when notify is missed (default `15`)
- `PIOS_MODEL_CONFIG_RECONNECT_SECONDS`: retry delay if listener DB connection drops (default `5`)
- `PIOS_ORCHESTRATOR_MODEL_CONFIG_KEY`: config key used for downloadable orchestrator models (default `orchestrator_models`)
- `PIOS_XLAM_MODEL_ID`: preferred orchestrator model id from the orchestrator catalog
- `DB_HOST`: Postgres host (default `localhost`; use `postgres` in docker-compose network, postgres backend only)
- `DB_PORT`: Postgres port (default `5432`)
- `DB_NAME`: Postgres database name (default `pios`)
- `DB_USER`: Postgres user (default `ml_user`)
- `DB_PASSWORD`: Postgres password (default `ml_pass`)
- `DB_CONNECT_TIMEOUT`: DB connect timeout in seconds for config lookups/listener reconnects (default `3`)
- `PIOS_AUTO_START_DB`: when `1` (default), startup tries to run `docker compose ... up -d postgres` if postgres backend is selected and DB is unreachable on local host.
- `PIOS_DB_COMPOSE_FILE`: optional override for the compose file used by postgres auto-start (default `infrastructure/docker-compose.yaml`).

## Feast + MLflow quick start

From `ml-backend/`:

```bash
# 1) Build Feast source parquet from datasets/synthetic_bloodbank_daily.csv
python -m pios_ml_backend.main mlops-prepare

# 2) Register Feast feature definitions
python -m pios_ml_backend.main mlops-apply

# 3) Train with Feast historical features and log to MLflow
# (includes online materialization by default)
python -m pios_ml_backend.main mlops-train

# 4) Score online features for one or more entities
python -m pios_ml_backend.main mlops-score --inventory-id Central__A+ --inventory-id Central__O-
```

To list valid entity ids generated from the dataset:

```bash
python -m pios_ml_backend.main mlops-entities --limit 30
```

## Notebook Models -> Feast + MLflow (single command)

This pipeline trains any number of notebook-defined models from
`notebooks/ml_models/config.json`, generates Feast feature definitions dynamically, and
logs all runs/models to MLflow:

```bash
python -m pios_ml_backend.main mlops-train-notebooks
```

Useful options:

```bash
# Train only selected model ids
python -m pios_ml_backend.main mlops-train-notebooks \
  --model-id donor_prediction \
  --model-id xgb_demand_forecast_j+30

# Override config/spec files
python -m pios_ml_backend.main mlops-train-notebooks \
  --model-config notebooks/ml_models/config.json \
  --training-spec notebooks/ml_models/training_spec.json

# Fail if any requested feature is missing (no auto-fill)
python -m pios_ml_backend.main mlops-train-notebooks --strict-features

# Register models in MLflow registry
python -m pios_ml_backend.main mlops-train-notebooks --register-models
```

Default training metadata for notebook models lives in:

- `ml-backend/notebooks/ml_models/training_spec.json`

## Commands

- `python -m pios_ml_backend.main run`
  - Initializes DB, starts the training scheduler, then runs a service loop.
- `python -m pios_ml_backend.main init-db`
  - Initializes Postgres config tables.
- `python -m pios_ml_backend.main mlops-prepare`
  - Writes Feast source parquet to `feature_repo/data/bloodbank_features.parquet`.
- `python -m pios_ml_backend.main mlops-apply`
  - Runs `feast apply` in `feature_repo`.
- `python -m pios_ml_backend.main mlops-materialize`
  - Materializes online features incrementally.
- `python -m pios_ml_backend.main mlops-train [--skip-materialize]`
  - Trains a logistic regression model with Feast historical features and logs run/model in MLflow.
- `python -m pios_ml_backend.main mlops-train-notebooks [options]`
  - Trains all notebook-defined models dynamically (features + Feast views + MLflow logging).
- `python -m pios_ml_backend.main mlops-score --inventory-id <hospital__blood_type> [--run-id <id>]`
  - Loads model from MLflow and scores live features from Feast online store.

## Optional environment variables

- `PIOS_FEATURE_SOURCE_CSV`
- `PIOS_FEAST_REPO`
- `PIOS_FEAST_FEATURE_PARQUET`
- `MLFLOW_TRACKING_URI` — **critical**: defaults to `http://localhost:8889` (Docker MLflow server)
- `MLFLOW_EXPERIMENT`
- `PIOS_MLFLOW_LAST_RUN_FILE`
- `MLFLOW_EXPERIMENT_NOTEBOOKS`
- `PIOS_MLFLOW_NOTEBOOK_RUNS_FILE`

## Dependencies (scheduler + drift)

The training scheduler requires `apscheduler` and drift detection requires
`scipy` (both already listed in `pyproject.toml`):

```bash
pip install "apscheduler>=3.10,<4.0"
# scipy is pulled in by scikit-learn which is already a dependency
```

These are installed automatically with `pip install -e .`.

---

## Quick Start (full stack)

```bash
# 1. Start Docker services (PostgreSQL + MLflow)
cd infrastructure/docker
docker compose -f docker-compose.local.yaml up -d postgres mlflow

# 2. Install ML backend
cd ../../ml-backend
pip install -e .

# 3. Apply Feast features
python -m pios_ml_backend.main mlops-apply

# 4. Run training (all models)
python scripts/train_models.py

# 5. Run drift detection
python scripts/drift_monitor.py

# 6. Open MLflow UI
#    → http://localhost:8889
#    → Click "model_drift_monitoring" experiment for drift results
#    → Click training experiments for model metrics

# 7. Start the training utilities loop (scheduler runs in background)
python -m pios_ml_backend.main run

# 8. Check drift status through the retained training and Django-integrated workflows
curl http://localhost:8000/drift/status
```
