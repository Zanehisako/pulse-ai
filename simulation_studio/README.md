# Simulation Studio

`simulation_studio` is the standalone FastAPI web app for the blood supply simulator. It serves a browser UI, REST APIs, and two WebSocket runtimes:

- step-by-step simulation playback
- continuous digital-twin mode

It also includes the Simulation Lab, which now supports comparing a dynamic number of policies: any selection from 2 policies up to the full available policy catalog.

DreamerV4 is not exposed in this web app anymore.

## What This App Does

Main capabilities:

- run a single scenario/strategy evaluation from the browser or API
- compare multiple scenarios and strategies in batch
- create custom scenarios
- stream a live simulation over WebSocket
- run a continuous digital twin with injected events and manual actions
- validate simulator output against reference benchmarks
- run Simulation Lab experiments with 2 to all available policies
- surface locally available learned policy artifacts for lab usage

## Folder Layout

Main files:

- `app/main.py`: FastAPI application factory, combined UI app, API-only app export
- `app/schemas.py`: request and response models
- `app/simulator_service.py`: simulator integration, strategy discovery, evaluation helpers
- `app/ws_simulation.py`: step-by-step simulation WebSocket
- `app/ws_digital_twin.py`: continuous digital-twin WebSocket
- `app/studio_routes.py`: Simulation Lab routes
- `app/studio_service.py`: Simulation Lab execution logic
- `app/validation_routes.py`: validation and sensitivity endpoints
- `app/snapshot_layer.py`: data snapshot and model registry integration
- `app/static/index.html`: UI shell
- `app/static/app.js`: front-end behavior
- `app/static/styles.css`: front-end styling
- `tests/test_api.py`: API coverage

## Requirements

Recommended environment:

- Python 3.11+
- dependencies installed from `simulation_studio/pyproject.toml`
- the simulator runtime now lives under `simulation_studio/simulator`

Recommended setup from the repo root:

```bash
cd /Users/mac/Documents/pios-1
python -m venv .venv
source .venv/bin/activate
pip install -e ./simulation_studio
```

Useful environment variables:

- `PIOS_SIM_OFFLINE=1`: use the offline city-graph path instead of live OSM downloads
- `MPLCONFIGDIR=/tmp/matplotlib`: avoids Matplotlib cache warnings on locked-down machines
- `PIOS_STANDALONE_PREDICTIONS=1`: run Django ML scheduled jobs against the in-memory operational world when the full Django/Postgres stack is not available
- `PIOS_ML_CONFIG_ROOT`: override path to shared `backendMulti/ml/config` (model catalog + scheduled prediction jobs)
- `PIOS_MODELS_DIR`: local ML artifact directory (defaults to `ml-backend/models` in the monorepo)
- `PIOS_MLFLOW_HTTP_TIMEOUT_SECONDS`: HTTP timeout for MLflow artifact preflight (default `30` when using `start-local.sh`)
- `MLFLOW_TRACKING_URI`: MLflow server for registry fallbacks (default `http://localhost:8889` via `start-local.sh`)

Standalone digital twin predictions are **decision support only**. When Django is running (normal `start-local.sh`), the twin UI reads **`PredictionResult` rows and model configs from Postgres** via `digital_twin.services.capture_operational_rows` — the same pipeline as the main dashboard, not a second ML run inside Studio. In-process scheduled jobs run only when Django is unavailable. Configure bindings in `simulation_studio/config/standalone_predictions.json`.

Recommended shell setup:

```bash
export PIOS_SIM_OFFLINE=1
export MPLCONFIGDIR=/tmp/matplotlib
```

## Running The App

From the repo root:

```bash
cd /Users/mac/Documents/pios-1
uvicorn simulation_studio.app.main:app --reload --port 8010
```

Equivalent explicit Python form:

```bash
python -m uvicorn simulation_studio.app.main:app --reload --port 8010
```

Once running:

- app UI: [http://localhost:8010](http://localhost:8010)
- OpenAPI docs: [http://localhost:8010/docs](http://localhost:8010/docs)
- ReDoc: [http://localhost:8010/redoc](http://localhost:8010/redoc)
- health check: [http://localhost:8010/api/health](http://localhost:8010/api/health)

## Running The Backend Only

To start the FastAPI backend without serving the bundled Simulation Studio web app:

```bash
cd /Users/mac/Documents/pios-1
uvicorn simulation_studio.app.main:api_app --reload --port 8010
```

Or use the repo runner:

```bash
cd /Users/mac/Documents/pios-1
./run.sh
```

When launched in backend-only mode:

- `/docs`, `/redoc`, and `/api/*` remain available
- `/` and `/static/*` are not served
- `./run.sh` now starts this API-only backend on port `8010`

## UI Areas

The current UI exposes five main working areas:

- standard simulation setup and report flow
- digital twin mode
- validation tools
- custom scenario builder
- Simulation Lab

Simulation Lab is the main policy-comparison surface for the current RL work. It can compare:

- fixed strategies such as `baseline`, `mass_campaign`, `lab_investment`, `emergency_network`, `full_response`
- adaptive controllers such as `ppo_shortage_minimizer`, `ppo_continuous`, `sac_continuous`, `iql_offline`, `cql_offline`
- `dreamerv3_official` when a usable local DreamerV3 run is available

The lab no longer uses a fixed A/B comparison model. The UI now supports:

- adding policies one at a time
- selecting all available policies
- resetting back to the default lab set
- sending a variable-length `universes` array to the backend
- loading an RL-first default branch set based on the currently available policy artifacts

## Current Simulation Lab Policy Contract

`GET /api/studio/setup` now returns:

- `policy_catalog`: all policies the lab can expose
- `policy_selection.min_policies`: `2`
- `policy_selection.default_policies`: current default count
- `policy_selection.max_policies`: all available policies in the catalog

The front-end uses this to allow a dynamic number of policy cards instead of a hard-coded pair.

Each policy branch is represented by a `UniversePolicyInput` object:

```json
{
  "key": "policy_sac",
  "label": "SAC Continuous",
  "strategy_key": "sac_continuous",
  "description": "Continuous off-policy controller.",
  "parameter_multipliers": {},
  "parameter_overrides": {},
  "replay_events": []
}
```

The Simulation Lab experiment request accepts an array of these objects in `universes`.

## REST API

### Core endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | simple health probe |
| `GET` | `/api/meta` | scenarios, strategies, actions, defaults, DreamerV3 runs |
| `GET` | `/api/evaluations/cached-summary` | load cached evaluation summaries from simulator results |
| `POST` | `/api/evaluations/compare` | batch compare scenarios and strategies |
| `POST` | `/api/evaluations/run` | run one scenario/strategy pair |
| `POST` | `/api/custom-scenarios` | create and persist a custom scenario |

### Simulation endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/sim/setup` | standard simulation setup payload |
| `WS` | `/api/sim/ws` | step-by-step simulation stream |

### Digital twin endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/twin/setup` | digital twin setup payload |
| `WS` | `/api/twin/ws` | continuous digital twin stream |

### Simulation Lab endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/studio/setup` | Simulation Lab setup, policy catalog, defaults |
| `POST` | `/api/studio/snapshots` | create a data snapshot for lab usage |
| `POST` | `/api/studio/experiments/run` | execute a multi-policy experiment |
| `POST` | `/api/studio/assistant/strategy` | run the strategy assistant |

### Validation endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/validation/references` | validation reference series |
| `GET` | `/api/validation/benchmarks` | benchmark metadata |
| `POST` | `/api/validation/quick` | quick validation |
| `POST` | `/api/validation/full` | full validation |
| `POST` | `/api/validation/sensitivity` | sensitivity analysis |
| `GET` | `/api/validation/metrics-info` | validation metric descriptions |

## Common API Commands

Health check:

```bash
curl http://localhost:8010/api/health
```

Fetch app metadata:

```bash
curl http://localhost:8010/api/meta
```

Run a single evaluation:

```bash
curl -X POST http://localhost:8010/api/evaluations/run \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_key": "baseline",
    "strategy_key": "sac_continuous",
    "seed": 100,
    "hours_override": 168,
    "include_timeline": true
  }'
```

Run a batch comparison:

```bash
curl -X POST http://localhost:8010/api/evaluations/compare \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_keys": ["baseline", "donor_decrease"],
    "strategy_keys": ["baseline", "ppo_continuous", "sac_continuous", "iql_offline", "cql_offline"],
    "runs": 2,
    "seed_start": 100,
    "hours_override": 168
  }'
```

Create a custom scenario:

```bash
curl -X POST http://localhost:8010/api/custom-scenarios \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Cold Snap Donor Drop",
    "description": "Lower turnout and slower transport.",
    "base_scenario_key": "baseline",
    "recommended_strategy_key": "mass_campaign",
    "sim_hours": 96,
    "donor_show_factor": 0.75,
    "transport_penalty": 1.4,
    "forced_weather": "snow"
  }'
```

Get Simulation Lab setup data:

```bash
curl http://localhost:8010/api/studio/setup
```

Run a Simulation Lab experiment with multiple policies:

```bash
curl -X POST http://localhost:8010/api/studio/experiments/run \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_key": "baseline",
    "seed": 100,
    "replications": 3,
    "hours_override": 168,
    "step_hours": 6.0,
    "timeline_months_ahead": 3,
    "include_timeline": true,
    "snapshot": {
      "source": "synthetic",
      "include_raw_donors": true,
      "include_alert_events": true,
      "include_model_registry": true
    },
    "universes": [
      {
        "key": "policy_baseline",
        "label": "Baseline",
        "strategy_key": "baseline",
        "description": "No active intervention."
      },
      {
        "key": "policy_ppo",
        "label": "PPO Continuous",
        "strategy_key": "ppo_continuous",
        "description": "Continuous PPO controller."
      },
      {
        "key": "policy_sac",
        "label": "SAC Continuous",
        "strategy_key": "sac_continuous",
        "description": "Continuous SAC controller."
      },
      {
        "key": "policy_iql",
        "label": "IQL Offline",
        "strategy_key": "iql_offline",
        "description": "Offline RL controller."
      }
    ],
    "replay": {
      "mode": "none",
      "events": []
    },
    "agent_mode": {
      "enabled": false
    }
  }'
```

Run a quick validation:

```bash
curl -X POST http://localhost:8010/api/validation/quick \
  -H "Content-Type: application/json" \
  -d '{"scenario_key": "baseline"}'
```

## WebSocket Contracts

### Step-by-step simulation

Endpoint:

```text
ws://localhost:8010/api/sim/ws
```

Client commands:

```json
{"command": "start", "scenario_key": "baseline", "strategy_key": "baseline", "seed": 100, "hours_override": 168, "step_hours": 6, "speed": 1.0}
{"command": "pause"}
{"command": "resume"}
{"command": "stop"}
{"command": "set_speed", "speed": 2.0}
```

Server messages:

```json
{"type": "init"}
{"type": "step"}
{"type": "complete"}
{"type": "error", "message": "..."}
{"type": "paused"}
{"type": "resumed"}
{"type": "stopped"}
```

### Digital twin

Endpoint:

```text
ws://localhost:8010/api/twin/ws
```

Client commands:

```json
{"command": "start", "scenario_key": "baseline", "strategy_key": "baseline", "seed": 100, "tick_interval_s": 2.0, "step_hours": 6.0, "enable_live_data": true, "budget_cycle_hours": 168}
{"command": "pause"}
{"command": "resume"}
{"command": "stop"}
{"command": "set_speed", "speed": 2.0}
{"command": "inject_event", "event_key": "demand_surge", "severity": 0.8}
{"command": "execute_action", "action_key": "campaign", "intensity": 0.7}
```

Server messages:

```json
{"type": "twin_init"}
{"type": "twin_tick"}
{"type": "twin_event"}
{"type": "budget_refresh"}
{"type": "action_result"}
{"type": "data_status"}
{"type": "paused"}
{"type": "resumed"}
{"type": "stopped"}
{"type": "error", "message": "..."}
```

## Simulation Lab Request Shapes

`POST /api/studio/snapshots` accepts `DataSnapshotRequest`:

```json
{
  "source": "synthetic",
  "donor_limit": 250,
  "include_raw_donors": true,
  "include_alert_events": true,
  "include_model_registry": true
}
```

`POST /api/studio/assistant/strategy` accepts `StrategyAssistantRequest`:

```json
{
  "scenario_key": "baseline",
  "seed": 200,
  "replications": 3,
  "hours_override": 168,
  "timeline_months_ahead": 3,
  "dropout_rise_pct": 12.0,
  "snapshot": {
    "source": "synthetic"
  },
  "candidate_universes": [
    {
      "key": "policy_sac",
      "label": "SAC Continuous",
      "strategy_key": "sac_continuous",
      "description": "Continuous off-policy controller."
    },
    {
      "key": "policy_iql",
      "label": "IQL Offline",
      "strategy_key": "iql_offline",
      "description": "Offline controller."
    }
  ]
}
```

## Custom Scenario Support

Custom scenarios are persisted to:

- `simulation_studio/data/custom_scenarios.json`

The custom scenario schema supports:

- duration overrides
- donor arrival and show-rate changes
- demand-rate and surge adjustments
- weather forcing
- transport penalties
- lab and processing time multipliers
- reserve and replenishment tuning
- demand-forecast noise and refresh intervals
- congestion thresholds and delay factors
- episode budget overrides

## Model Discovery

The app can surface local policy artifacts through the snapshot layer. It scans model locations under:

- `ml-backend/mlruns`
- `ml-backend/notebooks/mlruns`
- `simulation_studio/simulator`

This is how locally available PPO, SAC, IQL, and CQL artifacts can appear inside the Simulation Lab workflow.

## Testing

Run the API tests from the repo root:

```bash
cd /Users/mac/Documents/pios-1
pytest -q simulation_studio/tests/test_api.py
```

Syntax-check the main app modules:

```bash
python -m py_compile \
  simulation_studio/app/main.py \
  simulation_studio/app/simulator_service.py \
  simulation_studio/app/studio_routes.py \
  simulation_studio/app/ws_simulation.py \
  simulation_studio/app/ws_digital_twin.py
```

## Current Recommended Workflow

For the current RL work:

1. generate evaluation artifacts in `simulation_studio/simulator/results/`
2. start `simulation_studio`
3. open Simulation Lab
4. compare any subset from 2 policies up to the full policy catalog
5. use the strategy assistant and snapshot layer when you want more guided comparison

## Troubleshooting

App starts but strategies are missing:

- confirm the simulator models exist in `simulation_studio/simulator`
- check `GET /api/studio/setup` to see what the backend marks as available

DreamerV3 is missing from the UI:

- the app only exposes it when a usable DreamerV3 run is discoverable

DreamerV4 does not appear:

- this is expected; DreamerV4 has been removed from the studio-facing catalog

Slow startup or graph fetch issues:

- set `PIOS_SIM_OFFLINE=1`

Matplotlib cache warnings:

- set `MPLCONFIGDIR=/tmp/matplotlib`

## Summary

`simulation_studio` is now aligned with the current simulator stack:

- DreamerV4 removed from the studio UI and API surface
- Simulation Lab supports a dynamic number of policies
- SAC, IQL, and CQL are available for comparison alongside PPO and fixed baselines
- the lab and snapshot layer can consume locally generated simulator artifacts directly

## Notes

- `PIOS_SIM_OFFLINE=1` keeps the app fast and avoids downloading the city road graph.
- The backend does not reuse the existing Django backend or the existing web frontend.
- DreamerV3 is exposed only when its checkpoint is available.
