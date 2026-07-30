# Alerts system deep scan (backendMulti + linked services + web-app)

## Scope scanned

- `backendMulti/alerts/*` (models, API, services, tasks, signals, websocket, tests, commands)
- Linked backend paths: `backendMulti/backendMulti/{settings,asgi,urls}.py`, `backendMulti/ml/api/views.py`, `backendMulti/inventory/{models,tasks,views}.py`, `backendMulti/notifications/*`, `backendMulti/backendMulti/middleware/ws_auth.py`
- Web app paths: `web-app/pios-web/src/features/alerts/*`, websocket client, app provider/layout/routes, notifications hook, API URL helpers
- Infra/runtime: `infrastructure/docker/docker-compose.local.yaml`

---

## 1. End-to-end architecture (what happens when an alert is produced)

1. ML prediction endpoint runs (`/api/ml/predict/nl/` or `/api/ml/models/{id}/predict/`).
2. Backend calls `AlertEngine.evaluate_prediction(...)`.
3. Engine builds context from feature input + prediction output, filters active rules by scope, and evaluates conditions.
4. If matched, engine creates or refreshes `AlertEvent` (dedup by `event_key` + dedup window).
5. `notify_clients(event)` runs:
   - broadcasts to WebSocket group `alerts` (`/ws/alerts/`)
   - creates a Notification row and broadcasts to WebSocket group `notifications` (`/ws/notifications/`)
6. Event lifecycle updates (acknowledge / resolve / stale escalation) also call `notify_clients`.

Persistence is in PostgreSQL (`alert_rule`, `alert_event`, `alert_event_history`).

---

## 2. Alerts domain model (backendMulti/alerts/models.py)

### AlertRule

Defines trigger logic and behavior:

- Scope: `global | center | hospital | blood_type`
- Trigger type: `threshold_gap | stock_and_forecast | stale_alert | manual`
- Severity base: `warning | critical | resolved`
- Conditions JSON (`field/op/value`, or nested `all`/`any`)
- Delivery channels metadata, dedup window, escalation window

### AlertEvent

Represents an alert occurrence/state:

- Status: `open | acknowledged | escalated | resolved | muted`
- Severity: `warning | critical | resolved`
- Context payload + dimensions (`entity_type`, `entity_id`, `blood_type`)
- Time markers (`opened_at`, `last_evaluated_at`, `acknowledged_at`, `escalated_at`, `resolved_at`)
- `notification_id` link to mirrored notification

### AlertEventHistory

Audit trail for every transition: action, actor, note, full snapshot.

---

## 3. Rule evaluation and event creation internals

## Context construction (rule_engine.py)

- Flattens nested prediction output (including `execution_results[0].output`)
- Normalizes keys (`features__x` and dotted paths to simpler keys)
- Derives canonical fields like `entity_id`, `blood_type`, `predicted_value`

## Condition evaluation

Supported operators:

- `==`, `in`, `<`, `<=`, `>`, `>=`

Supports:

- single condition
- `all` (AND)
- `any` (OR)

Dynamic thresholds are supported via `threshold_mode`:

- `zscore`
- `percentile`
- `mad`

Thresholds are computed from `inventory.PredictionResult` history (`ThresholdEngine`).

## Dedup behavior (event_service.py)

`create_or_refresh_event` does:

- lock candidate event rows (`select_for_update`)
- reuse latest non-resolved event with same `event_key` inside dedup window
- otherwise create new event

On both create and refresh:

- writes history (`created` / `refreshed`)
- triggers client delivery

---

## 4. Lifecycle transitions

Implemented transitions (event_service.py):

- `acknowledge`: allowed from `open`, `escalated`
- `resolve`: allowed from `open`, `acknowledged`, `escalated`
- `escalate`: allowed from `open`, `acknowledged`

Each transition updates state fields, writes history, and broadcasts.

---

## 5. Alert rule generation and stale processing

## Auto rule generation

- Signal: creating a new `inventory.Hospital` triggers `AlertRuleFactory.generate_for_hospital(...)`.
- Factory creates 2 rules per blood type (`8 blood types => 16 rules/hospital`):
  - critical stock rule
  - trend warning rule
- Rules use dynamic thresholds (`zscore`, `percentile`).

## Stale escalation

- `AlertEngine.process_stale_alerts()` processes rules of type `stale_alert`.
- Finds open/acknowledged events older than rule escalation window, then escalates them.
- Exposed as:
  - management command: `python manage.py process_alerts`
  - direct callable: `alerts.tasks.run_process_stale_alerts`

---

## 6. Scheduling and local prediction flow

Scheduled jobs are intentionally brokerless in the current local setup. There is no Redis, Celery worker, or Celery beat in `docker-compose.local.yaml`.

For local demos and supervisor testing, predictions are generated when the local runner starts/seeds the platform:

- `start-local.sh` seeds inventory/donor data.
- `start-local.sh` calls `python manage.py run_prediction`.
- `run_prediction` writes `inventory.PredictionResult` rows.
- `run_prediction` calls `seed_dashboard_from_inventory` so dashboard snapshots include stockout predictions.

Manual refresh commands:

- `python manage.py run_prediction`
- `python manage.py process_alerts`

`run_prediction` now loads scheduled prediction jobs from `backendMulti/ml/config/scheduled_predictions.json`. The current configured jobs are:

- `dashboard_stockout_days`
- `dashboard_hospital_shortage`
- `dashboard_donor_propensity` (disabled until the operational donor DB has the trained feature schema)

Future real-time or periodic processing can be owned by another scheduler team. That scheduler should call the same Django management commands or the same configured executor rather than adding hardcoded model-specific loops.

## Runtime detail

WebSocket channel layer is **not Redis-backed**; it is currently:

- `channels.layers.InMemoryChannelLayer`

This keeps local Docker lighter, but WebSocket group delivery is process-local. If the backend is scaled to multiple processes/containers later, replace this with an explicitly configured shared channel layer.

---

## 7. WebSocket wiring and auth

ASGI router combines websocket routes from:

- alerts: `/ws/alerts/`
- notifications: `/ws/notifications/`
- dashboard: `/ws/dashboard/`

Alerts consumer (`alerts/consumers.py`):

- requires authenticated user (`scope["user"].is_authenticated`)
- joins `alerts` group
- sends `type: "alert_event"` payloads

Token middleware (`ws_auth.py`) can resolve `?token=...` to user.

Notifications consumer currently accepts connections without auth checks and broadcasts from `notifications` group.

---

## 8. Where alerts are triggered today

### Triggered

- `ml/api/views.py`:
  - `PredictNLView` (after successful orchestration prediction)
  - `PredictModelView` (after direct model prediction)

### Triggered by scheduled prediction tasks

`inventory/tasks.py` writes `PredictionResult` records through the configured scheduled prediction executor and calls `AlertEngine.evaluate_prediction` for each saved prediction.

In local demo mode, this happens when `python manage.py run_prediction` is called by `start-local.sh` or manually.

---

## 9. REST API surface (alerts app)

Base: `/alerts/`

- `GET /alerts/` (filters: status, severity, entity_type, entity_id, blood_type)
- `GET /alerts/summary/`
- `GET /alerts/<uuid>/`
- `POST /alerts/<uuid>/acknowledge/`
- `POST /alerts/<uuid>/resolve/`
- `POST /alerts/test-fire/`
- Rules CRUD under `/alerts/rules/`

---

## 10. Web app implementation (pios-web)

## Data source + realtime hook

`useAlerts.ts`:

- initial fetch: `GET /alerts/`
- realtime socket: `wsApi("/ws/alerts/")`
- maps backend payloads to UI `Alert` type
- prepends new alerts if id is not already present

## Global alerts banner

`GlobalAlertsContext` + `AlertsBanner`:

- app-wide alerts state
- banner shown in dashboard layout (`DashboardLayout` includes `AlertsBanner`)
- filters to critical/warning for top sticky notifications

## Alerts page

`/dashboard/alerts` route:

- chart built from alerts by day/severity
- list with filters (severity/center/type)
- resolve action: `POST /alerts/{id}/resolve/`
- escalate action in UI: `POST /alerts/{id}/escalate/`

---

## 11. Current integration mismatches / caveats found

1. **Frontend calls escalate endpoint that backend does not expose**
   - UI calls `POST /alerts/{id}/escalate/`
   - backend URLs include acknowledge + resolve, but no escalate API view/route.

2. **Alerts WebSocket auth may fail from web app depending on auth context**
   - alerts consumer requires authenticated user.
   - frontend `useAlerts` opens `/ws/alerts/` without query token.
   - if no valid session auth is present, connection is closed (`4003`).

3. **Channels layer is in-memory**
   - multi-process/server-worker websocket fanout is not shared.
   - broadcasts from other processes may not reach clients connected to the web server process unless a shared channel layer is configured.

4. **Notifications payload shape differs between backend and frontend expectations**
   - backend `Notification.to_dict()` uses keys `timestamp` and `data`.
   - frontend notification hook expects `created_at` and `extra_data`.
   - this can lead to missing mapped fields in UI.

---

## 12. Practical “how alerts work” summary

- Rules define what “bad state” means.
- ML prediction endpoints are the main realtime trigger point.
- Matching rules create/refresh persistent alert events.
- Every event update is pushed to websocket clients and mirrored into notifications.
- Local scheduled predictions are brokerless and run through `python manage.py run_prediction`.
- Stale open alerts are escalated when `python manage.py process_alerts` runs.
- Websocket pub/sub is currently local-process memory.

