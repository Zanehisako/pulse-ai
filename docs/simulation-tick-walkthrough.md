# What happens every 2 minutes — Simulation tick walkthrough

This document explains exactly what data changes on each scheduler tick, how ML predictions
run on top of that changed data, and how alerts reach the frontend in real time.

---

## The 3-step chain (triggered every 2 min by APScheduler)

```
advance_simulation_tick()
        ↓
run_configured_scheduled_predictions()
        ↓
manage.py seed_dashboard_from_inventory --force
        ↓
ModelResponse.post_save signal → WebSocket → dashboard
        ↓
AlertEngine.evaluate_prediction() → AlertEvent → WebSocket → alerts panel
```

---

## Step 1 — Simulation tick: what rows change and how

### 1a. BloodSupply rows (`inventory_bloodsupply`)

Every `BloodSupply` row (one per blood product type per hospital) is mutated:

| Field | What changes |
|---|---|
| `current_stock_units` | Decreases by `usage_today ± 10%` (random each tick). If the random roll < 8% probability, a restock of 50–300 units is added instead and the counter resets. Stock never goes below 0. |
| `usage_today` | Updated to the actual varied consumption used this tick. |
| `days_since_last_restock` | +1 every tick, reset to 0 on a restock event. |
| `stockout_count_90d` | +1 if stock hits 0 this tick (was > 0 before). |

**Concrete example (tick N vs tick N+1):**
```
O+  @ CHU-Québec   stock: 420 → 378  (consumed ~42 units, no restock)
A-  @ CHU-Québec   stock: 85  → 0    (consumed 90 units → STOCKOUT counted)
B+  @ Hôpital A    stock: 200 → 480  (restock fired: +285 units)
```
Because the RNG is deliberately unseeded, every tick produces different values.

---

### 1b. HospitalSupplyFeature rows (`inventory_hospitalsupplyfeature`)

One row per hospital. These are the context features the ML models use:

| Field | What changes |
|---|---|
| `temperature` | Random walk ±2 °C, clamped to −30 °C … +35 °C (realistic Québec range). |
| `rain_mm` | Gaussian noise centred at 0, max ~8 mm. Always ≥ 0. |
| `scheduled_surgeries` | Integer variance ±15% of current value, floor 0. |
| `trauma_cases` | Integer variance ±20% of current value, floor 0. |
| `current_inventory` | Synced from the real sum of `BloodSupply.current_stock_units` for that hospital after the supply tick above. |

---

### 1c. Donor rows (`inventory_donor`)

Every donor row is nudged forward in time:

| Field | What changes |
|---|---|
| `recency_days` | +1 every tick (donor gets "older" since last donation). |
| `days_until_eligible` | −1 every tick, floor 0 (eligible sooner). |
| `availability` | 2% chance an available donor becomes unavailable (churn). 3% chance an unavailable donor recovers. |

**Why this matters for predictions:** recency, frequency, and eligibility are the three
core features for the donor propensity model. As recency grows, predicted donation
propensity drops, which can eventually cross the alert threshold.

---

## Step 2 — ML predictions run on the new data

Two jobs are enabled in `ml/config/scheduled_predictions.json`:

### Job 1: `dashboard_stockout_days`
- **Model:** `stockout_days_predictor` (champion version)
- **Input features:** `current_stock_units`, `usage_today`, `days_since_last_restock`,
  `scheduled_surgeries`, `trauma_cases`, `temperature`, `rain_mm`, etc.
- **Output:** predicted number of days until stockout (float)
- **Alert fires when:** prediction ≤ **3.0 days**
- **Written to:** `PredictionResult` table, one row per (supply_id, date)

### Job 2: `dashboard_hospital_shortage`
- **Model:** `hospital_shortage_predictor` (champion version)
- **Input features:** hospital-level supply feature row
- **Output:** shortage probability 0.0 → 1.0
- **Alert fires when:** prediction ≥ **0.5** (50% shortage probability)
- **Written to:** `PredictionResult` table, one row per (hospital_id, date)

### Job 3: `dashboard_donor_propensity` — **disabled**
Disabled until the donor feature schema is fully populated in the operational DB.
Enable it in `scheduled_predictions.json` when ready.

---

## Step 3 — Dashboard sync and WebSocket push

After predictions are saved, `seed_dashboard_from_inventory --force` runs.
It rebuilds `ModelResponse` rows from the latest `PredictionResult` data.

`ModelResponse.post_save` signal fires → pushes to channel group `dashboard_updates`
→ every connected frontend receives a WebSocket message → `use-dashboard-data.ts`
updates the dashboard charts without a page reload.

---

## Step 4 — Alert evaluation and real-time alert panel

During prediction saving, `AlertEngine.evaluate_prediction()` is called for each result.

Flow:
```
prediction saved
    ↓
AlertEngine checks all active AlertRules for this model_id
    ↓
If operator condition met (e.g. stockout_days ≤ 3):
    create_or_refresh_event() creates an AlertEvent row
    ↓
notify_clients(event) calls broadcast_alert(event)
    ↓
channel_layer.group_send("alerts", {"type": "send_alert", ...})
    ↓
AlertConsumer.send_alert() pushes JSON to ws/alerts/
    ↓
useAlerts.ts receives the message → alert panel updates live
```

**Dedup:** if the same alert condition fires again within the dedup window (default: 60 min),
the existing `AlertEvent` is refreshed rather than a new one created. The frontend still
gets a WebSocket push.

---

## What you should see in the web-app after each tick

| Where | What you see |
|---|---|
| Dashboard charts | Stock levels trending down/up, hospital risk scores changing |
| Alert panel (bell icon) | New `OPEN` alerts appearing when stockout < 3 days or shortage > 50% |
| Alert severity | `warning` for borderline predictions, `critical` after escalation (after configured `escalation_after_minutes`) |
| Network tab → WS frames | Messages on `ws/dashboard/` and `ws/alerts/` arriving every 2 min |

---

## How to speed up / slow down the cycle

Edit `backendMulti/ml/config/simulation_config.json`:

```json
{
  "enabled": true,
  "tick_interval_minutes": 2,   ← change this
  ...
}
```

The scheduler reads this at startup. To change it on a running server, restart the container
(or the `daphne` process). No code change needed.

To **force an immediate tick** without waiting:
```bash
curl -X POST http://localhost:8000/api/predictions/trigger/ \
  -H "Authorization: Token <staff-token>"
```

---

## Config reference (`ml/config/simulation_config.json`)

```
tick_interval_minutes           How often the full chain runs (minutes)

blood_supply:
  consumption_variance_pct      ±% randomness on each supply's usage_today  (0.10 = ±10%)
  restock_probability           Probability of a restock event per supply per tick  (0.08 = 8%)
  restock_amount_min/max        Unit range for a restock delivery
  min_stock_floor               Minimum allowed stock value (0 = allow stockouts)

hospital_features:
  temperature_drift_max         Max °C change per tick (random walk)
  rain_mm_max                   Gaussian scale for rainfall (mm)
  scheduled_surgeries_variance_pct   ±% change on surgery count
  trauma_cases_variance_pct         ±% change on trauma count

donors:
  recency_increment             Days added to recency_days per tick (1 = 1 simulated day)
  availability_churn_rate       Probability an available donor becomes unavailable
  availability_recovery_rate    Probability an unavailable donor becomes available again
```
