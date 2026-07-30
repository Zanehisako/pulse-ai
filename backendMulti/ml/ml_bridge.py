"""
ml/ml_bridge.py

Feature engineering bridge: converts a BloodSupply snapshot (+ optional
HospitalSupplyFeature) into the 36-column DataFrame the stockout_days_predictor
LSTM model expects.

All derived feature formulas are aligned with train_stockout_lstm.py.
Rolling-window features are approximated from the current snapshot since the
operational DB does not store daily history — documented as future work.

Fixes applied vs previous version:
  FIX 1 — demand_pressure:       wrong formula (units_used/stock) → correct (trauma+surgeries)/inventory
  FIX 2 — medium_demand_pressure: was demand_pressure*0.5 → correct rolling-based formula
  FIX 3 — weather_load:          extra temperature term removed → rain_mm only
  FIX 4 — shock_intensity:       trauma indicator replaced with holiday flag
  FIX 5 — inventory_pressure:    wrong formula → 1/(1+current_inventory)
  FIX 6 — inventory_band:        wrong bins (50/150/300) and labels → correct (10/20/35/60/200)
  FIX 7 — shortage_7d_mean:      overwrite bug removed (was always 0)
  FIX 8 — shortage_14d_mean:     overwrite bug removed (was always 0)
  FIX 9 — inventory_trend_7d:    overwrite bug removed (was always 0)
"""
from __future__ import annotations

import math
from datetime import date

import pandas as pd


def build_stockout_features(
    supply,   # BloodSupply instance
    hsf=None, # HospitalSupplyFeature instance (optional, latest for this hospital)
) -> pd.DataFrame:
    """
    Build the 36-feature DataFrame expected by stockout_days_predictor@champion.

    Parameters
    ----------
    supply : BloodSupply
        Current inventory snapshot for one supply row.
    hsf : HospitalSupplyFeature | None
        Latest environmental/clinical row for the same hospital.
        If None, safe defaults are used.

    Returns
    -------
    pd.DataFrame with exactly one row and the 36 required columns.
    """
    today = date.today()

    # ── Date features ──────────────────────────────────────────────────────
    dow        = today.weekday()          # 0=Mon … 6=Sun
    weekend    = int(dow >= 5)
    month      = today.month
    season_sin = math.sin(2 * math.pi * month / 12)
    season_cos = math.cos(2 * math.pi * month / 12)

    # ── Environmental / clinical (from HospitalSupplyFeature if present) ───
    temperature         = float(hsf.temperature)       if hsf else 20.0
    rain_mm_val         = float(hsf.rain_mm)           if hsf else 0.0
    holiday             = int(hsf.holiday)             if hsf else 0
    disaster            = int(hsf.disaster)            if hsf else 0
    scheduled_surgeries = int(hsf.scheduled_surgeries) if hsf else int(supply.scheduled_surgeries_next7d / 7)
    trauma_cases        = int(hsf.trauma_cases)        if hsf else 0
    current_inventory   = float(hsf.current_inventory) if hsf else float(supply.current_stock_units)

    # ── Supply snapshot fields (direct from BloodSupply) ───────────────────
    stock_start       = float(supply.current_stock_units)
    units_used        = float(supply.usage_today)
    units_collected   = 0.0   # not stored — external pipeline would provide
    wastage           = 0.0   # not stored
    flu_index         = 0.0   # not tracked in operational DB
    donation_campaign = 0.0   # not tracked in operational DB

    # ── Derived flags ──────────────────────────────────────────────────────
    # critical_stock: from synthetic_bloodbank_daily.csv this was stock_end <= 0
    critical_stock = float(current_inventory <= 0)

    # ── FIX 1: demand_pressure ────────────────────────────────────────────
    # train_stockout_lstm.py:
    #   (trauma_cases + 0.6 * scheduled_surgeries) / (1 + current_inventory)
    demand_pressure = (
        float(trauma_cases) + 0.6 * float(scheduled_surgeries)
    ) / (1.0 + current_inventory)

    heavy_demand = float(demand_pressure > 0.8)

    # ── Rolling window approximations ──────────────────────────────────────
    # Without daily history these are approximated from today's snapshot.
    # A real pipeline would compute these from a BloodInventoryHistory table.
    inv_7d_mean = current_inventory + (units_used * 3.5)   # midpoint of 7-day window
    inv_3d_mean = current_inventory + (units_used * 1.5)   # midpoint of 3-day window
    inv_7d_std  = max(units_used * 0.3, 1.0)

    trauma_3d = float(trauma_cases)
    trauma_7d = float(trauma_cases)
    surg_daily = float(scheduled_surgeries) / 7.0
    surg_3d    = surg_daily
    surg_7d    = surg_daily

    # ── FIX 7+8: shortage rolling means — overwrite bug removed ──────────
    # train_stockout_lstm.py: rolling mean of blood_shortage flag over 7/14 days
    # Approximation: derive from current runway — do NOT overwrite after setting
    days_left         = current_inventory / max(units_used, 0.1)
    shortage_7d_mean  = 1.0 if days_left < 7  else 0.0
    shortage_14d_mean = 1.0 if days_left < 14 else 0.0

    # ── FIX 2: medium_demand_pressure ─────────────────────────────────────
    # train_stockout_lstm.py:
    #   (trauma_7d_mean + 0.6 * surgeries_7d_mean) / (1 + inventory_7d_mean)
    medium_demand_pressure = (
        trauma_7d + 0.6 * surg_7d
    ) / (1.0 + inv_7d_mean)

    # ── FIX 3: weather_load — rain_mm only, no temperature term ──────────
    # train_stockout_lstm.py: _safe_numeric(frame["rain_mm"])
    weather_load = rain_mm_val

    # ── FIX 4: shock_intensity — holiday flag, not trauma indicator ───────
    # train_stockout_lstm.py: disaster * 2.0 + holiday
    shock_intensity = float(disaster) * 2.0 + float(holiday)

    # ── FIX 5: inventory_pressure ─────────────────────────────────────────
    # train_stockout_lstm.py: 1.0 / (1.0 + current_inventory)
    inventory_pressure = 1.0 / (1.0 + current_inventory)

    # ── FIX 9: inventory_trend_7d — overwrite bug removed ────────────────
    # train_stockout_lstm.py: diff(7) of current_inventory per group
    # Best approximation without history: negative daily usage × 7
    inventory_trend_7d = -units_used * 7.0

    # ── FIX 6: inventory_band — correct bins and labels ──────────────────
    # train_stockout_lstm.py:
    #   bins=[-1, 10, 20, 35, 60, 200]
    #   labels=["critical", "low", "guarded", "healthy", "buffered"]
    if current_inventory <= 10:
        inventory_band = "critical"
    elif current_inventory <= 20:
        inventory_band = "low"
    elif current_inventory <= 35:
        inventory_band = "guarded"
    elif current_inventory <= 60:
        inventory_band = "healthy"
    else:
        inventory_band = "buffered"

    # ── Assemble exactly the 36 columns the model expects ─────────────────
    row = {
        "dow":                    dow,
        "weekend":                weekend,
        "month":                  month,
        "holiday":                holiday,
        "temperature":            temperature,
        "rain_mm":                rain_mm_val,
        "flu_index":              flu_index,
        "trauma_cases":           trauma_cases,
        "scheduled_surgeries":    scheduled_surgeries,
        "donation_campaign":      donation_campaign,
        "disaster":               disaster,
        "stock_start":            stock_start,
        "units_collected":        units_collected,
        "units_used":             units_used,
        "wastage":                wastage,
        "current_inventory":      current_inventory,
        "critical_stock":         critical_stock,
        "heavy_demand":           heavy_demand,
        "inventory_3d_mean":      inv_3d_mean,
        "inventory_7d_mean":      inv_7d_mean,
        "inventory_7d_std":       inv_7d_std,
        "trauma_3d_mean":         trauma_3d,
        "trauma_7d_mean":         trauma_7d,
        "surgeries_3d_mean":      surg_3d,
        "surgeries_7d_mean":      surg_7d,
        "shortage_7d_mean":       shortage_7d_mean,
        "shortage_14d_mean":      shortage_14d_mean,
        "season_sin":             season_sin,
        "season_cos":             season_cos,
        "demand_pressure":        demand_pressure,
        "medium_demand_pressure": medium_demand_pressure,
        "weather_load":           weather_load,
        "shock_intensity":        shock_intensity,
        "inventory_pressure":     inventory_pressure,
        "inventory_trend_7d":     inventory_trend_7d,
        "inventory_band":         inventory_band,
    }

    return pd.DataFrame([row])


def get_latest_hsf(hospital_id: str):
    """
    Return the most recent HospitalSupplyFeature row for a hospital, or None.
    Imported lazily to avoid circular imports.
    """
    from inventory.models import HospitalSupplyFeature
    return (
        HospitalSupplyFeature.objects
        .filter(hospital_id=hospital_id)
        .order_by("-event_timestamp")
        .first()
    )