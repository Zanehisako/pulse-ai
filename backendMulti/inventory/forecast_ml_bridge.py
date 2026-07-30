"""
inventory/forecast_ml_bridge.py

Feature engineering bridge: converts a BloodSupply snapshot (+ optional
HospitalSupplyFeature) into the 20-column DataFrame the blood_stock_forecast
XGBoost model expects.

Column alignment is with FORECAST_FEATURE_COLS defined in
datasets_featues_labels_seperated/generate_scripts/real_forcast.py:

    hospital_id,
    blood_product_type, current_stock_units, usage_today,
    usage_lag_1, usage_lag_3,
    lead_time_days, days_since_last_restock, stockout_count_90d,
    scheduled_surgeries_next7d,
    historical_demand_7d, historical_demand_30d,
    incoming_supply_scheduled_7d, incoming_supply_scheduled_30d,
    incoming_supply_day1,
    expiry_rate_7d, demand_spike_indicator, is_holiday, season,
    stock_trend_7d

Rolling-window features (historical_demand_7d/30d, usage_lag_1/3) are
computed from BloodSupplySnapshot rows which the simulation tick writes
every day.  If fewer rows are available the average of whatever is stored
is used; if none exist usage_today is the fallback.

Approximations
--------------
historical_demand_7d / historical_demand_30d
    Mean of the last 7 / 30 BloodSupplySnapshot.usage_today values for the
    same hospital + blood_product_type combination.  Falls back to
    usage_today when snapshot history is sparse.

usage_lag_1
    Most recent BloodSupplySnapshot.usage_today — yesterday's consumption.
    Falls back to usage_today when no snapshot exists.

usage_lag_3
    Mean of the three most recent BloodSupplySnapshot.usage_today values.
    Falls back to usage_today when snapshot history is sparse.

incoming_supply_scheduled_7d / incoming_supply_scheduled_30d / incoming_supply_day1
    Not stored in the operational DB.  Conservative default 0.0 avoids
    over-optimistic stock projections.  Future work: derive from a
    purchase-order model or lead_time_days heuristic.

expiry_rate_7d
    Approximated as current_stock_units × 0.20 — the mean of the
    Beta(1.5, 6) distribution used in real_forcast.py to generate this
    feature.

demand_spike_indicator
    Requires a rolling usage baseline; defaulted to 0.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Rolling demand helper
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_demand(supply, days: int) -> float:
    """
    Mean daily usage over the last *days* BloodSupplySnapshot rows for
    this hospital + blood_product_type pair.

    Falls back to supply.usage_today when snapshot history is too sparse
    (fewer than 2 rows) so that the feature is never a stale default 0.
    """
    from django.db.models import Avg, Count
    from inventory.models import BloodSupplySnapshot

    # Collect the IDs of the most recent `days` snapshots, then aggregate
    recent_ids = list(
        BloodSupplySnapshot.objects
        .filter(
            hospital=supply.hospital,
            blood_product_type=supply.blood_product_type,
        )
        .order_by("-recorded_at")
        .values_list("id", flat=True)[:days]
    )
    if len(recent_ids) >= 2:
        result = (
            BloodSupplySnapshot.objects
            .filter(id__in=recent_ids)
            .aggregate(avg=Avg("usage_today"))
        )
        return round(result["avg"], 2)
    return round(float(supply.usage_today), 2)


def _last_usage(supply) -> float:
    """
    Most recent BloodSupplySnapshot.usage_today for the lag-1 feature.
    Falls back to supply.usage_today when no snapshot history exists.
    """
    from inventory.models import BloodSupplySnapshot

    snap = (
        BloodSupplySnapshot.objects
        .filter(
            hospital=supply.hospital,
            blood_product_type=supply.blood_product_type,
        )
        .order_by("-recorded_at")
        .values_list("usage_today", flat=True)
        .first()
    )
    if snap is not None:
        return round(float(snap), 2)
    return round(float(supply.usage_today), 2)


def _stock_trend(supply, days: int = 7) -> float:
    """
    Rolling stock slope over the last *days* BloodSupplySnapshot rows:
      (stock_now − stock_{days}_ago) / days

    Negative → sustained depletion trend (strongest signal for t7_hard magnitude).
    Falls back to 0.0 when snapshot history is too short.
    """
    from inventory.models import BloodSupplySnapshot

    recent = list(
        BloodSupplySnapshot.objects
        .filter(
            hospital=supply.hospital,
            blood_product_type=supply.blood_product_type,
        )
        .order_by("-recorded_at")
        .values_list("current_stock_units", flat=True)[: days + 1]
    )
    if len(recent) > days:
        # recent[0] = latest (stock_now proxy), recent[-1] = oldest in window
        return round((float(recent[0]) - float(recent[-1])) / days, 3)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Calendar helpers (mirrored from real_forcast.py)
# ─────────────────────────────────────────────────────────────────────────────

def _season(d: date) -> int:
    """Return season: 1=Winter, 2=Spring, 3=Summer, 4=Autumn."""
    m = d.month
    if m in (12, 1, 2):
        return 1
    if m in (3, 4, 5):
        return 2
    if m in (6, 7, 8):
        return 3
    return 4


def _canadian_holidays(year: int) -> set:
    """Return a set of Canadian/Quebec public holiday *date* objects for *year*."""
    try:
        from dateutil.easter import easter as _easter
        easter_date = _easter(year)  # returns datetime.date
    except Exception:
        return set()

    good_friday   = easter_date - timedelta(days=2)
    easter_monday = easter_date + timedelta(days=1)

    # Victoria Day — last Monday before May 25
    may_25 = date(year, 5, 25)
    victoria_day = may_25 - timedelta(days=(may_25.weekday() + 1) % 7 or 7)

    # Labour Day — first Monday of September
    sep_1 = date(year, 9, 1)
    labour_day = sep_1 + timedelta(days=(7 - sep_1.weekday()) % 7)

    # Thanksgiving — second Monday of October
    oct_1 = date(year, 10, 1)
    first_monday_oct = oct_1 + timedelta(days=(7 - oct_1.weekday()) % 7)
    thanksgiving = first_monday_oct + timedelta(weeks=1)

    return {
        good_friday,
        easter_monday,
        victoria_day,
        labour_day,
        thanksgiving,
        date(year, 1, 1),    # New Year's Day
        date(year, 6, 24),   # Saint-Jean-Baptiste (Quebec)
        date(year, 7, 1),    # Canada Day
        date(year, 11, 11),  # Remembrance Day
        date(year, 12, 25),  # Christmas Day
        date(year, 12, 26),  # Boxing Day
    }


_holiday_cache: dict[int, set] = {}


def _is_holiday(d: date) -> int:
    """Return 1 if *d* is a Canadian/Quebec public holiday, else 0."""
    if d.year not in _holiday_cache:
        _holiday_cache[d.year] = _canadian_holidays(d.year)
    return int(d in _holiday_cache[d.year])


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_forecast_features(
    supply,   # BloodSupply instance
    hsf=None, # HospitalSupplyFeature instance (optional, latest for this hospital)
) -> pd.DataFrame:
    """
    Build the 20-feature DataFrame expected by blood_stock_forecast@champion.

    Parameters
    ----------
    supply : BloodSupply
        Current inventory snapshot for one supply row.
    hsf : HospitalSupplyFeature | None
        Latest environmental/clinical row for the same hospital.
        Accepted for API consistency with ml/ml_bridge.py; not currently
        used as the forecast model does not include HSF columns in its
        feature set.

    Returns
    -------
    pd.DataFrame with exactly one row and the 20 required columns.
    """
    today = date.today()

    # ── P9: hospital entity feature ───────────────────────────────────────
    hospital_id = str(supply.hospital_id)

    # ── Direct supply fields ──────────────────────────────────────────────
    blood_product_type         = str(supply.blood_product_type)
    current_stock_units        = float(supply.current_stock_units)
    usage_today_val            = float(supply.usage_today)
    lead_time_days             = int(supply.lead_time_days)
    # Cap at 30 to prevent the model from predicting phantom restocks for
    # rare blood types that haven't been restocked in a long while.
    days_since_last_restock    = min(int(supply.days_since_last_restock), 30)
    stockout_count_90d         = int(supply.stockout_count_90d)
    scheduled_surgeries_next7d = int(supply.scheduled_surgeries_next7d)

    # ── Rolling demand from snapshot history ────────────────────────────────
    historical_demand_7d  = _rolling_demand(supply, 7)
    historical_demand_30d = _rolling_demand(supply, 30)

    # ── P3: lag features ──────────────────────────────────────────────────
    usage_lag_1 = _last_usage(supply)         # yesterday's consumption
    usage_lag_3 = _rolling_demand(supply, 3)  # 3-day rolling mean

    # ── P7: stock trend (rolling slope for depletion magnitude) ──────────
    stock_trend_7d = _stock_trend(supply, days=7)

    # ── Scheduled supply approximations ───────────────────────────────────
    incoming_supply_scheduled_7d  = 0.0
    incoming_supply_scheduled_30d = 0.0
    # P6: next-day scheduled supply — purchase-order data not in operational
    # DB so conservative default 0.0 (same as 7d/30d approximation above).
    incoming_supply_day1 = 0.0

    # ── Expiry rate: Beta(1.5, 6) mean ≈ 0.20 of current stock ───────────
    expiry_rate_7d = round(current_stock_units * 0.20, 1)

    # ── Demand spike: requires rolling baseline — defaulted to 0 ─────────
    demand_spike_indicator = 0

    # ── Calendar features ─────────────────────────────────────────────────
    is_holiday_val = _is_holiday(today)
    season         = _season(today)

    row = {
        "hospital_id":                     hospital_id,
        "blood_product_type":              blood_product_type,
        "current_stock_units":             current_stock_units,
        "usage_today":                     usage_today_val,
        "usage_lag_1":                     usage_lag_1,
        "usage_lag_3":                     usage_lag_3,
        "lead_time_days":                  lead_time_days,
        "days_since_last_restock":         days_since_last_restock,
        "stockout_count_90d":              stockout_count_90d,
        "scheduled_surgeries_next7d":      scheduled_surgeries_next7d,
        "historical_demand_7d":            historical_demand_7d,
        "historical_demand_30d":           historical_demand_30d,
        "incoming_supply_scheduled_7d":    incoming_supply_scheduled_7d,
        "incoming_supply_scheduled_30d":   incoming_supply_scheduled_30d,
        "incoming_supply_day1":            incoming_supply_day1,
        "expiry_rate_7d":                  expiry_rate_7d,
        "demand_spike_indicator":          demand_spike_indicator,
        "is_holiday":                      is_holiday_val,
        "season":                          season,
        "stock_trend_7d":                  stock_trend_7d,
    }
    prefixed_row = {
        f"forecast_features_features__{key}": value
        for key, value in row.items()
    }

    return pd.DataFrame([prefixed_row])


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
