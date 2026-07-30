"""
Generates synthetic blood-inventory time-series data for TWO separate models.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL 1 — Days-Until-Stockout  (train_stockout_lstm.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Features  →  stockout_features.{parquet,csv}
    supply_id, blood_product_type, current_stock_units, usage_today,
    lead_time_days, days_since_last_restock, stockout_count_90d,
    scheduled_surgeries_next7d, event_timestamp

  Labels    →  stockout_labels.{parquet,csv}
    supply_id, event_timestamp, days_until_stockout

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL 2 — Multi-Horizon Stock Forecasting  (train_forecast_*.py  — to create)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Features  →  forecast_features.{parquet,csv}
    supply_id, blood_product_type, current_stock_units, usage_today,
    lead_time_days, days_since_last_restock, stockout_count_90d,
    scheduled_surgeries_next7d,
    historical_demand_7d, historical_demand_30d,
    incoming_supply_scheduled_7d, incoming_supply_scheduled_30d,
    expiry_rate_7d, demand_spike_indicator, is_holiday, season,
    event_timestamp

  Labels    →  forecast_labels.{parquet,csv}
    supply_id, event_timestamp,
    stock_units_t1   (projected stock in  1 day)
    stock_units_t7   (projected stock in  7 days)
    stock_units_t30  (projected stock in 30 days)

Outputs (all written to <ROOT>/datasets_featues_labels_seperated/)
──────────────────────────────────────────────────────────────────
  stockout_features.parquet / .csv
  stockout_labels.parquet   / .csv
  forecast_features.parquet / .csv
  forecast_labels.parquet   / .csv
"""

import json
import os
import sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from dateutil.easter import easter

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Import seed profiles from the Django app (no Django setup needed — no models)
# ─────────────────────────────────────────────────────────────────────────────
_BACKENDMULTI = Path(__file__).resolve().parent.parent.parent.parent / "backendMulti"
if str(_BACKENDMULTI) not in sys.path:
    sys.path.insert(0, str(_BACKENDMULTI))

from inventory.seed_data import (  # noqa: E402
    BLOOD_TYPE_PROFILES,
    HOSPITAL_SIZE_PROFILES,
    QUEBEC_HOSPITALS,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — loaded from blood_stock_forecast_model.json for reproducibility (P8)
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "blood_stock_forecast_model.json"
)


def _load_generation_config() -> dict:
    try:
        return json.loads(_MODEL_CONFIG_PATH.read_text(encoding="utf-8")).get(
            "data_generation", {}
        )
    except (OSError, json.JSONDecodeError):
        return {}


_GEN_CFG = _load_generation_config()

N_SUPPLY_IDS = int(_GEN_CFG.get("n_supply_ids", 20))
N_DAYS       = int(_GEN_CFG.get("n_days", 1000))   # ~2.75 years
_RNG_SEED    = int(_GEN_CFG.get("rng_seed", 42))
N_PATHS_MC   = int(_GEN_CFG.get("n_paths_mc", 30))  # MC paths per label row (P4)

START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# Relative demand weight per blood type (O+/A+ are highest demand)
DEMAND_WEIGHTS = {
    "O+":  1.40,
    "O-":  0.60,
    "A+":  1.30,
    "A-":  0.55,
    "B+":  0.90,
    "B-":  0.40,
    "AB+": 0.50,
    "AB-": 0.25,
}


RNG = np.random.default_rng(seed=_RNG_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_season(date: datetime) -> int:
    """Return season number: 1=Winter, 2=Spring, 3=Summer, 4=Autumn."""
    m = date.month
    if m in (12, 1, 2):
        return 1
    if m in (3, 4, 5):
        return 2
    if m in (6, 7, 8):
        return 3
    return 4

def get_canadian_holidays(year: int) -> set:
    easter_date = easter(year)
    good_friday = easter_date - timedelta(days=2)
    easter_monday = easter_date + timedelta(days=1)

    # Victoria Day = last Monday before May 25
    may_25 = datetime(year, 5, 25)
    victoria_day = may_25 - timedelta(days=(may_25.weekday() + 1) % 7 or 7)

    # Labour Day = first Monday of September
    sep_1 = datetime(year, 9, 1)
    labour_day = sep_1 + timedelta(days=(7 - sep_1.weekday()) % 7)

    # Thanksgiving = second Monday of October
    oct_1 = datetime(year, 10, 1)
    first_monday = oct_1 + timedelta(days=(7 - oct_1.weekday()) % 7)
    thanksgiving = first_monday + timedelta(weeks=1)

    floating = {good_friday, easter_monday, victoria_day, labour_day, thanksgiving}
    fixed = {
        datetime(year, 1, 1),   # New Year's
        datetime(year, 6, 24),  # Saint-Jean-Baptiste (Quebec)
        datetime(year, 7, 1),   # Canada Day
        datetime(year, 11, 11), # Remembrance Day
        datetime(year, 12, 25), # Christmas
        datetime(year, 12, 26), # Boxing Day
    }
    return floating | fixed


# Cache so you don't recompute per row
_holiday_cache: dict[int, set] = {}

def is_holiday(date: datetime) -> int:
    year = date.year
    if year not in _holiday_cache:
        _holiday_cache[year] = get_canadian_holidays(year)
    return int(date.date() in _holiday_cache[year])


# def compute_days_until_stockout(stock: float, avg_daily_usage: float) -> float:
#     """Forward-looking days until stock hits 0 given current stock and avg usage."""
#     if avg_daily_usage <= 0:
#         return 35.0  # effectively infinite; cap at 35
#     days = stock / avg_daily_usage
#     return float(np.clip(round(days), 1.0, 35.0))
def compute_days_until_stockout(stock, avg_daily_usage, scheduled_supply, current_day):
    proj = stock
    for horizon in range(1, 36):
        proj += scheduled_supply.get(current_day + horizon, 0)
        proj -= avg_daily_usage
        if proj <= 0:
            return float(horizon)
    return 35.0


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATE ONE SUPPLY_ID TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────

# def simulate_supply(supply_id: str, blood_type: str, n_days: int) -> pd.DataFrame:
#     """
#     Simulate daily inventory events for one supply (hospital / blood bank unit).

#     Returns a DataFrame with one row per calendar day.
#     """
#     weight = DEMAND_WEIGHTS[blood_type]

#     # ── Initial conditions ───────────────────────────────────────────────────
#     lead_time = int(RNG.integers(3, 10))           # days between order and arrival
#     restock_interval = int(RNG.integers(7, 21))    # how often they restock
#     restock_volume_mean = float(RNG.uniform(40, 120))

#     stock = float(RNG.uniform(40, 180))            # starting stock
#     stockout_count_90d = 0

#     # Per-day usage baseline (units/day), modulated by blood-type weight
#     base_usage = float(RNG.uniform(4, 14)) * weight

#     # Track 90-day rolling stockout events to replicate stockout_count_90d
#     stockout_days_ring = np.zeros(90, dtype=int)

#     # ── Scheduled supply pipeline: list of (arrival_day, units) tuples ──────
#     # Pre-generate future restock events for the whole series
#     # (mimics the "incoming_supply_scheduled" feature)
#     scheduled_supply: dict[int, float] = {}
#     next_restock_day = restock_interval
#     while next_restock_day < n_days:
#         vol = max(10.0, float(RNG.normal(restock_volume_mean, restock_volume_mean * 0.2)))
#         scheduled_supply[next_restock_day] = vol
#         next_restock_day += int(RNG.integers(restock_interval - 3, restock_interval + 4))

#     # Precompute day-level arrays for usage noise, surgeries, holidays, seasons
#     dates = [START_DATE + timedelta(days=d) for d in range(n_days)]

#     rows = []
#     history_usage: list[float] = []   # running list for rolling demand features

#     for d, date in enumerate(dates):
#         # ── Seasonality + weekend effect on demand ───────────────────────────
#         season = get_season(date)
#         holiday = is_holiday(date)
#         weekend_factor = 0.85 if date.weekday() >= 5 else 1.0
#         season_factor = 1.10 if season == 3 else (0.95 if season == 1 else 1.0)

#         # ── Scheduled surgeries (0–4) ────────────────────────────────────────
#         scheduled_surg = int(RNG.poisson(1.2 * weight))
#         scheduled_surg = min(scheduled_surg, 4)

#         # ── Today's actual usage ─────────────────────────────────────────────
#         surgery_boost = scheduled_surg * RNG.uniform(0.3, 0.8)
#         usage = max(
#             0.5,
#             float(RNG.normal(
#                 base_usage * weekend_factor * season_factor + surgery_boost,
#                 base_usage * 0.15,
#             )),
#         )

#         # Apply scheduled supply arrival (adds to stock at start of day)
#         if d in scheduled_supply:
#             stock += scheduled_supply[d]

#         # Consume stock (can't go below 0)
#         stock = max(0.0, stock - usage)

#         # ── Rolling demand history ───────────────────────────────────────────
#         history_usage.append(usage)
#         hist_7  = float(np.mean(history_usage[-7:]))
#         hist_30 = float(np.mean(history_usage[-30:]))

#         # ── Demand spike indicator ───────────────────────────────────────────
#         spike = int(usage > 1.5 * hist_30) if len(history_usage) >= 30 else 0

#         # ── Incoming supply in next 7 / 30 days ─────────────────────────────
#         incoming_7  = sum(v for day, v in scheduled_supply.items() if d < day <= d + 7)
#         incoming_30 = sum(v for day, v in scheduled_supply.items() if d < day <= d + 30)

#         # ── Expiry rate in next 7 days ────────────────────────────────────────
#         # Simulate: each unit has a random shelf life; proportion near expiry
#         # modelled as fraction of current stock that expires within 7 days
#         expiry_fraction = float(RNG.beta(1.5, 6)) * (1.1 if season == 3 else 1.0)
#         expiry_rate_7d = round(stock * expiry_fraction, 1)

#         # ── days_since_last_restock ──────────────────────────────────────────
#         past_restocks = [day for day in scheduled_supply if day <= d]
#         days_since_restock = (d - max(past_restocks)) if past_restocks else d

#         # ── stockout_count_90d ───────────────────────────────────────────────
#         stockout_days_ring[d % 90] = int(stock <= 0)
#         stockout_count_90d = int(stockout_days_ring.sum())

#         # ── days_until_stockout (forward label) ─────────────────────────────
#         avg_usage_14d = float(np.mean(history_usage[-14:])) if history_usage else base_usage
#         days_until_so = compute_days_until_stockout(stock, avg_usage_14d, scheduled_supply, d)

#         # ── Multi-horizon stock projections ─────────────────────────────────
#         # Compute projected stock at t+1, t+7, t+30 using scheduled supply
#         # and average recent demand (no look-ahead into random future noise)
#         def project_stock(horizon: int) -> float:
#             proj = stock
#             proj += sum(v for day, v in scheduled_supply.items() if d < day <= d + horizon)
#             proj -= avg_usage_14d * horizon
#             return float(np.clip(round(proj, 1), 0.0, None))

#         stock_t1  = project_stock(1)
#         stock_t7  = project_stock(7)
#         stock_t30 = project_stock(30)

#         rows.append({
#             # ── identifiers ─────────────────────────────────────────────────
#             "supply_id":                    supply_id,
#             "event_timestamp":              date,
#             # ── existing features ────────────────────────────────────────────
#             "blood_product_type":           blood_type,
#             "current_stock_units":          round(stock, 1),
#             "usage_today":                  round(usage, 2),
#             "lead_time_days":               lead_time,
#             "days_since_last_restock":      days_since_restock,
#             "stockout_count_90d":           stockout_count_90d,
#             "scheduled_surgeries_next7d":   scheduled_surg,
#             # ── new features ─────────────────────────────────────────────────
#             "historical_demand_7d":         round(hist_7, 2),
#             "historical_demand_30d":        round(hist_30, 2),
#             "incoming_supply_scheduled_7d": round(incoming_7, 1),
#             "incoming_supply_scheduled_30d":round(incoming_30, 1),
#             "expiry_rate_7d":               expiry_rate_7d,
#             "demand_spike_indicator":       spike,
#             "is_holiday":                   holiday,
#             "season":                       season,
#             # ── labels ───────────────────────────────────────────────────────
#             "days_until_stockout":          days_until_so,
#             "stock_units_t1":               stock_t1,
#             "stock_units_t7":               stock_t7,
#             "stock_units_t30":              stock_t30,
#         })

#     return pd.DataFrame(rows)
def simulate_supply(
    supply_id: str,
    blood_type: str,
    n_days: int,
    hospital_id: str = "H001",
    stock_base: float = 80.0,
    usage_base: float = 10.0,
    lead_time_days: int = 3,
    volatility: float = 0.15,
    surgery_base: int = 10,
) -> pd.DataFrame:
    """
    Simulate daily blood inventory for one supply_id.

    Parameters are calibrated from BLOOD_TYPE_PROFILES × HOSPITAL_SIZE_PROFILES
    in generate_dataset() so that mean stock oscillates around stock_base.
    """
    # ── Restock schedule calibrated to actual usage ───────────────────────────
    # Interval must stay within "days of supply at base stock" so stock rarely
    # hits zero mid-cycle (stockout floor causes ratchet accumulation otherwise).
    days_of_stock = stock_base / max(usage_base, 0.1)
    interval_lo = max(2, int(days_of_stock * 0.40))
    interval_hi = max(interval_lo + 1, int(days_of_stock * 0.85))
    restock_interval = int(RNG.integers(interval_lo, interval_hi + 1))
    # Replenish exactly one interval's worth of consumption ± 15% noise.
    restock_volume_mean = usage_base * restock_interval * float(RNG.uniform(0.85, 1.15))

    # Start near stock_base with ±25% noise
    stock = float(RNG.normal(stock_base, stock_base * 0.25))
    stock = max(stock_base * 0.3, stock)  # floor at 30% of base

    stockout_days_ring = np.zeros(90, dtype=int)

    # ── Scheduled supply with NOISE + UNCERTAINTY ─────────────────────────────
    scheduled_supply: dict[int, float] = {}
    next_restock_day = restock_interval
    while next_restock_day < n_days:
        vol = max(0.0, float(RNG.normal(restock_volume_mean, restock_volume_mean * 0.2)))
        delay = int(RNG.integers(0, max(1, lead_time_days) + 1))  # 0..lead_time, always forward
        actual_day = min(n_days - 1, next_restock_day + delay)
        scheduled_supply[actual_day] = scheduled_supply.get(actual_day, 0.0) + vol
        next_restock_day += max(1, int(RNG.integers(max(1, restock_interval - 1), restock_interval + 3)))

    dates = [START_DATE + timedelta(days=d) for d in range(n_days)]
    history_usage: list[float] = []
    history_stock: list[float] = []  # P7: rolling stock for trend feature

    rows = []

    for d, date in enumerate(dates):
        season = get_season(date)
        holiday = is_holiday(date)

        weekend_factor = 0.85 if date.weekday() >= 5 else 1.0
        season_factor = 1.10 if season == 3 else (0.95 if season == 1 else 1.0)

        # Surgeries driven by hospital surgery_base calibrated from size profile
        scheduled_surg = int(RNG.poisson(max(0.5, surgery_base * 0.08)))
        scheduled_surg = min(scheduled_surg, surgery_base)

        # noisy real demand around calibrated usage_base.
        # usage_base from seed_data already captures total consumption including surgeries.
        demand_noise = RNG.normal(1.0, volatility)

        usage = max(
            0.2,
            float(
                RNG.normal(
                    usage_base * weekend_factor * season_factor * demand_noise,
                    usage_base * 0.20,
                )
            ),
        )

        # apply supply
        if d in scheduled_supply:
            stock += scheduled_supply[d]

        # consume
        stock = max(0.0, stock - usage)

        history_usage.append(usage)
        history_stock.append(stock)

        # P7: rolling stock slope — (now − 7d ago) / 7, signals sustained depletion
        stock_trend_7d = (
            (stock - history_stock[-8]) / 7.0   # [-8] is 7 days ago when current is appended
            if len(history_stock) >= 8 else 0.0
        )

        hist_7 = float(np.mean(history_usage[-7:]))
        hist_30 = float(np.mean(history_usage[-30:]))

        # lag features (realistic signal)
        lag_1 = history_usage[-1] if len(history_usage) >= 1 else usage
        lag_3 = np.mean(history_usage[-3:]) if len(history_usage) >= 3 else usage

        spike = int(usage > 1.5 * hist_30) if len(history_usage) >= 30 else 0

        # 🔥 IMPERFECT knowledge of future supply (what model sees)
        incoming_7_est = sum(
            v * RNG.uniform(0.7, 1.1)
            for day, v in scheduled_supply.items()
            if d < day <= d + 7
        )

        incoming_30_est = sum(
            v * RNG.uniform(0.7, 1.1)
            for day, v in scheduled_supply.items()
            if d < day <= d + 30
        )

        expiry_rate_7d = round(stock * float(RNG.beta(1.5, 6)), 1)

        past_restocks = [day for day in scheduled_supply if day <= d]
        days_since_restock = (d - max(past_restocks)) if past_restocks else d

        stockout_days_ring[d % 90] = int(stock <= 0)
        stockout_count_90d = int(stockout_days_ring.sum())

        # ─────────────────────────────────────────────────────────────────
        # P4: MC-averaged labels (30 paths, local RNG seeded from global)
        #
        # Seeding a local RNG here (once per row) means the 30 path draws
        # never pollute the global RNG stream used for feature generation,
        # so adding/removing supply IDs does NOT silently shift every other
        # row's label (partial P8 fix — P8 completes the audit).
        # ─────────────────────────────────────────────────────────────────
        def simulate_future_path_mean(n_paths: int = 30) -> tuple[float, float, float]:
            local_rng = np.random.default_rng(int(RNG.integers(0, 2**32)))
            t1_acc: list[float] = []
            t7_acc: list[float] = []
            t30_acc: list[float] = []

            for _ in range(n_paths):
                proj = stock
                cumulative_drift = 0.0
                t1_val = t7_val = t30_val = 0.0

                for h in range(1, 31):
                    # demand drifts unpredictably — compounds each step
                    cumulative_drift += float(local_rng.normal(0, hist_7 * 0.08))
                    drifted_demand_mean = max(1.0, hist_7 + cumulative_drift)

                    # noise std grows with horizon — more aggressive at t30
                    horizon_std = hist_7 * (0.15 + (h / 30) * 0.6)
                    future_demand = max(0.5, float(local_rng.normal(drifted_demand_mean, horizon_std)))

                    future_supply = scheduled_supply.get(d + h, 0)

                    # supply failure probability grows with horizon
                    failure_prob = 0.05 + (h / 30) * 0.25  # 5% at t+1 → 30% at t+30
                    if local_rng.random() < failure_prob:
                        future_supply *= local_rng.uniform(0.0, 0.5)

                    proj = max(0.0, proj + future_supply - future_demand)
                    if h == 1:  t1_val  = round(proj, 1)
                    if h == 7:  t7_val  = round(proj, 1)
                    if h == 30: t30_val = round(proj, 1)

                t1_acc.append(t1_val)
                t7_acc.append(t7_val)
                t30_acc.append(t30_val)

            return (
                round(float(np.mean(t1_acc)), 1),
                round(float(np.mean(t7_acc)), 1),
                round(float(np.mean(t30_acc)), 1),
            )

        stock_t1, stock_t7, stock_t30 = simulate_future_path_mean(n_paths=N_PATHS_MC)

        rows.append({
            "supply_id": supply_id,
            "event_timestamp": date,

            # features
            "hospital_id": hospital_id,          # P9: hospital entity — systematic demand level
            "blood_product_type": blood_type,
            "current_stock_units": round(stock, 1),
            "usage_today": round(usage, 2),
            "usage_lag_1": round(lag_1, 2),
            "usage_lag_3": round(lag_3, 2),

            "lead_time_days": lead_time_days,
            "days_since_last_restock": days_since_restock,
            "stockout_count_90d": stockout_count_90d,
            "scheduled_surgeries_next7d": scheduled_surg,

            "historical_demand_7d": round(hist_7, 2),
            "historical_demand_30d": round(hist_30, 2),

            # 🔥 imperfect future info
            "incoming_supply_scheduled_7d": round(incoming_7_est, 1),
            "incoming_supply_scheduled_30d": round(incoming_30_est, 1),
            # exact day-1 scheduled supply (signal for restock-day bias reduction)
            "incoming_supply_day1": round(float(scheduled_supply.get(d + 1, 0.0)), 1),

            "expiry_rate_7d": expiry_rate_7d,
            "demand_spike_indicator": spike,
            "is_holiday": holiday,
            "season": season,
            "stock_trend_7d": round(stock_trend_7d, 3),  # P7

            # labels (now noisy & realistic)
            "stock_units_t1":  round(stock_t1, 1),  # negative = depleting
            "stock_units_t7":  round(stock_t7  - stock, 1),  # more negative further out
            "stock_units_t30": round(stock_t30 - stock, 1),  # most negative
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE ALL SUPPLY IDS
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate training data for all 7 QUEBEC_HOSPITALS × 8 BLOOD_TYPES = 56 supply IDs.
    Parameters are calibrated from BLOOD_TYPE_PROFILES × HOSPITAL_SIZE_PROFILES
    so mean stock per supply matches expected operational stock levels.
    """
    blood_types = list(BLOOD_TYPE_PROFILES.keys())  # 8 types in canonical order
    frames = []
    supply_counter = 1

    for hospital in QUEBEC_HOSPITALS:
        size_profile = HOSPITAL_SIZE_PROFILES[hospital.size_tier]
        for blood_type in blood_types:
            bt_profile  = BLOOD_TYPE_PROFILES[blood_type]
            stock_base  = bt_profile["stock_base"]  * size_profile["stock_scale"]
            usage_base  = bt_profile["usage_base"]  * size_profile["usage_scale"]
            lead_time   = size_profile["lead_time_days"]
            volatility  = size_profile["volatility"]
            surgery_base = size_profile["surgery_base"]

            sid = f"S{supply_counter:03d}"
            supply_counter += 1

            print(
                f"  {sid}  {hospital.hospital_id} {hospital.size_tier:<8}"
                f"  {blood_type:<4}  stock_base={stock_base:.0f}"
                f"  usage_base={usage_base:.1f}"
            )
            df = simulate_supply(
                supply_id=sid,
                blood_type=blood_type,
                n_days=N_DAYS,
                hospital_id=hospital.hospital_id,
                stock_base=stock_base,
                usage_base=usage_base,
                lead_time_days=lead_time,
                volatility=volatility,
                surgery_base=surgery_base,
            )
            frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.sort_values(["supply_id", "event_timestamp"]).reset_index(drop=True)

    # Trimmed version for forecast labels only
    full_df_forecast = (
        full_df.groupby("supply_id", group_keys=False)
        .apply(lambda g: g.iloc[:-30])
        .reset_index(drop=True)
    )
    return full_df, full_df_forecast  


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DEFINITIONS  (4 separate dataframes)
# ─────────────────────────────────────────────────────────────────────────────

# ── Model 1: Days-Until-Stockout ─────────────────────────────────────────────
STOCKOUT_FEATURE_COLS = [
    "supply_id",
    "blood_product_type",
    "current_stock_units",
    "usage_today",
    "lead_time_days",
    "days_since_last_restock",
    "stockout_count_90d",
    "scheduled_surgeries_next7d",
    "event_timestamp",          # required by Feast for point-in-time joins
]

STOCKOUT_LABEL_COLS = [
    "supply_id",
    "event_timestamp",
    "days_until_stockout",      # regression target
]

# ── Model 2: Multi-Horizon Stock Forecast ────────────────────────────────────
FORECAST_FEATURE_COLS = [
    "supply_id",
    "hospital_id",           # P9: hospital entity — systematic demand-level feature
    "blood_product_type",
    "current_stock_units",
    "usage_today",
    "usage_lag_1",           # P3: lag-1 daily usage — best predictor for t1
    "usage_lag_3",           # P3: 3-day rolling usage mean
    "lead_time_days",
    "days_since_last_restock",
    "stockout_count_90d",
    "scheduled_surgeries_next7d",
    # demand history
    "historical_demand_7d",
    "historical_demand_30d",
    # supply schedule (imperfect estimates)
    "incoming_supply_scheduled_7d",
    "incoming_supply_scheduled_30d",
    "incoming_supply_day1",  # P6: exact next-day scheduled supply — reduces t1 bias
    # other features
    "expiry_rate_7d",
    "demand_spike_indicator",
    "is_holiday",
    "season",
    "stock_trend_7d",            # P7: 7-day stock slope — helps t7 depletion magnitude
    "event_timestamp",       # required by Feast for point-in-time joins
]

FORECAST_LABEL_COLS = [
    "supply_id",
    "event_timestamp",
    "stock_units_t1",           # projected stock in  1 day  (24H filter)
    "stock_units_t7",           # projected stock in  7 days (1W  filter)
    "stock_units_t30",          # projected stock in 30 days (monthly)
]


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def _write(df: pd.DataFrame, stem: str, directory: str) -> None:
    """Write a dataframe as both parquet and csv."""
    df.to_parquet(os.path.join(directory, f"{stem}.parquet"), index=False)
    df.to_csv(os.path.join(directory, f"{stem}.csv"), index=False)
    print(f"  {stem:<30} → {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"    columns: {list(df.columns)}")


def save_datasets(full_df: pd.DataFrame, full_df_forecast: pd.DataFrame) -> None:
    SCRIPTS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIRECTORY = os.path.dirname(SCRIPTS_DIRECTORY)
    # DATASETS_DIRECTORY = os.path.join(ROOT_DIRECTORY, "datasets_featues_labels_seperated")
    # os.makedirs(DATASETS_DIRECTORY, exist_ok=True)

    print(f"\nSaving to: {ROOT_DIRECTORY}\n")
    #_write(full_df[STOCKOUT_FEATURE_COLS],          "stockout_features",  ROOT_DIRECTORY)
    #_write(full_df[STOCKOUT_LABEL_COLS],            "stockout_labels",    ROOT_DIRECTORY)
    _write(full_df_forecast[FORECAST_FEATURE_COLS], "forecast_features",  ROOT_DIRECTORY)  # ← trimmed
    _write(full_df_forecast[FORECAST_LABEL_COLS],   "forecast_labels",    ROOT_DIRECTORY)  # ← trimmed


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating stockout dataset…\n")
    full_df, full_df_forecast = generate_dataset()

    # Quick sanity checks
    # Weighted expected stock_base across 7 hospitals (4 large ×1.75, 3 urban ×1.3)
    n_large  = sum(1 for h in QUEBEC_HOSPITALS if h.size_tier == "large")
    n_urban  = sum(1 for h in QUEBEC_HOSPITALS if h.size_tier == "urban")
    n_total  = len(QUEBEC_HOSPITALS)
    large_scale = HOSPITAL_SIZE_PROFILES["large"]["stock_scale"]
    urban_scale = HOSPITAL_SIZE_PROFILES["urban"]["stock_scale"]

    print("\n-- Stock level validation (mean current_stock_units per blood type) --")
    stock_check = (
        full_df_forecast
        .groupby("blood_product_type")["current_stock_units"]
        .mean()
        .round(1)
        .sort_values(ascending=False)
    )
    for bt, mean_stock in stock_check.items():
        raw_base = BLOOD_TYPE_PROFILES[bt]["stock_base"]
        expected = raw_base * (n_large * large_scale + n_urban * urban_scale) / n_total
        ratio = mean_stock / expected
        status = "OK  " if 0.5 <= ratio <= 2.5 else "WARN"
        print(f"  {status} {bt:<4}  mean={mean_stock:>7.1f}  expected~{expected:.0f}  ratio={ratio:.2f}")

    print("\n-- Forecast feature sample (new cols only) --")
    new_cols = ["supply_id", "hospital_id", "event_timestamp", "blood_product_type",
                "current_stock_units", "usage_today", "historical_demand_7d"]
    print(full_df_forecast[new_cols].head(5).to_string())
    print("\n-- Forecast label sample --")
    print(full_df_forecast[FORECAST_LABEL_COLS].head(3).to_string())
    print("\n-- Label stats --")

    save_datasets(full_df, full_df_forecast)
    print("\nDone.")