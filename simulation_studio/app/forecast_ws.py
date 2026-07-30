"""Shared WebSocket helpers for live Demand Forecast panels (sim + digital twin)."""

from __future__ import annotations

from collections import deque
from typing import Any

from .operational_world import (
    BLOOD_DISTRIBUTION,
    BLOOD_TYPES,
    advance_operational_tick,
    utc_now_iso,
)
from .twin_forecast import build_tick_context, run_tick_forecast
from .twin_forecast_config import (
    default_job_id as default_forecast_job_id,
    forecast_panel_meta,
    list_panel_options,
    load_twin_forecast_config,
)


def forecast_panel_enabled() -> bool:
    try:
        return bool(load_twin_forecast_config().get("enabled"))
    except Exception:
        return False


def valid_forecast_job_id(job_id: str | None) -> str | None:
    clean = str(job_id or "").strip()
    if not clean or not forecast_panel_enabled():
        return None
    allowed = {str(row.get("job_id")) for row in list_panel_options()}
    return clean if clean in allowed else None


def resolve_forecast_job_id(start_msg: dict[str, Any] | None) -> str | None:
    start_msg = start_msg or {}
    return valid_forecast_job_id(
        str(start_msg.get("forecast_job_id") or default_forecast_job_id() or "")
    )


# --- Per-center realized demand (feeds usage_today) ------------------------
# The model's dominant feature is the rolling mean of `usage_today`. Feeding a
# flat proxy makes every entity predict the same value, so instead we derive each
# hospital's *actual* recent demand from its cumulative `transfused` count: a
# per-tick delta normalized to a daily rate (same technique as
# `compute_realized_metrics`), then split across blood types by BLOOD_DISTRIBUTION.
_DEMAND_STATE: dict[str, Any] = {
    "last_tick": None,
    "baseline_cum": {},
    "current_cum": {},
    "baseline_hour": None,
    "current_hour": None,
}


def reset_demand_history() -> None:
    """Clear accumulated per-center demand state (call when a new run starts)."""
    _DEMAND_STATE.update(
        last_tick=None,
        baseline_cum={},
        current_cum={},
        baseline_hour=None,
        current_hour=None,
    )


def _center_demand_per_day(
    centers: list[dict[str, Any]] | None,
    *,
    tick_number: int | None = None,
    simulated_hour: float | None = None,
) -> dict[str, float]:
    """Per-hospital realized demand rate (units/day) from cumulative transfusions.

    Advances once per tick; re-renders of the same tick reuse the stored window;
    a backward ``tick_number`` jump resets (new run).
    """
    st = _DEMAND_STATE
    last_tick = st["last_tick"]
    if tick_number is not None and last_tick is not None and tick_number < last_tick:
        reset_demand_history()
        last_tick = None
    advance = tick_number is None or last_tick is None or tick_number != last_tick

    cum: dict[str, float] = {}
    for center in centers or []:
        if not isinstance(center, dict):
            continue
        hid = str(center.get("external_id") or center.get("name") or "").strip()
        if not hid:
            continue
        stats = center.get("stats") if isinstance(center.get("stats"), dict) else {}
        cum[hid] = float(stats.get("transfused") or 0.0)

    now_h = float(simulated_hour) if simulated_hour is not None else None
    if advance:
        st["baseline_cum"] = st["current_cum"]
        st["current_cum"] = cum
        st["baseline_hour"] = st["current_hour"]
        st["current_hour"] = now_h
        if tick_number is not None:
            st["last_tick"] = tick_number

    base = st["baseline_cum"]
    curr = st["current_cum"] if st["current_cum"] else cum

    dh = None
    if st["current_hour"] is not None and st["baseline_hour"] is not None:
        dh = st["current_hour"] - st["baseline_hour"]

    def per_day(value: float) -> float:
        return value * 24.0 / dh if (dh and dh > 0) else value

    demand: dict[str, float] = {}
    for hid, value in curr.items():
        baseline_value = (base or {}).get(hid)
        delta = max(0.0, value - baseline_value) if baseline_value is not None else 0.0
        demand[hid] = per_day(delta)
    return demand


def blood_supplies_from_center_snapshots(
    centers: list[dict[str, Any]] | None,
    *,
    tick_number: int | None = None,
    simulated_hour: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    now = utc_now_iso()
    demand_by_center = _center_demand_per_day(
        centers, tick_number=tick_number, simulated_hour=simulated_hour
    )
    for center in centers or []:
        if not isinstance(center, dict):
            continue
        role = str(center.get("type") or center.get("role") or "").lower()
        if role and role not in {"hospital", "mobile"}:
            continue
        hospital_id = str(center.get("external_id") or center.get("name") or "").strip()
        if not hospital_id:
            continue
        hospital_name = str(center.get("name") or hospital_id)
        inventory = center.get("inventory") if isinstance(center.get("inventory"), dict) else {}
        # Dashboard demand model is RBC; split that pool across ABO/Rh types.
        rbc_units = float(inventory.get("RBC") or center.get("total_units") or 0.0)
        center_demand = float(demand_by_center.get(hospital_id, 0.0))
        for blood in BLOOD_TYPES:
            share = BLOOD_DISTRIBUTION.get(blood, 0.0)
            rows.append(
                {
                    "supply_id": f"{hospital_id}-{blood}",
                    "hospital_id": hospital_id,
                    "hospital_name": hospital_name,
                    "hospital": {"hospital_id": hospital_id, "name": hospital_name},
                    "blood_product_type": blood,
                    "current_stock_units": round(rbc_units * share, 2),
                    "usage_today": round(center_demand * share, 3),
                    "event_timestamp": now,
                    "updated_at": now,
                }
            )
    return rows


# --- Rolling demand-history enrichment -------------------------------------
# The live forecast payload only carries the *current* tick's usage/stock per
# entity, but the demand model leans hardest on recent-usage trend features
# (usage_30d_mean dominates its importance). We accumulate a short per-entity
# history here so the Demand Forecast panel can feed real 7d/30d rolling means
# and react to surges, instead of a flat baseline. Windows are sized in
# simulated hours so they are robust to the configured step size.
_FORECAST_WINDOW_7D_H = 7 * 24
_FORECAST_WINDOW_30D_H = 30 * 24
_FORECAST_HIST: dict[tuple[str, str], deque[tuple[float | None, float, float]]] = {}
_FORECAST_HIST_STATE: dict[str, Any] = {"last_tick": None}


def reset_forecast_history() -> None:
    """Clear accumulated rolling history (call when a new run starts)."""
    _FORECAST_HIST.clear()
    _FORECAST_HIST_STATE["last_tick"] = None


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def enrich_with_rolling_features(
    rows: list[dict[str, Any]],
    *,
    tick_number: int | None = None,
    simulated_hour: float | None = None,
    window_7d_h: float = _FORECAST_WINDOW_7D_H,
    window_30d_h: float = _FORECAST_WINDOW_30D_H,
) -> list[dict[str, Any]]:
    """Attach 7d/30d rolling usage & inventory means to each supply row.

    History is keyed by (hospital_id, blood_product_type) and advances once per
    tick (re-renders of the same tick recompute without double-counting). A
    backward jump in ``tick_number`` is treated as a new run and resets history.

    ``window_7d_h`` / ``window_30d_h`` size the trend windows in simulated hours.
    They default to true 7d/30d, but a short fast-forward sim can pass smaller
    values so ``usage_30d_mean`` reflects *recent* demand (the model's dominant
    feature) instead of the whole-run average — letting the forecast react.
    """
    last = _FORECAST_HIST_STATE.get("last_tick")
    if tick_number is not None and last is not None and tick_number < last:
        reset_forecast_history()
        last = None
    advance = tick_number is None or last is None or tick_number != last
    now_h = float(simulated_hour) if simulated_hour is not None else None

    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("hospital_id") or ""),
            str(row.get("blood_product_type") or ""),
        )
        hist = _FORECAST_HIST.setdefault(key, deque(maxlen=2048))
        if advance:
            hist.append(
                (
                    now_h,
                    float(row.get("usage_today") or 0.0),
                    float(row.get("current_stock_units") or 0.0),
                )
            )
        if now_h is not None:
            w7 = [(u, s) for (h, u, s) in hist if h is None or h >= now_h - window_7d_h]
            w30 = [(u, s) for (h, u, s) in hist if h is None or h >= now_h - window_30d_h]
        else:
            seq = list(hist)
            w30 = [(u, s) for (_h, u, s) in seq]
            w7 = w30[-28:]
        row["usage_7d_mean"] = _mean([u for u, _s in w7])
        row["usage_30d_mean"] = _mean([u for u, _s in w30])
        row["inventory_7d_mean"] = _mean([s for _u, s in w7])
        row["inventory_30d_mean"] = _mean([s for _u, s in w30])

    if tick_number is not None:
        _FORECAST_HIST_STATE["last_tick"] = tick_number
    return rows


# --- Realized ("actual") metrics for the model-validation panel ------------
# The validation panel overlays a model's prediction against the simulation's
# realized counterpart. We derive per-day realized rates from the cumulative
# per-center stats (transfused/donated/expired/collected) plus the digital-twin
# `simulator_metrics` (shortage). Cumulatives are turned into per-tick deltas and
# normalized to a daily rate so they sit on the same basis as the (daily) models.
_REALIZED_STATE: dict[str, Any] = {
    "last_tick": None,
    "baseline_cum": None,
    "current_cum": None,
    "baseline_hour": None,
    "current_hour": None,
    # Trailing (hour, raw_metrics) window for optional smoothing of the actual line.
    "hist": deque(maxlen=2048),
}


# --- Forward-looking stockout frequency (honest comparator for risk models) ---
# The stockout-risk / hazard models predict P(an entity stocks out within 7 days).
# The matching realized signal is *the frequency of that event*, not the
# instantaneous shortage rate (a probability vs a rate is apples-to-oranges). We
# track, per (hospital, blood type), the last simulated hour it was at/below the
# stockout threshold, then report the fraction of present entities that hit
# stockout at least once within a trailing horizon-length window. Under
# stationarity the trailing frequency is an unbiased estimate of the forward
# probability the model emits — the same assumption the demand panel's
# horizon_basis makes — so a calibrated model overlays the realized line.
_STOCKOUT_STATE: dict[str, Any] = {"last_stockout_hour": {}}


def reset_stockout_state() -> None:
    """Clear accumulated per-entity stockout history (call when a new run starts)."""
    _STOCKOUT_STATE["last_stockout_hour"] = {}


def _realized_stockout_freq(
    snapshot: dict[str, Any] | None,
    *,
    simulated_hour: float | None,
    window_hours: float = 168.0,
    threshold: float = 0.0,
) -> float | None:
    """Fraction of entities that stocked out within a trailing ``window_hours``.

    Reads per-entity ``current_stock_units`` from the snapshot's ``blood_supplies``
    (always present, so this works in both sim and twin modes). Returns ``None``
    when there are no entities to measure, so the caller leaves the actual at 0
    rather than fabricating a value.
    """
    rows = (snapshot or {}).get("blood_supplies") or []
    if not rows:
        return None
    last = _STOCKOUT_STATE["last_stockout_hour"]
    now_h = float(simulated_hour) if simulated_hour is not None else None
    present = 0
    stocked_out = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        hid = str(row.get("hospital_id") or "")
        if not hid:
            continue
        key = (hid, str(row.get("blood_product_type") or ""))
        present += 1
        stock = float(row.get("current_stock_units") or 0.0)
        if stock <= threshold and now_h is not None:
            last[key] = now_h
        last_h = last.get(key)
        if last_h is not None and (now_h is None or last_h >= now_h - window_hours):
            stocked_out += 1
    if present == 0:
        return None
    return round(stocked_out / present, 4)


def reset_realized_metrics() -> None:
    """Clear accumulated realized-metric state (call when a new run starts)."""
    _REALIZED_STATE.update(
        last_tick=None,
        baseline_cum=None,
        current_cum=None,
        baseline_hour=None,
        current_hour=None,
    )
    _REALIZED_STATE["hist"].clear()
    reset_stockout_state()


def _cumulative_totals(
    centers: list[dict[str, Any]] | None,
    simulated_state: dict[str, Any] | None,
) -> dict[str, float]:
    totals = {"transfused": 0.0, "donated": 0.0, "expired": 0.0, "collected": 0.0, "shortage": 0.0}
    seen_center = False
    for center in centers or []:
        if not isinstance(center, dict):
            continue
        stats = center.get("stats")
        if not isinstance(stats, dict):
            continue
        seen_center = True
        totals["transfused"] += float(stats.get("transfused") or 0.0)
        totals["donated"] += float(stats.get("donated") or 0.0)
        totals["expired"] += float(stats.get("expired") or 0.0)
        totals["collected"] += float(stats.get("collected") or 0.0)
    metrics = (simulated_state or {}).get("simulator_metrics") or {}
    # shortage is only exposed via the twin's simulator_metrics (not per-center)
    totals["shortage"] = float(metrics.get("total_shortage") or 0.0)
    if not seen_center:
        # No per-center stats this tick (e.g. snapshot-only) -> fall back to the
        # twin's aggregate simulator_metrics so realized wastage/supply still flow
        # instead of silently reading 0.
        totals["transfused"] = float(metrics.get("total_transfused") or 0.0)
        totals["donated"] = float(metrics.get("total_donated") or 0.0)
        totals["expired"] = float(metrics.get("total_expired") or 0.0)
        totals["collected"] = float(metrics.get("total_collected") or 0.0)
    return totals


def compute_realized_metrics(
    centers: list[dict[str, Any]] | None,
    simulated_state: dict[str, Any] | None,
    snapshot: dict[str, Any] | None = None,
    *,
    tick_number: int | None = None,
    simulated_hour: float | None = None,
    smoothing_hours: float = 0.0,
    stockout_window_hours: float = 168.0,
    stockout_threshold: float = 0.0,
) -> dict[str, float]:
    """Per-day realized rates the validation panel compares predictions against.

    Keys: ``realized_demand`` (transfused+shortage/day), ``realized_supply``
    (collected or donated/day), ``realized_wastage`` (expired/day),
    ``realized_shortage_rate`` (fraction of demand unmet this window), and
    ``realized_stockout_freq`` (fraction of entities that hit stockout within a
    trailing ``stockout_window_hours`` — the honest comparator for the stockout
    risk/hazard models). Advances once per tick; re-renders of the same tick reuse
    the stored window; a backward ``tick_number`` jump resets (new run).

    With ``smoothing_hours`` > 0, each metric is returned as the mean over a
    trailing window of that many simulated hours. A single tick annualizes a short
    window (``× 24/Δh``), which amplifies Poisson noise into spikes; smoothing puts
    the actual line on the same gentle basis as the model's rolling-mean forecast.
    """
    cum = _cumulative_totals(centers, simulated_state)
    now_h = float(simulated_hour) if simulated_hour is not None else None
    st = _REALIZED_STATE

    last_tick = st["last_tick"]
    if tick_number is not None and last_tick is not None and tick_number < last_tick:
        reset_realized_metrics()
        last_tick = None
    advance = tick_number is None or last_tick is None or tick_number != last_tick
    if advance:
        st["baseline_cum"] = st["current_cum"]
        st["current_cum"] = cum
        st["baseline_hour"] = st["current_hour"]
        st["current_hour"] = now_h
        if tick_number is not None:
            st["last_tick"] = tick_number

    base = st["baseline_cum"]
    curr = st["current_cum"] if st["current_cum"] is not None else cum
    if base is None:
        delta = {key: 0.0 for key in curr}
    else:
        delta = {key: max(0.0, curr.get(key, 0.0) - base.get(key, 0.0)) for key in curr}

    dh = None
    if st["current_hour"] is not None and st["baseline_hour"] is not None:
        dh = st["current_hour"] - st["baseline_hour"]

    def per_day(value: float) -> float:
        return value * 24.0 / dh if (dh and dh > 0) else value

    served = delta.get("transfused", 0.0) + delta.get("shortage", 0.0)
    raw = {
        "realized_demand": round(per_day(served), 4),
        "realized_supply": round(per_day(delta.get("collected") or delta.get("donated") or 0.0), 4),
        "realized_wastage": round(per_day(delta.get("expired", 0.0)), 4),
        "realized_shortage_rate": round(delta.get("shortage", 0.0) / served if served > 0 else 0.0, 4),
    }
    # Trailing stockout frequency is event-based (not a cumulative delta), so it is
    # sampled every call and merged in after any smoothing of the rate metrics.
    stockout_freq = _realized_stockout_freq(
        snapshot,
        simulated_hour=now_h,
        window_hours=stockout_window_hours,
        threshold=stockout_threshold,
    )
    if smoothing_hours and now_h is not None:
        hist = st["hist"]
        if advance:
            hist.append((now_h, raw))
        window = [m for (h, m) in hist if h >= now_h - smoothing_hours]
        if window:
            raw = {key: round(sum(m[key] for m in window) / len(window), 4) for key in raw}
    if stockout_freq is not None:
        raw["realized_stockout_freq"] = stockout_freq
    return raw


def merge_forecast_snapshot_payload(
    snapshot: dict[str, Any] | None,
    *,
    centers: list[dict[str, Any]] | None = None,
    tick_number: int | None = None,
    simulated_hour: float | None = None,
) -> dict[str, Any]:
    payload = dict((snapshot or {}).get("payload") or {})
    live_rows = blood_supplies_from_center_snapshots(
        centers, tick_number=tick_number, simulated_hour=simulated_hour
    )
    if live_rows:
        payload["blood_supplies"] = live_rows
    return payload


# Tracks which simulated day last committed a validation chart point, so we emit
# one forecast-vs-realized point per model-horizon day (not per 6h tick).
_FORECAST_CADENCE_STATE: dict[str, Any] = {"last_day": None}


def reset_forecast_cadence() -> None:
    """Clear the daily chart-cadence marker (call when a new run starts)."""
    _FORECAST_CADENCE_STATE["last_day"] = None


def build_forecast_panel(
    job_id: str | None,
    *,
    tick_number: int,
    simulated_hour: float,
    simulated_state: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
    centers: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    selected = valid_forecast_job_id(job_id)
    if not selected:
        return None
    tick_context = build_tick_context(
        tick_number=tick_number,
        simulated_hour=simulated_hour,
        simulated_state=simulated_state,
    )
    payload = merge_forecast_snapshot_payload(
        snapshot,
        centers=centers,
        tick_number=tick_number,
        simulated_hour=simulated_hour,
    )
    if not payload.get("blood_supplies"):
        advance_operational_tick()
    # Sim-timescale tuning for the validation panel (defaults preserve true 7d/30d
    # windows and unsmoothed actuals): shrink the trend windows so the forecast
    # reacts, and smooth the realized line so the two track over a comparable span.
    chart_cfg = (load_twin_forecast_config().get("chart") or {})
    enrich_with_rolling_features(
        payload.get("blood_supplies") or [],
        tick_number=tick_number,
        simulated_hour=simulated_hour,
        window_7d_h=float(chart_cfg.get("usage_window_7d_hours") or _FORECAST_WINDOW_7D_H),
        window_30d_h=float(chart_cfg.get("usage_window_30d_hours") or _FORECAST_WINDOW_30D_H),
    )
    # Forecast cadence. The demand model has a 1-day horizon, so by default we
    # commit one validation point per simulated day and measure realized demand
    # over that whole day — matching the model's horizon for an honest comparison
    # and removing the 6h annualization noise. The predictions table still updates
    # every tick. Set forecast_interval_hours to 0 for the legacy per-tick cadence.
    interval_h = float(chart_cfg.get("forecast_interval_hours") or 0.0)
    # Stockout-frequency comparator window/threshold (used by the risk/hazard panels).
    stockout_window_h = float(chart_cfg.get("stockout_window_hours") or 168.0)
    stockout_threshold = float(chart_cfg.get("stockout_threshold") or 0.0)
    new_interval = True
    if interval_h > 0 and simulated_hour is not None:
        day = int(float(simulated_hour) // interval_h)
        last_day = _FORECAST_CADENCE_STATE["last_day"]
        if last_day is not None and day < last_day:  # clock rewound -> new run
            last_day = None
        new_interval = day != last_day
        # Advance realized once per day so the delta spans the full interval
        # (dh ≈ interval -> per_day == the day's true total; no smoothing needed).
        realized = compute_realized_metrics(
            centers, simulated_state, payload,
            tick_number=day, simulated_hour=simulated_hour, smoothing_hours=0.0,
            stockout_window_hours=stockout_window_h, stockout_threshold=stockout_threshold,
        )
        if new_interval:
            _FORECAST_CADENCE_STATE["last_day"] = day
    else:
        realized = compute_realized_metrics(
            centers, simulated_state, payload,
            tick_number=tick_number, simulated_hour=simulated_hour,
            smoothing_hours=float(chart_cfg.get("realized_smoothing_hours") or 0.0),
            stockout_window_hours=stockout_window_h, stockout_threshold=stockout_threshold,
        )

    panel = run_tick_forecast(
        selected,
        snapshot_payload=payload,
        tick_context=tick_context,
        realized=realized,
    )
    if panel is not None and not new_interval:
        # Mid-day tick: keep the predictions table live but don't advance the chart.
        panel.pop("chart_point", None)
    return panel