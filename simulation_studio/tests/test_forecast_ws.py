from simulation_studio.app.forecast_ws import (
    blood_supplies_from_center_snapshots,
    compute_realized_metrics,
    enrich_with_rolling_features,
    merge_forecast_snapshot_payload,
    reset_demand_history,
    reset_forecast_history,
    reset_realized_metrics,
)


def _stockout_snapshot(*entities):
    """Build a snapshot payload with (hospital, blood_type, stock) entity rows."""
    return {
        "blood_supplies": [
            {"hospital_id": hid, "blood_product_type": bt, "current_stock_units": stock}
            for hid, bt, stock in entities
        ]
    }


def _hospital(transfused, *, rbc=80, hid="H-A"):
    return {
        "name": "Hospital A",
        "external_id": hid,
        "type": "hospital",
        "inventory": {"RBC": rbc},
        "stats": {"transfused": transfused},
    }


def _centers(transfused, donated, expired):
    return [{"stats": {"transfused": transfused, "donated": donated, "expired": expired}}]


def _supply_row(usage, stock, blood="O+"):
    return {
        "hospital_id": "H1",
        "blood_product_type": blood,
        "usage_today": usage,
        "current_stock_units": stock,
    }


def test_blood_supplies_from_center_snapshots_builds_entity_rows():
    centers = [
        {
            "name": "Hospital A",
            "external_id": "H-A",
            "type": "hospital",
            "total_units": 80,
            "inventory": {"RBC": 80},
            "stats": {"transfused": 40},
        }
    ]
    rows = blood_supplies_from_center_snapshots(centers)
    assert rows
    assert rows[0]["hospital_id"] == "H-A"
    assert rows[0]["blood_product_type"]


def test_blood_supplies_split_stock_by_abo_distribution():
    reset_demand_history()
    rows = blood_supplies_from_center_snapshots(
        [_hospital(0, rbc=100)], tick_number=1, simulated_hour=0.0
    )
    by_type = {r["blood_product_type"]: r["current_stock_units"] for r in rows}
    assert len(rows) == 8
    assert by_type["O+"] == 44.0  # 100 * 0.44
    assert by_type["AB-"] == 0.5  # 100 * 0.005 — no longer a uniform /8 split


def test_usage_today_tracks_transfused_delta_not_cumulative():
    reset_demand_history()
    # tick 1 establishes a baseline -> no demand yet (not the old 0.05*cumulative)
    rows1 = blood_supplies_from_center_snapshots(
        [_hospital(100)], tick_number=1, simulated_hour=0.0
    )
    assert all(r["usage_today"] == 0.0 for r in rows1)
    # tick 2: +12 transfused over 6h -> 48/day, split by ABO share
    rows2 = blood_supplies_from_center_snapshots(
        [_hospital(112)], tick_number=2, simulated_hour=6.0
    )
    by_type = {r["blood_product_type"]: r["usage_today"] for r in rows2}
    assert by_type["O+"] == round(48.0 * 0.44, 3)  # 21.12
    assert by_type["AB-"] == round(48.0 * 0.005, 3)  # 0.24


def test_demand_history_resets_on_backward_tick():
    reset_demand_history()
    blood_supplies_from_center_snapshots(
        [_hospital(500)], tick_number=5, simulated_hour=30.0
    )
    # a backward tick = new run: baseline cleared, so no spurious 500-unit delta
    rows = blood_supplies_from_center_snapshots(
        [_hospital(4)], tick_number=1, simulated_hour=0.0
    )
    assert all(r["usage_today"] == 0.0 for r in rows)


def test_merge_forecast_snapshot_payload_prefers_live_centers():
    payload = merge_forecast_snapshot_payload(
        {"payload": {"blood_supplies": [{"hospital_id": "OLD"}]}},
        centers=[
            {
                "name": "Live H",
                "external_id": "LIVE-1",
                "type": "hospital",
                "total_units": 10,
                "inventory": {"RBC": 10},
                "stats": {},
            }
        ],
    )
    assert payload["blood_supplies"][0]["hospital_id"] == "LIVE-1"


def test_enrich_adds_rolling_means_that_track_usage():
    reset_forecast_history()
    rows = None
    for tick, usage in enumerate([6.0, 9.0, 15.0], start=1):
        rows = [_supply_row(usage, 30.0)]
        enrich_with_rolling_features(rows, tick_number=tick, simulated_hour=tick * 6.0)
    assert rows[0]["usage_7d_mean"] == round((6 + 9 + 15) / 3, 3)
    assert rows[0]["usage_30d_mean"] == round((6 + 9 + 15) / 3, 3)
    assert rows[0]["inventory_7d_mean"] == 30.0


def test_enrich_does_not_double_count_same_tick():
    reset_forecast_history()
    enrich_with_rolling_features([_supply_row(10.0, 20.0)], tick_number=1, simulated_hour=6.0)
    rows = [_supply_row(10.0, 20.0)]
    enrich_with_rolling_features(rows, tick_number=1, simulated_hour=6.0)  # re-render
    assert rows[0]["usage_30d_mean"] == 10.0


def test_enrich_resets_history_on_new_run():
    reset_forecast_history()
    enrich_with_rolling_features([_supply_row(100.0, 5.0)], tick_number=5, simulated_hour=30.0)
    rows = [_supply_row(8.0, 40.0)]
    enrich_with_rolling_features(rows, tick_number=1, simulated_hour=6.0)  # tick rewinds
    assert rows[0]["usage_30d_mean"] == 8.0  # stale surge dropped


def test_enrich_windows_means_by_simulated_hour():
    reset_forecast_history()
    enrich_with_rolling_features([_supply_row(2.0, 10.0)], tick_number=1, simulated_hour=0.0)
    rows = [_supply_row(20.0, 50.0)]
    enrich_with_rolling_features(rows, tick_number=2, simulated_hour=8 * 24.0)  # +8 days
    assert rows[0]["usage_7d_mean"] == 20.0  # day-0 sample is outside the 7d window
    assert rows[0]["usage_30d_mean"] == round((2 + 20) / 2, 3)


def test_realized_metrics_converts_deltas_to_daily_rates():
    reset_realized_metrics()
    sim = {"simulator_metrics": {"total_shortage": 0}}
    compute_realized_metrics(_centers(0, 0, 0), sim, tick_number=1, simulated_hour=0.0)
    metrics = compute_realized_metrics(_centers(10, 8, 2), sim, tick_number=2, simulated_hour=6.0)
    # deltas over a 6h window are scaled to per-day (x4)
    assert metrics["realized_demand"] == 40.0
    assert metrics["realized_supply"] == 32.0
    assert metrics["realized_wastage"] == 8.0


def test_realized_metrics_shortage_rate_from_simulator_metrics():
    reset_realized_metrics()
    compute_realized_metrics(
        _centers(0, 0, 0), {"simulator_metrics": {"total_shortage": 0}}, tick_number=1, simulated_hour=0.0
    )
    metrics = compute_realized_metrics(
        _centers(8, 0, 0), {"simulator_metrics": {"total_shortage": 2}}, tick_number=2, simulated_hour=6.0
    )
    assert metrics["realized_shortage_rate"] == 0.2  # 2 unmet of 10 demanded
    assert metrics["realized_demand"] == 40.0  # (8 served + 2 unmet) x4


def test_realized_metrics_same_tick_not_double_counted():
    reset_realized_metrics()
    compute_realized_metrics(_centers(0, 0, 0), {}, tick_number=1, simulated_hour=0.0)
    first = compute_realized_metrics(_centers(10, 0, 0), {}, tick_number=2, simulated_hour=6.0)
    rerender = compute_realized_metrics(_centers(10, 0, 0), {}, tick_number=2, simulated_hour=6.0)
    assert first == rerender


def test_realized_metrics_resets_on_new_run():
    reset_realized_metrics()
    compute_realized_metrics(_centers(100, 0, 0), {}, tick_number=5, simulated_hour=30.0)
    compute_realized_metrics(_centers(110, 0, 0), {}, tick_number=6, simulated_hour=36.0)
    metrics = compute_realized_metrics(_centers(0, 0, 0), {}, tick_number=1, simulated_hour=0.0)
    assert metrics["realized_demand"] == 0.0  # stale run discarded


# --- Phase 2: realized smoothing + reactive forecast windows ---------------

def test_realized_smoothing_averages_over_window():
    reset_realized_metrics()
    sim = {"simulator_metrics": {"total_shortage": 0}}
    m1 = compute_realized_metrics(_centers(0, 0, 0), sim, tick_number=1, simulated_hour=0.0, smoothing_hours=48.0)
    assert m1["realized_demand"] == 0.0  # baseline tick
    # raw per-tick demand is 48/day each tick; smoothing reports the trailing mean
    m2 = compute_realized_metrics(_centers(12, 0, 0), sim, tick_number=2, simulated_hour=6.0, smoothing_hours=48.0)
    assert m2["realized_demand"] == 24.0  # mean(0, 48)
    m3 = compute_realized_metrics(_centers(24, 0, 0), sim, tick_number=3, simulated_hour=12.0, smoothing_hours=48.0)
    assert m3["realized_demand"] == 32.0  # mean(0, 48, 48)


def test_realized_smoothing_window_drops_old_entries():
    reset_realized_metrics()
    sim = {"simulator_metrics": {"total_shortage": 0}}
    compute_realized_metrics(_centers(0, 0, 0), sim, tick_number=1, simulated_hour=0.0, smoothing_hours=10.0)
    compute_realized_metrics(_centers(12, 0, 0), sim, tick_number=2, simulated_hour=6.0, smoothing_hours=10.0)
    # at hour 18 the window is [>= 8h], so only the hour-18 sample survives (no spike carry-over)
    m3 = compute_realized_metrics(_centers(24, 0, 0), sim, tick_number=3, simulated_hour=18.0, smoothing_hours=10.0)
    assert m3["realized_demand"] == 24.0  # 12 over 12h -> 24/day, alone in window


def test_realized_daily_total_over_full_interval():
    # Daily cadence drives realized by DAY index (as build_forecast_panel does):
    # mid-day ticks (same index) don't advance, so the day-boundary delta is the
    # true daily demand -- no 6h annualization spikes.
    reset_realized_metrics()
    sim = {"simulator_metrics": {"total_shortage": 0}}
    compute_realized_metrics(_centers(0, 0, 0), sim, tick_number=0, simulated_hour=0.0)
    compute_realized_metrics(_centers(20, 0, 0), sim, tick_number=0, simulated_hour=6.0)   # mid-day, frozen
    compute_realized_metrics(_centers(50, 0, 0), sim, tick_number=0, simulated_hour=12.0)  # mid-day, frozen
    m = compute_realized_metrics(_centers(90, 0, 0), sim, tick_number=1, simulated_hour=24.0)
    assert m["realized_demand"] == 90.0  # 90 transfused across the day, dh=24 -> 90/day (not x4 spiked)


# --- Stockout-frequency comparator (honest signal for risk/hazard panels) ---

def test_realized_stockout_freq_counts_entities_that_hit_zero():
    reset_realized_metrics()
    snap = _stockout_snapshot(("H1", "O+", 0), ("H1", "A+", 50))
    m = compute_realized_metrics([], {}, snap, tick_number=1, simulated_hour=0.0)
    assert m["realized_stockout_freq"] == 0.5  # 1 of 2 entities out


def test_realized_stockout_freq_persists_within_window_then_decays():
    reset_realized_metrics()
    out = _stockout_snapshot(("H1", "O+", 0))
    healthy = _stockout_snapshot(("H1", "O+", 50))
    m0 = compute_realized_metrics([], {}, out, tick_number=1, simulated_hour=0.0, stockout_window_hours=168)
    assert m0["realized_stockout_freq"] == 1.0
    # recovered but still inside the trailing 7-day window of the hour-0 stockout
    m1 = compute_realized_metrics([], {}, healthy, tick_number=2, simulated_hour=100.0, stockout_window_hours=168)
    assert m1["realized_stockout_freq"] == 1.0
    # past the window -> the stale stockout no longer counts
    m2 = compute_realized_metrics([], {}, healthy, tick_number=3, simulated_hour=200.0, stockout_window_hours=168)
    assert m2["realized_stockout_freq"] == 0.0


def test_realized_stockout_freq_absent_without_entities():
    reset_realized_metrics()
    m = compute_realized_metrics(_centers(0, 0, 0), {}, None, tick_number=1, simulated_hour=0.0)
    assert "realized_stockout_freq" not in m  # no per-entity stock -> don't fabricate


def test_realized_stockout_freq_resets_on_new_run():
    reset_realized_metrics()
    out = _stockout_snapshot(("H1", "O+", 0))
    healthy = _stockout_snapshot(("H1", "O+", 50))
    compute_realized_metrics([], {}, out, tick_number=5, simulated_hour=30.0)
    # backward tick = new run: stockout history cleared, healthy entity not flagged
    m = compute_realized_metrics([], {}, healthy, tick_number=1, simulated_hour=0.0)
    assert m["realized_stockout_freq"] == 0.0


def test_realized_stockout_freq_respects_threshold():
    reset_realized_metrics()
    snap = _stockout_snapshot(("H1", "O+", 1.0))
    strict = compute_realized_metrics([], {}, snap, tick_number=1, simulated_hour=0.0, stockout_threshold=0.0)
    assert strict["realized_stockout_freq"] == 0.0  # 1.0 > 0 -> not out
    reset_realized_metrics()
    lenient = compute_realized_metrics([], {}, snap, tick_number=1, simulated_hour=0.0, stockout_threshold=1.0)
    assert lenient["realized_stockout_freq"] == 1.0  # 1.0 <= 1.0 -> out


def test_shrunk_window_tracks_recent_usage_not_lifetime():
    samples = [(10.0, 0.0), (10.0, 6.0), (40.0, 12.0)]
    # default (true 30d) window -> lifetime average
    reset_forecast_history()
    rows_default = None
    for tick, (usage, hour) in enumerate(samples, start=1):
        rows_default = [_supply_row(usage, 30.0)]
        enrich_with_rolling_features(rows_default, tick_number=tick, simulated_hour=hour)
    assert rows_default[0]["usage_30d_mean"] == round((10 + 10 + 40) / 3, 3)  # 20.0 lifetime
    # shrunk 30d window (8h) at hour 12 -> only hours 6 & 12 -> tracks the recent surge
    reset_forecast_history()
    rows_recent = None
    for tick, (usage, hour) in enumerate(samples, start=1):
        rows_recent = [_supply_row(usage, 30.0)]
        enrich_with_rolling_features(
            rows_recent, tick_number=tick, simulated_hour=hour, window_7d_h=6, window_30d_h=8
        )
    assert rows_recent[0]["usage_30d_mean"] == round((10 + 40) / 2, 3)  # 25.0 recent