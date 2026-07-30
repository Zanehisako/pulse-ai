from simulation_studio.app.twin_forecast import (
    _apply_panel_value_mode,
    _chart_point,
    _horizon_days,
    _resolve_value_mode,
    build_tick_context,
)

_VALIDATION_CONFIG = {
    "chart": {"label_template": "h{simulated_hour}", "supply_source": "snapshot_stock_sum"},
    "prediction_value_mode": "demand",
    "validation": {
        "forecast": {
            "predicted": {"label": "Forecast demand", "aggregate": "sum", "value_mode": "demand"},
            "actual": {"label": "Realized demand", "source": "realized_demand", "aggregate": "sum"},
        },
        "stockout_risk": {
            "predicted": {"label": "Predicted risk", "aggregate": "mean", "value_mode": "raw"},
            "actual": {"label": "Realized shortage rate", "source": "realized_shortage_rate", "aggregate": "mean"},
        },
    },
}


def test_build_tick_context_computes_calendar_features():
    ctx = build_tick_context(tick_number=3, simulated_hour=48.0)
    assert ctx["tick_number"] == 3
    assert ctx["simulated_hour"] == 48.0
    overrides = ctx.get("feature_overrides") or {}
    assert overrides.get("month") == 3
    assert overrides.get("dow") == 3
    assert "T" in str(overrides.get("event_timestamp") or "")


def _stock_forecast_job():
    # Mirrors the shared dashboard_forecast_t7 job: scale daily demand to the
    # horizon, then project remaining stock (the step that pinned the panel to 0).
    return {
        "id": "dashboard_forecast_t7",
        "prediction_value": {"path": "0.demand_units"},
        "postprocess": [
            {"type": "multiply", "factor": 7},
            {"type": "subtract_from_attr", "attr": "current_stock_units", "min_value": 0},
        ],
    }


def test_demand_mode_drops_stock_projection_keeps_multiply():
    job = _stock_forecast_job()
    patched = _apply_panel_value_mode(job, {"prediction_value_mode": "demand"})
    types = [step["type"] for step in patched["postprocess"]]
    assert types == ["multiply"]  # subtract_from_attr removed -> demand, not leftover stock
    # original job is not mutated
    assert [s["type"] for s in job["postprocess"]] == ["multiply", "subtract_from_attr"]


def test_projected_stock_mode_is_unchanged():
    job = _stock_forecast_job()
    for cfg in ({}, {"prediction_value_mode": "projected_stock"}):
        patched = _apply_panel_value_mode(job, cfg)
        assert [s["type"] for s in patched["postprocess"]] == [
            "multiply",
            "subtract_from_attr",
        ]
        assert "usage_30d_mean" not in (patched.get("feature_mapping") or {})


def test_demand_mode_injects_rolling_usage_features():
    job = _stock_forecast_job()
    job["feature_mapping"] = {"current_inventory": {"attr": "current_stock_units"}}
    patched = _apply_panel_value_mode(job, {"prediction_value_mode": "demand"})
    fm = patched["feature_mapping"]
    # recent-usage trend features the model leans on are fed
    assert fm["usage_30d_mean"] == {"attr": "usage_30d_mean", "default": 0}
    assert {"usage_7d_mean", "inventory_7d_mean", "inventory_30d_mean"} <= set(fm)
    # existing mappings are preserved
    assert fm["current_inventory"] == {"attr": "current_stock_units"}


def test_raw_mode_injects_features_without_stripping_postprocess():
    job = {
        "dashboard_role": "stockout_risk",
        "postprocess": [{"type": "multiply", "factor": 7}],
        "feature_mapping": {},
    }
    patched = _apply_panel_value_mode(job, _VALIDATION_CONFIG)
    assert [s["type"] for s in patched["postprocess"]] == ["multiply"]  # not stripped
    assert "usage_30d_mean" in patched["feature_mapping"]  # rolling means still fed


def test_resolve_value_mode_per_role():
    assert _resolve_value_mode({"dashboard_role": "forecast"}, _VALIDATION_CONFIG) == "demand"
    assert _resolve_value_mode({"dashboard_role": "stockout_risk"}, _VALIDATION_CONFIG) == "raw"
    # role without a validation entry falls back to the top-level default
    assert _resolve_value_mode({"dashboard_role": "other"}, _VALIDATION_CONFIG) == "demand"


def test_chart_point_demand_role_predicted_vs_realized():
    cp = _chart_point(
        config=_VALIDATION_CONFIG,
        tick_context={"tick_number": 5, "simulated_hour": 30.0},
        predictions=[{"predicted_value": 33.0}, {"predicted_value": 47.0}],
        snapshot_payload={"blood_supplies": []},
        job={"dashboard_role": "forecast"},
        realized={"realized_demand": 120.0},
    )
    assert cp["predicted"] == 80.0  # sum aggregate
    assert cp["actual"] == 120.0  # realized_demand extractor
    assert cp["predicted_label"] == "Forecast demand"
    assert cp["actual_label"] == "Realized demand"
    # back-compat aliases for the frontend series arrays
    assert cp["demand"] == cp["predicted"] and cp["supply"] == cp["actual"]


def test_chart_point_applies_predicted_scale():
    # A `scale` on the predicted block calibrates a model trained at a different
    # population scale down to the simulation's regime (here 400 -> 100).
    config = {
        "validation": {
            "forecast": {
                "predicted": {"label": "Forecast demand", "aggregate": "sum", "scale": 0.25},
                "actual": {"label": "Realized demand", "source": "realized_demand", "aggregate": "sum"},
            }
        }
    }
    cp = _chart_point(
        config=config,
        tick_context={"tick_number": 5, "simulated_hour": 30.0},
        predictions=[{"predicted_value": 250.0}, {"predicted_value": 150.0}],
        snapshot_payload={},
        job={"dashboard_role": "forecast"},
        realized={"realized_demand": 100.0},
    )
    assert cp["predicted"] == 100.0  # 400 sum * 0.25 scale -> matches realized
    assert cp["actual"] == 100.0


def test_chart_point_scale_defaults_to_one():
    # No scale field -> predicted unchanged (other roles / legacy unaffected).
    cp = _chart_point(
        config=_VALIDATION_CONFIG,
        tick_context={"tick_number": 5, "simulated_hour": 30.0},
        predictions=[{"predicted_value": 33.0}, {"predicted_value": 47.0}],
        snapshot_payload={"blood_supplies": []},
        job={"dashboard_role": "forecast"},
        realized={"realized_demand": 120.0},
    )
    assert cp["predicted"] == 80.0


def test_chart_point_stockout_role_uses_mean_and_rate():
    cp = _chart_point(
        config=_VALIDATION_CONFIG,
        tick_context={"tick_number": 1, "simulated_hour": 6.0},
        predictions=[{"predicted_value": 0.4}, {"predicted_value": 0.6}],
        snapshot_payload={},
        job={"dashboard_role": "stockout_risk"},
        realized={"realized_shortage_rate": 0.2},
    )
    assert cp["predicted"] == 0.5  # mean aggregate
    assert cp["actual"] == 0.2


def test_horizon_days_parses_dashboard_horizon():
    assert _horizon_days({"dashboard_horizon": "t1"}) == 1
    assert _horizon_days({"dashboard_horizon": "t7"}) == 7
    assert _horizon_days({"dashboard_horizon": "T30"}) == 30  # case-insensitive
    # missing / malformed horizon -> 1 (no-op multiplier for non-forecast roles)
    assert _horizon_days({"dashboard_horizon": ""}) == 1
    assert _horizon_days({"dashboard_horizon": "weekly"}) == 1
    assert _horizon_days({}) == 1
    assert _horizon_days(None) == 1


_HORIZON_CONFIG = {
    "validation": {
        "forecast": {
            "predicted": {"label": "Forecast demand", "aggregate": "sum"},
            "actual": {
                "label": "Realized demand",
                "source": "realized_demand",
                "aggregate": "sum",
                "horizon_basis": True,
            },
        }
    }
}


def test_chart_point_horizon_basis_lifts_realized_to_cumulative():
    # T7 predicts daily demand × 7 (cumulative); horizon_basis lifts the per-day
    # realized rate onto the same basis (120/day -> 840 over 7 days) so the two
    # series share units instead of being 7× apart.
    cp = _chart_point(
        config=_HORIZON_CONFIG,
        tick_context={"tick_number": 5, "simulated_hour": 30.0},
        predictions=[{"predicted_value": 560.0}],
        snapshot_payload={},
        job={"dashboard_role": "forecast", "dashboard_horizon": "t7"},
        realized={"realized_demand": 120.0},
    )
    assert cp["predicted"] == 560.0
    assert cp["actual"] == 840.0  # 120/day * 7 days


def test_chart_point_horizon_basis_noop_for_t1():
    cp = _chart_point(
        config=_HORIZON_CONFIG,
        tick_context={"tick_number": 5, "simulated_hour": 30.0},
        predictions=[{"predicted_value": 80.0}],
        snapshot_payload={},
        job={"dashboard_role": "forecast", "dashboard_horizon": "t1"},
        realized={"realized_demand": 120.0},
    )
    assert cp["actual"] == 120.0  # horizon 1 -> unchanged


def test_chart_point_horizon_basis_ignored_without_flag():
    # The default validation config (no horizon_basis) leaves realized per-day.
    cp = _chart_point(
        config=_VALIDATION_CONFIG,
        tick_context={"tick_number": 5, "simulated_hour": 30.0},
        predictions=[{"predicted_value": 560.0}],
        snapshot_payload={"blood_supplies": []},
        job={"dashboard_role": "forecast", "dashboard_horizon": "t7"},
        realized={"realized_demand": 120.0},
    )
    assert cp["actual"] == 120.0  # not lifted -> opt-in only


def test_chart_point_unknown_role_falls_back_to_stock_sum():
    cp = _chart_point(
        config={"chart": {"supply_source": "snapshot_stock_sum", "demand_aggregate": "sum"}},
        tick_context={"tick_number": 1, "simulated_hour": 6.0},
        predictions=[{"predicted_value": 10.0}],
        snapshot_payload={"blood_supplies": [{"current_stock_units": 12}, {"current_stock_units": 8}]},
        job={"dashboard_role": "forecast"},
        realized=None,
    )
    assert cp["predicted"] == 10.0
    assert cp["actual"] == 20.0  # legacy snapshot stock-sum fallback
