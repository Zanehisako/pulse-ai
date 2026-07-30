"""
ML-enhanced simulation bridge.

Pre-fills simulation scenario parameters from ML model predictions,
adds a point-query mode for short single-component simulations,
and wires ML demand forecasts into the simulation's forecast noise.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Config key for this module in simulation_tool.json
CONFIG_KEY = "ml_bridge"


def _bridge_config() -> dict[str, Any]:
    from ml.core.simulation_tool import _simulation_config

    return dict(_simulation_config().get(CONFIG_KEY) or {})


def _predict_model(model_id: str, features: dict[str, Any]) -> dict[str, Any] | None:
    """Run a single ML model prediction, returning None on failure."""
    try:
        from ml.core.feature_store import resolve_prediction_features
        from ml.core.prediction import run_prediction
        from ml.core.startup import get_registry

        registry = get_registry()
        rt = next(
            (m for m in registry.list_models() if m.model_id == model_id), None
        )
        if rt is None or rt.status != "loaded":
            return None
        resolved = resolve_prediction_features(rt, features)
        return run_prediction(rt, resolved.features)
    except Exception as exc:
        logger.debug("ml_bridge prediction failed for %s: %s", model_id, exc)
        return None


def prefill_scenario_params(
    query: str,
    base_params: dict[str, Any],
    *,
    blood_type: str | None = None,
    component_type: str | None = None,
) -> dict[str, Any]:
    """
    Enhance simulation scenario params with ML model predictions.

    Calls demand/supply forecast models to get realistic current-state
    estimates, then adjusts simulation parameters accordingly.
    """
    config = _bridge_config()
    if not config.get("prefill_enabled", True):
        return {}

    overrides: dict[str, Any] = {}
    features: dict[str, Any] = {}
    if blood_type:
        features["blood_type"] = blood_type
    if component_type:
        features["component_type"] = component_type

    # Get demand forecast to calibrate demand_surge_factor
    demand_model = config.get("demand_model", "component_demand_quantile_forecast_model")
    demand_result = _predict_model(demand_model, features)
    if demand_result and "q50" in demand_result:
        predicted_demand = float(demand_result["q50"])
        baseline_demand = float(base_params.get("avg_units_per_order", 4.23))
        if baseline_demand > 0 and predicted_demand > 0:
            surge = predicted_demand / baseline_demand
            overrides["demand_surge_factor"] = round(min(max(surge, 0.2), 4.0), 3)
            overrides["_ml_demand_q50"] = round(predicted_demand, 2)

    # Get supply forecast to calibrate donor_show_factor
    supply_model = config.get("supply_model", "component_supply_forecast_model")
    supply_result = _predict_model(supply_model, features)
    if supply_result and "q50" in supply_result:
        predicted_supply = float(supply_result["q50"])
        baseline_supply = float(base_params.get("avg_units_per_order", 4.23))
        if baseline_supply > 0 and predicted_supply > 0:
            supply_ratio = predicted_supply / baseline_supply
            overrides["donor_show_factor"] = round(min(max(supply_ratio, 0.1), 2.5), 3)
            overrides["_ml_supply_q50"] = round(predicted_supply, 2)

    # Get stockout risk to calibrate demand_forecast_noise
    stockout_model = config.get("stockout_model", "stockout_time_to_event_hazard_model")
    stockout_features = dict(features)
    stockout_result = _predict_model(stockout_model, stockout_features)
    if stockout_result and "probability" in stockout_result:
        p_stockout = float(stockout_result["probability"])
        # Higher stockout risk → lower forecast noise (more conservative)
        noise = max(0.02, 0.1 * (1 - p_stockout))
        overrides["demand_forecast_noise"] = round(noise, 4)
        overrides["_ml_stockout_probability"] = round(p_stockout, 4)

    return overrides


def point_query_params(
    query: str,
    *,
    blood_type: str | None = None,
    component_type: str | None = None,
    current_inventory: float | None = None,
    units_used: float | None = None,
) -> dict[str, Any]:
    """
    Build simulation params for a short point-query (24-48h).

    Uses ML model predictions to set realistic initial conditions
    for a focused single-component simulation.
    """
    config = _bridge_config()
    hours = int(config.get("point_query_hours", 48))

    params: dict[str, Any] = {
        "sim_hours": hours,
        "name": f"Point query: {(query or '')[:50]}",
        "description": "ML-calibrated short-horizon point query.",
        "base_scenario_key": "baseline",
        "recommended_strategy_key": "baseline",
    }

    features: dict[str, Any] = {}
    if blood_type:
        features["blood_type"] = blood_type
    if component_type:
        features["component_type"] = component_type
    if current_inventory is not None:
        features["current_inventory"] = current_inventory
    if units_used is not None:
        features["units_used"] = units_used

    # Set initial inventory from provided value or ML prediction
    if current_inventory is not None:
        # Convert absolute units to days-of-supply
        daily_usage = (units_used or 4.23) * 24
        if daily_usage > 0:
            params["initial_inventory_days"] = round(
                min(max(current_inventory / daily_usage, 0.5), 14.0), 2
            )

    # Set demand intensity from provided usage or ML forecast
    if units_used is not None:
        baseline_rate = 1.22  # baseline demand_rate_h
        baseline_units = 4.23
        # More usage → faster demand arrival
        surge = units_used / baseline_units
        params["demand_surge_factor"] = round(min(max(surge, 0.2), 4.0), 3)

    # Get ML forecast noise calibration
    ml_overrides = prefill_scenario_params(
        query, {"avg_units_per_order": 4.23}, blood_type=blood_type, component_type=component_type
    )
    if "demand_forecast_noise" in ml_overrides:
        params["demand_forecast_noise"] = ml_overrides["demand_forecast_noise"]

    return params


def is_point_query(query: str) -> bool:
    """Detect if a query is asking for a short-horizon point prediction via simulation."""
    config = _bridge_config()
    terms = config.get("point_query_terms", [])
    if not terms:
        return False
    lowered = (query or "").lower()
    return any(term in lowered for term in terms)
