from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.core.prediction import _apply_prediction_output_contract
from ml.core.feature_extraction import route_models
from ml.core.registry import ModelRuntime


CONFIG_DIR = BACKEND_ROOT / "ml" / "config"
ARCHIVED_MODEL_IDS = {
    "ideal_donor_classifier",
    "days_until_stockout_regression",
    "hospital_shortage_predictor",
    "blood_stock_forecast",
    "short_horizon_probability",
    "medium_horizon_probability",
    "hospital_long_term_shortage_prophet",
}
NEW_MODEL_IDS = {
    "donor_next_donation_hazard_model",
    "donor_contact_response_model",
    "donor_contact_propensity_model",
    "donor_priority_policy_model",
    "stockout_time_to_event_hazard_model",
    "component_demand_quantile_forecast_model",
    "component_supply_forecast_model",
    "component_expiry_waste_model",
    "component_inventory_risk_simulator",
    "ideal_donor_probability_model",
}


def _load_json(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _configured_model_ids(value) -> set[str]:
    if isinstance(value, dict):
        found = {
            str(value[key])
            for key in ("model_id", "regression_model_id")
            if isinstance(value.get(key), str)
        }
        for child in value.values():
            found.update(_configured_model_ids(child))
        return found
    if isinstance(value, list):
        found = set()
        for child in value:
            found.update(_configured_model_ids(child))
        return found
    return set()


def _runtime_from_config(row: dict) -> ModelRuntime:
    return ModelRuntime(
        model_id=str(row["id"]),
        aliases=[],
        slug=str(row["id"]),
        file_path=str(row.get("file_path") or ""),
        description=str(row.get("description") or ""),
        feature_names=list(row.get("features") or []),
        examples=list(row.get("examples") or []),
        defaults=dict(row.get("defaults") or {}),
    )


def test_orchestrator_model_configured_models_have_examples():
    payload = _load_json("config.json")
    configured = {
        str(row["id"]): row
        for row in payload["models"]
        if isinstance(row, dict) and row.get("id")
    }

    assert configured
    for model_id, row in configured.items():
        examples = row.get("examples")
        assert str(row.get("description") or "").strip(), model_id
        assert isinstance(examples, list) and examples, model_id
        assert any(
            item.get("kind") == "good" for item in examples if isinstance(item, dict)
        ), model_id
        assert any(
            item.get("kind") == "bad" for item in examples if isinstance(item, dict)
        ), model_id


def test_donor_profile_queries_route_by_config_intent():
    payload = _load_json("config.json")
    candidates = [
        _runtime_from_config(row)
        for row in payload["models"]
        if isinstance(row, dict) and row.get("enabled")
    ]

    donate_now_ranked = route_models(
        "Can a 35 year old donor with BMI 24 donate blood now?", candidates, top_k=3
    )
    ideal_ranked = route_models(
        "Is a  35 year old donor with BMI 24 an ideal donor?", candidates, top_k=3
    )

    assert donate_now_ranked[0].model_id == "donor_next_donation_hazard_model"
    assert ideal_ranked[0].model_id == "ideal_donor_probability_model"


def test_backend_routing_configs_do_not_reference_archived_models():
    for name in (
        "scheduled_predictions.json",
        "stockout_hybrid.json",
        "model_prediction_defaults.json",
        "config.json",
    ):
        configured_model_ids = _configured_model_ids(_load_json(name))
        for model_id in ARCHIVED_MODEL_IDS:
            assert model_id not in configured_model_ids, f"{name} still configures {model_id}"


def test_scheduled_predictions_route_to_new_shadow_models():
    payload = _load_json("scheduled_predictions.json")
    configured = {row["model_id"] for row in payload["jobs"]}
    assert "component_inventory_risk_simulator" in configured
    assert "stockout_time_to_event_hazard_model" in configured
    assert "donor_priority_policy_model" in configured
    assert "donor_contact_response_model" in configured
    assert all(row["model_version"] == "challenger" for row in payload["jobs"])


def test_dashboard_donor_snapshot_uses_configured_priority_model_stats():
    payload = _load_json("scheduled_predictions.json")
    ideal_config = payload["dashboard_snapshots"]["ideal"]
    candidates = {tuple(row.items()) for row in ideal_config["stats_candidates"]}
    assert tuple({"model_id": "donor_priority_policy_model"}.items()) in candidates
    assert tuple({"model_id": "donor_priority_policy_model_shadow"}.items()) in candidates
    assert ideal_config["fallback_filter"] == {"model_id__icontains": "donor_priority"}
    aliases = ideal_config["field_aliases"]
    assert "age" not in aliases
    assert "eligible_to_donate" in aliases
    assert "readiness_score" in aliases
    assert set(ideal_config["binary_features"]) >= {
        "eligible_to_donate",
        "is_regular_donor",
        "is_rare_type",
    }


def test_model_prediction_defaults_cover_new_runtime_models():
    payload = _load_json("model_prediction_defaults.json")
    configured = set(payload["models"])
    assert NEW_MODEL_IDS <= configured
    assert payload["models"]["donor_uplift_model"]["defaults"]["enabled"] is False


def test_sota_model_stats_runtime_config_is_path_and_marker_driven():
    payload = _load_json("prediction_runtime.json")
    discovery_config = payload["model_discovery"]
    stats_config = payload["sota_model_stats"]
    assert discovery_config["models_dir_env"] == "PIOS_MODELS_DIR"
    assert discovery_config["default_models_dir"] == "../ml-backend/models"
    assert stats_config["enabled"] is True
    assert stats_config["models_dir_env"] == "PIOS_MODELS_DIR"
    assert stats_config["default_models_dir"] == "../ml-backend/models"
    assert stats_config["catalog_path_env"] == "PIOS_SOTA_CATALOG_PATH"
    assert stats_config["datasets_dir_env"] == "PIOS_SOTA_DATASETS_DIR"
    assert "donor_priority_policy_model" not in json.dumps(stats_config)
    assert set(stats_config["donor_feature_markers"]) >= {
        "eligible_to_donate",
        "is_rare_type",
        "readiness_score",
    }


def test_db_tool_config_covers_current_stock_status_queries():
    payload = _load_json("config.json")
    db_tool = next(
        tool
        for tool in payload["external_tools"]
        if tool.get("adapter") == "db_query"
    )
    example = next(
        row
        for row in db_tool["examples"]
        if row.get("user_query") == "What is the stock status?"
    )

    assert db_tool["enabled"] is True
    assert "stock/status questions" in db_tool["input_hints"]
    assert example["arguments"] == {
        "table": "hospitals",
        "aggregate": "list",
        "fields": ["date", "hospital_id", "blood_type", "stock_end", "critical_stock"],
        "latest_only": True,
        "limit": 10,
    }


def test_prediction_output_contract_clips_multioutput_pyfunc_fields():
    runtime = ModelRuntime(
        model_id="component_inventory_risk_simulator",
        aliases=[],
        slug="component_inventory_risk_simulator",
        file_path="models:/component_inventory_risk_simulator@challenger",
        description="",
        defaults={},
    )

    result = _apply_prediction_output_contract(
        runtime,
        {
            "p_stockout_0_7d": 1.4,
            "p_stockout_0_30d": -0.2,
            "expected_days_until_stockout": -3,
            "median_status": "not_reached",
        },
    )

    assert result["p_stockout_0_7d"] == 1.0
    assert result["p_stockout_0_7d_clipped"] is True
    assert result["p_stockout_0_30d"] == 0.0
    assert result["expected_days_until_stockout"] == 0.0
    assert result["median_status"] == "not_reached"
