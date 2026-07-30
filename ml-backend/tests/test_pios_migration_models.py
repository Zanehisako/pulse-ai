from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_TRAINING_SCRIPTS = Path(__file__).resolve().parents[1] / "training_scripts"
_ROBOT_DIR = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _ROBOT_DIR / "config"

if str(_TRAINING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TRAINING_SCRIPTS))
if str(_ROBOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROBOT_DIR))

from model_training_utils import (
    _feature_columns_for_dataset,
    _is_future_or_label_column,
    TaskDataset,
)
from model_architecture_config import (
    load_tool_registry_policy,
    validate_tool_registry_config,
    validate_tool_registry_configs,
)


def _load_json(filename: str) -> dict:
    return json.loads((_CONFIG_DIR / filename).read_text(encoding="utf-8"))


class TestWastageLeakageGuard:
    def test_wastage_blocked_as_feature_when_target(self, tmp_path: Path):
        frame = pd.DataFrame(
            {
                "hospital_id": ["H001"],
                "blood_type": ["O+"],
                "component_type": ["RBC"],
                "event_timestamp": pd.to_datetime(["2026-01-01"]),
                "wastage_next_1d": [5.0],
                "wastage_next_7d": [30.0],
                "current_inventory": [50.0],
                "units_used": [8.0],
            }
        )
        dataset = TaskDataset(
            task_name="test_expiry_waste",
            model_name="component_expiry_waste_model",
            experiment_name="test",
            task_type="regression",
            frame=frame,
            target_column="wastage_next_1d",
            timestamp_column="event_timestamp",
            feature_service="test_service",
            entity_keys=("hospital_id", "blood_type", "component_type"),
            label_file=tmp_path / "labels.parquet",
            feature_file=tmp_path / "features.parquet",
            description="test",
            hard_mask_builder=lambda df: pd.Series(False, index=df.index),
        )
        features = _feature_columns_for_dataset(frame, dataset)
        assert "wastage_next_1d" not in features
        assert "wastage_next_7d" not in features
        assert "current_inventory" in features

    def test_wastage_plain_available_as_feature_for_inventory_models(self, tmp_path: Path):
        frame = pd.DataFrame(
            {
                "hospital_id": ["H001"],
                "blood_type": ["O+"],
                "component_type": ["RBC"],
                "event_timestamp": pd.to_datetime(["2026-01-01"]),
                "wastage": [3.0],
                "wastage_next_1d": [5.0],
                "current_inventory": [50.0],
                "stockout_within_0_7d": [0],
            }
        )
        dataset = TaskDataset(
            task_name="test_stockout",
            model_name="test_model",
            experiment_name="test",
            task_type="classification",
            frame=frame,
            target_column="stockout_within_0_7d",
            timestamp_column="event_timestamp",
            feature_service="test_service",
            entity_keys=("hospital_id", "blood_type", "component_type"),
            label_file=tmp_path / "labels.parquet",
            feature_file=tmp_path / "features.parquet",
            description="test",
            hard_mask_builder=lambda df: pd.Series(False, index=df.index),
        )
        features = _feature_columns_for_dataset(frame, dataset)
        assert "wastage" in features
        assert "wastage_next_1d" not in features


class TestOperationalOutreachPriority:
    def test_outreach_priority_inputs_are_point_in_time_safe(self):
        inputs = [
            "emergency_outreach_priority",
            "digital_contactability_score",
            "eligible_to_donate",
        ]
        for col in inputs:
            assert not _is_future_or_label_column(col), f"{col} should be point-in-time safe"

    def test_operational_outreach_priority_in_donor_features(self):
        donor_cfg = _load_json("donor_contact_response_model.json")
        seq_features = donor_cfg["feature_engineering"].get("sequence_features", [])
        static_features = donor_cfg["feature_engineering"].get("static_features", [])
        all_features = seq_features + static_features
        assert "operational_outreach_priority" in all_features


class TestSimulatorConfig:
    def test_simulator_config_required_fields_present(self):
        cfg = _load_json("component_inventory_risk_simulator.json")
        assert cfg["enabled"] is True
        assert cfg["model_type"] == "composite_pyfunc"
        sub = cfg["sub_models"]
        assert "demand" in sub
        assert "supply" in sub
        assert "waste" in sub
        assert sub["demand"]["registered_model_name"] == "component_demand_quantile_forecast_model"
        assert sub["supply"]["registered_model_name"] == "component_supply_forecast_model"
        assert sub["waste"]["registered_model_name"] == "component_expiry_waste_model"
        sim = cfg["simulation"]
        assert sim["n_paths"] >= 100
        assert len(sim["horizons_days"]) >= 2
        assert "stock_column" in sim
        contract = cfg["output_contract"]
        assert "p_stockout_0_7d" in contract
        assert "p_stockout_0_30d" in contract
        assert "p_stockout_0_90d" in contract
        assert "p_stockout_0_180d" in contract
        assert "expected_days_until_stockout" in contract
        assert "median_status" in contract

    def test_simulator_child_model_aliases_are_champion(self):
        cfg = _load_json("component_inventory_risk_simulator.json")
        for key in ("demand", "supply", "waste"):
            assert cfg["sub_models"][key]["alias"] == "champion"


class TestReportOnlyRouting:
    def test_new_models_have_report_only_gates(self):
        gates = _load_json("thesis_readiness_gates.json")
        new_models = [
            "donor_next_donation_hazard_model",
            "donor_contact_response_model",
            "donor_priority_policy_model",
            "stockout_time_to_event_hazard_model",
            "component_demand_quantile_forecast_model",
            "component_supply_forecast_model",
            "component_expiry_waste_model",
            "component_inventory_risk_simulator",
        ]
        for model in new_models:
            model_gates = gates["models"].get(model)
            assert model_gates is not None, f"Missing thesis gates for {model}"
            assert (
                model_gates.get("shadow_deployment") is True
            ), f"{model} should be shadow_deployment"
            assert (
                "disabled_until_thresholds_defined"
                in str(model_gates.get("production_routing", ""))
            ), f"{model} should have disabled production routing"

    def test_report_only_models_not_in_scheduled_predictions(self):
        gates = _load_json("thesis_readiness_gates.json")
        new_models = {
            "component_expiry_waste_model",
            "component_inventory_risk_simulator",
            "donor_contact_response_model",
        }
        for model in new_models:
            model_gates = gates["models"].get(model)
            assert model_gates is not None, f"Missing gates for {model}"
            assert model_gates.get("production_routing") == "disabled_until_thresholds_defined"
            assert model_gates.get("shadow_deployment") is True


class TestExpiryWasteProxyMode:
    def test_expiry_waste_config_has_fallback_targets(self):
        cfg = _load_json("component_expiry_waste_model.json")
        target = cfg["target"]
        assert target["column"] == "wastage_next_1d"
        fallback = target.get("fallback_targets", [])
        assert len(fallback) > 0
        assert "wastage" in fallback

    def test_expiry_waste_has_data_availability_note(self):
        cfg = _load_json("component_expiry_waste_model.json")
        note = cfg.get("data_availability_note", "")
        assert "shifted labels" in note.lower() or "fallback" in note.lower()


class TestToolRegistryConfig:
    MODEL_CONFIG_FILES = [
        "donor_next_donation_hazard_model.json",
        "donor_contact_propensity_model.json",
        "donor_contact_response_model.json",
        "donor_priority_policy_model.json",
        "donor_uplift_model.json",
        "stockout_time_to_event_hazard_model.json",
        "component_demand_quantile_forecast_model.json",
        "component_supply_forecast_model.json",
        "component_expiry_waste_model.json",
        "component_inventory_risk_simulator.json",
    ]

    def test_new_model_configs_validate_as_tools(self):
        policy = load_tool_registry_policy(_CONFIG_DIR)
        configs = [_load_json(filename) for filename in self.MODEL_CONFIG_FILES]
        validated = validate_tool_registry_configs(configs, policy=policy)
        assert {row["id"] for row in validated} == {
            "donor_next_donation_hazard_model",
            "donor_contact_propensity_model",
            "donor_contact_response_model",
            "donor_priority_policy_model",
            "donor_uplift_model",
            "stockout_time_to_event_hazard_model",
            "component_demand_quantile_forecast_model",
            "component_supply_forecast_model",
            "component_expiry_waste_model",
            "component_inventory_risk_simulator",
        }

    def test_tool_config_missing_required_field_fails_clearly(self):
        policy = load_tool_registry_policy(_CONFIG_DIR)
        cfg = _load_json("donor_contact_response_model.json")
        cfg.pop("timeout_seconds")
        with pytest.raises(ValueError, match="missing required field timeout_seconds"):
            validate_tool_registry_config(cfg, policy=policy)

    def test_tool_config_duplicate_ids_fail(self):
        policy = load_tool_registry_policy(_CONFIG_DIR)
        cfg = _load_json("donor_contact_response_model.json")
        with pytest.raises(ValueError, match="unique"):
            validate_tool_registry_configs([cfg, dict(cfg)], policy=policy)
