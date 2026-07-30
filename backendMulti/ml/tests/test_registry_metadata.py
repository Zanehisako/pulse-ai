from __future__ import annotations

import pickle
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from ml.core.registry import ModelRegistry


class _DummyEstimator:
    feature_names_in_ = [
        "temp_c",
        "rain_mm",
        "holiday",
        "trauma_cases",
        "scheduled_surgeries",
        "stock_end",
    ]

    def predict(self, _x):
        return [1]


def test_registry_applies_runtime_metadata_defaults(tmp_path: Path):
    model_path = tmp_path / "component_demand_quantile_forecast_model.pkl"
    model_path.write_bytes(pickle.dumps(_DummyEstimator()))

    registry = ModelRegistry(
        models_dir=tmp_path,
        config_path=tmp_path / "unused.json",
        config_loader=lambda: [
            {
                "id": "component_demand_quantile_forecast_model",
                "file_path": str(model_path),
                "type": "xgboost",
                "description": "",
                "features": [],
                "examples": [],
                "defaults": {},
            }
        ],
    )

    changed = registry.refresh(force=True)
    runtime = registry.get("component_demand_quantile_forecast_model")

    assert changed is True
    assert runtime is not None
    assert runtime.status == "loaded"
    assert "component demand" in runtime.description.lower()
    assert "forecast" in runtime.examples[0]["user_query"].lower()
    assert runtime.defaults["prediction_source"] == "feast_online"
    assert runtime.defaults["feast"]["feature_service"] == "quebec_arima_sarima_forecast_service"


def test_registry_applies_inventory_simulator_defaults(tmp_path: Path):
    registry = ModelRegistry(
        models_dir=tmp_path,
        config_path=tmp_path / "unused.json",
        config_loader=lambda: [
            {
                "id": "component_inventory_risk_simulator",
                "file_path": "models:/component_inventory_risk_simulator@challenger",
                "type": "mlflow",
                "description": "",
                "features": [],
                "examples": [],
                "defaults": {},
            }
        ],
    )

    changed = registry.refresh(force=True)
    runtime = registry.get("component_inventory_risk_simulator")

    assert changed is True
    assert runtime is not None
    assert runtime.status == "loaded"
    assert "inventory simulation" in runtime.description.lower()
    assert runtime.defaults["prediction_source"] == "direct_input"
    assert runtime.defaults["output"]["name"] == "inventory_risk"
    assert runtime.defaults["output_contract"]["p_stockout_0_7d"]["clip_max"] == 1.0
    assert runtime.defaults["horizon_routing"]["timeframes"] == ["short", "medium", "long"]


def test_registry_applies_configured_donor_priority_defaults(tmp_path: Path):
    registry = ModelRegistry(
        models_dir=tmp_path,
        config_path=tmp_path / "unused.json",
        config_loader=lambda: [
            {
                "id": "donor_priority_policy_model",
                "file_path": "models:/donor_priority_policy_model@challenger",
                "type": "mlflow",
                "description": "",
                "features": [],
                "examples": [],
                "defaults": {},
            }
        ],
    )

    changed = registry.refresh(force=True)
    runtime = registry.get("donor_priority_policy_model")

    assert changed is True
    assert runtime is not None
    assert runtime.status == "loaded"
    assert "expected-utility" in runtime.description.lower()
    assert runtime.defaults["output"]["name"] == "donor_priority_score"
    assert runtime.defaults["output"]["task_type"] == "policy"
    assert runtime.defaults["output"]["clip_min"] == 0.0
    assert runtime.defaults["output"]["clip_max"] == 1.0
