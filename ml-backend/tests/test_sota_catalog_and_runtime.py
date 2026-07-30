from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ML_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(ML_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_BACKEND_ROOT))

from pios_ml_backend.sota import (  # noqa: E402
    SotaCatalogError,
    SotaNotebookModel,
    load_notebook_bundle,
    load_sota_catalog,
    validate_sota_catalog,
)
from pios_ml_backend.sota.notebook_runtime import NeuralPreprocessor  # noqa: E402
from pios_ml_backend.sota.notebook_runtime import _normalize_neural_state_dict  # noqa: E402


class _Predictor:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame: pd.DataFrame):
        return np.full(len(frame), self.value, dtype=float)

    def predict_proba(self, frame: pd.DataFrame):
        positive = np.full(len(frame), self.value, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def _catalog(tmp_path: Path, **model_overrides):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"artifact")
    model = {
        "id": "model_a",
        "enabled": True,
        "registered_model_name": "registered_model_a",
        "artifact_filename": artifact.name,
        "task_type": "classification",
        "features": ["x"],
        "outputs": {"primary": "probability"},
        **model_overrides,
    }
    return {
        "default_models_dir": str(tmp_path),
        "models": [model],
    }


def test_sota_catalog_accepts_valid_configured_artifacts(tmp_path: Path):
    payload = _catalog(tmp_path)

    assert validate_sota_catalog(payload, require_artifacts=True) is payload


def test_sota_catalog_rejects_duplicate_model_ids(tmp_path: Path):
    payload = _catalog(tmp_path)
    payload["models"].append(dict(payload["models"][0]))

    with pytest.raises(SotaCatalogError, match="Duplicate"):
        validate_sota_catalog(payload, require_artifacts=True)


def test_sota_catalog_rejects_duplicate_registered_names(tmp_path: Path):
    payload = _catalog(tmp_path)
    second = dict(payload["models"][0], id="model_b")
    payload["models"].append(second)

    with pytest.raises(SotaCatalogError, match="registered_model_name"):
        validate_sota_catalog(payload, require_artifacts=True)


def test_sota_catalog_resolves_default_models_dir_relative_to_catalog_file(tmp_path: Path):
    config_dir = tmp_path / "config"
    models_dir = tmp_path / "models"
    config_dir.mkdir()
    models_dir.mkdir()
    (models_dir / "model.joblib").write_bytes(b"artifact")
    catalog_path = config_dir / "sota_model_catalog.json"
    payload = _catalog(models_dir)
    payload["default_models_dir"] = "../models"
    catalog_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_sota_catalog(catalog_path, require_artifacts=True)

    assert loaded["models"][0]["registered_model_name"] == "registered_model_a"


def test_sota_catalog_rejects_missing_artifact(tmp_path: Path):
    payload = _catalog(tmp_path, artifact_filename="missing.joblib")

    with pytest.raises(SotaCatalogError, match="does not exist"):
        validate_sota_catalog(payload, require_artifacts=True)


def test_sota_catalog_allows_disabled_missing_artifact(tmp_path: Path):
    payload = _catalog(tmp_path, enabled=False, artifact_filename="missing.joblib")

    assert validate_sota_catalog(payload, require_artifacts=True) is payload


def test_sota_catalog_rejects_invalid_outputs(tmp_path: Path):
    payload = _catalog(tmp_path, outputs={})

    with pytest.raises(SotaCatalogError, match="outputs"):
        validate_sota_catalog(payload, require_artifacts=True)


def test_sota_catalog_rejects_missing_artifact_filename(tmp_path: Path):
    payload = _catalog(tmp_path)
    del payload["models"][0]["artifact_filename"]

    with pytest.raises(SotaCatalogError, match="artifact_filename"):
        validate_sota_catalog(payload, require_artifacts=True)


def test_sota_catalog_rejects_unsupported_task_type(tmp_path: Path):
    payload = _catalog(tmp_path, task_type="custom")

    with pytest.raises(SotaCatalogError, match="task_type"):
        validate_sota_catalog(payload, require_artifacts=True)


def test_notebook_pickle_compatibility_loader_restores_main_preprocessor(tmp_path: Path):
    main_module = sys.modules["__main__"]
    legacy_class = type("NeuralPreprocessor", (), {})
    legacy_class.__module__ = "__main__"
    previous = getattr(main_module, "NeuralPreprocessor", None)
    had_previous = hasattr(main_module, "NeuralPreprocessor")
    setattr(main_module, "NeuralPreprocessor", legacy_class)
    try:
        instance = legacy_class()
        instance.feature_cols = ["x"]
        instance.extra_state = "loaded"
        path = tmp_path / "legacy.joblib"
        joblib.dump({"neural_preprocessor": instance}, path)
    finally:
        if had_previous:
            setattr(main_module, "NeuralPreprocessor", previous)
        else:
            delattr(main_module, "NeuralPreprocessor")

    loaded = load_notebook_bundle(path)

    assert isinstance(loaded["neural_preprocessor"], NeuralPreprocessor)
    assert loaded["neural_preprocessor"].extra_state == "loaded"


def test_neural_state_dict_normalization_strips_compiled_prefix():
    state = {
        "_orig_mod.cls": "cls",
        "_orig_mod.blocks.0.norm1.weight": "weight",
    }

    assert _normalize_neural_state_dict(state) == {
        "cls": "cls",
        "blocks.0.norm1.weight": "weight",
    }


def test_quantile_output_normalization_orders_and_clips_outputs():
    model = SotaNotebookModel(
        {"task_type": "quantile", "features": ["x"], "outputs": {"primary": "demand_units"}}
    )
    model._bundle = {
        "winner": "tabular_gbdt",
        "feature_cols": ["x"],
        "tabular_columns": ["x"],
        "tabular_models": {
            "q10": _Predictor(8.0),
            "q50": _Predictor(-2.0),
            "q90": _Predictor(3.0),
        },
    }

    result = model.predict(None, [{"x": 1.0}]).iloc[0].to_dict()

    assert result["prediction"] == 0.0
    assert result["demand_units"] == 0.0
    assert result["q90"] == 8.0


def test_regression_output_is_clipped_to_non_negative():
    model = SotaNotebookModel(
        {"task_type": "regression", "features": ["x"], "outputs": {"primary": "wastage_units"}}
    )
    model._bundle = {
        "winner": "tabular_gbdt",
        "feature_cols": ["x"],
        "tabular_columns": ["x"],
        "tabular_model": _Predictor(-3.0),
    }

    result = model.predict(None, [{"x": 1.0}]).iloc[0].to_dict()

    assert result["prediction"] == 0.0
    assert result["wastage_units"] == 0.0


def test_regression_score_output_uses_configured_range():
    model = SotaNotebookModel(
        {
            "task_type": "regression",
            "features": ["x"],
            "outputs": {"primary": "ideal_donor_probability"},
            "score_output_schema": {"range": [0.0, 1.0]},
        }
    )
    model._bundle = {
        "winner": "tabular_gbdt",
        "feature_cols": ["x"],
        "tabular_columns": ["x"],
        "tabular_model": _Predictor(1.4),
    }

    result = model.predict(None, [{"x": 1.0}]).iloc[0].to_dict()

    assert result["prediction"] == 1.0
    assert result["ideal_donor_probability"] == 1.0


def test_classification_probability_threshold_output():
    model = SotaNotebookModel(
        {
            "task_type": "classification",
            "features": ["x"],
            "threshold": 0.7,
            "outputs": {"primary": "p_stockout"},
        }
    )
    model._bundle = {
        "winner": "tabular_gbdt",
        "feature_cols": ["x"],
        "tabular_columns": ["x"],
        "tabular_model": _Predictor(0.61),
    }

    result = model.predict(None, [{"x": 1.0}]).iloc[0].to_dict()

    assert result["probability"] == pytest.approx(0.61)
    assert result["p_stockout"] == pytest.approx(0.61)
    assert result["prediction"] == 0


def test_policy_and_simulator_outputs_are_json_stable():
    policy = SotaNotebookModel(
        {
            "task_type": "policy",
            "features": ["contact_response_score", "eligibility_score"],
            "policy_formula": {"contact_response_score": 0.7, "eligibility_score": 0.3},
            "action_thresholds": {"urgent_contact": 0.8},
        }
    )
    policy._bundle = {"type": "policy"}
    policy_result = policy.predict(
        None,
        [
            {"contact_response_score": 0.1, "eligibility_score": 0.1},
            {"contact_response_score": 0.9, "eligibility_score": 0.9},
        ],
    )

    simulator = SotaNotebookModel(
        {
            "task_type": "simulator",
            "features": ["current_inventory", "units_used", "units_collected", "wastage"],
            "simulator": {
                "horizons_days": [7],
                "stock_column": "current_inventory",
                "demand_column": "units_used",
                "supply_column": "units_collected",
                "waste_column": "wastage",
                "safety_threshold": 1.0,
            },
        }
    )
    simulator._bundle = {"type": "hybrid_simulator"}
    simulator_result = simulator.predict(
        None,
        [{"current_inventory": 8.0, "units_used": 2.0, "units_collected": 0.0, "wastage": 0.0}],
    ).iloc[0].to_dict()

    assert policy_result["donor_priority_score"].tolist() == [0.0, 1.0]
    assert policy_result["recommended_action"].tolist() == [
        "do not contact low score",
        "urgent contact",
    ]
    assert simulator_result["expected_days_until_stockout"] == pytest.approx(3.5)
    assert 0.0 <= simulator_result["p_stockout_0_7d"] <= 1.0


def test_real_sota_catalog_references_all_global_sota_joblib_artifacts():
    catalog_path = ML_BACKEND_ROOT / "config" / "sota_model_catalog.json"
    catalog = load_sota_catalog(catalog_path, require_artifacts=True)
    models_dir = (catalog_path.parent / str(catalog["default_models_dir"])).resolve()
    configured = {
        str(row["artifact_filename"])
        for row in catalog["models"]
        if bool(row.get("enabled"))
    }
    artifacts = {path.name for path in models_dir.glob("global_sota_*.joblib")}

    assert artifacts <= configured
