from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.core.scheduled_predictions import (
    attr_value,
    feature_frame_from_mapping,
    run_scheduled_prediction_jobs,
)


class _FakePredictor:
    def __init__(self, resolution=None):
        self.resolution = resolution or type("R", (), {"runtime": type("RT", (), {"model_id": "fake"})()})()

    def predict(self, frame: pd.DataFrame) -> list[dict]:
        return [{"score": float(len(frame))}] * len(frame)


def test_attr_value_supports_nested_dicts():
    row = {"hospital": {"hospital_id": "H-1"}, "usage_today": 4}
    assert attr_value(row, "hospital.hospital_id") == "H-1"
    assert attr_value(row, "usage_today") == 4


def test_run_scheduled_prediction_jobs_with_dict_entities(tmp_path: Path):
    config = {
        "jobs": [
            {
                "id": "test_job",
                "enabled": True,
                "model_id": "demo_model",
                "model_version": "champion",
                "entity_source": "tests.supply_entities",
                "feature_builder": "tests.feature_builder",
                "feature_mapping": {
                    "hospital_id": {"attr": "hospital.hospital_id", "default": ""},
                    "usage_today": {"attr": "usage_today", "default": 0},
                },
                "prediction_value": {"path": "score"},
                "result": {
                    "entity_id_attr": "supply_id",
                    "entity_type": "supply",
                    "hospital_id_attr": "hospital.hospital_id",
                    "blood_type_attr": "blood_product_type",
                },
                "alert": {"enabled": False},
            }
        ]
    }
    config_path = tmp_path / "scheduled_predictions.json"
    config_path.write_text(__import__("json").dumps(config), encoding="utf-8")

    entities = [
        {
            "supply_id": "S-1",
            "hospital": {"hospital_id": "H-1"},
            "blood_product_type": "O+",
            "usage_today": 3,
        }
    ]

    def entity_resolver(path: str):
        assert path == "tests.supply_entities"
        return entities

    def model_loader(model_id, version, loading_policy=None):
        return _FakePredictor(), type(
            "Res",
            (),
            {"source_used": "test", "source_ref": "test", "fallback_reason": ""},
        )()

    summary, predictions = run_scheduled_prediction_jobs(
        config_path=config_path,
        entity_resolver=entity_resolver,
        model_loader=model_loader,
    )

    assert summary["jobs"][0]["status"] == "ok"
    assert len(predictions) == 1
    assert predictions[0]["entity_id"] == "S-1"
    assert predictions[0]["predicted_value"] == 1.0


def test_feature_frame_from_mapping():
    frame = feature_frame_from_mapping(
        {"blood_product_type": "A+", "usage_today": 2},
        {"blood_type": {"attr": "blood_product_type"}, "usage_today": {"attr": "usage_today"}},
    )
    assert frame.iloc[0]["blood_type"] == "A+"