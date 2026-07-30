from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from simulation_studio.app.main import app


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    config_path = tmp_path / "operational_simulation.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "2026-06-13",
                "enabled": True,
                "max_hospitals": 12,
                "min_hospitals": 1,
                "include_blood_banks": True,
                "include_mobile_from_strategy": True,
                "hospital_capacity": {"nurses": 3, "lab": 2, "processing": 1},
                "hospital_hours": [0, 24],
                "geo_bbox": {"north": 46.86, "south": 46.76, "east": -71.17, "west": -71.31},
                "component_split": {"RBC": 0.58, "PLATELETS": 0.27, "PLASMA": 0.15},
                "fallback_to_default_centers": True,
                "demand_weight_from_usage": True,
                "usage_demand_weight_floor": 0.5,
                "usage_demand_weight_cap": 2.5,
                "use_django_centers": True,
                "max_django_centers": 50,
            }
        ),
        encoding="utf-8",
    )
    from simulation_studio.app import config_routes, operational_seed

    monkeypatch.setattr(config_routes, "CONFIG_PATH", config_path)
    monkeypatch.setattr(operational_seed, "CONFIG_PATH", config_path)
    return config_path


def test_get_config_returns_current_file(isolated_config):
    client = TestClient(app)
    response = client.get("/api/studio/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["max_hospitals"] == 12


def test_put_config_updates_supported_fields(isolated_config):
    client = TestClient(app)
    response = client.put(
        "/api/studio/config",
        json={
            "enabled": False,
            "max_hospitals": 8,
            "min_hospitals": 2,
            "hospital_capacity": {"nurses": 5},
            "hospital_hours": {"open": 7.0, "close": 19.0},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["max_hospitals"] == 8
    assert payload["min_hospitals"] == 2
    assert payload["hospital_capacity"]["nurses"] == 5
    assert payload["hospital_capacity"]["lab"] == 2
    assert payload["hospital_hours"] == [7.0, 19.0]

    # Verify file was actually written
    written = json.loads(isolated_config.read_text(encoding="utf-8"))
    assert written["enabled"] is False


def test_put_config_rejects_invalid_geo_bounds(isolated_config):
    client = TestClient(app)
    response = client.put(
        "/api/studio/config",
        json={"geo_bbox": {"north": 46.0, "south": 47.0}},
    )
    assert response.status_code == 422


def test_put_config_rejects_component_split_not_summing_to_one(isolated_config):
    client = TestClient(app)
    response = client.put(
        "/api/studio/config",
        json={
            "component_split": {"RBC": 0.5, "PLATELETS": 0.5, "PLASMA": 0.5},
        },
    )
    assert response.status_code == 422


def test_put_config_rejects_max_less_than_min_hospitals(isolated_config):
    client = TestClient(app)
    response = client.put(
        "/api/studio/config",
        json={"max_hospitals": 1, "min_hospitals": 5},
    )
    assert response.status_code == 422
