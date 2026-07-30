from __future__ import annotations

import pytest

from simulation_studio.app.center_loader import (
    _capacity_from_daily,
    _center_to_config,
    _first_open_hours,
    _map_db_type,
    _parse_hours_window,
    load_django_centre_configs,
    resolve_center_configs,
)


@pytest.fixture
def config() -> dict:
    return {
        "use_django_centers": True,
        "max_django_centers": 50,
        "django_center_type_map": {
            "fixe": "blood_bank",
            "mobile": "mobile",
            "hopital": "hospital",
            "clinique": "hospital",
            "*": "blood_bank",
        },
        "django_capacity_divisors": {
            "nurses": 30,
            "lab": 60,
            "processing": 90,
        },
        "django_default_capacity": {
            "hospital": {"nurses": 4, "lab": 3, "processing": 2},
            "blood_bank": {"nurses": 4, "lab": 2, "processing": 2},
            "mobile": {"nurses": 2, "lab": 1, "processing": 1},
        },
        "django_default_hours": {
            "hospital": [0, 24],
            "blood_bank": [7.25, 19.75],
            "mobile": [9, 18],
        },
    }


def test_parse_hours_window():
    assert _parse_hours_window("08:30-16:00") == (8.5, 16.0)
    assert _parse_hours_window("fermé") is None
    assert _parse_hours_window("closed") is None
    assert _parse_hours_window("invalid") is None


def test_first_open_hours_dict(config):
    horaires = {
        "lun": "07:00-19:00",
        "mar": "fermé",
    }
    assert _first_open_hours(horaires) == (7.0, 19.0)


def test_first_open_hours_tuple():
    assert _first_open_hours(["09:00", "17:00"]) == (9.0, 17.0)


def test_map_db_type(config):
    assert _map_db_type("fixe", config) == "blood_bank"
    assert _map_db_type("mobile", config) == "mobile"
    assert _map_db_type("HOPITAL", config) == "hospital"
    assert _map_db_type("clinique", config) == "hospital"
    assert _map_db_type("unknown", config) == "blood_bank"


def test_capacity_from_daily(config):
    # 120 daily -> 4 nurses, 2 lab, 1 processing
    cap = _capacity_from_daily(120, "blood_bank", config)
    assert cap == {"nurses": 4, "lab": 2, "processing": 1}


def test_capacity_from_daily_defaults(config):
    cap = _capacity_from_daily(None, "hospital", config)
    assert cap == {"nurses": 4, "lab": 3, "processing": 2}


def test_center_to_config_preserves_original_type(config):
    row = {
        "nom": "Centre Test",
        "code_centre": "CT001",
        "type": "fixe",
        "region": "Québec",
        "latitude": 46.812,
        "longitude": -71.21,
        "capacite_journaliere": 90,
        "horaires": {"lun": "08:00-18:00"},
    }
    cfg = _center_to_config(row, config)
    assert cfg["name"] == "Centre Test"
    assert cfg["type"] == "blood_bank"
    assert cfg["original_type"] == "fixe"
    assert cfg["external_id"] == "CT001"
    assert cfg["lat"] == 46.812
    assert cfg["hours"] == (8.0, 18.0)


def test_resolve_center_configs_prefers_operational(config, monkeypatch):
    operational_payload = {
        "hospitals": [{"id": "H1", "name": "Hospital One"}],
    }

    class FakePlan:
        enabled = True
        center_configs = [{"name": "Operational Center"}]

    configs, source = resolve_center_configs(
        config,
        operational_payload,
        _build_operational_seed_plan=lambda payload, config=None: FakePlan,
    )
    assert source == "operational_rows"
    assert configs == [{"name": "Operational Center"}]


def test_load_django_centre_configs_disabled(config):
    config["use_django_centers"] = False
    configs, source = load_django_centre_configs(config)
    assert configs == []
    assert source is None


def test_resolve_center_configs_fallback_to_default(config, monkeypatch):
    # Make Django unavailable and ensure fallback is returned.
    monkeypatch.setattr(
        "simulation_studio.app.center_loader.load_django_centre_configs",
        lambda config: ([], None),
    )
    config["fallback_to_default_centers"] = True
    configs, source = resolve_center_configs(config)
    assert configs is None
    assert source == "default_centers"
