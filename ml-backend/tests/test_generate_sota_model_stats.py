from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_sota_model_stats.py"
CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "sota_model_catalog.json"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_sota_model_stats", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dashboard_profile_payload_is_catalog_driven():
    module = _load_script_module()
    model_entry = {
        "dashboard_profile": {
            "enabled": True,
            "fields": {
                "eligible_to_donate": {"mode": 1},
                "blood_type": "High-need or rare type",
            },
        },
    }

    assert module.dashboard_profile_payload(model_entry) == {
        "eligible_to_donate": {"mode": 1},
        "blood_type": {"mode": "High-need or rare type"},
    }


def test_merge_dashboard_profile_overrides_training_distribution_for_card():
    module = _load_script_module()
    stats = {
        "eligible_to_donate": {"lower": 0, "upper": 1, "median": 0},
        "center_distance_km": {"lower": 1, "upper": 60, "median": 30},
    }
    model_entry = {
        "dashboard_profile": {
            "enabled": True,
            "fields": {
                "eligible_to_donate": {"mode": 1},
                "center_distance_km": {"lower": 0, "upper": 20, "median": 10},
            },
        },
    }

    merged = module.merge_dashboard_profile(stats, model_entry)

    assert merged["eligible_to_donate"] == {"mode": 1}
    assert merged["center_distance_km"] == {"lower": 0, "upper": 20, "median": 10}


def test_donor_priority_catalog_defines_dashboard_profile():
    import json

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    donor_priority = next(
        row for row in catalog["models"]
        if row["registered_model_name"] == "donor_priority_policy_model"
    )

    profile = donor_priority["dashboard_profile"]
    assert profile["enabled"] is True
    assert set(profile["fields"]) >= {
        "eligible_to_donate",
        "blood_type",
        "center_distance_km",
        "readiness_score",
    }
