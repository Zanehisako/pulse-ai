import json

import pytest

from simulation_studio.app.twin_forecast_config import (
    TwinForecastConfigError,
    default_job_id,
    forecast_panel_meta,
    list_panel_options,
    load_twin_forecast_config,
)

def test_load_twin_forecast_config_from_repo():
    cfg = load_twin_forecast_config()
    assert cfg.get("enabled") is True
    assert cfg.get("default_job_id")


def test_list_panel_options_includes_forecast_jobs():
    options = list_panel_options()
    assert options
    job_ids = {row["job_id"] for row in options}
    assert "dashboard_forecast_t7" in job_ids
    assert all(row.get("label") for row in options)


def test_forecast_panel_meta_exposes_chart_cap():
    meta = forecast_panel_meta()
    assert meta.get("enabled") is True
    assert isinstance(meta.get("chart_max_points"), int)
    assert meta["chart_max_points"] >= 1
    assert meta.get("default_job_id") == default_job_id()


def test_validate_rejects_unknown_default_job(tmp_path):
    cfg_path = tmp_path / "twin_forecast_panel.json"
    cfg_path.write_text(
        json.dumps(
            {
                "version": "test",
                "enabled": True,
                "scheduled_predictions_file": "scheduled_predictions.json",
                "panel_job_filter": {"dashboard_roles": ["forecast"]},
                "default_job_id": "nonexistent_job",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TwinForecastConfigError):
        load_twin_forecast_config(cfg_path)