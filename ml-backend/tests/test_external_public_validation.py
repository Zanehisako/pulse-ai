from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import validate_quebec_public_data as validation  # noqa: E402


def _base_config(tmp_path: Path) -> dict:
    donor_path = tmp_path / "donor.csv"
    inventory_path = tmp_path / "inventory.csv"
    pd.DataFrame(
        {
            "donor_id": ["D1", "D2", "D3", "D4"],
            "event_timestamp": pd.date_range("2024-01-01", periods=4),
            "annual_blood_products_delivered": [300000, 300000, 300000, 300000],
            "blood_type": ["O+", "A+", "O-", "B+"],
            "region": ["Montreal", "Monteregie", "Montreal", "Laval"],
            "age": [20, 32, 45, 68],
            "sex": ["F", "M", "F", "X"],
            "city": ["Montreal", "Laval", "Gatineau", "Sherbrooke"],
        }
    ).to_csv(donor_path, index=False)
    pd.DataFrame(
        {
            "wilaya": ["Montreal", "Quebec"],
            "short_horizon_stockout_risk_score": [0.2, 0.8],
        }
    ).to_csv(inventory_path, index=False)
    return {
        "id": "test_external_validation",
        "enabled": True,
        "output": {
            "directory": str(tmp_path / "out"),
            "json_report": "external_validation_report.json",
            "markdown_report": "external_validation_report.md",
            "tables_directory": "tables",
            "plots_directory": "plots",
        },
        "synthetic_datasets": {
            "donor_features": {
                "enabled": True,
                "path": str(donor_path),
                "format": "csv",
            },
            "inventory_features": {
                "enabled": True,
                "path": str(inventory_path),
                "format": "csv",
            },
        },
        "sources": {
            "annual": {
                "enabled": True,
                "source_type": "published_aggregate",
                "url": "https://example.test/annual",
                "metrics": {"blood_products_distributed": 300478},
            },
            "isq": {
                "enabled": True,
                "source_type": "statistical_table",
                "url": "https://example.test/isq",
                "region_population_weights": {
                    "Montreal": 2,
                    "Monteregie": 1,
                    "Laval": 1,
                },
            },
            "centres": {
                "enabled": True,
                "source_type": "html_reference",
                "url": "https://example.test/centres",
                "centre_cities": ["Montreal", "Laval", "Gatineau", "Sherbrooke"],
            },
            "er": {
                "enabled": False,
                "source_type": "csv",
                "url": "https://example.test/er",
                "data_url": "https://example.test/er.csv",
                "fetch": {"enabled": False, "fail_on_error": False},
                "derived_metrics": {
                    "er_occupancy_rate": {
                        "numerator": "occupied",
                        "denominator": "functional",
                    }
                },
            },
        },
        "comparisons": {
            "annual_products": {
                "enabled": True,
                "comparison_type": "scalar_column_vs_source_metric",
                "source_id": "annual",
                "dataset": "donor_features",
                "synthetic_column": "annual_blood_products_delivered",
                "aggregator": "median",
                "source_metric": "blood_products_distributed",
                "threshold": {"relative_error_max": 0.01},
                "plot": False,
            },
            "region_distribution": {
                "enabled": True,
                "comparison_type": "distribution_vs_expected_weights",
                "source_id": "isq",
                "dataset": "donor_features",
                "synthetic_column": "region",
                "dedupe_key": "donor_id",
                "source_weights": "region_population_weights",
                "threshold": {"max_total_variation_distance": 0.3},
                "plot": False,
            },
            "centre_coverage": {
                "enabled": True,
                "comparison_type": "coverage_values",
                "source_id": "centres",
                "dataset": "donor_features",
                "synthetic_column": "city",
                "dedupe_key": "donor_id",
                "source_values": "centre_cities",
                "normalization": "ascii_lower",
                "threshold": {"min_coverage_rate": 1.0},
                "plot": False,
            },
            "er_proxy": {
                "enabled": False,
                "comparison_type": "external_proxy_profile",
                "source_id": "er",
                "dataset": "inventory_features",
                "synthetic_column": "short_horizon_stockout_risk_score",
                "external_metric": "er_occupancy_rate",
                "quantiles": [0.25, 0.5, 0.75],
                "threshold": {
                    "minimum_synthetic_rows": 1,
                    "minimum_external_rows": 1
                },
                "plot": False,
            },
        },
        "limitations": ["fixture limitation"],
    }


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_config_validation_requires_enabled_source_url(tmp_path: Path):
    config = _base_config(tmp_path)
    config["sources"]["annual"]["url"] = ""
    with pytest.raises(ValueError, match="url must be present"):
        validation.validate_public_validation_config(config)


def test_project_external_validation_config_validates():
    config = validation.load_public_validation_config()
    assert config["id"] == "quebec_public_external_validation"
    assert "hema_quebec_annual_report_2023_2024" in config["sources"]
    assert "annual_blood_products_delivered" in config["comparisons"]


def test_load_enabled_sources_excludes_disabled_sources(tmp_path: Path):
    config = validation.validate_public_validation_config(_base_config(tmp_path))
    enabled = validation.load_enabled_sources(config)
    assert "annual" in enabled
    assert "er" not in enabled


def test_invalid_metric_mapping_fails_clearly(tmp_path: Path):
    config = _base_config(tmp_path)
    config["comparisons"]["annual_products"]["source_metric"] = "missing_metric"
    path = _write_config(tmp_path, config)
    with pytest.raises(ValueError, match="source metric not found"):
        validation.build_external_validation_report(path)


def test_build_external_validation_report_from_fixtures(tmp_path: Path):
    path = _write_config(tmp_path, _base_config(tmp_path))
    report = validation.build_external_validation_report(path)

    assert report["summary"]["comparison_count"] == 3
    assert report["comparisons"]["annual_products"]["status"] == "PASS"
    assert report["comparisons"]["centre_coverage"]["status"] == "PASS"
    assert (tmp_path / "out" / "external_validation_report.json").exists()
    assert (tmp_path / "out" / "external_validation_report.md").exists()
    assert (tmp_path / "out" / "tables" / "annual_products.csv").exists()


def test_unavailable_public_download_can_fail_closed(tmp_path: Path):
    config = _base_config(tmp_path)
    config["sources"]["er"]["enabled"] = True
    config["sources"]["er"]["data_url"] = str(tmp_path / "missing.csv")
    config["sources"]["er"]["fetch"] = {"enabled": True, "fail_on_error": True}
    path = _write_config(tmp_path, config)

    report = validation.build_external_validation_report(path)

    assert report["sources"]["er"]["status"] == "error"
    assert "Could not download public source" in report["sources"]["er"]["detail"]
