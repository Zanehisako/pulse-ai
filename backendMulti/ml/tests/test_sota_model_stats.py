from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
from django.core.management import call_command
from django.test import SimpleTestCase

from ml.core.sota_model_stats import (
    dashboard_profile_payload,
    generate_sota_model_stats,
    merge_dashboard_profile,
    resolve_sota_model_stats_paths,
)


def _write_runtime_config(path: Path, *, catalog_path: Path, datasets_dir: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "sota_model_stats": {
                    "enabled": True,
                    "models_dir_env": "PIOS_TEST_MODELS_DIR",
                    "default_models_dir": "unused",
                    "catalog_path": str(catalog_path),
                    "datasets_dir": str(datasets_dir),
                    "generate_missing_datasets": False,
                    "donor_feature_markers": [
                        "eligible_to_donate",
                        "is_rare_type",
                        "readiness_score",
                    ],
                }
            }
        ),
        encoding="utf-8",
    )


def _write_catalog(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "donor_profile_catalog_row",
                        "enabled": True,
                        "registered_model_name": "donor_profile_runtime_model",
                        "artifact_filename": "donor_profile_runtime_model.joblib",
                        "task_type": "policy",
                        "features": [
                            "eligible_to_donate",
                            "is_rare_type",
                            "readiness_score",
                        ],
                        "outputs": {"primary": "donor_priority_score"},
                        "dashboard_profile": {
                            "enabled": True,
                            "fields": {
                                "eligible_to_donate": {"mode": 1},
                                "readiness_score": {
                                    "lower": 0.7,
                                    "upper": 1.0,
                                    "median": 0.85,
                                },
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


class SotaModelStatsUnitTests(SimpleTestCase):
    def test_dashboard_profile_payload_normalizes_literal_values(self):
        payload = dashboard_profile_payload(
            {
                "dashboard_profile": {
                    "enabled": True,
                    "fields": {
                        "blood_type": "High-need or rare type",
                        "eligible_to_donate": {"mode": 1},
                    },
                }
            }
        )

        self.assertEqual(payload["blood_type"], {"mode": "High-need or rare type"})
        self.assertEqual(payload["eligible_to_donate"], {"mode": 1})

    def test_merge_dashboard_profile_overrides_distribution_stats(self):
        merged = merge_dashboard_profile(
            {"readiness_score": {"lower": 0.1, "upper": 0.5}},
            {
                "dashboard_profile": {
                    "enabled": True,
                    "fields": {"readiness_score": {"lower": 0.7, "upper": 1.0}},
                }
            },
        )

        self.assertEqual(merged["readiness_score"], {"lower": 0.7, "upper": 1.0})

    def test_resolves_paths_from_runtime_config_and_environment(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog_path = root / "catalog.json"
            datasets_dir = root / "datasets"
            models_dir = root / "models"
            catalog_path.write_text('{"models": []}', encoding="utf-8")
            datasets_dir.mkdir()
            models_dir.mkdir()
            runtime_config = root / "prediction_runtime.json"
            _write_runtime_config(runtime_config, catalog_path=catalog_path, datasets_dir=datasets_dir)

            with patch.dict(os.environ, {"PIOS_TEST_MODELS_DIR": str(models_dir)}):
                paths = resolve_sota_model_stats_paths(runtime_config)

        self.assertEqual(paths["catalog_path"], catalog_path.resolve())
        self.assertEqual(paths["datasets_dir"], datasets_dir.resolve())
        self.assertEqual(paths["models_dir"], models_dir.resolve())


class SotaModelStatsIntegrationTests(SimpleTestCase):
    def test_generate_sota_model_stats_persists_profile_from_configured_catalog(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            datasets_dir = root / "datasets"
            models_dir.mkdir()
            datasets_dir.mkdir()
            (models_dir / "donor_profile_runtime_model.joblib").write_bytes(b"placeholder")
            catalog_path = root / "catalog.json"
            runtime_config = root / "prediction_runtime.json"
            _write_catalog(catalog_path)
            _write_runtime_config(runtime_config, catalog_path=catalog_path, datasets_dir=datasets_dir)

            donor_features = pd.DataFrame(
                {
                    "eligible_to_donate": [0, 1, 1],
                    "is_rare_type": [0, 1, 0],
                    "readiness_score": [0.2, 0.6, 0.8],
                }
            )
            inventory_features = pd.DataFrame({"stock": [1, 2, 3]})
            manager = MagicMock()

            with (
                patch.dict(os.environ, {"PIOS_TEST_MODELS_DIR": str(models_dir)}),
                patch(
                    "ml.core.sota_model_stats.load_or_generate_datasets",
                    return_value=(inventory_features, donor_features),
                ),
                patch("ml.core.sota_model_stats.ModelStats.objects", manager),
            ):
                summary = generate_sota_model_stats(runtime_config_path=runtime_config)

        self.assertEqual(summary["persisted"], 1)
        manager.update_or_create.assert_called_once()
        kwargs = manager.update_or_create.call_args.kwargs
        self.assertEqual(kwargs["model_id"], "donor_profile_runtime_model")
        self.assertEqual(kwargs["defaults"]["stats"]["eligible_to_donate"], {"mode": 1})
        self.assertEqual(
            kwargs["defaults"]["stats"]["readiness_score"],
            {"lower": 0.7, "upper": 1.0, "median": 0.85},
        )

    def test_management_command_runs_backend_generator(self):
        with patch(
            "ml.management.commands.generate_sota_model_stats.generate_sota_model_stats",
            return_value={
                "persisted": 1,
                "processed": [
                    {
                        "model_id": "donor_profile_runtime_model",
                        "available_features": 3,
                        "configured_features": 3,
                        "task_type": "policy",
                    }
                ],
                "skipped": [],
            },
        ) as generator:
            call_command("generate_sota_model_stats")

        generator.assert_called_once()
