import json
from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase

from ml.services.training_scheduler import get_schedule_config_path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _enabled_scheduled_model_ids() -> set[str]:
    payload = _read_json(Path(settings.PIOS_SCHEDULED_PREDICTION_CONFIG_PATH))
    return {
        str(job["model_id"])
        for job in payload.get("jobs", [])
        if isinstance(job, dict) and bool(job.get("enabled")) and job.get("model_id")
    }


def _enabled_training_models(schedule_path: Path) -> dict[str, dict]:
    payload = _read_json(schedule_path)
    return {
        str(row["registered_model_name"]): dict(row, schedule_key=key)
        for key, row in payload.get("scripts", {}).items()
        if isinstance(row, dict)
        and bool(row.get("enabled"))
        and row.get("registered_model_name")
    }


def _enabled_catalog_models(catalog_path: Path) -> dict[str, dict]:
    payload = _read_json(catalog_path)
    return {
        str(row["registered_model_name"]): dict(row)
        for row in payload.get("models", [])
        if isinstance(row, dict)
        and bool(row.get("enabled"))
        and row.get("registered_model_name")
    }


class ScheduledPredictionModelContractTests(SimpleTestCase):
    def test_dashboard_jobs_match_django_training_schedule_registered_names(self):
        scheduled_model_ids = _enabled_scheduled_model_ids()
        training_models = _enabled_training_models(get_schedule_config_path())

        self.assertFalse(scheduled_model_ids - set(training_models))

    def test_dashboard_jobs_match_start_local_training_schedule_registered_names(self):
        start_local_schedule = (
            Path(settings.BASE_DIR).parent
            / "ml-backend"
            / "config"
            / "training_schedule.json"
        )
        scheduled_model_ids = _enabled_scheduled_model_ids()
        training_models = _enabled_training_models(start_local_schedule)

        self.assertFalse(scheduled_model_ids - set(training_models))

    def test_dashboard_jobs_match_enabled_local_catalog_models(self):
        catalog_path = (
            Path(settings.BASE_DIR).parent
            / "ml-backend"
            / "config"
            / "sota_model_catalog.json"
        )
        scheduled_model_ids = _enabled_scheduled_model_ids()
        catalog_models = _enabled_catalog_models(catalog_path)

        self.assertFalse(scheduled_model_ids - set(catalog_models))

    def test_configured_child_models_are_trained_before_parent(self):
        workspace_root = Path(settings.BASE_DIR).parent
        config_dir = workspace_root / "ml-backend" / "config"
        schedule_path = config_dir / "training_schedule.json"
        schedule = _read_json(schedule_path)

        enabled_order = [
            str(row["registered_model_name"])
            for row in schedule.get("scripts", {}).values()
            if isinstance(row, dict)
            and bool(row.get("enabled"))
            and row.get("registered_model_name")
        ]
        position = {
            model_name: index for index, model_name in enumerate(enabled_order)
        }

        checked_configs = 0
        for config_path in sorted(config_dir.glob("*.json")):
            payload = _read_json(config_path)
            sub_models = payload.get("sub_models")
            if not isinstance(sub_models, dict) or not sub_models:
                continue

            parent = str(payload["registered_model_name"])
            child_models = {
                str(row["registered_model_name"])
                for row in sub_models.values()
                if isinstance(row, dict) and row.get("registered_model_name")
            }

            self.assertFalse(child_models - set(enabled_order))
            for child in child_models:
                self.assertLess(position[child], position[parent])
            checked_configs += 1

        self.assertGreater(checked_configs, 0)
