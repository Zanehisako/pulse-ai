import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

from ml.core.model_config_hot_reload import ModelConfigHotReloadWatcher
from ml.core.model_config_sync import (
    ModelConfigSyncError,
    sync_model_config_file,
    validate_model_config_payload,
)
from ml.core.registry import ModelRegistry


def _model_row(model_id="config_model", *, enabled=True, model_type="mlflow"):
    return {
        "id": model_id,
        "description": f"Configured model {model_id}",
        "file_path": f"models:/{model_id}@champion",
        "type": model_type,
        "enabled": enabled,
        "features": ["age", "bmi"],
        "feature_info": {},
        "examples": [{"kind": "good", "user_query": f"Run {model_id}"}],
        "defaults": {"output": {"name": "score"}},
    }


def _payload(*rows):
    return {
        "runtime": {
            "config_hot_reload": {
                "enabled": True,
                "poll_interval_seconds": 0.25,
                "debounce_seconds": 0.25,
            }
        },
        "models": list(rows),
        "external_tools": [],
        "orchestrator": {},
    }


class _FakeModelConfig:
    def __init__(self, model_id, **values):
        self.model_id = model_id
        for key, value in values.items():
            setattr(self, key, value)

    def save(self, update_fields=None):
        return None

    def to_registry_dict(self):
        return {
            "id": self.model_id,
            "description": self.description,
            "file_path": self.file_path,
            "type": self.model_type,
            "features": self.features,
            "feature_info": self.feature_info,
            "examples": self.examples,
            "defaults": self.defaults,
        }


class _FakeQuerySet:
    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def exclude(self, **kwargs):
        rows = self.rows
        if "model_id__in" in kwargs:
            excluded = set(kwargs["model_id__in"])
            rows = [row for row in rows if row.model_id not in excluded]
        return _FakeQuerySet(rows)

    def update(self, **kwargs):
        for row in self.rows:
            for key, value in kwargs.items():
                setattr(row, key, value)
        return len(self.rows)


class _FakeModelConfigManager:
    def __init__(self):
        self.rows = {}

    def select_for_update(self):
        return self

    def filter(self, **kwargs):
        rows = list(self.rows.values())
        if "model_id__in" in kwargs:
            ids = set(kwargs["model_id__in"])
            rows = [row for row in rows if row.model_id in ids]
        if "config_source_path" in kwargs:
            source = kwargs["config_source_path"]
            rows = [row for row in rows if row.config_source_path == source]
        if "is_active" in kwargs:
            active = kwargs["is_active"]
            rows = [row for row in rows if row.is_active is active]
        return _FakeQuerySet(rows)

    def create(self, model_id, **defaults):
        row = _FakeModelConfig(model_id, **defaults)
        self.rows[model_id] = row
        return row

    def get(self, model_id):
        return self.rows[model_id]

    def active_registry_rows(self):
        return [
            row.to_registry_dict()
            for row in self.rows.values()
            if row.is_active
        ]


class ModelConfigSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.config_path = Path(self.tmp.name) / "config.json"
        self.manager = _FakeModelConfigManager()

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, payload):
        self.config_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_sync_adds_and_updates_config_rows(self):
        self._write(_payload(_model_row("donor_score")))
        summary = sync_model_config_file(
            self.config_path,
            reason="test-add",
            model_manager=self.manager,
            atomic_context=nullcontext,
        )

        self.assertTrue(summary["changed"])
        self.assertEqual(summary["created_total"], 1)
        row = self.manager.get("donor_score")
        self.assertTrue(row.is_active)
        self.assertEqual(row.config_source_path, str(self.config_path.resolve()))

        updated = _model_row("donor_score")
        updated["description"] = "Updated from config"
        self._write(_payload(updated))
        summary = sync_model_config_file(
            self.config_path,
            reason="test-update",
            model_manager=self.manager,
            atomic_context=nullcontext,
        )

        self.assertEqual(summary["updated_total"], 1)
        self.assertEqual(row.description, "Updated from config")

    def test_sync_maps_enabled_false_to_inactive(self):
        self._write(_payload(_model_row("disabled_model", enabled=False)))
        sync_model_config_file(
            self.config_path,
            reason="test-disabled",
            model_manager=self.manager,
            atomic_context=nullcontext,
        )

        row = self.manager.get("disabled_model")
        self.assertFalse(row.is_active)

    def test_sync_rejects_duplicate_ids(self):
        payload = _payload(_model_row("dup_model"), _model_row("dup_model"))

        with self.assertRaisesRegex(ModelConfigSyncError, "Duplicate model id"):
            validate_model_config_payload(payload)

    def test_sync_rejects_invalid_model_type(self):
        payload = _payload(_model_row("bad_type", model_type="unknown_adapter"))

        with self.assertRaisesRegex(ModelConfigSyncError, "Unsupported model type"):
            validate_model_config_payload(payload)

    def test_sync_rejects_missing_required_field(self):
        row = _model_row("missing_field")
        row.pop("enabled")

        with self.assertRaisesRegex(ModelConfigSyncError, "missing required fields"):
            validate_model_config_payload(_payload(row))

    def test_removed_config_rows_deactivate_only_same_source_path(self):
        self._write(_payload(_model_row("kept_model"), _model_row("removed_model")))
        sync_model_config_file(
            self.config_path,
            reason="test-seed",
            model_manager=self.manager,
            atomic_context=nullcontext,
        )
        uploaded = self.manager.create(
            model_id="uploaded_model",
            description="Uploaded through API",
            file_path="uploads/uploaded_model.pkl",
            model_type="sklearn",
            features=["age"],
            feature_info={},
            examples=[],
            defaults={},
            is_active=True,
            config_source_path="",
        )

        self._write(_payload(_model_row("kept_model")))
        summary = sync_model_config_file(
            self.config_path,
            reason="test-remove",
            model_manager=self.manager,
            atomic_context=nullcontext,
        )

        self.assertEqual(summary["deactivated_removed_total"], 1)
        self.assertTrue(self.manager.get("kept_model").is_active)
        self.assertFalse(self.manager.get("removed_model").is_active)
        self.assertTrue(uploaded.is_active)

    def test_synced_rows_load_through_registry_loader(self):
        self._write(_payload(_model_row("registry_model")))
        sync_model_config_file(
            self.config_path,
            reason="test-registry",
            model_manager=self.manager,
            atomic_context=nullcontext,
        )
        rows = self.manager.active_registry_rows()

        registry = ModelRegistry(
            models_dir=Path(self.tmp.name),
            config_path=self.config_path,
            config_loader=lambda: rows,
        )
        registry.refresh(force=True)

        runtime = registry.get("registry_model")
        self.assertIsNotNone(runtime)
        self.assertEqual(runtime.status, "loaded")


class ModelConfigWatcherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.config_path = Path(self.tmp.name) / "config.json"
        self.config_path.write_text(json.dumps({"version": 1}), encoding="utf-8")
        self.reloads = []

    def tearDown(self):
        self.tmp.cleanup()

    def _watcher(self, callback=None):
        return ModelConfigHotReloadWatcher(
            path=self.config_path,
            poll_interval_seconds=0.25,
            debounce_seconds=0.25,
            reload_callback=callback or self._record_reload,
        )

    def _record_reload(self, path):
        self.reloads.append(json.loads(path.read_text(encoding="utf-8")))
        return {"ok": True}

    def test_poll_once_detects_changed_file_after_debounce(self):
        watcher = self._watcher()
        watcher.prime()
        self.config_path.write_text(json.dumps({"version": 2}), encoding="utf-8")

        first = watcher.poll_once(now=1.0)
        second = watcher.poll_once(now=1.3)

        self.assertTrue(first["debounced"])
        self.assertTrue(second["applied"])
        self.assertEqual(self.reloads, [{"version": 2}])

    def test_poll_once_debounces_unstable_writes(self):
        watcher = self._watcher()
        watcher.prime()
        self.config_path.write_text(json.dumps({"version": 2}), encoding="utf-8")
        watcher.poll_once(now=1.0)
        self.config_path.write_text(json.dumps({"version": 3}), encoding="utf-8")

        second = watcher.poll_once(now=1.1)
        third = watcher.poll_once(now=1.2)
        fourth = watcher.poll_once(now=1.4)

        self.assertTrue(second["debounced"])
        self.assertTrue(third["debounced"])
        self.assertTrue(fourth["applied"])
        self.assertEqual(self.reloads, [{"version": 3}])

    def test_poll_once_rejects_invalid_json_without_applying(self):
        applied = []

        def callback(path):
            applied.append(json.loads(path.read_text(encoding="utf-8")))
            return {"ok": True}

        watcher = self._watcher(callback)
        watcher.prime()
        self.config_path.write_text("{", encoding="utf-8")
        watcher.poll_once(now=1.0)
        result = watcher.poll_once(now=1.3)

        self.assertFalse(result["applied"])
        self.assertIn("JSONDecodeError", result["error"])
        self.assertEqual(applied, [])
        self.assertIn("JSONDecodeError", watcher.status()["last_file_error"])

    def test_poll_once_calls_generic_reload_callback(self):
        calls = []

        def callback(path):
            calls.append(path)
            return {"file_sync": {"changed": True}}

        watcher = self._watcher(callback)
        watcher.prime()
        self.config_path.write_text(json.dumps({"version": 2}), encoding="utf-8")
        watcher.poll_once(now=1.0)
        result = watcher.poll_once(now=1.3)

        self.assertTrue(result["applied"])
        self.assertEqual(calls, [self.config_path.resolve()])
        self.assertEqual(
            watcher.status()["last_reload_result"],
            {"file_sync": {"changed": True}},
        )
