"""Tests for the ModelUploadView endpoint.

Covers:
- Happy path: valid file + valid config → 201, DB written, registry reloaded.
- Missing model_file → 400.
- Missing config → 400.
- Invalid JSON config → 400.
- Missing model_id in config → 422.
- Empty features list → 422.
- Unsupported model_type → 422.
- Unsupported file extension → 422.
- Extension / model_type mismatch → 422.
- Duplicate active model_id → 409.
"""

import io
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, mock_open

import django
from django.test import SimpleTestCase
from rest_framework.test import APIRequestFactory

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")

if "keycloak" not in sys.modules:
    keycloak_stub = SimpleNamespace(KeycloakOpenID=object, KeycloakAdmin=object)
    sys.modules["keycloak"] = keycloak_stub

django.setup()

from django.core.files.uploadedfile import InMemoryUploadedFile
from ml.api.views import ModelUploadView, _validate_upload_config


def _make_file(name="model.pkl", content=b"fake-model-bytes"):
    buf = io.BytesIO(content)
    return InMemoryUploadedFile(buf, "model_file", name, "application/octet-stream", len(content), None)


def _valid_config(**overrides) -> str:
    base = {
        "model_id": "ext_donor_v1",
        "model_type": "sklearn",
        "features": ["age", "weight"],
        "description": "External donor eligibility model",
        "feature_info": {"age": {"type": "int"}},
        "examples": [{"age": 30, "weight": 70}],
        "defaults": {"age": 25},
    }
    base.update(overrides)
    return json.dumps(base)


def _make_request(factory, file=None, config=None):
    data = {}
    if file is not None:
        data["model_file"] = file
    if config is not None:
        data["config"] = config
    return factory.post("/api/ml/models/upload/", data, format="multipart")


class ModelUploadViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()

    # ── Happy path ────────────────────────────────────────────────────────

    def test_valid_upload_returns_201_and_registers_model(self):
        request = _make_request(self.factory, file=_make_file(), config=_valid_config())

        manager = MagicMock()
        manager.filter.return_value.exists.return_value = False  # no duplicate
        uoc_result = (MagicMock(), True)
        manager.update_or_create.return_value = uoc_result

        with (
            patch("ml.api.views.MLModelConfig.objects", manager),
            patch("ml.api.views.transaction.atomic", return_value=_nullctx()),
            patch("ml.api.views.refresh_runtime_state", return_value={}) as reload_mock,
            patch("os.makedirs"),
            patch("builtins.open", mock_open()),
        ):
            response = ModelUploadView.as_view()(request)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["status"], "registered")
        self.assertEqual(response.data["model_id"], "ext_donor_v1")
        self.assertEqual(response.data["model_type"], "sklearn")
        self.assertTrue(response.data["is_active"])
        self.assertIn("uploads", response.data["file_path"])
        reload_mock.assert_called_once_with("model-upload", force=True, warmup=False)
        manager.update_or_create.assert_called_once()

    def test_pytorch_pt_extension_accepted(self):
        request = _make_request(
            self.factory,
            file=_make_file("net.pt"),
            config=_valid_config(model_id="ext_net", model_type="pytorch"),
        )

        manager = MagicMock()
        manager.filter.return_value.exists.return_value = False

        with (
            patch("ml.api.views.MLModelConfig.objects", manager),
            patch("ml.api.views.transaction.atomic", return_value=_nullctx()),
            patch("ml.api.views.refresh_runtime_state", return_value={}),
            patch("os.makedirs"),
            patch("builtins.open", mock_open()),
        ):
            response = ModelUploadView.as_view()(request)

        self.assertEqual(response.status_code, 201)
        self.assertIn(".pt", response.data["file_path"])

    # ── Missing fields ────────────────────────────────────────────────────

    def test_missing_model_file_returns_400(self):
        request = _make_request(self.factory, config=_valid_config())
        response = ModelUploadView.as_view()(request)
        self.assertEqual(response.status_code, 400)
        self.assertIn("model_file", response.data["error"])

    def test_missing_config_returns_400(self):
        request = _make_request(self.factory, file=_make_file())
        response = ModelUploadView.as_view()(request)
        self.assertEqual(response.status_code, 400)
        self.assertIn("config", response.data["error"])

    def test_invalid_json_config_returns_400(self):
        request = _make_request(self.factory, file=_make_file(), config="not-json{{{")
        response = ModelUploadView.as_view()(request)
        self.assertEqual(response.status_code, 400)
        self.assertIn("JSON", response.data["error"])

    # ── Config validation (422) ───────────────────────────────────────────

    def test_missing_model_id_returns_422(self):
        cfg = json.dumps({"model_type": "sklearn", "features": ["age"]})
        request = _make_request(self.factory, file=_make_file(), config=cfg)
        response = ModelUploadView.as_view()(request)
        self.assertEqual(response.status_code, 422)
        self.assertIn("model_id", response.data["error"])

    def test_empty_features_returns_422(self):
        cfg = _valid_config(features=[])
        request = _make_request(self.factory, file=_make_file(), config=cfg)
        response = ModelUploadView.as_view()(request)
        self.assertEqual(response.status_code, 422)
        self.assertIn("features", response.data["error"])

    def test_unsupported_model_type_returns_422(self):
        cfg = _valid_config(model_type="tensorflow")
        request = _make_request(self.factory, file=_make_file(), config=cfg)
        response = ModelUploadView.as_view()(request)
        self.assertEqual(response.status_code, 422)
        self.assertIn("model_type", response.data["error"])

    def test_unsupported_extension_returns_422(self):
        request = _make_request(
            self.factory, file=_make_file("model.h5"), config=_valid_config()
        )
        response = ModelUploadView.as_view()(request)
        self.assertEqual(response.status_code, 422)
        self.assertIn("extension", response.data["error"].lower())

    def test_extension_type_mismatch_returns_422(self):
        # .ubj is xgboost-only; sklearn is not compatible
        request = _make_request(
            self.factory,
            file=_make_file("model.ubj"),
            config=_valid_config(model_type="sklearn"),
        )
        response = ModelUploadView.as_view()(request)
        self.assertEqual(response.status_code, 422)
        self.assertIn("compatible", response.data["error"].lower())

    # ── Duplicate (409) ──────────────────────────────────────────────────

    def test_duplicate_active_model_returns_409(self):
        request = _make_request(self.factory, file=_make_file(), config=_valid_config())

        manager = MagicMock()
        manager.filter.return_value.exists.return_value = True  # already active

        with patch("ml.api.views.MLModelConfig.objects", manager):
            response = ModelUploadView.as_view()(request)

        self.assertEqual(response.status_code, 409)
        self.assertIn("ext_donor_v1", response.data["error"])


# ── Unit tests for _validate_upload_config ────────────────────────────────

class ValidateUploadConfigTests(SimpleTestCase):
    def test_minimal_valid_config(self):
        result = _validate_upload_config({"model_id": "m1", "features": ["age"]})
        model_id, model_type, desc, features, *_ = result
        self.assertEqual(model_id, "m1")
        self.assertEqual(model_type, "sklearn")  # default
        self.assertEqual(features, ["age"])

    def test_missing_model_id_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_upload_config({"features": ["age"]})
        self.assertIn("model_id", str(ctx.exception))

    def test_empty_features_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_upload_config({"model_id": "m1", "features": []})
        self.assertIn("features", str(ctx.exception))

    def test_bad_model_type_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_upload_config({"model_id": "m1", "features": ["x"], "model_type": "keras"})
        self.assertIn("model_type", str(ctx.exception))

    def test_feature_info_must_be_dict(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_upload_config({"model_id": "m1", "features": ["x"], "feature_info": "bad"})
        self.assertIn("feature_info", str(ctx.exception))


# ── helpers ───────────────────────────────────────────────────────────────

from contextlib import contextmanager

@contextmanager
def _nullctx():
    yield
