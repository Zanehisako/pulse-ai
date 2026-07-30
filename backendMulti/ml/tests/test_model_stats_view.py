import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
from django.apps import apps
if not apps.ready:
    django.setup()

if "keycloak" not in sys.modules:
    keycloak_stub = ModuleType("keycloak")
    keycloak_stub.KeycloakOpenID = object
    keycloak_stub.KeycloakAdmin = object
    sys.modules["keycloak"] = keycloak_stub

from django.test import SimpleTestCase
from rest_framework import status
from rest_framework.test import APIRequestFactory, force_authenticate

from ml.api.views import ModelsStatsListView, ModelsStatsView, _serialize_model_stats_entry


def authenticate(request):
    force_authenticate(
        request,
        user=SimpleNamespace(is_authenticated=True, id=1, pk=1, email="ci@example.com"),
    )
    return request


class ModelsStatsListViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.view = ModelsStatsListView.as_view()

    @patch("ml.api.views._collect_model_stats_entries")
    def test_list_returns_runtime_models_with_stats_overlay(self, mock_collect):
        mock_collect.return_value = [
            {
                "model_id": "model_a",
                "identifier": "model_a",
                "stats": {"age": 30},
                "made_at": "2026-03-28T10:00:00",
            },
            {
                "model_id": "model_b",
                "identifier": "model_b",
                "stats": {},
                "made_at": "",
            },
        ]

        request = self.factory.get("/api/ml/models_stats/")
        authenticate(request)
        response = self.view(request)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["models"]), 2)
        self.assertEqual(response.data["models"][1]["model_id"], "model_b")
        self.assertEqual(response.data["models"][1]["stats"], {})


class ModelsStatsDetailViewTests(SimpleTestCase):
    def setUp(self):
        self.factory = APIRequestFactory()
        self.view = ModelsStatsView.as_view()

    @patch("ml.api.views.ModelStats.objects.filter")
    @patch("ml.api.views.get_registry")
    @patch("ml.api.views._load_model_feature_names")
    @patch("ml.api.views.refresh_runtime_state")
    @patch("ml.api.views.is_ready")
    def test_detail_returns_runtime_placeholder_when_stats_missing(
        self,
        mock_is_ready,
        mock_refresh,
        mock_feature_names,
        mock_get_registry,
        mock_filter,
    ):
        mock_is_ready.return_value = True
        mock_feature_names.return_value = {}
        mock_registry = MagicMock()
        mock_registry.get.return_value = SimpleNamespace(model_id="model_b", slug="model_b")
        mock_get_registry.return_value = mock_registry
        mock_filter.return_value.first.return_value = None

        request = self.factory.get("/api/ml/models_stats/model_b/")
        authenticate(request)
        response = self.view(request, model_id="model_b")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["model_id"], "model_b")
        self.assertEqual(response.data["identifier"], "model_b")
        self.assertEqual(response.data["stats"], {})
        self.assertEqual(response.data["made_at"], "")

    @patch("ml.api.views.ModelStats.objects.filter")
    @patch("ml.api.views.get_registry")
    @patch("ml.api.views._load_model_feature_names")
    @patch("ml.api.views.refresh_runtime_state")
    @patch("ml.api.views.is_ready")
    def test_detail_uses_configured_metrics_path_when_stats_missing(
        self,
        mock_is_ready,
        mock_refresh,
        mock_feature_names,
        mock_get_registry,
        mock_filter,
    ):
        with TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            metrics_path.write_text(json.dumps({"auc": 0.91}), encoding="utf-8")
            mock_is_ready.return_value = True
            mock_feature_names.return_value = {}
            mock_registry = MagicMock()
            mock_registry.get.return_value = SimpleNamespace(
                model_id="model_b",
                slug="model_b",
                defaults={"metrics_path": str(metrics_path)},
            )
            mock_get_registry.return_value = mock_registry
            mock_filter.return_value.first.return_value = None

            request = self.factory.get("/api/ml/models_stats/model_b/")
            authenticate(request)
            response = self.view(request, model_id="model_b")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["stats"]["auc"], 0.91)
        self.assertEqual(response.data["stats"]["_metrics_source"], str(metrics_path))

    @patch("ml.api.views.ModelStats.objects.filter")
    @patch("ml.api.views._load_model_feature_names")
    @patch("ml.api.views.is_ready")
    def test_detail_returns_404_when_runtime_and_stats_missing(
        self, mock_is_ready, mock_feature_names, mock_filter
    ):
        mock_is_ready.return_value = False
        mock_feature_names.return_value = {}
        mock_filter.return_value.first.return_value = None

        request = self.factory.get("/api/ml/models_stats/unknown/")
        authenticate(request)
        response = self.view(request, model_id="unknown")

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
        self.assertIn("error", response.data)

    @patch("ml.api.views.ModelStats.objects.filter")
    @patch("ml.api.views._load_model_feature_names")
    @patch("ml.api.views.is_ready")
    def test_detail_parses_legacy_stringified_json_stats(
        self, mock_is_ready, mock_feature_names, mock_filter
    ):
        mock_is_ready.return_value = False
        mock_feature_names.return_value = {
            "donor_propensity_model": [
                "donor_features_features__lat",
                "donor_features_features__lon",
                "donor_features_features__availability",
                "donor_features_features__blood_group",
                "donor_features_features__recency_days",
                "donor_features_features__frequency_365",
                "donor_features_features__days_until_eligible",
            ]
        }
        mock_filter.return_value.first.return_value = SimpleNamespace(
            model_id="donor_propensity_model",
            identifier="donor",
            stats=(
                "[[0.0,0.0],[30.4,30.5],[1.4,1.5],[1.0,1.0],[3.0,4.0],"
                "[176.0,184.0],[2.0,2.0],[44.0,46.0],[0.0,0.1]]"
            ),
            made_at="2026-04-22T16:40:18.639242+00:00",
        )

        request = self.factory.get("/api/ml/models_stats/donor_propensity_model/")
        authenticate(request)
        response = self.view(request, model_id="donor_propensity_model")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["stats"]["label"], [0.0, 0.0])
        self.assertEqual(
            response.data["stats"]["donor_features_features__lat"], [30.4, 30.5]
        )
        self.assertEqual(response.data["stats"]["prediction"], [0.0, 0.1])


class ModelStatsSerializationTests(SimpleTestCase):
    def test_serialize_parses_legacy_stringified_json_stats(self):
        payload = _serialize_model_stats_entry(
            model_id="hospital_shortage_predictor",
            identifier="hospital",
            stats=(
                "[[1.0,1.0],[19.7,19.9],[2.7,3.2],[0.0,0.0],[0.0,0.0],"
                "[3.0,3.0],[2.0,2.0],[0.0,0.0],[0.9,1.0]]"
            ),
            made_at="2026-04-22T16:48:12.571413+00:00",
            feature_names=[
                "hospital_supply_features_features__temperature",
                "hospital_supply_features_features__rain_mm",
                "hospital_supply_features_features__holiday",
                "hospital_supply_features_features__disaster",
                "hospital_supply_features_features__scheduled_surgeries",
                "hospital_supply_features_features__trauma_cases",
                "hospital_supply_features_features__current_inventory",
            ],
        )

        self.assertEqual(payload["stats"]["label"], [1.0, 1.0])
        self.assertEqual(
            payload["stats"]["hospital_supply_features_features__temperature"],
            [19.7, 19.9],
        )
        self.assertEqual(payload["stats"]["prediction"], [0.9, 1.0])
