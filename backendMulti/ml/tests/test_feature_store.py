import datetime as dt
import json
import os
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import MagicMock, patch

import django
from django.apps import apps
from django.test import SimpleTestCase

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
if not apps.ready:
    django.setup()

from ml.core.feature_store import (
    _repo_path,
    _resolve_model_default_features,
    resolve_component_prediction_features,
    resolve_prediction_features,
)
from ml.core.registry import ModelRuntime


class ResolvePredictionFeaturesTests(SimpleTestCase):
    def _runtime(self) -> ModelRuntime:
        return ModelRuntime(
            model_id="stockout_days_predictor",
            aliases=["stockout_days_predictor"],
            slug="stockout_days_predictor",
            file_path=Path("/tmp/unused.pkl"),
            description="stockout model",
            feature_names=[
                "blood_product_type",
                "current_stock_units",
                "usage_today",
                "lead_time_days",
                "days_since_last_restock",
                "stockout_count_90d",
                "scheduled_surgeries_next7d",
            ],
            defaults={
                "prediction_source": "feast_online",
                "feast": {
                    "feature_service": "stockout_sequence_service",
                    "entity_keys": ["hospital_id", "blood_type"],
                    "allow_direct_input": True,
                    "allow_feature_overrides": True,
                },
            },
        )

    def _prefixed_runtime(self) -> ModelRuntime:
        return ModelRuntime(
            model_id="hospital_shortage_predictor",
            aliases=["hospital_shortage_predictor"],
            slug="hospital_shortage_predictor",
            file_path=Path("/tmp/unused.pkl"),
            description="hospital model",
            feature_names=[
                "hospital_supply_features_features__temperature",
                "hospital_supply_features_features__rain_mm",
                "hospital_supply_features_features__holiday",
                "hospital_supply_features_features__disaster",
                "hospital_supply_features_features__scheduled_surgeries",
                "hospital_supply_features_features__trauma_cases",
                "hospital_supply_features_features__current_inventory",
            ],
            defaults={
                "prediction_source": "feast_online",
                "feast": {
                    "feature_service": "hospital_shortage_service",
                    "entity_keys": ["hospital_id", "blood_type"],
                    "allow_direct_input": True,
                    "allow_feature_overrides": True,
                },
            },
        )

    def test_default_repo_path_points_to_workspace_feast_repo(self):
        repo_path = _repo_path({})

        self.assertEqual(repo_path.name, "feature_repo")
        self.assertEqual(repo_path.parent.name, "ml-backend")
        self.assertTrue((repo_path / "feature_store.yaml").exists())

    def test_configured_repo_path_overrides_default(self):
        repo_path = _repo_path({"repo_path": "/tmp/custom-feast-repo"})

        self.assertEqual(repo_path, Path("/tmp/custom-feast-repo").resolve())

    @patch("ml.core.feature_store._get_feature_store")
    def test_resolve_prediction_features_fetches_feast_online(self, mock_get_store):
        store = MagicMock()
        store.get_feature_service.return_value = object()
        store.get_online_features.return_value.to_dict.return_value = {
            "hospital_id": ["H001"],
            "blood_type": ["A+"],
            "current_stock_units": [5],
            "usage_today": [3.5],
            "lead_time_days": [2],
            "blood_product_type": ["A+"],
            "days_since_last_restock": [1],
            "stockout_count_90d": [0],
            "scheduled_surgeries_next7d": [4],
        }
        mock_get_store.return_value = store

        resolved = resolve_prediction_features(
            self._runtime(),
            {"hospital": "H001", "blood_type": "A+", "current_stock_units": 7},
        )

        self.assertEqual(resolved.metadata["source"], "feast_online")
        self.assertEqual(
            resolved.metadata["feature_service"], "stockout_sequence_service"
        )
        self.assertEqual(
            resolved.metadata["entity_row"],
            {"hospital_id": "H001", "blood_type": "A+"},
        )
        self.assertEqual(resolved.features["current_stock_units"], 7)
        self.assertEqual(resolved.features["usage_today"], 3.5)
        self.assertEqual(resolved.features["blood_product_type"], "A+")

    @patch("ml.core.feature_store.resolve_prediction_features")
    @patch("ml.core.feature_store._available_entity_key_values")
    def test_resolve_component_prediction_features_fans_out_missing_component(
        self, mock_components, mock_resolve
    ):
        mock_components.return_value = ["A+", "B-", "PLASMA"]
        payloads = {
            component: MagicMock(metadata={"entity_row": {"blood_type": component}})
            for component in mock_components.return_value
        }
        mock_resolve.side_effect = lambda runtime, request, forecast_horizon=None: payloads[
            request["blood_type"]
        ]

        runtime = self._runtime()
        runtime.defaults = {
            "prediction_source": "feast_online",
            "feast": {
                "feature_service": "stockout_sequence_service",
                "entity_keys": ["hospital_id"],
            },
        }

        resolved = resolve_component_prediction_features(
            runtime,
            {"hospital_id": "H001"},
            forecast_horizon={"timeframe": "medium", "days": 14},
        )

        self.assertEqual([component for component, _ in resolved], ["A+", "B-", "PLASMA"])
        self.assertEqual(
            [call.args[1]["blood_type"] for call in mock_resolve.call_args_list],
            ["A+", "B-", "PLASMA"],
        )

    def test_resolve_prediction_features_accepts_complete_direct_payload(self):
        runtime = self._runtime()
        direct_features = {
            "blood_product_type": "A+",
            "current_stock_units": 5,
            "usage_today": 3.5,
            "lead_time_days": 2,
            "days_since_last_restock": 1,
            "stockout_count_90d": 0,
            "scheduled_surgeries_next7d": 4,
        }

        resolved = resolve_prediction_features(runtime, direct_features)

        self.assertEqual(resolved.metadata["source"], "request")
        self.assertEqual(resolved.features, direct_features)

    @patch("ml.core.feature_store._resolve_model_default_features")
    def test_resolve_prediction_features_fills_null_padded_payload_from_stats(
        self, mock_defaults
    ):
        runtime = self._runtime()
        runtime.defaults = {
            "prediction_source": "feast_online",
            "feast": {
                "feature_service": "stockout_sequence_service",
                "entity_keys": ["hospital_id", "blood_type"],
                "allow_direct_input": True,
            },
        }
        null_padded_features = {
            "blood_product_type": "A+",
            "current_stock_units": 5,
            "usage_today": None,
            "lead_time_days": 2,
            "days_since_last_restock": 1,
            "stockout_count_90d": 0,
            "scheduled_surgeries_next7d": 4,
        }
        mock_defaults.return_value = {
            "features": {"usage_today": 3.5},
            "metadata": {
                "provider": "model_stats",
                "status": "found",
                "resolved_feature_names": ["usage_today"],
            },
        }

        resolved = resolve_prediction_features(runtime, null_padded_features)

        self.assertEqual(resolved.metadata["source"], "feature_completion")
        self.assertEqual(resolved.features["usage_today"], 3.5)
        self.assertNotIn("hospital_id", resolved.features)

    def test_resolve_prediction_features_completes_partial_donor_features_from_store(
        self,
    ):
        import pandas as pd

        runtime = ModelRuntime(
            model_id="donor_propensity_model",
            aliases=["donor_propensity_model"],
            slug="donor_propensity_model",
            file_path=Path("/tmp/unused.pkl"),
            description="donor model",
            feature_names=["donor_id", "age", "bmi", "sex", "blood_type"],
            defaults={
                "prediction_source": "feast_online",
                "feast": {
                    "feature_service": "donor_propensity_service",
                    "entity_keys": ["donor_id"],
                    "allow_direct_input": True,
                    "allow_feature_overrides": True,
                },
            },
        )

        with TemporaryDirectory() as tmpdir:
            feature_path = Path(tmpdir) / "donor_propensity_features.parquet"
            pd.DataFrame(
                [
                    {
                        "donor_id": "D001",
                        "event_timestamp": "2026-01-01T00:00:00Z",
                        "age": 35,
                        "bmi": 24.0,
                        "sex": "M",
                        "blood_type": "O+",
                    },
                    {
                        "donor_id": "D002",
                        "event_timestamp": "2026-01-02T00:00:00Z",
                        "age": 35,
                        "bmi": 24.0,
                        "sex": "F",
                        "blood_type": "A+",
                    },
                ]
            ).to_parquet(feature_path, index=False)

            with (
                patch(
                    "ml.core.feature_store._lookup_postgres_feature_match_context",
                    return_value={
                        "metadata": {
                            "provider": "postgres",
                            "status": "unavailable",
                        }
                    },
                ),
                patch(
                    "ml.core.feature_store._offline_feature_paths",
                    return_value=[feature_path],
                ),
            ):
                resolved = resolve_prediction_features(
                    runtime,
                    {"age": 35, "bmi": 24},
                )

        self.assertEqual(resolved.metadata["source"], "feature_completion")
        self.assertEqual(resolved.features["age"], 35)
        self.assertEqual(resolved.features["bmi"], 24)
        self.assertEqual(resolved.features["sex"], "F")
        self.assertEqual(resolved.features["blood_type"], "A+")
        self.assertNotIn("donor_id", resolved.features)

    def test_default_features_merge_model_stats_with_offline_stats(self):
        runtime = ModelRuntime(
            model_id="donor_propensity_model",
            aliases=["donor_propensity_model"],
            slug="donor_propensity_model",
            file_path=Path("/tmp/unused.pkl"),
            description="donor model",
            feature_names=["age", "bmi", "snapshot_recency_days"],
        )

        with (
            patch(
                "ml.core.feature_store._model_stats_default_mapping",
                return_value={"age": 35, "bmi": 24.0},
            ),
            patch(
                "ml.core.feature_store._offline_feature_stats_defaults",
                return_value={
                    "metadata": {
                        "provider": "offline_feature_stats",
                        "status": "found",
                        "resolved_feature_names": ["snapshot_recency_days"],
                    },
                    "features": {"snapshot_recency_days": 63.0},
                },
            ),
        ):
            resolved = _resolve_model_default_features(
                runtime=runtime,
                expected_features=("age", "bmi", "snapshot_recency_days"),
                feature_service_names=["donor_propensity_service"],
            )

        self.assertEqual(
            resolved["features"],
            {"age": 35, "bmi": 24.0, "snapshot_recency_days": 63.0},
        )
        self.assertEqual(resolved["metadata"]["provider"], "combined_defaults")

    @patch("ml.core.feature_store._get_feature_store")
    def test_resolve_prediction_features_maps_online_values_to_prefixed_signature(
        self, mock_get_store
    ):
        store = MagicMock()
        store.get_feature_service.return_value = object()
        store.get_online_features.return_value.to_dict.return_value = {
            "hospital_id": ["H001"],
            "blood_type": ["A+"],
            "temperature": [20.0],
            "rain_mm": [0.0],
            "holiday": [0],
            "disaster": [0],
            "scheduled_surgeries": [10],
            "trauma_cases": [5],
            "current_inventory": [20],
        }
        mock_get_store.return_value = store

        with (
            patch(
                "ml.core.feature_store._lookup_postgres_entity_context",
                return_value={"metadata": {"provider": "postgres", "status": "not_found"}},
            ),
            patch(
                "ml.core.feature_store._lookup_offline_entity_context",
                return_value={"providers": []},
            ),
        ):
            resolved = resolve_prediction_features(
                self._prefixed_runtime(),
                {"hospital_id": "H001", "blood_type": "A+", "temperature": 21.0},
            )

        self.assertEqual(resolved.metadata["source"], "feast_online")
        self.assertEqual(
            resolved.features["hospital_supply_features_features__temperature"], 21.0
        )
        self.assertEqual(
            resolved.features["hospital_supply_features_features__rain_mm"], 0.0
        )
        self.assertEqual(
            resolved.features[
                "hospital_supply_features_features__scheduled_surgeries"
            ],
            10,
        )

    @patch("ml.core.feature_store._get_feature_store")
    def test_resolve_prediction_features_falls_back_to_compatible_feature_service(
        self, mock_get_store
    ):
        requested_service = MagicMock()
        requested_service.name = "hospital_shortage_service"
        compatible_service = MagicMock()
        compatible_service.name = "hospital_supply_service"
        fallback_service = MagicMock()
        fallback_service.name = "hospital_id_service"
        store = MagicMock()

        def get_feature_service(name):
            if name == "hospital_supply_service":
                return compatible_service
            if name == "hospital_id_service":
                return fallback_service
            return requested_service

        store.get_feature_service.side_effect = get_feature_service

        def get_online_features(*, features, entity_rows):
            result = MagicMock()
            if features is requested_service:
                result.to_dict.return_value = {
                    "hospital_id": ["H001"],
                    "blood_type": ["A+"],
                    "temperature": [None],
                    "rain_mm": [None],
                }
            else:
                result.to_dict.return_value = {
                    "hospital_id": ["H001"],
                    "blood_type": ["A+"],
                    "temperature": [20.0],
                    "rain_mm": [0.0],
                    "holiday": [0],
                    "disaster": [0],
                    "scheduled_surgeries": [10],
                    "trauma_cases": [5],
                    "current_inventory": [20],
                }
            return result

        store.get_online_features.side_effect = get_online_features
        mock_get_store.return_value = store

        with (
            patch(
                "ml.core.feature_store._lookup_postgres_entity_context",
                return_value={"metadata": {"provider": "postgres", "status": "not_found"}},
            ),
            patch(
                "ml.core.feature_store._lookup_offline_entity_context",
                return_value={"providers": []},
            ),
        ):
            resolved = resolve_prediction_features(
                self._prefixed_runtime(),
                {
                    "hospital_id": "H001",
                    "blood_type": "A+",
                    "hospital_supply_features_features__temperature": None,
                },
            )

        self.assertEqual(resolved.metadata["feature_service"], "hospital_supply_service")
        self.assertEqual(
            resolved.metadata["requested_feature_service"], "hospital_shortage_service"
        )
        self.assertEqual(resolved.metadata["overrides"], [])
        self.assertEqual(resolved.features["hospital_supply_features_features__temperature"], 20.0)
        self.assertEqual(store.get_online_features.call_count, 2)
        self.assertIs(
            store.get_online_features.call_args.kwargs["features"], compatible_service
        )

    def test_resolve_prediction_features_enriches_weather_with_coordinates(self):
        store = MagicMock()
        service = MagicMock()
        service.name = "hospital_supply_service"
        store.get_feature_service.return_value = service
        store.get_online_features.return_value.to_dict.return_value = {
            "hospital_id": ["H001"],
            "blood_type": ["A+"],
            "temperature": [8.0],
            "rain_mm": [0.0],
            "holiday": [0],
            "disaster": [0],
            "scheduled_surgeries": [10],
            "trauma_cases": [5],
            "current_inventory": [20],
        }
        today = dt.datetime.now(dt.timezone.utc).date()
        postgres_context = {
            "metadata": {
                "provider": "postgres",
                "enabled": True,
                "exists": True,
                "status": "found",
                "as_of": today.isoformat(),
            },
            "row": {
                "hospital_id": "H001",
                "latitude": 36.75,
                "longitude": 3.04,
                "event_timestamp": today.isoformat(),
            },
            "features": {},
        }
        weather_response = {
            "data": {
                "daily": {
                    "temperature_2m_max": [22.0],
                    "temperature_2m_min": [14.0],
                    "precipitation_sum": [3.5],
                }
            },
            "cache": {"hit": False, "key": "weather-cache-key"},
        }

        with (
            patch("ml.core.feature_store._get_feature_store", return_value=store),
            patch(
                "ml.core.feature_store._lookup_postgres_entity_context",
                return_value=postgres_context,
            ),
            patch(
                "ml.core.feature_store._lookup_offline_entity_context",
                return_value={"providers": []},
            ),
            patch(
                "ml.core.feature_store._cached_http_json_request",
                return_value=weather_response,
            ) as mock_weather_request,
        ):
            resolved = resolve_prediction_features(
                self._prefixed_runtime(),
                {"hospital_id": "H001", "blood_type": "A+"},
                forecast_horizon={"timeframe": "short", "days": 3},
            )

        target_date = today + dt.timedelta(days=3)
        self.assertEqual(
            resolved.features["hospital_supply_features_features__temperature"], 18.0
        )
        self.assertEqual(
            resolved.features["hospital_supply_features_features__rain_mm"], 3.5
        )
        weather_metadata = resolved.metadata["external_features"]["providers"][0]
        self.assertEqual(weather_metadata["status"], "found")
        self.assertEqual(weather_metadata["target_date"], target_date.isoformat())
        self.assertEqual(weather_metadata["cache"]["key"], "weather-cache-key")
        mock_weather_request.assert_called_once()
        self.assertEqual(
            mock_weather_request.call_args.kwargs["params"]["start_date"],
            target_date.isoformat(),
        )

    def test_resolve_prediction_features_uses_forecast_base_date_for_external_apis(self):
        store = MagicMock()
        service = MagicMock()
        service.name = "hospital_supply_service"
        store.get_feature_service.return_value = service
        store.get_online_features.return_value.to_dict.return_value = {
            "hospital_id": ["H001"],
            "blood_type": ["A+"],
            "temperature": [8.0],
            "rain_mm": [0.0],
            "holiday": [0],
            "disaster": [0],
            "scheduled_surgeries": [10],
            "trauma_cases": [5],
            "current_inventory": [20],
        }
        postgres_context = {
            "metadata": {
                "provider": "postgres",
                "enabled": True,
                "exists": True,
                "status": "found",
                "as_of": "2022-12-31",
            },
            "row": {
                "hospital_id": "H001",
                "latitude": 36.75,
                "longitude": 3.04,
                "event_timestamp": "2022-12-31",
            },
            "features": {},
        }

        with (
            patch.dict(os.environ, {"PIOS_FORECAST_BASE_DATE": "2026-04-28"}),
            patch("ml.core.feature_store._get_feature_store", return_value=store),
            patch(
                "ml.core.feature_store._lookup_postgres_entity_context",
                return_value=postgres_context,
            ),
            patch(
                "ml.core.feature_store._lookup_offline_entity_context",
                return_value={"providers": []},
            ),
            patch(
                "ml.core.feature_store._cached_http_json_request",
                return_value={
                    "data": {
                        "daily": {
                            "temperature_2m_max": [22.0],
                            "temperature_2m_min": [14.0],
                            "precipitation_sum": [3.5],
                        }
                    },
                    "cache": {"hit": False, "key": "weather-cache-key"},
                },
            ) as mock_weather_request,
        ):
            resolved = resolve_prediction_features(
                self._prefixed_runtime(),
                {"hospital_id": "H001", "blood_type": "A+"},
                forecast_horizon={"timeframe": "short", "days": 3},
            )

        weather_metadata = resolved.metadata["external_features"]["providers"][0]
        self.assertEqual(weather_metadata["status"], "found")
        self.assertEqual(weather_metadata["target_date"], "2026-05-01")
        self.assertEqual(
            mock_weather_request.call_args.kwargs["params"]["start_date"],
            "2026-05-01",
        )

    def test_resolve_prediction_features_updates_calendar_features_from_horizon(self):
        runtime = ModelRuntime(
            model_id="calendar_stockout_model",
            aliases=[],
            slug="calendar_stockout_model",
            file_path=Path("/tmp/unused.pkl"),
            description="calendar-aware stockout model",
            feature_names=["dow", "weekend", "month", "season_sin", "season_cos"],
            defaults={
                "prediction_source": "feast_online",
                "feast": {
                    "feature_service": "calendar_stockout_service",
                    "entity_keys": ["hospital_id", "blood_type"],
                    "allow_direct_input": True,
                    "allow_feature_overrides": True,
                },
            },
        )
        store = MagicMock()
        service = MagicMock()
        service.name = "calendar_stockout_service"
        store.get_feature_service.return_value = service
        store.get_online_features.return_value.to_dict.return_value = {
            "hospital_id": ["H001"],
            "blood_type": ["A+"],
            "dow": [5],
            "weekend": [1],
            "month": [12],
            "season_sin": [0.0],
            "season_cos": [1.0],
        }

        with (
            patch.dict(os.environ, {"PIOS_FORECAST_BASE_DATE": "2026-04-24"}),
            patch("ml.core.feature_store._get_feature_store", return_value=store),
            patch(
                "ml.core.feature_store._lookup_postgres_entity_context",
                return_value={"metadata": {"selected_provider": None}, "features": {}},
            ),
            patch(
                "ml.core.feature_store._lookup_offline_entity_context",
                return_value={"providers": []},
            ),
        ):
            resolved = resolve_prediction_features(
                runtime,
                {"hospital_id": "H001", "blood_type": "A+"},
                forecast_horizon={"timeframe": "long", "days": 30},
            )

        self.assertEqual(resolved.features["dow"], 6)
        self.assertEqual(resolved.features["weekend"], 1)
        self.assertEqual(resolved.features["month"], 5)
        calendar_metadata = next(
            provider
            for provider in resolved.metadata["external_features"]["providers"]
            if provider.get("provider") == "forecast_calendar"
        )
        self.assertEqual(calendar_metadata["status"], "found")
        self.assertEqual(calendar_metadata["target_date"], "2026-05-24")

    def test_resolve_prediction_features_uses_configured_external_provider(self):
        runtime = ModelRuntime(
            model_id="traffic_aware_stockout_model",
            aliases=[],
            slug="traffic_aware_stockout_model",
            file_path=Path("/tmp/unused.pkl"),
            description="traffic-aware stockout model",
            feature_names=["traffic_delay_minutes"],
            defaults={
                "prediction_source": "feast_online",
                "feast": {
                    "feature_service": "traffic_stockout_service",
                    "entity_keys": ["hospital_id"],
                },
            },
        )
        store = MagicMock()
        service = MagicMock()
        service.name = "traffic_stockout_service"
        store.get_feature_service.return_value = service
        store.get_online_features.return_value.to_dict.return_value = {
            "hospital_id": ["H001"],
            "traffic_delay_minutes": [1.0],
        }
        today = dt.datetime.now(dt.timezone.utc).date()
        postgres_context = {
            "metadata": {
                "provider": "postgres",
                "enabled": True,
                "exists": True,
                "status": "found",
                "as_of": today.isoformat(),
            },
            "row": {
                "hospital_id": "H001",
                "latitude": 36.75,
                "longitude": 3.04,
                "event_timestamp": today.isoformat(),
            },
            "features": {},
        }
        provider_config = {
            "traffic": {
                "url": "https://traffic.example/forecast",
                "params": {
                    "lat": "{latitude}",
                    "lon": "{longitude}",
                    "date": "{target_date}",
                    "key": "{env:TRAFFIC_KEY}",
                },
                "response_features": {
                    "traffic_delay_minutes": "routes.0.duration_minutes"
                },
            }
        }

        with (
            patch.dict(
                os.environ,
                {
                    "PIOS_FEATURE_API_PROVIDERS_JSON": json.dumps(provider_config),
                    "TRAFFIC_KEY": "test-key",
                },
            ),
            patch("ml.core.feature_store._get_feature_store", return_value=store),
            patch(
                "ml.core.feature_store._lookup_postgres_entity_context",
                return_value=postgres_context,
            ),
            patch(
                "ml.core.feature_store._lookup_offline_entity_context",
                return_value={"providers": []},
            ),
            patch(
                "ml.core.feature_store._cached_http_json_request",
                return_value={
                    "data": {"routes": [{"duration_minutes": 12.0}]},
                    "cache": {"hit": False, "key": "traffic-cache-key"},
                },
            ) as mock_api_request,
        ):
            resolved = resolve_prediction_features(
                runtime,
                {"hospital_id": "H001"},
                forecast_horizon={"timeframe": "short", "days": 2},
            )

        self.assertEqual(resolved.features["traffic_delay_minutes"], 12.0)
        provider_metadata = next(
            provider
            for provider in resolved.metadata["external_features"]["providers"]
            if provider.get("provider") == "traffic"
        )
        self.assertEqual(provider_metadata["provider"], "traffic")
        self.assertEqual(provider_metadata["status"], "found")
        self.assertEqual(provider_metadata["cache"]["key"], "traffic-cache-key")
        params = mock_api_request.call_args.kwargs["params"]
        self.assertEqual(params["key"], "test-key")
        self.assertEqual(params["date"], (today + dt.timedelta(days=2)).isoformat())

    @patch("ml.core.feature_store._resolve_model_default_features")
    @patch("ml.core.feature_store._get_feature_store")
    def test_resolve_prediction_features_uses_stats_when_online_store_is_empty(
        self, mock_get_store, mock_defaults
    ):
        store = MagicMock()
        store.get_feature_service.return_value = object()
        store.get_online_features.return_value.to_dict.return_value = {
            "hospital_id": ["H001"],
            "blood_type": ["A+"],
            "current_stock_units": [None],
            "usage_today": [None],
        }
        mock_get_store.return_value = store
        mock_defaults.return_value = {
            "features": {
                "blood_product_type": "A+",
                "current_stock_units": 5,
                "usage_today": 3.5,
                "lead_time_days": 2,
                "days_since_last_restock": 1,
                "stockout_count_90d": 0,
                "scheduled_surgeries_next7d": 4,
            },
            "metadata": {
                "provider": "model_stats",
                "status": "found",
            },
        }

        resolved = resolve_prediction_features(
            self._runtime(),
            {"hospital": "H001", "blood_type": "A+"},
        )

        self.assertEqual(resolved.metadata["source"], "feature_completion")
        self.assertEqual(resolved.features["usage_today"], 3.5)
        self.assertEqual(resolved.metadata["default_features"]["status"], "found")
