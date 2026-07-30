from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.core.http_client import http_json_request
from ml.core.model_metadata import runtime_prediction_defaults
from ml.core.model_stats import coerce_model_default_mapping
from ml.core.registry import ModelRuntime

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = BACKEND_ROOT.parent
DEFAULT_FEAST_REPO_CANDIDATES = (
    WORKSPACE_ROOT / "ml-backend" / "feature_repo",
    BACKEND_ROOT / "ml" / "config" / "feature_repo",
)
DEFAULT_FEAST_REPO = DEFAULT_FEAST_REPO_CANDIDATES[0].resolve()
FEATURE_DATA_DIR = WORKSPACE_ROOT / "ml-backend" / "datasets_featues_labels_seperated"
FEATURE_API_CACHE_DIR = BACKEND_ROOT / "ml" / ".cache" / "feature_enrichment"
DEFAULT_SQLITE_REFERENCE_PATH = BACKEND_ROOT / "ml" / ".cache" / "orchestrator_reference.sqlite3"
DEFAULT_FEATURE_DOMAIN_CONFIG_PATH = (
    BACKEND_ROOT / "ml" / "config" / "feature_store_dynamic.json"
)

_feature_store_cache: dict[Path, Any] = {}
_feature_store_lock = threading.Lock()
_domain_config_cache: tuple[Path, float, dict[str, Any]] | None = None


def _domain_config_path() -> Path:
    raw = os.getenv("PIOS_FEATURE_DOMAIN_CONFIG", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_FEATURE_DOMAIN_CONFIG_PATH


def _domain_config() -> dict[str, Any]:
    global _domain_config_cache
    path = _domain_config_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if (
        _domain_config_cache is not None
        and _domain_config_cache[0] == path
        and _domain_config_cache[1] == mtime
    ):
        return _domain_config_cache[2]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Invalid feature domain config at %s", path)
        return {}
    if not isinstance(payload, dict):
        return {}
    _domain_config_cache = (path, mtime, payload)
    return payload


def _configured_mapping(name: str) -> dict[str, Any]:
    value = _domain_config().get(name)
    return dict(value) if isinstance(value, dict) else {}


def _configured_list(name: str) -> list[str]:
    return _as_clean_list(_domain_config().get(name))


def _component_config() -> dict[str, Any]:
    value = _domain_config().get("component")
    return dict(value) if isinstance(value, dict) else {}


def _component_entity_keys() -> list[str]:
    return _as_clean_list(_component_config().get("entity_keys"))


def _component_parent_entity_keys() -> list[str]:
    return _as_clean_list(_component_config().get("parent_entity_keys"))


def _component_all_values() -> set[str]:
    return {
        _normalize_component_value(value)
        for value in _as_clean_list(_component_config().get("all_values"))
    }


@dataclass(frozen=True)
class ResolvedFeaturePayload:
    features: dict[str, Any]
    metadata: dict[str, Any]


def resolve_prediction_features(
    runtime: ModelRuntime,
    request_features: dict[str, Any],
    *,
    forecast_horizon: dict[str, Any] | None = None,
) -> ResolvedFeaturePayload:
    requested = dict(request_features)
    defaults = _merged_prediction_defaults(runtime)
    source = _prediction_source(defaults)
    raw_expected_features = (
        tuple(runtime.feature_names)
        if runtime.feature_names
        else tuple(
            str(name)
            for name in defaults.get("feast", {}).get("expected_features", [])
            if isinstance(name, str)
        )
    )
    expected_features = tuple(
        feature for feature in raw_expected_features if _is_model_input_feature(feature)
    )
    canonical_requested = _canonicalize_request_features(
        requested,
        expected_features=expected_features,
    )
    direct_features = _model_feature_subset(canonical_requested, expected_features)

    direct_payload = ResolvedFeaturePayload(
        features=direct_features,
        metadata={
            "source": "request",
            "requested_feature_names": sorted(requested.keys()),
            "resolved_feature_names": sorted(direct_features.keys()),
        },
    )
    if source != "feast_online":
        missing = tuple(f for f in expected_features if f not in direct_features)
        if missing:
            stats_defaults = _model_stats_default_mapping(runtime, expected_features)
            if stats_defaults:
                filled = {**stats_defaults, **direct_features}
                return ResolvedFeaturePayload(
                    features=filled,
                    metadata={
                        "source": "request_with_model_stats",
                        "requested_feature_names": sorted(requested.keys()),
                        "resolved_feature_names": sorted(filled.keys()),
                        "model_stats_filled": sorted(
                            k for k in stats_defaults if k not in direct_features
                        ),
                    },
                )
        return direct_payload

    feast_config = defaults.get("feast", {})
    if not isinstance(feast_config, dict):
        feast_config = {}

    feature_service = str(feast_config.get("feature_service", "")).strip()
    if not feature_service:
        raise ValueError(
            f"{runtime.model_id} is configured for Feast online inference, "
            "but no feature_service is defined."
        )

    entity_keys = _entity_keys(
        feast_config.get("entity_keys"),
        feature_service=feature_service,
        request_features=requested,
    )
    allow_direct_input = bool(feast_config.get("allow_direct_input", True))
    allow_feature_overrides = bool(feast_config.get("allow_feature_overrides", True))
    fail_open = bool(feast_config.get("fail_open", False))

    has_complete_direct_vector = _has_complete_direct_vector(
        canonical_requested, expected_features, entity_keys
    )

    entity_row = _extract_entity_row(requested, entity_keys)
    if not entity_row:
        if has_complete_direct_vector and allow_direct_input:
            return direct_payload

    repo_path = _repo_path(feast_config)
    fallback_feature_services = _as_clean_list(
        feast_config.get("fallback_feature_services")
        or feast_config.get("fallback_services")
    )
    feature_service_names = [feature_service, *fallback_feature_services]
    if entity_row:
        store = _get_feature_store(repo_path)
        entity_context = _resolve_entity_context(
            entity_row=entity_row,
            expected_features=expected_features,
            feature_service_names=feature_service_names,
        )
    else:
        store = None
        entity_context = _resolve_request_feature_context(
            request_features=canonical_requested,
            expected_features=expected_features,
            feature_service_names=feature_service_names,
        )
    resolved_feature_service = None
    resolved_from_store: dict[str, Any] = {}
    online_errors: list[str] = []
    if store is not None:
        for candidate_service in _resolve_feature_services(
            store,
            feature_service=feature_service,
            entity_keys=entity_keys,
            expected_features=expected_features,
            fallback_feature_services=fallback_feature_services,
        ):
            try:
                online_result = store.get_online_features(
                    features=candidate_service,
                    entity_rows=[entity_row],
                ).to_dict()
            except Exception as exc:
                service_name = getattr(candidate_service, "name", str(candidate_service))
                online_errors.append(f"{service_name}: {exc}")
                logger.warning(
                    "Feast feature service %s failed during online lookup: %s",
                    service_name,
                    exc,
                )
                continue
            candidate_resolved = _flatten_online_row(
                online_result,
                expected_features=expected_features,
                entity_keys=entity_keys,
            )
            if candidate_resolved:
                resolved_feature_service = candidate_service
                resolved_from_store = candidate_resolved
                break

    selected_feature_service = getattr(
        resolved_feature_service, "name", feature_service
    )
    if resolved_feature_service is not None and selected_feature_service != feature_service:
        logger.warning(
            "Feast feature service %s returned no usable features; using %s instead.",
            feature_service,
            selected_feature_service,
        )

    if not resolved_from_store:
        entity_features = entity_context.get("features")
        if isinstance(entity_features, dict) and entity_features:
            resolved_from_store = {
                key: value
                for key, value in entity_features.items()
                if _has_feature_value(value)
                and (not expected_features or key in expected_features)
            }
            selected_feature_service = f"{feature_service}:entity_context"
        elif allow_direct_input and fail_open:
            return direct_payload
        elif entity_row:
            error_suffix = (
                f" Online lookup errors: {'; '.join(online_errors)}"
                if online_errors
                else ""
            )
            logger.warning(
                "Feast online store returned no materialized features for %s and entity %s.%s",
                runtime.model_id,
                entity_row,
                error_suffix,
            )

    resolved = dict(resolved_from_store)
    entity_features = entity_context.get("features")
    entity_metadata = entity_context.get("metadata", {})
    if (
        isinstance(entity_features, dict)
        and isinstance(entity_metadata, dict)
        and entity_metadata.get("selected_provider") in {"postgres", "sqlite"}
    ):
        resolved.update(
            {
                key: value
                for key, value in entity_features.items()
                if _has_feature_value(value) and (not expected_features or key in expected_features)
            }
        )

    external_context = _resolve_external_feature_context(
        entity_row=entity_row,
        entity_context=entity_context,
        expected_features=expected_features,
        base_features=resolved,
        forecast_horizon=forecast_horizon,
    )
    external_features = external_context.get("features")
    if isinstance(external_features, dict):
        resolved.update(
            {
                key: value
                for key, value in external_features.items()
                if _has_feature_value(value) and (not expected_features or key in expected_features)
            }
        )

    default_context = _resolve_model_default_features(
        runtime=runtime,
        expected_features=expected_features,
        feature_service_names=feature_service_names,
    )
    default_features = default_context.get("features")
    if isinstance(default_features, dict):
        for key, value in default_features.items():
            if _has_feature_value(value) and (not expected_features or key in expected_features):
                resolved.setdefault(key, value)

    if allow_feature_overrides:
        resolved.update(
            {
                key: value
                for key, value in direct_features.items()
                if _has_feature_value(value)
            }
        )
    else:
        for key, value in direct_features.items():
            if not _has_feature_value(value):
                continue
            resolved.setdefault(key, value)

    resolved = _model_feature_subset(resolved, expected_features)
    missing_features = [
        feature
        for feature in expected_features
        if feature not in resolved or not _has_feature_value(resolved.get(feature))
    ]
    if expected_features and missing_features:
        raise ValueError(
            f"{runtime.model_id} could not resolve required model features "
            f"{missing_features}. Provide more features or configure model stats defaults."
        )

    override_keys = sorted(
        key
        for key, value in canonical_requested.items()
        if key not in entity_row and key in resolved_from_store
        and _has_feature_value(value)
    )
    metadata_source = (
        "feast_online"
        if resolved_from_store and entity_row and not str(selected_feature_service).endswith(":entity_context")
        else "feature_completion"
    )

    return ResolvedFeaturePayload(
        features=resolved,
        metadata={
            "source": metadata_source,
            "requested_feature_names": sorted(requested.keys()),
            "resolved_feature_names": sorted(resolved.keys()),
            "feature_service": selected_feature_service,
            "requested_feature_service": feature_service,
            "repo_path": str(repo_path),
            "entity_row": entity_row,
            "entity_keys": sorted(entity_keys),
            "entity_lookup": entity_context.get("metadata", {}),
            "external_features": external_context.get("metadata", {}),
            "default_features": default_context.get("metadata", {}),
            "missing_features": missing_features,
            "overrides": override_keys,
        },
    )


def resolve_component_prediction_features(
    runtime: ModelRuntime,
    request_features: dict[str, Any],
    *,
    forecast_horizon: dict[str, Any] | None = None,
    component_key: str = "",
) -> list[tuple[str, ResolvedFeaturePayload]]:
    requested = dict(request_features or {})
    component_key = component_key or next(iter(_component_entity_keys()), "")
    if not component_key:
        return []
    defaults = _merged_prediction_defaults(runtime)
    feast_config = defaults.get("feast")
    if not isinstance(feast_config, dict):
        return []
    entity_keys = _entity_keys(
        feast_config.get("entity_keys"),
        feature_service=str(feast_config.get("feature_service") or ""),
        request_features=requested,
    )
    if component_key not in entity_keys:
        entity_keys = (*entity_keys, component_key)
    if _value_for_entity_key(requested, component_key) not in {None, ""}:
        resolved = resolve_prediction_features(
            runtime,
            requested,
            forecast_horizon=forecast_horizon,
        )
        return [(str(_value_for_entity_key(requested, component_key)), resolved)]

    partial_entity_row: dict[str, Any] = {}
    for key in entity_keys:
        if key == component_key:
            continue
        value = _value_for_entity_key(requested, key)
        if value not in {None, ""}:
            partial_entity_row[key] = str(value).strip()
    if not partial_entity_row:
        return []

    feature_service = str(feast_config.get("feature_service") or "").strip()
    fallback_feature_services = _as_clean_list(
        feast_config.get("fallback_feature_services")
        or feast_config.get("fallback_services")
    )
    feature_service_names = [feature_service, *fallback_feature_services]
    components = _available_entity_key_values(
        partial_entity_row,
        component_key,
        feature_service_names=feature_service_names,
    )
    resolved_payloads: list[tuple[str, ResolvedFeaturePayload]] = []
    for component in components:
        component_request = dict(requested)
        component_request[component_key] = component
        try:
            resolved_payloads.append(
                (
                    component,
                    resolve_prediction_features(
                        runtime,
                        component_request,
                        forecast_horizon=forecast_horizon,
                    ),
                )
            )
        except Exception as exc:
            logger.warning(
                "Component feature resolution failed for %s=%s on %s: %s",
                component_key,
                component,
                runtime.model_id,
                exc,
            )
    return resolved_payloads


def _prediction_source(defaults: dict[str, Any]) -> str:
    for key in ("prediction_source", "feature_source", "inference_source"):
        value = defaults.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _merged_prediction_defaults(runtime: ModelRuntime) -> dict[str, Any]:
    base_defaults = runtime_prediction_defaults(runtime.model_id)
    merged = _deep_merge({}, base_defaults)
    if isinstance(runtime.defaults, dict):
        merged = _deep_merge(merged, runtime.defaults)
    base_feast = base_defaults.get("feast")
    merged_feast = merged.get("feast")
    if isinstance(base_feast, dict) and isinstance(merged_feast, dict):
        base_entity_keys = _as_clean_list(base_feast.get("entity_keys"))
        merged_entity_keys = _as_clean_list(merged_feast.get("entity_keys"))
        if base_entity_keys:
            merged_feast["entity_keys"] = [
                *merged_entity_keys,
                *[key for key in base_entity_keys if key not in merged_entity_keys],
            ]
    return merged


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _repo_path(feast_config: dict[str, Any]) -> Path:
    configured = feast_config.get("repo_path") or os.getenv("PIOS_FEAST_REPO")
    if configured:
        return Path(str(configured)).expanduser().resolve()
    for candidate in DEFAULT_FEAST_REPO_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_FEAST_REPO


def _resolve_feature_services(
    store: Any,
    *,
    feature_service: str,
    entity_keys: tuple[str, ...],
    expected_features: tuple[str, ...],
    fallback_feature_services: list[str] | None = None,
) -> list[Any]:
    candidates = [feature_service]
    candidates.extend(fallback_feature_services or [])
    candidates.extend(_feature_service_candidates_from_expected(expected_features))
    candidates.extend(f"{entity_key}_service" for entity_key in entity_keys)

    seen: set[str] = set()
    unique_candidates = []
    for candidate in candidates:
        clean_candidate = str(candidate).strip()
        if clean_candidate and clean_candidate not in seen:
            seen.add(clean_candidate)
            unique_candidates.append(clean_candidate)

    first_error: Exception | None = None
    services: list[Any] = []
    for candidate in unique_candidates:
        try:
            service = store.get_feature_service(candidate)
        except Exception as exc:
            if first_error is None:
                first_error = exc
            continue
        services.append(service)

    if services:
        return services
    if first_error is not None:
        raise first_error
    raise ValueError("No Feast feature service candidates were available.")


def _feature_service_candidates_from_expected(
    expected_features: tuple[str, ...],
) -> list[str]:
    candidates: list[str] = []
    for feature_name in expected_features:
        if "__" not in feature_name:
            continue
        service_base = feature_name.split("__", 1)[0]
        while service_base.endswith("_features"):
            service_base = service_base[: -len("_features")]
        if service_base:
            candidates.append(f"{service_base}_service")
    return candidates


def _as_clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _resolve_entity_context(
    *,
    entity_row: dict[str, Any],
    expected_features: tuple[str, ...],
    feature_service_names: list[str],
) -> dict[str, Any]:
    providers: list[dict[str, Any]] = []
    features: dict[str, Any] = {}
    selected_row: dict[str, Any] | None = None
    selected_provider = ""

    postgres_context = _lookup_postgres_entity_context(entity_row, expected_features)
    providers.append(postgres_context["metadata"])
    if postgres_context.get("row"):
        selected_row = postgres_context["row"]
        selected_provider = "postgres"
        features.update(postgres_context.get("features", {}))

    postgres_status = str(postgres_context.get("metadata", {}).get("status") or "")
    component_key = _component_entity_key_in(entity_row)
    should_try_sqlite = postgres_status in {"unavailable", "query_failed"} or (
        postgres_status == "not_found"
        and component_key is not None
        and _is_component_value(entity_row.get(component_key))
        and any(key != component_key for key in entity_row)
    )
    if should_try_sqlite:
        sqlite_context = _lookup_sqlite_entity_context(entity_row, expected_features)
        providers.append(sqlite_context["metadata"])
        if sqlite_context.get("row"):
            sqlite_features = sqlite_context.get("features", {})
            if isinstance(sqlite_features, dict):
                for key, value in sqlite_features.items():
                    if _has_feature_value(value):
                        features.setdefault(key, value)
            if not selected_row:
                selected_row = sqlite_context["row"]
                selected_provider = "sqlite"

    offline_context = _lookup_offline_entity_context(
        entity_row,
        expected_features,
        feature_service_names=feature_service_names,
    )
    providers.extend(offline_context.get("providers", []))
    offline_features = offline_context.get("features")
    if isinstance(offline_features, dict):
        for key, value in offline_features.items():
            if _has_feature_value(value):
                features.setdefault(key, value)
    if not selected_row and offline_context.get("row"):
        selected_row = offline_context["row"]
        selected_provider = str(offline_context.get("provider") or "offline_features")

    return {
        "features": features,
        "row": selected_row or {},
        "metadata": {
            "entity_row": entity_row,
            "exists": bool(selected_row),
            "selected_provider": selected_provider or None,
            "as_of": _row_as_of(selected_row or {}),
            "providers": providers,
        },
    }


def _resolve_request_feature_context(
    *,
    request_features: dict[str, Any],
    expected_features: tuple[str, ...],
    feature_service_names: list[str],
) -> dict[str, Any]:
    providers: list[dict[str, Any]] = []
    features: dict[str, Any] = {}
    selected_row: dict[str, Any] | None = None
    selected_provider = ""

    postgres_context = _lookup_postgres_feature_match_context(
        request_features,
        expected_features,
    )
    providers.append(postgres_context["metadata"])
    if postgres_context.get("row"):
        selected_row = postgres_context["row"]
        selected_provider = "postgres"
        features.update(postgres_context.get("features", {}))

    postgres_status = str(postgres_context.get("metadata", {}).get("status") or "")
    if postgres_status in {"unavailable", "query_failed"}:
        sqlite_context = _lookup_sqlite_feature_match_context(
            request_features,
            expected_features,
        )
        providers.append(sqlite_context["metadata"])
        if sqlite_context.get("row"):
            sqlite_features = sqlite_context.get("features", {})
            if isinstance(sqlite_features, dict):
                for key, value in sqlite_features.items():
                    if _has_feature_value(value):
                        features.setdefault(key, value)
            if not selected_row:
                selected_row = sqlite_context["row"]
                selected_provider = "sqlite"

    offline_context = _lookup_offline_feature_match_context(
        request_features,
        expected_features,
        feature_service_names=feature_service_names,
    )
    providers.extend(offline_context.get("providers", []))
    offline_features = offline_context.get("features")
    if isinstance(offline_features, dict):
        for key, value in offline_features.items():
            if _has_feature_value(value):
                features.setdefault(key, value)
    if not selected_row and offline_context.get("row"):
        selected_row = offline_context["row"]
        selected_provider = str(offline_context.get("provider") or "offline_features")

    return {
        "features": features,
        "row": selected_row or {},
        "metadata": {
            "entity_row": {},
            "exists": bool(selected_row),
            "selected_provider": selected_provider or None,
            "as_of": _row_as_of(selected_row or {}),
            "providers": providers,
            "lookup": "request_feature_match",
        },
    }


def _lookup_postgres_entity_context(
    entity_row: dict[str, Any],
    expected_features: tuple[str, ...],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": "postgres",
        "enabled": _env_truthy("PIOS_FEATURE_DB_LOOKUP_ENABLED", default=True),
        "exists": False,
    }
    if not metadata["enabled"]:
        metadata["status"] = "disabled"
        return {"metadata": metadata}

    table_name = _entity_postgres_table(entity_row)
    if not table_name:
        metadata["status"] = "no_table_mapping"
        return {"metadata": metadata}

    schema_name = os.getenv("PIOS_ORCH_DB_SCHEMA", "public").strip() or "public"
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(
            host=os.getenv("DB_HOST") or os.getenv("PGHOST") or "localhost",
            port=int(os.getenv("DB_PORT") or os.getenv("PGPORT") or "5432"),
            dbname=os.getenv("DB_NAME") or os.getenv("PGDATABASE") or "pios",
            user=os.getenv("DB_USER") or os.getenv("PGUSER") or "admin",
            password=os.getenv("DB_PASSWORD") or os.getenv("PGPASSWORD") or "admin",
            connect_timeout=max(1, int(os.getenv("DB_CONNECT_TIMEOUT", "1"))),
        )
    except Exception as exc:
        metadata.update({"status": "unavailable", "error": str(exc)})
        return {"metadata": metadata}

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                """,
                [schema_name, table_name],
            )
            columns = [str(row["column_name"]) for row in cur.fetchall()]
            if not columns or any(key not in columns for key in entity_row):
                metadata.update(
                    {
                        "status": "unsupported_table",
                        "schema": schema_name,
                        "table": table_name,
                    }
                )
                return {"metadata": metadata}

            order_column = next(
                (candidate for candidate in _order_columns() if candidate in columns),
                None,
            )
            where_clause = " AND ".join(f"{key} = %s" for key in entity_row)
            order_clause = f" ORDER BY {order_column} DESC" if order_column else ""
            cur.execute(
                f'SELECT * FROM "{schema_name}"."{table_name}" WHERE {where_clause}{order_clause} LIMIT 1',
                list(entity_row.values()),
            )
            row = cur.fetchone()
    except Exception as exc:
        metadata.update({"status": "query_failed", "error": str(exc)})
        return {"metadata": metadata}
    finally:
        conn.close()

    if not row:
        metadata.update(
            {
                "status": "not_found",
                "schema": schema_name,
                "table": table_name,
            }
        )
        return {"metadata": metadata}

    row_dict = {str(key): _jsonable(value) for key, value in dict(row).items()}
    features = _feature_values_from_context_row(row_dict, expected_features)
    metadata.update(
        {
            "status": "found",
            "exists": True,
            "schema": schema_name,
            "table": table_name,
            "as_of": _row_as_of(row_dict),
            "resolved_feature_names": sorted(features),
        }
    )
    return {"metadata": metadata, "row": row_dict, "features": features}


def _lookup_postgres_feature_match_context(
    request_features: dict[str, Any],
    expected_features: tuple[str, ...],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": "postgres",
        "enabled": _env_truthy("PIOS_FEATURE_DB_LOOKUP_ENABLED", default=True),
        "exists": False,
        "lookup": "feature_match",
    }
    if not metadata["enabled"]:
        metadata["status"] = "disabled"
        return {"metadata": metadata}

    query_features = _query_feature_values(request_features, expected_features)
    if not query_features:
        metadata["status"] = "no_query_features"
        return {"metadata": metadata}

    table_name = _feature_match_postgres_table(expected_features)
    if not table_name:
        metadata["status"] = "no_table_mapping"
        return {"metadata": metadata}

    schema_name = os.getenv("PIOS_ORCH_DB_SCHEMA", "public").strip() or "public"
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except Exception as exc:
        metadata.update({"status": "unavailable", "error": str(exc)})
        return {"metadata": metadata}

    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST") or os.getenv("PGHOST") or "localhost",
            port=int(os.getenv("DB_PORT") or os.getenv("PGPORT") or "5432"),
            dbname=os.getenv("DB_NAME") or os.getenv("PGDATABASE") or "pios",
            user=os.getenv("DB_USER") or os.getenv("PGUSER") or "admin",
            password=os.getenv("DB_PASSWORD") or os.getenv("PGPASSWORD") or "admin",
            connect_timeout=max(1, int(os.getenv("DB_CONNECT_TIMEOUT", "1"))),
        )
    except Exception as exc:
        metadata.update({"status": "unavailable", "error": str(exc)})
        return {"metadata": metadata}

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                """,
                [schema_name, table_name],
            )
            columns = [str(row["column_name"]) for row in cur.fetchall()]
            filters = _postgres_feature_filters(query_features, columns, expected_features)
            if not filters:
                metadata.update(
                    {
                        "status": "unsupported_filter_columns",
                        "schema": schema_name,
                        "table": table_name,
                    }
                )
                return {"metadata": metadata}
            order_column = next(
                (candidate for candidate in _order_columns() if candidate in columns),
                None,
            )
            id_columns = [
                column
                for column in _order_columns()
                if column in columns and (column == "id" or column.endswith("_id"))
            ]
            where_clause = " AND ".join(f'"{column}" = %s' for column, _ in filters)
            order_parts = []
            if order_column:
                order_parts.append(f'"{order_column}" DESC')
            order_parts.extend(f'"{column}" ASC' for column in id_columns)
            order_clause = f" ORDER BY {', '.join(order_parts)}" if order_parts else ""
            cur.execute(
                f'SELECT * FROM "{schema_name}"."{table_name}" WHERE {where_clause}{order_clause} LIMIT 1',
                [value for _, value in filters],
            )
            row = cur.fetchone()
    except Exception as exc:
        metadata.update({"status": "query_failed", "error": str(exc)})
        return {"metadata": metadata}
    finally:
        conn.close()

    if not row:
        metadata.update(
            {
                "status": "not_found",
                "schema": schema_name,
                "table": table_name,
            }
        )
        return {"metadata": metadata}

    row_dict = {str(key): _jsonable(value) for key, value in dict(row).items()}
    features = _feature_values_from_context_row(row_dict, expected_features)
    metadata.update(
        {
            "status": "found",
            "exists": True,
            "schema": schema_name,
            "table": table_name,
            "as_of": _row_as_of(row_dict),
            "resolved_feature_names": sorted(features),
            "matched_features": sorted(column for column, _ in filters),
        }
    )
    return {"metadata": metadata, "row": row_dict, "features": features}


def _entity_postgres_table(entity_row: dict[str, Any]) -> str:
    mapping_raw = os.getenv("PIOS_FEATURE_ENTITY_TABLES_JSON", "").strip()
    if mapping_raw:
        try:
            mapping = json.loads(mapping_raw)
        except json.JSONDecodeError:
            mapping = {}
        if isinstance(mapping, dict):
            for key in entity_row:
                table_name = str(mapping.get(key) or "").strip()
                if table_name:
                    return table_name

    mappings = _domain_config().get("entity_table_mappings")
    if isinstance(mappings, list):
        entity_keys = set(entity_row)
        for row in mappings:
            if not isinstance(row, dict):
                continue
            entity_key = str(row.get("entity_key") or "").strip()
            if entity_key not in entity_keys:
                continue
            env_name = str(row.get("env") or "").strip()
            default = str(row.get("default") or "").strip()
            table_name = os.getenv(env_name, "").strip() if env_name else ""
            return table_name or default
    return ""


def _feature_match_postgres_table(expected_features: tuple[str, ...]) -> str:
    normalized = {_normalize_feature_name(feature).lower() for feature in expected_features}
    mappings = _domain_config().get("feature_match_table_mappings")
    if isinstance(mappings, list):
        for row in mappings:
            if not isinstance(row, dict):
                continue
            features = {
                _normalize_feature_name(value).lower()
                for value in _as_clean_list(row.get("features"))
            }
            if not normalized & features:
                continue
            env_name = str(row.get("env") or "").strip()
            default = str(row.get("default") or "").strip()
            table_name = os.getenv(env_name, "").strip() if env_name else ""
            return table_name or default
    return ""


def _sqlite_reference_path() -> Path:
    raw = os.getenv("PIOS_ORCH_DB_SQLITE_PATH", "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_SQLITE_REFERENCE_PATH


def _normalize_entity_row(entity_row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in (entity_row or {}).items():
        if value in {None, ""}:
            continue
        clean_key = str(key).strip()
        clean_value = str(value).strip()
        if _is_component_entity_key(clean_key):
            clean_value = _normalize_component_value(clean_value)
        normalized[clean_key] = clean_value
    return normalized


def _sqlite_fetch_entity_row(
    conn: sqlite3.Connection,
    table_name: str,
    entity_row: dict[str, Any],
    columns: list[str],
) -> sqlite3.Row | None:
    where_clause = " AND ".join(f'"{key}" = ?' for key in entity_row)
    order_column = next(
        (
            candidate
            for candidate in _order_columns()
            if candidate in columns
        ),
        None,
    )
    order_clause = f' ORDER BY "{order_column}" DESC' if order_column else ""
    return conn.execute(
        f'SELECT * FROM "{table_name}" WHERE {where_clause}{order_clause} LIMIT 1',
        list(entity_row.values()),
    ).fetchone()


def _normalize_component_value(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    raw_aliases = _component_config().get("aliases")
    if not isinstance(raw_aliases, dict):
        raw_aliases = {}
    aliases = {
        str(key).strip().upper().replace(" ", "_"): str(value).strip().upper()
        for key, value in raw_aliases.items()
        if str(key).strip() and str(value).strip()
    }
    return aliases.get(text, text)


def _is_component_value(value: Any) -> bool:
    values = {
        _normalize_component_value(item)
        for item in _as_clean_list(_component_config().get("values"))
    }
    return _normalize_component_value(value) in values


def _componentized_hospital_row(
    row: dict[str, Any],
    component: str,
) -> dict[str, Any]:
    component = _normalize_component_value(component)
    factors = _component_config().get("factors")
    factor = 1.0
    if isinstance(factors, dict):
        try:
            factor = float(factors.get(component, factor))
        except (TypeError, ValueError):
            factor = 1.0
    result = dict(row)
    component_key = _component_entity_keys()[0] if _component_entity_keys() else ""
    if component_key:
        result[component_key] = component
    for key in _as_clean_list(_component_config().get("scaled_quantity_fields")):
        if key in result and result.get(key) not in {None, ""}:
            try:
                result[key] = int(round(float(result[key]) * factor))
            except (TypeError, ValueError):
                pass
    for key in _as_clean_list(_component_config().get("scaled_count_fields")):
        if key in result and result.get(key) not in {None, ""}:
            try:
                result[key] = max(0, int(round(float(result[key]) * factor)))
            except (TypeError, ValueError):
                pass
    critical_field = str(_component_config().get("critical_stock_field") or "").strip()
    source_field = str(_component_config().get("critical_stock_source_field") or "").strip()
    threshold = _component_config().get("critical_stock_threshold")
    if critical_field and source_field and critical_field in result:
        try:
            result[critical_field] = int(
                float(result.get(source_field, 0)) <= float(threshold)
            )
        except (TypeError, ValueError):
            pass
    return result


def _lookup_sqlite_entity_context(
    entity_row: dict[str, Any],
    expected_features: tuple[str, ...],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": "sqlite",
        "enabled": _env_truthy("PIOS_FEATURE_SQLITE_LOOKUP_ENABLED", default=True),
        "exists": False,
    }
    if not metadata["enabled"]:
        metadata["status"] = "disabled"
        return {"metadata": metadata}

    db_path = _sqlite_reference_path()
    if not db_path.exists():
        metadata.update({"status": "missing_db", "path": str(db_path)})
        return {"metadata": metadata}
    table_name = _entity_postgres_table(entity_row)
    if not table_name:
        metadata["status"] = "no_table_mapping"
        return {"metadata": metadata}

    normalized_entity = _normalize_entity_row(entity_row)
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        columns = [
            str(row["name"])
            for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        ]
        if not columns or any(key not in columns for key in normalized_entity):
            metadata.update({"status": "unsupported_table", "path": str(db_path), "table": table_name})
            return {"metadata": metadata}
        row = _sqlite_fetch_entity_row(conn, table_name, normalized_entity, columns)
        component_key = _component_entity_key_in(normalized_entity)
        if row is None and component_key is not None:
            fallback_entity = dict(normalized_entity)
            requested_component = fallback_entity[component_key]
            for all_value in _component_all_values():
                fallback_entity[component_key] = all_value
                row = _sqlite_fetch_entity_row(conn, table_name, fallback_entity, columns)
                if row is not None:
                    row = _componentized_hospital_row(dict(row), requested_component)
                    break
        if row is None:
            metadata.update({"status": "not_found", "path": str(db_path), "table": table_name})
            return {"metadata": metadata}
        row_dict = {str(key): _jsonable(value) for key, value in dict(row).items()}
    except Exception as exc:
        metadata.update({"status": "query_failed", "path": str(db_path), "error": str(exc)})
        return {"metadata": metadata}
    finally:
        try:
            conn.close()
        except Exception:
            pass

    features = _feature_values_from_context_row(row_dict, expected_features)
    metadata.update(
        {
            "status": "found",
            "exists": True,
            "path": str(db_path),
            "table": table_name,
            "as_of": _row_as_of(row_dict),
            "resolved_feature_names": sorted(features),
        }
    )
    return {"metadata": metadata, "row": row_dict, "features": features}


def _lookup_sqlite_feature_match_context(
    request_features: dict[str, Any],
    expected_features: tuple[str, ...],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": "sqlite",
        "enabled": _env_truthy("PIOS_FEATURE_SQLITE_LOOKUP_ENABLED", default=True),
        "exists": False,
        "lookup": "feature_match",
    }
    if not metadata["enabled"]:
        metadata["status"] = "disabled"
        return {"metadata": metadata}

    query_features = _query_feature_values(request_features, expected_features)
    if not query_features:
        metadata["status"] = "no_query_features"
        return {"metadata": metadata}
    table_name = _feature_match_postgres_table(expected_features)
    db_path = _sqlite_reference_path()
    if not table_name or not db_path.exists():
        metadata.update({"status": "missing_mapping_or_db", "path": str(db_path)})
        return {"metadata": metadata}
    supported_tables = _sqlite_feature_match_tables()
    if supported_tables and table_name not in supported_tables:
        metadata.update({"status": "not_supported_for_table", "table": table_name})
        return {"metadata": metadata}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        columns = [
            str(row["name"])
            for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        ]
        filters = _postgres_feature_filters(query_features, columns, expected_features)
        if not filters:
            metadata.update({"status": "unsupported_filter_columns", "path": str(db_path), "table": table_name})
            return {"metadata": metadata}
        where_clause = " AND ".join(f'"{column}" = ?' for column, _ in filters)
        order_column = next((c for c in _order_columns() if c in columns), None)
        order_clause = f' ORDER BY "{order_column}" DESC' if order_column else ""
        row = conn.execute(
            f'SELECT * FROM "{table_name}" WHERE {where_clause}{order_clause} LIMIT 1',
            [value for _, value in filters],
        ).fetchone()
        if row is None:
            metadata.update({"status": "not_found", "path": str(db_path), "table": table_name})
            return {"metadata": metadata}
        row_dict = {str(key): _jsonable(value) for key, value in dict(row).items()}
    except Exception as exc:
        metadata.update({"status": "query_failed", "path": str(db_path), "error": str(exc)})
        return {"metadata": metadata}
    finally:
        try:
            conn.close()
        except Exception:
            pass
    features = _feature_values_from_context_row(row_dict, expected_features)
    metadata.update(
        {
            "status": "found",
            "exists": True,
            "path": str(db_path),
            "table": table_name,
            "as_of": _row_as_of(row_dict),
            "resolved_feature_names": sorted(features),
            "matched_features": sorted(column for column, _ in filters),
        }
    )
    return {"metadata": metadata, "row": row_dict, "features": features}


def _postgres_feature_filters(
    query_features: dict[str, Any],
    columns: list[str],
    expected_features: tuple[str, ...],
) -> list[tuple[str, Any]]:
    column_lookup = {column.lower(): column for column in columns}
    filters: list[tuple[str, Any]] = []
    for feature, value in query_features.items():
        for candidate in _context_column_candidates(feature, expected_features):
            column = column_lookup.get(candidate.lower())
            if column is not None:
                filters.append((column, value))
                break
    return filters


def _lookup_offline_entity_context(
    entity_row: dict[str, Any],
    expected_features: tuple[str, ...],
    *,
    feature_service_names: list[str],
) -> dict[str, Any]:
    providers: list[dict[str, Any]] = []
    for path in _offline_feature_paths(feature_service_names):
        metadata = {
            "provider": "offline_feature_store",
            "path": str(path),
            "exists": False,
        }
        if not path.exists():
            metadata["status"] = "missing_file"
            providers.append(metadata)
            continue
        try:
            import pandas as pd

            frame = pd.read_parquet(path)
            mask = None
            for key, value in entity_row.items():
                if key not in frame.columns:
                    mask = None
                    break
                current = frame[key].astype(str) == str(value)
                mask = current if mask is None else mask & current
            if mask is None:
                metadata["status"] = "unsupported_entity_key"
                providers.append(metadata)
                continue
            matches = frame.loc[mask].copy()
            if matches.empty:
                metadata["status"] = "not_found"
                providers.append(metadata)
                continue
            order_column = next(
                (
                    column
                    for column in _timestamp_order_columns()
                    if column in matches.columns
                ),
                None,
            )
            if order_column:
                matches[order_column] = pd.to_datetime(
                    matches[order_column],
                    errors="coerce",
                    utc=True,
                )
                matches = matches.sort_values(order_column)
            row_dict = {
                str(key): _jsonable(value)
                for key, value in matches.iloc[-1].to_dict().items()
            }
        except Exception as exc:
            metadata.update({"status": "read_failed", "error": str(exc)})
            providers.append(metadata)
            continue

        features = _feature_values_from_context_row(row_dict, expected_features)
        metadata.update(
            {
                "status": "found",
                "exists": True,
                "as_of": _row_as_of(row_dict),
                "resolved_feature_names": sorted(features),
            }
        )
        providers.append(metadata)
        return {
            "providers": providers,
            "provider": "offline_feature_store",
            "row": row_dict,
            "features": features,
        }
    return {"providers": providers}


def _lookup_offline_feature_match_context(
    request_features: dict[str, Any],
    expected_features: tuple[str, ...],
    *,
    feature_service_names: list[str],
) -> dict[str, Any]:
    providers: list[dict[str, Any]] = []
    query_features = _query_feature_values(request_features, expected_features)
    if not query_features:
        return {
            "providers": [
                {
                    "provider": "offline_feature_store",
                    "exists": False,
                    "status": "no_query_features",
                    "lookup": "feature_match",
                }
            ]
        }

    for path in _offline_feature_paths(feature_service_names):
        metadata = {
            "provider": "offline_feature_store",
            "path": str(path),
            "exists": False,
            "lookup": "feature_match",
        }
        if not path.exists():
            metadata["status"] = "missing_file"
            providers.append(metadata)
            continue
        try:
            import pandas as pd

            frame = pd.read_parquet(path)
            mask = None
            matched_columns: list[str] = []
            for feature, value in query_features.items():
                column = _matching_context_column(
                    feature,
                    frame.columns,
                    expected_features,
                )
                if column is None:
                    continue
                current = _series_matches_value(frame[column], value)
                mask = current if mask is None else mask & current
                matched_columns.append(column)
            if mask is None or not matched_columns:
                metadata["status"] = "unsupported_filter_columns"
                providers.append(metadata)
                continue
            matches = frame.loc[mask].copy()
            if matches.empty:
                metadata["status"] = "not_found"
                metadata["matched_features"] = matched_columns
                providers.append(metadata)
                continue
            order_columns = [
                column for column in _order_columns() if column in matches.columns
            ]
            for column in _timestamp_order_columns():
                if column in matches.columns:
                    matches[column] = pd.to_datetime(
                        matches[column],
                        errors="coerce",
                        utc=True,
                    )
            if order_columns:
                ascending = [
                    False if column in set(_timestamp_order_columns()) else True
                    for column in order_columns
                ]
                matches = matches.sort_values(order_columns, ascending=ascending)
            row_dict = {
                str(key): _jsonable(value)
                for key, value in matches.iloc[0].to_dict().items()
            }
        except Exception as exc:
            metadata.update({"status": "read_failed", "error": str(exc)})
            providers.append(metadata)
            continue

        features = _feature_values_from_context_row(row_dict, expected_features)
        metadata.update(
            {
                "status": "found",
                "exists": True,
                "as_of": _row_as_of(row_dict),
                "resolved_feature_names": sorted(features),
                "matched_features": matched_columns,
            }
        )
        providers.append(metadata)
        return {
            "providers": providers,
            "provider": "offline_feature_store",
            "row": row_dict,
            "features": features,
        }
    return {"providers": providers}


def _offline_feature_paths(feature_service_names: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for service_name in feature_service_names:
        service_base = str(service_name or "").strip()
        if service_base.endswith("_service"):
            service_base = service_base[: -len("_service")]
        if not service_base:
            continue
        candidate = (FEATURE_DATA_DIR / f"{service_base}_features.parquet").resolve()
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            paths.append(candidate)
    return paths


def _feature_values_from_context_row(
    row: dict[str, Any],
    expected_features: tuple[str, ...],
) -> dict[str, Any]:
    if not row:
        return {}
    candidates: dict[str, Any] = {}
    for raw_key, value in row.items():
        key = _normalize_feature_name(str(raw_key))
        if _is_target_like_context_column(key):
            continue
        candidates[key] = value
        alias = _configured_mapping("context_field_aliases").get(key)
        if alias:
            candidates[alias] = value
    canonical = _canonicalize_request_features(candidates, expected_features=expected_features)
    return {
        key: value
        for key, value in canonical.items()
        if _has_feature_value(value) and (not expected_features or key in expected_features)
    }


def _resolve_model_default_features(
    *,
    runtime: ModelRuntime,
    expected_features: tuple[str, ...],
    feature_service_names: list[str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": "model_stats",
        "status": "not_required" if not expected_features else "missing",
    }
    if not expected_features:
        return {"metadata": metadata, "features": {}}

    features: dict[str, Any] = {}
    providers: list[dict[str, Any]] = []

    stats = _model_stats_default_mapping(runtime, expected_features)
    if stats:
        features.update(stats)
        providers.append(
            {
                "provider": "model_stats",
                "status": "found",
                "resolved_feature_names": sorted(stats),
            }
        )

    remaining_features = tuple(
        feature for feature in expected_features if feature not in features
    )
    offline_stats = _offline_feature_stats_defaults(
        remaining_features,
        feature_service_names=feature_service_names,
    ) if remaining_features else {"metadata": {}, "features": {}}
    offline_features = offline_stats.get("features")
    if isinstance(offline_features, dict) and offline_features:
        features.update(offline_features)
        offline_metadata = offline_stats.get("metadata")
        if isinstance(offline_metadata, dict):
            providers.append(offline_metadata)

    if features:
        metadata.update(
            {
                "status": "found",
                "resolved_feature_names": sorted(features),
                "providers": providers,
            }
        )
        provider_names = {
            str(provider.get("provider") or "")
            for provider in providers
            if isinstance(provider, dict)
        }
        if len(provider_names) == 1 and provider_names:
            metadata["provider"] = next(iter(provider_names))
        else:
            metadata["provider"] = "combined_defaults"
        return {"metadata": metadata, "features": features}

    if offline_stats.get("metadata"):
        return offline_stats
    return {"metadata": metadata, "features": {}}


def _model_stats_default_mapping(
    runtime: ModelRuntime,
    expected_features: tuple[str, ...],
) -> dict[str, Any]:
    clean_id = _clean_model_id(runtime.model_id)
    try:
        from ml.models import ModelStats

        stats_row = ModelStats.objects.filter(model_id=clean_id).first()
    except Exception as exc:
        logger.debug("ModelStats lookup failed for %s: %s", clean_id, exc)
        return {}
    if stats_row is None:
        return {}
    return _model_feature_subset(
        coerce_model_default_mapping(
            stats_row.stats,
            feature_names=list(expected_features),
        ),
        expected_features,
    )


def _offline_feature_stats_defaults(
    expected_features: tuple[str, ...],
    *,
    feature_service_names: list[str],
) -> dict[str, Any]:
    providers: list[dict[str, Any]] = []
    for path in _offline_feature_paths(feature_service_names):
        metadata = {
            "provider": "offline_feature_stats",
            "path": str(path),
            "exists": False,
        }
        if not path.exists():
            metadata["status"] = "missing_file"
            providers.append(metadata)
            continue
        try:
            import pandas as pd

            frame = pd.read_parquet(path)
            defaults: dict[str, Any] = {}
            for feature in expected_features:
                column = _matching_context_column(feature, frame.columns, expected_features)
                if column is None:
                    continue
                value = _default_value_from_series(frame[column])
                if value is not None:
                    defaults[feature] = value
        except Exception as exc:
            metadata.update({"status": "read_failed", "error": str(exc)})
            providers.append(metadata)
            continue
        if defaults:
            metadata.update(
                {
                    "status": "found",
                    "exists": True,
                    "resolved_feature_names": sorted(defaults),
                }
            )
            providers.append(metadata)
            return {
                "metadata": {
                    "provider": "offline_feature_stats",
                    "status": "found",
                    "providers": providers,
                    "resolved_feature_names": sorted(defaults),
                },
                "features": defaults,
            }
        metadata["status"] = "no_matching_features"
        providers.append(metadata)
    return {
        "metadata": {
            "provider": "offline_feature_stats",
            "status": "missing",
            "providers": providers,
        },
        "features": {},
    }


def _default_value_from_series(series: Any) -> Any:
    try:
        import pandas as pd

        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                return None
            return _jsonable(numeric.median())
        text = series.dropna().astype(str).str.strip()
        text = text[text != ""]
        if text.empty:
            return None
        return str(text.mode().iloc[0] if not text.mode().empty else text.iloc[0])
    except Exception:
        return None


def _is_target_like_context_column(column: str) -> bool:
    lowered = str(column or "").strip().lower()
    return (
        lowered.startswith("days_until_")
        or lowered in {"blood_shortage", "stockout_next_day", "heavy_demand_next_day"}
        or lowered.endswith("_target")
        or lowered.endswith("_label")
        or "_next_day" in lowered
        or lowered.startswith("shortage_risk_next_")
    )


def _resolve_external_feature_context(
    *,
    entity_row: dict[str, Any],
    entity_context: dict[str, Any],
    expected_features: tuple[str, ...],
    base_features: dict[str, Any],
    forecast_horizon: dict[str, Any] | None,
) -> dict[str, Any]:
    weather_context = _resolve_weather_features(
        entity_row=entity_row,
        entity_context=entity_context,
        expected_features=expected_features,
        base_features=base_features,
        forecast_horizon=forecast_horizon,
    )
    calendar_context = _resolve_forecast_calendar_features(
        entity_context=entity_context,
        expected_features=expected_features,
        forecast_horizon=forecast_horizon,
    )
    configured_contexts = _resolve_configured_external_feature_contexts(
        entity_row=entity_row,
        entity_context=entity_context,
        expected_features=expected_features,
        forecast_horizon=forecast_horizon,
    )
    features: dict[str, Any] = {}
    features.update(weather_context.get("features", {}))
    features.update(calendar_context.get("features", {}))
    providers = [
        weather_context.get("metadata", {}),
        calendar_context.get("metadata", {}),
    ]
    for context in configured_contexts:
        context_features = context.get("features")
        if isinstance(context_features, dict):
            features.update(context_features)
        context_metadata = context.get("metadata")
        if isinstance(context_metadata, dict):
            providers.append(context_metadata)
    return {
        "features": features,
        "metadata": {
            "providers": providers,
        },
    }


def _resolve_forecast_calendar_features(
    *,
    entity_context: dict[str, Any],
    expected_features: tuple[str, ...],
    forecast_horizon: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"provider": "forecast_calendar"}
    wanted = {_normalize_feature_name(name) for name in expected_features}
    supported = {
        "dow",
        "day_of_week",
        "weekday",
        "weekend",
        "month",
        "season_sin",
        "season_cos",
    }
    if not wanted.intersection(supported):
        metadata["status"] = "not_required"
        return {"metadata": metadata, "features": {}}

    target_date = _weather_target_date(entity_context, forecast_horizon)
    if target_date is None:
        metadata["status"] = "missing_target_date"
        return {"metadata": metadata, "features": {}}

    features: dict[str, Any] = {}
    dow = target_date.weekday()
    month = target_date.month
    values = {
        "dow": dow,
        "day_of_week": dow,
        "weekday": dow,
        "weekend": int(dow >= 5),
        "month": month,
        "season_sin": float(math.sin(2.0 * math.pi * month / 12.0)),
        "season_cos": float(math.cos(2.0 * math.pi * month / 12.0)),
    }
    for name, value in values.items():
        canonical = _canonical_expected_name(name, expected_features)
        if canonical in expected_features:
            features[canonical] = value

    metadata.update(
        {
            "status": "found" if features else "no_matching_features",
            "target_date": target_date.isoformat(),
            "resolved_feature_names": sorted(features),
        }
    )
    return {"metadata": metadata, "features": features}


def _resolve_weather_features(
    *,
    entity_row: dict[str, Any],
    entity_context: dict[str, Any],
    expected_features: tuple[str, ...],
    base_features: dict[str, Any],
    forecast_horizon: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": "open_meteo",
        "enabled": _env_truthy("PIOS_WEATHER_API_ENABLED", default=True),
    }
    wanted = {
        _normalize_feature_name(name)
        for name in expected_features
        if _normalize_feature_name(name) in {"temperature", "rain_mm"}
    }
    if not wanted:
        metadata["status"] = "not_required"
        return {"metadata": metadata}
    if not metadata["enabled"]:
        metadata["status"] = "disabled"
        return {"metadata": metadata}

    coordinates = _resolve_entity_coordinates(entity_row, entity_context)
    if coordinates is None:
        metadata["status"] = "missing_coordinates"
        return {"metadata": metadata}

    target_date = _weather_target_date(entity_context, forecast_horizon)
    if target_date is None:
        metadata["status"] = "missing_target_date"
        return {"metadata": metadata}

    today = _parse_date(os.getenv("PIOS_FORECAST_BASE_DATE"))
    if today is None:
        today = dt.datetime.now(dt.timezone.utc).date()
    if target_date < today or target_date > today + dt.timedelta(days=16):
        metadata.update(
            {
                "status": "outside_forecast_window",
                "target_date": target_date.isoformat(),
            }
        )
        return {"metadata": metadata}

    url = os.getenv(
        "PIOS_WEATHER_API_URL",
        "https://api.open-meteo.com/v1/forecast",
    ).strip()
    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "UTC",
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
    }
    try:
        response = _cached_http_json_request(
            provider="open_meteo",
            url=url,
            params=params,
            timeout_s=float(os.getenv("PIOS_WEATHER_API_TIMEOUT_S", "5")),
        )
    except Exception as exc:
        metadata.update({"status": "request_failed", "error": str(exc)})
        return {"metadata": metadata}

    daily = response.get("data", {}).get("daily", {})
    features: dict[str, Any] = {}
    if "temperature" in wanted:
        max_values = daily.get("temperature_2m_max")
        min_values = daily.get("temperature_2m_min")
        if isinstance(max_values, list) and isinstance(min_values, list) and max_values and min_values:
            features[_canonical_expected_name("temperature", expected_features)] = (
                float(max_values[0]) + float(min_values[0])
            ) / 2.0
    if "rain_mm" in wanted:
        precipitation = daily.get("precipitation_sum")
        if isinstance(precipitation, list) and precipitation:
            features[_canonical_expected_name("rain_mm", expected_features)] = float(
                precipitation[0]
            )

    metadata.update(
        {
            "status": "found" if features else "empty_response",
            "target_date": target_date.isoformat(),
            "coordinates": coordinates,
            "resolved_feature_names": sorted(features),
            "cache": response.get("cache", {}),
        }
    )
    return {"metadata": metadata, "features": features}


def _canonical_expected_name(name: str, expected_features: tuple[str, ...]) -> str:
    aliases = _expected_feature_aliases(expected_features)
    return aliases.get(name, name)


def _resolve_entity_coordinates(
    entity_row: dict[str, Any],
    entity_context: dict[str, Any],
) -> dict[str, float] | None:
    row = entity_context.get("row")
    if isinstance(row, dict):
        coordinates = _coordinates_from_row(row)
        if coordinates is not None:
            return coordinates

    mapping_raw = os.getenv("PIOS_ENTITY_COORDS_JSON", "").strip()
    if not mapping_raw:
        return None
    try:
        mapping = json.loads(mapping_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(mapping, dict):
        return None
    for value in entity_row.values():
        row_value = mapping.get(str(value))
        if isinstance(row_value, dict):
            return _coordinates_from_row(row_value)
    return None


def _coordinates_from_row(row: dict[str, Any]) -> dict[str, float] | None:
    lat = row.get("latitude", row.get("lat"))
    lon = row.get("longitude", row.get("lon", row.get("lng")))
    try:
        latitude = float(lat)
        longitude = float(lon)
    except (TypeError, ValueError):
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    return {"latitude": latitude, "longitude": longitude}


def _weather_target_date(
    entity_context: dict[str, Any],
    forecast_horizon: dict[str, Any] | None,
) -> dt.date | None:
    base_date = _parse_date(os.getenv("PIOS_FORECAST_BASE_DATE"))
    if base_date is None:
        base_date = dt.datetime.now(dt.timezone.utc).date()
    days = None
    if isinstance(forecast_horizon, dict):
        try:
            days = float(forecast_horizon.get("days"))
        except (TypeError, ValueError):
            days = None
    return base_date + dt.timedelta(days=max(0, int(round(days or 0))))


def _cached_http_json_request(
    *,
    provider: str,
    url: str,
    params: dict[str, Any],
    timeout_s: float,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    cache_dir = Path(
        os.getenv("PIOS_FEATURE_API_CACHE_DIR", str(FEATURE_API_CACHE_DIR))
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_payload = json.dumps(
        {
            "provider": provider,
            "method": method,
            "url": url,
            "params": params,
            "body": body or {},
            "headers": headers or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    cache_key = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{cache_key}.json"
    refresh = _env_truthy("PIOS_FEATURE_API_CACHE_REFRESH", default=False)
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["cache"] = {"hit": True, "path": str(cache_path), "key": cache_key}
        return payload

    payload = http_json_request(
        method=method,
        url=url,
        params=params,
        body=body,
        headers=headers,
        timeout_s=timeout_s,
    )
    payload["cache"] = {"hit": False, "path": str(cache_path), "key": cache_key}
    cache_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def _resolve_configured_external_feature_contexts(
    *,
    entity_row: dict[str, Any],
    entity_context: dict[str, Any],
    expected_features: tuple[str, ...],
    forecast_horizon: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    providers = _configured_external_providers()
    if not providers:
        return []

    row = entity_context.get("row") if isinstance(entity_context, dict) else {}
    if not isinstance(row, dict):
        row = {}
    coordinates = _resolve_entity_coordinates(entity_row, entity_context)
    target_date = _weather_target_date(entity_context, forecast_horizon)
    context_values: dict[str, Any] = {
        **{str(key): value for key, value in row.items()},
        **{str(key): value for key, value in entity_row.items()},
    }
    if coordinates is not None:
        context_values.update(coordinates)
    if target_date is not None:
        context_values["target_date"] = target_date.isoformat()

    contexts: list[dict[str, Any]] = []
    for provider_name, config in providers.items():
        contexts.append(
            _resolve_configured_external_provider(
                provider_name=provider_name,
                config=config,
                context_values=context_values,
                expected_features=expected_features,
            )
        )
    return contexts


def _configured_external_providers() -> dict[str, dict[str, Any]]:
    raw = os.getenv("PIOS_FEATURE_API_PROVIDERS_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid PIOS_FEATURE_API_PROVIDERS_JSON; ignoring providers.")
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(name): config
        for name, config in parsed.items()
        if str(name).strip() and isinstance(config, dict)
    }


def _resolve_configured_external_provider(
    *,
    provider_name: str,
    config: dict[str, Any],
    context_values: dict[str, Any],
    expected_features: tuple[str, ...],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": provider_name,
        "enabled": bool(config.get("enabled", True)),
        "source": "configured_api",
    }
    if not metadata["enabled"]:
        metadata["status"] = "disabled"
        return {"metadata": metadata}

    feature_mapping = _configured_provider_feature_mapping(config)
    wanted_features = {
        feature_name: path
        for feature_name, path in feature_mapping.items()
        if _canonical_expected_name(feature_name, expected_features) in expected_features
        or not expected_features
    }
    if not wanted_features:
        metadata["status"] = "not_required"
        return {"metadata": metadata}

    url = str(config.get("url") or "").strip()
    if not url:
        metadata["status"] = "missing_url"
        return {"metadata": metadata}

    try:
        rendered_url = str(_render_api_value(url, context_values))
        params = _render_api_mapping(config.get("params"), context_values)
        headers = _render_api_mapping(config.get("headers"), context_values)
        body = _render_api_mapping(config.get("body"), context_values)
    except ValueError as exc:
        metadata.update({"status": "missing_context", "error": str(exc)})
        return {"metadata": metadata}

    try:
        response = _cached_http_json_request(
            provider=provider_name,
            url=rendered_url,
            params=params,
            method=str(config.get("method") or "GET").strip().upper() or "GET",
            body=body or None,
            headers={str(key): str(value) for key, value in headers.items()} or None,
            timeout_s=float(config.get("timeout_s") or 5),
        )
    except Exception as exc:
        metadata.update({"status": "request_failed", "error": str(exc)})
        return {"metadata": metadata}

    payload = response.get("data", {})
    features: dict[str, Any] = {}
    for feature_name, response_path in wanted_features.items():
        value = _json_path(payload, response_path)
        if value is None:
            continue
        canonical_name = _canonical_expected_name(feature_name, expected_features)
        features[canonical_name] = value

    metadata.update(
        {
            "status": "found" if features else "empty_response",
            "resolved_feature_names": sorted(features),
            "cache": response.get("cache", {}),
        }
    )
    return {"metadata": metadata, "features": features}


def _configured_provider_feature_mapping(config: dict[str, Any]) -> dict[str, str]:
    raw_mapping = config.get("response_features") or config.get("features")
    if not isinstance(raw_mapping, dict):
        return {}
    mapping: dict[str, str] = {}
    for feature_name, path in raw_mapping.items():
        clean_feature = str(feature_name or "").strip()
        clean_path = str(path or clean_feature).strip()
        if clean_feature and clean_path:
            mapping[clean_feature] = clean_path
    return mapping


def _render_api_mapping(value: Any, context_values: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    rendered: dict[str, Any] = {}
    for key, raw_value in value.items():
        rendered_value = _render_api_value(raw_value, context_values)
        if rendered_value not in {None, ""}:
            rendered[str(key)] = rendered_value
    return rendered


def _render_api_value(value: Any, context_values: dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value

    def replace(match: Any) -> str:
        expression = str(match.group(1)).strip()
        if expression.startswith("env:"):
            env_name = expression.split(":", 1)[1].strip()
            env_value = os.getenv(env_name)
            if env_value is None:
                raise ValueError(f"Missing environment variable {env_name}")
            return env_value
        if expression not in context_values or context_values[expression] is None:
            raise ValueError(f"Missing API context value {expression}")
        return str(context_values[expression])

    return re.sub(r"\{([^{}]+)\}", replace, value)


def _json_path(payload: Any, path: str) -> Any:
    current = payload
    for part in str(path or "").split("."):
        clean_part = part.strip()
        if not clean_part:
            continue
        if isinstance(current, dict):
            current = current.get(clean_part)
        elif isinstance(current, list):
            try:
                current = current[int(clean_part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if current is None:
            return None
    return current


def _row_as_of(row: dict[str, Any]) -> str | None:
    for key in _timestamp_order_columns():
        value = row.get(key)
        if value not in {None, ""}:
            return str(value)
    return None


def _parse_date(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return dt.date.fromisoformat(text[:10])
        except ValueError:
            return None


def _env_truthy(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _order_columns() -> list[str]:
    return _configured_list("order_columns")


def _sqlite_feature_match_tables() -> set[str]:
    supported_tables = set(_configured_list("sqlite_feature_match_tables"))
    mappings = _domain_config().get("entity_table_mappings")
    if not isinstance(mappings, list):
        return supported_tables
    for row in mappings:
        if not isinstance(row, dict):
            continue
        default = str(row.get("default") or "").strip()
        env_name = str(row.get("env") or "").strip()
        if default in supported_tables and env_name:
            env_value = os.getenv(env_name, "").strip()
            if env_value:
                supported_tables.add(env_value)
    return supported_tables


def _timestamp_order_columns() -> list[str]:
    columns = []
    for column in _order_columns():
        lowered = column.lower()
        if "date" in lowered or "time" in lowered or lowered.endswith("_at"):
            columns.append(column)
    return columns


def _is_component_entity_key(key: str) -> bool:
    return str(key or "").strip() in set(_component_entity_keys())


def _component_entity_key_in(values: dict[str, Any]) -> str | None:
    for key in _component_entity_keys():
        if key in values:
            return key
    return None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _clean_model_id(model_id: str) -> str:
    return str(model_id or "").split("_champion")[0].split("_@")[0]


def _has_feature_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        import math

        if isinstance(value, float) and math.isnan(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip()
        return bool(text) and text.lower() not in {"null", "none", "nan"}
    return True


def _is_identifier_feature(name: str) -> bool:
    normalized = _normalize_feature_name(str(name or "")).strip().lower()
    identifier_names = {
        _normalize_feature_name(value).lower()
        for value in _configured_list("identifier_names")
    }
    return (
        normalized == "id"
        or normalized.endswith("_id")
        or normalized in identifier_names
    )


def _is_model_input_feature(name: str) -> bool:
    return bool(str(name or "").strip()) and not _is_identifier_feature(name)


def _model_feature_subset(
    features: dict[str, Any],
    expected_features: tuple[str, ...],
) -> dict[str, Any]:
    if expected_features:
        allowed = set(expected_features)
        return {
            key: value
            for key, value in features.items()
            if key in allowed and _has_feature_value(value)
        }
    return {
        key: value
        for key, value in features.items()
        if _is_model_input_feature(key) and _has_feature_value(value)
    }


def _query_feature_values(
    request_features: dict[str, Any],
    expected_features: tuple[str, ...],
) -> dict[str, Any]:
    canonical = _canonicalize_request_features(
        request_features,
        expected_features=expected_features,
    )
    return {
        key: value
        for key, value in canonical.items()
        if key in expected_features
        and _is_model_input_feature(key)
        and _has_feature_value(value)
    }


def _context_column_candidates(
    feature: str,
    expected_features: tuple[str, ...],
) -> list[str]:
    normalized = _normalize_feature_name(feature)
    candidates = [feature, normalized]
    for raw_name, alias in _configured_mapping("context_field_aliases").items():
        if alias == normalized:
            candidates.append(raw_name)
    canonical = _canonical_expected_name(normalized, expected_features)
    candidates.append(canonical)
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        clean = str(candidate or "").strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            unique.append(clean)
    return unique


def _matching_context_column(
    feature: str,
    columns: Any,
    expected_features: tuple[str, ...],
) -> str | None:
    column_lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in _context_column_candidates(feature, expected_features):
        column = column_lookup.get(candidate.lower())
        if column is not None:
            return column
    normalized_feature = _normalize_feature_name(feature).lower()
    for column in columns:
        clean_column = str(column)
        if _normalize_feature_name(clean_column).lower() == normalized_feature:
            return clean_column
    return None


def _series_matches_value(series: Any, value: Any) -> Any:
    try:
        import pandas as pd

        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric_value) and numeric_series.notna().any():
            return (numeric_series - float(numeric_value)).abs() <= 1e-9
    except Exception:
        pass
    return series.astype(str).str.strip().str.lower() == str(value).strip().lower()


def _entity_keys(
    raw_keys: Any, *, feature_service: str, request_features: dict[str, Any]
) -> tuple[str, ...]:
    def with_component_key(keys: tuple[str, ...]) -> tuple[str, ...]:
        component_key = next(iter(_component_entity_keys()), "")
        parent_keys = set(_component_parent_entity_keys())
        if (
            component_key
            and component_key not in keys
            and (not parent_keys or any(key in keys for key in parent_keys))
            and _is_component_value(_value_for_entity_key(request_features, component_key))
        ):
            return (*keys, component_key)
        return keys

    if isinstance(raw_keys, str) and raw_keys.strip():
        return with_component_key((raw_keys.strip(),))
    if isinstance(raw_keys, list):
        cleaned = tuple(str(value).strip() for value in raw_keys if str(value).strip())
        if cleaned:
            return with_component_key(cleaned)

    if feature_service.endswith("_service"):
        candidate = feature_service[: -len("_service")].strip()
        if candidate:
            return with_component_key((candidate,))

    request_entity_keys = sorted(
        key
        for key in request_features.keys()
        if key.endswith("_id") or key == "entity_id"
    )
    return with_component_key(tuple(request_entity_keys))


def _has_complete_direct_vector(
    request_features: dict[str, Any],
    expected_features: tuple[str, ...],
    entity_keys: tuple[str, ...],
) -> bool:
    if not expected_features:
        return False
    return all(
        feature in request_features
        and feature not in entity_keys
        and _is_model_input_feature(feature)
        and _has_feature_value(request_features.get(feature))
        for feature in expected_features
    )


def _extract_entity_row(
    request_features: dict[str, Any], entity_keys: tuple[str, ...]
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in entity_keys:
        value = _value_for_entity_key(request_features, key)
        if value is None or str(value).strip() == "":
            return {}
        row[key] = str(value).strip()
    return row


def _value_for_entity_key(request_features: dict[str, Any], key: str) -> Any:
    aliases = _as_clean_list(_configured_mapping("entity_key_aliases").get(key)) or [key]
    return next(
        (
            request_features.get(alias)
            for alias in aliases
            if alias in request_features
        ),
        None,
    )


def _available_entity_key_values(
    partial_entity_row: dict[str, Any],
    key: str,
    *,
    feature_service_names: list[str],
) -> list[str]:
    values: list[str] = []
    values.extend(_available_entity_key_values_from_postgres(partial_entity_row, key))
    values.extend(
        _available_entity_key_values_from_offline(
            partial_entity_row,
            key,
            feature_service_names=feature_service_names,
        )
    )
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        if not clean:
            continue
        normalized = _normalize_component_value(clean) if _is_component_entity_key(key) else clean
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    if _is_component_entity_key(key) and len(unique) > 1:
        all_values = _component_all_values()
        unique = [
            value
            for value in unique
            if _normalize_component_value(value) not in all_values
        ]
    return unique


def _available_entity_key_values_from_offline(
    partial_entity_row: dict[str, Any],
    key: str,
    *,
    feature_service_names: list[str],
) -> list[str]:
    values: list[str] = []
    for path in _offline_feature_paths(feature_service_names):
        if not path.exists():
            continue
        try:
            import pandas as pd

            frame = pd.read_parquet(path, columns=list(partial_entity_row) + [key])
        except Exception:
            continue
        if key not in frame.columns or any(column not in frame.columns for column in partial_entity_row):
            continue
        mask = None
        for entity_key, entity_value in partial_entity_row.items():
            current = frame[entity_key].astype(str) == str(entity_value)
            mask = current if mask is None else mask & current
        matches = frame.loc[mask] if mask is not None else frame
        if matches.empty:
            continue
        values.extend(
            str(value).strip()
            for value in matches[key].dropna().astype(str).unique().tolist()
        )
    return values


def _available_entity_key_values_from_postgres(
    partial_entity_row: dict[str, Any],
    key: str,
) -> list[str]:
    table_name = _entity_postgres_table(partial_entity_row)
    if not table_name:
        return []
    schema_name = os.getenv("PIOS_ORCH_DB_SCHEMA", "public").strip() or "public"
    try:
        import psycopg2
        from psycopg2 import sql

        conn = psycopg2.connect(
            host=os.getenv("DB_HOST") or os.getenv("PGHOST") or "localhost",
            port=int(os.getenv("DB_PORT") or os.getenv("PGPORT") or "5432"),
            dbname=os.getenv("DB_NAME") or os.getenv("PGDATABASE") or "pios",
            user=os.getenv("DB_USER") or os.getenv("PGUSER") or "admin",
            password=os.getenv("DB_PASSWORD") or os.getenv("PGPASSWORD") or "admin",
            connect_timeout=max(1, int(os.getenv("DB_CONNECT_TIMEOUT", "1"))),
        )
    except Exception:
        return []
    try:
        with conn.cursor() as cur:
            where = sql.SQL(" AND ").join(
                sql.SQL("{} = %s").format(sql.Identifier(entity_key))
                for entity_key in partial_entity_row
            )
            query = sql.SQL("SELECT DISTINCT {} FROM {}.{} WHERE {} LIMIT 100").format(
                sql.Identifier(key),
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
                where,
            )
            cur.execute(query, list(partial_entity_row.values()))
            return [str(row[0]).strip() for row in cur.fetchall() if row and row[0] is not None]
    except Exception:
        return []
    finally:
        conn.close()


def _flatten_online_row(
    online_result: dict[str, list[Any]],
    *,
    expected_features: tuple[str, ...],
    entity_keys: tuple[str, ...],
) -> dict[str, Any]:
    alias_lookup = _expected_feature_aliases(expected_features)
    row: dict[str, Any] = {}

    for raw_name, values in online_result.items():
        if not isinstance(values, list) or not values:
            continue
        value = values[0]
        if value is None:
            continue

        name = _normalize_feature_name(str(raw_name))
        if name in entity_keys:
            continue
        target_name = alias_lookup.get(name, name)
        if expected_features and target_name not in expected_features:
            continue
        if target_name not in row:
            row[target_name] = value
    return row


def _normalize_feature_name(raw_name: str) -> str:
    if "__" in raw_name:
        return raw_name.split("__", 1)[1]
    if ":" in raw_name:
        return raw_name.split(":", 1)[1]
    return raw_name


def _expected_feature_aliases(expected_features: tuple[str, ...]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for feature_name in expected_features:
        normalized = _normalize_feature_name(feature_name)
        aliases.setdefault(normalized, feature_name)
    return aliases


def _canonicalize_request_features(
    request_features: dict[str, Any],
    *,
    expected_features: tuple[str, ...],
) -> dict[str, Any]:
    if not expected_features:
        return dict(request_features)

    aliases = _expected_feature_aliases(expected_features)
    canonical: dict[str, Any] = {}
    for key, value in request_features.items():
        target_key = aliases.get(_normalize_feature_name(key), key)
        canonical[target_key] = value
    return canonical


def _get_feature_store(repo_path: Path):
    if repo_path in _feature_store_cache:
        return _feature_store_cache[repo_path]

    try:
        from feast import FeatureStore
    except ImportError as exc:
        raise RuntimeError(
            "Feast is not installed in the backend environment. "
            "Install it before enabling online feature serving."
        ) from exc

    if not repo_path.exists():
        raise RuntimeError(f"Feast repo not found: {repo_path}")

    logger.info("Loading Feast FeatureStore from %s", repo_path)
    store = FeatureStore(repo_path=str(repo_path))
    with _feature_store_lock:
        _feature_store_cache[repo_path] = store
    return store
