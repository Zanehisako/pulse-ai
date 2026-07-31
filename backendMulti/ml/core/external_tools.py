"""
External tool implementations: Postgres-backed db_tool, search, llm_api, simulation.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, execute_values

from ml.core.feature_extraction import extract_nl_features
from ml.core.http_client import http_json_request
from ml.core.simulation_tool import (
    SimulationToolConfig,
    execute_simulation_query,
)
from ml.core.utils import as_float, as_int, safe_slug


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ML_DIR = PROJECT_ROOT / "ml"
ML_BACKEND_DIR = PROJECT_ROOT.parent / "ml-backend"
_VALID_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_POSTGRES_BOOTSTRAP_LOCK = threading.Lock()
_POSTGRES_BOOTSTRAP_CACHE: set[tuple[str, str, str, str]] = set()
_SQLITE_BOOTSTRAP_LOCK = threading.Lock()
_SQLITE_BOOTSTRAP_CACHE: set[tuple[str, str, str, str]] = set()
DEFAULT_SQLITE_DB_PATH = BACKEND_ML_DIR / ".cache" / "orchestrator_reference.sqlite3"
DEFAULT_FEATURE_DOMAIN_CONFIG_PATH = BACKEND_ML_DIR / "config" / "feature_store_dynamic.json"
DEFAULT_DB_REFERENCE_TABLE_CONFIG_PATH = BACKEND_ML_DIR / "config" / "db_reference_tables.json"
_DOMAIN_CONFIG_CACHE: tuple[Path, float, dict[str, Any]] | None = None
_DB_REFERENCE_CONFIG_CACHE: tuple[Path, float, dict[str, Any]] | None = None
_NUMERIC_COLUMN_TYPES = {
    "integer",
    "bigint",
    "double precision",
}


def _clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _domain_config_path() -> Path:
    raw = os.getenv("PIOS_FEATURE_DOMAIN_CONFIG", "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_FEATURE_DOMAIN_CONFIG_PATH


def _domain_config() -> dict[str, Any]:
    global _DOMAIN_CONFIG_CACHE
    path = _domain_config_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if (
        _DOMAIN_CONFIG_CACHE is not None
        and _DOMAIN_CONFIG_CACHE[0] == path
        and _DOMAIN_CONFIG_CACHE[1] == mtime
    ):
        return _DOMAIN_CONFIG_CACHE[2]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    _DOMAIN_CONFIG_CACHE = (path, mtime, payload)
    return payload


def _component_config() -> dict[str, Any]:
    value = _domain_config().get("component")
    return dict(value) if isinstance(value, dict) else {}


def _db_reference_config_path() -> Path:
    raw = os.getenv("PIOS_DB_REFERENCE_TABLE_CONFIG", "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_DB_REFERENCE_TABLE_CONFIG_PATH


def _db_reference_config() -> dict[str, Any]:
    global _DB_REFERENCE_CONFIG_CACHE
    path = _db_reference_config_path()
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        raise RuntimeError(f"DB reference table config not found: {path}") from exc
    if (
        _DB_REFERENCE_CONFIG_CACHE is not None
        and _DB_REFERENCE_CONFIG_CACHE[0] == path
        and _DB_REFERENCE_CONFIG_CACHE[1] == mtime
    ):
        return _DB_REFERENCE_CONFIG_CACHE[2]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid DB reference table config JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("DB reference table config must be a JSON object.")
    _DB_REFERENCE_CONFIG_CACHE = (path, mtime, payload)
    return payload


def _target_like_column_patterns() -> list[re.Pattern[str]]:
    patterns = _db_reference_config().get("target_column_patterns", [])
    if not isinstance(patterns, list):
        raise RuntimeError("DB reference target_column_patterns must be a list.")
    return [re.compile(str(pattern), flags=re.IGNORECASE) for pattern in patterns]


def _db_query_defaults() -> dict[str, Any]:
    value = _db_reference_config().get("query_defaults")
    if not isinstance(value, dict):
        raise RuntimeError("DB reference table config must define query_defaults.")
    return value


def _db_default_limit() -> int:
    return max(1, as_int(_db_query_defaults().get("default_limit"), 10))


def _db_max_limit() -> int:
    return max(1, as_int(_db_query_defaults().get("max_limit"), _db_default_limit()))


def db_reference_default_limit() -> int:
    return _db_default_limit()


def db_reference_series_limit() -> int:
    value = as_int(_db_query_defaults().get("series_limit"), 0)
    if value <= 0:
        raise RuntimeError("DB reference query_defaults.series_limit must be positive.")
    return value


def db_reference_default_schema() -> str:
    value = safe_slug(_db_query_defaults().get("default_schema"))
    if not value:
        raise RuntimeError("DB reference query_defaults.default_schema must be configured.")
    return value


def db_reference_default_sqlite_table() -> str:
    value = safe_slug(_db_query_defaults().get("default_sqlite_table"))
    if not value:
        raise RuntimeError(
            "DB reference query_defaults.default_sqlite_table must be configured."
        )
    return value


def db_reference_default_table_name(logical_table: str) -> str:
    config = _TABLE_CONFIGS.get(safe_slug(logical_table))
    if not config:
        raise RuntimeError(f"Unknown DB reference table: {logical_table}")
    value = safe_slug(config.get("default_table_name"))
    if not value:
        raise RuntimeError(
            f"DB reference table '{logical_table}' default_table_name must be configured."
        )
    return value


def _aggregate_aliases() -> dict[str, str]:
    raw = _db_query_defaults().get("aggregate_aliases")
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError("DB reference query_defaults.aggregate_aliases must be configured.")
    return {
        safe_slug(key): safe_slug(value)
        for key, value in raw.items()
        if safe_slug(key) and safe_slug(value)
    }


def _db_nl_config() -> dict[str, Any]:
    value = _db_reference_config().get("natural_language_query")
    if not isinstance(value, dict):
        raise RuntimeError("DB reference natural_language_query must be configured.")
    return value


def _nl_terms(key: str) -> list[str]:
    return _clean_list(_db_nl_config().get(key))


def _nl_patterns(key: str) -> list[str]:
    return _clean_list(_db_nl_config().get(key))


def _contains_any_term(text: str, terms: list[str]) -> bool:
    return any(term.lower() in text for term in terms)


def _required_term_groups(key: str) -> list[list[str]]:
    rows = _db_nl_config().get(key)
    if not isinstance(rows, list):
        return []
    groups: list[list[str]] = []
    for row in rows:
        terms = _clean_list(row)
        if terms:
            groups.append(terms)
    return groups


def is_target_like_column(column_name: str) -> bool:
    normalized = str(column_name or "").strip().lower()
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _target_like_column_patterns())


def _reference_table_columns(
    raw_columns: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    return [
        (name, dtype)
        for name, dtype in raw_columns
        if not is_target_like_column(name)
    ]


def _column_rows(raw_columns: Any, *, table_name: str) -> list[tuple[str, str]]:
    if not isinstance(raw_columns, list) or not raw_columns:
        raise RuntimeError(f"DB table '{table_name}' must define non-empty columns.")
    rows: list[tuple[str, str]] = []
    for row in raw_columns:
        if (
            isinstance(row, (list, tuple))
            and len(row) == 2
            and str(row[0]).strip()
            and str(row[1]).strip()
        ):
            rows.append((str(row[0]).strip(), str(row[1]).strip()))
            continue
        if isinstance(row, dict) and str(row.get("name") or "").strip():
            rows.append(
                (
                    str(row.get("name")).strip(),
                    str(row.get("type") or "TEXT").strip(),
                )
            )
            continue
        raise RuntimeError(f"Invalid column entry in DB table '{table_name}'.")
    return rows


def _load_table_configs() -> dict[str, dict[str, Any]]:
    payload = _db_reference_config()
    tables = payload.get("tables")
    if not isinstance(tables, dict) or not tables:
        raise RuntimeError("DB reference table config must define tables.")
    configs: dict[str, dict[str, Any]] = {}
    for logical_name, raw_config in tables.items():
        logical = safe_slug(logical_name)
        if not logical or not isinstance(raw_config, dict):
            raise RuntimeError(f"Invalid DB reference table entry: {logical_name!r}.")
        ddl_columns = _reference_table_columns(
            _column_rows(raw_config.get("columns"), table_name=logical)
        )
        aliases = set(_clean_list(raw_config.get("aliases")))
        if not aliases:
            raise RuntimeError(f"DB table '{logical}' must define aliases.")
        configs[logical] = {
            "aliases": aliases,
            "default_table_name": str(raw_config.get("default_table_name") or logical).strip(),
            "columns": {name: dtype.lower() for name, dtype in ddl_columns},
            "ddl_columns": ddl_columns,
            "identity_fields": _clean_list(raw_config.get("identity_fields")),
            "field_aliases": {
                safe_slug(key): str(value).strip()
                for key, value in dict(raw_config.get("field_aliases") or {}).items()
                if safe_slug(key) and str(value).strip()
            },
            "ignored_filter_fields": {
                safe_slug(item)
                for item in _clean_list(raw_config.get("ignored_filter_fields"))
                if safe_slug(item)
            },
            "filter_value_paths": {
                str(key).strip(): _clean_list(value)
                for key, value in dict(raw_config.get("filter_value_paths") or {}).items()
                if str(key).strip()
            },
            "filter_value_patterns": {
                str(key).strip(): _clean_list(value)
                for key, value in dict(raw_config.get("filter_value_patterns") or {}).items()
                if str(key).strip()
            },
            "filter_value_case": {
                str(key).strip(): str(value).strip().lower()
                for key, value in dict(raw_config.get("filter_value_case") or {}).items()
                if str(key).strip() and str(value).strip()
            },
            "filter_value_map": {
                str(key).strip(): {
                    str(map_key).strip().lower(): map_value
                    for map_key, map_value in dict(value).items()
                    if str(map_key).strip()
                }
                for key, value in dict(raw_config.get("filter_value_map") or {}).items()
                if str(key).strip() and isinstance(value, dict)
            },
            "seed_field_aliases": {
                str(key).strip(): _clean_list(value)
                for key, value in dict(raw_config.get("seed_field_aliases") or {}).items()
                if str(key).strip()
            },
            "seed_derived_fields": {
                str(key).strip(): dict(value)
                for key, value in dict(raw_config.get("seed_derived_fields") or {}).items()
                if str(key).strip() and isinstance(value, dict)
            },
            "seed_refresh_when_empty_fields": _clean_list(
                raw_config.get("seed_refresh_when_empty_fields")
            ),
            "inverse_field_aliases": set(_clean_list(raw_config.get("inverse_field_aliases"))),
            "csv_kind": str(raw_config.get("csv_kind") or logical).strip(),
        }
    return configs


_TABLE_CONFIGS: dict[str, dict[str, Any]] = _load_table_configs()

_SEED_FIELD_ALIASES = {
    "date": ("event_timestamp", "as_of_date"),
    "temp_c": ("temperature",),
    "supply_shock": ("disaster",),
    "stock_start": ("current_inventory",),
    "stock_end": ("current_inventory",),
}
_NO_SEED_VALUE = object()


def _is_blank_seed_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() == ""


def _seed_field_aliases(field_name: str) -> list[str]:
    aliases = list(_SEED_FIELD_ALIASES.get(field_name, ()))
    for config in _TABLE_CONFIGS.values():
        seed_aliases = config.get("seed_field_aliases")
        if isinstance(seed_aliases, dict):
            aliases.extend(_clean_list(seed_aliases.get(field_name)))
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _coerce_seed_compare_value(value: Any) -> Any:
    if _is_blank_seed_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value).strip().lower()


def _seed_rule_matches(raw_value: Any, rule: dict[str, Any]) -> bool:
    operator = str(rule.get("operator") or "not_blank").strip().lower()
    if operator == "not_blank":
        return not _is_blank_seed_value(raw_value)
    left = _coerce_seed_compare_value(raw_value)
    right = _coerce_seed_compare_value(rule.get("value"))
    if left is None:
        return False
    if operator in {"eq", "equals"}:
        return left == right
    if operator in {"ne", "not_equals"}:
        return left != right
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    if operator == "lt":
        return left < right
    if operator == "lte":
        return left <= right
    if operator == "gt":
        return left > right
    if operator == "gte":
        return left >= right
    return False


def _seed_derived_value(row: dict[str, Any], *, field_name: str) -> Any:
    for config in _TABLE_CONFIGS.values():
        derived_fields = config.get("seed_derived_fields")
        if not isinstance(derived_fields, dict):
            continue
        rule = derived_fields.get(field_name)
        if not isinstance(rule, dict):
            continue
        source = str(rule.get("source") or "").strip()
        if not source:
            continue
        raw_value = row.get(source)
        if _is_blank_seed_value(raw_value):
            continue
        matched = _seed_rule_matches(raw_value, rule)
        if "true_value" in rule or "false_value" in rule:
            return rule.get("true_value") if matched else rule.get("false_value")
        return matched
    return _NO_SEED_VALUE

def _dedupe_list_fields(fields: list[str], config: dict[str, Any]) -> list[str]:
    columns = config.get("columns", {})
    seen: set[str] = set()
    selected: list[str] = []
    for field in fields:
        normalized = _resolve_reference_field_name(field, config=config)
        if not normalized or normalized in seen or normalized not in columns:
            continue
        seen.add(normalized)
        selected.append(normalized)
    return selected


def _resolve_reference_field_name(
    field: Any,
    *,
    config: dict[str, Any],
) -> str:
    normalized = str(field or "").strip()
    if not normalized:
        return ""
    aliases = config.get("field_aliases")
    if isinstance(aliases, dict):
        aliased = str(aliases.get(safe_slug(normalized)) or "").strip()
        if aliased:
            return aliased
    return normalized


def _lookup_filter_value_path(value: Any, path: str) -> Any:
    current = value
    for part in [segment for segment in str(path or "").split(".") if segment]:
        if isinstance(current, dict):
            current = current.get(part)
            continue
        return _NO_SEED_VALUE
    return current


def _normalize_configured_filter_value(
    value: Any,
    *,
    field_name: str,
    config: dict[str, Any],
) -> Any:
    if isinstance(value, dict):
        paths = config.get("filter_value_paths")
        paths = paths if isinstance(paths, dict) else {}
        for path in _clean_list(paths.get(field_name)):
            candidate = _lookup_filter_value_path(value, path)
            if candidate is not _NO_SEED_VALUE and not _is_blank_seed_value(candidate):
                value = candidate
                break
    if isinstance(value, list):
        return [
            _normalize_configured_filter_value(
                item,
                field_name=field_name,
                config=config,
            )
            for item in value
            if not _is_blank_seed_value(item)
        ]
    if isinstance(value, str):
        text = value.strip()
        patterns = config.get("filter_value_patterns")
        patterns = patterns if isinstance(patterns, dict) else {}
        for pattern in _clean_list(patterns.get(field_name)):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            groups = [group for group in match.groups() if group not in {None, ""}]
            text = "".join(groups) if groups else match.group(0)
            break
        value = text
    cases = config.get("filter_value_case")
    cases = cases if isinstance(cases, dict) else {}
    case_mode = str(cases.get(field_name) or "").strip().lower()
    if isinstance(value, str) and case_mode == "upper":
        return value.upper()
    if isinstance(value, str) and case_mode == "lower":
        value = value.lower()
    value_maps = config.get("filter_value_map")
    value_maps = value_maps if isinstance(value_maps, dict) else {}
    field_map = value_maps.get(field_name)
    if isinstance(field_map, dict):
        key = str(value).strip().lower()
        if key in field_map:
            return field_map[key]
    return value


def _normalize_reference_filters(
    filters: dict[str, Any] | None,
    *,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(filters, dict):
        return filters
    normalized_filters: dict[str, Any] = {}
    ignored_fields = config.get("ignored_filter_fields")
    ignored_fields = ignored_fields if isinstance(ignored_fields, set) else set()
    for raw_field, raw_value in filters.items():
        field_key = safe_slug(raw_field)
        if field_key in ignored_fields:
            continue
        field_name = _resolve_reference_field_name(raw_field, config=config)
        if not field_name:
            continue
        normalized_filters[field_name] = _normalize_configured_filter_value(
            raw_value,
            field_name=field_name,
            config=config,
        )
    return normalized_filters


def _query_column_tokens(query: str) -> set[str]:
    tokens = {token for token in re.split(r"[^a-z0-9_+-]+", (query or "").lower()) if token}
    singularized = {
        token[:-1]
        for token in tokens
        if len(token) > 3 and token.endswith("s")
    }
    return tokens | singularized


def _infer_reference_list_fields(
    query: str,
    *,
    logical_table: str,
    config: dict[str, Any],
) -> list[str]:
    lowered = (query or "").lower()
    tokens = _query_column_tokens(query)
    columns = list(config.get("columns") or {})
    identity_fields = _dedupe_list_fields(list(config.get("identity_fields") or []), config)

    requested_fields: list[str] = []
    for column in columns:
        column_parts = {part for part in column.lower().split("_") if part}
        if column.lower() in lowered or tokens.intersection(column_parts):
            requested_fields.append(column)

    if requested_fields:
        data_fields = [field for field in requested_fields if field not in identity_fields]
        if data_fields:
            return _dedupe_list_fields(identity_fields + data_fields, config)
        return _dedupe_list_fields(requested_fields, config)
    return identity_fields[:1]


def _infer_reference_list_request(query: str) -> tuple[str, list[str], bool] | None:
    lowered = (query or "").lower()
    if any(
        token in lowered
        for token in _nl_terms("blocked_prediction_tokens")
    ):
        return None
    wants_list = any(
        token in lowered
        for token in _nl_terms("list_intent_tokens")
    )
    if not wants_list:
        return None
    tokens = _query_column_tokens(query)
    latest_terms = _nl_terms("latest_tokens")
    latest_only = bool(tokens.intersection(set(latest_terms))) or _contains_any_term(
        lowered,
        latest_terms,
    )
    for logical_table, config in _TABLE_CONFIGS.items():
        aliases = set(config.get("aliases") or ())
        if not tokens.intersection(aliases) and not _contains_any_term(
            lowered,
            list(aliases),
        ):
            continue
        return (
            logical_table,
            _infer_reference_list_fields(
                query,
                logical_table=logical_table,
                config=config,
            ),
            latest_only,
        )
    return None


def is_supply_query(query: str) -> bool:
    lowered = (query or "").lower()
    tokens = {
        token for token in re.split(r"[^a-z0-9_+-]+", lowered) if token
    }
    supply_terms = _nl_terms("supply_intent_tokens")
    has_supply_intent = bool(tokens.intersection(set(supply_terms))) or any(
        term.lower() in lowered
        for term in supply_terms
        if any(char.isspace() for char in term)
    )
    has_blood_context = _contains_any_term(
        lowered,
        _nl_terms("supply_context_terms"),
    ) or any(
        re.search(pattern, lowered) for pattern in _nl_patterns("supply_blood_group_patterns")
    )
    return has_supply_intent and has_blood_context


def is_blood_prevalence_query(query: str) -> bool:
    lowered = (query or "").lower()
    blood_context = _contains_any_term(
        lowered,
        _nl_terms("blood_context_phrases"),
    ) or any(
        all(term.lower() in lowered for term in group)
        for group in _required_term_groups("blood_context_required_terms")
    )
    population_context = _contains_any_term(lowered, _nl_terms("population_context_terms"))
    return blood_context and population_context and any(
        token in lowered for token in _nl_terms("prevalence_tokens")
    )


def is_donor_outreach_query(query: str) -> bool:
    lowered = (query or "").lower()
    if not all(term.lower() in lowered for term in _nl_terms("outreach_required_terms")):
        return False
    return any(token in lowered for token in _nl_terms("outreach_tokens"))


def is_structured_db_query(query: str) -> bool:
    return (
        is_supply_query(query)
        or is_blood_prevalence_query(query)
        or is_donor_outreach_query(query)
        or _infer_reference_list_request(query) is not None
    )


def extract_supply_blood_type(query: str) -> str | None:
    extracted = extract_nl_features(query)
    raw = extracted.get("blood_type")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().upper()
    pattern_config = _db_nl_config().get("blood_type_patterns")
    pattern_config = pattern_config if isinstance(pattern_config, dict) else {}
    sign_match = re.search(str(pattern_config.get("signed") or ""), query, flags=re.IGNORECASE)
    if sign_match:
        return f"{sign_match.group(1).upper()}{sign_match.group(2)}"
    word_match = re.search(str(pattern_config.get("worded") or ""), query, flags=re.IGNORECASE)
    if not word_match:
        return None
    word_signs = pattern_config.get("word_signs")
    word_signs = word_signs if isinstance(word_signs, dict) else {}
    sign = str(word_signs.get(word_match.group(2).lower()) or "").strip()
    if not sign:
        return None
    return f"{word_match.group(1).upper()}{sign}"


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "eligible"}


def _extract_country_code(query: str) -> str | None:
    code_match = re.search(
        str(_db_nl_config().get("country_code_pattern") or ""),
        query,
        flags=re.IGNORECASE,
    )
    if code_match:
        return code_match.group(1).upper()

    lowered = query.lower()
    aliases = _db_nl_config().get("country_aliases")
    aliases = aliases if isinstance(aliases, dict) else {}
    for phrase, code in aliases.items():
        if str(phrase).strip().lower() in lowered:
            return str(code).strip().upper()
    return None


def _requested_limit(query: str, default: int = 10, maximum: int = 25) -> int:
    lowered = (query or "").lower()
    for pattern in _nl_patterns("limit_patterns"):
        match = re.search(pattern, lowered)
        if match:
            return max(1, min(maximum, int(match.group(1))))
    return default


def _wants_ethnicity_breakdown(query: str) -> bool:
    lowered = (query or "").lower()
    return any(token in lowered for token in _nl_terms("ethnicity_terms"))


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    rows: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        rows.append(path)
    return rows


def _dataset_donor_csv_candidates() -> list[Path]:
    return [
        BACKEND_ML_DIR
        / "datasets"
        / "blood_registry_sythentic"
        / "data"
        / "blood_donation_registry_ml_ready.csv",
        BACKEND_ML_DIR / "datasets" / "synthetic_donor_candidates_clustered.csv",
        BACKEND_ML_DIR / "datasets" / "donor_snapshots_with_targets.csv",
        ML_BACKEND_DIR
        / "datasets"
        / "blood_registry_sythentic"
        / "data"
        / "blood_donation_registry_ml_ready.csv",
        ML_BACKEND_DIR / "datasets" / "synthetic_donor_candidates_clustered.csv",
        ML_BACKEND_DIR / "datasets" / "donor_snapshots_with_targets.csv",
    ]


def _default_donor_csv_fallback() -> Path:
    candidates = _dedupe_paths(
        [path.expanduser().resolve() for path in _dataset_donor_csv_candidates()]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _default_donor_csv_candidates() -> list[Path]:
    env_paths: list[Path] = []
    for env_name in ("PIOS_ORCH_DB_DONOR_CSV_PATH", "PIOS_ORCH_DB_BOOTSTRAP_CSV"):
        raw = os.getenv(env_name, "").strip()
        if raw:
            env_paths.append(Path(raw).expanduser().resolve())

    return _dedupe_paths(
        env_paths
        + [path.expanduser().resolve() for path in _dataset_donor_csv_candidates()]
    )


def _validate_identifier(value: str, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise RuntimeError(f"{label} cannot be empty.")
    if not _VALID_SQL_IDENTIFIER.match(normalized):
        raise RuntimeError(
            f"Invalid {label}: {value!r}. Use letters, digits, and underscores only."
        )
    return normalized


def _normalize_db_schema(schema: str | None) -> str:
    return _validate_identifier(schema or db_reference_default_schema(), "schema")


def _normalize_db_limit(limit: Any, *, default: int | None = None) -> int:
    if default is None:
        default = _db_default_limit()
    requested = as_int(limit, default)
    return max(1, min(_db_max_limit(), requested))


def _normalize_aggregate(aggregate: str | None) -> str:
    normalized = safe_slug(aggregate or "")
    if not normalized:
        raise RuntimeError("DB aggregate is required.")
    aliases = _aggregate_aliases()
    if normalized not in aliases:
        supported = ", ".join(sorted(aliases))
        raise RuntimeError(f"Unsupported DB aggregate '{aggregate}'. Supported: {supported}.")
    return aliases[normalized]


def _resolve_table_request(
    table: str | None,
    *,
    donors_table_name: str,
    hospitals_table_name: str,
) -> tuple[str, str, dict[str, Any]]:
    normalized = safe_slug(table or "")
    for logical_name, config in _TABLE_CONFIGS.items():
        if normalized in config["aliases"]:
            actual_name = donors_table_name if logical_name == "donors" else hospitals_table_name
            return logical_name, _validate_identifier(actual_name, "table name"), config
    supported = ", ".join(sorted(_TABLE_CONFIGS))
    raise RuntimeError(f"Unsupported DB table '{table}'. Supported tables: {supported}.")


def _postgres_connect():
    dsn = os.getenv("PIOS_ORCH_DB_DSN", "").strip()
    connect_timeout = max(1, int(os.getenv("DB_CONNECT_TIMEOUT", "3")))
    if dsn:
        return psycopg2.connect(dsn=dsn, connect_timeout=connect_timeout)
    return psycopg2.connect(
        host=os.getenv("DB_HOST") or os.getenv("PGHOST") or "localhost",
        port=int(os.getenv("DB_PORT") or os.getenv("PGPORT") or "5432"),
        dbname=os.getenv("DB_NAME") or os.getenv("PGDATABASE") or "pios",
        user=os.getenv("DB_USER") or os.getenv("PGUSER") or "admin",
        password=os.getenv("DB_PASSWORD") or os.getenv("PGPASSWORD") or "admin",
        connect_timeout=connect_timeout,
    )


def _serialize_db_value(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def _serialize_db_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _serialize_db_value(value) for key, value in row.items()}


def _parse_optional_bool(value: Any) -> bool | None:
    if _is_blank_seed_value(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _parse_optional_int(value: Any) -> int | None:
    if _is_blank_seed_value(value):
        return None
    text = str(value).strip()
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _parse_optional_float(value: Any) -> float | None:
    if _is_blank_seed_value(value):
        return None
    text = str(value).strip()
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_optional_date(value: Any) -> dt.date | None:
    if _is_blank_seed_value(value):
        return None
    if hasattr(value, "date"):
        try:
            return value.date()
        except Exception:
            pass
    text = str(value).strip()
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return dt.date.fromisoformat(text[:10])
        except ValueError:
            return None


def _qualified_table(schema_name: str, table_name: str):
    return sql.SQL("{}.{}").format(
        sql.Identifier(schema_name),
        sql.Identifier(table_name),
    )


def _create_schema_if_needed(cur, schema_name: str) -> None:
    cur.execute(
        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name))
    )


def _create_reference_table(
    cur,
    *,
    schema_name: str,
    table_name: str,
    columns: list[tuple[str, str]],
    logical_name: str,
) -> None:
    column_defs = [
        sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(dtype))
        for name, dtype in columns
    ]
    if logical_name == "hospitals":
        column_defs.append(
            sql.SQL("PRIMARY KEY ({}, {}, {})").format(
                sql.Identifier("date"),
                sql.Identifier("hospital_id"),
                sql.Identifier("blood_type"),
            )
        )
    cur.execute(
        sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            _qualified_table(schema_name, table_name),
            sql.SQL(", ").join(column_defs),
        )
    )


def _ensure_postgres_table_columns(
    cur,
    *,
    schema_name: str,
    table_name: str,
    columns: list[tuple[str, str]],
) -> None:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        """,
        [schema_name, table_name],
    )
    existing = {str(row[0]) for row in cur.fetchall()}
    for name, dtype in columns:
        if name in existing:
            continue
        normalized_dtype = str(dtype).replace("PRIMARY KEY", "").strip()
        cur.execute(
            sql.SQL("ALTER TABLE {} ADD COLUMN {} {}").format(
                _qualified_table(schema_name, table_name),
                sql.Identifier(name),
                sql.SQL(normalized_dtype),
            )
        )


def _create_reference_indexes(
    cur,
    *,
    schema_name: str,
    donors_table_name: str,
    hospitals_table_name: str,
    required_tables: set[str],
) -> None:
    if "donors" in required_tables:
        cur.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {} ON {} ({}, {})"
            ).format(
                sql.Identifier(f"{donors_table_name}_country_blood_idx"),
                _qualified_table(schema_name, donors_table_name),
                sql.Identifier("country_code"),
                sql.Identifier("blood_type"),
            )
        )
    if "hospitals" in required_tables:
        cur.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {} ON {} ({}, {}, {})"
            ).format(
                sql.Identifier(f"{hospitals_table_name}_hospital_blood_date_idx"),
                _qualified_table(schema_name, hospitals_table_name),
                sql.Identifier("hospital_id"),
                sql.Identifier("blood_type"),
                sql.Identifier("date"),
            )
        )


def _table_row_count(cur, *, schema_name: str, table_name: str) -> int:
    cur.execute(
        sql.SQL("SELECT COUNT(*) FROM {}").format(
            _qualified_table(schema_name, table_name)
        )
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


def _prepare_seed_value(row: dict[str, Any], *, field_name: str, dtype: str) -> Any:
    raw = row.get(field_name)
    if _is_blank_seed_value(raw):
        for alias in _seed_field_aliases(field_name):
            raw = row.get(alias)
            if not _is_blank_seed_value(raw):
                break
    if _is_blank_seed_value(raw):
        derived = _seed_derived_value(row, field_name=field_name)
        if derived is not _NO_SEED_VALUE:
            raw = derived
    component_keys = set(_clean_list(_component_config().get("entity_keys")))
    if _is_blank_seed_value(raw) and field_name in component_keys:
        parent_keys = set(_clean_list(_component_config().get("parent_entity_keys")))
        all_values = _clean_list(_component_config().get("all_values"))
        if (not parent_keys or parent_keys.intersection(row)) and all_values:
            raw = all_values[0]
    normalized_dtype = str(dtype or "").strip().lower()
    if "primary key" in normalized_dtype:
        normalized_dtype = normalized_dtype.replace("primary key", "").strip()

    if normalized_dtype in {"integer", "bigint"}:
        return _parse_optional_int(raw)
    if normalized_dtype == "double precision":
        return _parse_optional_float(raw)
    if normalized_dtype == "boolean":
        return _parse_optional_bool(raw)
    if normalized_dtype == "date":
        return _parse_optional_date(raw)
    text = str(raw or "").strip()
    if not text:
        return None
    if field_name in {"country_code", "blood_type"}:
        return text.upper()
    return text


def _prepare_seed_row(
    row: dict[str, Any],
    *,
    columns: list[tuple[str, str]],
) -> tuple[Any, ...]:
    return tuple(
        _prepare_seed_value(row, field_name=field_name, dtype=dtype)
        for field_name, dtype in columns
    )


def _stockout_seed_components() -> list[str]:
    cleaned = [
        _normalize_stockout_component(value)
        for value in _clean_list(_component_config().get("values"))
    ]
    return [value for value in cleaned if value]


def _component_all_markers() -> list[str]:
    cleaned = [
        _normalize_stockout_component(value)
        for value in _clean_list(_component_config().get("all_values"))
    ]
    return [value for value in cleaned if value]


def _normalize_stockout_component(value: Any) -> str:
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


def _component_factor(component: str) -> float:
    factors = _component_config().get("factors")
    if not isinstance(factors, dict):
        return 1.0
    try:
        return float(factors.get(_normalize_stockout_component(component), 1.0))
    except (TypeError, ValueError):
        return 1.0


def _componentize_hospital_seed_row(
    row: dict[str, Any],
    component: str,
) -> dict[str, Any]:
    factor = _component_factor(component)
    output = dict(row)
    component_keys = _clean_list(_component_config().get("entity_keys"))
    if component_keys:
        output[component_keys[0]] = _normalize_stockout_component(component)
    for key in _clean_list(_component_config().get("scaled_quantity_fields")):
        if output.get(key) is not None:
            output[key] = int(round(float(output[key]) * factor))
    for key in _clean_list(_component_config().get("scaled_count_fields")):
        if output.get(key) is not None:
            output[key] = max(0, int(round(float(output[key]) * factor)))
    critical_field = str(_component_config().get("critical_stock_field") or "").strip()
    source_field = str(_component_config().get("critical_stock_source_field") or "").strip()
    threshold = _component_config().get("critical_stock_threshold")
    if critical_field and source_field and output.get(source_field) is not None:
        try:
            output[critical_field] = bool(float(output[source_field]) <= float(threshold))
        except (TypeError, ValueError):
            pass
    return output


def _prepare_seed_rows(
    row: dict[str, Any],
    *,
    logical_table: str,
    columns: list[tuple[str, str]],
) -> list[tuple[Any, ...]]:
    column_names = [name for name, _ in columns]
    prepared = dict(zip(column_names, _prepare_seed_row(row, columns=columns)))
    if logical_table != "hospitals":
        return [tuple(prepared.get(name) for name in column_names)]
    component_keys = _clean_list(_component_config().get("entity_keys"))
    component_key = component_keys[0] if component_keys else ""
    blood_type = str(prepared.get(component_key) or "").strip().upper()
    all_values = set(_component_all_markers())
    if blood_type and _normalize_stockout_component(blood_type) not in all_values:
        return [tuple(prepared.get(name) for name in column_names)]
    rows = [prepared]
    rows.extend(
        _componentize_hospital_seed_row(prepared, component)
        for component in _stockout_seed_components()
    )
    return [tuple(item.get(name) for name in column_names) for item in rows]


def _seed_postgres_table_from_csv(
    cur,
    *,
    schema_name: str,
    logical_table: str,
    table_name: str,
    csv_path: Path,
) -> None:
    if not csv_path.exists():
        raise RuntimeError(f"Bootstrap CSV not found for {logical_table}: {csv_path}")

    config = _TABLE_CONFIGS[logical_table]
    column_names = [name for name, _ in config["ddl_columns"]]
    if csv_path.suffix.lower() == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(csv_path).to_dict(orient="records")
    else:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if isinstance(row, dict)]

    if not rows:
        raise RuntimeError(f"Bootstrap CSV for {logical_table} is empty: {csv_path}")

    prepared_rows = [
        prepared
        for row in rows
        for prepared in _prepare_seed_rows(
            row,
            logical_table=logical_table,
            columns=config["ddl_columns"],
        )
    ]

    insert_sql = sql.SQL(
        "INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING"
    ).format(
        _qualified_table(schema_name, table_name),
        sql.SQL(", ").join(sql.Identifier(name) for name in column_names),
    )
    execute_values(cur, insert_sql.as_string(cur.connection), prepared_rows, page_size=1000)


def _ensure_postgres_hospital_component_rows(
    cur,
    *,
    schema_name: str,
    table_name: str,
) -> None:
    columns = [name for name, _ in _TABLE_CONFIGS["hospitals"]["ddl_columns"]]
    all_markers = _component_all_markers()
    if not all_markers:
        return
    cur.execute(
        sql.SQL("SELECT {} FROM {} WHERE UPPER(blood_type) = ANY(%s)").format(
            sql.SQL(", ").join(sql.Identifier(name) for name in columns),
            _qualified_table(schema_name, table_name),
        ),
        [all_markers],
    )
    source_rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    if not source_rows:
        return
    prepared_rows = [
        tuple(_componentize_hospital_seed_row(row, component).get(name) for name in columns)
        for row in source_rows
        for component in _stockout_seed_components()
    ]
    if not prepared_rows:
        return
    insert_sql = sql.SQL(
        "INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING"
    ).format(
        _qualified_table(schema_name, table_name),
        sql.SQL(", ").join(sql.Identifier(name) for name in columns),
    )
    execute_values(cur, insert_sql.as_string(cur.connection), prepared_rows, page_size=1000)


def ensure_postgres_reference_tables(
    *,
    db_schema: str,
    donors_table_name: str,
    hospitals_table_name: str,
    donor_csv_path: Path,
    hospital_csv_path: Path,
    required_tables: set[str] | None = None,
) -> None:
    schema_name = _normalize_db_schema(db_schema)
    donors_table_name = _validate_identifier(donors_table_name, "donors table name")
    hospitals_table_name = _validate_identifier(
        hospitals_table_name,
        "hospitals table name",
    )
    required = set(required_tables or {"donors", "hospitals"})
    cache_key = (
        schema_name,
        donors_table_name,
        hospitals_table_name,
        ",".join(sorted(required)),
    )
    if cache_key in _POSTGRES_BOOTSTRAP_CACHE:
        return

    with _POSTGRES_BOOTSTRAP_LOCK:
        if cache_key in _POSTGRES_BOOTSTRAP_CACHE:
            return
        conn = _postgres_connect()
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                _create_schema_if_needed(cur, schema_name)
                if "donors" in required:
                    _create_reference_table(
                        cur,
                        schema_name=schema_name,
                        table_name=donors_table_name,
                        columns=_TABLE_CONFIGS["donors"]["ddl_columns"],
                        logical_name="donors",
                    )
                    _ensure_postgres_table_columns(
                        cur,
                        schema_name=schema_name,
                        table_name=donors_table_name,
                        columns=_TABLE_CONFIGS["donors"]["ddl_columns"],
                    )
                if "hospitals" in required:
                    _create_reference_table(
                        cur,
                        schema_name=schema_name,
                        table_name=hospitals_table_name,
                        columns=_TABLE_CONFIGS["hospitals"]["ddl_columns"],
                        logical_name="hospitals",
                    )
                    _ensure_postgres_table_columns(
                        cur,
                        schema_name=schema_name,
                        table_name=hospitals_table_name,
                        columns=_TABLE_CONFIGS["hospitals"]["ddl_columns"],
                    )
                _create_reference_indexes(
                    cur,
                    schema_name=schema_name,
                    donors_table_name=donors_table_name,
                    hospitals_table_name=hospitals_table_name,
                    required_tables=required,
                )
                if "donors" in required and _table_row_count(
                    cur,
                    schema_name=schema_name,
                    table_name=donors_table_name,
                ) == 0:
                    _seed_postgres_table_from_csv(
                        cur,
                        schema_name=schema_name,
                        logical_table="donors",
                        table_name=donors_table_name,
                        csv_path=donor_csv_path,
                    )
                if "hospitals" in required:
                    _seed_postgres_table_from_csv(
                        cur,
                        schema_name=schema_name,
                        logical_table="hospitals",
                        table_name=hospitals_table_name,
                        csv_path=hospital_csv_path,
                    )
                    _ensure_postgres_hospital_component_rows(
                        cur,
                        schema_name=schema_name,
                        table_name=hospitals_table_name,
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        _POSTGRES_BOOTSTRAP_CACHE.add(cache_key)


def _build_filter_clauses(
    filters: dict[str, Any] | None,
    *,
    config: dict[str, Any],
) -> tuple[list[sql.Composable], list[Any]]:
    filters = _normalize_reference_filters(filters, config=config)
    if not isinstance(filters, dict):
        return [], []

    clauses: list[sql.Composable] = []
    params: list[Any] = []
    allowed_columns = config["columns"]

    for raw_field, raw_value in sorted(filters.items()):
        field_name = _validate_identifier(raw_field, "filter field")
        if field_name not in allowed_columns:
            logger.debug("Skipping unsupported filter field '%s' for table.", field_name)
            continue
        identifier = sql.Identifier(field_name)
        if isinstance(raw_value, list):
            values = [value for value in raw_value if not _is_blank_seed_value(value)]
            if not values:
                continue
            clauses.append(
                sql.SQL("{} IN ({})").format(
                    identifier,
                    sql.SQL(", ").join(sql.Placeholder() for _ in values),
                )
            )
            params.extend(values)
            continue
        if raw_value is None:
            clauses.append(sql.SQL("{} IS NULL").format(identifier))
            continue
        clauses.append(sql.SQL("{} = {}").format(identifier, sql.Placeholder()))
        params.append(raw_value)
    return clauses, params


def _sqlite_db_path(raw_path: str | None) -> Path:
    path = Path(str(raw_path or "").strip()).expanduser() if raw_path else DEFAULT_SQLITE_DB_PATH
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _sqlite_type(dtype: str) -> str:
    normalized = str(dtype or "").strip().lower()
    if "int" in normalized or "bool" in normalized:
        return "INTEGER"
    if "double" in normalized or "real" in normalized or "float" in normalized:
        return "REAL"
    return "TEXT"


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def _sqlite_identifier(value: str, label: str = "identifier") -> str:
    return _validate_identifier(value, label)


def _sqlite_create_reference_table(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    columns: list[tuple[str, str]],
    logical_name: str,
) -> None:
    column_defs = []
    for name, dtype in columns:
        sql_type = _sqlite_type(dtype)
        if logical_name == "donors" and name == "donor_id":
            column_defs.append(f'"{name}" {sql_type} PRIMARY KEY')
        else:
            column_defs.append(f'"{name}" {sql_type}')
    if logical_name == "hospitals":
        column_defs.append('PRIMARY KEY ("date", "hospital_id", "blood_type")')
    conn.execute(
        f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(column_defs)})'
    )
    existing = {
        str(row["name"])
        for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    }
    for name, dtype in columns:
        if name not in existing:
            conn.execute(
                f'ALTER TABLE "{table_name}" ADD COLUMN "{name}" {_sqlite_type(dtype)}'
            )


def _seed_sqlite_table_from_csv(
    conn: sqlite3.Connection,
    *,
    logical_table: str,
    table_name: str,
    csv_path: Path,
) -> None:
    if not csv_path.exists():
        raise RuntimeError(f"Bootstrap CSV not found for {logical_table}: {csv_path}")

    config = _TABLE_CONFIGS[logical_table]
    column_names = [name for name, _ in config["ddl_columns"]]
    if csv_path.suffix.lower() == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(csv_path).to_dict(orient="records")
    else:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if isinstance(row, dict)]

    if not rows:
        raise RuntimeError(f"Bootstrap CSV for {logical_table} is empty: {csv_path}")

    prepared_rows = [
        tuple(_sqlite_value(value) for value in prepared)
        for row in rows
        for prepared in _prepare_seed_rows(
            row,
            logical_table=logical_table,
            columns=config["ddl_columns"],
        )
    ]
    placeholders = ", ".join("?" for _ in column_names)
    columns_sql = ", ".join(f'"{name}"' for name in column_names)
    conn.executemany(
        f'INSERT OR IGNORE INTO "{table_name}" ({columns_sql}) VALUES ({placeholders})',
        prepared_rows,
    )


def _ensure_sqlite_hospital_component_rows(
    conn: sqlite3.Connection,
    *,
    table_name: str,
) -> None:
    column_names = [name for name, _ in _TABLE_CONFIGS["hospitals"]["ddl_columns"]]
    all_markers = _component_all_markers()
    if not all_markers:
        return
    select_columns_sql = ", ".join(f'"{name}"' for name in column_names)
    placeholders = ", ".join("?" for _ in all_markers)
    fetched = conn.execute(
        f'SELECT {select_columns_sql} '
        f'FROM "{table_name}" WHERE UPPER(blood_type) IN ({placeholders})',
        all_markers,
    ).fetchall()
    if not fetched:
        return
    prepared_rows = [
        tuple(
            _sqlite_value(_componentize_hospital_seed_row(dict(row), component).get(name))
            for name in column_names
        )
        for row in fetched
        for component in _stockout_seed_components()
    ]
    if not prepared_rows:
        return
    placeholders = ", ".join("?" for _ in column_names)
    columns_sql = ", ".join(f'"{name}"' for name in column_names)
    conn.executemany(
        f'INSERT OR IGNORE INTO "{table_name}" ({columns_sql}) VALUES ({placeholders})',
        prepared_rows,
    )


def _sqlite_row_count(conn: sqlite3.Connection, *, table_name: str) -> int:
    row = conn.execute(f'SELECT COUNT(*) AS value FROM "{table_name}"').fetchone()
    return int(row["value"] if row else 0)


def _sqlite_nonempty_field_count(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    field_name: str,
) -> int:
    field = _sqlite_identifier(field_name, "seed refresh field")
    row = conn.execute(
        f'SELECT COUNT(*) AS value FROM "{table_name}" '
        f'WHERE "{field}" IS NOT NULL AND TRIM(CAST("{field}" AS TEXT)) != \'\''
    ).fetchone()
    return int(row["value"] if row else 0)


def _sqlite_seed_refresh_needed(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    config: dict[str, Any],
) -> bool:
    refresh_fields = _clean_list(config.get("seed_refresh_when_empty_fields"))
    if not refresh_fields:
        return False
    if _sqlite_row_count(conn, table_name=table_name) == 0:
        return False
    columns = config.get("columns", {})
    for field_name in refresh_fields:
        if field_name not in columns:
            continue
        if _sqlite_nonempty_field_count(
            conn,
            table_name=table_name,
            field_name=field_name,
        ) == 0:
            return True
    return False


def ensure_sqlite_reference_tables(
    *,
    db_sqlite_path: str | None,
    donors_table_name: str,
    hospitals_table_name: str,
    donor_csv_path: Path,
    hospital_csv_path: Path,
    required_tables: set[str] | None = None,
) -> Path:
    sqlite_path = _sqlite_db_path(db_sqlite_path)
    donors_table_name = _validate_identifier(donors_table_name, "donors table name")
    hospitals_table_name = _validate_identifier(
        hospitals_table_name,
        "hospitals table name",
    )
    required = set(required_tables or {"donors", "hospitals"})
    cache_key = (
        str(sqlite_path),
        donors_table_name,
        hospitals_table_name,
        ",".join(sorted(required)),
    )
    if cache_key in _SQLITE_BOOTSTRAP_CACHE:
        return sqlite_path

    with _SQLITE_BOOTSTRAP_LOCK:
        if cache_key in _SQLITE_BOOTSTRAP_CACHE:
            return sqlite_path
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            if "donors" in required:
                _sqlite_create_reference_table(
                    conn,
                    table_name=donors_table_name,
                    columns=_TABLE_CONFIGS["donors"]["ddl_columns"],
                    logical_name="donors",
                )
                donor_count = _sqlite_row_count(conn, table_name=donors_table_name)
                if donor_count > 0 and _sqlite_seed_refresh_needed(
                    conn,
                    table_name=donors_table_name,
                    config=_TABLE_CONFIGS["donors"],
                ):
                    conn.execute(f'DELETE FROM "{donors_table_name}"')
                    donor_count = 0
                if donor_count == 0:
                    _seed_sqlite_table_from_csv(
                        conn,
                        logical_table="donors",
                        table_name=donors_table_name,
                        csv_path=donor_csv_path,
                    )
                conn.execute(
                    f'CREATE INDEX IF NOT EXISTS "{donors_table_name}_country_blood_idx" '
                    f'ON "{donors_table_name}" ("country_code", "blood_type")'
                )
            if "hospitals" in required:
                _sqlite_create_reference_table(
                    conn,
                    table_name=hospitals_table_name,
                    columns=_TABLE_CONFIGS["hospitals"]["ddl_columns"],
                    logical_name="hospitals",
                )
                if _sqlite_row_count(conn, table_name=hospitals_table_name) == 0:
                    _seed_sqlite_table_from_csv(
                        conn,
                        logical_table="hospitals",
                        table_name=hospitals_table_name,
                        csv_path=hospital_csv_path,
                    )
                _ensure_sqlite_hospital_component_rows(
                    conn,
                    table_name=hospitals_table_name,
                )
                conn.execute(
                    f'CREATE INDEX IF NOT EXISTS "{hospitals_table_name}_hospital_blood_date_idx" '
                    f'ON "{hospitals_table_name}" ("hospital_id", "blood_type", "date")'
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        _SQLITE_BOOTSTRAP_CACHE.add(cache_key)
    return sqlite_path


def _build_sqlite_filter_clauses(
    filters: dict[str, Any] | None,
    *,
    config: dict[str, Any],
) -> tuple[list[str], list[Any]]:
    filters = _normalize_reference_filters(filters, config=config)
    if not isinstance(filters, dict):
        return [], []

    clauses: list[str] = []
    params: list[Any] = []
    allowed_columns = config["columns"]
    for raw_field, raw_value in sorted(filters.items()):
        field_name = _validate_identifier(raw_field, "filter field")
        if field_name not in allowed_columns:
            logger.debug("Skipping unsupported filter field '%s' for table.", field_name)
            continue
        if isinstance(raw_value, list):
            values = [
                _sqlite_value(value)
                for value in raw_value
                if not _is_blank_seed_value(value)
            ]
            if not values:
                continue
            clauses.append(f'"{field_name}" IN ({", ".join("?" for _ in values)})')
            params.extend(values)
            continue
        if raw_value is None:
            clauses.append(f'"{field_name}" IS NULL')
            continue
        clauses.append(f'"{field_name}" = ?')
        params.append(_sqlite_value(raw_value))
    return clauses, params


def _preview_rows(rows: list[dict[str, Any]], *, limit: int = 3) -> str:
    preview = rows[:limit]
    if not preview:
        return ""
    return json.dumps(preview, ensure_ascii=True, default=str)


def _normalize_structured_db_request(
    *,
    query: str | None,
    logical_table: str,
    field: str,
    aggregate: str,
    sort: str | None,
    limit: Any,
    group_by: str | None,
    config: dict[str, Any],
) -> tuple[str, str, str, int]:
    requested_field = "*" if str(field or "").strip() == "*" else _validate_identifier(field, "field")
    aggregate_name = _normalize_aggregate(aggregate)
    sort_order = "ASC" if str(sort or "").strip().lower() == "asc" else "DESC"
    normalized_limit = _normalize_db_limit(limit)

    if requested_field == "*":
        return requested_field, aggregate_name, sort_order, normalized_limit

    alias_key = safe_slug(requested_field)
    field_aliases = config.get("field_aliases", {})
    inverse_aliases = set(config.get("inverse_field_aliases", set()))
    resolved_field = field_aliases.get(alias_key, requested_field)
    lowered_query = str(query or "").lower()

    if alias_key in inverse_aliases:
        if aggregate_name == "max":
            aggregate_name = "min"
        elif aggregate_name == "min":
            aggregate_name = "max"
        sort_order = "DESC" if sort_order == "ASC" else "ASC"

    if logical_table == "donors" and not group_by:
        age_like_fields = {"age", "birthdate", "date_of_birth", "dob", "donor_age"}
        if alias_key in age_like_fields or resolved_field == "age":
            if "oldest" in lowered_query:
                resolved_field = "age"
                aggregate_name = "max"
                sort_order = "DESC"
                if limit is None:
                    normalized_limit = 1
            elif "youngest" in lowered_query:
                resolved_field = "age"
                aggregate_name = "min"
                sort_order = "ASC"
                if limit is None:
                    normalized_limit = 1

    return resolved_field, aggregate_name, sort_order, normalized_limit


def _normalize_reference_list_fields(
    fields: list[str] | str | None,
    *,
    config: dict[str, Any],
) -> list[str]:
    if isinstance(fields, str):
        fields = [fields]
    elif fields is None:
        fields = []
    selected = _dedupe_list_fields(fields, config)
    if not selected:
        raise RuntimeError("No supported list fields were inferred for this query.")
    identity_fields = _dedupe_list_fields(list(config.get("identity_fields") or []), config)
    data_fields = [field for field in selected if field not in identity_fields]
    if data_fields:
        selected = _dedupe_list_fields(identity_fields + data_fields, config)
    return selected


def _execute_reference_list_postgres(
    *,
    query: str | None,
    table: str,
    fields: list[str] | str | None,
    filters: dict[str, Any] | None,
    limit: Any,
    latest_only: bool,
    db_schema: str,
    donors_table_name: str,
    hospitals_table_name: str,
    donor_csv_path: Path,
    hospital_csv_path: Path,
) -> dict[str, Any]:
    logical_table, actual_table_name, config = _resolve_table_request(
        table,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
    )
    schema_name = _normalize_db_schema(db_schema)
    selected_fields = _normalize_reference_list_fields(fields, config=config)
    normalized_limit = _normalize_db_limit(limit)
    normalized_filters = _normalize_reference_filters(filters, config=config)
    ensure_postgres_reference_tables(
        db_schema=schema_name,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
        donor_csv_path=donor_csv_path,
        hospital_csv_path=hospital_csv_path,
        required_tables={logical_table},
    )

    where_clauses, params = _build_filter_clauses(normalized_filters, config=config)
    table_sql = _qualified_table(schema_name, actual_table_name)
    if latest_only and "date" in config["columns"]:
        where_clauses.append(
            sql.SQL("{} = (SELECT MAX({}) FROM {})").format(
                sql.Identifier("date"),
                sql.Identifier("date"),
                table_sql,
            )
        )
    where_sql = (
        sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)
        if where_clauses
        else sql.SQL("")
    )
    select_list = sql.SQL(", ").join(
        sql.SQL("{} AS {}").format(sql.Identifier(field), sql.Identifier(field))
        for field in selected_fields
    )
    order_fields = [
        field
        for field in config.get("identity_fields", [])
        if field in selected_fields
    ] or selected_fields
    order_sql = sql.SQL(", ").join(sql.Identifier(field) for field in order_fields)

    conn = _postgres_connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                sql.SQL(
                    "SELECT DISTINCT {select_list} FROM {table}{where_clause} "
                    "ORDER BY {order_fields} ASC LIMIT %s"
                ).format(
                    select_list=select_list,
                    table=table_sql,
                    where_clause=where_sql,
                    order_fields=order_sql,
                ),
                [*params, normalized_limit + 1],
            )
            fetched = cur.fetchall()
    finally:
        conn.close()

    rows = [_serialize_db_row(dict(row)) for row in fetched[:normalized_limit]]
    preview = _preview_rows(rows)
    return {
        "tool_type": "db_tool",
        "provider": f"postgres:{schema_name}.{actual_table_name}",
        "table": logical_table,
        "fields": selected_fields,
        "aggregate": "list",
        "filters": normalized_filters or {},
        "limit": normalized_limit,
        "latest_only": latest_only,
        "truncated": len(fetched) > normalized_limit,
        "rows": rows,
        "answer": (
            f"Returned {len(rows)} bounded result row(s) for list of "
            f"{logical_table} columns: {', '.join(selected_fields)}."
            + (f" Preview: {preview}" if preview else "")
        ),
    }


def _execute_structured_postgres_query(
    *,
    query: str | None,
    table: str,
    field: str,
    aggregate: str,
    group_by: str | None,
    filters: dict[str, Any] | None,
    limit: Any,
    sort: str | None,
    db_schema: str,
    donors_table_name: str,
    hospitals_table_name: str,
    donor_csv_path: Path,
    hospital_csv_path: Path,
) -> dict[str, Any]:
    logical_table, actual_table_name, config = _resolve_table_request(
        table,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
    )
    schema_name = _normalize_db_schema(db_schema)
    requested_field = "*" if str(field or "").strip() == "*" else _validate_identifier(field, "field")
    group_by_name = (
        _validate_identifier(group_by, "group_by field") if group_by else None
    )
    field_name, aggregate_name, sort_order, normalized_limit = _normalize_structured_db_request(
        query=query,
        logical_table=logical_table,
        field=requested_field,
        aggregate=aggregate,
        sort=sort,
        limit=limit,
        group_by=group_by_name,
        config=config,
    )

    if field_name != "*" and field_name not in config["columns"]:
        raise RuntimeError(f"Unsupported field '{field_name}' for table '{logical_table}'.")
    if group_by_name and group_by_name not in config["columns"]:
        raise RuntimeError(
            f"Unsupported group_by field '{group_by_name}' for table '{logical_table}'."
        )

    ensure_postgres_reference_tables(
        db_schema=schema_name,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
        donor_csv_path=donor_csv_path,
        hospital_csv_path=hospital_csv_path,
        required_tables={logical_table},
    )

    normalized_filters = _normalize_reference_filters(filters, config=config)
    where_clauses, params = _build_filter_clauses(normalized_filters, config=config)
    where_sql = (
        sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)
        if where_clauses
        else sql.SQL("")
    )
    table_sql = _qualified_table(schema_name, actual_table_name)
    rows: list[dict[str, Any]]
    query_label = f"{aggregate_name} on {logical_table}.{field_name}"
    has_more_rows = False

    conn = _postgres_connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if aggregate_name == "list":
                if field_name == "*":
                    raise RuntimeError("list requires a concrete field.")
                cur.execute(
                    sql.SQL(
                        "SELECT DISTINCT {field} AS {field_alias} "
                        "FROM {table}{where_clause}{extra_predicate} "
                        "ORDER BY {field} ASC "
                        "LIMIT %s"
                    ).format(
                        field=sql.Identifier(field_name),
                        field_alias=sql.Identifier(field_name),
                        table=table_sql,
                        where_clause=where_sql,
                        extra_predicate=(
                            sql.SQL(" AND {} IS NOT NULL").format(sql.Identifier(field_name))
                            if where_clauses
                            else sql.SQL(" WHERE {} IS NOT NULL").format(sql.Identifier(field_name))
                        ),
                    ),
                    [*params, normalized_limit + 1],
                )
                fetched = cur.fetchall()
                has_more_rows = len(fetched) > normalized_limit
                rows = [_serialize_db_row(dict(row)) for row in fetched[:normalized_limit]]
                query_label = f"list of {logical_table}.{field_name}"
            elif aggregate_name == "count":
                grouping_field = group_by_name or (field_name if field_name != "*" else None)
                if grouping_field is None:
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) AS value FROM {}{}").format(
                            table_sql,
                            where_sql,
                        ),
                        params,
                    )
                    scalar_row = cur.fetchone() or {"value": 0}
                    rows = [_serialize_db_row(dict(scalar_row))]
                else:
                    cur.execute(
                        sql.SQL(
                            "SELECT {group_field} AS group_value, COUNT(*) AS value "
                            "FROM {table}{where_clause} "
                            "GROUP BY {group_field} "
                            "ORDER BY value DESC, group_value ASC "
                            "LIMIT %s"
                        ).format(
                            group_field=sql.Identifier(grouping_field),
                            table=table_sql,
                            where_clause=where_sql,
                        ),
                        [*params, normalized_limit + 1],
                    )
                    fetched = cur.fetchall()
                    has_more_rows = len(fetched) > normalized_limit
                    rows = [_serialize_db_row(dict(row)) for row in fetched[:normalized_limit]]
                    query_label = f"count by {logical_table}.{grouping_field}"
            elif aggregate_name == "distinct_count":
                if field_name == "*":
                    raise RuntimeError("distinct_count requires a concrete field.")
                if group_by_name:
                    cur.execute(
                        sql.SQL(
                            "SELECT {group_by} AS group_value, COUNT(DISTINCT {field}) AS value "
                            "FROM {table}{where_clause} "
                            "GROUP BY {group_by} "
                            "ORDER BY value DESC, group_value ASC "
                            "LIMIT %s"
                        ).format(
                            group_by=sql.Identifier(group_by_name),
                            field=sql.Identifier(field_name),
                            table=table_sql,
                            where_clause=where_sql,
                        ),
                        [*params, normalized_limit + 1],
                    )
                    fetched = cur.fetchall()
                    has_more_rows = len(fetched) > normalized_limit
                    rows = [_serialize_db_row(dict(row)) for row in fetched[:normalized_limit]]
                    query_label = f"distinct count of {field_name} by {group_by_name}"
                else:
                    cur.execute(
                        sql.SQL(
                            "SELECT COUNT(DISTINCT {field}) AS value FROM {table}{where_clause}"
                        ).format(
                            field=sql.Identifier(field_name),
                            table=table_sql,
                            where_clause=where_sql,
                        ),
                        params,
                    )
                    scalar_row = cur.fetchone() or {"value": 0}
                    rows = [_serialize_db_row(dict(scalar_row))]
            elif aggregate_name in {"avg", "sum"}:
                if field_name == "*":
                    raise RuntimeError(f"{aggregate_name} requires a numeric field.")
                field_type = config["columns"].get(field_name)
                if field_type not in _NUMERIC_COLUMN_TYPES:
                    raise RuntimeError(
                        f"Field '{field_name}' is not numeric and cannot use aggregate '{aggregate_name}'."
                    )
                agg_sql = sql.SQL("AVG") if aggregate_name == "avg" else sql.SQL("SUM")
                if group_by_name:
                    cur.execute(
                        sql.SQL(
                            "SELECT {group_by} AS group_value, {aggregate}({field}) AS value "
                            "FROM {table}{where_clause} "
                            "GROUP BY {group_by} "
                            "ORDER BY value {sort_order}, group_value ASC "
                            "LIMIT %s"
                        ).format(
                            group_by=sql.Identifier(group_by_name),
                            aggregate=agg_sql,
                            field=sql.Identifier(field_name),
                            table=table_sql,
                            where_clause=where_sql,
                            sort_order=sql.SQL(sort_order),
                        ),
                        [*params, normalized_limit + 1],
                    )
                    fetched = cur.fetchall()
                    has_more_rows = len(fetched) > normalized_limit
                    rows = [_serialize_db_row(dict(row)) for row in fetched[:normalized_limit]]
                    query_label = f"{aggregate_name} of {field_name} by {group_by_name}"
                else:
                    cur.execute(
                        sql.SQL(
                            "SELECT {aggregate}({field}) AS value FROM {table}{where_clause}"
                        ).format(
                            aggregate=agg_sql,
                            field=sql.Identifier(field_name),
                            table=table_sql,
                            where_clause=where_sql,
                        ),
                        params,
                    )
                    scalar_row = cur.fetchone() or {"value": None}
                    rows = [_serialize_db_row(dict(scalar_row))]
            elif aggregate_name in {"min", "max"}:
                if field_name == "*":
                    raise RuntimeError(f"{aggregate_name} requires a concrete field.")
                agg_sql = sql.SQL("MIN") if aggregate_name == "min" else sql.SQL("MAX")
                direction = sql.SQL(sort_order)
                if group_by_name:
                    cur.execute(
                        sql.SQL(
                            "SELECT {group_by} AS group_value, {aggregate}({field}) AS value "
                            "FROM {table}{where_clause} "
                            "GROUP BY {group_by} "
                            "ORDER BY value {direction}, group_value ASC "
                            "LIMIT %s"
                        ).format(
                            group_by=sql.Identifier(group_by_name),
                            aggregate=agg_sql,
                            field=sql.Identifier(field_name),
                            table=table_sql,
                            where_clause=where_sql,
                            direction=direction,
                        ),
                        [*params, normalized_limit + 1],
                    )
                    fetched = cur.fetchall()
                    has_more_rows = len(fetched) > normalized_limit
                    rows = [_serialize_db_row(dict(row)) for row in fetched[:normalized_limit]]
                    query_label = f"{aggregate_name} of {field_name} by {group_by_name}"
                else:
                    identity_fields = [
                        name
                        for name in config["identity_fields"]
                        if name in config["columns"] and name != field_name
                    ]
                    select_fields = [
                        sql.SQL("{} AS {}").format(
                            sql.Identifier(name),
                            sql.Identifier(name),
                        )
                        for name in identity_fields
                    ]
                    select_fields.append(
                        sql.SQL("{} AS value").format(sql.Identifier(field_name))
                    )
                    cur.execute(
                        sql.SQL(
                            "SELECT {select_list} "
                            "FROM {table}{where_clause}{extra_predicate} "
                            "ORDER BY {field} {direction} "
                            "LIMIT %s"
                        ).format(
                            select_list=sql.SQL(", ").join(select_fields),
                            table=table_sql,
                            where_clause=where_sql,
                            extra_predicate=(
                                sql.SQL(" AND {} IS NOT NULL").format(sql.Identifier(field_name))
                                if where_clauses
                                else sql.SQL(" WHERE {} IS NOT NULL").format(
                                    sql.Identifier(field_name)
                                )
                            ),
                            field=sql.Identifier(field_name),
                            direction=direction,
                        ),
                        [*params, normalized_limit + 1],
                    )
                    fetched = cur.fetchall()
                    has_more_rows = len(fetched) > normalized_limit
                    rows = [_serialize_db_row(dict(row)) for row in fetched[:normalized_limit]]
                    query_label = f"top {normalized_limit} {aggregate_name} values for {field_name}"
            else:
                raise RuntimeError(f"Unsupported aggregate '{aggregate_name}'.")
    finally:
        conn.close()

    truncated = has_more_rows
    preview = _preview_rows(rows)
    return {
        "tool_type": "db_tool",
        "provider": f"postgres:{schema_name}.{actual_table_name}",
        "table": logical_table,
        "requested_field": requested_field,
        "field": field_name,
        "aggregate": aggregate_name,
        "group_by": group_by_name,
        "filters": normalized_filters or {},
        "limit": normalized_limit,
        "truncated": truncated,
        "rows": rows,
        "answer": (
            f"Returned {len(rows)} bounded result row(s) for {query_label}."
            + (f" Preview: {preview}" if preview else "")
        ),
    }


def _execute_structured_sqlite_query(
    *,
    query: str | None,
    table: str,
    field: str,
    aggregate: str,
    group_by: str | None,
    filters: dict[str, Any] | None,
    limit: Any,
    sort: str | None,
    db_sqlite_path: str | None,
    donors_table_name: str,
    hospitals_table_name: str,
    donor_csv_path: Path,
    hospital_csv_path: Path,
    postgres_error: Exception | None = None,
) -> dict[str, Any]:
    logical_table, actual_table_name, config = _resolve_table_request(
        table,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
    )
    requested_field = "*" if str(field or "").strip() == "*" else _validate_identifier(field, "field")
    group_by_name = (
        _validate_identifier(group_by, "group_by field") if group_by else None
    )
    field_name, aggregate_name, sort_order, normalized_limit = _normalize_structured_db_request(
        query=query,
        logical_table=logical_table,
        field=requested_field,
        aggregate=aggregate,
        sort=sort,
        limit=limit,
        group_by=group_by_name,
        config=config,
    )
    if field_name != "*" and field_name not in config["columns"]:
        raise RuntimeError(f"Unsupported field '{field_name}' for table '{logical_table}'.")
    if group_by_name and group_by_name not in config["columns"]:
        raise RuntimeError(
            f"Unsupported group_by field '{group_by_name}' for table '{logical_table}'."
        )

    sqlite_path = ensure_sqlite_reference_tables(
        db_sqlite_path=db_sqlite_path,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
        donor_csv_path=donor_csv_path,
        hospital_csv_path=hospital_csv_path,
        required_tables={logical_table},
    )
    normalized_filters = _normalize_reference_filters(filters, config=config)
    where_clauses, params = _build_sqlite_filter_clauses(
        normalized_filters,
        config=config,
    )
    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    rows: list[dict[str, Any]]
    query_label = f"{aggregate_name} on {logical_table}.{field_name}"
    has_more_rows = False

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        if aggregate_name == "list":
            if field_name == "*":
                raise RuntimeError("list requires a concrete field.")
            extra_predicate = (
                f' AND "{field_name}" IS NOT NULL'
                if where_clauses
                else f' WHERE "{field_name}" IS NOT NULL'
            )
            fetched = conn.execute(
                f'SELECT DISTINCT "{field_name}" AS "{field_name}" '
                f'FROM "{actual_table_name}"{where_sql}{extra_predicate} '
                f'ORDER BY "{field_name}" ASC LIMIT ?',
                [*params, normalized_limit + 1],
            ).fetchall()
            has_more_rows = len(fetched) > normalized_limit
            rows = [dict(row) for row in fetched[:normalized_limit]]
            query_label = f"list of {logical_table}.{field_name}"
        elif aggregate_name == "count":
            grouping_field = group_by_name or (field_name if field_name != "*" else None)
            if grouping_field is None:
                fetched = conn.execute(
                    f'SELECT COUNT(*) AS value FROM "{actual_table_name}"{where_sql}',
                    params,
                ).fetchall()
                rows = [dict(fetched[0])] if fetched else [{"value": 0}]
            else:
                fetched = conn.execute(
                    f'SELECT "{grouping_field}" AS group_value, COUNT(*) AS value '
                    f'FROM "{actual_table_name}"{where_sql} '
                    f'GROUP BY "{grouping_field}" '
                    f'ORDER BY value DESC, group_value ASC LIMIT ?',
                    [*params, normalized_limit + 1],
                ).fetchall()
                has_more_rows = len(fetched) > normalized_limit
                rows = [dict(row) for row in fetched[:normalized_limit]]
                query_label = f"count by {logical_table}.{grouping_field}"
        elif aggregate_name == "distinct_count":
            if field_name == "*":
                raise RuntimeError("distinct_count requires a concrete field.")
            if group_by_name:
                fetched = conn.execute(
                    f'SELECT "{group_by_name}" AS group_value, COUNT(DISTINCT "{field_name}") AS value '
                    f'FROM "{actual_table_name}"{where_sql} '
                    f'GROUP BY "{group_by_name}" '
                    f'ORDER BY value DESC, group_value ASC LIMIT ?',
                    [*params, normalized_limit + 1],
                ).fetchall()
                has_more_rows = len(fetched) > normalized_limit
                rows = [dict(row) for row in fetched[:normalized_limit]]
                query_label = f"distinct count of {field_name} by {group_by_name}"
            else:
                fetched = conn.execute(
                    f'SELECT COUNT(DISTINCT "{field_name}") AS value FROM "{actual_table_name}"{where_sql}',
                    params,
                ).fetchall()
                rows = [dict(fetched[0])] if fetched else [{"value": 0}]
        elif aggregate_name in {"avg", "sum"}:
            if field_name == "*":
                raise RuntimeError(f"{aggregate_name} requires a numeric field.")
            field_type = config["columns"].get(field_name)
            if field_type not in _NUMERIC_COLUMN_TYPES:
                raise RuntimeError(
                    f"Field '{field_name}' is not numeric and cannot use aggregate '{aggregate_name}'."
                )
            agg_sql = "AVG" if aggregate_name == "avg" else "SUM"
            if group_by_name:
                fetched = conn.execute(
                    f'SELECT "{group_by_name}" AS group_value, {agg_sql}("{field_name}") AS value '
                    f'FROM "{actual_table_name}"{where_sql} '
                    f'GROUP BY "{group_by_name}" '
                    f'ORDER BY value {sort_order}, group_value ASC LIMIT ?',
                    [*params, normalized_limit + 1],
                ).fetchall()
                has_more_rows = len(fetched) > normalized_limit
                rows = [dict(row) for row in fetched[:normalized_limit]]
                query_label = f"{aggregate_name} of {field_name} by {group_by_name}"
            else:
                fetched = conn.execute(
                    f'SELECT {agg_sql}("{field_name}") AS value FROM "{actual_table_name}"{where_sql}',
                    params,
                ).fetchall()
                rows = [dict(fetched[0])] if fetched else [{"value": None}]
        elif aggregate_name in {"min", "max"}:
            if field_name == "*":
                raise RuntimeError(f"{aggregate_name} requires a concrete field.")
            agg_sql = "MIN" if aggregate_name == "min" else "MAX"
            if group_by_name:
                fetched = conn.execute(
                    f'SELECT "{group_by_name}" AS group_value, {agg_sql}("{field_name}") AS value '
                    f'FROM "{actual_table_name}"{where_sql} '
                    f'GROUP BY "{group_by_name}" '
                    f'ORDER BY value {sort_order}, group_value ASC LIMIT ?',
                    [*params, normalized_limit + 1],
                ).fetchall()
                has_more_rows = len(fetched) > normalized_limit
                rows = [dict(row) for row in fetched[:normalized_limit]]
                query_label = f"{aggregate_name} of {field_name} by {group_by_name}"
            else:
                identity_fields = [
                    name
                    for name in config["identity_fields"]
                    if name in config["columns"] and name != field_name
                ]
                select_list = ", ".join(
                    [f'"{name}" AS "{name}"' for name in identity_fields]
                    + [f'"{field_name}" AS value']
                )
                extra_predicate = (
                    f' AND "{field_name}" IS NOT NULL'
                    if where_clauses
                    else f' WHERE "{field_name}" IS NOT NULL'
                )
                fetched = conn.execute(
                    f'SELECT {select_list} FROM "{actual_table_name}"{where_sql}{extra_predicate} '
                    f'ORDER BY "{field_name}" {sort_order} LIMIT ?',
                    [*params, normalized_limit + 1],
                ).fetchall()
                has_more_rows = len(fetched) > normalized_limit
                rows = [dict(row) for row in fetched[:normalized_limit]]
                query_label = f"top {normalized_limit} {aggregate_name} values for {field_name}"
        else:
            raise RuntimeError(f"Unsupported aggregate '{aggregate_name}'.")
    finally:
        conn.close()

    rows = [_serialize_db_row(row) for row in rows]
    preview = _preview_rows(rows)
    payload = {
        "tool_type": "db_tool",
        "provider": f"sqlite:{sqlite_path}:{actual_table_name}",
        "table": logical_table,
        "requested_field": requested_field,
        "field": field_name,
        "aggregate": aggregate_name,
        "group_by": group_by_name,
        "filters": normalized_filters or {},
        "limit": normalized_limit,
        "truncated": has_more_rows,
        "rows": rows,
        "answer": (
            f"Returned {len(rows)} bounded SQLite fallback row(s) for {query_label}."
            + (f" Preview: {preview}" if preview else "")
        ),
    }
    if postgres_error is not None:
        payload["postgres_error"] = str(postgres_error)
    return payload


def _execute_reference_list_sqlite(
    *,
    query: str | None,
    table: str,
    fields: list[str] | str | None,
    filters: dict[str, Any] | None,
    limit: Any,
    latest_only: bool,
    db_sqlite_path: str | None,
    donors_table_name: str,
    hospitals_table_name: str,
    donor_csv_path: Path,
    hospital_csv_path: Path,
    postgres_error: Exception | None = None,
) -> dict[str, Any]:
    logical_table, actual_table_name, config = _resolve_table_request(
        table,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
    )
    selected_fields = _normalize_reference_list_fields(fields, config=config)
    normalized_limit = _normalize_db_limit(limit)
    normalized_filters = _normalize_reference_filters(filters, config=config)
    sqlite_path = ensure_sqlite_reference_tables(
        db_sqlite_path=db_sqlite_path,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
        donor_csv_path=donor_csv_path,
        hospital_csv_path=hospital_csv_path,
        required_tables={logical_table},
    )
    where_clauses, params = _build_sqlite_filter_clauses(
        normalized_filters,
        config=config,
    )
    if latest_only and "date" in config["columns"]:
        where_clauses.append(
            f'"date" = (SELECT MAX("date") FROM "{actual_table_name}")'
        )
    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    select_list = ", ".join(f'"{field}" AS "{field}"' for field in selected_fields)
    order_fields = [
        field
        for field in config.get("identity_fields", [])
        if field in selected_fields
    ] or selected_fields
    order_sql = ", ".join(f'"{field}" ASC' for field in order_fields)

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        fetched = conn.execute(
            f'SELECT DISTINCT {select_list} FROM "{actual_table_name}"{where_sql} '
            f"ORDER BY {order_sql} LIMIT ?",
            [*params, normalized_limit + 1],
        ).fetchall()
    finally:
        conn.close()

    rows = [_serialize_db_row(dict(row)) for row in fetched[:normalized_limit]]
    preview = _preview_rows(rows)
    payload = {
        "tool_type": "db_tool",
        "provider": f"sqlite:{sqlite_path}:{actual_table_name}",
        "table": logical_table,
        "fields": selected_fields,
        "aggregate": "list",
        "filters": normalized_filters or {},
        "limit": normalized_limit,
        "latest_only": latest_only,
        "truncated": len(fetched) > normalized_limit,
        "rows": rows,
        "answer": (
            f"Returned {len(rows)} bounded SQLite fallback row(s) for list of "
            f"{logical_table} columns: {', '.join(selected_fields)}."
            + (f" Preview: {preview}" if preview else "")
        ),
    }
    if postgres_error is not None:
        payload["postgres_error"] = str(postgres_error)
    return payload


def load_supply_series_from_csv(
    csv_path: Path,
    blood_type: str | None,
) -> list[dict[str, Any]]:
    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")

    by_date: dict[str, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not isinstance(row, dict):
                continue
            date_value = str(row.get("date", "")).strip()
            if not date_value:
                continue

            row_blood_type = str(row.get("blood_type", "")).strip().upper()
            if blood_type and row_blood_type != blood_type:
                continue

            bucket = by_date.setdefault(
                date_value,
                {
                    "date": date_value,
                    "stock_end_total": 0.0,
                    "units_collected_total": 0.0,
                    "units_used_total": 0.0,
                    "records": 0,
                },
            )
            bucket["stock_end_total"] += as_float(row.get("stock_end"), 0.0)
            bucket["units_collected_total"] += as_float(row.get("units_collected"), 0.0)
            bucket["units_used_total"] += as_float(row.get("units_used"), 0.0)
            bucket["records"] = int(bucket["records"]) + 1

    series = [by_date[key] for key in sorted(by_date.keys())]
    if not series:
        label = blood_type or "all blood types"
        raise RuntimeError(f"No supply rows found for {label} in {csv_path}.")
    return series


def load_supply_series_from_sqlite(
    db_path: Path,
    blood_type: str | None,
    table_name: str | None = None,
) -> list[dict[str, Any]]:
    if not db_path.exists():
        raise RuntimeError(f"SQLite DB not found: {db_path}")

    table_name = table_name or db_reference_default_sqlite_table()
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
        table_name = db_reference_default_sqlite_table()

    where_clause = ""
    params: list[Any] = []
    if blood_type:
        where_clause = "WHERE UPPER(REPLACE(CAST(blood_type AS TEXT), ' ', '')) = ?"
        params.append(blood_type.upper())

    sql = (
        f"SELECT date, "
        f"SUM(COALESCE(stock_end, 0)) AS stock_end_total, "
        f"SUM(COALESCE(units_collected, 0)) AS units_collected_total, "
        f"SUM(COALESCE(units_used, 0)) AS units_used_total, "
        f"COUNT(*) AS records "
        f"FROM {table_name} "
        f"{where_clause} "
        f"GROUP BY date "
        f"ORDER BY date ASC"
    )

    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        raise RuntimeError(f"SQLite query failed on table '{table_name}': {exc}") from exc
    finally:
        conn.close()

    series = [
        {
            "date": str(row["date"]),
            "stock_end_total": float(row["stock_end_total"] or 0.0),
            "units_collected_total": float(row["units_collected_total"] or 0.0),
            "units_used_total": float(row["units_used_total"] or 0.0),
            "records": int(row["records"] or 0),
        }
        for row in rows
    ]
    if not series:
        label = blood_type or "all blood types"
        raise RuntimeError(
            f"No supply rows found for {label} in SQLite table '{table_name}'."
        )
    return series


def load_donor_rows_from_csv(
    csv_path: Path,
    *,
    country_code: str | None = None,
    blood_type: str | None = None,
) -> list[dict[str, Any]]:
    if not csv_path.exists():
        raise RuntimeError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = [str(name).strip() for name in (reader.fieldnames or []) if name]
        header_set = {name.lower() for name in header}
        if "donor_id" not in header_set or "blood_type" not in header_set:
            raise RuntimeError(f"CSV is not a donor registry dataset: {csv_path.name}")

        rows: list[dict[str, Any]] = []
        for row in reader:
            if not isinstance(row, dict):
                continue
            row_country = str(row.get("country_code", "")).strip().upper()
            row_blood_type = str(row.get("blood_type", "")).strip().upper()
            if country_code and row_country != country_code:
                continue
            if blood_type and row_blood_type != blood_type:
                continue
            rows.append(row)

    if not rows:
        scope_parts = []
        if country_code:
            scope_parts.append(country_code)
        if blood_type:
            scope_parts.append(blood_type)
        label = " / ".join(scope_parts) if scope_parts else "requested scope"
        raise RuntimeError(f"No donor rows found for {label} in {csv_path.name}.")
    return rows


def _compute_blood_type_prevalence(
    rows: list[dict[str, Any]],
    *,
    country_code: str | None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        blood_type = str(row.get("blood_type", "")).strip().upper()
        if not blood_type:
            continue
        counts[blood_type] = counts.get(blood_type, 0) + 1

    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("No blood type values found in donor registry.")

    output_rows = [
        {
            "blood_type": blood_type,
            "count": count,
            "prevalence_pct": round((count / total) * 100.0, 2),
        }
        for blood_type, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    scope = f"{country_code} donor registry" if country_code else "donor registry"
    preview = ", ".join(
        f"{row['blood_type']} {row['prevalence_pct']:.1f}%"
        for row in output_rows[:3]
    )
    return {
        "query_kind": "blood_type_prevalence",
        "total_population": total,
        "rows": output_rows,
        "answer": f"Blood type prevalence in the {scope}: {preview}.",
    }


def _is_eligible_donor_row(row: dict[str, Any]) -> bool:
    eligible_raw = row.get("eligible_to_donate")
    if eligible_raw not in {None, ""}:
        return _truthy(eligible_raw)
    eligibility_status = str(row.get("eligibility_status", "")).strip().lower()
    if eligibility_status:
        return eligibility_status == "eligible"
    deferral_reason = str(row.get("deferral_reason", "")).strip().lower()
    return deferral_reason in {"", "none", "null"}


def _donor_outreach_score(row: dict[str, Any]) -> tuple[float, list[str]]:
    propensity_raw = row.get("donation_propensity_score")
    propensity = None
    if propensity_raw not in {None, ""}:
        propensity = max(0.0, min(1.0, as_float(propensity_raw, 0.0)))

    regular = _truthy(row.get("is_regular_donor"))
    rare = _truthy(row.get("is_rare_type"))
    recent_donations = max(0.0, as_float(row.get("donation_count_last_12m"), 0.0))
    lifetime = max(0.0, as_float(row.get("lifetime_donation_count"), 0.0))
    next_6m_count = max(0.0, as_float(row.get("next_6m_donation_count"), 0.0))

    recency = None
    recency_raw = row.get("recency_days")
    if recency_raw not in {None, ""}:
        recency = max(0.0, as_float(recency_raw, 0.0))

    derived_propensity = 0.32
    derived_propensity += min(0.20, recent_donations * 0.06)
    derived_propensity += min(0.12, lifetime * 0.015)
    derived_propensity += min(0.16, next_6m_count * 0.08)
    if regular:
        derived_propensity += 0.10
    if rare:
        derived_propensity += 0.08
    if recency is not None:
        if recency < 30:
            derived_propensity -= 0.14
        elif recency < 56:
            derived_propensity -= 0.05
        elif recency <= 180:
            derived_propensity += 0.10
        elif recency <= 365:
            derived_propensity += 0.06
        else:
            derived_propensity += 0.02

    base = propensity if propensity is not None else derived_propensity
    score = base
    score += min(0.07, recent_donations * 0.02)
    if regular:
        score += 0.06
    if rare:
        score += 0.05
    if recency is not None and 56 <= recency <= 365:
        score += 0.04
    score = max(0.0, min(0.99, score))

    rationale: list[str] = []
    if propensity is not None and propensity >= 0.6:
        rationale.append("high propensity")
    if regular:
        rationale.append("regular donor")
    if rare:
        rationale.append("rare blood type")
    if recent_donations >= 2:
        rationale.append("recent donation history")
    if recency is not None and 56 <= recency <= 365:
        rationale.append("good recontact window")
    if not rationale:
        rationale.append("eligible donor profile")
    return score, rationale


def _recommend_donor_outreach(
    rows: list[dict[str, Any]],
    *,
    country_code: str | None,
    blood_type: str | None,
    top_n: int,
) -> dict[str, Any]:
    eligible_rows = [row for row in rows if _is_eligible_donor_row(row)]
    if not eligible_rows:
        raise RuntimeError("No eligible donors found in the local donor registry.")

    ranked_rows: list[dict[str, Any]] = []
    for row in eligible_rows:
        score, rationale = _donor_outreach_score(row)
        ranked_rows.append(
            {
                "donor_id": str(row.get("donor_id", "")).strip(),
                "blood_type": str(row.get("blood_type", "")).strip().upper(),
                "country_code": str(row.get("country_code", "")).strip().upper(),
                "region": str(row.get("region", "")).strip(),
                "preferred_site": str(row.get("preferred_site", "")).strip(),
                "recommendation_score": round(score, 4),
                "donation_propensity_score": (
                    round(as_float(row.get("donation_propensity_score"), 0.0), 4)
                    if row.get("donation_propensity_score") not in {None, ""}
                    else None
                ),
                "recency_days": (
                    int(as_float(row.get("recency_days"), 0.0))
                    if row.get("recency_days") not in {None, ""}
                    else None
                ),
                "donation_count_last_12m": int(as_float(row.get("donation_count_last_12m"), 0.0)),
                "is_regular_donor": _truthy(row.get("is_regular_donor")),
                "is_rare_type": _truthy(row.get("is_rare_type")),
                "rationale": ", ".join(rationale),
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            -float(row["recommendation_score"]),
            -int(bool(row["is_rare_type"])),
            -int(bool(row["is_regular_donor"])),
            row["donor_id"],
        )
    )
    selected = ranked_rows[:top_n]
    if not selected:
        raise RuntimeError("No strong donor candidates found in the local registry.")

    scope_parts = []
    if country_code:
        scope_parts.append(country_code)
    if blood_type:
        scope_parts.append(blood_type)
    scope = " / ".join(scope_parts) if scope_parts else "local donor registry"
    preview = ", ".join(
        f"{row['donor_id']} ({row['blood_type']}, score {row['recommendation_score']:.2f})"
        for row in selected[:3]
    )
    strongest_score = float(selected[0]["recommendation_score"])
    decision = "Yes" if strongest_score >= 0.45 else "Maybe"
    return {
        "query_kind": "donor_outreach",
        "total_candidates": len(ranked_rows),
        "recommended_count": len(selected),
        "rows": selected,
        "answer": (
            f"{decision}. Found {len(selected)} eligible donor candidates for next month in {scope}. "
            f"Prioritize {preview}."
        ),
    }


def _execute_blood_prevalence_postgres(
    *,
    query: str,
    country_code: str | None,
    blood_type: str | None,
    db_schema: str,
    donors_table_name: str,
    hospitals_table_name: str,
    donor_csv_path: Path,
    hospital_csv_path: Path,
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if country_code:
        filters["country_code"] = country_code
    if blood_type:
        filters["blood_type"] = blood_type

    payload = _execute_structured_postgres_query(
        query=query,
        table="donors",
        field="blood_type",
        aggregate="count",
        group_by=None,
        filters=filters,
        limit=_db_max_limit(),
        sort="desc",
        db_schema=db_schema,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
        donor_csv_path=donor_csv_path,
        hospital_csv_path=hospital_csv_path,
    )

    count_rows = payload.get("rows", [])
    total = sum(int(as_int(row.get("value"), 0)) for row in count_rows)
    if total <= 0:
        raise RuntimeError("No blood type values found in donor registry.")

    output_rows = [
        {
            "blood_type": str(row.get("group_value", "")).strip().upper(),
            "count": int(as_int(row.get("value"), 0)),
            "prevalence_pct": round((int(as_int(row.get("value"), 0)) / total) * 100.0, 2),
        }
        for row in count_rows
        if str(row.get("group_value", "")).strip()
    ]
    scope = f"{country_code} donor registry" if country_code else "donor registry"
    preview = ", ".join(
        f"{row['blood_type']} {row['prevalence_pct']:.1f}%"
        for row in output_rows[:3]
    )
    return {
        **payload,
        "query": query,
        "country_code": country_code,
        "blood_type": blood_type,
        "query_kind": "blood_type_prevalence",
        "total_population": total,
        "rows": output_rows,
        "answer": f"Blood type prevalence in the {scope}: {preview}.",
    }


def _execute_blood_prevalence_sqlite(
    *,
    query: str,
    country_code: str | None,
    blood_type: str | None,
    db_sqlite_path: str | None,
    donors_table_name: str,
    hospitals_table_name: str,
    donor_csv_path: Path,
    hospital_csv_path: Path,
    postgres_error: Exception | None = None,
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if country_code:
        filters["country_code"] = country_code
    if blood_type:
        filters["blood_type"] = blood_type
    payload = _execute_structured_sqlite_query(
        query=query,
        table="donors",
        field="blood_type",
        aggregate="count",
        group_by=None,
        filters=filters,
        limit=_db_max_limit(),
        sort="desc",
        db_sqlite_path=db_sqlite_path,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
        donor_csv_path=donor_csv_path,
        hospital_csv_path=hospital_csv_path,
        postgres_error=postgres_error,
    )
    count_rows = payload.get("rows", [])
    total = sum(int(as_int(row.get("value"), 0)) for row in count_rows)
    if total <= 0:
        raise RuntimeError("No blood type values found in donor registry.")
    output_rows = [
        {
            "blood_type": str(row.get("group_value", "")).strip().upper(),
            "count": int(as_int(row.get("value"), 0)),
            "prevalence_pct": round((int(as_int(row.get("value"), 0)) / total) * 100.0, 2),
        }
        for row in count_rows
        if str(row.get("group_value", "")).strip()
    ]
    scope = f"{country_code} donor registry" if country_code else "donor registry"
    preview = ", ".join(
        f"{row['blood_type']} {row['prevalence_pct']:.1f}%"
        for row in output_rows[:3]
    )
    return {
        **payload,
        "query": query,
        "country_code": country_code,
        "blood_type": blood_type,
        "query_kind": "blood_type_prevalence",
        "total_population": total,
        "rows": output_rows,
        "answer": f"Blood type prevalence in the {scope}: {preview}.",
    }


def _execute_supply_snapshot_postgres(
    *,
    query: str,
    blood_type: str | None,
    db_schema: str,
    donors_table_name: str,
    hospitals_table_name: str,
    donor_csv_path: Path,
    hospital_csv_path: Path,
    db_series_limit: int,
) -> dict[str, Any]:
    schema_name = _normalize_db_schema(db_schema)
    hospitals_table_name = _validate_identifier(
        hospitals_table_name,
        "hospitals table name",
    )
    series_limit = _normalize_db_limit(db_series_limit)
    filters = {"blood_type": blood_type} if blood_type else {}
    ensure_postgres_reference_tables(
        db_schema=schema_name,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
        donor_csv_path=donor_csv_path,
        hospital_csv_path=hospital_csv_path,
        required_tables={"hospitals"},
    )

    config = _TABLE_CONFIGS["hospitals"]
    where_clauses, params = _build_filter_clauses(filters, config=config)
    where_sql = (
        sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)
        if where_clauses
        else sql.SQL("")
    )
    table_sql = _qualified_table(schema_name, hospitals_table_name)

    conn = _postgres_connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                sql.SQL(
                    "SELECT date, "
                    "SUM(COALESCE(stock_end, 0)) AS stock_end_total, "
                    "SUM(COALESCE(units_collected, 0)) AS units_collected_total, "
                    "SUM(COALESCE(units_used, 0)) AS units_used_total, "
                    "COUNT(*) AS records "
                    "FROM {table}{where_clause} "
                    "GROUP BY date "
                    "ORDER BY date DESC "
                    "LIMIT %s"
                ).format(
                    table=table_sql,
                    where_clause=where_sql,
                ),
                [*params, series_limit],
            )
            fetched = cur.fetchall()
    finally:
        conn.close()

    if not fetched:
        scope = blood_type or "all blood types"
        raise RuntimeError(f"No hospital supply rows found for {scope}.")

    series = [
        _serialize_db_row(dict(row))
        for row in reversed(fetched)
    ]
    latest = series[-1]
    scope = blood_type or "all blood types"
    latest_date = str(latest.get("date", "latest date"))
    stock_end_total = float(as_float(latest.get("stock_end_total"), 0.0))
    records = int(as_int(latest.get("records"), 0))
    return {
        "tool_type": "db_tool",
        "provider": f"postgres:{schema_name}.{hospitals_table_name}",
        "query": query,
        "query_kind": "hospital_supply_snapshot",
        "table": "hospitals",
        "field": "stock_end",
        "aggregate": "sum",
        "blood_type": blood_type,
        "latest": latest,
        "series": series,
        "answer": (
            f"Latest available blood supply for {scope} is "
            f"{stock_end_total:.0f} units on {latest_date} "
            f"(aggregated from {records} records)."
        ),
    }


def _execute_supply_snapshot_sqlite(
    *,
    query: str,
    blood_type: str | None,
    db_sqlite_path: str | None,
    donors_table_name: str,
    hospitals_table_name: str,
    donor_csv_path: Path,
    hospital_csv_path: Path,
    db_series_limit: int,
    postgres_error: Exception | None = None,
) -> dict[str, Any]:
    sqlite_path = ensure_sqlite_reference_tables(
        db_sqlite_path=db_sqlite_path,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
        donor_csv_path=donor_csv_path,
        hospital_csv_path=hospital_csv_path,
        required_tables={"hospitals"},
    )
    filters = {"blood_type": blood_type} if blood_type else {}
    where_clauses, params = _build_sqlite_filter_clauses(
        filters,
        config=_TABLE_CONFIGS["hospitals"],
    )
    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    series_limit = _normalize_db_limit(db_series_limit)
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        fetched = conn.execute(
            f'SELECT date, SUM(COALESCE(stock_end, 0)) AS stock_end_total, '
            f'SUM(COALESCE(units_collected, 0)) AS units_collected_total, '
            f'SUM(COALESCE(units_used, 0)) AS units_used_total, COUNT(*) AS records '
            f'FROM "{hospitals_table_name}"{where_sql} '
            f'GROUP BY date ORDER BY date DESC LIMIT ?',
            [*params, series_limit],
        ).fetchall()
    finally:
        conn.close()
    if not fetched:
        scope = blood_type or "all blood types"
        raise RuntimeError(f"No hospital supply rows found for {scope}.")
    series = [_serialize_db_row(dict(row)) for row in reversed(fetched)]
    latest = series[-1]
    scope = blood_type or "all blood types"
    latest_date = str(latest.get("date", "latest date"))
    stock_end_total = float(as_float(latest.get("stock_end_total"), 0.0))
    records = int(as_int(latest.get("records"), 0))
    payload = {
        "tool_type": "db_tool",
        "provider": f"sqlite:{sqlite_path}:{hospitals_table_name}",
        "query": query,
        "query_kind": "hospital_supply_snapshot",
        "table": "hospitals",
        "field": "stock_end",
        "aggregate": "sum",
        "blood_type": blood_type,
        "latest": latest,
        "series": series,
        "answer": (
            f"Latest available blood supply for {scope} is "
            f"{stock_end_total:.0f} units on {latest_date} "
            f"(aggregated from {records} records)."
        ),
    }
    if postgres_error is not None:
        payload["postgres_error"] = str(postgres_error)
    return payload


def _execute_donor_outreach_postgres(
    *,
    query: str,
    country_code: str | None,
    blood_type: str | None,
    top_n: int,
    db_schema: str,
    donors_table_name: str,
    hospitals_table_name: str,
    donor_csv_path: Path,
    hospital_csv_path: Path,
) -> dict[str, Any]:
    schema_name = _normalize_db_schema(db_schema)
    donors_table_name = _validate_identifier(donors_table_name, "donors table name")
    ensure_postgres_reference_tables(
        db_schema=schema_name,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
        donor_csv_path=donor_csv_path,
        hospital_csv_path=hospital_csv_path,
        required_tables={"donors"},
    )

    filters: dict[str, Any] = {"eligible_to_donate": True}
    if country_code:
        filters["country_code"] = country_code
    if blood_type:
        filters["blood_type"] = blood_type
    where_clauses, params = _build_filter_clauses(filters, config=_TABLE_CONFIGS["donors"])
    where_sql = (
        sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)
        if where_clauses
        else sql.SQL("")
    )

    conn = _postgres_connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                sql.SQL(
                    "SELECT donor_id, blood_type, country_code, region, preferred_site, "
                    "donation_propensity_score, recency_days, donation_count_last_12m, "
                    "is_regular_donor, is_rare_type, "
                    "LEAST(0.99, GREATEST(0.0, "
                    "COALESCE(donation_propensity_score / 100.0, 0.32) "
                    "+ LEAST(0.07, COALESCE(donation_count_last_12m, 0) * 0.02) "
                    "+ CASE WHEN COALESCE(is_regular_donor, FALSE) THEN 0.06 ELSE 0 END "
                    "+ CASE WHEN COALESCE(is_rare_type, FALSE) THEN 0.05 ELSE 0 END "
                    "+ CASE WHEN COALESCE(recency_days, 0) BETWEEN 56 AND 365 THEN 0.04 ELSE 0 END"
                    ")) AS recommendation_score "
                    "FROM {table}{where_clause} "
                    "ORDER BY recommendation_score DESC, is_rare_type DESC, is_regular_donor DESC, donor_id ASC "
                    "LIMIT %s"
                ).format(
                    table=_qualified_table(schema_name, donors_table_name),
                    where_clause=where_sql,
                ),
                [*params, _normalize_db_limit(top_n)],
            )
            fetched = cur.fetchall()
    finally:
        conn.close()

    rows = []
    for raw_row in fetched:
        row = dict(raw_row)
        rationale = []
        if as_float(row.get("donation_propensity_score"), 0.0) >= 60.0:
            rationale.append("high propensity")
        if row.get("is_regular_donor"):
            rationale.append("regular donor")
        if row.get("is_rare_type"):
            rationale.append("rare blood type")
        if as_int(row.get("donation_count_last_12m"), 0) >= 2:
            rationale.append("recent donation history")
        if 56 <= as_int(row.get("recency_days"), 0) <= 365:
            rationale.append("good recontact window")
        if not rationale:
            rationale.append("eligible donor profile")
        rows.append(
            {
                "donor_id": str(row.get("donor_id", "")).strip(),
                "blood_type": str(row.get("blood_type", "")).strip().upper(),
                "country_code": str(row.get("country_code", "")).strip().upper(),
                "region": str(row.get("region", "") or "").strip(),
                "preferred_site": str(row.get("preferred_site", "") or "").strip(),
                "recommendation_score": round(as_float(row.get("recommendation_score"), 0.0), 4),
                "donation_propensity_score": round(as_float(row.get("donation_propensity_score"), 0.0), 4),
                "recency_days": as_int(row.get("recency_days"), 0),
                "donation_count_last_12m": as_int(row.get("donation_count_last_12m"), 0),
                "is_regular_donor": bool(row.get("is_regular_donor")),
                "is_rare_type": bool(row.get("is_rare_type")),
                "rationale": ", ".join(rationale),
            }
        )

    if not rows:
        raise RuntimeError("No eligible donors found in the donor registry.")

    scope_parts = []
    if country_code:
        scope_parts.append(country_code)
    if blood_type:
        scope_parts.append(blood_type)
    scope = " / ".join(scope_parts) if scope_parts else "local donor registry"
    preview = ", ".join(
        f"{row['donor_id']} ({row['blood_type']}, score {row['recommendation_score']:.2f})"
        for row in rows[:3]
    )
    strongest_score = float(rows[0]["recommendation_score"])
    decision = "Yes" if strongest_score >= 0.45 else "Maybe"
    return {
        "tool_type": "db_tool",
        "provider": f"postgres:{schema_name}.{donors_table_name}",
        "query": query,
        "query_kind": "donor_outreach",
        "country_code": country_code,
        "blood_type": blood_type,
        "total_candidates": len(rows),
        "recommended_count": len(rows),
        "rows": rows,
        "answer": (
            f"{decision}. Found {len(rows)} eligible donor candidates for next month in {scope}. "
            f"Prioritize {preview}."
        ),
    }


def _execute_donor_outreach_sqlite(
    *,
    query: str,
    country_code: str | None,
    blood_type: str | None,
    top_n: int,
    db_sqlite_path: str | None,
    donors_table_name: str,
    hospitals_table_name: str,
    donor_csv_path: Path,
    hospital_csv_path: Path,
    postgres_error: Exception | None = None,
) -> dict[str, Any]:
    sqlite_path = ensure_sqlite_reference_tables(
        db_sqlite_path=db_sqlite_path,
        donors_table_name=donors_table_name,
        hospitals_table_name=hospitals_table_name,
        donor_csv_path=donor_csv_path,
        hospital_csv_path=hospital_csv_path,
        required_tables={"donors"},
    )
    filters: dict[str, Any] = {"eligible_to_donate": True}
    if country_code:
        filters["country_code"] = country_code
    if blood_type:
        filters["blood_type"] = blood_type
    where_clauses, params = _build_sqlite_filter_clauses(
        filters,
        config=_TABLE_CONFIGS["donors"],
    )
    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        fetched = conn.execute(
            f'SELECT donor_id, blood_type, country_code, region, preferred_site, '
            f'donation_propensity_score, recency_days, donation_count_last_12m, '
            f'is_regular_donor, is_rare_type, '
            f'MIN(0.99, MAX(0.0, COALESCE(donation_propensity_score / 100.0, 0.32) '
            f'+ MIN(0.07, COALESCE(donation_count_last_12m, 0) * 0.02) '
            f'+ CASE WHEN COALESCE(is_regular_donor, 0) THEN 0.06 ELSE 0 END '
            f'+ CASE WHEN COALESCE(is_rare_type, 0) THEN 0.05 ELSE 0 END '
            f'+ CASE WHEN COALESCE(recency_days, 0) BETWEEN 56 AND 365 THEN 0.04 ELSE 0 END'
            f')) AS recommendation_score '
            f'FROM "{donors_table_name}"{where_sql} '
            f'ORDER BY recommendation_score DESC, is_rare_type DESC, is_regular_donor DESC, donor_id ASC '
            f'LIMIT ?',
            [*params, _normalize_db_limit(top_n)],
        ).fetchall()
    finally:
        conn.close()

    rows = []
    for raw_row in fetched:
        row = dict(raw_row)
        rationale = []
        if as_float(row.get("donation_propensity_score"), 0.0) >= 60.0:
            rationale.append("high propensity")
        if row.get("is_regular_donor"):
            rationale.append("regular donor")
        if row.get("is_rare_type"):
            rationale.append("rare blood type")
        if as_int(row.get("donation_count_last_12m"), 0) >= 2:
            rationale.append("recent donation history")
        if 56 <= as_int(row.get("recency_days"), 0) <= 365:
            rationale.append("good recontact window")
        if not rationale:
            rationale.append("eligible donor profile")
        rows.append(
            {
                "donor_id": str(row.get("donor_id", "")).strip(),
                "blood_type": str(row.get("blood_type", "")).strip().upper(),
                "country_code": str(row.get("country_code", "")).strip().upper(),
                "region": str(row.get("region", "") or "").strip(),
                "preferred_site": str(row.get("preferred_site", "") or "").strip(),
                "recommendation_score": round(as_float(row.get("recommendation_score"), 0.0), 4),
                "donation_propensity_score": round(as_float(row.get("donation_propensity_score"), 0.0), 4),
                "recency_days": as_int(row.get("recency_days"), 0),
                "donation_count_last_12m": as_int(row.get("donation_count_last_12m"), 0),
                "is_regular_donor": bool(row.get("is_regular_donor")),
                "is_rare_type": bool(row.get("is_rare_type")),
                "rationale": ", ".join(rationale),
            }
        )
    if not rows:
        raise RuntimeError("No eligible donors found in the donor registry.")

    scope_parts = []
    if country_code:
        scope_parts.append(country_code)
    if blood_type:
        scope_parts.append(blood_type)
    scope = " / ".join(scope_parts) if scope_parts else "local donor registry"
    preview = ", ".join(
        f"{row['donor_id']} ({row['blood_type']}, score {row['recommendation_score']:.2f})"
        for row in rows[:3]
    )
    strongest_score = float(rows[0]["recommendation_score"])
    decision = "Yes" if strongest_score >= 0.45 else "Maybe"
    payload = {
        "tool_type": "db_tool",
        "provider": f"sqlite:{sqlite_path}:{donors_table_name}",
        "query": query,
        "query_kind": "donor_outreach",
        "country_code": country_code,
        "blood_type": blood_type,
        "total_candidates": len(rows),
        "recommended_count": len(rows),
        "rows": rows,
        "answer": (
            f"{decision}. Found {len(rows)} eligible donor candidates for next month in {scope}. "
            f"Prioritize {preview}."
        ),
    }
    if postgres_error is not None:
        payload["postgres_error"] = str(postgres_error)
    return payload


def flatten_duckduckgo_topics(payload: list[Any], limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in payload:
        if len(rows) >= limit:
            break
        if not isinstance(row, dict):
            continue
        nested = row.get("Topics")
        if isinstance(nested, list):
            rows.extend(flatten_duckduckgo_topics(nested, limit - len(rows)))
            continue
        title = str(row.get("Text", "")).strip()
        url = str(row.get("FirstURL", "")).strip()
        if title:
            rows.append({"title": title, "url": url})
    return rows[:limit]


def execute_db_tool(
    query: str | None = None,
    *,
    db_supply_csv_path: Path,
    db_sqlite_path: str,
    db_sqlite_table: str,
    db_series_limit: int,
    db_schema: str | None = None,
    donors_table_name: str | None = None,
    hospitals_table_name: str | None = None,
    table: str | None = None,
    field: str | None = None,
    fields: list[str] | str | None = None,
    aggregate: str | None = None,
    group_by: str | None = None,
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
    sort: str | None = None,
    latest_only: bool | None = None,
    csv_candidates: list[Path] | None = None,
) -> dict[str, Any]:
    prepared_query = (query or "").strip()
    db_schema = db_schema or db_reference_default_schema()
    donors_table_name = donors_table_name or db_reference_default_table_name("donors")
    hospitals_table_name = hospitals_table_name or db_reference_default_table_name(
        "hospitals"
    )
    donor_csv_candidates = _dedupe_paths(
        list(csv_candidates or [])
        + _default_donor_csv_candidates()
    )
    donor_csv_path = next(
        (candidate for candidate in donor_csv_candidates if candidate.exists()),
        (
            Path(os.getenv("PIOS_ORCH_DB_DONOR_CSV_PATH", "")).expanduser().resolve()
            if os.getenv("PIOS_ORCH_DB_DONOR_CSV_PATH", "").strip()
            else _default_donor_csv_fallback()
        ),
    )
    hospital_csv_path = db_supply_csv_path.expanduser().resolve()
    sqlite_fallback_enabled = os.getenv(
        "PIOS_ORCH_DB_SQLITE_FALLBACK",
        "1",
    ).strip().lower() in {"1", "true", "yes", "y", "on"}

    try:
        normalized_aggregate = _normalize_aggregate(aggregate) if aggregate else None
        if table and normalized_aggregate == "list" and (fields or field):
            return _execute_reference_list_postgres(
                query=prepared_query,
                table=table,
                fields=fields or field,
                filters=filters,
                limit=limit,
                latest_only=bool(latest_only),
                db_schema=db_schema,
                donors_table_name=donors_table_name,
                hospitals_table_name=hospitals_table_name,
                donor_csv_path=donor_csv_path,
                hospital_csv_path=hospital_csv_path,
            )

        if table and field and aggregate:
            return _execute_structured_postgres_query(
                query=prepared_query,
                table=table,
                field=field,
                aggregate=aggregate,
                group_by=group_by,
                filters=filters,
                limit=limit,
                sort=sort,
                db_schema=db_schema,
                donors_table_name=donors_table_name,
                hospitals_table_name=hospitals_table_name,
                donor_csv_path=donor_csv_path,
                hospital_csv_path=hospital_csv_path,
            )

        blood_type = extract_supply_blood_type(prepared_query)
        country_code = _extract_country_code(prepared_query)

        if is_blood_prevalence_query(prepared_query):
            if _wants_ethnicity_breakdown(prepared_query):
                raise RuntimeError(
                    "Ethnicity breakdown is not available in the orchestrator donor registry."
                )
            return _execute_blood_prevalence_postgres(
                query=prepared_query,
                country_code=country_code,
                blood_type=blood_type,
                db_schema=db_schema,
                donors_table_name=donors_table_name,
                hospitals_table_name=hospitals_table_name,
                donor_csv_path=donor_csv_path,
                hospital_csv_path=hospital_csv_path,
            )

        if is_donor_outreach_query(prepared_query):
            return _execute_donor_outreach_postgres(
                query=prepared_query,
                country_code=country_code,
                blood_type=blood_type,
                top_n=_requested_limit(
                    prepared_query,
                    default=_db_default_limit(),
                    maximum=_db_max_limit(),
                ),
                db_schema=db_schema,
                donors_table_name=donors_table_name,
                hospitals_table_name=hospitals_table_name,
                donor_csv_path=donor_csv_path,
                hospital_csv_path=hospital_csv_path,
            )

        list_request = _infer_reference_list_request(prepared_query)
        if list_request is not None:
            list_table, list_fields, latest_only = list_request
            return _execute_reference_list_postgres(
                query=prepared_query,
                table=list_table,
                fields=list_fields,
                filters=filters,
                limit=limit,
                latest_only=latest_only,
                db_schema=db_schema,
                donors_table_name=donors_table_name,
                hospitals_table_name=hospitals_table_name,
                donor_csv_path=donor_csv_path,
                hospital_csv_path=hospital_csv_path,
            )

        if not is_supply_query(prepared_query):
            raise RuntimeError("DB fallback template does not match this query.")

        return _execute_supply_snapshot_postgres(
            query=prepared_query,
            blood_type=blood_type,
            db_schema=db_schema,
            donors_table_name=donors_table_name,
            hospitals_table_name=hospitals_table_name,
            donor_csv_path=donor_csv_path,
            hospital_csv_path=hospital_csv_path,
            db_series_limit=db_series_limit,
        )
    except Exception as postgres_error:
        if not sqlite_fallback_enabled:
            raise
        normalized_aggregate = _normalize_aggregate(aggregate) if aggregate else None
        if table and normalized_aggregate == "list" and (fields or field):
            return _execute_reference_list_sqlite(
                query=prepared_query,
                table=table,
                fields=fields or field,
                filters=filters,
                limit=limit,
                latest_only=bool(latest_only),
                db_sqlite_path=db_sqlite_path,
                donors_table_name=donors_table_name,
                hospitals_table_name=hospitals_table_name,
                donor_csv_path=donor_csv_path,
                hospital_csv_path=hospital_csv_path,
                postgres_error=postgres_error,
            )

        if table and field and aggregate:
            return _execute_structured_sqlite_query(
                query=prepared_query,
                table=table,
                field=field,
                aggregate=aggregate,
                group_by=group_by,
                filters=filters,
                limit=limit,
                sort=sort,
                db_sqlite_path=db_sqlite_path,
                donors_table_name=donors_table_name,
                hospitals_table_name=hospitals_table_name,
                donor_csv_path=donor_csv_path,
                hospital_csv_path=hospital_csv_path,
                postgres_error=postgres_error,
            )

        blood_type = extract_supply_blood_type(prepared_query)
        country_code = _extract_country_code(prepared_query)
        if is_blood_prevalence_query(prepared_query):
            if _wants_ethnicity_breakdown(prepared_query):
                raise
            return _execute_blood_prevalence_sqlite(
                query=prepared_query,
                country_code=country_code,
                blood_type=blood_type,
                db_sqlite_path=db_sqlite_path,
                donors_table_name=donors_table_name,
                hospitals_table_name=hospitals_table_name,
                donor_csv_path=donor_csv_path,
                hospital_csv_path=hospital_csv_path,
                postgres_error=postgres_error,
            )
        if is_donor_outreach_query(prepared_query):
            return _execute_donor_outreach_sqlite(
                query=prepared_query,
                country_code=country_code,
                blood_type=blood_type,
                top_n=_requested_limit(
                    prepared_query,
                    default=_db_default_limit(),
                    maximum=_db_max_limit(),
                ),
                db_sqlite_path=db_sqlite_path,
                donors_table_name=donors_table_name,
                hospitals_table_name=hospitals_table_name,
                donor_csv_path=donor_csv_path,
                hospital_csv_path=hospital_csv_path,
                postgres_error=postgres_error,
            )
        list_request = _infer_reference_list_request(prepared_query)
        if list_request is not None:
            list_table, list_fields, latest_only = list_request
            return _execute_reference_list_sqlite(
                query=prepared_query,
                table=list_table,
                fields=list_fields,
                filters=filters,
                limit=limit,
                latest_only=latest_only,
                db_sqlite_path=db_sqlite_path,
                donors_table_name=donors_table_name,
                hospitals_table_name=hospitals_table_name,
                donor_csv_path=donor_csv_path,
                hospital_csv_path=hospital_csv_path,
                postgres_error=postgres_error,
            )
        if not is_supply_query(prepared_query):
            raise
        return _execute_supply_snapshot_sqlite(
            query=prepared_query,
            blood_type=blood_type,
            db_sqlite_path=db_sqlite_path,
            donors_table_name=donors_table_name,
            hospitals_table_name=hospitals_table_name,
            donor_csv_path=donor_csv_path,
            hospital_csv_path=hospital_csv_path,
            db_series_limit=db_series_limit,
            postgres_error=postgres_error,
        )


def execute_search_tool(
    query: str,
    *,
    search_api_url: str,
    search_timeout_s: float,
    search_result_limit: int,
) -> dict[str, Any]:

    # ── Custom API configured ────────────────────────
    if search_api_url:
        response = http_json_request(
            method="GET",
            url=search_api_url,
            params={"q": query},
            timeout_s=search_timeout_s,
        )
        return {
            "tool_type": "search",
            "provider": "custom_search_api",
            "query": query,
            "status_code": response.get("status_code"),
            "data": response.get("data"),
        }

    # ── DuckDuckGo: try Instant Answer first ─────────
    response = http_json_request(
        method="GET",
        url="https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        timeout_s=search_timeout_s,
    )
    payload = response.get("data")
    if not isinstance(payload, dict):
        payload = {}

    answer = str(payload.get("AbstractText", "")).strip()
    related = payload.get("RelatedTopics", [])
    if not isinstance(related, list):
        related = []
    topics = flatten_duckduckgo_topics(related, search_result_limit)

    # If Instant Answer returned something, use it
    if answer or topics:
        return {
            "tool_type": "search",
            "provider": "duckduckgo_instant_answer",
            "query": query,
            "answer": answer,
            "answer_url": str(payload.get("AbstractURL", "")).strip(),
            "results": topics,
            "status_code": response.get("status_code"),
        }

    # ── DuckDuckGo: fallback to HTML search ──────────
    import re as _re
    try:
        html_response = http_json_request(
            method="GET",
            url="https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; PIOS/1.0)"},
            timeout_s=search_timeout_s,
        )
        raw_text = html_response.get("data", {})
        if isinstance(raw_text, dict):
            raw_text = raw_text.get("text", "")
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)

        # Parse result snippets from HTML
        results = []
        # Match DuckDuckGo result blocks
        snippet_matches = _re.findall(
            r'class="result__snippet"[^>]*>(.*?)</a',
            raw_text,
            flags=_re.DOTALL,
        )
        title_matches = _re.findall(
            r'class="result__a"[^>]*>(.*?)</a',
            raw_text,
            flags=_re.DOTALL,
        )
        url_matches = _re.findall(
            r'class="result__url"[^>]*href="([^"]*)"',
            raw_text,
            flags=_re.DOTALL,
        )

        for i in range(min(search_result_limit, len(snippet_matches))):
            title = _re.sub(r"<[^>]+>", "", title_matches[i]).strip() if i < len(title_matches) else ""
            snippet = _re.sub(r"<[^>]+>", "", snippet_matches[i]).strip()
            url = url_matches[i].strip() if i < len(url_matches) else ""
            if snippet:
                results.append({"title": title, "url": url, "snippet": snippet})

        if results:
            return {
                "tool_type": "search",
                "provider": "duckduckgo_html",
                "query": query,
                "answer": results[0]["snippet"] if results else "",
                "results": results,
                "status_code": 200,
            }
    except Exception:
        pass

    # ── Nothing worked ───────────────────────────────
    return {
        "tool_type": "search",
        "provider": "duckduckgo_instant_answer",
        "query": query,
        "answer": "",
        "answer_url": "",
        "results": [],
        "status_code": response.get("status_code"),
    }


def execute_llm_api_tool(
    query: str,
    *,
    llm_api_url: str,
    llm_api_key: str,
    llm_api_model: str,
    llm_api_timeout_s: float,
    llm_api_system_prompt: str,
    local_llm: Any = None,
    local_llm_active_model_id: str | None = None,
    local_llm_prompt_mode: str | None = None,
    safe_generation_tokens_fn: Any = None,
) -> dict[str, Any]:
    if not llm_api_url:
        # Fall back to the local orchestrator LLM
        if local_llm is None:
            raise RuntimeError("LLM API fallback is not configured and no local LLM is available.")
        prompt = (
            f"{llm_api_system_prompt}\n\n"
            f"User question: {query}\n\n"
            "Answer concisely with factual caveats when uncertain."
        )
        max_tokens = 256
        if safe_generation_tokens_fn:
            max_tokens = safe_generation_tokens_fn(prompt, requested=256)
        prompt_mode = str(local_llm_prompt_mode or "").strip().lower()
        text = ""
        if prompt_mode == "chat" and hasattr(local_llm, "create_chat_completion"):
            try:
                output = local_llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": llm_api_system_prompt},
                        {"role": "user", "content": query},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                choices = output.get("choices", []) if isinstance(output, dict) else []
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get("message", {})
                    text = str(message.get("content", "")).strip()
            except Exception:
                text = ""
        if not text:
            output = local_llm(
                prompt, max_tokens=max_tokens, temperature=0.1, echo=False
            )
            text = str(output["choices"][0]["text"]).strip()
        return {
            "tool_type": "llm_api",
            "provider": "local_orchestrator_llm_fallback",
            "model": local_llm_active_model_id,
            "status_code": 200,
            "answer": text,
        }

    headers_dict = {"Accept": "application/json", "Content-Type": "application/json"}
    if llm_api_key:
        headers_dict["Authorization"] = f"Bearer {llm_api_key}"

    body: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": llm_api_system_prompt},
            {"role": "user", "content": query},
        ],
        "temperature": 0.1,
    }
    if llm_api_model:
        body["model"] = llm_api_model

    response = http_json_request(
        method="POST",
        url=llm_api_url,
        body=body,
        headers=headers_dict,
        timeout_s=llm_api_timeout_s,
    )
    data = response.get("data")
    answer = extract_llm_api_text(data)
    return {
        "tool_type": "llm_api",
        "model": llm_api_model,
        "status_code": response.get("status_code"),
        "answer": answer,
        "data": data if not answer else None,
    }


def execute_simulation_tool(
    query: str,
    *,
    simulation_base_url: str,
    simulation_timeout_s: float,
    simulation_default_policy: str,
    simulation_compare_on_recommend: bool,
    simulation_default_compare_policies: list[str] | None = None,
) -> dict[str, Any]:
    config = SimulationToolConfig(
        base_url=simulation_base_url,
        timeout_s=simulation_timeout_s,
        default_policy=simulation_default_policy,
        compare_on_recommend=simulation_compare_on_recommend,
        default_compare_policies=list(simulation_default_compare_policies or []),
    )
    return execute_simulation_query(query, config=config)


def execute_digital_twin_tool(
    *,
    operation: str,
    query: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from digital_twin.models import TwinAction, TwinBranch, TwinRun
    from digital_twin.services import (
        DECISION_SUPPORT,
        branch_payload,
        capture_snapshot,
        create_twin_run,
        current_state,
        promote_action,
        recommend_action,
        recommendation_payload,
        run_branch,
    )

    args = dict(arguments or {})
    source_tool_id = str(args.get("tool_id") or "").strip()
    run_id = str(args.get("run_id") or "").strip()
    run = None
    if run_id:
        run = TwinRun.objects.get(run_id=run_id)
    else:
        run = TwinRun.objects.order_by("-started_at").first()
    if run is None:
        run_payload: dict[str, Any] = {}
        for key in ("source_mode", "scenario_key", "strategy_key", "seed"):
            if args.get(key) not in (None, ""):
                run_payload[key] = args[key]
        run = create_twin_run(run_payload)
        capture_snapshot(run)

    def _source_snapshot_ids() -> list[str]:
        return [
            str(snapshot_id)
            for snapshot_id in run.snapshots.order_by("-captured_at").values_list("snapshot_id", flat=True)[:5]
        ]

    if operation == "current_state":
        return {
            "tool_type": "digital_twin",
            "operation": operation,
            "run_id": str(run.run_id),
            "config_version": run.config_version,
            "source_snapshot_ids": _source_snapshot_ids(),
            "results": current_state(run),
            "answer": "Current digital twin state returned.",
            "decision_support": DECISION_SUPPORT,
        }

    if operation in {"run_branch", "compare_policies"}:
        policies = args.get("policies")
        if operation == "compare_policies" and isinstance(policies, list) and policies:
            branches = [
                branch_payload(
                    run_branch(
                        run,
                        branch_key=f"policy-{index + 1}-{safe_slug(policy)}",
                        policy_overrides={"strategy_key": str(policy)},
                    )
                )
                for index, policy in enumerate(policies)
            ]
            return {
                "tool_type": "digital_twin",
                "operation": operation,
                "run_id": str(run.run_id),
                "config_version": run.config_version,
                "source_snapshot_ids": _source_snapshot_ids(),
                "results": {"branches": branches},
                "answer": f"Compared {len(branches)} digital twin policy branches.",
                "decision_support": DECISION_SUPPORT,
            }
        branch = run_branch(
            run,
            branch_key=str(args.get("branch_key") or "agent-branch"),
            policy_overrides=dict(args.get("policy_overrides") or {}),
            action_overrides=list(args.get("action_overrides") or []),
            event_overrides=list(args.get("event_overrides") or []),
        )
        return {
            "tool_type": "digital_twin",
            "operation": operation,
            "run_id": str(run.run_id),
            "config_version": run.config_version,
            "source_snapshot_ids": _source_snapshot_ids(),
            "results": branch_payload(branch),
            "answer": "Digital twin branch created.",
            "decision_support": DECISION_SUPPORT,
        }

    if operation == "recommend_action":
        branch = None
        branch_id = str(args.get("branch_id") or "").strip()
        if branch_id:
            branch = TwinBranch.objects.get(branch_id=branch_id, run=run)
        recommendation = recommend_action(
            run,
            branch=branch,
            source_tool_ids=[source_tool_id] if source_tool_id else [],
        )
        return {
            "tool_type": "digital_twin",
            "operation": operation,
            "run_id": str(run.run_id),
            "config_version": run.config_version,
            "source_snapshot_ids": _source_snapshot_ids(),
            "results": recommendation_payload(recommendation),
            "answer": "Digital twin recommendation created.",
            "decision_support": DECISION_SUPPORT,
        }

    if operation == "promote_action":
        action_id = str(args.get("action_id") or "").strip()
        if not action_id:
            raise RuntimeError("Digital twin promote operation requires action_id.")
        action = TwinAction.objects.get(action_id=action_id)
        promoted = promote_action(action)
        return {
            "tool_type": "digital_twin",
            "operation": operation,
            "run_id": str(promoted.run_id),
            "config_version": promoted.run.config_version,
            "source_snapshot_ids": [
                str(snapshot_id)
                for snapshot_id in promoted.run.snapshots.order_by("-captured_at").values_list("snapshot_id", flat=True)[:5]
            ],
            "results": {"action_id": str(promoted.action_id), "promotion_status": promoted.promotion_status},
            "answer": "Digital twin action promoted.",
            "decision_support": DECISION_SUPPORT,
        }

    raise RuntimeError(f"Unsupported digital twin operation: {operation}")


def extract_llm_api_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        return ""

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return str(message["content"]).strip()
            if isinstance(first.get("text"), str):
                return str(first["text"]).strip()

    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()

    output = payload.get("output")
    if isinstance(output, list):
        for block in output:
            if not isinstance(block, dict):
                continue
            content = block.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    return str(item["text"]).strip()
    return ""


def is_external_output_success(output: dict[str, Any]) -> bool:
    if not isinstance(output, dict):
        return bool(output)
    answer = output.get("answer")
    if isinstance(answer, str) and answer.strip():
        return True
    results = output.get("results")
    if isinstance(results, list) and len(results) > 0:
        return True
    return bool(output.get("data"))
