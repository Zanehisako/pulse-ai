"""Config-driven public response compaction.

The output-key whitelist used to shape public API responses lives in the
orchestrator config (``orchestrator.api_output_compact_keys``) so response
shaping can be updated without code changes. When no configured key list is
available the built-in defaults below apply.
"""

from __future__ import annotations

from typing import Any

DEFAULT_COMPACT_KEYS: frozenset[str] = frozenset(
    {
        "answer",
        "results",
        "data",
        "query_kind",
        "model_id",
        "entity_key",
        "row_count",
        "prediction",
        "probability",
        "prediction_name",
        "prediction_unit",
        "prediction_task_type",
        "stockout_probability",
        "estimated_days_until_stockout",
        "horizon",
        "risk_level",
        "recommended_action",
        "confidence",
        "blood_component",
        "blood_group",
        "location",
        "worst_component",
        "worst_blood_group",
        "component_count",
        "worst_component_prediction",
        "warnings",
        "filters",
        "fields",
        "table",
        "aggregate",
        "limit",
        "tables",
        "table_names",
        "logical_table",
        "default_table_name",
        "column_count",
        "columns",
        "identity_fields",
        "field_aliases",
        "ignored_filter_fields",
        "aliases",
    }
)

DEFAULT_COMPACT_ROW_KEYS: tuple[str, ...] = (
    "index",
    "rank",
    "donor_id",
    "blood_type",
    "country_code",
    "region",
    "preferred_site",
    "recency_days",
    "donation_count_last_12m",
    "eligibility_status",
    "model_id",
    "model_score",
    "prediction",
    "probability",
    "prediction_name",
    "prediction_unit",
    "prediction_task_type",
)

DEFAULT_MODEL_OUTPUT_ROW_KEYS: tuple[str, ...] = (
    "prediction",
    "probability",
    "prediction_name",
    "prediction_unit",
    "prediction_task_type",
)


def configured_compact_keys(config_source: Any) -> frozenset[str] | None:
    """Return the configured compact-key whitelist or None for defaults."""
    if not isinstance(config_source, dict):
        return None
    keys = config_source.get("api_output_compact_keys")
    if isinstance(keys, (list, tuple, set)) and keys:
        return frozenset(str(key) for key in keys if str(key).strip())
    return None


def compact_prediction_row(
    row: Any,
    *,
    row_keys: tuple[str, ...] | None = None,
    model_output_keys: tuple[str, ...] | None = None,
) -> Any:
    if not isinstance(row, dict):
        return row
    public_keys = row_keys if row_keys is not None else DEFAULT_COMPACT_ROW_KEYS
    compact = {key: row[key] for key in public_keys if key in row}
    model_output = row.get("model_output")
    if isinstance(model_output, dict):
        for key in (
            model_output_keys
            if model_output_keys is not None
            else DEFAULT_MODEL_OUTPUT_ROW_KEYS
        ):
            if key in model_output and key not in compact:
                compact[key] = model_output[key]
    return compact or {
        key: value
        for key, value in row.items()
        if key not in {"model_output", "feature_resolution"}
    }


def compact_prediction_output(
    output: Any,
    *,
    compact_keys: frozenset[str] | None = None,
) -> Any:
    if not isinstance(output, dict):
        return output
    keys = compact_keys if compact_keys is not None else DEFAULT_COMPACT_KEYS
    compact = {
        key: value
        for key, value in output.items()
        if key in keys or str(key).startswith("risk_")
    }
    rows = output.get("rows")
    if isinstance(rows, list):
        compact["rows"] = [compact_prediction_row(row) for row in rows]
    return compact or output
