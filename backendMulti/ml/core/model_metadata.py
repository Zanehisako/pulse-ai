"""
Runtime model metadata overrides, task inference, and default examples.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_MODEL_DEFAULTS_CONFIG_CACHE: tuple[Path, float, dict[str, Any]] | None = None


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _model_prediction_defaults_path() -> Path:
    configured = os.getenv("PIOS_MODEL_PREDICTION_DEFAULTS_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "config" / "model_prediction_defaults.json"


def _model_prediction_defaults_config() -> dict[str, Any]:
    global _MODEL_DEFAULTS_CONFIG_CACHE
    path = _model_prediction_defaults_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if (
        _MODEL_DEFAULTS_CONFIG_CACHE is not None
        and _MODEL_DEFAULTS_CONFIG_CACHE[0] == path
        and _MODEL_DEFAULTS_CONFIG_CACHE[1] == mtime
    ):
        return _MODEL_DEFAULTS_CONFIG_CACHE[2]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    _MODEL_DEFAULTS_CONFIG_CACHE = (path, mtime, payload)
    return payload


def _configured_model_row(model_id: str) -> dict[str, Any]:
    models = _model_prediction_defaults_config().get("models")
    if not isinstance(models, dict):
        return {}
    row = models.get(model_id)
    return row if isinstance(row, dict) else {}


def _clean_runtime_model_id(model_id: str) -> str:
    clean_id = str(model_id or "")
    for token in ("_champion", "_challenger", "_@"):
        clean_id = clean_id.split(token)[0]
    return clean_id


def runtime_model_overrides() -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    models = _model_prediction_defaults_config().get("models")
    if isinstance(models, dict):
        for model_id, row in models.items():
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata")
            if isinstance(metadata, dict):
                clean_id = str(model_id)
                overrides[clean_id] = dict(metadata)
    return overrides


def runtime_prediction_defaults(model_id: str) -> dict[str, Any]:
    clean_id = _clean_runtime_model_id(model_id)
    configured = _configured_model_row(clean_id).get("defaults")
    if not isinstance(configured, dict):
        return {}
    return dict(configured)


def infer_model_task(model_id: str, model_type: str, features: list[str]) -> str:
    overrides = runtime_model_overrides()
    clean_id = _clean_runtime_model_id(model_id)
    override = overrides.get(model_id) or overrides.get(clean_id)
    if isinstance(override, dict) and isinstance(override.get("task"), str):
        return str(override["task"])

    lowered = model_id.lower()
    if "stockout" in lowered or "demand" in lowered:
        return "inventory_forecast"
    if "donation_30d" in lowered or "days_to_next" in lowered:
        return "donor_horizon"
    if "donor" in lowered or "agent" in lowered:
        return "donor_individual"

    joined = " ".join(features).lower()
    if any(token in joined for token in {"stock", "hospital", "units_used", "units_collected"}):
        return "inventory_forecast"
    if any(token in joined for token in {"recency", "donation_count", "blood_type", "eligible"}):
        return "donor_individual"
    if model_type.lower().startswith("xgb"):
        return "inventory_forecast"
    return "general"


def default_description_for_task(task: str) -> str:
    if task == "donor_individual":
        return (
            "Individual donor propensity/eligibility model. "
            "Use for single-donor scoring or classification from donor features such as "
            "age, sex, BMI, recency, blood type, and donation history. "
            "Not for population-level statistics, grouped analytics, or historical database lookups."
        )
    if task == "donor_horizon":
        return (
            "Donor horizon model for a single donor. "
            "Use it to estimate the next donation date or whether a donor will donate within a time window "
            "from donor-level features. "
            "Not for aggregate trends, cohort summaries, or hospital inventory analytics."
        )
    if task == "inventory_forecast":
        return (
            "Hospital inventory and stockout horizon forecasting model. "
            "Use for prediction or forecasting from operational inputs such as temperature, rain, holidays, "
            "trauma cases, scheduled surgeries, stock levels, units used, and units collected. "
            "For stockout models, output is expected days until stockout. "
            "Not for descriptive SQL-style averages/counts over stored historical rows."
        )
    return (
        "General prediction model. "
        "Use with provided feature values for direct model inference, not for aggregate database analytics."
    )


def default_examples_for_task(task: str) -> list[dict[str, Any]]:
    if task == "donor_individual":
        return [
            {
                "kind": "good",
                "user_query": "Is donor age 35 with BMI 24 and recency 30 days likely to donate?",
                "why": "Single donor-level prediction request with concrete donor features.",
            },
            {
                "kind": "bad",
                "user_query": "What is the U.S. blood type prevalence by ethnicity over time?",
                "why": "Population analytics query; requires database/search, not donor-level model inference.",
            },
        ]
    if task == "donor_horizon":
        return [
            {
                "kind": "good",
                "user_query": "For this donor profile, how many days until the next donation?",
                "why": "Individual time-to-event estimation request.",
            },
            {
                "kind": "bad",
                "user_query": "Break down national donation percentages by age band over time.",
                "why": "Aggregate trend analytics request outside donor horizon scope.",
            },
        ]
    if task == "inventory_forecast":
        return [
            {
                "kind": "good",
                "user_query": (
                    "Given temperature 30, rain 0, holiday false, trauma cases 10, "
                    "scheduled surgeries 50, and the current stock inputs, forecast days until stockout."
                ),
                "why": "Operational inventory forecast request aligned with model scope.",
            },
            {
                "kind": "bad",
                "user_query": (
                    "What was the average stock_end in historical hospital rows where holiday was false "
                    "and trauma_cases was 10?"
                ),
                "why": "Historical aggregate analytics should use the structured database tool, not a forecasting model.",
            },
        ]
    return [
        {
            "kind": "good",
            "user_query": "Run this model with the provided feature values.",
            "why": "Direct model execution request.",
        },
        {
            "kind": "bad",
            "user_query": "Compute national prevalence trends from U.S. donor population tables.",
            "why": "Aggregate analytics should use database/search tools.",
        },
    ]
