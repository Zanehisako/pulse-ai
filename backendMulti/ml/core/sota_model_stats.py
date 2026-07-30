from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from django.utils import timezone

from ml.core.model_loading import load_prediction_runtime_config
from ml.core.model_stats import prepare_model_stats_for_storage
from ml.models import ModelStats


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SotaModelStatsConfigError(ValueError):
    """Raised when SOTA model stats generation cannot be configured."""


def _resolve_path(raw: Any, *, base_dir: Path = PROJECT_ROOT) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise SotaModelStatsConfigError("Configured SOTA model stats path cannot be empty.")
    path = Path(os.path.expandvars(text)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _clean_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _path_from_env_or_config(config: dict[str, Any], *, env_key: str, config_key: str) -> Path:
    env_name = str(config.get(env_key) or "").strip()
    if env_name:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return _resolve_path(env_value)
    return _resolve_path(config.get(config_key))


def sota_model_stats_config(runtime_config_path: str | Path | None = None) -> dict[str, Any]:
    payload = load_prediction_runtime_config(runtime_config_path)
    config = payload.get("sota_model_stats")
    if not isinstance(config, dict):
        raise SotaModelStatsConfigError("prediction_runtime.sota_model_stats must be configured.")
    if not bool(config.get("enabled", False)):
        raise SotaModelStatsConfigError("SOTA model stats generation is disabled by config.")
    return dict(config)


def resolve_sota_model_stats_paths(
    runtime_config_path: str | Path | None = None,
) -> dict[str, Path]:
    config = sota_model_stats_config(runtime_config_path)
    return {
        "catalog_path": _path_from_env_or_config(
            config,
            env_key="catalog_path_env",
            config_key="catalog_path",
        ),
        "datasets_dir": _path_from_env_or_config(
            config,
            env_key="datasets_dir_env",
            config_key="datasets_dir",
        ),
        "models_dir": _path_from_env_or_config(
            config,
            env_key="models_dir_env",
            config_key="default_models_dir",
        ),
    }


def feature_stats_payload(frame: pd.DataFrame, feature_names: list[str]) -> dict:
    """Compute per-feature statistics used by the ModelStats API."""
    stats: dict = {}
    for column in feature_names:
        if column not in frame.columns:
            stats[column] = {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "lower": 0.0,
                "upper": 0.0,
                "default": 0.0,
            }
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                stats[column] = {
                    "median": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "lower": 0.0,
                    "upper": 0.0,
                    "default": 0.0,
                }
                continue
            quantiles = numeric.quantile([0.025, 0.975]).to_numpy()
            stats[column] = {
                "median": float(numeric.median()),
                "mean": float(numeric.mean()),
                "std": float(numeric.std(ddof=0)) if len(numeric) > 1 else 0.0,
                "lower": float(quantiles[0]),
                "upper": float(quantiles[1]),
                "default": float(numeric.median()),
            }
        else:
            text = series.fillna("").astype(str).str.strip().replace({"nan": "", "None": ""})
            non_empty = text[text != ""]
            mode = str(non_empty.mode().iloc[0]) if not non_empty.mode().empty else ""
            stats[column] = {
                "mode": mode,
                "default": mode,
                "unique_values": int(non_empty.nunique()),
                "missing_rate": float((text == "").mean()),
            }
    return stats


def dashboard_profile_payload(model_entry: dict) -> dict:
    profile = model_entry.get("dashboard_profile")
    if not isinstance(profile, dict) or not bool(profile.get("enabled", False)):
        return {}

    fields = profile.get("fields")
    if not isinstance(fields, dict):
        return {}

    result: dict = {}
    for key, value in fields.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(value, dict):
            result[key_text] = dict(value)
        else:
            result[key_text] = {"mode": value}
    return result


def merge_dashboard_profile(stats: dict, model_entry: dict) -> dict:
    profile = dashboard_profile_payload(model_entry)
    if not profile:
        return stats
    return {**stats, **profile}


def load_or_generate_datasets(
    datasets_dir: Path,
    *,
    generate_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv_path = datasets_dir / "sota_inventory_features.parquet"
    donor_path = datasets_dir / "sota_donor_features.parquet"

    if inv_path.exists() and donor_path.exists():
        return pd.read_parquet(inv_path), pd.read_parquet(donor_path)

    if not generate_missing:
        raise SotaModelStatsConfigError(
            f"SOTA feature datasets not found in {datasets_dir}."
        )

    generate_scripts_dir = datasets_dir / "generate_scripts"
    sys.path.insert(0, str(generate_scripts_dir))
    try:
        from generate_sota_dataset import (  # type: ignore
            generate_quebec_donor_dataset,
            generate_quebec_inventory_dataset,
        )
    except ModuleNotFoundError as exc:
        raise SotaModelStatsConfigError(
            f"SOTA dataset generator not found in {generate_scripts_dir}."
        ) from exc

    datasets_dir.mkdir(parents=True, exist_ok=True)
    inv_features, inv_labels = generate_quebec_inventory_dataset()
    donor_features, donor_labels = generate_quebec_donor_dataset()

    inv_features.to_parquet(inv_path, index=False)
    inv_labels.to_parquet(datasets_dir / "sota_inventory_labels.parquet", index=False)
    donor_features.to_parquet(donor_path, index=False)
    donor_labels.to_parquet(datasets_dir / "sota_donor_labels.parquet", index=False)
    return inv_features, donor_features


def is_donor_model(features: list[str], donor_markers: list[str]) -> bool:
    return len(set(features) & set(donor_markers)) > 2


def generate_sota_model_stats(
    *,
    runtime_config_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    datasets_dir: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = sota_model_stats_config(runtime_config_path)
    paths = resolve_sota_model_stats_paths(runtime_config_path)
    if catalog_path is not None:
        paths["catalog_path"] = _resolve_path(catalog_path)
    if datasets_dir is not None:
        paths["datasets_dir"] = _resolve_path(datasets_dir)
    if models_dir is not None:
        paths["models_dir"] = _resolve_path(models_dir)

    catalog = json.loads(paths["catalog_path"].read_text(encoding="utf-8"))
    models = catalog.get("models")
    if not isinstance(models, list):
        raise SotaModelStatsConfigError("SOTA catalog must define a models list.")

    inv_features, donor_features = load_or_generate_datasets(
        paths["datasets_dir"],
        generate_missing=bool(config.get("generate_missing_datasets", True)),
    )
    donor_markers = _clean_strings(config.get("donor_feature_markers"))
    if not donor_markers:
        raise SotaModelStatsConfigError(
            "prediction_runtime.sota_model_stats.donor_feature_markers must not be empty."
        )

    persisted = 0
    skipped: list[dict[str, str]] = []
    processed: list[dict[str, Any]] = []

    for model_entry in models:
        if not isinstance(model_entry, dict) or not model_entry.get("enabled"):
            continue

        model_id = str(model_entry.get("registered_model_name") or "").strip()
        artifact = str(model_entry.get("artifact_filename") or "").strip()
        features = _clean_strings(model_entry.get("features"))
        task_type = str(model_entry.get("task_type") or "").strip()
        artifact_path = paths["models_dir"] / artifact

        if not model_id or not artifact or not features:
            skipped.append({"model_id": model_id, "reason": "incomplete catalog entry"})
            continue
        if not artifact_path.exists():
            skipped.append({"model_id": model_id, "reason": f"artifact not found at {artifact_path}"})
            continue

        dataset = donor_features if is_donor_model(features, donor_markers) else inv_features
        available = [feature for feature in features if feature in dataset.columns]
        if not available:
            skipped.append({"model_id": model_id, "reason": "no matching dataset features"})
            continue

        stats = merge_dashboard_profile(
            feature_stats_payload(dataset[available], available),
            model_entry,
        )
        ModelStats.objects.update_or_create(
            model_id=model_id,
            defaults={
                "identifier": model_id,
                "stats": prepare_model_stats_for_storage(stats),
                "made_at": timezone.now(),
            },
        )
        persisted += 1
        processed.append(
            {
                "model_id": model_id,
                "available_features": len(available),
                "configured_features": len(features),
                "task_type": task_type,
            }
        )

    return {
        "persisted": persisted,
        "processed": processed,
        "skipped": skipped,
        "paths": {key: str(value) for key, value in paths.items()},
    }
