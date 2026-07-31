import os as _os

# Force single-threaded OpenMP / MKL in the serving process to prevent
# deadlocks between XGBoost/LightGBM OpenMP threads and the async event loop.
for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_v, "1")

import logging
import os
import json
import subprocess
import sys
import threading
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numpy.ma.extras import isin

from ml.core.model_metadata import runtime_prediction_defaults
from ml.core.model_stats import coerce_model_default_mapping
from ml.core.registry import ModelRuntime, _prepare_vector
from ml.core.utils import sigmoid
from ml.models import ModelStats

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MLFLOW_RUNS_DIR = (
    PROJECT_ROOT / "ml" / "artifacts" / "training_scripts" / "mlruns"
)
DEFAULT_PREDICTION_RUNTIME_CONFIG_PATH = (
    PROJECT_ROOT / "ml" / "config" / "prediction_runtime.json"
)
LOCAL_MLFLOW_ARTIFACT_PREFERENCES = {
    "hospital_shortage_predictor": ("best_hospital_model", "model"),
    "donor_propensity_model": ("best_donor_model", "model"),
    "ideal_donor_classifier": ("ideal_donor_classifier", "model"),
    "stockout_days_predictor": ("model", "best_estimator"),
}

# import mlflow
# mlflow.set_tracking_uri("file:./mlruns")

# ── MLflow model cache ────────────────────────────────────────
# Avoids re-downloading + re-deserializing the model on every request.
_mlflow_model_cache: dict[str, Any] = {}
_mlflow_cache_lock = threading.Lock()
_PREDICTION_RUNTIME_CONFIG_CACHE: tuple[Path, float, dict[str, Any]] | None = None


def _prediction_runtime_config_path() -> Path:
    raw = os.getenv("PIOS_PREDICTION_RUNTIME_CONFIG", "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_PREDICTION_RUNTIME_CONFIG_PATH


def _prediction_runtime_config() -> dict[str, Any]:
    global _PREDICTION_RUNTIME_CONFIG_CACHE
    path = _prediction_runtime_config_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if (
        _PREDICTION_RUNTIME_CONFIG_CACHE is not None
        and _PREDICTION_RUNTIME_CONFIG_CACHE[0] == path
        and _PREDICTION_RUNTIME_CONFIG_CACHE[1] == mtime
    ):
        return _PREDICTION_RUNTIME_CONFIG_CACHE[2]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Invalid prediction runtime config: %s", path)
        return {}
    if not isinstance(payload, dict):
        return {}
    _PREDICTION_RUNTIME_CONFIG_CACHE = (path, mtime, payload)
    return payload


def _mlflow_subprocess_config() -> dict[str, Any]:
    mlflow_config = _prediction_runtime_config().get("mlflow")
    if not isinstance(mlflow_config, dict):
        return {}
    subprocess_config = mlflow_config.get("subprocess")
    return dict(subprocess_config) if isinstance(subprocess_config, dict) else {}


def _mlflow_subprocess_enabled() -> bool:
    if os.getenv("PIOS_MLFLOW_ISOLATED_WORKER", "").strip():
        return False
    raw = os.getenv("PIOS_MLFLOW_SUBPROCESS", "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "y", "on"}
    return bool(_mlflow_subprocess_config().get("enabled"))


def _mlflow_subprocess_timeout() -> float:
    raw = os.getenv("PIOS_MLFLOW_SUBPROCESS_TIMEOUT_SECONDS", "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    config = _mlflow_subprocess_config()
    try:
        return max(1.0, float(config.get("timeout_seconds", 60)))
    except (TypeError, ValueError):
        return 60.0


def _mlflow_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    config_env = _mlflow_subprocess_config().get("env")
    if isinstance(config_env, dict):
        for key, value in config_env.items():
            key_text = str(key).strip()
            if key_text:
                env[key_text] = str(value)
    env["PIOS_MLFLOW_ISOLATED_WORKER"] = "1"
    env.setdefault(
        "DJANGO_SETTINGS_MODULE",
        os.getenv("DJANGO_SETTINGS_MODULE", "backendMulti.settings"),
    )
    env.setdefault("MLFLOW_TRACKING_URI", _mlflow_tracking_uri())
    env["PIOS_MLFLOW_HTTP_TIMEOUT_SECONDS"] = str(_mlflow_http_timeout())
    env["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(int(_mlflow_http_timeout()))
    env["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
    _append_env_pythonpath(env, _mlflow_local_load_python_paths())
    return env


def _mlflow_configured_paths(key: str) -> list[Path]:
    config = _prediction_runtime_config().get("mlflow")
    if not isinstance(config, dict):
        config = {}
    raw_paths = config.get(key)
    if not isinstance(raw_paths, list):
        return []
    paths: list[Path] = []
    for raw in raw_paths:
        text = str(raw or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        paths.append(path.resolve())
    return paths


def _mlflow_local_artifact_roots() -> list[Path]:
    paths = _mlflow_configured_paths("local_artifact_roots")
    if paths:
        return paths
    default = PROJECT_ROOT / "mlflow_artifacts"
    if LOCAL_MLFLOW_RUNS_DIR.exists():
        return [default, LOCAL_MLFLOW_RUNS_DIR]
    return [default]


def _mlflow_local_load_python_paths() -> list[Path]:
    return [path for path in _mlflow_configured_paths("local_load_python_paths") if path.exists()]


def _mlflow_tracking_uri() -> str:
    configured = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if configured:
        return configured

    config = _prediction_runtime_config().get("mlflow")
    raw_uri = ""
    if isinstance(config, dict):
        raw_uri = str(config.get("tracking_uri") or "").strip()
    if raw_uri:
        if "://" in raw_uri:
            return raw_uri
        path = Path(raw_uri).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve().as_uri()

    roots = _mlflow_local_artifact_roots()
    if roots:
        return roots[0].resolve().as_uri()
    return "http://localhost:8889"


def _append_env_pythonpath(env: dict[str, str], paths: list[Path]) -> None:
    values = [str(path) for path in paths]
    existing = env.get("PYTHONPATH", "").strip()
    if existing:
        values.append(existing)
    if values:
        env["PYTHONPATH"] = os.pathsep.join(values)


def _load_mlflow_pyfunc_model(mlflow: Any, model_uri: str):
    paths = [str(path) for path in _mlflow_local_load_python_paths()]
    original_sys_path = list(sys.path)
    try:
        for path in reversed(paths):
            if path not in sys.path:
                sys.path.insert(0, path)
        return mlflow.pyfunc.load_model(model_uri)
    finally:
        sys.path[:] = original_sys_path


def _mlflow_preflight_failure_policy() -> dict[str, Any]:
    config = _prediction_runtime_config().get("mlflow")
    if not isinstance(config, dict):
        return {}
    policy = config.get("preflight_failure_policy")
    return dict(policy) if isinstance(policy, dict) else {}


def _mlflow_http_timeout() -> float:
    raw = os.getenv("PIOS_MLFLOW_HTTP_TIMEOUT_SECONDS", "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    config = _prediction_runtime_config().get("mlflow")
    if isinstance(config, dict):
        try:
            return max(1.0, float(config.get("http_timeout_seconds", 10)))
        except (TypeError, ValueError):
            pass
    return 10.0


def _mlflow_missing_datetime_policy() -> dict[str, Any]:
    config = _prediction_runtime_config().get("mlflow")
    if not isinstance(config, dict):
        return {}
    policy = config.get("missing_datetime_policy")
    return dict(policy) if isinstance(policy, dict) else {}


def _default_mlflow_model_version() -> str:
    config = _prediction_runtime_config().get("model_loading")
    if isinstance(config, dict):
        default_policy = config.get("default_policy")
        if isinstance(default_policy, dict):
            version = str(default_policy.get("default_model_version") or "").strip()
            if version:
                return version
    return "champion"


def _mlflow_run_metadata_sources(metadata_key: str) -> tuple[dict[str, Any], ...]:
    config = _prediction_runtime_config().get("mlflow")
    if not isinstance(config, dict):
        return ()
    sources_config = config.get("run_metadata_sources")
    if not isinstance(sources_config, dict):
        return ()
    sources = sources_config.get(metadata_key)
    if not isinstance(sources, list):
        return ()
    return tuple(dict(source) for source in sources if isinstance(source, dict))


def _runtime_payload(runtime: ModelRuntime) -> dict[str, Any]:
    return {
        "model_id": runtime.model_id,
        "aliases": list(runtime.aliases or []),
        "slug": runtime.slug,
        "file_path": str(runtime.file_path),
        "model_type": runtime.model_type,
        "description": runtime.description,
        "feature_names": list(runtime.feature_names or []),
        "feature_info": dict(runtime.feature_info or {}),
        "examples": list(runtime.examples or []),
        "defaults": dict(runtime.defaults or {}),
        "status": runtime.status,
        "load_error": runtime.load_error,
    }


def _predict_mlflow_subprocess(runtime: ModelRuntime, features: dict[str, Any]) -> dict[str, Any]:
    outputs = _predict_mlflow_subprocess_batch(runtime, [features])
    if not outputs:
        raise ValueError("Isolated MLflow prediction returned no outputs.")
    return outputs[0]


def _predict_mlflow_subprocess_batch(
    runtime: ModelRuntime,
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = {
        "runtime": _runtime_payload(runtime),
        "feature_rows": [dict(row or {}) for row in feature_rows],
    }
    completed = subprocess.run(
        [sys.executable, "-m", "ml.core.mlflow_predict_worker"],
        input=json.dumps(payload, ensure_ascii=True, default=str),
        text=True,
        capture_output=True,
        timeout=_mlflow_subprocess_timeout(),
        env=_mlflow_subprocess_env(),
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"worker exited with code {completed.returncode}"
        raise ValueError(f"Isolated MLflow prediction failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Isolated MLflow prediction returned invalid JSON: {completed.stdout[:500]}"
        ) from exc
    if not isinstance(result, dict):
        raise ValueError("Isolated MLflow prediction returned a non-object payload.")
    if result.get("error"):
        raise ValueError(str(result.get("error")))
    outputs = result.get("outputs")
    if isinstance(outputs, list):
        return [row for row in outputs if isinstance(row, dict)]
    output = result.get("output")
    if isinstance(output, dict):
        return [output]
    raise ValueError("Isolated MLflow prediction returned no output object.")


def _repair_loaded_model_compatibility(model: Any) -> Any:
    """Patch known pickle/runtime compatibility gaps on loaded model objects."""
    visited: set[int] = set()

    def visit(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if (
            obj.__class__.__name__ == "SimpleImputer"
            and not hasattr(obj, "_fill_dtype")
            and hasattr(obj, "_fit_dtype")
        ):
            setattr(obj, "_fill_dtype", getattr(obj, "_fit_dtype"))

        if isinstance(obj, dict):
            for value in obj.values():
                visit(value)
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for value in obj:
                visit(value)
            return

        attrs = getattr(obj, "__dict__", None)
        if isinstance(attrs, dict):
            for value in attrs.values():
                visit(value)

    visit(model)
    return model


def _get_mlflow_module():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow support is unavailable in this backend environment. "
            "Install the base runtime dependency `mlflow-skinny` or a full MLflow package."
        ) from exc

    mlflow.set_tracking_uri(_mlflow_tracking_uri())
    return mlflow


def _try_load_local_mlflow_artifact(mlflow: Any, model_name: str) -> Any | None:
    local_model_dir = _find_local_mlflow_model_dir(model_name)
    if local_model_dir is None:
        return None
    logger.info(
        "Loading %s from local MLflow artifact cache at %s",
        model_name,
        local_model_dir,
    )
    model = _load_mlflow_pyfunc_model(mlflow, str(local_model_dir))
    return _repair_loaded_model_compatibility(model)


def _get_cached_mlflow_model(model_uri: str):
    """Load an MLflow pyfunc model, caching it after first load."""
    if model_uri in _mlflow_model_cache:
        logger.debug("Cache HIT for %s", model_uri)
        return _repair_loaded_model_compatibility(_mlflow_model_cache[model_uri])

    logger.info("Cache MISS for %s — loading from MLflow...", model_uri)
    mlflow = _get_mlflow_module()

    model = None
    load_error = None
    tracking_uri = mlflow.get_tracking_uri()
    policy = _mlflow_preflight_failure_policy()
    use_local_first = bool(policy.get("use_local_before_remote"))

    if model_uri.startswith("models:/"):
        model_name, version_or_alias = _parse_registry_model_uri(model_uri)
        if use_local_first:
            model = _try_load_local_mlflow_artifact(mlflow, model_name)
        if model is None:
            preflight_ok = _registry_artifact_is_available(
                model_name,
                version_or_alias=version_or_alias,
                tracking_uri=tracking_uri,
            )
            if not preflight_ok:
                model = _try_load_local_mlflow_artifact(mlflow, model_name)
                if model is None:
                    logger.warning(
                        "Registry artifact preflight failed for %s at %s. "
                        "Proceeding with direct MLflow load before local fallback.",
                        model_name,
                        tracking_uri,
                    )

    if model is None:
        try:
            model = _load_mlflow_pyfunc_model(mlflow, model_uri)
            logger.info("✅ Model loaded successfully: %s", model_uri)
        except mlflow.exceptions.MlflowException as e:
            load_error = str(e)
            logger.warning("Failed to load %s: %s", model_uri, e)

            # Fallback: try latest version if it's a registry model
            if model_uri.startswith("models:/"):
                model_name = model_uri.split("/")[1].split("@")[0]
                try:
                    from mlflow import MlflowClient

                    client = MlflowClient()
                    versions = client.search_model_versions(f"name='{model_name}'")
                    if versions:
                        latest_version = max(int(v.version) for v in versions)
                        fallback_uri = f"models:/{model_name}/{latest_version}"
                        logger.info("🔄 Trying fallback: %s", fallback_uri)
                        model = _load_mlflow_pyfunc_model(mlflow, fallback_uri)
                        logger.info("✅ Loaded with fallback version: %s", latest_version)
                    else:
                        raise ValueError(f"No versions found for model '{model_name}'")
                except Exception as fallback_error:
                    logger.error("Fallback also failed: %s", fallback_error)
                    model = _try_load_local_mlflow_artifact(mlflow, model_name)
                    if model is None:
                        roots = ", ".join(str(path) for path in _mlflow_local_artifact_roots())
                        raise ValueError(
                            f"Model '{model_name}' could not be loaded from the MLflow registry. "
                            f"Original error: {load_error}. "
                            f"Registry fallback error: {fallback_error}. "
                            f"MLflow is reachable, but the artifact files appear to be missing from the server artifact root. "
                            f"Expected a local artifact fallback under one of: {roots}"
                        ) from fallback_error
            else:
                raise

    if model is None:
        raise ValueError(f"Failed to load model from {model_uri}: {load_error}")

    with _mlflow_cache_lock:
        _mlflow_model_cache[model_uri] = model
    return model


def _find_local_mlflow_model_dir_under_root(
    model_name: str,
    runs_root: Path,
) -> Path | None:
    if not runs_root.exists():
        return None

    expected_feature_service = _expected_feature_service(model_name)
    preferences = _artifact_preferences(model_name)
    candidates: list[tuple[int, float, Path]] = []

    for experiment_dir in runs_root.iterdir():
        if not experiment_dir.is_dir():
            continue
        for run_dir in experiment_dir.iterdir():
            if not run_dir.is_dir():
                continue
            artifacts_dir = run_dir / "artifacts"
            if not artifacts_dir.is_dir():
                continue

            if expected_feature_service is not None:
                feast_tag = _read_text_if_exists(
                    run_dir / "tags" / "feast_feature_service"
                )
                if feast_tag != expected_feature_service:
                    continue

            run_score = 100 if expected_feature_service is not None else 0
            for index, artifact_name in enumerate(preferences):
                candidate_dir = artifacts_dir / artifact_name
                if not (candidate_dir / "MLmodel").exists():
                    continue
                priority = len(preferences) - index
                try:
                    mtime = (candidate_dir / "MLmodel").stat().st_mtime
                except OSError:
                    mtime = 0.0
                candidates.append((run_score + priority, mtime, candidate_dir))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _find_local_mlflow_model_dir(model_name: str) -> Path | None:
    for runs_root in _mlflow_local_artifact_roots():
        found = _find_local_mlflow_model_dir_under_root(model_name, runs_root)
        if found is not None:
            return found
    if LOCAL_MLFLOW_RUNS_DIR.exists():
        return _find_local_mlflow_model_dir_under_root(model_name, LOCAL_MLFLOW_RUNS_DIR)
    return None


def _expected_feature_service(model_name: str) -> str | None:
    defaults = runtime_prediction_defaults(model_name)
    if not isinstance(defaults, dict):
        return None
    feast = defaults.get("feast")
    if not isinstance(feast, dict):
        return None
    feature_service = feast.get("feature_service")
    if isinstance(feature_service, str) and feature_service.strip():
        return feature_service.strip()
    return None


def _artifact_preferences(model_name: str) -> tuple[str, ...]:
    preferences = list(LOCAL_MLFLOW_ARTIFACT_PREFERENCES.get(model_name, ()))
    if model_name not in preferences:
        preferences.append(model_name)
    if "model" not in preferences:
        preferences.append("model")
    return tuple(preferences)


def _read_text_if_exists(path: Path) -> str | None:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return None


def _parse_registry_model_uri(model_uri: str) -> tuple[str, str]:
    payload = model_uri[len("models:/") :].strip()
    if "@" in payload:
        model_name, version_or_alias = payload.split("@", 1)
        return model_name.strip(), version_or_alias.strip()
    if "/" in payload:
        model_name, version_or_alias = payload.split("/", 1)
        return model_name.strip(), version_or_alias.strip()
    return payload, ""


def _runtime_mlflow_model_uri(runtime: ModelRuntime) -> str:
    raw_path = str(runtime.file_path)
    raw_path = raw_path.replace("\\", "/")
    raw_path = raw_path.replace("mlflow:/", "models:/")

    if (
        raw_path.startswith("models:/")
        and "@" not in raw_path
        and raw_path.count("/") == 1
    ):
        raw_path = f"{raw_path}@{_default_mlflow_model_version()}"

    if raw_path.startswith(("file:", "models:", "runs:", "mlflow-artifacts:")):
        return raw_path
    return Path(raw_path).resolve().as_uri()


def _registry_artifact_is_available(
    model_name: str,
    *,
    version_or_alias: str,
    tracking_uri: str,
) -> bool:
    if not tracking_uri.startswith(("http://", "https://")):
        return True

    try:
        from mlflow import MlflowClient

        client = MlflowClient()
        if version_or_alias and not version_or_alias.isdigit():
            model_version = client.get_model_version_by_alias(
                model_name, version_or_alias
            )
        elif version_or_alias:
            model_version = client.get_model_version(model_name, version_or_alias)
        else:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                return False
            model_version = max(versions, key=lambda item: int(item.version))
    except Exception as exc:
        logger.warning(
            "Could not resolve MLflow model version metadata for %s: %s",
            model_name,
            exc,
        )
        return True

    source = str(getattr(model_version, "source", "") or "")
    if not source.startswith("mlflow-artifacts:/"):
        return True

    artifact_path = source[len("mlflow-artifacts:/") :].lstrip("/")
    artifact_url = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow-artifacts/artifacts/{artifact_path}/"
    request = urllib.request.Request(artifact_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=_mlflow_http_timeout()) as response:
            return 200 <= getattr(response, "status", 200) < 400
    except urllib.error.HTTPError as exc:
        logger.warning(
            "MLflow artifact preflight failed for %s (%s): HTTP %s",
            model_name,
            artifact_url,
            exc.code,
        )
        return False
    except urllib.error.URLError as exc:
        logger.warning(
            "MLflow artifact preflight failed for %s (%s): %s",
            model_name,
            artifact_url,
            exc,
        )
        return False


def flush_mlflow_cache():
    """Clear the MLflow model cache (called on model reload)."""
    with _mlflow_cache_lock:
        _mlflow_model_cache.clear()
    logger.info("MLflow model cache flushed.")


def preload_prediction_runtime(runtime: ModelRuntime) -> None:
    """Load enough of a runtime to prove it is usable for prediction."""
    if runtime.status != "loaded":
        raise ValueError(runtime.load_error or "Model is not loaded.")
    if runtime.model_type == "mlflow":
        _get_cached_mlflow_model(_runtime_mlflow_model_uri(runtime))


def _get_stats(
    model_id: str, *, feature_names: list[str] | None = None
) -> dict[str, Any]:
    # Strip alias suffix (e.g. "hospital_shortage_predictor_champion" → "hospital_shortage_predictor")
    clean_id = model_id.split("_champion")[0].split("_@")[0]
    try:
        stats_row = ModelStats.objects.filter(model_id=clean_id).first()
    except Exception as exc:
        logger.warning("Model stats lookup failed for %s: %s", clean_id, exc)
        return {}
    if stats_row is None:
        return {}
    stats = coerce_model_default_mapping(stats_row.stats, feature_names=feature_names)
    return {
        key: value
        for key, value in stats.items()
        if not _is_identifier_feature_name(key)
    }


def _is_identifier_feature_name(name: str) -> bool:
    lowered = str(name or "").strip().lower()
    if "__" in lowered:
        lowered = lowered.split("__", 1)[1]
    if ":" in lowered:
        lowered = lowered.split(":", 1)[1]
    return lowered == "id" or lowered.endswith("_id")


def _handle_missing_values(
    runtime: ModelRuntime, features: dict[str, Any]
) -> dict[str, Any]:
    stats = _get_stats(runtime.model_id, feature_names=runtime.feature_names)
    merged_features = dict(features)
    for key, value in stats.items():
        if (
            key not in merged_features
            or merged_features.get(key) is None
            or merged_features.get(key) == ""
        ):
            merged_features[key] = value
    return merged_features


def _merged_runtime_defaults(runtime: ModelRuntime) -> dict[str, Any]:
    clean_id = runtime.model_id.split("_champion")[0].split("_@")[0]
    merged = dict(runtime_prediction_defaults(clean_id))
    runtime_defaults = runtime.defaults if isinstance(runtime.defaults, dict) else {}
    for key, value in runtime_defaults.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _apply_prediction_output_contract(
    runtime: ModelRuntime, details: dict[str, Any]
) -> dict[str, Any]:
    defaults = _merged_runtime_defaults(runtime)
    output = defaults.get("output")
    output_contract = defaults.get("output_contract")
    if not isinstance(output, dict):
        output = {}

    prediction_name = str(output.get("name") or "").strip()
    prediction_unit = str(output.get("unit") or "").strip()
    task_type = str(output.get("task_type") or "").strip()
    result = dict(details)

    if isinstance(output_contract, dict):
        for key, contract in output_contract.items():
            if key not in result or not isinstance(contract, dict):
                continue
            if contract.get("type") == "string":
                continue
            value = result.get(key)
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            raw_value = numeric_value
            clip_min = contract.get("clip_min")
            clip_max = contract.get("clip_max")
            if clip_min is not None:
                numeric_value = max(float(clip_min), numeric_value)
            if clip_max is not None:
                numeric_value = min(float(clip_max), numeric_value)
            result[key] = numeric_value
            if numeric_value != raw_value:
                result[f"{key}_raw"] = raw_value
                result[f"{key}_clipped"] = True

    if "prediction" in result:
        prediction = result.get("prediction")
        try:
            numeric_prediction = float(prediction)
        except (TypeError, ValueError):
            numeric_prediction = None
        if numeric_prediction is not None:
            apply_clipping = (
                output.get("apply_clipping") is True
                or output.get("clip_output") is True
            )
            if apply_clipping:
                raw_prediction = numeric_prediction
                clip_min = output.get("clip_min")
                clip_max = output.get("clip_max")
                if clip_min is not None:
                    numeric_prediction = max(float(clip_min), numeric_prediction)
                if clip_max is not None:
                    numeric_prediction = min(float(clip_max), numeric_prediction)
                if numeric_prediction != raw_prediction:
                    result["raw_prediction"] = raw_prediction
                    result["prediction_clipped"] = True
            result["prediction"] = numeric_prediction
            if prediction_name:
                result[prediction_name] = numeric_prediction

    if prediction_name:
        result["prediction_name"] = prediction_name
    if prediction_unit:
        result["prediction_unit"] = prediction_unit
    if task_type:
        result["prediction_task_type"] = task_type
    return result


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_mlflow_prediction_output(predictions: Any) -> dict[str, Any]:
    if isinstance(predictions, pd.DataFrame):
        if predictions.empty:
            return {"prediction": None}
        row = {
            str(key): _to_python_scalar(value)
            for key, value in predictions.iloc[0].to_dict().items()
        }
        if "prediction" not in row and len(row) == 1:
            row["prediction"] = next(iter(row.values()))
        return row

    if isinstance(predictions, pd.Series):
        if predictions.empty:
            return {"prediction": None}
        return {"prediction": _to_python_scalar(predictions.iloc[0])}

    if isinstance(predictions, np.ndarray):
        if predictions.ndim == 0:
            return {"prediction": _to_python_scalar(predictions.item())}
        if predictions.ndim == 1:
            return {
                "prediction": (
                    _to_python_scalar(predictions[0]) if len(predictions) > 0 else None
                )
            }
        if predictions.ndim >= 2 and predictions.shape[0] > 0:
            first_row = predictions[0]
            if len(first_row) == 1:
                return {"prediction": _to_python_scalar(first_row[0])}
            return {"prediction": [_to_python_scalar(value) for value in first_row]}

    if isinstance(predictions, list):
        if not predictions:
            return {"prediction": None}
        first = predictions[0]
        if isinstance(first, dict):
            return {
                str(key): _to_python_scalar(value) for key, value in first.items()
            }
        return {"prediction": _to_python_scalar(first)}

    return {"prediction": _to_python_scalar(predictions)}


def _mlflow_missing_datetime_value() -> Any:
    policy = _mlflow_missing_datetime_policy()
    if str(policy.get("action") or "").strip().lower() != "fill":
        return pd.NaT

    if "value" in policy:
        configured = pd.to_datetime(policy.get("value"), errors="coerce", utc=True)
        if not pd.isna(configured):
            return configured.tz_convert(None)

    if str(policy.get("source") or "").strip().lower() == "request_time":
        return pd.Timestamp.now(tz="UTC").tz_convert(None)

    return pd.NaT


def _coerce_mlflow_datetime_column(values: pd.Series) -> pd.Series:
    coerced = pd.to_datetime(values, errors="coerce", utc=True)
    if coerced.isna().any():
        fill_value = _mlflow_missing_datetime_value()
        if not pd.isna(fill_value):
            coerced = coerced.fillna(fill_value)
    return coerced.dt.tz_convert(None).astype("datetime64[ns]")


def _json_safe_prediction_input_value(value: Any) -> Any:
    value = _to_python_scalar(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _predict_linear_custom(
    runtime: ModelRuntime, features: dict[str, Any]
) -> dict[str, Any]:
    """
    Hand-rolled logistic regression with manual standardization.
    Used by models saved with architecture="linear_custom".

    The pickle dict contains:
        params.weights  — {feature_name: weight}
        params.means    — {feature_name: mean} for standardization
        params.stds     — {feature_name: std}
        params.intercept — float
        params.encoders — {feature_name: [class_list]} for categoricals
    """
    params = runtime.params
    weights = (
        params.get("weights", {}) if isinstance(params.get("weights"), dict) else {}
    )
    means = params.get("means", {}) if isinstance(params.get("means"), dict) else {}
    stds = params.get("stds", {}) if isinstance(params.get("stds"), dict) else {}
    encoders_raw = (
        params.get("encoders", {}) if isinstance(params.get("encoders"), dict) else {}
    )
    intercept = float(params.get("intercept", 0.0))

    feature_names = runtime.feature_names or list(weights.keys())
    x_vector, used_inputs, missing = _prepare_vector(
        feature_names,
        features,
        encoders={n: list(c) for n, c in encoders_raw.items() if isinstance(c, list)},
    )
    values = x_vector[0]

    # Manual dot product with standardization
    score = intercept
    for i, name in enumerate(feature_names):
        val = float(values[i])
        if name in means:
            std_val = float(stds.get(name, 1.0))
            std_val = std_val if abs(std_val) > 1e-12 else 1.0
            val = (val - float(means.get(name, 0.0))) / std_val
        score += float(weights.get(name, 0.0)) * val

    prob = sigmoid(score)
    return {
        "prediction": int(prob >= 0.5),
        "probability": prob,
        "score": score,
        "used_inputs": used_inputs,
        "missing_features": missing,
    }


def _predict_online_agent(
    runtime: ModelRuntime, features: dict[str, Any]
) -> dict[str, Any]:
    """
    PersistentAdamAgent snapshot — online logistic regression with interactions.
    Used by models saved from DonorPredictiveService.

    The pickle dict contains:
        state.weights            — {feature_name: weight}
        state.intercept          — float
        state.standardizer_stats — {means: {}, stds: {}}
        state.encoders           — {feature_name: {classes: [...]}}
        state.interaction_weights — {"feat1||feat2": weight}
    """
    state = runtime.state
    weights = state.get("weights", {}) if isinstance(state.get("weights"), dict) else {}
    intercept = float(state.get("intercept", 0.0))

    # Standardizer stats
    stats = state.get("standardizer_stats", {})
    means = (
        stats.get("means", {})
        if isinstance(stats, dict) and isinstance(stats.get("means"), dict)
        else {}
    )
    stds = (
        stats.get("stds", {})
        if isinstance(stats, dict) and isinstance(stats.get("stds"), dict)
        else {}
    )

    # Categorical encoders
    encoder_payload = state.get("encoders", {})
    encoders: dict[str, list[Any]] = {}
    if isinstance(encoder_payload, dict):
        for feat_name, enc_state in encoder_payload.items():
            if isinstance(enc_state, dict) and isinstance(
                enc_state.get("classes"), list
            ):
                encoders[str(feat_name)] = list(enc_state["classes"])

    feature_names = runtime.feature_names or list(weights.keys())
    x_vector, used_inputs, missing = _prepare_vector(
        feature_names, features, encoders=encoders
    )

    # Standardize
    scaled_values: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        val = float(x_vector[0][i])
        if name in means:
            std_val = float(stds.get(name, 1.0))
            std_val = std_val if abs(std_val) > 1e-12 else 1.0
            val = (val - float(means.get(name, 0.0))) / std_val
        scaled_values[name] = val

    # Main effects
    score = intercept
    for name in feature_names:
        score += float(weights.get(name, 0.0)) * scaled_values.get(name, 0.0)

    # Interaction effects (feature pairs like "age||recency_days")
    interaction_weights = state.get("interaction_weights", {})
    if isinstance(interaction_weights, dict):
        for key, weight in interaction_weights.items():
            if not isinstance(key, str) or "||" not in key:
                continue
            first, second = key.split("||", 1)
            score += (
                float(weight)
                * scaled_values.get(first, 0.0)
                * scaled_values.get(second, 0.0)
            )

    prob = sigmoid(score)
    return {
        "prediction": int(prob >= 0.5),
        "probability": prob,
        "score": score,
        "used_inputs": used_inputs,
        "missing_features": missing,
    }


def _predict_estimator(
    runtime: ModelRuntime, features: dict[str, Any]
) -> dict[str, Any]:
    """
    Standard sklearn/xgboost/catboost model with .predict() and .predict_proba().
    Works with any model that follows the sklearn estimator interface.
    """
    estimator = runtime.estimator
    if estimator is None:
        raise ValueError("Estimator model is not available.")

    feature_names = runtime.feature_names
    if not feature_names:
        if not features:
            raise ValueError("No feature schema found and no features provided.")
        feature_names = sorted(features.keys())

    x_vector, used_inputs, missing = _prepare_vector(feature_names, features)
    result: dict[str, Any] = {
        "used_inputs": used_inputs,
        "missing_features": missing,
    }

    # Probabilities (if available)
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_vector)
        prob_list = probabilities[0].tolist()
        result["probabilities"] = prob_list
        if len(prob_list) == 2:
            result["probability"] = float(prob_list[1])
        else:
            max_idx = int(np.argmax(prob_list))
            result["probability"] = float(prob_list[max_idx])

    # Prediction
    if hasattr(estimator, "predict"):
        prediction = estimator.predict(x_vector)
        val = prediction[0]
        if isinstance(val, np.generic):
            val = val.item()
        result["prediction"] = val
    else:
        raise ValueError("Model has no predict method.")

    return result


def _predict_mlflow(runtime: ModelRuntime, features: dict[str, Any]) -> dict[str, Any]:
    """
    Run MLflow pyfunc model with caching and fallback strategies.
    """
    if _mlflow_subprocess_enabled():
        return _predict_mlflow_subprocess(runtime, features)

    request_features = dict(features)
    features = _handle_missing_values(runtime=runtime, features=request_features)

    used_inputs: dict[str, Any] = dict(features)
    missing_features: list[str] = []

    # Convert integers to floats
    for name, value in features.items():
        if isinstance(value, int):
            features[name] = float(value)

    model_uri = _runtime_mlflow_model_uri(runtime)

    try:
        # Use cached model (avoids 9s download+deserialize per request)
        model = _get_cached_mlflow_model(model_uri)

        # Convert features to DataFrame
        df = pd.DataFrame([features])

        # Align DataFrame to model signature
        if (
            model.metadata
            and model.metadata.signature
            and model.metadata.signature.inputs
        ):
            input_specs = list(model.metadata.signature.inputs)
            required_cols = [inp.name for inp in input_specs]
            missing_features = [
                col for col in required_cols if col not in request_features
            ]
            extra_cols = [c for c in df.columns if c not in required_cols]

            if extra_cols:
                df = df.drop(columns=extra_cols)

            input_types = {
                spec.name: str(spec.type).lower()
                for spec in input_specs
                if spec.name is not None
            }

            for col in required_cols:
                if col not in df.columns:
                    if "datetime" in input_types.get(col, ""):
                        df[col] = _mlflow_missing_datetime_value()
                    else:
                        df[col] = 0.0

            for spec in input_specs:
                col = spec.name
                expected = str(spec.type).lower()
                if "string" in expected:
                    df[col] = df[col].where(df[col].notna(), "").astype(str)
                    continue
                if "datetime" in expected:
                    df[col] = _coerce_mlflow_datetime_column(df[col])
                    continue

                numeric = pd.to_numeric(df[col], errors="coerce").fillna(0)
                if "long" in expected or "int" in expected:
                    df[col] = numeric.astype("int64")
                else:
                    df[col] = numeric.astype("float32")

            df = df[required_cols]
            used_inputs = {
                str(key): _json_safe_prediction_input_value(value)
                for key, value in df.iloc[0].to_dict().items()
            }

        predictions = model.predict(df)

    except Exception as e:
        logger.error(f"MLflow prediction error: {e}")
        traceback.print_exc()
        raise ValueError(
            f"Failed to load or predict with MLflow model: {e}. "
            f"Check that the model is registered and the tracking URI is correct."
        )

    result = _normalize_mlflow_prediction_output(predictions)
    result["used_inputs"] = used_inputs
    result["missing_features"] = missing_features
    return result


def _predict_sota_joblib(
    runtime: ModelRuntime,
    features: dict[str, Any],
) -> dict[str, Any]:
    adapter = runtime.artifact
    if adapter is None or not hasattr(adapter, "predict_frame"):
        raise ValueError("SOTA joblib model adapter is not available.")

    request_features = dict(features)
    predictions = adapter.predict_frame(request_features)
    from ml.core.sota_joblib import normalize_prediction_output

    result = normalize_prediction_output(predictions)
    feature_names = runtime.feature_names or getattr(adapter, "feature_names", [])
    used_inputs = {
        name: request_features.get(name)
        for name in feature_names
        if name in request_features
    }
    result["used_inputs"] = used_inputs
    result["missing_features"] = [
        name for name in feature_names if name not in request_features
    ]
    return result


# ── Main dispatcher ──────────────────────────────────────────


def run_prediction(runtime: ModelRuntime, features: dict[str, Any]) -> dict[str, Any]:
    """
    Run a prediction on any loaded model.

    This is the SINGLE ENTRY POINT for all predictions.
    It dispatches to the right function based on runtime.model_type.

    Returns a dict with:
        model_id, aliases, model_type, file_path,
        prediction, probability, score, used_inputs, missing_features
    """
    if runtime.status != "loaded":
        raise ValueError(runtime.load_error or "Model is not loaded.")

    if runtime.model_type == "linear_custom":
        details = _predict_linear_custom(runtime, features)
    elif runtime.model_type == "online_agent_snapshot":
        details = _predict_online_agent(runtime, features)
    elif runtime.model_type == "mlflow":
        details = _predict_mlflow(runtime, features)
    elif runtime.model_type == "sota_joblib":
        details = _predict_sota_joblib(runtime, features)
    else:
        details = _predict_estimator(runtime, features)
    details = _apply_prediction_output_contract(runtime, details)

    return {
        "model_id": runtime.model_id,
        "aliases": runtime.aliases,
        "model_type": runtime.model_type,
        "file_path": str(runtime.file_path),
        **details,
    }


def run_prediction_batch(
    runtime: ModelRuntime,
    feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row or {}) for row in feature_rows if isinstance(row, dict)]
    if not rows:
        return []
    if runtime.model_type == "mlflow" and _mlflow_subprocess_enabled():
        outputs = _predict_mlflow_subprocess_batch(runtime, rows)
        return [
            {
                "model_id": runtime.model_id,
                "aliases": runtime.aliases,
                "model_type": runtime.model_type,
                "file_path": str(runtime.file_path),
                **_apply_prediction_output_contract(runtime, output),
            }
            for output in outputs
        ]
    return [run_prediction(runtime, row) for row in rows]
