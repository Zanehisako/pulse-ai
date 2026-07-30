"""
ModelRuntime + ModelRegistry — extracted from api.py.
Discovers .pkl files, loads them, provides lookup by id/alias.
ZERO framework dependency — only stdlib + numpy + pickle.

WHERE THIS CAME FROM:
    api.py::ModelRuntime (dataclass)
    api.py::ModelRegistry (class with refresh/get/loaded/list_models)
    api.py::_resolve_model_path, _coerce_feature_value, _prepare_vector
"""

import hashlib
import json
import logging
import pickle
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ── Changed: import from ml_core.utils instead of local _functions ──
from ml.core.model_metadata import (
    default_description_for_task,
    default_examples_for_task,
    infer_model_task,
    runtime_model_overrides,
    runtime_prediction_defaults,
)
from ml.core.utils import encode_with_classes, numeric, safe_slug

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# This resolves to D:\Projects\pios\backendMulti\
# Which is correct — it's the Django project root.

# MLflow tracking URI points to where training scripts store models
# All training scripts run from ml-backend/scripts/ and use default ./mlruns

#tracking_path = PROJECT_ROOT.parent / "ml-backend" / "scripts" / "mlruns"
#mlflow.set_tracking_uri(tracking_path.as_uri())
# ── Helper functions (were private in api.py) ────────────────────

_MLFLOW_URI_PREFIXES = ("models:/", "runs:/", "mlflow-artifacts:/", "file:")


def _clean_model_identifier(model_id: str) -> str:
    clean_id = str(model_id or "")
    for token in ("_champion", "_challenger", "_@"):
        clean_id = clean_id.split(token)[0]
    return clean_id


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _looks_like_placeholder_description(
    description: str,
    *,
    model_id: str,
    file_path: Path | str,
) -> bool:
    text = str(description or "").strip()
    if not text:
        return True

    lowered = text.lower()
    candidates = {str(model_id).strip().lower(), safe_slug(model_id).lower()}

    if isinstance(file_path, Path):
        candidates.add(file_path.name.lower())
        candidates.add(str(file_path).lower())
    else:
        raw_path = str(file_path).strip()
        if raw_path:
            candidates.add(raw_path.lower())
            candidates.add(raw_path.rstrip("/").rsplit("/", 1)[-1].lower())

    return lowered in candidates


def _is_mlflow_uri(path: str) -> bool:
    """Return True if *path* is an MLflow model URI rather than a filesystem path."""
    return any(path.startswith(prefix) for prefix in _MLFLOW_URI_PREFIXES)


def _resolve_model_path(raw_path: str, models_dir: Path) -> Path | str:
    """
    Resolve a model file_path from config to an actual Path.
    Handles: absolute paths, relative paths, fallback to models_dir.
    MLflow URIs (models:/, runs:/, mlflow-artifacts:/, file:) are returned
    as-is (plain strings) — they must NOT be wrapped in Path().

    This is needed because config.json has paths like:
        "/Users/mac/Documents/pios-1/ml-backend/models/donor_v1.pkl"  (absolute, maybe wrong machine)
        "models/donor_v1.pkl"  (relative to project root)
        "models:/hospital_shortage_predictor@champion"  (MLflow URI)

    We try the path as-is, then fall back to just the filename in models_dir.
    """
    if _is_mlflow_uri(raw_path):
        return raw_path

    candidate = Path(raw_path)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        # Absolute path doesn't exist (different machine) — try just the filename
        return models_dir / candidate.name
    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    model_candidate = (models_dir / candidate).resolve()
    if model_candidate.exists():
        return model_candidate
    return (models_dir / candidate.name).resolve()


def _coerce_feature_value(
    feature_name: str,
    value: Any,
    all_inputs: dict[str, Any],
    encoders: dict[str, list[Any]] | None = None,
) -> float:
    """
    Convert a single feature value to float for model input.

    Priority:
    1. If there's an encoder for this feature → use encode_with_classes
    2. If value is not None → use numeric()
    3. If feature name has underscore (like blood_type_A-) → check one-hot encoding
    4. Otherwise → 0.0
    """
    if encoders and feature_name in encoders:
        return encode_with_classes(value, encoders[feature_name])
    if value is not None:
        return numeric(value)
    # Handle one-hot encoded features like "hospital_East" or "blood_type_A-"
    if "_" in feature_name:
        prefix, suffix = feature_name.split("_", 1)
        base_value = all_inputs.get(prefix)
        if base_value is not None and str(base_value).strip():
            return 1.0 if str(base_value).strip().lower() == suffix.lower() else 0.0
    return 0.0


def _prepare_vector(
    feature_names: list[str],
    features: dict[str, Any],
    encoders: dict[str, list[Any]] | None = None,
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    """
    Build a numpy feature vector from a dict of features.

    Returns:
        - np.ndarray of shape (1, n_features) — ready for model.predict()
        - dict of used_inputs (feature_name → float value used)
        - list of missing feature names
    """
    row: list[float] = []
    used_inputs: dict[str, float] = {}
    missing: list[str] = []

    for name in feature_names:
        value = features.get(name)
        if value is None and name not in features:
            missing.append(name)
        coerced = _coerce_feature_value(name, value, features, encoders=encoders)
        used_inputs[name] = float(coerced)
        row.append(float(coerced))

    return np.array([row], dtype=float), used_inputs, missing


def _config_aliases(config_row: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    defaults = config_row.get("defaults")
    if isinstance(defaults, dict):
        discovery = defaults.get("model_discovery")
        if isinstance(discovery, dict):
            catalog_model_id = str(discovery.get("catalog_model_id") or "").strip()
            if catalog_model_id:
                aliases.append(catalog_model_id)
    return aliases


# ── ModelRuntime (identical to api.py) ───────────────────────────


@dataclass
class ModelRuntime:
    model_id: str
    aliases: list[str]
    slug: str
    file_path: Path | str
    description: str
    feature_names: list[str] = field(default_factory=list)
    feature_info: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)
    defaults: dict[str, Any] = field(default_factory=dict)
    model_type: str = "unknown"
    status: str = "not_loaded"
    load_error: str | None = None
    artifact: Any = None
    estimator: Any = None
    params: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)


# ── ModelRegistry (identical to api.py) ────────────��─────────────


class ModelRegistry:
    """
    Discovers model files, loads them, provides lookup by id/alias.

    This class has ZERO framework dependency. It works the same whether
    called from FastAPI, Django, or a plain Python script.

    Config can come from:
        1. config_loader callback (Django ORM in our case)
        2. config.json file (fallback)

    Change detection: SHA-256 of config + file mtimes/sizes.
    If nothing changed, refresh() is a no-op.
    """

    def __init__(
        self,
        models_dir: Path,
        config_path: Path,
        config_loader: Callable[[], list[dict[str, Any]]] | None = None,
    ):
        self.models_dir = models_dir
        self.config_path = config_path
        self._config_loader = config_loader
        self._lock = threading.RLock()
        self._models_by_id: dict[str, ModelRuntime] = {}
        self._alias_to_id: dict[str, str] = {}
        self._config_signature: str | None = None

    # ── Load config ──────────────────────────────────────

    def _load_config_models(self) -> list[dict[str, Any]]:
        if self._config_loader is not None:
            try:
                rows = self._config_loader()
            except Exception:
                return []
            if not isinstance(rows, list):
                return []
            return [row for row in rows if isinstance(row, dict)]

        # Lazy import — only runs after Django apps are ready
        try:
            from ml.models import MLModelConfig
        except Exception:
            return []

        try:
            rows = list(MLModelConfig.objects.values())
            return rows
        except Exception:
            return []

    # ── Change detection ─────────────────────────────────

    def _compute_signature(
        self,
        config_models: list[dict[str, Any]],
        model_files: list[Path],
    ) -> str:
        """SHA-256 of config + file stats. If unchanged, skip reload."""
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                config_models,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        for f in model_files:
            digest.update(f.name.encode("utf-8"))
            try:
                stat = f.stat()
                digest.update(str(stat.st_mtime_ns).encode("utf-8"))
                digest.update(str(stat.st_size).encode("utf-8"))
            except OSError:
                continue
        return digest.hexdigest()

    # ── Core refresh logic ───────────────────────────────

    def refresh(self, *, force: bool = False) -> bool:
        """
        Scan models_dir for .pkl files, match with config, load each one.
        Returns True if anything changed.

        THIS IS THE MAIN ENTRY POINT. Called:
        - Once at startup (force=True)
        - On config reload
        - On periodic poll
        """
        config_models = self._load_config_models()
        """
        -Todo we will have to change this to accept everything
        g"""

        discovered_files: list[Path | str] = []
        for config_model in config_models:
            raw = config_model["file_path"]
            if isinstance(raw, str) and _is_mlflow_uri(raw):
                discovered_files.append(raw)
            else:
                discovered_files.append(_resolve_model_path(str(raw), self.models_dir))

        logger.debug("Discovered model files during refresh: %s", discovered_files)
        # signature = self._compute_signature(config_models, discovered_files)

        # with self._lock:
        #     if not force and signature == self._config_signature:
        #         return False

        models_by_id: dict[str, ModelRuntime] = {}
        alias_to_id: dict[str, str] = {}

        # Build a lookup: filename → config row
        config_by_filename: dict[str, dict[str, Any]] = {}
        for config_row in config_models:
            file_path = config_row.get("file_path")
            if config_row.get("model_type", config_row.get("type")) == "mlflow":
                config_by_filename[str(file_path)] = config_row

            elif isinstance(file_path, str):
                resolved = _resolve_model_path(file_path, self.models_dir)
                key = resolved.name if isinstance(resolved, Path) else resolved
                config_by_filename[key] = config_row

        # First pass: load every .pkl file found on disk
        logger.debug("First registry pass: load discovered model files.")
        for model_file in discovered_files:
            if isinstance(model_file, str):
                lookup_key = model_file
            else:
                lookup_key = model_file.name
            config_row = config_by_filename.get(lookup_key, {})
            logger.debug("Registry config row for %s: %s", lookup_key, config_row)
            config_id = config_row.get("id") or config_row.get("model_id")
            if isinstance(config_id, str) and config_id.strip():
                model_id = safe_slug(config_id)
            elif isinstance(model_file, Path):
                model_id = safe_slug(model_file.stem)
            else:
                model_id = safe_slug(model_file.rstrip("/").rsplit("/", 1)[-1])
            aliases = [model_id]
            if isinstance(model_file, Path):
                aliases.append(safe_slug(model_file.stem))
            else:
                aliases.append(safe_slug(model_file.rstrip("/").rsplit("/", 1)[-1]))
            if isinstance(config_id, str) and config_id.strip():
                aliases.append(config_id.strip())
            aliases.extend(_config_aliases(config_row))

            runtime = ModelRuntime(
                model_id=model_id,
                aliases=list(dict.fromkeys(aliases)),
                slug=model_id,
                file_path=model_file,
                model_type=str(
                    config_row.get("type", config_row.get("model_type", "unknown"))
                ),
                description=str(config_row.get("description", model_file)),
                feature_names=[
                    str(f) for f in config_row.get("features", []) if isinstance(f, str)
                ],
                feature_info=config_row.get("feature_info", {})
                if isinstance(config_row.get("feature_info"), dict)
                else {},
                examples=config_row.get("examples", [])
                if isinstance(config_row.get("examples"), list)
                else [],
                defaults=config_row.get("defaults", {})
                if isinstance(config_row.get("defaults"), dict)
                else {},
            )
            self._load_runtime(runtime)
            self._apply_runtime_metadata(runtime)
            self._register(runtime, models_by_id=models_by_id, alias_to_id=alias_to_id)

        # Second pass: config entries pointing to files not yet registered
        logger.debug("Second registry pass: config entries not yet registered.")
        for config_row in config_models:
            file_path = config_row.get("file_path")
            model_id = config_row.get("id") or config_row.get("model_id")
            if not isinstance(file_path, str) or not isinstance(model_id, str):
                continue
            resolved = _resolve_model_path(file_path, self.models_dir)
            # MLflow URIs cannot be checked with exists(); skip that gate
            if not _is_mlflow_uri(file_path):
                if not isinstance(resolved, Path) or not resolved.exists():
                    continue
            alias_key = model_id.strip().lower()
            if alias_key in alias_to_id:
                continue

            runtime = ModelRuntime(
                model_id=safe_slug(model_id),
                aliases=[safe_slug(model_id), model_id, *_config_aliases(config_row)],
                slug=safe_slug(model_id),
                file_path=resolved,
                description=str(
                    config_row.get(
                        "description",
                        resolved.name if isinstance(resolved, Path) else resolved,
                    )
                ),
                feature_names=[
                    str(f) for f in config_row.get("features", []) if isinstance(f, str)
                ],
                feature_info=config_row.get("feature_info", {})
                if isinstance(config_row.get("feature_info"), dict)
                else {},
                examples=config_row.get("examples", [])
                if isinstance(config_row.get("examples"), list)
                else [],
                defaults=config_row.get("defaults", {})
                if isinstance(config_row.get("defaults"), dict)
                else {},
                model_type=str(
                    config_row.get("type", config_row.get("model_type", "unknown"))
                ),
            )
            self._load_runtime(runtime)
            self._apply_runtime_metadata(runtime)
            self._register(runtime, models_by_id=models_by_id, alias_to_id=alias_to_id)

        with self._lock:
            self._models_by_id = models_by_id
            self._alias_to_id = alias_to_id
            # self._config_signature = signature
        logger.debug("Finished registry refresh.")
        return True

    # ── Internal helpers ─────────────────────────────────

    def _register(
        self,
        runtime: ModelRuntime,
        *,
        models_by_id: dict[str, ModelRuntime],
        alias_to_id: dict[str, str],
    ) -> None:
        """Add a runtime to the lookup dicts."""
        models_by_id[runtime.model_id] = runtime
        for alias in runtime.aliases + [runtime.slug]:
            alias_to_id[alias.lower()] = runtime.model_id

    def _extract_feature_names(self, runtime: ModelRuntime, payload: Any) -> list[str]:
        """
        Try to extract feature names from the model artifact.
        Checks (in order): config, metadata dict, sklearn feature_names_in_, n_features_in_.
        """
        if runtime.feature_names:
            return runtime.feature_names

        if isinstance(payload, dict):
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                metadata_features = metadata.get("features")
                if isinstance(metadata_features, list):
                    return [str(f) for f in metadata_features]
            state = payload.get("state")
            if isinstance(state, dict):
                active = state.get("active_features")
                if isinstance(active, list):
                    return [str(f) for f in active]
            params = payload.get("params")
            if isinstance(params, dict):
                weights = params.get("weights")
                if isinstance(weights, dict):
                    return [str(f) for f in weights.keys()]

        estimator = payload
        if isinstance(payload, dict) and "model_binary" in payload:
            estimator = payload["model_binary"]

        feature_names = getattr(estimator, "feature_names_in_", None)
        if feature_names is not None:
            return [str(f) for f in list(feature_names)]

        n_features = getattr(estimator, "n_features_in_", None)
        if isinstance(n_features, int) and n_features > 0:
            return [f"feature_{i + 1}" for i in range(n_features)]

        return []

    def _load_runtime(self, runtime: ModelRuntime) -> None:
        """
        Load a the file and classify it by architecture type.

        #TODO load any file (sklearn,xgboost,pytorch,mlflow ...etc)

        Four model types are detected:
        1. linear_custom  — dict with "architecture": "linear_custom"
        2. online_agent_snapshot — dict with "state" key (PersistentAdamAgent)
        3. estimator — anything with .predict() (sklearn, xgb, etc.)
        4. MLFlow Models uri "mlflow::"
        """
        logger.debug("Loading runtime: %s", runtime)
        if runtime.model_type == "mlflow":
            runtime.status = "loaded"
            return

        if runtime.model_type == "sota_joblib":
            if not isinstance(runtime.file_path, Path) or not runtime.file_path.exists():
                runtime.status = "error"
                runtime.load_error = f"Model file not found: {runtime.file_path}"
                return
            try:
                from ml.core.sota_joblib import load_sota_joblib_model

                adapter_config = runtime.defaults.get("sota_joblib", {})
                if not isinstance(adapter_config, dict):
                    adapter_config = {}
                adapter_config = {
                    **adapter_config,
                    "features": runtime.feature_names,
                }
                runtime.artifact = load_sota_joblib_model(
                    runtime.file_path,
                    adapter_config,
                )
                runtime.feature_names = runtime.feature_names or runtime.artifact.feature_names
                runtime.status = "loaded"
            except Exception as exc:
                runtime.status = "error"
                runtime.load_error = f"{type(exc).__name__}: {exc}"
            return

        elif isinstance(runtime.file_path, Path) and not runtime.file_path.exists():
            runtime.status = "error"
            runtime.load_error = f"Model file not found: {runtime.file_path}"
            return

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fp = runtime.file_path
                if not isinstance(fp, Path):
                    raise TypeError(f"Expected Path for pickle load, got: {fp!r}")
                artifact = pickle.loads(fp.read_bytes())
        except Exception as exc:
            runtime.status = "error"
            runtime.load_error = f"{type(exc).__name__}: {exc}"
            return

        runtime.artifact = artifact
        runtime.feature_names = self._extract_feature_names(runtime, artifact)
        runtime.status = "loaded"

        # Type 1: linear_custom (hand-rolled logistic regression)
        if isinstance(artifact, dict) and "architecture" in artifact:
            arch = str(artifact.get("architecture", "unknown"))
            runtime.model_type = arch
            if arch == "linear_custom":
                runtime.params = (
                    artifact.get("params", {})
                    if isinstance(artifact.get("params"), dict)
                    else {}
                )
                return
            estimator = artifact.get("model_binary")
            runtime.estimator = estimator
            return

        # Type 2: online_agent_snapshot (PersistentAdamAgent saved state)
        if isinstance(artifact, dict) and "state" in artifact:
            runtime.model_type = "online_agent_snapshot"
            runtime.state = (
                artifact["state"] if isinstance(artifact["state"], dict) else {}
            )
            return

        # Type 3: estimator (sklearn, xgboost, catboost, etc.)
        runtime.estimator = artifact
        runtime.model_type = type(artifact).__name__

    def _apply_runtime_metadata(self, runtime: ModelRuntime) -> None:
        lookup_id = safe_slug(_clean_model_identifier(runtime.model_id))
        overrides = runtime_model_overrides()
        override = overrides.get(lookup_id) or overrides.get(runtime.model_id) or {}
        task = infer_model_task(
            lookup_id or runtime.model_id,
            runtime.model_type,
            runtime.feature_names,
        )

        override_description = str(override.get("description") or "").strip()
        if _looks_like_placeholder_description(
            runtime.description,
            model_id=lookup_id or runtime.model_id,
            file_path=runtime.file_path,
        ):
            runtime.description = (
                override_description or default_description_for_task(task)
            )

        if not isinstance(runtime.examples, list) or not any(
            isinstance(item, dict) for item in runtime.examples
        ):
            runtime.examples = default_examples_for_task(task)

        defaults = {}
        runtime_defaults = runtime_prediction_defaults(lookup_id or runtime.model_id)
        if isinstance(runtime_defaults, dict):
            defaults = _deep_merge_dicts(defaults, runtime_defaults)
        if isinstance(runtime.defaults, dict):
            defaults = _deep_merge_dicts(defaults, runtime.defaults)
        runtime.defaults = defaults

    # ── Public API ───────────────────────────────────────

    def list_models(self) -> list[ModelRuntime]:
        """Return all discovered models, sorted by id."""
        with self._lock:
            return sorted(self._models_by_id.values(), key=lambda m: m.model_id)

    # def remove(self,model_id: str) -> None:
    #     """Remove a model by id."""

    def get(self, identifier: str) -> ModelRuntime | None:
        """Lookup a model by id or alias (case-insensitive)."""
        key = identifier.lower()
        with self._lock:
            mid = self._alias_to_id.get(key)
            if mid:
                return self._models_by_id.get(mid)
            return self._models_by_id.get(key)

    def loaded(self) -> list[ModelRuntime]:
        """Return only successfully loaded models."""
        return [rt for rt in self.list_models() if rt.status == "loaded"]
