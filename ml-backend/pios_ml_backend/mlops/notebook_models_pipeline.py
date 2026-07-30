from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pyspark.ml.classification import (
    GBTClassifier,
    LogisticRegression as SparkLogisticRegression,
    RandomForestClassifier as SparkRandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (
    GBTRegressor,
    LinearRegression as SparkLinearRegression,
    RandomForestRegressor as SparkRandomForestRegressor,
)
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark.sql import functions as F

from pios_ml_backend.mlops.feast_mlflow_pipeline import DependencyError, PipelineError


def _require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name, fromlist=["*"])
    except ModuleNotFoundError as exc:
        raise DependencyError(
            f"Missing optional dependency '{module_name}'. Install it with: {install_hint}"
        ) from exc


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_") or "model"


def _feature_ref(feature_view: str, feature_name: str) -> str:
    return f"{feature_view}:{feature_name}"


@dataclass(frozen=True)
class NotebookPipelineSettings:
    backend_root: Path
    feature_repo: Path
    feature_data_dir: Path
    generated_feature_definitions: Path
    metadata_path: Path
    tracking_uri: str
    experiment_name: str
    default_model_config: Path
    default_training_spec: Path

    @classmethod
    def from_env(cls) -> "NotebookPipelineSettings":
        backend_root = Path(__file__).resolve().parents[2]
        feature_repo = Path(
            os.getenv("PIOS_FEAST_REPO", str(backend_root / "feature_repo"))
        ).resolve()

        return cls(
            backend_root=backend_root,
            feature_repo=feature_repo,
            feature_data_dir=(feature_repo / "data").resolve(),
            generated_feature_definitions=(
                feature_repo / "notebook_generated_feature_definitions.py"
            ).resolve(),
            metadata_path=Path(
                os.getenv(
                    "PIOS_MLFLOW_NOTEBOOK_RUNS_FILE",
                    str(backend_root / "artifacts" / "mlflow_notebook_runs.json"),
                )
            ).resolve(),
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI", f"sqlite:///{backend_root / 'mlflow.db'}"
            ),
            experiment_name=os.getenv(
                "MLFLOW_EXPERIMENT_NOTEBOOKS", "pios-notebook-models"
            ),
            default_model_config=(
                backend_root / "notebooks" / "ml_models" / "config.json"
            ).resolve(),
            default_training_spec=(
                backend_root / "notebooks" / "ml_models" / "training_spec.json"
            ).resolve(),
        )


@dataclass(frozen=True)
class NotebookModelSpec:
    model_id: str
    description: str
    features: tuple[str, ...]
    source_model_path: str | None
    dataset_csv: Path
    target_column: str
    task: str
    algorithm: str
    timestamp_column: str | None
    entity_column: str | None
    entity_columns: tuple[str, ...]
    group_by_columns: tuple[str, ...]
    feature_view_name: str
    feature_service_name: str
    model_registry_name: str
    model_output_path: Path
    register_model: bool
    test_size: float
    random_state: int


@dataclass(frozen=True)
class PreparedModelData:
    spec: NotebookModelSpec
    entity_column: str
    event_timestamp_column: str
    feature_columns: tuple[str, ...]
    target_column: str
    feature_types: Mapping[str, str]
    feature_parquet_relative_path: str
    prepared_entity_target_df: pd.DataFrame
    filled_missing_features: tuple[str, ...]
    categorical_encodings: Mapping[str, Mapping[str, int]]


class NotebookModelsFeastMLflowPipeline:
    ENTITY_COLUMN = "entity_id"
    EVENT_TIMESTAMP_COLUMN = "event_timestamp"

    def __init__(self, settings: NotebookPipelineSettings | None = None):
        self.settings = settings or NotebookPipelineSettings.from_env()
        self._spark_session: SparkSession | None = None

    @classmethod
    def from_env(cls) -> "NotebookModelsFeastMLflowPipeline":
        return cls(settings=NotebookPipelineSettings.from_env())

    def run(
        self,
        model_config_path: Path | None = None,
        training_spec_path: Path | None = None,
        selected_model_ids: Sequence[str] | None = None,
        materialize_online: bool = True,
        allow_missing_features: bool = True,
        register_models: bool = False,
    ) -> dict[str, Any]:
        config_path = (model_config_path or self.settings.default_model_config).resolve()
        training_path = (
            training_spec_path.resolve()
            if training_spec_path is not None
            else self.settings.default_training_spec
        )

        catalog = self._load_notebook_model_catalog(config_path)
        training_specs = self._load_training_specs(training_path)
        specs = self._build_specs(
            catalog=catalog,
            training_specs=training_specs,
            selected_model_ids=selected_model_ids,
        )

        if not specs:
            raise PipelineError(
                "No notebook model found to process. Check --model-id values or config file."
            )

        prepared_models = [
            self._prepare_model_data(spec, allow_missing_features=allow_missing_features)
            for spec in specs
        ]

        self._write_generated_feature_definitions(prepared_models)
        self.apply_feast_repo()
        if materialize_online:
            self.materialize_incremental()

        results = [
            self._train_and_log_model(model, register_models=register_models)
            for model in prepared_models
        ]

        payload = {
            "experiment": self.settings.experiment_name,
            "tracking_uri": self.settings.tracking_uri,
            "feature_repo": str(self.settings.feature_repo),
            "generated_feature_definitions": str(
                self.settings.generated_feature_definitions
            ),
            "processed_models": results,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._write_run_summary(payload)
        return payload

    def apply_feast_repo(self) -> None:
        self._run_feast_cli(["apply"])

    def materialize_incremental(self) -> None:
        feast_module = _require_dependency("feast", "pip install feast")
        store = feast_module.FeatureStore(repo_path=str(self.settings.feature_repo))
        store.materialize_incremental(end_date=datetime.now(timezone.utc))

    def _load_notebook_model_catalog(self, config_path: Path) -> list[dict[str, Any]]:
        if not config_path.exists():
            raise PipelineError(f"Notebook model config file not found: {config_path}")

        payload = json.loads(config_path.read_text(encoding="utf-8"))
        models = payload.get("models")
        if not isinstance(models, list):
            raise PipelineError(
                f"Invalid notebook model config format in {config_path}: 'models' must be a list."
            )
        return models

    def _load_training_specs(self, training_path: Path) -> dict[str, dict[str, Any]]:
        if not training_path.exists():
            return {}

        payload = json.loads(training_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if isinstance(payload.get("models"), list):
                result: dict[str, dict[str, Any]] = {}
                for item in payload["models"]:
                    if isinstance(item, dict) and item.get("id"):
                        result[str(item["id"])] = item
                return result
            if isinstance(payload.get("models"), dict):
                return {
                    str(model_id): spec
                    for model_id, spec in payload["models"].items()
                    if isinstance(spec, dict)
                }
            return {}

        if isinstance(payload, list):
            result = {}
            for item in payload:
                if isinstance(item, dict) and item.get("id"):
                    result[str(item["id"])] = item
            return result

        return {}

    def _build_specs(
        self,
        catalog: Sequence[dict[str, Any]],
        training_specs: Mapping[str, dict[str, Any]],
        selected_model_ids: Sequence[str] | None,
    ) -> list[NotebookModelSpec]:
        selected = {item.strip() for item in selected_model_ids or [] if item.strip()}
        if selected:
            missing = sorted(
                selected - {str(item.get("id", "")).strip() for item in catalog}
            )
            if missing:
                raise PipelineError(
                    f"Model id(s) not present in notebook config: {', '.join(missing)}"
                )

        specs: list[NotebookModelSpec] = []
        for model_payload in catalog:
            model_id = str(model_payload.get("id", "")).strip()
            if not model_id:
                continue
            if selected and model_id not in selected:
                continue

            features_raw = model_payload.get("features", [])
            if not isinstance(features_raw, list) or not features_raw:
                raise PipelineError(
                    f"Model '{model_id}' has no features in notebook model config."
                )
            features = tuple(str(item) for item in features_raw)

            training_payload = dict(training_specs.get(model_id, {}))
            fallback = self._infer_training_defaults(model_id)

            dataset_csv = self._resolve_path(
                training_payload.get("dataset_csv") or fallback["dataset_csv"]
            )
            if dataset_csv is None or not dataset_csv.exists():
                raise PipelineError(
                    f"Dataset for model '{model_id}' was not found. "
                    "Set it in training spec with 'dataset_csv'."
                )

            task = str(training_payload.get("task") or fallback["task"]).strip().lower()
            if task not in {"classification", "regression"}:
                raise PipelineError(
                    f"Unsupported task '{task}' for model '{model_id}'. "
                    "Use 'classification' or 'regression'."
                )

            algorithm = str(
                training_payload.get("algorithm") or fallback["algorithm"]
            ).strip()
            register_model = bool(training_payload.get("register_model", False))

            feature_view_name = str(
                training_payload.get("feature_view_name")
                or f"{_slugify(model_id)}_features"
            )
            feature_service_name = str(
                training_payload.get("feature_service_name")
                or f"{_slugify(model_id)}_service"
            )
            registry_name = str(
                training_payload.get("model_registry_name")
                or f"pios_{_slugify(model_id)}"
            )

            model_output_path = self._resolve_model_output_path(
                training_payload.get("model_output_path"), model_id
            )

            entity_columns = training_payload.get("entity_columns") or fallback.get(
                "entity_columns", []
            )
            if not isinstance(entity_columns, list):
                entity_columns = []

            group_by_columns = training_payload.get("group_by_columns") or fallback.get(
                "group_by_columns", []
            )
            if not isinstance(group_by_columns, list):
                group_by_columns = []

            spec = NotebookModelSpec(
                model_id=model_id,
                description=str(model_payload.get("description", "")),
                features=features,
                source_model_path=model_payload.get("file_path"),
                dataset_csv=dataset_csv,
                target_column=str(
                    training_payload.get("target_column") or fallback["target_column"]
                ),
                task=task,
                algorithm=algorithm,
                timestamp_column=training_payload.get("timestamp_column")
                or fallback.get("timestamp_column"),
                entity_column=training_payload.get("entity_column")
                or fallback.get("entity_column"),
                entity_columns=tuple(str(item) for item in entity_columns),
                group_by_columns=tuple(str(item) for item in group_by_columns),
                feature_view_name=feature_view_name,
                feature_service_name=feature_service_name,
                model_registry_name=registry_name,
                model_output_path=model_output_path,
                register_model=register_model,
                test_size=float(training_payload.get("test_size", 0.2)),
                random_state=int(training_payload.get("random_state", 42)),
            )
            specs.append(spec)

        return specs

    def _infer_training_defaults(self, model_id: str) -> dict[str, Any]:
        slug = _slugify(model_id)
        if "donor" in slug:
            return {
                "dataset_csv": self.settings.backend_root
                / "datasets"
                / "blood_registry_sythentic"
                / "data"
                / "blood_donation_registry_ml_ready.csv",
                "target_column": "donated_next_6m",
                "task": "classification",
                "algorithm": "logistic_regression",
                "timestamp_column": "as_of_date",
                "entity_column": "donor_id",
                "entity_columns": [],
                "group_by_columns": [],
            }

        if "demand" in slug or "forecast" in slug or "xgb" in slug:
            return {
                "dataset_csv": self.settings.backend_root
                / "datasets"
                / "synthetic_bloodbank_daily.csv",
                "target_column": "heavy_demand_next_day",
                "task": "classification",
                "algorithm": "xgboost_classifier",
                "timestamp_column": "date",
                "entity_column": None,
                "entity_columns": ["hospital", "blood_type"],
                "group_by_columns": ["hospital", "blood_type"],
            }

        return {
            "dataset_csv": self.settings.backend_root
            / "datasets"
            / "synthetic_bloodbank_daily.csv",
            "target_column": "heavy_demand_next_day",
            "task": "classification",
            "algorithm": "logistic_regression",
            "timestamp_column": "date",
            "entity_column": None,
            "entity_columns": ["hospital", "blood_type"],
            "group_by_columns": ["hospital", "blood_type"],
        }

    def _prepare_model_data(
        self, spec: NotebookModelSpec, allow_missing_features: bool
    ) -> PreparedModelData:
        if not spec.dataset_csv.exists():
            raise PipelineError(f"Dataset not found for model '{spec.model_id}': {spec.dataset_csv}")

        frame = pd.read_csv(spec.dataset_csv)
        if spec.target_column not in frame.columns:
            raise PipelineError(
                f"Target column '{spec.target_column}' is missing in dataset "
                f"for model '{spec.model_id}'."
            )

        timestamp_source = self._select_timestamp_column(frame, spec)
        frame[self.EVENT_TIMESTAMP_COLUMN] = pd.to_datetime(
            frame[timestamp_source], utc=True, errors="coerce"
        )
        if frame[self.EVENT_TIMESTAMP_COLUMN].isna().any():
            raise PipelineError(
                f"Unable to parse timestamp values from '{timestamp_source}' "
                f"for model '{spec.model_id}'."
            )

        frame[self.ENTITY_COLUMN] = self._build_entity_column(frame, spec)

        working = frame.copy()
        working = self._apply_feature_engineering(working, spec)

        filled_missing: list[str] = []
        missing_features = sorted(set(spec.features) - set(working.columns))
        if missing_features and not allow_missing_features:
            raise PipelineError(
                f"Model '{spec.model_id}' is missing required features: {missing_features}"
            )

        for feature in missing_features:
            working[feature] = 0.0
            filled_missing.append(feature)

        typed_features, encodings = self._coerce_feature_columns(
            frame=working, feature_columns=spec.features
        )
        working = typed_features
        working[spec.target_column] = self._coerce_target(
            series=working[spec.target_column], task=spec.task, model_id=spec.model_id
        )

        prepared = working.dropna(
            subset=[
                self.ENTITY_COLUMN,
                self.EVENT_TIMESTAMP_COLUMN,
                spec.target_column,
                *spec.features,
            ]
        ).copy()
        if prepared.empty:
            raise PipelineError(
                f"Prepared dataset is empty for model '{spec.model_id}' "
                "after feature processing and NA filtering."
            )

        feature_types = self._resolve_feature_types(prepared, spec.features)
        model_slug = _slugify(spec.model_id)
        parquet_name = f"notebook_{model_slug}_features.parquet"
        parquet_path = (self.settings.feature_data_dir / parquet_name).resolve()
        self.settings.feature_data_dir.mkdir(parents=True, exist_ok=True)

        final_columns = [
            self.ENTITY_COLUMN,
            self.EVENT_TIMESTAMP_COLUMN,
            *spec.features,
            spec.target_column,
        ]
        prepared[final_columns].to_parquet(parquet_path, index=False)

        return PreparedModelData(
            spec=spec,
            entity_column=self.ENTITY_COLUMN,
            event_timestamp_column=self.EVENT_TIMESTAMP_COLUMN,
            feature_columns=spec.features,
            target_column=spec.target_column,
            feature_types=feature_types,
            feature_parquet_relative_path=f"data/{parquet_name}",
            prepared_entity_target_df=prepared[
                [self.ENTITY_COLUMN, self.EVENT_TIMESTAMP_COLUMN, spec.target_column]
            ].copy(),
            filled_missing_features=tuple(filled_missing),
            categorical_encodings=encodings,
        )

    def _select_timestamp_column(
        self, frame: pd.DataFrame, spec: NotebookModelSpec
    ) -> str:
        if spec.timestamp_column and spec.timestamp_column in frame.columns:
            return spec.timestamp_column

        for candidate in ("event_timestamp", "date", "as_of_date", "timestamp"):
            if candidate in frame.columns:
                return candidate

        raise PipelineError(
            f"No timestamp column found for model '{spec.model_id}'. "
            "Set 'timestamp_column' in training spec."
        )

    def _build_entity_column(self, frame: pd.DataFrame, spec: NotebookModelSpec) -> pd.Series:
        if spec.entity_column and spec.entity_column in frame.columns:
            return frame[spec.entity_column].astype(str).str.strip()

        if spec.entity_columns:
            missing = sorted(set(spec.entity_columns) - set(frame.columns))
            if missing:
                raise PipelineError(
                    f"Entity columns {missing} are missing for model '{spec.model_id}'."
                )
            values = (
                frame[list(spec.entity_columns)]
                .astype(str)
                .apply(lambda row: "__".join(item.strip() for item in row), axis=1)
            )
            return values

        if "inventory_id" in frame.columns:
            return frame["inventory_id"].astype(str).str.strip()
        if {"hospital", "blood_type"}.issubset(frame.columns):
            return (
                frame["hospital"].astype(str).str.strip()
                + "__"
                + frame["blood_type"].astype(str).str.strip()
            )
        if "donor_id" in frame.columns:
            return frame["donor_id"].astype(str).str.strip()

        raise PipelineError(
            f"Unable to infer entity id for model '{spec.model_id}'. "
            "Set 'entity_column' or 'entity_columns' in training spec."
        )

    def _apply_feature_engineering(
        self, frame: pd.DataFrame, spec: NotebookModelSpec
    ) -> pd.DataFrame:
        engineered = frame.copy()
        features = set(spec.features)

        # Les modèles de demande issus de notebooks nécessitent souvent des variables retardées/glissantes dérivées de units_used.
        lag_features = {"lag1", "lag7", "lag30", "roll7", "roll30"}
        if lag_features & features and "units_used" in engineered.columns:
            group_cols = [
                item for item in spec.group_by_columns if item in engineered.columns
            ]
            if not group_cols and {"hospital", "blood_type"}.issubset(engineered.columns):
                group_cols = ["hospital", "blood_type"]

            if group_cols:
                ordered = engineered.sort_values(
                    group_cols + [self.EVENT_TIMESTAMP_COLUMN]
                ).copy()
                grouped = ordered.groupby(group_cols, dropna=False)
            else:
                ordered = engineered.sort_values(self.EVENT_TIMESTAMP_COLUMN).copy()
                grouped = ordered.groupby(lambda _: 0)

            if "lag1" in features:
                ordered["lag1"] = grouped["units_used"].shift(1)
            if "lag7" in features:
                ordered["lag7"] = grouped["units_used"].shift(7)
            if "lag30" in features:
                ordered["lag30"] = grouped["units_used"].shift(30)
            if "roll7" in features:
                ordered["roll7"] = grouped["units_used"].transform(
                    lambda series: series.shift(1).rolling(7, min_periods=1).mean()
                )
            if "roll30" in features:
                ordered["roll30"] = grouped["units_used"].transform(
                    lambda series: series.shift(1).rolling(30, min_periods=1).mean()
                )

            engineered = ordered

        # Pour les exports de notebooks qui incluent des noms de features one-hot.
        for prefix, source_column in (
            ("hospital_", "hospital"),
            ("blood_type_", "blood_type"),
        ):
            requested = [name for name in features if name.startswith(prefix)]
            if not requested or source_column not in engineered.columns:
                continue
            for feature_name in requested:
                category_value = feature_name[len(prefix) :]
                engineered[feature_name] = (
                    engineered[source_column].astype(str) == category_value
                ).astype(int)

        if "dow" in features and "dow" not in engineered.columns:
            engineered["dow"] = engineered[self.EVENT_TIMESTAMP_COLUMN].dt.dayofweek
        if "weekend" in features and "weekend" not in engineered.columns:
            engineered["weekend"] = (
                engineered[self.EVENT_TIMESTAMP_COLUMN].dt.dayofweek >= 5
            ).astype(int)
        if "month" in features and "month" not in engineered.columns:
            engineered["month"] = engineered[self.EVENT_TIMESTAMP_COLUMN].dt.month

        return engineered

    def _coerce_feature_columns(
        self, frame: pd.DataFrame, feature_columns: Iterable[str]
    ) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
        coerced = frame.copy()
        encodings: dict[str, dict[str, int]] = {}

        for feature in feature_columns:
            series = coerced[feature]

            if pd.api.types.is_bool_dtype(series):
                coerced[feature] = series.astype(int)
                continue

            if pd.api.types.is_numeric_dtype(series):
                coerced[feature] = pd.to_numeric(series, errors="coerce")
                continue

            # Conserver un encodage déterministe pour les features catégorielles de notebook.
            values = series.fillna("__missing__").astype(str)
            unique_values = sorted(values.unique().tolist())
            mapping = {item: index for index, item in enumerate(unique_values)}
            encodings[feature] = mapping
            coerced[feature] = values.map(mapping).astype(float)

        for feature in feature_columns:
            if coerced[feature].isna().all():
                coerced[feature] = 0.0
            else:
                median = float(coerced[feature].median())
                coerced[feature] = coerced[feature].fillna(median)

        return coerced, encodings

    def _coerce_target(self, series: pd.Series, task: str, model_id: str) -> pd.Series:
        if task == "regression":
            result = pd.to_numeric(series, errors="coerce")
            if result.isna().all():
                raise PipelineError(
                    f"Target column is not numeric for regression model '{model_id}'."
                )
            return result

        if pd.api.types.is_numeric_dtype(series):
            result = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
        else:
            values = series.fillna("__missing__").astype(str)
            unique_values = sorted(values.unique().tolist())
            mapping = {item: index for index, item in enumerate(unique_values)}
            result = values.map(mapping).astype(int)

        if result.nunique() < 2:
            raise PipelineError(
                f"Classification target in model '{model_id}' must have at least two classes."
            )
        return result

    def _resolve_feature_types(
        self, frame: pd.DataFrame, feature_columns: Iterable[str]
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for feature in feature_columns:
            column = pd.to_numeric(frame[feature], errors="coerce")
            values = column.dropna().to_numpy()
            if values.size == 0:
                result[feature] = "Float32"
                continue

            if np.allclose(values, np.round(values), equal_nan=True):
                result[feature] = "Int64"
            else:
                result[feature] = "Float32"
        return result

    def _write_generated_feature_definitions(
        self, prepared_models: Sequence[PreparedModelData]
    ) -> None:
        if not prepared_models:
            raise PipelineError("No prepared model data available for Feast generation.")

        lines: list[str] = []
        lines.append("from datetime import timedelta")
        lines.append("")
        lines.append("from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType")
        lines.append("from feast.types import Float32, Int64")
        lines.append("")

        for prepared in prepared_models:
            slug = _slugify(prepared.spec.model_id)
            entity_var = f"{slug}_entity"
            source_var = f"{slug}_source"
            view_var = f"{slug}_view"
            service_var = f"{slug}_service"

            lines.append(
                f"{entity_var} = Entity("
                f"name=\"{slug}_entity\", "
                f"join_keys=[\"{prepared.entity_column}\"], "
                "value_type=ValueType.STRING)"
            )
            lines.append("")
            lines.append(
                f"{source_var} = FileSource("
                f"name=\"{slug}_source\", "
                f"path=\"{prepared.feature_parquet_relative_path}\", "
                f"timestamp_field=\"{prepared.event_timestamp_column}\")"
            )
            lines.append("")
            lines.append(f"{view_var} = FeatureView(")
            lines.append(f"    name=\"{prepared.spec.feature_view_name}\",")
            lines.append(f"    entities=[{entity_var}],")
            lines.append("    ttl=timedelta(days=365),")
            lines.append("    schema=[")
            for feature in prepared.feature_columns:
                dtype = prepared.feature_types.get(feature, "Float32")
                lines.append(f"        Field(name=\"{feature}\", dtype={dtype}),")
            lines.append("    ],")
            lines.append("    online=True,")
            lines.append(f"    source={source_var},")
            lines.append(
                f"    tags={{\"owner\": \"notebook_pipeline\", \"model_id\": \"{prepared.spec.model_id}\"}},"
            )
            lines.append(")")
            lines.append("")
            lines.append(
                f"{service_var} = FeatureService("
                f"name=\"{prepared.spec.feature_service_name}\", "
                f"features=[{view_var}])"
            )
            lines.append("")

        self.settings.generated_feature_definitions.parent.mkdir(
            parents=True, exist_ok=True
        )
        self.settings.generated_feature_definitions.write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _train_and_log_model(
        self, prepared: PreparedModelData, register_models: bool
    ) -> dict[str, Any]:
        feast_module = _require_dependency("feast", "pip install feast")
        mlflow = _require_dependency("mlflow", "pip install mlflow")
        _require_dependency("mlflow.spark", "pip install mlflow pyspark")

        store = feast_module.FeatureStore(repo_path=str(self.settings.feature_repo))
        feature_refs = [
            _feature_ref(prepared.spec.feature_view_name, item)
            for item in prepared.feature_columns
        ]

        training_pdf = store.get_historical_features(
            entity_df=prepared.prepared_entity_target_df,
            features=feature_refs,
        ).to_df()
        if training_pdf.empty:
            raise PipelineError(
                f"Feast historical dataset is empty for model '{prepared.spec.model_id}'."
            )

        spark = self._get_spark()
        training_df = spark.createDataFrame(training_pdf)
        if prepared.target_column not in training_df.columns:
            targets = spark.createDataFrame(prepared.prepared_entity_target_df)
            training_df = training_df.join(
                targets,
                on=[prepared.entity_column, prepared.event_timestamp_column],
                how="left",
            )

        for feature in prepared.feature_columns:
            for candidate in (
                feature,
                f"{prepared.spec.feature_view_name}:{feature}",
                f"{prepared.spec.feature_view_name}__{feature}",
            ):
                if candidate in training_df.columns:
                    if candidate != feature:
                        training_df = training_df.withColumnRenamed(candidate, feature)
                    break
            else:
                raise PipelineError(
                    f"Feast output for model '{prepared.spec.model_id}' "
                    f"is missing feature '{feature}'. Columns: {list(training_df.columns)}"
                )

        training_df = training_df.select(
            *[
                F.col(feature).cast("double").alias(feature)
                for feature in prepared.feature_columns
            ],
            F.col(prepared.target_column).cast("double").alias("label"),
        )
        training_df = training_df.dropna(subset=[*prepared.feature_columns, "label"])
        if training_df.limit(1).count() == 0:
            raise PipelineError(
                f"Feast historical dataset is empty for model '{prepared.spec.model_id}'."
            )

        class_count = int(training_df.select(F.countDistinct("label")).first()[0])
        if prepared.spec.task == "classification" and class_count < 2:
            raise PipelineError(
                f"Classification target in model '{prepared.spec.model_id}' must have at least two classes."
            )
        normalized_algorithm = _slugify(prepared.spec.algorithm)
        if (
            prepared.spec.task == "classification"
            and normalized_algorithm
            in {"xgboost", "xgb", "xgboost_classifier", "xgb_classifier"}
            and class_count != 2
        ):
            raise PipelineError(
                "Spark GBTClassifier fallback for xgboost currently supports binary labels only."
            )

        weighted_df, weight_col = self._apply_class_weights(
            training_df, task=prepared.spec.task
        )
        assembler = VectorAssembler(
            inputCols=list(prepared.feature_columns), outputCol="features"
        )
        model_input = assembler.transform(weighted_df)

        train_df, test_df = self._split_data(frame=model_input, spec=prepared.spec)
        if train_df.limit(1).count() == 0 or test_df.limit(1).count() == 0:
            raise PipelineError(
                f"Train/test split produced an empty partition for model '{prepared.spec.model_id}'."
            )

        estimator = self._build_model(spec=prepared.spec, weight_col=weight_col)
        model = estimator.fit(train_df)
        predictions = model.transform(test_df)

        metrics = self._evaluate_model(
            predictions=predictions,
            task=prepared.spec.task,
        )

        self._configure_mlflow_experiment(mlflow)
        run_name = f"notebook-{prepared.spec.model_id}-{_slugify(prepared.spec.algorithm)}"

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("model_id", prepared.spec.model_id)
            mlflow.log_param("algorithm", prepared.spec.algorithm)
            mlflow.log_param("task", prepared.spec.task)
            mlflow.log_param("target", prepared.target_column)
            mlflow.log_param("feature_count", len(prepared.feature_columns))
            mlflow.log_param("feature_view", prepared.spec.feature_view_name)
            mlflow.log_param("feature_service", prepared.spec.feature_service_name)
            mlflow.log_param("dataset_csv", str(prepared.spec.dataset_csv))
            if prepared.spec.source_model_path:
                mlflow.log_param(
                    "source_notebook_model_path", prepared.spec.source_model_path
                )
            if prepared.filled_missing_features:
                mlflow.log_param(
                    "filled_missing_features",
                    ",".join(prepared.filled_missing_features),
                )

            mlflow.log_metrics(metrics)
            mlflow.log_dict(
                {
                    "features": list(prepared.feature_columns),
                    "feature_types": dict(prepared.feature_types),
                    "categorical_encodings": prepared.categorical_encodings,
                    "model_output_path": str(prepared.spec.model_output_path),
                },
                artifact_file="feature_spec.json",
            )

            self._log_model_artifact(mlflow=mlflow, model=model)
            run_id = run.info.run_id

            should_register = register_models or prepared.spec.register_model
            if should_register:
                self._register_mlflow_model(
                    mlflow=mlflow,
                    run_id=run_id,
                    registry_name=prepared.spec.model_registry_name,
                )

        saved_model_path = self._save_spark_model(
            model=model, output_path=prepared.spec.model_output_path
        )

        return {
            "model_id": prepared.spec.model_id,
            "run_id": run_id,
            "algorithm": prepared.spec.algorithm,
            "task": prepared.spec.task,
            "metrics": metrics,
            "feature_count": len(prepared.feature_columns),
            "feature_view": prepared.spec.feature_view_name,
            "feature_service": prepared.spec.feature_service_name,
            "dataset_csv": str(prepared.spec.dataset_csv),
            "saved_model_path": str(saved_model_path),
            "filled_missing_features": list(prepared.filled_missing_features),
            "registered_model": prepared.spec.model_registry_name
            if register_models or prepared.spec.register_model
            else None,
        }

    def _split_data(
        self, frame: SparkDataFrame, spec: NotebookModelSpec
    ) -> tuple[SparkDataFrame, SparkDataFrame]:
        test_ratio = min(max(spec.test_size, 0.05), 0.5)
        train_ratio = 1.0 - test_ratio
        return frame.randomSplit([train_ratio, test_ratio], seed=spec.random_state)

    def _build_model(self, spec: NotebookModelSpec, weight_col: str | None) -> Any:
        name = _slugify(spec.algorithm)

        if spec.task == "classification":
            if name in {"logistic_regression", "logreg"}:
                kwargs: dict[str, Any] = {
                    "featuresCol": "features",
                    "labelCol": "label",
                    "maxIter": 200,
                }
                if weight_col:
                    kwargs["weightCol"] = weight_col
                return SparkLogisticRegression(**kwargs)
            if name in {"random_forest", "random_forest_classifier", "rf_classifier"}:
                kwargs = {
                    "featuresCol": "features",
                    "labelCol": "label",
                    "numTrees": 400,
                    "seed": spec.random_state,
                }
                if weight_col:
                    kwargs["weightCol"] = weight_col
                return SparkRandomForestClassifier(**kwargs)
            if name in {"xgboost", "xgb", "xgboost_classifier", "xgb_classifier"}:
                return GBTClassifier(
                    featuresCol="features",
                    labelCol="label",
                    maxIter=200,
                    maxDepth=6,
                    stepSize=0.05,
                    seed=spec.random_state,
                )
            raise PipelineError(
                f"Unsupported classification algorithm '{spec.algorithm}' "
                f"for model '{spec.model_id}'."
            )

        if name in {"linear_regression", "linreg"}:
            return SparkLinearRegression(
                featuresCol="features", labelCol="label", maxIter=200
            )
        if name in {"random_forest", "random_forest_regressor", "rf_regressor"}:
            return SparkRandomForestRegressor(
                featuresCol="features",
                labelCol="label",
                numTrees=400,
                seed=spec.random_state,
            )
        if name in {"xgboost", "xgb", "xgboost_regressor", "xgb_regressor"}:
            return GBTRegressor(
                featuresCol="features",
                labelCol="label",
                maxIter=200,
                maxDepth=6,
                stepSize=0.05,
                seed=spec.random_state,
            )

        raise PipelineError(
            f"Unsupported regression algorithm '{spec.algorithm}' for model '{spec.model_id}'."
        )

    def _evaluate_model(
        self, predictions: SparkDataFrame, task: str
    ) -> dict[str, float]:
        if task == "classification":
            metrics: dict[str, float] = {
                "accuracy": float(
                    MulticlassClassificationEvaluator(
                        labelCol="label",
                        predictionCol="prediction",
                        metricName="accuracy",
                    ).evaluate(predictions)
                ),
                "precision": float(
                    MulticlassClassificationEvaluator(
                        labelCol="label",
                        predictionCol="prediction",
                        metricName="weightedPrecision",
                    ).evaluate(predictions)
                ),
                "recall": float(
                    MulticlassClassificationEvaluator(
                        labelCol="label",
                        predictionCol="prediction",
                        metricName="weightedRecall",
                    ).evaluate(predictions)
                ),
                "f1": float(
                    MulticlassClassificationEvaluator(
                        labelCol="label",
                        predictionCol="prediction",
                        metricName="f1",
                    ).evaluate(predictions)
                ),
            }

            class_count = int(predictions.select(F.countDistinct("label")).first()[0])
            if class_count == 2:
                metrics["roc_auc"] = float(
                    BinaryClassificationEvaluator(
                        labelCol="label",
                        rawPredictionCol="rawPrediction",
                        metricName="areaUnderROC",
                    ).evaluate(predictions)
                )
            return metrics

        return {
            "rmse": float(
                RegressionEvaluator(
                    labelCol="label",
                    predictionCol="prediction",
                    metricName="rmse",
                ).evaluate(predictions)
            ),
            "mae": float(
                RegressionEvaluator(
                    labelCol="label",
                    predictionCol="prediction",
                    metricName="mae",
                ).evaluate(predictions)
            ),
            "r2": float(
                RegressionEvaluator(
                    labelCol="label",
                    predictionCol="prediction",
                    metricName="r2",
                ).evaluate(predictions)
            ),
        }

    def _log_model_artifact(self, mlflow: Any, model: Any) -> None:
        mlflow.spark.log_model(model, artifact_path="model")

    def _save_spark_model(self, model: Any, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        model.write().overwrite().save(str(output_path))
        return output_path

    def _configure_mlflow_experiment(self, mlflow: Any) -> None:
        mlflow.set_tracking_uri(self.settings.tracking_uri)
        try:
            mlflow.set_experiment(self.settings.experiment_name)
        except Exception as exc:
            message = str(exc)
            if "Can't locate revision identified by" in message:
                raise PipelineError(
                    "MLflow tracking database schema is incompatible with the installed MLflow version. "
                    "Use a fresh SQLite tracking URI, for example: "
                    "export MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow_local.db"
                ) from exc
            raise

    def _register_mlflow_model(self, mlflow: Any, run_id: str, registry_name: str) -> None:
        model_uri = f"runs:/{run_id}/model"
        try:
            mlflow.register_model(model_uri=model_uri, name=registry_name)
        except Exception:
            # L'enregistrement est optionnel et ne doit pas faire échouer l'exécution du pipeline.
            return

    def _run_feast_cli(self, args: Sequence[str]) -> None:
        if not self.settings.feature_repo.exists():
            raise PipelineError(f"Feature repo not found: {self.settings.feature_repo}")

        command = ["feast", *args]
        try:
            subprocess.run(command, cwd=str(self.settings.feature_repo), check=True)
        except FileNotFoundError as exc:
            raise DependencyError(
                "Feast CLI is not available. Install it with: pip install feast"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise PipelineError(
                f"Command failed in feature repo ({self.settings.feature_repo}): "
                f"{' '.join(command)}"
            ) from exc

    def _write_run_summary(self, payload: Mapping[str, Any]) -> None:
        self.settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.metadata_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _resolve_path(self, raw_path: Any) -> Path | None:
        if raw_path is None:
            return None
        value = str(raw_path).strip()
        if not value:
            return None

        candidate = Path(value)
        if candidate.exists():
            return candidate.resolve()

        backend_relative = (self.settings.backend_root / value).resolve()
        if backend_relative.exists():
            return backend_relative

        # Les configurations de notebooks peuvent contenir des chemins absolus obsolètes. Repli sur le nom de base.
        basename = candidate.name
        search_roots = [
            self.settings.backend_root / "models",
            self.settings.backend_root / "notebooks" / "models",
            self.settings.backend_root / "datasets",
        ]
        for root in search_roots:
            if not root.exists():
                continue
            matches = list(root.rglob(basename))
            if matches:
                return matches[0].resolve()

        return candidate.resolve()

    def _resolve_model_output_path(self, raw_path: Any, model_id: str) -> Path:
        if raw_path:
            resolved = self._resolve_path(raw_path)
            if resolved is not None:
                return resolved
        return (
            self.settings.backend_root
            / "models"
            / f"{_slugify(model_id)}_feast_mlflow.sparkml"
        ).resolve()

    def _get_spark(self) -> SparkSession:
        if self._spark_session is None:
            java_major = self._ensure_supported_java_runtime()
            builder = SparkSession.builder.appName("pios-notebook-models-mlflow")
            if java_major == 23:
                # JDK 23 nécessite d'autoriser les API de security manager dépréciées utilisées par Hadoop UGI.
                builder = builder.config(
                    "spark.driver.extraJavaOptions", "-Djava.security.manager=allow"
                ).config(
                    "spark.executor.extraJavaOptions", "-Djava.security.manager=allow"
                )
            self._spark_session = builder.getOrCreate()
            self._spark_session.conf.set("spark.sql.session.timeZone", "UTC")
        return self._spark_session

    def _apply_class_weights(
        self, frame: SparkDataFrame, task: str
    ) -> tuple[SparkDataFrame, str | None]:
        if task != "classification":
            return frame, None

        counts = frame.groupBy("label").count().collect()
        if len(counts) != 2:
            return frame, None

        by_label = {float(row["label"]): int(row["count"]) for row in counts}
        if any(count <= 0 for count in by_label.values()):
            return frame, None

        total = float(sum(by_label.values()))
        labels = sorted(by_label.keys())
        weights = {
            label: total / (2.0 * float(by_label[label]))
            for label in labels
        }
        weighted = frame.withColumn(
            "class_weight",
            F.when(F.col("label") == F.lit(labels[0]), F.lit(weights[labels[0]])).otherwise(
                F.lit(weights[labels[1]])
            ),
        )
        return weighted, "class_weight"

    def _ensure_supported_java_runtime(self) -> int | None:
        try:
            output = subprocess.run(
                ["java", "-version"],
                check=False,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, OSError):
            return None

        merged = "\n".join(
            part for part in [output.stderr.strip(), output.stdout.strip()] if part
        )
        major = self._parse_java_major(merged)
        if major is not None and major >= 24:
            raise DependencyError(
                f"Unsupported Java runtime detected (Java {major}). "
                "PySpark/Hadoop in this pipeline requires Java 17 or 21. "
                "Set JAVA_HOME to a JDK 17/21 installation and retry."
            )
        return major

    def _parse_java_major(self, version_output: str) -> int | None:
        for line in version_output.splitlines():
            if "version" not in line:
                continue
            token: str | None = None
            if '"' in line:
                parts = line.split('"')
                if len(parts) >= 2:
                    token = parts[1]
            if not token:
                words = line.strip().split()
                if words:
                    token = words[-1]
            if not token:
                continue

            chunks = token.split(".")
            if not chunks:
                continue
            if chunks[0] == "1" and len(chunks) > 1:
                candidate = chunks[1]
            else:
                candidate = chunks[0]
            if candidate.isdigit():
                return int(candidate)
        return None
