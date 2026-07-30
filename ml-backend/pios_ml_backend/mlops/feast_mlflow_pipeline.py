from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from pyspark.sql import functions as F


class PipelineError(RuntimeError):
    pass


class DependencyError(RuntimeError):
    pass


def _require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name, fromlist=["*"])
    except ModuleNotFoundError as exc:
        raise DependencyError(
            f"Missing optional dependency '{module_name}'. Install it with: {install_hint}"
        ) from exc


@dataclass(frozen=True)
class FeastMLflowSettings:
    backend_root: Path
    dataset_csv: Path
    feature_repo: Path
    feature_parquet: Path
    metadata_path: Path
    tracking_uri: str
    experiment_name: str

    @classmethod
    def from_env(cls) -> "FeastMLflowSettings":
        backend_root = Path(__file__).resolve().parents[2]
        feature_repo = Path(
            os.getenv("PIOS_FEAST_REPO", str(backend_root / "feature_repo"))
        ).resolve()
        feature_repo_data = feature_repo / "data"

        return cls(
            backend_root=backend_root,
            dataset_csv=Path(
                os.getenv(
                    "PIOS_FEATURE_SOURCE_CSV",
                    str(backend_root / "datasets" / "synthetic_bloodbank_daily.csv"),
                )
            ).resolve(),
            feature_repo=feature_repo,
            feature_parquet=Path(
                os.getenv(
                    "PIOS_FEAST_FEATURE_PARQUET",
                    str(feature_repo_data / "bloodbank_features.parquet"),
                )
            ).resolve(),
            metadata_path=Path(
                os.getenv(
                    "PIOS_MLFLOW_LAST_RUN_FILE",
                    str(backend_root / "artifacts" / "mlflow_last_run.json"),
                )
            ).resolve(),
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI", f"sqlite:///{backend_root / 'mlflow.db'}"
            ),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT", "pios-bloodbank-demand"),
        )


class FeastMLflowPipeline:
    ENTITY_COLUMN = "inventory_id"
    EVENT_TIMESTAMP_COLUMN = "event_timestamp"
    TARGET_COLUMN = "heavy_demand_next_day"
    FEATURE_VIEW_NAME = "bloodbank_daily_features"
    FEATURE_SERVICE_NAME = "bloodbank_demand_v1"
    FEATURE_COLUMNS = (
        "dow",
        "weekend",
        "month",
        "holiday",
        "temp_c",
        "rain_mm",
        "flu_index",
        "trauma_cases",
        "scheduled_surgeries",
        "donation_campaign",
        "supply_shock",
        "stock_start",
    )

    def __init__(self, settings: FeastMLflowSettings | None = None):
        self.settings = settings or FeastMLflowSettings.from_env()
        self._spark_session: SparkSession | None = None

    @classmethod
    def from_env(cls) -> "FeastMLflowPipeline":
        return cls(settings=FeastMLflowSettings.from_env())

    @property
    def feature_references(self) -> list[str]:
        return [f"{self.FEATURE_VIEW_NAME}:{name}" for name in self.FEATURE_COLUMNS]

    def prepare_feature_data(self) -> SparkDataFrame:
        if not self.settings.dataset_csv.exists():
            raise PipelineError(f"Dataset not found: {self.settings.dataset_csv}")

        spark = self._get_spark()
        raw_df = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .csv(str(self.settings.dataset_csv))
        )
        required = {
            "date",
            "hospital",
            "blood_type",
            self.TARGET_COLUMN,
            *self.FEATURE_COLUMNS,
        }
        missing = sorted(required - set(raw_df.columns))
        if missing:
            raise PipelineError(
                f"Dataset {self.settings.dataset_csv} is missing columns: {missing}"
            )

        prepared = (
            raw_df.withColumn(
                self.EVENT_TIMESTAMP_COLUMN, F.to_timestamp(F.col("date"))
            )
            .withColumn(
                self.ENTITY_COLUMN,
                F.concat_ws(
                    "__",
                    F.trim(F.col("hospital").cast("string")),
                    F.trim(F.col("blood_type").cast("string")),
                ),
            )
            .withColumn(self.TARGET_COLUMN, F.col(self.TARGET_COLUMN).cast("int"))
        )

        bad_timestamps = prepared.filter(F.col(self.EVENT_TIMESTAMP_COLUMN).isNull()).count()
        if bad_timestamps:
            raise PipelineError(
                "Failed to parse one or more values in 'date' into event_timestamp."
            )

        for feature in self.FEATURE_COLUMNS:
            prepared = prepared.withColumn(feature, F.col(feature).cast("double"))

        final_columns = [
            self.ENTITY_COLUMN,
            self.EVENT_TIMESTAMP_COLUMN,
            *self.FEATURE_COLUMNS,
            self.TARGET_COLUMN,
        ]
        prepared = prepared.select(*final_columns)

        self.settings.feature_parquet.parent.mkdir(parents=True, exist_ok=True)
        if self.settings.feature_parquet.exists() and self.settings.feature_parquet.is_file():
            self.settings.feature_parquet.unlink()
        prepared.write.mode("overwrite").parquet(str(self.settings.feature_parquet))
        return prepared

    def apply_feast_repo(self) -> None:
        self._run_feast_cli(["apply"])

    def materialize_incremental(self) -> None:
        feast_module = _require_dependency("feast", "pip install feast")
        store = feast_module.FeatureStore(repo_path=str(self.settings.feature_repo))
        store.materialize_incremental(end_date=datetime.now(timezone.utc))

    def build_training_frame(self) -> SparkDataFrame:
        feast_module = _require_dependency("feast", "pip install feast")

        prepared = self.prepare_feature_data()
        store = feast_module.FeatureStore(repo_path=str(self.settings.feature_repo))
        entity_df = (
            prepared.select(
                self.ENTITY_COLUMN, self.EVENT_TIMESTAMP_COLUMN, self.TARGET_COLUMN
            ).toPandas()
        )

        spark = self._get_spark()
        training_pdf = store.get_historical_features(
            entity_df=entity_df,
            features=self.feature_references,
        ).to_df()
        if training_pdf.empty:
            raise PipelineError("Training dataset is empty after Feast feature retrieval.")
        training_df = spark.createDataFrame(training_pdf)

        if self.TARGET_COLUMN not in training_df.columns:
            target_df = prepared.select(
                self.ENTITY_COLUMN, self.EVENT_TIMESTAMP_COLUMN, self.TARGET_COLUMN
            )
            training_df = training_df.join(
                target_df,
                on=[self.ENTITY_COLUMN, self.EVENT_TIMESTAMP_COLUMN],
                how="left",
            )

        for feature in self.FEATURE_COLUMNS:
            candidates = (
                feature,
                f"{self.FEATURE_VIEW_NAME}__{feature}",
                f"{self.FEATURE_VIEW_NAME}:{feature}",
            )
            for candidate in candidates:
                if candidate in training_df.columns:
                    if candidate != feature:
                        training_df = training_df.withColumnRenamed(candidate, feature)
                    break
            else:
                raise PipelineError(
                    "Unable to find feature column "
                    f"'{feature}' in Feast output columns: {list(training_df.columns)}"
                )

        training_df = training_df.dropna(
            subset=[*self.FEATURE_COLUMNS, self.TARGET_COLUMN]
        )
        training_df = training_df.select(
            *[
                F.col(column).cast("double").alias(column)
                for column in self.FEATURE_COLUMNS
            ],
            F.col(self.TARGET_COLUMN).cast("double").alias("label"),
        )
        return training_df

    def train_and_log_model(self) -> dict[str, Any]:
        mlflow = _require_dependency("mlflow", "pip install mlflow")
        _require_dependency("mlflow.spark", "pip install mlflow pyspark")

        training_df = self.build_training_frame()
        if training_df.limit(1).count() == 0:
            raise PipelineError("Training dataset is empty after Feast feature retrieval.")

        class_count = int(training_df.select(F.countDistinct("label")).first()[0])
        if class_count < 2:
            raise PipelineError(
                f"Target '{self.TARGET_COLUMN}' must contain at least two classes."
            )

        weighted_df, weight_col = self._apply_class_weights(training_df)

        assembler = VectorAssembler(
            inputCols=list(self.FEATURE_COLUMNS), outputCol="features"
        )
        model_input = assembler.transform(weighted_df)

        train_df, test_df = model_input.randomSplit([0.8, 0.2], seed=42)
        if train_df.limit(1).count() == 0 or test_df.limit(1).count() == 0:
            raise PipelineError("Train/test split produced an empty partition.")

        estimator = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=200,
            weightCol=weight_col,
        )
        model = estimator.fit(train_df)
        pred = model.transform(test_df)

        metrics: dict[str, float] = {
            "accuracy": float(
                MulticlassClassificationEvaluator(
                    labelCol="label", predictionCol="prediction", metricName="accuracy"
                ).evaluate(pred)
            ),
            "precision": float(
                MulticlassClassificationEvaluator(
                    labelCol="label",
                    predictionCol="prediction",
                    metricName="weightedPrecision",
                ).evaluate(pred)
            ),
            "recall": float(
                MulticlassClassificationEvaluator(
                    labelCol="label",
                    predictionCol="prediction",
                    metricName="weightedRecall",
                ).evaluate(pred)
            ),
            "f1": float(
                MulticlassClassificationEvaluator(
                    labelCol="label", predictionCol="prediction", metricName="f1"
                ).evaluate(pred)
            ),
        }

        if class_count == 2:
            metrics["roc_auc"] = float(
                BinaryClassificationEvaluator(
                    labelCol="label",
                    rawPredictionCol="rawPrediction",
                    metricName="areaUnderROC",
                ).evaluate(pred)
            )

        self._configure_mlflow_experiment(mlflow)

        with mlflow.start_run(run_name="feast-mlflow-logreg") as run:
            mlflow.log_param("model_type", "SparkLogisticRegression")
            mlflow.log_param("target", self.TARGET_COLUMN)
            mlflow.log_param("feature_view", self.FEATURE_VIEW_NAME)
            mlflow.log_param("feature_service", self.FEATURE_SERVICE_NAME)
            mlflow.log_param("feature_count", len(self.FEATURE_COLUMNS))
            mlflow.log_metrics(metrics)

            mlflow.log_text(
                json.dumps(
                    {
                        "features": list(self.FEATURE_COLUMNS),
                        "feature_repo": str(self.settings.feature_repo),
                        "dataset": str(self.settings.dataset_csv),
                    },
                    indent=2,
                ),
                artifact_file="feast_feature_spec.json",
            )
            mlflow.spark.log_model(model, artifact_path="model")

            run_id = run.info.run_id
            self._write_last_run(run_id=run_id, metrics=metrics)

        return {"run_id": run_id, "metrics": metrics}

    def sync_and_train(self, materialize_online: bool = True) -> dict[str, Any]:
        self.prepare_feature_data()
        self.apply_feast_repo()
        if materialize_online:
            self.materialize_incremental()
        return self.train_and_log_model()

    def score_online(
        self, inventory_ids: Sequence[str], run_id: str | None = None
    ) -> list[dict[str, Any]]:
        if not inventory_ids:
            raise PipelineError("Provide at least one inventory id to score.")

        feast_module = _require_dependency("feast", "pip install feast")
        mlflow = _require_dependency("mlflow", "pip install mlflow")
        _require_dependency("mlflow.spark", "pip install mlflow pyspark")

        selected_run = run_id or self.get_latest_run_id()
        mlflow.set_tracking_uri(self.settings.tracking_uri)

        model_uri = f"runs:/{selected_run}/model"
        model = mlflow.spark.load_model(model_uri=model_uri)
        store = feast_module.FeatureStore(repo_path=str(self.settings.feature_repo))

        online_result = store.get_online_features(
            features=self.feature_references,
            entity_rows=[{self.ENTITY_COLUMN: value} for value in inventory_ids],
        ).to_dict()

        rows: list[dict[str, Any]] = []
        for index, _ in enumerate(inventory_ids):
            row: dict[str, Any] = {"_index": index, "inventory_id": inventory_ids[index]}
            for feature in self.FEATURE_COLUMNS:
                value = self._extract_online_feature(online_result, feature, index)
                row[feature] = 0.0 if value is None else float(value)
            rows.append(row)

        spark = self._get_spark()
        features_df = spark.createDataFrame(rows)
        for feature in self.FEATURE_COLUMNS:
            features_df = features_df.withColumn(
                feature, F.coalesce(F.col(feature).cast("double"), F.lit(0.0))
            )

        assembler = VectorAssembler(
            inputCols=list(self.FEATURE_COLUMNS), outputCol="features"
        )
        scored = model.transform(assembler.transform(features_df)).withColumn(
            "probability_score",
            F.when(
                F.size(vector_to_array("probability")) > 1,
                vector_to_array("probability")[1],
            ).otherwise(F.col("prediction").cast("double")),
        )
        records = scored.select(
            "_index",
            "inventory_id",
            F.col("prediction").cast("int").alias("prediction"),
            F.col("probability_score").cast("double").alias("probability"),
        ).orderBy("_index")

        output: list[dict[str, Any]] = []
        for row in records.collect():
            output.append(
                {
                    "inventory_id": row["inventory_id"],
                    "prediction": int(row["prediction"]),
                    "probability": float(row["probability"]),
                    "run_id": selected_run,
                }
            )
        return output

    def list_inventory_ids(self, limit: int = 20) -> list[str]:
        prepared = self.prepare_feature_data()
        rows = (
            prepared.select(self.ENTITY_COLUMN)
            .distinct()
            .orderBy(self.ENTITY_COLUMN)
            .limit(limit)
            .collect()
        )
        return [str(row[self.ENTITY_COLUMN]) for row in rows]

    def get_latest_run_id(self) -> str:
        if self.settings.metadata_path.exists():
            payload = json.loads(self.settings.metadata_path.read_text(encoding="utf-8"))
            run_id = payload.get("run_id")
            if run_id:
                return str(run_id)

        mlflow = _require_dependency("mlflow", "pip install mlflow")
        mlflow.set_tracking_uri(self.settings.tracking_uri)
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.settings.experiment_name)
        if experiment is None:
            raise PipelineError(
                f"No MLflow experiment named '{self.settings.experiment_name}' found."
            )

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise PipelineError(
                f"No runs found in experiment '{self.settings.experiment_name}'."
            )
        return runs[0].info.run_id

    def _run_feast_cli(self, args: Sequence[str]) -> None:
        if not self.settings.feature_repo.exists():
            raise PipelineError(f"Feature repo not found: {self.settings.feature_repo}")

        command = ["feast", *args]
        try:
            subprocess.run(
                command,
                cwd=str(self.settings.feature_repo),
                check=True,
            )
        except FileNotFoundError as exc:
            raise DependencyError(
                "Feast CLI is not available. Install it with: pip install feast"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise PipelineError(
                f"Command failed in feature repo ({self.settings.feature_repo}): "
                f"{' '.join(command)}"
            ) from exc

    def _write_last_run(self, run_id: str, metrics: dict[str, float]) -> None:
        self.settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "metrics": metrics,
            "tracking_uri": self.settings.tracking_uri,
            "experiment": self.settings.experiment_name,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.settings.metadata_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _extract_online_feature(
        self, online_result: dict[str, list[Any]], feature: str, index: int
    ) -> Any:
        candidates = (
            feature,
            f"{self.FEATURE_VIEW_NAME}__{feature}",
            f"{self.FEATURE_VIEW_NAME}:{feature}",
        )
        for key in candidates:
            values = online_result.get(key)
            if values is not None and len(values) > index:
                return values[index]
        return None

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

    def _get_spark(self) -> SparkSession:
        if self._spark_session is None:
            java_major = self._ensure_supported_java_runtime()
            builder = SparkSession.builder.appName("pios-feast-mlflow")
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
        self, frame: SparkDataFrame
    ) -> tuple[SparkDataFrame, str | None]:
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
