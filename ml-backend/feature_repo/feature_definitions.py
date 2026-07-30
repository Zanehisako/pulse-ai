import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pv
import pyarrow.parquet as pq
from feast import (
    Entity,
    FeatureService,
    # FeatureStore,
    FeatureView,
    Field,
    FileSource,
    OnDemandFeatureView,
    ValueType,
)
from feast.types import Float32, Int64, String

logger = logging.getLogger(__name__)

dtype_map = {
    "float64": Float32,
    "int64": Int64,
    "object": String,
}

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
DATASET_FOLDER = os.path.join(ROOT_DIRECTORY, "datasets_featues_labels_seperated")
ENTITY_CONFIG_PATH = Path(
    os.getenv(
        "PIOS_FEAST_ENTITY_CONFIG",
        str(Path(SCRIPT_DIRECTORY) / "feature_entities.json"),
    )
)


def _load_entity_config() -> dict:
    try:
        payload = json.loads(ENTITY_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _configured_entities(dataset_stem: str, columns) -> list[str]:
    config = _load_entity_config()
    dataset_entities = config.get("dataset_entities")
    configured = []
    if isinstance(dataset_entities, dict):
        for key in (dataset_stem, "*"):
            values = dataset_entities.get(key)
            if isinstance(values, list):
                configured.extend(str(value).strip() for value in values)
    suffixes = config.get("entity_suffixes")
    if not isinstance(suffixes, list):
        suffixes = []
    inferred = [
        str(column)
        for column in columns
        if any(str(column).lower().endswith(str(suffix).lower()) for suffix in suffixes)
    ]
    entity_columns = [*configured, *inferred]
    seen = set()
    return [
        column
        for column in entity_columns
        if column in columns and not (column.lower() in seen or seen.add(column.lower()))
    ]

datasets_files_csv = list(Path(DATASET_FOLDER).glob("*.csv"))
if len(datasets_files_csv) != 0:
    for dataset_file in datasets_files_csv:
        logger.debug("Preparing feature CSV %s", dataset_file)
        # convert csv to parquet
        if dataset_file.with_suffix(".parquet").exists():
            logger.debug(
                "%s already exists, skipping conversion.",
                dataset_file.with_suffix(".parquet"),
            )
            continue
        table = pv.read_csv(dataset_file)
        pq.write_table(table, dataset_file.with_suffix(".parquet"))
else:
    logger.debug("No CSV feature datasets found.")


datasets_files_parquet = sorted(Path(DATASET_FOLDER).glob("*_features.parquet"))
logger.debug("Feature parquet files: %s", datasets_files_parquet)
feature_views_registry: dict[str, list[FeatureView | OnDemandFeatureView]] = {}
services_registry = {}
entities_registry = {}
for dataset_file in datasets_files_parquet:
    df = pd.read_parquet(dataset_file)
    if "event_timestamp" not in df.columns:
        start = datetime(2024, 1, 1)

        df["event_timestamp"] = [
            start + timedelta(minutes=np.random.randint(0, 60 * 24 * 30))
            for _ in range(len(df))
        ]

    entity_columns = _configured_entities(dataset_file.stem, df.columns)
    if not entity_columns:
        logger.debug("Skipping %s because no entity columns were found.", dataset_file)
        continue
    entity_name: str = entity_columns[0].lower()
    for entity_column in entity_columns:
        entity_column = entity_column.lower()
        if entity_column in entities_registry:
            continue
        entities_registry[entity_column] = Entity(
            name=entity_column, join_keys=[entity_column], value_type=ValueType.STRING
        )
        globals()[entity_column] = entities_registry[entity_column]

    source = FileSource(
        name="source" + dataset_file.stem,
        path=str(dataset_file),
        timestamp_field="event_timestamp",
    )
    schema = []
    for column in df.columns:
        if column.lower() in {entity.lower() for entity in entity_columns}:
            continue
        if column.lower() == "event_timestamp":
            continue
        else:
            schema_type = dtype_map.get(str(df[column].dtype), String)
            schema.append(Field(name=column, dtype=schema_type))

    feature_view = FeatureView(
        name=dataset_file.stem + "_features",
        entities=[entities_registry[entity.lower()] for entity in entity_columns],
        ttl=timedelta(days=365),
        schema=schema,
        # These views are used both for offline training joins and online inference.
        online=True,
        source=source,
        tags={"domain": "bloodbank", "team": "ml"},
    )
    if entity_name in feature_views_registry.keys():
        feature_views_registry[entity_name].append(feature_view)
    else:
        feature_views_registry[entity_name] = [feature_view]
    globals()[feature_view.name] = feature_view

    service_base = dataset_file.stem
    if service_base.endswith("_features"):
        service_base = service_base[: -len("_features")]
    feature_view_service = FeatureService(
        name=service_base + "_service",
        features=[feature_view],
    )
    services_registry[feature_view_service.name] = feature_view_service
    globals()[feature_view_service.name] = feature_view_service

for entity_name, feature_view in feature_views_registry.items():
    logger.debug("Entity %s FeatureView count: %s", entity_name, len(feature_view))

for entity in entities_registry.values():
    feature_views = feature_views_registry.get(entity.name)
    if feature_views is not None:
        service = FeatureService(
            name=entity.name + "_service",
            features=feature_views,
        )
        services_registry[service.name] = service
        globals()[service.name] = service
