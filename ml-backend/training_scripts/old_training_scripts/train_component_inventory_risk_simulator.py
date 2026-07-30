"""
train_component_inventory_risk_simulator.py

Standalone training script for the inventory risk simulator. Does NOT use
train_and_log(). Instead:
1. Resolves child model champion aliases to pinned version URIs
2. Extracts child model input schemas from MLflow signatures
3. Builds InventoryRiskSimulatorPyfunc with pinned URIs and schemas
4. Registers as a pyfunc artifact in MLflow
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

_TRAINING_SCRIPTS = Path(__file__).resolve().parent
_ROBOT_DIRECTORY = _TRAINING_SCRIPTS.parent
if str(_ROBOT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_ROBOT_DIRECTORY))
if str(_TRAINING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TRAINING_SCRIPTS))

import django
from dotenv import load_dotenv

_BACKEND = _ROBOT_DIRECTORY.parent / "backendMulti"
for path in (_BACKEND, _BACKEND.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
load_dotenv(_BACKEND / ".env")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backendMulti.settings")
django.setup()

from pios_ml_backend.inventory.simulation_pyfunc import InventoryRiskSimulatorPyfunc
from pios_ml_backend.training_resource_profiles import (
    apply_profile_to_model_config,
    nested_bool,
)

_CONFIG_PATH = _ROBOT_DIRECTORY / "config" / "component_inventory_risk_simulator.json"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8889")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _load_config() -> dict:
    return apply_profile_to_model_config(
        json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    )


def _local_artifact_roots() -> list[Path]:
    configured = os.getenv("PIOS_MLFLOW_ARTIFACT_ROOTS", "").strip()
    roots = [Path(value).expanduser() for value in configured.split(os.pathsep) if value.strip()]
    roots.append(_ROBOT_DIRECTORY.parent / "mlflow_artifacts")
    return [path.resolve() for path in roots if path.exists()]


def _latest_local_model_dir(model_name: str) -> Path | None:
    candidates: list[Path] = []
    for root in _local_artifact_roots():
        candidates.extend(root.glob(f"*/**/artifacts/{model_name}/MLmodel"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def _resolve_child_uri(client: MlflowClient, model_name: str, alias: str) -> str:
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        return f"models:/{model_name}/{mv.version}"
    except Exception:
        local_dir = _latest_local_model_dir(model_name)
        if local_dir is None:
            raise
        return str(local_dir)


def _extract_schema(client: MlflowClient, model_name: str, alias: str) -> dict[str, Any]:
    mv = None
    run = None
    model_info = None
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        run = client.get_run(mv.run_id)
        model_info = mlflow.models.get_model_info(f"models:/{model_name}/{mv.version}")
    except Exception:
        local_dir = _latest_local_model_dir(model_name)
        if local_dir is not None:
            try:
                model_info = mlflow.models.get_model_info(str(local_dir))
            except Exception:
                model_info = None
    input_cols: list[str] = []
    dtypes: dict[str, str] = {}
    defaults: dict[str, Any] = {}
    output_cols: list[str] = []

    signature = getattr(model_info, "signature", None)
    if signature and signature.inputs and signature.inputs.inputs:
        for col in signature.inputs.inputs:
            if col.name:
                input_cols.append(col.name)
                dtype_str = str(col.type) if hasattr(col, "type") else "float64"
                dtypes[col.name] = dtype_str

    if signature and signature.outputs:
        if hasattr(signature.outputs, "inputs") and signature.outputs.inputs:
            for col in signature.outputs.inputs:
                if col.name:
                    output_cols.append(col.name)

    prediction_defaults_path = None
    if mv is not None:
        for artifact in client.list_artifacts(mv.run_id):
            if artifact.path == "prediction_defaults.json":
                prediction_defaults_path = artifact.path
                break
    if prediction_defaults_path and mv is not None:
        try:
            local_path = client.download_artifacts(mv.run_id, prediction_defaults_path)
            pd_data = json.loads(Path(local_path).read_text())
            if isinstance(pd_data, dict):
                fill_values = pd_data.get("fill_values", pd_data.get("defaults", {}))
                if isinstance(fill_values, dict):
                    defaults.update(fill_values)
        except Exception:
            pass
    elif (local_dir := _latest_local_model_dir(model_name)) is not None:
        local_defaults = local_dir / "prediction_defaults.json"
        if local_defaults.exists():
            try:
                pd_data = json.loads(local_defaults.read_text(encoding="utf-8"))
                if isinstance(pd_data, dict):
                    fill_values = pd_data.get("fill_values", pd_data.get("defaults", {}))
                    if isinstance(fill_values, dict):
                        defaults.update(fill_values)
            except Exception:
                pass

    for col in input_cols:
        dtype_text = str(dtypes.get(col, "")).lower()
        if "string" in dtype_text or "str" in dtype_text:
            defaults[col] = str(defaults.get(col) or "")
        elif col not in defaults:
            defaults[col] = 0.0

    return {
        "input_columns": input_cols,
        "dtypes": dtypes,
        "defaults": defaults,
        "output_columns": output_cols,
    }


def main() -> None:
    config = _load_config()
    sub_models_cfg = config["sub_models"]
    sim_cfg = config["simulation"]
    client = MlflowClient()

    print("Resolving child model champion aliases...")
    demand_model_name = sub_models_cfg["demand"]["registered_model_name"]
    supply_model_name = sub_models_cfg["supply"]["registered_model_name"]
    waste_model_name = sub_models_cfg["waste"]["registered_model_name"]

    try:
        demand_uri = _resolve_child_uri(client, demand_model_name, "champion")
        print(f"  demand  → {demand_uri}")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot resolve champion for {demand_model_name}: {exc}"
        ) from exc

    try:
        supply_uri = _resolve_child_uri(client, supply_model_name, "champion")
        print(f"  supply  → {supply_uri}")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot resolve champion for {supply_model_name}: {exc}"
        ) from exc

    try:
        waste_uri = _resolve_child_uri(client, waste_model_name, "champion")
        print(f"  waste   → {waste_uri}")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot resolve champion for {waste_model_name}: {exc}"
        ) from exc

    print("Extracting child model schemas...")
    demand_schema = _extract_schema(client, demand_model_name, "champion")
    print(f"  demand  input cols: {len(demand_schema['input_columns'])}")
    supply_schema = _extract_schema(client, supply_model_name, "champion")
    print(f"  supply  input cols: {len(supply_schema['input_columns'])}")
    waste_schema = _extract_schema(client, waste_model_name, "champion")
    print(f"  waste   input cols: {len(waste_schema['input_columns'])}")

    horizons = sim_cfg.get("horizons_days", [7, 30, 90, 180])
    seed = sim_cfg.get("seed") if sim_cfg.get("seed") is not None else None

    print(f"Building simulator (paths={sim_cfg['n_paths']}, horizons={horizons})...")
    simulator = InventoryRiskSimulatorPyfunc(
        demand_model_uri=demand_uri,
        supply_model_uri=supply_uri,
        waste_model_uri=waste_uri,
        demand_schema=demand_schema,
        supply_schema=supply_schema,
        waste_schema=waste_schema,
        n_paths=int(sim_cfg["n_paths"]),
        horizons_days=horizons,
        demand_cv=float(sim_cfg.get("demand_cv", 0.15)),
        supply_cv=float(sim_cfg.get("supply_cv", 0.20)),
        waste_cv=float(sim_cfg.get("waste_cv", 0.15)),
        safety_threshold=float(sim_cfg.get("safety_threshold", 1.0)),
        stock_column=str(sim_cfg.get("stock_column", "current_inventory")),
        inventory_policy=dict(config.get("inventory_policy", {})),
        seed=seed,
    )

    input_example = pd.DataFrame(
        {
            "hospital_id": ["HOSP-1"],
            "blood_type": ["O+"],
            "component_type": ["RBC"],
            "event_timestamp": [pd.Timestamp.utcnow().tz_localize(None)],
            "current_inventory": [50.0],
            "usage_today": [8.0],
            "current_stock_units": [50.0],
            "shelf_life_days": [42.0],
            "units_used": [8.0],
            "units_collected": [10.0],
            "wastage": [1.0],
            "temp_c": [20.0],
            "is_holiday": [0],
            "dow": [3],
            "month": [5],
            "is_weekend": [0],
            "is_ice_storm": [0],
            "is_heat_wave": [0],
        }
    )

    sample_output = simulator.predict(None, input_example)
    print(f"Sample output columns: {list(sample_output.columns)}")
    print(f"Sample output:\n{sample_output.to_string()}")
    simulator.clear_child_model_cache()

    signature = infer_signature(input_example, sample_output)

    registered_name = str(config["registered_model_name"])
    experiment_name = str(config["id"])
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{registered_name}-simulator"):
        mlflow.set_tag("model_type", "composite_pyfunc")
        mlflow.set_tag("simulator_n_paths", str(sim_cfg["n_paths"]))
        mlflow.set_tag("simulator_horizons", str(horizons))
        mlflow.set_tag("demand_child_uri", demand_uri)
        mlflow.set_tag("supply_child_uri", supply_uri)
        mlflow.set_tag("waste_child_uri", waste_uri)

        mlflow.log_param("n_paths", sim_cfg["n_paths"])
        mlflow.log_param("horizons_days", str(horizons))
        mlflow.log_param("demand_cv", sim_cfg.get("demand_cv", 0.15))
        mlflow.log_param("supply_cv", sim_cfg.get("supply_cv", 0.20))
        mlflow.log_param("waste_cv", sim_cfg.get("waste_cv", 0.15))

        model_info = mlflow.pyfunc.log_model(
            artifact_path=registered_name,
            python_model=simulator,
            signature=signature,
            input_example=input_example if nested_bool(
                config.get("training", {}),
                ("mlflow", "input_example", "enabled"),
                True,
            ) else None,
            registered_model_name=registered_name,
        )

        client = MlflowClient()
        client.set_registered_model_alias(
            name=registered_name,
            alias="challenger",
            version=model_info.registered_model_version,
        )
        if os.getenv("PIOS_PROMOTE_TRAINED_MODELS", "0").strip().lower() in {
            "1", "true", "yes", "y", "on",
        }:
            client.set_registered_model_alias(
                name=registered_name,
                alias="champion",
                version=model_info.registered_model_version,
            )

    print(f"\nSimulator registered: {registered_name} v{model_info.registered_model_version}")
    print(f"Child models pinned:")
    print(f"  demand: {demand_uri}")
    print(f"  supply: {supply_uri}")
    print(f"  waste:  {waste_uri}")
    print("Done.")


if __name__ == "__main__":
    main()
