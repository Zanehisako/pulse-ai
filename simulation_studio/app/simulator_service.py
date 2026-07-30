from __future__ import annotations

import copy
import csv
import json
import os
import re
import sys
import threading
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

from .center_loader import resolve_center_configs
from .schemas import ComparisonRequest, CustomScenarioCreate, SingleRunRequest

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = APP_ROOT / "data"
CUSTOM_SCENARIOS_PATH = DATA_ROOT / "custom_scenarios.json"
SIMULATOR_ROOT = APP_ROOT / "simulator"
CACHED_SUMMARY_CANDIDATES = (
    SIMULATOR_ROOT / "results" / "strategy_eval_all_summary.csv",
    SIMULATOR_ROOT / "results" / "strategy_eval_summary.csv",
    SIMULATOR_ROOT / "strategy_eval_all_summary.csv",
    SIMULATOR_ROOT / "strategy_eval_summary.csv",
)
DREAMERV3_RUNS_ROOT = SIMULATOR_ROOT / "dreamerv3_runs"
CACHE_ROOT = APP_ROOT / ".cache"
STORE_LOCK = threading.Lock()
WEATHER_OPTIONS = ["clear", "cloudy", "snow", "ice_storm", "blizzard"]


os.environ.setdefault("PIOS_SIM_OFFLINE", "1")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))

from core import load_city_graph  # noqa: E402
from engine import (  # noqa: E402
    LEARNED_CONTINUOUS_CONTROLLER_KEYS,
    default_official_dreamerv3_checkpoint,
    resolve_learned_continuous_model_path,
    run_scenario,
    summarize_state,
)
from eval_metrics import DETAIL_METRICS, extract_eval_metrics  # noqa: E402
from scenarios import ACTION_CATALOG, SCENARIOS, STRATEGIES, ScenarioParams  # noqa: E402

STUDIO_STRATEGY_KEYS = [
    "baseline",
    "ppo_shortage_minimizer",
    "ppo_continuous",
    "sac_continuous",
    "iql_offline",
    "cql_offline",
    "dreamerv3_official",
]
DEFAULT_COMPARISON_STRATEGY_KEYS = [
    "ppo_continuous",
    "sac_continuous",
    "iql_offline",
    "cql_offline",
    "dreamerv3_official",
]
SIMULATION_LAB_POLICY_KEYS = [
    "ppo_continuous",
    "sac_continuous",
    "iql_offline",
    "cql_offline",
    "dreamerv3_official",
    "baseline",
    "mass_campaign",
    "lab_investment",
    "emergency_network",
    "full_response",
    "ppo_shortage_minimizer",
]
WEATHER_OPTION_LABELS = {
    "clear": "Clear",
    "cloudy": "Cloudy",
    "snow": "Snow",
    "ice_storm": "Ice Storm",
    "blizzard": "Blizzard",
}
CUSTOM_SCENARIO_GROUPS = [
    {
        "key": "episode",
        "label": "Episode",
        "description": "High-level run length and environment controls.",
    },
    {
        "key": "donor_supply",
        "label": "Donor Supply",
        "description": "Arrival pace, turnout, and screening assumptions for donors.",
    },
    {
        "key": "hospital_demand",
        "label": "Hospital Demand",
        "description": "Order timing and how intense hospital demand becomes.",
    },
    {
        "key": "operations_inventory",
        "label": "Operations & Inventory",
        "description": "Processing speed, network friction, and reserve buffers.",
    },
    {
        "key": "forecast_transport",
        "label": "Forecast & Congestion",
        "description": "Forecast refresh cadence, uncertainty, and backlog-driven delays.",
    },
    {
        "key": "budget",
        "label": "Budget",
        "description": "How much intervention spend the scenario allows.",
    },
]
CUSTOM_SCENARIO_FIELDS = [
    {
        "key": "sim_hours",
        "label": "Simulation Hours",
        "group": "episode",
        "input": "integer",
        "min": 24,
        "max": 1440,
        "step": 1,
        "unit": "hours",
        "description": "Total simulated hours before the episode ends.",
    },
    {
        "key": "forced_weather",
        "label": "Forced Weather",
        "group": "episode",
        "input": "select",
        "nullable": True,
        "options": [
            {"value": option, "label": WEATHER_OPTION_LABELS[option]}
            for option in WEATHER_OPTIONS
        ],
        "description": "Pins the whole episode to one weather state instead of using the winter weather engine.",
    },
    {
        "key": "donor_inter_arrival_h",
        "label": "Donor Inter-Arrival",
        "group": "donor_supply",
        "input": "number",
        "min": 0.01,
        "max": 240.0,
        "step": 0.01,
        "unit": "hours",
        "description": "Average hours between donor arrivals. Lower values mean donors arrive more often.",
    },
    {
        "key": "donor_show_factor",
        "label": "Donor Show Factor",
        "group": "donor_supply",
        "input": "number",
        "min": 0.1,
        "max": 2.5,
        "step": 0.01,
        "description": "Multiplier applied to donor show-up probability after registration or outreach.",
    },
    {
        "key": "eligible_rate",
        "label": "Eligibility Rate",
        "group": "donor_supply",
        "input": "number",
        "min": 0.1,
        "max": 1.0,
        "step": 0.01,
        "description": "Share of arriving donors who pass screening and can donate.",
    },
    {
        "key": "demand_rate_h",
        "label": "Demand Inter-Arrival",
        "group": "hospital_demand",
        "input": "number",
        "min": 0.01,
        "max": 240.0,
        "step": 0.01,
        "unit": "hours",
        "description": "Average hours between hospital orders. Lower values mean demand arrives faster.",
    },
    {
        "key": "demand_surge_factor",
        "label": "Demand Surge Factor",
        "group": "hospital_demand",
        "input": "number",
        "min": 0.2,
        "max": 4.0,
        "step": 0.01,
        "description": "Multiplier on overall hospital demand pressure and order intensity.",
    },
    {
        "key": "avg_units_per_order",
        "label": "Average Units Per Order",
        "group": "hospital_demand",
        "input": "number",
        "min": 0.1,
        "max": 25.0,
        "step": 0.01,
        "unit": "units",
        "description": "Average number of blood units requested in each hospital order.",
    },
    {
        "key": "transport_penalty",
        "label": "Transport Penalty",
        "group": "operations_inventory",
        "input": "number",
        "min": 0.5,
        "max": 5.0,
        "step": 0.01,
        "description": "Base multiplier on travel and shipment time across the network.",
    },
    {
        "key": "lab_time_factor",
        "label": "Lab Time Factor",
        "group": "operations_inventory",
        "input": "number",
        "min": 0.1,
        "max": 3.0,
        "step": 0.01,
        "description": "Multiplier on laboratory testing and release time.",
    },
    {
        "key": "proc_time_factor",
        "label": "Processing Time Factor",
        "group": "operations_inventory",
        "input": "number",
        "min": 0.1,
        "max": 3.0,
        "step": 0.01,
        "description": "Multiplier on post-lab processing and component-prep time.",
    },
    {
        "key": "initial_inventory_days",
        "label": "Initial Inventory Buffer",
        "group": "operations_inventory",
        "input": "number",
        "min": 0.5,
        "max": 14.0,
        "step": 0.1,
        "unit": "days",
        "description": "Starting citywide inventory buffer measured in days of supply.",
    },
    {
        "key": "reserve_target_days",
        "label": "Reserve Target",
        "group": "operations_inventory",
        "input": "number",
        "min": 0.5,
        "max": 14.0,
        "step": 0.1,
        "unit": "days",
        "description": "Target reserve buffer the replenishment system tries to maintain.",
    },
    {
        "key": "regional_replenishment_rate",
        "label": "Regional Replenishment Rate",
        "group": "operations_inventory",
        "input": "number",
        "min": 0.0,
        "max": 2.0,
        "step": 0.01,
        "description": "Share of daily demand the wider provincial network can top up.",
    },
    {
        "key": "demand_forecast_noise",
        "label": "Demand Forecast Noise",
        "group": "forecast_transport",
        "input": "number",
        "min": 0.0,
        "max": 2.0,
        "step": 0.01,
        "description": "Random forecast error added when the simulator refreshes demand outlook.",
    },
    {
        "key": "demand_forecast_interval_h",
        "label": "Forecast Refresh Interval",
        "group": "forecast_transport",
        "input": "number",
        "min": 1.0,
        "max": 168.0,
        "step": 0.5,
        "unit": "hours",
        "description": "Hours between demand forecast refresh cycles.",
    },
    {
        "key": "demand_shock_scale",
        "label": "Demand Shock Scale",
        "group": "forecast_transport",
        "input": "number",
        "min": 0.0,
        "max": 2.0,
        "step": 0.01,
        "description": "Standard deviation of random demand shocks in the forecast model.",
    },
    {
        "key": "transport_lead_time_noise",
        "label": "Transport Lead-Time Noise",
        "group": "forecast_transport",
        "input": "number",
        "min": 0.0,
        "max": 2.0,
        "step": 0.01,
        "description": "Randomness applied to shipment lead times on top of the base transport model.",
    },
    {
        "key": "congestion_threshold_units",
        "label": "Congestion Threshold",
        "group": "forecast_transport",
        "input": "number",
        "min": 0.0,
        "max": 200.0,
        "step": 0.5,
        "unit": "units",
        "description": "Transport backlog level where congestion penalties start to appear.",
    },
    {
        "key": "congestion_delay_factor",
        "label": "Congestion Delay Factor",
        "group": "forecast_transport",
        "input": "number",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "description": "Additional transport delay applied per unit above the congestion threshold.",
    },
    {
        "key": "episode_budget",
        "label": "Episode Budget",
        "group": "budget",
        "input": "number",
        "min": 0.0,
        "max": 100000.0,
        "step": 10.0,
        "unit": "budget units",
        "description": "Total intervention budget available to the selected strategy during the episode.",
    },
]
SCENARIO_PARAM_KEYS = tuple(field["key"] for field in CUSTOM_SCENARIO_FIELDS)
DEFAULT_SIM_BOUNDS = (None, 46.90, 46.70, -71.10, -71.35)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized.lower()).strip("-")
    return slug or "custom-scenario"


def ensure_store_exists() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not CUSTOM_SCENARIOS_PATH.exists():
        CUSTOM_SCENARIOS_PATH.write_text("[]\n", encoding="utf-8")


def load_custom_records() -> list[dict[str, Any]]:
    ensure_store_exists()
    try:
        with CUSTOM_SCENARIOS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid custom scenario store: {exc}") from exc
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="Custom scenario store must be a JSON array.")
    return data


def save_custom_records(records: list[dict[str, Any]]) -> None:
    ensure_store_exists()
    with CUSTOM_SCENARIOS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")


@lru_cache(maxsize=1)
def get_sim_context():
    try:
        return load_city_graph()
    except Exception:
        return DEFAULT_SIM_BOUNDS


def _read_latest_marker(base_dir: Path) -> tuple[Path | None, str | None]:
    latest_path = base_dir / "latest"
    if not latest_path.exists():
        return None, None

    try:
        if latest_path.is_symlink():
            target = Path(os.readlink(str(latest_path)))
        else:
            target = Path(latest_path.read_text(encoding="utf-8").strip())
    except OSError:
        return None, None

    if not str(target):
        return None, None

    resolved = target if target.is_absolute() else (base_dir / target).resolve()
    return resolved, str(target)


def dreamerv3_run_preference(item: dict[str, Any]) -> tuple[int, str, str]:
    run_key = str(item.get("key", "")).lower()
    run_name = str(item.get("run_name", "")).lower()
    preferred = os.environ.get("PIOS_DREAMERV3_PREFERRED_RUN", "").strip().lower()

    score = 0
    if preferred and preferred in {run_key, run_name}:
        score += 100
    if "aligned" in run_key or "aligned" in run_name:
        score += 40
    if "kpi" in run_key or "kpi" in run_name:
        score += 15
    if "budget" in run_key or "budget" in run_name:
        score -= 25
    return (score, str(item.get("updated_at", "")), run_name)


def scenario_params_payload(params: ScenarioParams) -> dict[str, Any]:
    return {key: getattr(params, key) for key in SCENARIO_PARAM_KEYS}


def custom_scenario_editor_config() -> dict[str, Any]:
    return {
        "groups": copy.deepcopy(CUSTOM_SCENARIO_GROUPS),
        "fields": copy.deepcopy(CUSTOM_SCENARIO_FIELDS),
    }


def discover_dreamerv3_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not DREAMERV3_RUNS_ROOT.exists():
        return runs

    for ckpt_dir in DREAMERV3_RUNS_ROOT.rglob("ckpt"):
        if not ckpt_dir.is_dir():
            continue

        run_dir = ckpt_dir.parent
        resolved_checkpoint: Path | None = None
        latest_name: str | None = None

        candidate, latest_name = _read_latest_marker(ckpt_dir)
        if candidate is not None and candidate.is_dir() and (candidate / "done").exists():
            resolved_checkpoint = candidate

        if resolved_checkpoint is None:
            completed_runs = sorted(
                [
                    candidate
                    for candidate in ckpt_dir.iterdir()
                    if candidate.is_dir() and (candidate / "done").exists()
                ],
                key=lambda candidate: candidate.stat().st_mtime,
                reverse=True,
            )
            if completed_runs:
                resolved_checkpoint = completed_runs[0]
                latest_name = completed_runs[0].name

        if resolved_checkpoint is None:
            continue

        relative_key = run_dir.relative_to(DREAMERV3_RUNS_ROOT).as_posix()
        updated_at = (
            datetime.fromtimestamp(resolved_checkpoint.stat().st_mtime, timezone.utc)
            .replace(microsecond=0)
            .isoformat()
        )
        scores_path = run_dir / "scores.jsonl"
        runs.append(
            {
                "key": relative_key,
                "label": run_dir.name.replace("_", " "),
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "checkpoint_input": str(run_dir),
                "resolved_checkpoint": str(resolved_checkpoint),
                "latest_checkpoint": latest_name,
                "updated_at": updated_at,
                "has_scores": scores_path.exists(),
            }
        )

    runs.sort(
        key=dreamerv3_run_preference,
        reverse=True,
    )
    return runs


def public_dreamerv3_run_payload(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": run["key"],
        "label": run["label"],
        "run_name": run["run_name"],
        "updated_at": run["updated_at"],
        "latest_checkpoint": run["latest_checkpoint"],
        "has_scores": run["has_scores"],
    }


def resolve_dreamerv3_run(run_key: str | None = None) -> dict[str, Any]:
    runs = discover_dreamerv3_runs()
    if not runs:
        raise HTTPException(
            status_code=400,
            detail=(
                "No DreamerV3 checkpoints were discovered under "
                f"{DREAMERV3_RUNS_ROOT}."
            ),
        )

    if run_key is None:
        return runs[0]

    for run in runs:
        if run["key"] == run_key:
            return run

    raise HTTPException(
        status_code=404,
        detail=f"Unknown DreamerV3 run '{run_key}'.",
    )


def strategy_availability(strategy_key: str) -> tuple[bool, str | None]:
    if strategy_key != "dreamerv3_official":
        return True, None

    runs = discover_dreamerv3_runs()
    if runs:
        return True, None

    try:
        checkpoint = Path(default_official_dreamerv3_checkpoint())
    except Exception as exc:
        return False, f"DreamerV3 checkpoint lookup failed: {exc}"

    if strategy_key in LEARNED_CONTINUOUS_CONTROLLER_KEYS:
        try:
            resolved = resolve_learned_continuous_model_path(strategy_key)
            if resolved is not None:
                return True, None
            labels = {
                "ppo_continuous": "PPO continuous",
                "sac_continuous": "SAC continuous",
                "iql_offline": "IQL offline",
                "cql_offline": "CQL offline",
            }
            return False, (
                f"{labels.get(strategy_key, strategy_key)} checkpoint not found in the "
                "standard simulator model locations."
            )
        except Exception as exc:
            return False, f"{strategy_key} model lookup failed: {exc}"

    return True, None


def strategy_payload(strategy_key: str) -> dict[str, Any]:
    strategy = STRATEGIES[strategy_key]
    available, disabled_reason = strategy_availability(strategy_key)
    action_keys = list(strategy.action_keys or strategy.controller_action_keys)
    available_runs = []
    if strategy_key == "dreamerv3_official":
        available_runs = discover_dreamerv3_runs()
    return {
        "key": strategy_key,
        "name": strategy.name,
        "description": strategy.description,
        "action_keys": action_keys,
        "actions": [
            {
                "key": action.key,
                "name": action.name,
                "description": action.description,
                "operational_cost": action.operational_cost,
            }
            for action in strategy.action_space
        ],
        "is_adaptive": strategy.is_adaptive,
        "controller_key": strategy.controller_key,
        "estimated_action_cost": float(strategy.action_cost),
        "available": available,
        "disabled_reason": disabled_reason,
        "available_run_count": len(available_runs),
    }


def scenario_payload(
    scenario_key: str,
    params: ScenarioParams,
    *,
    source: str,
    base_scenario_key: str | None = None,
    created_at: str | None = None,
    default_strategy_key: str | None = None,
) -> dict[str, Any]:
    return {
        "key": scenario_key,
        "name": params.name,
        "description": params.description,
        "narrative": params.narrative,
        "source": source,
        "is_holdout": bool(params.is_holdout),
        "tags": list(params.tags),
        "base_scenario_key": base_scenario_key,
        "created_at": created_at,
        "default_strategy_key": default_strategy_key or params.strategy_key,
        "params": scenario_params_payload(params),
    }


def build_custom_params(
    record: dict[str, Any], records: list[dict[str, Any]] | None = None
) -> ScenarioParams:
    records = records if records is not None else load_custom_records()
    records_by_key = {str(item.get("key", "")): item for item in records}
    params = _build_custom_params(record, records_by_key, seen_keys=set())
    params.scenario_key = record["key"]
    return params


def _build_custom_params(
    record: dict[str, Any],
    records_by_key: dict[str, dict[str, Any]],
    *,
    seen_keys: set[str],
) -> ScenarioParams:
    record_key = str(record.get("key", ""))
    if record_key:
        if record_key in seen_keys:
            raise HTTPException(
                status_code=500,
                detail=f"Cyclic custom scenario base chain detected at '{record_key}'.",
            )
        seen_keys = set(seen_keys)
        seen_keys.add(record_key)

    base_key = str(record.get("base_scenario_key") or "baseline")
    if base_key in SCENARIOS:
        params = copy.deepcopy(SCENARIOS[base_key])
    elif base_key in records_by_key:
        params = _build_custom_params(
            records_by_key[base_key],
            records_by_key,
            seen_keys=seen_keys,
        )
    else:
        params = copy.deepcopy(SCENARIOS["baseline"])

    for field_name, field_value in (record.get("params") or {}).items():
        if hasattr(params, field_name):
            setattr(params, field_name, field_value)

    params.strategy_key = record.get("default_strategy_key", params.strategy_key)
    return params


def list_scenarios() -> list[dict[str, Any]]:
    scenarios = [
        scenario_payload(key, params, source="built_in", default_strategy_key=params.strategy_key)
        for key, params in SCENARIOS.items()
    ]
    custom_records = load_custom_records()
    custom_scenarios = [
        scenario_payload(
            record["key"],
            build_custom_params(record, custom_records),
            source="custom",
            base_scenario_key=record.get("base_scenario_key"),
            created_at=record.get("created_at"),
            default_strategy_key=record.get("default_strategy_key"),
        )
        for record in custom_records
    ]
    return sorted(scenarios, key=lambda item: item["name"]) + sorted(
        custom_scenarios, key=lambda item: item["created_at"] or ""
    )


def list_strategies() -> list[dict[str, Any]]:
    return [
        strategy_payload(key)
        for key in STUDIO_STRATEGY_KEYS
        if key in STRATEGIES
    ]


def list_simulation_lab_policies() -> list[dict[str, Any]]:
    return [
        strategy_payload(key)
        for key in SIMULATION_LAB_POLICY_KEYS
        if key in STRATEGIES
    ]


def default_simulation_lab_universe_specs(
    *,
    min_policies: int = 2,
) -> list[dict[str, str]]:
    minimum = max(2, int(min_policies))
    catalog = {
        policy["key"]: policy
        for policy in list_simulation_lab_policies()
        if policy.get("available", True)
    }
    defaults: list[dict[str, str]] = []
    seen: set[str] = set()

    def append_strategy(strategy_key: str) -> None:
        if strategy_key in seen:
            return
        policy = catalog.get(strategy_key)
        if not policy:
            return
        seen.add(strategy_key)
        defaults.append(
            {
                "key": f"policy-{len(defaults) + 1}",
                "label": str(policy.get("name") or STRATEGIES[strategy_key].name),
                "strategy_key": strategy_key,
                "description": str(
                    policy.get("description") or STRATEGIES[strategy_key].description
                ),
            }
        )

    for strategy_key in DEFAULT_COMPARISON_STRATEGY_KEYS:
        if strategy_key in STRATEGIES:
            append_strategy(strategy_key)

    if len(defaults) < minimum:
        for strategy_key in SIMULATION_LAB_POLICY_KEYS:
            if strategy_key in STRATEGIES:
                append_strategy(strategy_key)
            if len(defaults) >= minimum:
                break

    return defaults


def get_app_metadata() -> dict[str, Any]:
    scenarios = list_scenarios()
    strategies = list_strategies()
    dreamerv3_runs = discover_dreamerv3_runs()
    return {
        "generated_at": utc_now_iso(),
        "offline_mode": os.environ.get("PIOS_SIM_OFFLINE", "1") not in {"0", "false", "no"},
        "weather_options": WEATHER_OPTIONS,
        "custom_scenario_editor": custom_scenario_editor_config(),
        "scenarios": scenarios,
        "strategies": strategies,
        "dreamerv3_runs": [public_dreamerv3_run_payload(run) for run in dreamerv3_runs],
        "default_dreamerv3_run_key": dreamerv3_runs[0]["key"]
        if dreamerv3_runs
        else None,
        "actions": [
            {
                "key": action.key,
                "name": action.name,
                "description": action.description,
                "operational_cost": action.operational_cost,
            }
            for action in ACTION_CATALOG.values()
        ],
        "defaults": {
            "comparison_scenarios": ["baseline", "combined_crisis", "holiday_flu_wave"],
            "comparison_strategies": list(DEFAULT_COMPARISON_STRATEGY_KEYS),
            "single_run_hours": 168,
            "comparison_hours": 168,
        },
    }


def scenario_name_lookup() -> dict[str, str]:
    lookup = {key: params.name for key, params in SCENARIOS.items()}
    for record in load_custom_records():
        lookup[record["key"]] = record["params"]["name"]
    return lookup


def resolve_scenario(scenario_key: str) -> ScenarioParams:
    if scenario_key in SCENARIOS:
        params = copy.deepcopy(SCENARIOS[scenario_key])
        params.scenario_key = scenario_key
        return params

    records = load_custom_records()
    for record in records:
        if record["key"] == scenario_key:
            return copy.deepcopy(build_custom_params(record, records))

    raise HTTPException(status_code=404, detail=f"Unknown scenario '{scenario_key}'.")


def ensure_strategy(strategy_key: str) -> None:
    if strategy_key not in STRATEGIES:
        raise HTTPException(status_code=404, detail=f"Unknown strategy '{strategy_key}'.")
    available, disabled_reason = strategy_availability(strategy_key)
    if not available:
        raise HTTPException(status_code=400, detail=disabled_reason or "Strategy is unavailable.")


def build_metric_row(
    *,
    state,
    strategy_key: str,
    scenario_key: str,
    seed: int,
    scenario_names: dict[str, str],
) -> dict[str, Any]:
    summary = summarize_state(state)
    row: dict[str, Any] = {
        "strategy": strategy_key,
        "strategy_name": STRATEGIES[strategy_key].name,
        "scenario": scenario_key,
        "scenario_name": scenario_names[scenario_key],
        "seed": seed,
    }
    row.update(extract_eval_metrics(summary, state))
    return row


def execute_simulation(params: ScenarioParams, *, seed: int, include_timeline: bool, **kwargs):
    try:
        return run_scenario(
            params,
            *get_sim_context(),
            seed=seed,
            enable_logs=False,
            fast_mode=not include_timeline,
            **kwargs,
        )
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["strategy"]), str(row["scenario"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (strategy_key, scenario_key), grouped_rows in sorted(grouped.items()):
        summary_row: dict[str, Any] = {
            "strategy": strategy_key,
            "strategy_name": STRATEGIES[strategy_key].name,
            "scenario": scenario_key,
            "scenario_name": str(grouped_rows[0]["scenario_name"]),
            "runs": float(len(grouped_rows)),
        }
        for metric_name in DETAIL_METRICS:
            values = np.asarray(
                [float(row[metric_name]) for row in grouped_rows],
                dtype=np.float32,
            )
            summary_row[f"{metric_name}_mean"] = float(values.mean())
            summary_row[f"{metric_name}_std"] = float(values.std())
        summary_rows.append(summary_row)
    return summary_rows


def best_by_scenario(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["scenario"])].append(row)

    for scenario_key, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                float(row["shortage_rate_mean"]),
                -float(row["episode_score_mean"]),
            )
        )
        best = rows[0]
        winners.append(
            {
                "scenario": scenario_key,
                "scenario_name": best["scenario_name"],
                "strategy": best["strategy"],
                "strategy_name": best["strategy_name"],
                "shortage_rate_mean": float(best["shortage_rate_mean"]),
                "episode_score_mean": float(best["episode_score_mean"]),
                "reward_total_mean": float(best["reward_total_mean"]),
            }
        )
    return sorted(winners, key=lambda item: item["scenario_name"])


def run_comparison(request: ComparisonRequest) -> dict[str, Any]:
    scenario_keys = list(dict.fromkeys(request.scenario_keys))
    strategy_keys = list(dict.fromkeys(request.strategy_keys))
    if not scenario_keys or not strategy_keys:
        raise HTTPException(status_code=400, detail="Select at least one scenario and one strategy.")

    for strategy_key in strategy_keys:
        ensure_strategy(strategy_key)

    scenario_names = scenario_name_lookup()
    rows: list[dict[str, Any]] = []
    selected_dreamerv3_run = (
        resolve_dreamerv3_run(request.dreamerv3_run_key)
        if "dreamerv3_official" in strategy_keys
        else None
    )
    center_configs, _ = resolve_center_configs()

    for scenario_key in scenario_keys:
        for strategy_key in strategy_keys:
            for run_idx in range(request.runs):
                seed = request.seed_start + run_idx
                params = resolve_scenario(scenario_key)
                params.strategy_key = strategy_key
                if request.hours_override is not None:
                    params.sim_hours = int(request.hours_override)
                run_kwargs: dict[str, Any] = {}
                if (
                    strategy_key == "dreamerv3_official"
                    and selected_dreamerv3_run is not None
                ):
                    run_kwargs["official_dreamerv3_checkpoint"] = (
                        selected_dreamerv3_run["checkpoint_input"]
                    )
                state = execute_simulation(
                    params,
                    seed=seed,
                    include_timeline=False,
                    center_configs=center_configs,
                    **run_kwargs,
                )
                rows.append(
                    build_metric_row(
                        state=state,
                        strategy_key=strategy_key,
                        scenario_key=scenario_key,
                        seed=seed,
                        scenario_names=scenario_names,
                    )
                )

    summary_rows = aggregate_metric_rows(rows)
    return {
        "generated_at": utc_now_iso(),
        "request": request.model_dump(),
        "dreamerv3_run": (
            public_dreamerv3_run_payload(selected_dreamerv3_run)
            if selected_dreamerv3_run is not None
            else None
        ),
        "summary_rows": summary_rows,
        "detail_rows": rows,
        "best_by_scenario": best_by_scenario(summary_rows),
    }


def serialize_center(center) -> dict[str, Any]:
    return {
        "name": center.name,
        "type": getattr(center, "original_type", None) or center.ctype,
        "role": center.ctype,
        "inventory": center.inventory_by_component(),
        "stats": {
            "donated": center.stats["donated"],
            "rejected": center.stats["rejected"],
            "no_show": center.stats["no_show"],
            "transfused": center.stats["transfused"],
            "expired": center.stats["expired"],
        },
    }


def run_single(request: SingleRunRequest) -> dict[str, Any]:
    ensure_strategy(request.strategy_key)
    params = resolve_scenario(request.scenario_key)
    params.strategy_key = request.strategy_key
    if request.hours_override is not None:
        params.sim_hours = int(request.hours_override)
    selected_dreamerv3_run = (
        resolve_dreamerv3_run(request.dreamerv3_run_key)
        if request.strategy_key == "dreamerv3_official"
        else None
    )

    center_configs, _ = resolve_center_configs()

    state = execute_simulation(
        params,
        seed=request.seed,
        include_timeline=request.include_timeline,
        center_configs=center_configs,
        official_dreamerv3_checkpoint=(
            selected_dreamerv3_run["checkpoint_input"]
            if selected_dreamerv3_run is not None
            else None
        ),
    )
    summary = summarize_state(state)
    reward = summary["reward"]

    return {
        "generated_at": utc_now_iso(),
        "scenario": scenario_payload(
            request.scenario_key,
            params,
            source="custom" if request.scenario_key not in SCENARIOS else "built_in",
            default_strategy_key=params.strategy_key,
        ),
        "strategy": strategy_payload(request.strategy_key),
        "dreamerv3_run": (
            public_dreamerv3_run_payload(selected_dreamerv3_run)
            if selected_dreamerv3_run is not None
            else None
        ),
        "seed": request.seed,
        "hours": params.sim_hours,
        "summary": {
            **extract_eval_metrics(summary, state),
            "active_action_keys": list(summary["active_action_keys"]),
            "shortage_by_component": summary["shortage_by_component"],
            "shortage_by_hospital": dict(summary["shortage_by_hospital"]),
        },
        "reward_terms": {key: float(value) for key, value in reward.terms.items()},
        "inventory_timeline": state.hourly_inventory,
        "centers": [serialize_center(center) for center in state.centers],
    }


def create_custom_scenario(payload: CustomScenarioCreate) -> dict[str, Any]:
    ensure_strategy(payload.recommended_strategy_key)
    base = resolve_scenario(payload.base_scenario_key)
    scenario_key_base = slugify(payload.name)

    overrides = payload.model_dump(exclude={"base_scenario_key", "recommended_strategy_key"})
    for field_name, field_value in overrides.items():
        if field_name in {"name", "description", "narrative"}:
            setattr(base, field_name, field_value or getattr(base, field_name))
            continue
        if field_value is not None:
            setattr(base, field_name, field_value)

    base.strategy_key = payload.recommended_strategy_key

    with STORE_LOCK:
        records = load_custom_records()
        candidate = scenario_key_base
        suffix = 2
        existing_keys = {record["key"] for record in records}.union(SCENARIOS.keys())
        while candidate in existing_keys:
            candidate = f"{scenario_key_base}-{suffix}"
            suffix += 1

        base.scenario_key = candidate
        record = {
            "key": candidate,
            "created_at": utc_now_iso(),
            "base_scenario_key": payload.base_scenario_key,
            "default_strategy_key": payload.recommended_strategy_key,
            "params": {
                "name": base.name,
                "description": base.description,
                "narrative": base.narrative,
                **scenario_params_payload(base),
            },
        }
        records.append(record)
        save_custom_records(records)

    return {
        "message": "Custom scenario created.",
        "scenario": scenario_payload(
            candidate,
            base,
            source="custom",
            base_scenario_key=payload.base_scenario_key,
            created_at=record["created_at"],
            default_strategy_key=payload.recommended_strategy_key,
        ),
    }


def load_cached_summary() -> dict[str, Any]:
    cached_summary_path = resolve_cached_summary_path()
    if cached_summary_path is None:
        return {
            "available": False,
            "generated_at": None,
            "summary_rows": [],
        }

    with cached_summary_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if key in {"strategy", "strategy_name", "scenario", "scenario_name"}:
                parsed[key] = value
            else:
                parsed[key] = float(value) if value not in {"", None} else None
        parsed_rows.append(parsed)

    return {
        "available": True,
        "generated_at": datetime.fromtimestamp(
            cached_summary_path.stat().st_mtime,
            timezone.utc,
        )
        .replace(microsecond=0)
        .isoformat(),
        "summary_rows": parsed_rows,
    }


def resolve_cached_summary_path() -> Path | None:
    existing_paths = [path for path in CACHED_SUMMARY_CANDIDATES if path.exists()]
    if not existing_paths:
        return None
    return max(existing_paths, key=lambda path: path.stat().st_mtime)
