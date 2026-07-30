"""
End-to-end orchestrator benchmark for local GGUF planner LLMs.

The benchmark is designed to be reproducible and thesis-friendly:

- It evaluates every locally available orchestrator GGUF model.
- It uses natural-language prompts that exercise predictive models,
  the simulation tool, and the db_search/db_tool path.
- It varies the available tool menu per case to measure dynamic routing.
- It writes raw results, summary tables, plots, and a markdown report.

The predictive-model tool path is executed against local MLflow artifacts.
The simulation tool path is executed through the real simulation adapter
against a deterministic local HTTP stub. The database tool path is routed
through the real orchestrator adapter, while the underlying DB execution is
replaced with deterministic fixture responses because the local benchmark
environment does not provide a live Postgres instance.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.core.registry import ModelRegistry
from ml.core.utils import as_float, safe_slug
from ml.orchestrator.service import DynamicXLAMOrchestrator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_GGUF_DIR = (WORKSPACE_ROOT / "ml-backend" / "models" / "gguf").resolve()
DEFAULT_OUTPUT_ROOT = (PROJECT_ROOT / "benchmark_artifacts" / "orchestrator_dynamic").resolve()

PREDICTIVE_MODEL_SPECS: tuple[dict[str, str], ...] = (
    {
        "id": "donor_propensity_model",
        "artifact_relpath": "mlflow_artifacts/2/0254ca274a1346c0ab4ae1cffa54ced9/artifacts/donor_propensity_model",
        "description": (
            "Individual donor propensity model. Use it for donor-level scoring or "
            "probability estimation from donor feature vectors, not for aggregate "
            "analytics or scenario simulation."
        ),
        "good_query": "Predict donor donation propensity for the donor described by the provided donor features.",
        "bad_query": "What would happen if blood donors dropped by 50% next week?",
    },
    {
        "id": "hospital_shortage_predictor",
        "artifact_relpath": "mlflow_artifacts/1/e7e9bd7652c74ec4924e4ce22f6a40fe/artifacts/hospital_shortage_predictor",
        "description": (
            "Hospital shortage risk model. Use it to forecast shortage risk from "
            "operational hospital inputs such as weather, surgeries, trauma load, "
            "and current inventory."
        ),
        "good_query": "Forecast hospital shortage risk from the provided hospital operating conditions.",
        "bad_query": "What is the average stock_end in the historical hospital table?",
    },
    {
        "id": "stockout_days_predictor",
        "artifact_relpath": "mlflow_artifacts/3/4d373827cc384906a3eeb07401e1740b/artifacts/stockout_days_predictor",
        "description": (
            "Days-to-stockout model for a specific blood product inventory state. "
            "Use it when the user provides inventory features and asks how many "
            "days remain until stockout."
        ),
        "good_query": "Estimate how many days remain until stockout for the provided blood product inventory state.",
        "bad_query": "Which hospitals currently have the highest O+ stock?",
    },
)

SIMULATION_METADATA = {
    "scenarios": [
        {
            "key": "baseline",
            "name": "Normal Operations",
            "description": "Balanced baseline blood-supply operations.",
            "params": {
                "sim_hours": 168,
                "donor_show_factor": 1.0,
                "demand_surge_factor": 1.0,
                "transport_penalty": 1.0,
                "initial_inventory_days": 3.0,
                "reserve_target_days": 2.0,
                "demand_forecast_interval_h": 24.0,
                "congestion_delay_factor": 0.05,
                "episode_budget": 4500.0,
            },
        }
    ],
    "strategies": [
        {"key": "baseline", "name": "Baseline", "available": True},
        {
            "key": "dreamerv3_official",
            "name": "Official DreamerV3",
            "available": True,
        },
        {"key": "ppo_continuous", "name": "PPO Continuous", "available": True},
        {"key": "sac_continuous", "name": "SAC Continuous", "available": True},
        {"key": "iql_offline", "name": "IQL Offline", "available": True},
        {"key": "cql_offline", "name": "CQL Offline", "available": True},
    ],
    "actions": [
        {
            "key": "campaign",
            "name": "Targeted donor campaign",
            "description": "Focused outreach to recover donor attendance.",
            "operational_cost": 60,
        },
        {
            "key": "emergency_share",
            "name": "Emergency sharing and courier routing",
            "description": "Emergency inventory rebalancing and courier response.",
            "operational_cost": 65,
        },
    ],
    "custom_scenario_editor": {
        "fields": [
            {"key": "sim_hours", "label": "Simulation Hours", "input": "integer", "min": 24, "max": 1440},
            {"key": "donor_show_factor", "label": "Donor Show Factor", "input": "number", "min": 0.1, "max": 2.5},
            {"key": "demand_surge_factor", "label": "Demand Surge Factor", "input": "number", "min": 0.2, "max": 4.0},
            {"key": "transport_penalty", "label": "Transport Penalty", "input": "number", "min": 0.5, "max": 5.0},
            {"key": "initial_inventory_days", "label": "Initial Inventory Buffer", "input": "number", "min": 0.5, "max": 14.0},
            {"key": "reserve_target_days", "label": "Reserve Target", "input": "number", "min": 0.5, "max": 14.0},
            {"key": "demand_forecast_interval_h", "label": "Forecast Refresh Interval", "input": "number", "min": 1.0, "max": 168.0},
            {"key": "congestion_delay_factor", "label": "Congestion Delay Factor", "input": "number", "min": 0.0, "max": 1.0},
            {"key": "episode_budget", "label": "Episode Budget", "input": "number", "min": 0.0, "max": 100000.0},
        ]
    },
    "default_dreamerv3_run_key": "benchmark_stub_run",
    "defaults": {
        "comparison_strategies": [
            "dreamerv3_official",
            "ppo_continuous",
            "sac_continuous",
            "iql_offline",
            "cql_offline",
        ]
    },
}

SIMULATION_POLICY_FACTORS = {
    "baseline": 1.0,
    "dreamerv3_official": 0.68,
    "ppo_continuous": 0.8,
    "sac_continuous": 0.86,
    "iql_offline": 0.93,
    "cql_offline": 1.02,
}

TOOL_FAMILY_ORDER = ["db_search", "simulation", "predictive_model", "other"]
TOOL_FAMILY_COLORS = {
    "db_search": "#0B7285",
    "simulation": "#E8590C",
    "predictive_model": "#2B8A3E",
    "other": "#6C757D",
}
QUALITY_METRIC_ORDER = [
    "planned_tool_match",
    "executed_tool_match",
    "execution_success",
    "menu_adherence",
    "argument_match",
    "case_score",
]


@dataclass(frozen=True)
class ReferencePredictiveModel:
    model_id: str
    artifact_uri: str
    artifact_path: Path
    description: str
    features: tuple[str, ...]
    sample_features: dict[str, Any]
    examples: tuple[dict[str, Any], ...]

    def config_row(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "file_path": self.artifact_uri,
            "type": "mlflow",
            "features": list(self.features),
            "description": self.description,
            "examples": [dict(row) for row in self.examples],
        }


@dataclass(frozen=True)
class ToolMenu:
    menu_id: str
    label: str
    db_enabled: bool
    simulation_enabled: bool
    allowed_model_ids: tuple[str, ...] | None = None

    def forced_model_ids(self, all_model_ids: tuple[str, ...]) -> list[str] | None:
        if self.allowed_model_ids is None:
            return None
        if not self.allowed_model_ids:
            return ["__disabled__"]
        return list(self.allowed_model_ids)

    def available_tools(self, all_model_ids: tuple[str, ...]) -> list[str]:
        model_ids = list(all_model_ids if self.allowed_model_ids is None else self.allowed_model_ids)
        tools: list[str] = []
        if self.db_enabled:
            tools.append("db_tool")
        if self.simulation_enabled:
            tools.append("simulation")
        tools.extend(model_ids)
        return tools


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    family: str
    label: str
    category: str
    query: str
    menu_id: str
    expected_tool: str
    provided_features: dict[str, Any] = field(default_factory=dict)
    required_plan_arguments: dict[str, Any] = field(default_factory=dict)
    required_plan_argument_keys: tuple[str, ...] = ()
    expected_execution_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkRun:
    orchestrator_model_id: str
    orchestrator_model_name: str
    load_time_ms: float
    case: BenchmarkCase
    menu: ToolMenu
    available_tools: tuple[str, ...]
    result: dict[str, Any]
    latency_ms: float
    planned_tool: str
    executed_tool: str
    planned_family: str
    executed_family: str
    planned_tool_match: bool
    executed_tool_match: bool
    execution_success: bool
    menu_adherence: bool
    argument_match: bool
    case_score: float
    planning_steps: list[dict[str, Any]]
    execution_rows: list[dict[str, Any]]

    def flat_row(self) -> dict[str, Any]:
        return {
            "orchestrator_model_id": self.orchestrator_model_id,
            "orchestrator_model_name": self.orchestrator_model_name,
            "load_time_ms": round(self.load_time_ms, 2),
            "case_id": self.case.case_id,
            "family": self.case.family,
            "label": self.case.label,
            "category": self.case.category,
            "menu_id": self.menu.menu_id,
            "menu_label": self.menu.label,
            "available_tool_count": len(self.available_tools),
            "available_tools": ", ".join(self.available_tools),
            "query": self.case.query,
            "expected_tool": self.case.expected_tool,
            "planned_tool": self.planned_tool,
            "executed_tool": self.executed_tool,
            "planned_family": self.planned_family,
            "executed_family": self.executed_family,
            "planned_tool_match": self.planned_tool_match,
            "executed_tool_match": self.executed_tool_match,
            "execution_success": self.execution_success,
            "menu_adherence": self.menu_adherence,
            "argument_match": self.argument_match,
            "case_score": round(self.case_score, 4),
            "latency_ms": round(self.latency_ms, 2),
            "planner_mode": str(self.result.get("planner_mode") or ""),
            "summary": str(self.result.get("natural_language_response") or ""),
            "plan_json": json.dumps(self.planning_steps, ensure_ascii=True, default=str),
            "execution_json": json.dumps(self.execution_rows, ensure_ascii=True, default=str),
        }


@dataclass(frozen=True)
class BenchmarkSummary:
    output_dir: Path
    raw_results_path: Path
    summary_path: Path
    report_path: Path
    plot_paths: tuple[Path, ...]
    overall_table: pd.DataFrame
    family_table: pd.DataFrame
    menu_table: pd.DataFrame
    results_df: pd.DataFrame


def build_reference_predictive_models(workspace_root: Path | None = None) -> dict[str, ReferencePredictiveModel]:
    root = workspace_root or WORKSPACE_ROOT
    models: dict[str, ReferencePredictiveModel] = {}
    for spec in PREDICTIVE_MODEL_SPECS:
        artifact_path = (root / spec["artifact_relpath"]).resolve()
        if not artifact_path.exists():
            raise FileNotFoundError(f"Benchmark artifact not found: {artifact_path}")
        example_payload = json.loads((artifact_path / "input_example.json").read_text(encoding="utf-8"))
        columns = tuple(str(item) for item in example_payload.get("columns", []))
        rows = example_payload.get("data", [])
        if not columns or not isinstance(rows, list) or not rows:
            raise RuntimeError(f"Invalid input_example.json for benchmark artifact: {artifact_path}")
        sample_features = dict(zip(columns, rows[0]))
        examples = (
            {
                "kind": "good",
                "user_query": spec["good_query"],
                "why": "This prompt matches the model's intended inference scope.",
            },
            {
                "kind": "bad",
                "user_query": spec["bad_query"],
                "why": "This prompt should be routed to another tool family.",
            },
        )
        model = ReferencePredictiveModel(
            model_id=spec["id"],
            artifact_uri=artifact_path.as_uri(),
            artifact_path=artifact_path,
            description=spec["description"],
            features=columns,
            sample_features=sample_features,
            examples=examples,
        )
        models[model.model_id] = model
    return models


def build_tool_menus(reference_models: dict[str, ReferencePredictiveModel]) -> dict[str, ToolMenu]:
    all_model_ids = tuple(reference_models.keys())
    return {
        "all_tools_all_models": ToolMenu(
            menu_id="all_tools_all_models",
            label="All tools + all predictive models",
            db_enabled=True,
            simulation_enabled=True,
            allowed_model_ids=all_model_ids,
        ),
        "db_only": ToolMenu(
            menu_id="db_only",
            label="DB search only",
            db_enabled=True,
            simulation_enabled=False,
            allowed_model_ids=(),
        ),
        "simulation_only": ToolMenu(
            menu_id="simulation_only",
            label="Simulation only",
            db_enabled=False,
            simulation_enabled=True,
            allowed_model_ids=(),
        ),
        "models_plus_db": ToolMenu(
            menu_id="models_plus_db",
            label="All predictive models + DB",
            db_enabled=True,
            simulation_enabled=False,
            allowed_model_ids=all_model_ids,
        ),
        "models_plus_simulation": ToolMenu(
            menu_id="models_plus_simulation",
            label="All predictive models + simulation",
            db_enabled=False,
            simulation_enabled=True,
            allowed_model_ids=all_model_ids,
        ),
        "donor_only": ToolMenu(
            menu_id="donor_only",
            label="Donor model only",
            db_enabled=False,
            simulation_enabled=False,
            allowed_model_ids=("donor_propensity_model",),
        ),
        "hospital_only": ToolMenu(
            menu_id="hospital_only",
            label="Hospital shortage model only",
            db_enabled=False,
            simulation_enabled=False,
            allowed_model_ids=("hospital_shortage_predictor",),
        ),
        "stockout_only": ToolMenu(
            menu_id="stockout_only",
            label="Stockout-days model only",
            db_enabled=False,
            simulation_enabled=False,
            allowed_model_ids=("stockout_days_predictor",),
        ),
    }


def build_benchmark_cases(reference_models: dict[str, ReferencePredictiveModel]) -> list[BenchmarkCase]:
    donor_features = dict(reference_models["donor_propensity_model"].sample_features)
    hospital_features = dict(reference_models["hospital_shortage_predictor"].sample_features)
    stockout_features = dict(reference_models["stockout_days_predictor"].sample_features)
    return [
        BenchmarkCase(
            case_id="inventory_lookup_all",
            family="inventory_lookup",
            label="Inventory lookup with full menu",
            category="db_search",
            query="Which hospitals currently have the highest O+ stock?",
            menu_id="all_tools_all_models",
            expected_tool="db_tool",
            required_plan_arguments={"table": "hospitals", "field": "stock_end"},
        ),
        BenchmarkCase(
            case_id="inventory_lookup_db_only",
            family="inventory_lookup",
            label="Inventory lookup with DB only",
            category="db_search",
            query="Which hospitals currently have the highest O+ stock?",
            menu_id="db_only",
            expected_tool="db_tool",
            required_plan_arguments={"table": "hospitals", "field": "stock_end"},
        ),
        BenchmarkCase(
            case_id="donor_age_average_models_db",
            family="donor_age_average",
            label="Average donor age with models + DB",
            category="db_search",
            query="What is the average donor age by country code?",
            menu_id="models_plus_db",
            expected_tool="db_tool",
            required_plan_arguments={"table": "donors", "field": "age", "aggregate": "avg"},
        ),
        BenchmarkCase(
            case_id="historical_stock_average_all",
            family="historical_stock_average",
            label="Historical stock average with full menu",
            category="db_search",
            query=(
                "In historical hospital data, what was the average stock_end when holiday "
                "was false, rain_mm was 0, trauma_cases was 10, and scheduled_surgeries was 50?"
            ),
            menu_id="all_tools_all_models",
            expected_tool="db_tool",
            required_plan_arguments={"table": "hospitals", "field": "stock_end", "aggregate": "avg"},
        ),
        BenchmarkCase(
            case_id="shock_forecast_all",
            family="shock_forecast",
            label="Scenario forecast with full menu",
            category="simulation",
            query="What would happen if blood donors dropped by 50% in the next week?",
            menu_id="all_tools_all_models",
            expected_tool="simulation",
            expected_execution_fields={"mode": "forecast"},
        ),
        BenchmarkCase(
            case_id="shock_forecast_sim_only",
            family="shock_forecast",
            label="Scenario forecast with simulation only",
            category="simulation",
            query="What would happen if blood donors dropped by 50% in the next week?",
            menu_id="simulation_only",
            expected_tool="simulation",
            expected_execution_fields={"mode": "forecast"},
        ),
        BenchmarkCase(
            case_id="shock_recommend_models_sim",
            family="shock_recommend",
            label="Scenario recommendation with models + simulation",
            category="simulation",
            query="What should we do if blood donors dropped by 50% in the next week?",
            menu_id="models_plus_simulation",
            expected_tool="simulation",
            expected_execution_fields={"mode": "recommend"},
        ),
        BenchmarkCase(
            case_id="policy_compare_all",
            family="policy_compare",
            label="Policy comparison with full menu",
            category="simulation",
            query="Compare DreamerV3, PPO, and SAC for a 30% donor drop next week.",
            menu_id="all_tools_all_models",
            expected_tool="simulation",
            expected_execution_fields={"mode": "compare"},
        ),
        BenchmarkCase(
            case_id="donor_propensity_all",
            family="donor_propensity",
            label="Donor propensity with full menu",
            category="predictive_model",
            query=(
                "Predict donor donation propensity for an available donor with roughly "
                "90 recency days and around 85 days until eligibility."
            ),
            menu_id="all_tools_all_models",
            expected_tool="donor_propensity_model",
            provided_features=donor_features,
        ),
        BenchmarkCase(
            case_id="donor_propensity_only",
            family="donor_propensity",
            label="Donor propensity with donor model only",
            category="predictive_model",
            query=(
                "Predict donor donation propensity for an available donor with roughly "
                "90 recency days and around 85 days until eligibility."
            ),
            menu_id="donor_only",
            expected_tool="donor_propensity_model",
            provided_features=donor_features,
        ),
        BenchmarkCase(
            case_id="hospital_shortage_all",
            family="hospital_shortage",
            label="Hospital shortage risk with full menu",
            category="predictive_model",
            query=(
                "Forecast hospital shortage risk from the provided operating conditions "
                "with low current inventory, no disaster, and a modest surgery load."
            ),
            menu_id="all_tools_all_models",
            expected_tool="hospital_shortage_predictor",
            provided_features=hospital_features,
        ),
        BenchmarkCase(
            case_id="hospital_shortage_only",
            family="hospital_shortage",
            label="Hospital shortage risk with hospital model only",
            category="predictive_model",
            query=(
                "Forecast hospital shortage risk from the provided operating conditions "
                "with low current inventory, no disaster, and a modest surgery load."
            ),
            menu_id="hospital_only",
            expected_tool="hospital_shortage_predictor",
            provided_features=hospital_features,
        ),
        BenchmarkCase(
            case_id="stockout_days_models_db",
            family="stockout_days",
            label="Days to stockout with models + DB",
            category="predictive_model",
            query=(
                "Estimate how many days remain until stockout for an A+ inventory state "
                "with about 40 units on hand and roughly 8 units used today."
            ),
            menu_id="models_plus_db",
            expected_tool="stockout_days_predictor",
            provided_features=stockout_features,
        ),
        BenchmarkCase(
            case_id="stockout_days_only",
            family="stockout_days",
            label="Days to stockout with stockout model only",
            category="predictive_model",
            query=(
                "Estimate how many days remain until stockout for an A+ inventory state "
                "with about 40 units on hand and roughly 8 units used today."
            ),
            menu_id="stockout_only",
            expected_tool="stockout_days_predictor",
            provided_features=stockout_features,
        ),
    ]


class DeterministicDbToolBackend:
    """Fixture-backed DB tool results for reproducible routing benchmarks."""

    def execute(self, query: str | None = None, **kwargs: Any) -> dict[str, Any]:
        prepared_query = str(query or "").strip().lower()
        table = safe_slug(str(kwargs.get("table") or ""))
        field = safe_slug(str(kwargs.get("field") or ""))
        aggregate = safe_slug(str(kwargs.get("aggregate") or ""))
        if (
            "average donor age by country code" in prepared_query
            or (table == "donors" and field == "age" and aggregate == "avg")
        ):
            rows = [
                {"group_value": "US", "value": 38.4},
                {"group_value": "CA", "value": 36.9},
                {"group_value": "DZ", "value": 34.2},
            ]
            return {
                "tool_type": "db_tool",
                "query_kind": "aggregate",
                "table": "donors",
                "field": "age",
                "aggregate": "avg",
                "rows": rows,
                "answer": "Average donor age by country code: US 38.4 years, CA 36.9 years, DZ 34.2 years.",
            }
        if (
            "average stock_end" in prepared_query
            or (table == "hospitals" and field == "stock_end" and aggregate == "avg")
        ):
            rows = [{"value": 47.2}]
            return {
                "tool_type": "db_tool",
                "query_kind": "aggregate",
                "table": "hospitals",
                "field": "stock_end",
                "aggregate": "avg",
                "rows": rows,
                "answer": "Average historical stock_end under the requested hospital conditions was 47.2 units.",
            }
        rows = [
            {"hospital_id": "North Hub", "stock_end": 512},
            {"hospital_id": "Central Hub", "stock_end": 487},
            {"hospital_id": "East Hub", "stock_end": 465},
        ]
        return {
            "tool_type": "db_tool",
            "query_kind": "inventory_lookup",
            "table": "hospitals",
            "field": "stock_end",
            "aggregate": "max",
            "rows": rows,
            "latest": {"date": "2026-04-20", "stock_end_total": 512, "records": 8},
            "series": [
                {"date": "2026-04-18", "stock_end_total": 476, "records": 8},
                {"date": "2026-04-19", "stock_end_total": 498, "records": 8},
                {"date": "2026-04-20", "stock_end_total": 512, "records": 8},
            ],
            "answer": "Highest O+ stock snapshot: North Hub 512 units, Central Hub 487 units, East Hub 465 units.",
        }


class _SimulationStubServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        super().__init__((host, port), _SimulationStubHandler)
        self.metadata_payload = json.loads(json.dumps(SIMULATION_METADATA))
        self.custom_scenarios: dict[str, dict[str, Any]] = {}
        self._scenario_counter = 0

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"

    def next_custom_key(self) -> str:
        self._scenario_counter += 1
        return f"bench-custom-{self._scenario_counter}"

    def create_scenario(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = self.next_custom_key()
        self.custom_scenarios[key] = dict(payload)
        title = str(payload.get("name") or payload.get("narrative") or "Benchmark Custom Scenario").strip()
        if not title:
            title = f"Benchmark Custom Scenario {self._scenario_counter}"
        scenario = {
            "key": key,
            "name": title[:80],
            "description": str(payload.get("narrative") or "Benchmark-generated custom scenario.").strip()[:240],
        }
        return {"scenario": scenario}

    def scenario_payload(self, scenario_key: str) -> dict[str, Any]:
        if scenario_key in self.custom_scenarios:
            return dict(self.custom_scenarios[scenario_key])
        for row in self.metadata_payload.get("scenarios", []):
            if isinstance(row, dict) and safe_slug(str(row.get("key") or "")) == safe_slug(scenario_key):
                return dict(row.get("params") or {})
        return dict(self.metadata_payload["scenarios"][0]["params"])

    def _severity_score(self, scenario_payload: dict[str, Any], hours_override: int | None) -> float:
        donor_drop = max(0.0, 1.0 - as_float(scenario_payload.get("donor_show_factor"), 1.0))
        demand_surge = max(0.0, as_float(scenario_payload.get("demand_surge_factor"), 1.0) - 1.0)
        transport_penalty = max(0.0, as_float(scenario_payload.get("transport_penalty"), 1.0) - 1.0)
        low_inventory = max(0.0, 3.0 - as_float(scenario_payload.get("initial_inventory_days"), 3.0))
        congestion = max(0.0, as_float(scenario_payload.get("congestion_delay_factor"), 0.05) - 0.05)
        long_horizon = max(0.0, (as_float(hours_override, as_float(scenario_payload.get("sim_hours"), 168.0)) / 168.0) - 1.0)
        return donor_drop + (demand_surge * 0.8) + (transport_penalty * 0.35) + (low_inventory * 0.3) + (congestion * 2.0) + (long_horizon * 0.2)

    def _build_run_payload(
        self,
        *,
        scenario_key: str,
        strategy_key: str,
        hours_override: int | None,
    ) -> dict[str, Any]:
        scenario_payload = self.scenario_payload(scenario_key)
        severity = self._severity_score(scenario_payload, hours_override)
        policy_factor = SIMULATION_POLICY_FACTORS.get(strategy_key, 1.0)
        shortage_rate = round(min(45.0, max(2.5, (6.0 + severity * 18.0) * policy_factor)), 1)
        service_rate = round(max(55.0, 100.0 - shortage_rate), 1)
        total_net_requested = int(max(900, 1000 + severity * 140))
        total_shortage = int(round(total_net_requested * shortage_rate / 100.0))
        total_transfused = max(0, total_net_requested - total_shortage)
        budget_spent = round(1800.0 + severity * 700.0 + (1.0 - policy_factor) * 600.0, 2)
        budget_remaining = round(max(500.0, as_float(scenario_payload.get("episode_budget"), 4500.0) - budget_spent), 2)
        episode_score = round(max(180.0, 520.0 - total_shortage * 0.6 - severity * 20.0 + (1.0 - policy_factor) * 75.0), 2)
        reward_total = round(episode_score + 40.0, 2)
        active_action_keys = ["campaign"]
        if shortage_rate >= 10.0:
            active_action_keys.append("emergency_share")
        rbc_units = int(round(total_shortage * 0.62))
        platelet_units = int(round(total_shortage * 0.24))
        plasma_units = max(0, total_shortage - rbc_units - platelet_units)
        hours = int(hours_override or as_float(scenario_payload.get("sim_hours"), 168.0))
        inventory_total = max(10, int(round(550 - total_shortage * 0.8)))
        scenario_name = scenario_key
        if scenario_key in self.custom_scenarios:
            title = str(self.custom_scenarios[scenario_key].get("name") or "").strip()
            if title:
                scenario_name = title[:80]
        elif scenario_key == "baseline":
            scenario_name = "Normal Operations"
        return {
            "scenario": {
                "key": scenario_key,
                "name": scenario_name,
                "description": "Deterministic local benchmark simulation response.",
            },
            "strategy": {
                "key": strategy_key,
                "name": next(
                    (
                        row["name"]
                        for row in self.metadata_payload.get("strategies", [])
                        if isinstance(row, dict) and safe_slug(str(row.get("key") or "")) == safe_slug(strategy_key)
                    ),
                    strategy_key,
                ),
                "description": "Benchmark strategy profile.",
            },
            "dreamerv3_run": {"key": "benchmark_stub_run"} if strategy_key == "dreamerv3_official" else None,
            "hours": hours,
            "summary": {
                "shortage_rate": shortage_rate,
                "service_rate": service_rate,
                "total_shortage": total_shortage,
                "total_net_requested": total_net_requested,
                "total_transfused": total_transfused,
                "budget_spent": budget_spent,
                "budget_remaining": budget_remaining,
                "episode_score": episode_score,
                "reward_total": reward_total,
                "exact_match_rate": round(max(70.0, 96.0 - severity * 8.0), 1),
                "compatible_substitution_rate": round(min(20.0, 4.0 + severity * 4.0), 1),
                "active_action_keys": active_action_keys,
                "shortage_by_component": {
                    "RBC": rbc_units,
                    "PLATELETS": platelet_units,
                    "PLASMA": plasma_units,
                },
                "shortage_by_hospital": {
                    "North Hub": round(total_shortage * 0.45, 2),
                    "Central Hub": round(total_shortage * 0.33, 2),
                    "East Hub": round(total_shortage * 0.22, 2),
                },
            },
            "inventory_timeline": [
                {"hour": max(24, hours // 4), "RBC": max(20, inventory_total + 90), "PLATELETS": 55, "PLASMA": 60, "shortages": 0},
                {"hour": max(48, hours // 2), "RBC": max(12, inventory_total + 30), "PLATELETS": 28, "PLASMA": 34, "shortages": max(0, int(total_shortage * 0.35))},
                {"hour": hours, "RBC": max(5, inventory_total // 2), "PLATELETS": max(4, inventory_total // 6), "PLASMA": max(5, inventory_total // 5), "shortages": total_shortage},
            ],
        }

    def run_scenario(self, body: dict[str, Any]) -> dict[str, Any]:
        scenario_key = str(body.get("scenario_key") or "baseline").strip() or "baseline"
        strategy_key = safe_slug(str(body.get("strategy_key") or "baseline"))
        hours_override = body.get("hours_override")
        return self._build_run_payload(
            scenario_key=scenario_key,
            strategy_key=strategy_key or "baseline",
            hours_override=int(hours_override) if hours_override is not None else None,
        )

    def compare_scenarios(self, body: dict[str, Any]) -> dict[str, Any]:
        scenario_key = "baseline"
        scenario_keys = body.get("scenario_keys")
        if isinstance(scenario_keys, list) and scenario_keys:
            scenario_key = str(scenario_keys[0] or "baseline").strip() or "baseline"
        hours_override = body.get("hours_override")
        rows = []
        for strategy_key in [safe_slug(str(item)) for item in body.get("strategy_keys", [])]:
            payload = self._build_run_payload(
                scenario_key=scenario_key,
                strategy_key=strategy_key or "baseline",
                hours_override=int(hours_override) if hours_override is not None else None,
            )
            summary = payload["summary"]
            rows.append(
                {
                    "strategy": strategy_key,
                    "strategy_name": payload["strategy"]["name"],
                    "scenario": payload["scenario"]["key"],
                    "scenario_name": payload["scenario"]["name"],
                    "shortage_rate_mean": summary["shortage_rate"],
                    "service_rate_mean": summary["service_rate"],
                    "total_shortage_mean": summary["total_shortage"],
                    "budget_spent_mean": summary["budget_spent"],
                    "episode_score_mean": summary["episode_score"],
                    "reward_total_mean": summary["reward_total"],
                    "exact_match_rate_mean": summary["exact_match_rate"],
                    "compatible_substitution_rate_mean": summary["compatible_substitution_rate"],
                }
            )
        rows.sort(key=lambda row: (row["shortage_rate_mean"], -row["service_rate_mean"], -row["episode_score_mean"]))
        return {
            "summary_rows": rows,
            "best_by_scenario": rows[:1],
            "dreamerv3_run": {"key": "benchmark_stub_run"},
        }


class _SimulationStubHandler(BaseHTTPRequestHandler):
    server: _SimulationStubServer

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover
        del format, args

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        if not raw.strip():
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _send_json(self, payload: dict[str, Any], status_code: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # pragma: no cover - covered through client requests
        if self.path.rstrip("/") == "/api/meta":
            self._send_json(self.server.metadata_payload)
            return
        self._send_json({"error": "not_found"}, status_code=404)

    def do_POST(self) -> None:  # pragma: no cover - covered through client requests
        body = self._read_json_body()
        if self.path.rstrip("/") == "/api/custom-scenarios":
            self._send_json(self.server.create_scenario(body))
            return
        if self.path.rstrip("/") == "/api/evaluations/run":
            self._send_json(self.server.run_scenario(body))
            return
        if self.path.rstrip("/") == "/api/evaluations/compare":
            self._send_json(self.server.compare_scenarios(body))
            return
        self._send_json({"error": "not_found"}, status_code=404)


@contextlib.contextmanager
def simulation_stub_server() -> Any:
    server = _SimulationStubServer()
    thread = threading.Thread(target=server.serve_forever, name="simulation-benchmark-stub", daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@contextlib.contextmanager
def temporary_env(overrides: dict[str, str]) -> Any:
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def build_registry(reference_models: dict[str, ReferencePredictiveModel], *, workspace_root: Path | None = None) -> ModelRegistry:
    root = workspace_root or WORKSPACE_ROOT
    registry = ModelRegistry(
        models_dir=(root / "backendMulti" / "ml" / "models").resolve(),
        config_path=(root / "backendMulti" / "ml" / "config" / "config.json").resolve(),
        config_loader=lambda: [model.config_row() for model in reference_models.values()],
    )
    registry.refresh(force=True)
    return registry


def discover_local_orchestrator_models(
    *,
    registry: ModelRegistry,
    gguf_dir: Path,
) -> list[dict[str, Any]]:
    env = {
        "PIOS_XLAM_DISABLE_LLM": "1",
        "PIOS_XLAM_MODEL_DIR": str(gguf_dir.resolve()),
        "PIOS_ORCH_ENABLE_DB_TOOL": "0",
        "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
        "PIOS_ORCH_ENABLE_SIMULATION_TOOL": "0",
    }
    with temporary_env(env):
        orchestrator = DynamicXLAMOrchestrator(registry=registry)
        rows = [row for row in orchestrator.list_available_models() if row.get("local_exists")]
    rows.sort(key=lambda row: (str(row.get("name") or row.get("id")), str(row.get("id"))))
    return rows


def _tool_family(tool_name: str, predictive_model_ids: set[str]) -> str:
    normalized = safe_slug(tool_name)
    if tool_name in predictive_model_ids:
        return "predictive_model"
    if normalized in {"db_tool", "db_search", "database", "sql", "search_db"}:
        return "db_search"
    if normalized in {"simulation", "simulate", "scenario_simulation", "policy_compare"}:
        return "simulation"
    return "other"


def _first_executed_tool(execution_rows: list[dict[str, Any]]) -> str:
    if not execution_rows:
        return ""
    for row in execution_rows:
        if row.get("success"):
            return str(row.get("tool") or "").strip()
    return str(execution_rows[0].get("tool") or "").strip()


def _menu_adherence(plan_steps: list[dict[str, Any]], available_tools: set[str]) -> bool:
    allowed = set(available_tools)
    allowed.update({"default", "fallback"})
    for row in plan_steps:
        tool_name = str(row.get("tool") or "").strip()
        if tool_name and tool_name not in allowed:
            return False
    return True


def _argument_match(case: BenchmarkCase, plan_steps: list[dict[str, Any]], execution_rows: list[dict[str, Any]]) -> bool:
    if not case.required_plan_arguments and not case.required_plan_argument_keys and not case.expected_execution_fields:
        return True
    if not plan_steps:
        return False
    first_step = plan_steps[0]
    arguments = first_step.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    for key, expected in case.required_plan_arguments.items():
        actual = arguments.get(key)
        if actual != expected:
            return False
    for key in case.required_plan_argument_keys:
        if key not in arguments:
            return False
    if case.expected_execution_fields:
        if not execution_rows:
            return False
        successful = None
        for row in execution_rows:
            if row.get("success"):
                successful = row
                break
        successful = successful or execution_rows[0]
        output = successful.get("output")
        if not isinstance(output, dict):
            return False
        for key, expected in case.expected_execution_fields.items():
            if output.get(key) != expected:
                return False
    return True


def _case_score(
    *,
    planned_tool_match: bool,
    executed_tool_match: bool,
    execution_success: bool,
    menu_adherence: bool,
    argument_match: bool,
) -> float:
    weights = {
        "planned_tool_match": 0.3,
        "executed_tool_match": 0.2,
        "execution_success": 0.2,
        "menu_adherence": 0.15,
        "argument_match": 0.15,
    }
    return round(
        (weights["planned_tool_match"] * float(planned_tool_match))
        + (weights["executed_tool_match"] * float(executed_tool_match))
        + (weights["execution_success"] * float(execution_success))
        + (weights["menu_adherence"] * float(menu_adherence))
        + (weights["argument_match"] * float(argument_match)),
        4,
    )


def _ensure_output_dir(output_root: Path) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (output_root / timestamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(_markdown_escape(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_markdown_escape(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "font.family": "DejaVu Serif",
            "savefig.bbox": "tight",
        }
    )


def _save_metric_bars(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    path = (output_dir / "overall_quality_metrics.png").resolve()
    metric_df = summary_df.melt(
        id_vars=["orchestrator_model_name"],
        value_vars=QUALITY_METRIC_ORDER[:-1],
        var_name="metric",
        value_name="score",
    )
    labels = list(summary_df["orchestrator_model_name"])
    metrics = QUALITY_METRIC_ORDER[:-1]
    x_positions = list(range(len(labels)))
    width = 0.14

    fig, ax = plt.subplots(figsize=(12, 6))
    for index, metric in enumerate(metrics):
        subset = metric_df[metric_df["metric"] == metric]
        values = [float(value) for value in subset["score"]]
        offset = (index - (len(metrics) - 1) / 2.0) * width
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=metric.replace("_", " ").title(),
        )

    ax.set_title("Routing and Execution Quality by Orchestrator LLM")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.legend(ncols=3, fontsize=9)
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_family_accuracy_plot(family_df: pd.DataFrame, output_dir: Path) -> Path:
    path = (output_dir / "planned_accuracy_by_tool_family.png").resolve()
    pivot = family_df.pivot(index="category", columns="orchestrator_model_name", values="planned_tool_match")
    categories = list(pivot.index)
    models = list(pivot.columns)
    x_positions = list(range(len(categories)))
    width = 0.16 if models else 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for index, model_name in enumerate(models):
        values = [float(value) for value in pivot[model_name]]
        offset = (index - (len(models) - 1) / 2.0) * width
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=model_name,
        )

    ax.set_title("Planned Tool Accuracy by Expected Tool Family")
    ax.set_ylabel("Mean planned-tool match")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label.replace("_", " ").title() for label in categories])
    ax.legend(fontsize=9)
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_tool_count_plot(results_df: pd.DataFrame, output_dir: Path) -> Path:
    path = (output_dir / "dynamic_score_by_available_tool_count.png").resolve()
    grouped = (
        results_df.groupby(["orchestrator_model_name", "available_tool_count"], as_index=False)["case_score"]
        .mean()
        .sort_values(["orchestrator_model_name", "available_tool_count"])
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for model_name, subset in grouped.groupby("orchestrator_model_name"):
        ax.plot(
            subset["available_tool_count"],
            subset["case_score"],
            marker="o",
            linewidth=2,
            label=model_name,
        )
    ax.set_title("Dynamic Benchmark Score vs. Number of Available Tools")
    ax.set_xlabel("Available tool count in the menu")
    ax.set_ylabel("Mean composite case score")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9)
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_menu_heatmap(menu_df: pd.DataFrame, output_dir: Path) -> Path:
    path = (output_dir / "menu_responsiveness_heatmap.png").resolve()
    pivot = menu_df.pivot(index="menu_label", columns="orchestrator_model_name", values="case_score")
    fig, ax = plt.subplots(figsize=(11, 6))
    heatmap = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("Mean Composite Score by Tool Menu")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    for row_index in range(len(pivot.index)):
        for col_index in range(len(pivot.columns)):
            ax.text(col_index, row_index, f"{pivot.iloc[row_index, col_index]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_latency_boxplot(results_df: pd.DataFrame, output_dir: Path) -> Path:
    path = (output_dir / "latency_distribution.png").resolve()
    labels = []
    values = []
    for model_name, subset in results_df.groupby("orchestrator_model_name"):
        labels.append(model_name)
        values.append([float(value) for value in subset["latency_ms"]])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(values, tick_labels=labels, patch_artist=True)
    ax.set_title("End-to-End Latency Distribution by Orchestrator LLM")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticklabels(labels, rotation=15, ha="right")
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_confusion_heatmaps(results_df: pd.DataFrame, output_dir: Path) -> Path:
    path = (output_dir / "tool_family_confusion_matrices.png").resolve()
    models = list(results_df["orchestrator_model_name"].drop_duplicates())
    family_index = {name: idx for idx, name in enumerate(TOOL_FAMILY_ORDER[:-1])}
    rows = max(1, math.ceil(len(models) / 2))
    cols = 2 if len(models) > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if hasattr(axes, "flat"):
        flat_axes = list(axes.flat)
    elif isinstance(axes, (list, tuple)):
        flat_axes = list(axes)
    else:
        flat_axes = [axes]

    for axis, model_name in zip(flat_axes, models):
        subset = results_df[results_df["orchestrator_model_name"] == model_name]
        matrix = [[0, 0, 0] for _ in range(3)]
        for _, row in subset.iterrows():
            expected_family = row["category"]
            planned_family = row["planned_family"]
            if expected_family not in family_index or planned_family not in family_index:
                continue
            matrix[family_index[expected_family]][family_index[planned_family]] += 1
        heatmap = axis.imshow(matrix, cmap="Oranges", vmin=0)
        axis.set_title(model_name)
        axis.set_xticks(range(3))
        axis.set_xticklabels([name.replace("_", " ") for name in TOOL_FAMILY_ORDER[:-1]], rotation=20, ha="right")
        axis.set_yticks(range(3))
        axis.set_yticklabels([name.replace("_", " ") for name in TOOL_FAMILY_ORDER[:-1]])
        for row_index in range(3):
            for col_index in range(3):
                axis.text(col_index, row_index, str(matrix[row_index][col_index]), ha="center", va="center", fontsize=10)
        fig.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)

    for axis in flat_axes[len(models):]:
        axis.axis("off")

    fig.suptitle("Expected vs. Planned Tool Family")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_plots(
    *,
    results_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    family_df: pd.DataFrame,
    menu_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, ...]:
    configure_plot_style()
    plot_paths = (
        _save_metric_bars(overall_df, output_dir),
        _save_family_accuracy_plot(family_df, output_dir),
        _save_tool_count_plot(results_df, output_dir),
        _save_menu_heatmap(menu_df, output_dir),
        _save_latency_boxplot(results_df, output_dir),
        _save_confusion_heatmaps(results_df, output_dir),
    )
    return plot_paths


def _report_tables(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_df = (
        results_df.groupby(["orchestrator_model_id", "orchestrator_model_name"], as_index=False)
        .agg(
            runs=("case_id", "count"),
            load_time_ms=("load_time_ms", "max"),
            planned_tool_match=("planned_tool_match", "mean"),
            executed_tool_match=("executed_tool_match", "mean"),
            execution_success=("execution_success", "mean"),
            menu_adherence=("menu_adherence", "mean"),
            argument_match=("argument_match", "mean"),
            case_score=("case_score", "mean"),
            median_latency_ms=("latency_ms", "median"),
        )
        .sort_values("case_score", ascending=False)
    )

    family_df = (
        results_df.groupby(["orchestrator_model_name", "category"], as_index=False)
        .agg(
            planned_tool_match=("planned_tool_match", "mean"),
            executed_tool_match=("executed_tool_match", "mean"),
            case_score=("case_score", "mean"),
            latency_ms=("latency_ms", "median"),
        )
        .sort_values(["category", "orchestrator_model_name"])
    )

    menu_df = (
        results_df.groupby(["orchestrator_model_name", "menu_id", "menu_label"], as_index=False)
        .agg(
            runs=("case_id", "count"),
            planned_tool_match=("planned_tool_match", "mean"),
            executed_tool_match=("executed_tool_match", "mean"),
            menu_adherence=("menu_adherence", "mean"),
            case_score=("case_score", "mean"),
        )
        .sort_values(["menu_label", "orchestrator_model_name"])
    )
    return overall_df, family_df, menu_df


def write_report(
    *,
    output_dir: Path,
    results_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    family_df: pd.DataFrame,
    menu_df: pd.DataFrame,
    plot_paths: tuple[Path, ...],
) -> Path:
    report_path = (output_dir / "report.md").resolve()
    sections: list[str] = []
    sections.append("# Dynamic Orchestrator LLM Benchmark")
    sections.append("")
    sections.append(f"Generated at: `{dt.datetime.now(dt.timezone.utc).isoformat()}`")
    sections.append("")
    sections.append("## Experimental Setup")
    sections.append("")
    sections.append(
        "This benchmark evaluates each locally available GGUF orchestrator model against "
        "natural-language prompts that require predictive-model inference, simulation, or "
        "db_search routing. The tool menu is varied per case to quantify whether the "
        "planner remains accurate as tools appear and disappear."
    )
    sections.append("")
    sections.append(f"- Number of runs: `{len(results_df)}`")
    sections.append(f"- Number of prompt/menu cases: `{results_df['case_id'].nunique()}`")
    sections.append(f"- Number of orchestrator LLMs: `{results_df['orchestrator_model_id'].nunique()}`")
    sections.append("")
    sections.append("## Overall Summary")
    sections.append("")
    sections.append(dataframe_to_markdown(overall_df.round(4)))
    sections.append("")
    sections.append("## Tool-Family Summary")
    sections.append("")
    sections.append(dataframe_to_markdown(family_df.round(4)))
    sections.append("")
    sections.append("## Tool-Menu Responsiveness")
    sections.append("")
    sections.append(dataframe_to_markdown(menu_df.round(4)))
    sections.append("")

    for model_name, subset in results_df.groupby("orchestrator_model_name"):
        sections.append(f"## Dynamic Trace: {model_name}")
        sections.append("")
        trace_df = subset[
            [
                "family",
                "menu_label",
                "available_tools",
                "expected_tool",
                "planned_tool",
                "executed_tool",
                "case_score",
            ]
        ].sort_values(["family", "menu_label"])
        sections.append(dataframe_to_markdown(trace_df.round(4)))
        sections.append("")

    sections.append("## Plot Files")
    sections.append("")
    for path in plot_paths:
        sections.append(f"- `{path.name}`")
    sections.append("")
    report_path.write_text("\n".join(sections), encoding="utf-8")
    return report_path


class OrchestratorDynamicBenchmark:
    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        gguf_dir: Path | None = None,
        output_root: Path | None = None,
        cpu_threads: int = 4,
        n_ctx: int = 2048,
        n_batch: int = 128,
        n_gpu_layers: int = 0,
    ) -> None:
        self.workspace_root = (workspace_root or WORKSPACE_ROOT).resolve()
        self.gguf_dir = (gguf_dir or DEFAULT_GGUF_DIR).resolve()
        self.output_root = (output_root or DEFAULT_OUTPUT_ROOT).resolve()
        self.cpu_threads = max(1, int(cpu_threads))
        self.n_ctx = max(512, int(n_ctx))
        self.n_batch = max(32, int(n_batch))
        self.n_gpu_layers = int(n_gpu_layers)

        self.reference_models = build_reference_predictive_models(self.workspace_root)
        self.registry = build_registry(self.reference_models, workspace_root=self.workspace_root)
        self.menus = build_tool_menus(self.reference_models)
        self.cases = build_benchmark_cases(self.reference_models)
        self.predictive_model_ids = set(self.reference_models.keys())

    def _orchestrator_env(self, model_id: str) -> dict[str, str]:
        return {
            "DJANGO_SKIP_ML_INIT": "1",
            "PIOS_XLAM_MODEL_DIR": str(self.gguf_dir),
            "PIOS_XLAM_MODEL_ID": model_id,
            "PIOS_XLAM_INIT_MODE": "blocking",
            "PIOS_XLAM_DISABLE_PROGRESS": "1",
            "PIOS_XLAM_CPU_THREADS": str(self.cpu_threads),
            "PIOS_XLAM_N_CTX": str(self.n_ctx),
            "PIOS_XLAM_N_BATCH": str(self.n_batch),
            "PIOS_XLAM_N_GPU_LAYERS": str(self.n_gpu_layers),
            "PIOS_ORCH_ENABLE_SEARCH_TOOL": "0",
            "PIOS_ORCH_LLM_API_URL": "",
        }

    def _load_orchestrator(self, model_id: str) -> tuple[DynamicXLAMOrchestrator, float]:
        with temporary_env(self._orchestrator_env(model_id)):
            started = time.perf_counter()
            orchestrator = DynamicXLAMOrchestrator(registry=self.registry)
            orchestrator.initialize_on_startup()
            load_time_ms = round((time.perf_counter() - started) * 1000.0, 2)
        status = orchestrator.status()
        if not status.get("llm_ready"):
            raise RuntimeError(
                f"Orchestrator model '{model_id}' failed to load: {status.get('llm_error')}"
            )
        return orchestrator, load_time_ms

    def _configure_menu(self, orchestrator: DynamicXLAMOrchestrator, menu: ToolMenu, simulation_base_url: str) -> tuple[list[str], list[str] | None]:
        all_model_ids = tuple(self.reference_models.keys())
        forced_model_ids = menu.forced_model_ids(all_model_ids)
        available_tools = menu.available_tools(all_model_ids)

        orchestrator.enable_external_fallback = True
        orchestrator.db_fallback_enabled = menu.db_enabled
        orchestrator.search_fallback_enabled = False
        orchestrator.simulation_tool_enabled = menu.simulation_enabled
        orchestrator.llm_api_url = ""
        orchestrator.simulation_base_url = simulation_base_url

        external_order: list[str] = []
        if menu.db_enabled:
            external_order.append(orchestrator.db_tool_name)
        if menu.simulation_enabled:
            external_order.append(orchestrator.simulation_tool_name)
        orchestrator.external_fallback_order = external_order or [orchestrator.db_tool_name, orchestrator.simulation_tool_name]

        original_tools_fn = getattr(orchestrator, "_benchmark_original_planner_external_tools", None)
        if original_tools_fn is None:
            original_tools_fn = orchestrator._planner_external_tools
            setattr(orchestrator, "_benchmark_original_planner_external_tools", original_tools_fn)

        def filtered_planner_external_tools() -> list[dict[str, Any]]:
            tools = original_tools_fn()
            return [
                tool
                for tool in tools
                if safe_slug(str(tool.get("name") or "")) not in {"llm_api", "search"}
            ]

        orchestrator._planner_external_tools = filtered_planner_external_tools  # type: ignore[assignment]
        return available_tools, forced_model_ids

    def _run_case(
        self,
        *,
        orchestrator: DynamicXLAMOrchestrator,
        orchestrator_model: dict[str, Any],
        load_time_ms: float,
        case: BenchmarkCase,
        menu: ToolMenu,
        simulation_base_url: str,
    ) -> BenchmarkRun:
        available_tools, forced_model_ids = self._configure_menu(orchestrator, menu, simulation_base_url)
        started = time.perf_counter()
        result = orchestrator.run(
            case.query,
            provided_features=dict(case.provided_features),
            forced_model_ids=forced_model_ids,
        )
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        plan_steps = list(result.get("plan", {}).get("steps") or [])
        execution_rows = list(result.get("execution_results") or [])
        planned_tool = str(plan_steps[0].get("tool") or "").strip() if plan_steps else ""
        executed_tool = _first_executed_tool(execution_rows)
        planned_family = _tool_family(planned_tool, self.predictive_model_ids)
        executed_family = _tool_family(executed_tool, self.predictive_model_ids)
        planned_tool_match = planned_tool == case.expected_tool
        executed_tool_match = executed_tool == case.expected_tool
        execution_success = bool(result.get("success"))
        menu_adherence = _menu_adherence(plan_steps, set(available_tools))
        argument_match = _argument_match(case, plan_steps, execution_rows)
        case_score = _case_score(
            planned_tool_match=planned_tool_match,
            executed_tool_match=executed_tool_match,
            execution_success=execution_success,
            menu_adherence=menu_adherence,
            argument_match=argument_match,
        )
        return BenchmarkRun(
            orchestrator_model_id=str(orchestrator_model.get("id") or ""),
            orchestrator_model_name=str(orchestrator_model.get("name") or orchestrator_model.get("id") or ""),
            load_time_ms=load_time_ms,
            case=case,
            menu=menu,
            available_tools=tuple(available_tools),
            result=result,
            latency_ms=latency_ms,
            planned_tool=planned_tool,
            executed_tool=executed_tool,
            planned_family=planned_family,
            executed_family=executed_family,
            planned_tool_match=planned_tool_match,
            executed_tool_match=executed_tool_match,
            execution_success=execution_success,
            menu_adherence=menu_adherence,
            argument_match=argument_match,
            case_score=case_score,
            planning_steps=plan_steps,
            execution_rows=execution_rows,
        )

    def run(
        self,
        *,
        selected_model_ids: list[str] | None = None,
        selected_case_ids: list[str] | None = None,
        quick: bool = False,
    ) -> BenchmarkSummary:
        output_dir = _ensure_output_dir(self.output_root)
        discovered_models = discover_local_orchestrator_models(
            registry=self.registry,
            gguf_dir=self.gguf_dir,
        )
        if selected_model_ids:
            allowed_ids = {safe_slug(item) for item in selected_model_ids}
            discovered_models = [
                row for row in discovered_models if safe_slug(str(row.get("id") or "")) in allowed_ids
            ]
        if quick and not selected_model_ids:
            discovered_models = discovered_models[:1]
        if not discovered_models:
            raise RuntimeError(f"No local GGUF orchestrator models found in {self.gguf_dir}")

        cases = list(self.cases)
        if selected_case_ids:
            allowed_case_ids = {str(item).strip() for item in selected_case_ids if str(item).strip()}
            cases = [case for case in cases if case.case_id in allowed_case_ids]
        if quick and not selected_case_ids:
            quick_case_ids = {
                "inventory_lookup_all",
                "shock_forecast_all",
                "donor_propensity_all",
                "hospital_shortage_all",
                "stockout_days_models_db",
            }
            cases = [case for case in cases if case.case_id in quick_case_ids]
        if not cases:
            raise RuntimeError("No benchmark cases selected.")

        db_backend = DeterministicDbToolBackend()
        raw_rows: list[dict[str, Any]] = []
        result_rows: list[dict[str, Any]] = []

        with simulation_stub_server() as simulation_server:
            with patch("ml.orchestrator.service.execute_db_tool", side_effect=db_backend.execute):
                with patch("ml.core.prediction._get_stats", return_value={}):
                    for orchestrator_model in discovered_models:
                        orchestrator, load_time_ms = self._load_orchestrator(str(orchestrator_model["id"]))
                        try:
                            for case in cases:
                                menu = self.menus[case.menu_id]
                                run = self._run_case(
                                    orchestrator=orchestrator,
                                    orchestrator_model=orchestrator_model,
                                    load_time_ms=load_time_ms,
                                    case=case,
                                    menu=menu,
                                    simulation_base_url=simulation_server.base_url,
                                )
                                raw_rows.append(
                                    {
                                        **run.flat_row(),
                                        "raw_result": run.result,
                                    }
                                )
                                result_rows.append(run.flat_row())
                        finally:
                            try:
                                del orchestrator
                            except Exception:
                                pass

        results_df = pd.DataFrame(result_rows)
        overall_df, family_df, menu_df = _report_tables(results_df)
        plot_paths = generate_plots(
            results_df=results_df,
            overall_df=overall_df,
            family_df=family_df,
            menu_df=menu_df,
            output_dir=output_dir,
        )

        raw_results_path = (output_dir / "results.json").resolve()
        raw_results_path.write_text(json.dumps(raw_rows, indent=2, ensure_ascii=True, default=str), encoding="utf-8")

        csv_path = (output_dir / "results.csv").resolve()
        results_df.to_csv(csv_path, index=False)

        summary_path = (output_dir / "summary.json").resolve()
        summary_payload = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "models": sorted(results_df["orchestrator_model_id"].drop_duplicates().tolist()),
            "overall": json.loads(overall_df.to_json(orient="records")),
            "family": json.loads(family_df.to_json(orient="records")),
            "menus": json.loads(menu_df.to_json(orient="records")),
            "plot_files": [path.name for path in plot_paths],
            "results_csv": csv_path.name,
            "results_json": raw_results_path.name,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")

        report_path = write_report(
            output_dir=output_dir,
            results_df=results_df,
            overall_df=overall_df,
            family_df=family_df,
            menu_df=menu_df,
            plot_paths=plot_paths,
        )

        return BenchmarkSummary(
            output_dir=output_dir,
            raw_results_path=raw_results_path,
            summary_path=summary_path,
            report_path=report_path,
            plot_paths=plot_paths,
            overall_table=overall_df,
            family_table=family_df,
            menu_table=menu_df,
            results_df=results_df,
        )
