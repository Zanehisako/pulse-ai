from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


WeatherState = Literal["clear", "cloudy", "snow", "ice_storm", "blizzard"]


class ComparisonRequest(BaseModel):
    scenario_keys: list[str] = Field(default_factory=lambda: ["baseline"])
    strategy_keys: list[str] = Field(default_factory=lambda: ["baseline"])
    runs: int = Field(default=1, ge=1, le=5)
    seed_start: int = Field(default=100, ge=0, le=100_000)
    hours_override: int | None = Field(default=168, ge=24, le=1440)
    dreamerv3_run_key: str | None = None


class SingleRunRequest(BaseModel):
    scenario_key: str
    strategy_key: str
    seed: int = Field(default=100, ge=0, le=100_000)
    hours_override: int | None = Field(default=168, ge=24, le=1440)
    include_timeline: bool = True
    dreamerv3_run_key: str | None = None


class CustomScenarioCreate(BaseModel):
    name: str = Field(min_length=3, max_length=80)
    description: str = Field(default="", max_length=280)
    narrative: str = Field(default="", max_length=2000)
    base_scenario_key: str = "baseline"
    recommended_strategy_key: str = "baseline"
    sim_hours: int | None = Field(
        default=None,
        ge=24,
        le=1440,
        description="Total simulated hours before the episode ends.",
    )
    donor_inter_arrival_h: float | None = Field(
        default=None,
        gt=0.01,
        le=240.0,
        description="Average hours between donor arrivals. Lower values mean more donors.",
    )
    donor_show_factor: float | None = Field(
        default=None,
        ge=0.1,
        le=2.5,
        description="Multiplier applied to donor show-up probability.",
    )
    eligible_rate: float | None = Field(
        default=None,
        ge=0.1,
        le=1.0,
        description="Share of arriving donors who pass screening.",
    )
    demand_rate_h: float | None = Field(
        default=None,
        gt=0.01,
        le=240.0,
        description="Average hours between hospital orders. Lower values mean faster demand.",
    )
    demand_surge_factor: float | None = Field(
        default=None,
        ge=0.2,
        le=4.0,
        description="Multiplier on hospital demand intensity.",
    )
    avg_units_per_order: float | None = Field(
        default=None,
        gt=0.1,
        le=25.0,
        description="Average blood units requested per hospital order.",
    )
    transport_penalty: float | None = Field(
        default=None,
        ge=0.5,
        le=5.0,
        description="Base multiplier on travel and shipment time across the network.",
    )
    forced_weather: WeatherState | None = None
    lab_time_factor: float | None = Field(
        default=None,
        ge=0.1,
        le=3.0,
        description="Multiplier on laboratory testing and release time.",
    )
    proc_time_factor: float | None = Field(
        default=None,
        ge=0.1,
        le=3.0,
        description="Multiplier on post-lab processing time.",
    )
    initial_inventory_days: float | None = Field(
        default=None,
        ge=0.5,
        le=14.0,
        description="Starting inventory buffer expressed in days of supply.",
    )
    reserve_target_days: float | None = Field(
        default=None,
        ge=0.5,
        le=14.0,
        description="Reserve target the simulator tries to maintain in days of supply.",
    )
    regional_replenishment_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Share of daily demand the provincial network can replenish.",
    )
    demand_forecast_noise: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Random forecast error added when the demand outlook is refreshed.",
    )
    demand_forecast_interval_h: float | None = Field(
        default=None,
        ge=1.0,
        le=168.0,
        description="Hours between demand-forecast refresh cycles.",
    )
    demand_shock_scale: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Standard deviation of random demand shocks in the forecast model.",
    )
    transport_lead_time_noise: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Randomness applied to shipment lead times.",
    )
    congestion_threshold_units: float | None = Field(
        default=None,
        ge=0.0,
        le=200.0,
        description="Transport backlog level where congestion penalties begin.",
    )
    congestion_delay_factor: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Additional transport delay applied per unit above the congestion threshold.",
    )
    episode_budget: float | None = Field(
        default=None,
        ge=0.0,
        le=100_000.0,
        description="Total intervention budget available during the episode.",
    )


SnapshotSource = Literal["django", "synthetic"]
ReplayMode = Literal["none", "manual", "alerts", "hybrid"]


class DataSnapshotRequest(BaseModel):
    source: SnapshotSource = "django"
    donor_model: str | None = None
    donor_table: str | None = None
    donor_limit: int = Field(default=250, ge=1, le=5_000)
    include_raw_donors: bool = True
    include_alert_events: bool = True
    include_model_registry: bool = True
    as_of: str | None = None


class ReplayEventInput(BaseModel):
    key: str = Field(min_length=2, max_length=80)
    title: str = Field(default="", max_length=160)
    hour: float = Field(default=0.0, ge=0.0, le=17_520.0)
    duration_hours: float = Field(default=24.0, ge=1.0, le=2_160.0)
    severity: float = Field(default=0.5, ge=0.0, le=1.0)
    forced_weather: WeatherState | None = None
    modifiers: dict[str, float] = Field(default_factory=dict)
    universe_keys: list[str] = Field(default_factory=list)
    source: str = Field(default="manual", max_length=40)
    description: str = Field(default="", max_length=500)


class ReplayConfig(BaseModel):
    mode: ReplayMode = "none"
    divergence_hour: float | None = Field(default=None, ge=0.0, le=17_520.0)
    events: list[ReplayEventInput] = Field(default_factory=list)
    historical_event_limit: int = Field(default=12, ge=1, le=200)


class AgentModeConfig(BaseModel):
    enabled: bool = False
    dropout_sensitivity: float = Field(default=1.0, ge=0.1, le=5.0)
    reactivation_boost: float = Field(default=0.12, ge=0.0, le=1.0)
    intervention_effectiveness: float = Field(default=0.18, ge=0.0, le=1.5)
    fatigue_rate: float = Field(default=0.05, ge=0.0, le=1.0)


class UniversePolicyInput(BaseModel):
    key: str = Field(min_length=2, max_length=80)
    label: str = Field(min_length=2, max_length=120)
    strategy_key: str = "baseline"
    description: str = Field(default="", max_length=280)
    parameter_multipliers: dict[str, float] = Field(default_factory=dict)
    parameter_overrides: dict[str, float | int | str | None] = Field(
        default_factory=dict
    )
    replay_events: list[ReplayEventInput] = Field(default_factory=list)


class StudioExperimentRequest(BaseModel):
    scenario_key: str = "baseline"
    seed: int = Field(default=100, ge=0, le=100_000)
    replications: int = Field(default=3, ge=1, le=8)
    hours_override: int | None = Field(default=168, ge=24, le=1440)
    step_hours: float = Field(default=6.0, ge=1.0, le=168.0)
    timeline_months_ahead: int = Field(default=3, ge=0, le=24)
    include_timeline: bool = True
    snapshot: DataSnapshotRequest = Field(default_factory=DataSnapshotRequest)
    universes: list[UniversePolicyInput] = Field(default_factory=list)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    agent_mode: AgentModeConfig = Field(default_factory=AgentModeConfig)
    dreamerv3_run_key: str | None = None


class StrategyAssistantRequest(BaseModel):
    scenario_key: str = "baseline"
    seed: int = Field(default=200, ge=0, le=100_000)
    replications: int = Field(default=3, ge=1, le=8)
    hours_override: int | None = Field(default=168, ge=24, le=1440)
    timeline_months_ahead: int = Field(default=3, ge=0, le=24)
    dropout_rise_pct: float = Field(default=12.0, ge=0.0, le=100.0)
    snapshot: DataSnapshotRequest = Field(default_factory=DataSnapshotRequest)
    candidate_universes: list[UniversePolicyInput] = Field(default_factory=list)
    dreamerv3_run_key: str | None = None
