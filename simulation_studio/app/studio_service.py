from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import HTTPException

from .schemas import (
    AgentModeConfig,
    ReplayConfig,
    ReplayEventInput,
    StrategyAssistantRequest,
    StudioExperimentRequest,
    UniversePolicyInput,
)
from .simulator_service import (
    default_simulation_lab_universe_specs,
    execute_simulation,
    ensure_strategy,
    public_dreamerv3_run_payload,
    resolve_dreamerv3_run,
    resolve_scenario,
    scenario_payload,
    strategy_payload,
    utc_now_iso,
)
from .snapshot_layer import DonorRecord, SimulationSnapshot, capture_data_snapshot, clamp
from .ws_simulation import (
    build_report,
    build_snapshot,
    initialize_simulation_from_params,
)

from engine import step_simulation  # noqa: E402
from scenarios import SCENARIOS, STRATEGIES, ScenarioParams  # noqa: E402

RUNTIME_FACTOR_FIELDS = {
    "donor_inter_arrival_h",
    "donor_show_factor",
    "eligible_rate",
    "demand_rate_h",
    "demand_surge_factor",
    "avg_units_per_order",
    "transport_penalty",
    "lab_time_factor",
    "proc_time_factor",
    "initial_inventory_days",
    "reserve_target_days",
    "regional_replenishment_rate",
    "episode_budget",
    "demand_forecast_noise",
    "demand_shock_scale",
    "transport_lead_time_noise",
    "congestion_delay_factor",
}
TIMELINE_METRICS = (
    "shortage_rate",
    "budget_remaining",
    "total_donated",
    "total_shortage",
    "total_transfused",
    "score",
)


@dataclass
class ScheduledReplayEvent:
    key: str
    title: str
    start_hour: float
    end_hour: float
    severity: float
    modifiers: dict[str, float] = field(default_factory=dict)
    forced_weather: str | None = None
    source: str = "manual"
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "start_hour": round(self.start_hour, 2),
            "end_hour": round(self.end_hour, 2),
            "severity": round(self.severity, 4),
            "modifiers": dict(self.modifiers),
            "forced_weather": self.forced_weather,
            "source": self.source,
            "description": self.description,
        }


def default_experiment_universes() -> list[UniversePolicyInput]:
    return [
        UniversePolicyInput(**payload)
        for payload in default_simulation_lab_universe_specs()
    ]


def default_assistant_universes() -> list[UniversePolicyInput]:
    return [
        UniversePolicyInput(**payload)
        for payload in default_simulation_lab_universe_specs()
    ]


def numeric_summary(summary: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in summary.items()
        if isinstance(value, (int, float))
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def confidence_triplet(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "std": float(arr.std()) if len(arr) else 0.0,
        "low": percentile(values, 10),
        "high": percentile(values, 90),
    }


def aggregate_report_summaries(reports: list[dict[str, Any]]) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        for key, value in numeric_summary(report["summary"]).items():
            metric_values[key].append(value)

    aggregate: dict[str, Any] = {}
    for key, values in sorted(metric_values.items()):
        stats = confidence_triplet(values)
        aggregate[f"{key}_mean"] = stats["mean"]
        aggregate[f"{key}_std"] = stats["std"]
        aggregate[f"{key}_low"] = stats["low"]
        aggregate[f"{key}_high"] = stats["high"]
    return aggregate


def build_confidence_bands(replications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not replications:
        return []
    max_steps = max(len(replication["timeline"]) for replication in replications)
    bands: list[dict[str, Any]] = []
    for step_index in range(max_steps):
        row: dict[str, Any] = {"step": step_index + 1}
        hours = [
            float(replication["timeline"][step_index]["hour"])
            for replication in replications
            if len(replication["timeline"]) > step_index
        ]
        row["hour"] = float(np.mean(hours)) if hours else 0.0
        for metric_name in TIMELINE_METRICS:
            values = [
                float(replication["timeline"][step_index][metric_name])
                for replication in replications
                if len(replication["timeline"]) > step_index
            ]
            if not values:
                continue
            stats = confidence_triplet(values)
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_low"] = stats["low"]
            row[f"{metric_name}_high"] = stats["high"]
        bands.append(row)
    return bands


def donor_heatmap(donors: list[DonorRecord]) -> list[dict[str, Any]]:
    buckets: Counter[tuple[float, float]] = Counter()
    blood_mix: dict[tuple[float, float], Counter[str]] = defaultdict(Counter)
    for donor in donors:
        if donor.latitude is None or donor.longitude is None:
            continue
        bucket = (round(donor.latitude, 2), round(donor.longitude, 2))
        buckets[bucket] += 1
        blood_mix[bucket][donor.blood_type] += 1
    rows: list[dict[str, Any]] = []
    for (lat, lon), count in buckets.items():
        rows.append(
            {
                "lat": lat,
                "lon": lon,
                "count": count,
                "blood_types": dict(sorted(blood_mix[(lat, lon)].items())),
            }
        )
    return sorted(rows, key=lambda item: item["count"], reverse=True)


def universe_policy_strength(universe: UniversePolicyInput) -> float:
    strategy = STRATEGIES[universe.strategy_key]
    weights = {
        "campaign": 0.24,
        "mobile_unit": 0.14,
        "extend_hours": 0.08,
        "lab_fast_track": 0.03,
        "surge_staff": 0.06,
        "emergency_share": 0.03,
        "clinical_conservation": 0.02,
        "national_mutual_aid": 0.02,
    }
    action_keys = strategy.action_keys or strategy.controller_action_keys
    score = sum(weights.get(key, 0.02) for key in action_keys)
    for field_name, value in universe.parameter_multipliers.items():
        if field_name == "donor_show_factor":
            score += max(float(value) - 1.0, -0.2) * 0.25
        elif field_name == "donor_inter_arrival_h":
            score += max(1.0 - float(value), -0.2) * 0.25
    return clamp(score, 0.0, 1.0)


class DonorAgentCoordinator:
    def __init__(
        self,
        donors: list[DonorRecord],
        config: AgentModeConfig,
        *,
        policy_strength: float,
    ) -> None:
        self.donors = donors
        self.config = config
        self.policy_strength = policy_strength
        self._baseline = self._simulate(hour=0.0, horizon_hours=720.0)

    def _donor_probabilities(
        self,
        donor: DonorRecord,
        *,
        hour: float,
        horizon_hours: float,
    ) -> tuple[float, float]:
        timeline_ratio = hour / max(horizon_hours, 1.0)
        recency = donor.recency_days if donor.recency_days is not None else 140.0
        if recency < 56.0:
            recency_modifier = 0.30
        elif recency < 120.0:
            recency_modifier = 1.18
        elif recency < 240.0:
            recency_modifier = 0.98
        else:
            recency_modifier = 0.78
        regular_modifier = 1.12 if donor.regular_donor else 0.93
        donation_modifier = 1.0 + min(donor.donation_count, 10) * 0.015
        dropout_modifier = 1.0 - (
            donor.dropout_risk * 0.35 * self.config.dropout_sensitivity
        )
        fatigue_modifier = 1.0 - (timeline_ratio * self.config.fatigue_rate)
        intervention_modifier = 1.0 + (
            self.policy_strength * self.config.intervention_effectiveness
        )
        reactivation_modifier = 1.0 + (
            max(recency - 120.0, 0.0) / 365.0
        ) * self.config.reactivation_boost * self.policy_strength

        show_probability = clamp(
            0.34
            * recency_modifier
            * regular_modifier
            * donation_modifier
            * dropout_modifier
            * fatigue_modifier
            * intervention_modifier
            * reactivation_modifier,
            0.02,
            0.98,
        )
        eligible_probability = clamp(
            (0.92 if donor.eligible else 0.58)
            * (0.42 if recency < 56.0 else 1.0)
            * (1.0 - (donor.dropout_risk * 0.06)),
            0.05,
            0.99,
        )
        return show_probability, eligible_probability

    def _simulate(self, *, hour: float, horizon_hours: float) -> dict[str, float]:
        if not self.donors:
            return {
                "expected_active_donors": 1.0,
                "show_probability": 0.85,
                "eligibility_probability": 0.85,
            }
        expected_active = 0.0
        show_probs: list[float] = []
        eligible_probs: list[float] = []
        for donor in self.donors:
            show_probability, eligible_probability = self._donor_probabilities(
                donor,
                hour=hour,
                horizon_hours=horizon_hours,
            )
            expected_active += show_probability * eligible_probability
            show_probs.append(show_probability)
            eligible_probs.append(eligible_probability)
        return {
            "expected_active_donors": expected_active,
            "show_probability": float(np.mean(show_probs)),
            "eligibility_probability": float(np.mean(eligible_probs)),
        }

    def adjustments_for_hour(self, *, hour: float, horizon_hours: float) -> dict[str, Any]:
        current = self._simulate(hour=hour, horizon_hours=horizon_hours)
        baseline_active = max(self._baseline["expected_active_donors"], 1e-6)
        availability_ratio = current["expected_active_donors"] / baseline_active
        show_ratio = current["show_probability"] / max(
            self._baseline["show_probability"],
            1e-6,
        )
        eligible_ratio = current["eligibility_probability"] / max(
            self._baseline["eligibility_probability"],
            1e-6,
        )
        return {
            "parameter_multipliers": {
                "donor_inter_arrival_h": clamp(1.0 / max(availability_ratio, 0.35), 0.45, 2.5),
                "donor_show_factor": clamp(show_ratio, 0.55, 1.45),
                "eligible_rate": clamp(eligible_ratio, 0.65, 1.25),
            },
            "summary": {
                "expected_active_donors": round(current["expected_active_donors"], 2),
                "average_show_probability": round(current["show_probability"], 4),
                "average_eligibility_probability": round(
                    current["eligibility_probability"], 4
                ),
                "availability_ratio": round(availability_ratio, 4),
            },
        }


class RuntimeAdjustmentController:
    def __init__(
        self,
        state,
        base_params: ScenarioParams,
        scheduled_events: list[ScheduledReplayEvent],
    ) -> None:
        self.state = state
        self.base_params = {
            field_name: float(getattr(base_params, field_name))
            for field_name in RUNTIME_FACTOR_FIELDS
            if hasattr(base_params, field_name)
            and isinstance(getattr(base_params, field_name), (int, float))
        }
        self.default_weather = base_params.forced_weather or state.weather.state
        self.pending_events = sorted(scheduled_events, key=lambda event: event.start_hour)
        self.active_events: list[ScheduledReplayEvent] = []
        self.applied_events: list[dict[str, Any]] = []
        self.agent_adjustments: dict[str, float] = {}

    def set_agent_adjustments(self, adjustments: dict[str, float]) -> None:
        self.agent_adjustments = dict(adjustments)
        self._recompute()

    def advance_to(self, hour: float) -> list[dict[str, Any]]:
        transitions: list[dict[str, Any]] = []
        retained: list[ScheduledReplayEvent] = []
        for event in self.active_events:
            if event.end_hour <= hour + 1e-6:
                transitions.append(
                    {
                        "kind": "expired",
                        "hour": round(hour, 2),
                        "event": event.as_dict(),
                    }
                )
            else:
                retained.append(event)
        self.active_events = retained

        while self.pending_events and self.pending_events[0].start_hour <= hour + 1e-6:
            event = self.pending_events.pop(0)
            self.active_events.append(event)
            transitions.append(
                {"kind": "activated", "hour": round(hour, 2), "event": event.as_dict()}
            )
            self.applied_events.append(
                {
                    **event.as_dict(),
                    "activated_at": round(hour, 2),
                }
            )

        self._recompute()
        return transitions

    def _recompute(self) -> None:
        event_multipliers = {field_name: 1.0 for field_name in self.base_params}
        forced_weather: tuple[float, str] | None = None
        for event in self.active_events:
            for field_name, factor in event.modifiers.items():
                if field_name in event_multipliers:
                    event_multipliers[field_name] *= float(factor)
            if event.forced_weather is not None:
                if forced_weather is None or event.severity >= forced_weather[0]:
                    forced_weather = (event.severity, event.forced_weather)

        for field_name, base_value in self.base_params.items():
            value = base_value
            value *= event_multipliers.get(field_name, 1.0)
            value *= self.agent_adjustments.get(field_name, 1.0)
            setattr(self.state.params, field_name, float(value))

        self.state.weather.state = forced_weather[1] if forced_weather else self.default_weather


def merge_multiplier_dict(
    base: dict[str, float],
    extra: dict[str, float],
) -> dict[str, float]:
    merged = dict(base)
    for key, value in extra.items():
        merged[key] = float(merged.get(key, 1.0)) * float(value)
    return merged


def normalized_universe(universe: UniversePolicyInput) -> UniversePolicyInput:
    ensure_strategy(universe.strategy_key)
    payload = universe.model_dump()
    payload["key"] = str(payload.get("key") or "").strip()
    if not payload["key"]:
        raise HTTPException(status_code=400, detail="Universe key is required.")
    return UniversePolicyInput(**payload)


def next_universe_key(seen_keys: set[str]) -> str:
    index = 1
    candidate = f"policy-{index}"
    while candidate in seen_keys:
        index += 1
        candidate = f"policy-{index}"
    return candidate


def unique_universe_list(
    universes: list[UniversePolicyInput],
) -> list[UniversePolicyInput]:
    normalized: list[UniversePolicyInput] = []
    seen_keys: set[str] = set()
    for universe in universes:
        prepared = normalized_universe(universe)
        if prepared.key in seen_keys:
            payload = prepared.model_dump()
            payload["key"] = next_universe_key(seen_keys)
            prepared = UniversePolicyInput(**payload)
        seen_keys.add(prepared.key)
        normalized.append(prepared)
    return normalized


def experiment_universes(request: StudioExperimentRequest) -> list[UniversePolicyInput]:
    universes = request.universes or default_experiment_universes()
    return unique_universe_list(universes)


def apply_snapshot_calibration(
    scenario_key: str,
    snapshot: SimulationSnapshot,
    universe: UniversePolicyInput,
    *,
    hours_override: int | None,
    timeline_months_ahead: int,
) -> tuple[ScenarioParams, list[str]]:
    params = copy.deepcopy(resolve_scenario(scenario_key))
    params.strategy_key = universe.strategy_key
    notes: list[str] = []
    base_hours = int(hours_override) if hours_override is not None else int(params.sim_hours)
    sim_hours = max(24, base_hours)
    if timeline_months_ahead > 0:
        sim_hours = max(sim_hours, timeline_months_ahead * 30 * 24)
    params.sim_hours = sim_hours
    if timeline_months_ahead > 0:
        notes.append(f"Projected horizon set to {params.sim_hours} hours.")
    else:
        notes.append(f"Classic mission horizon set to {params.sim_hours} hours.")

    if timeline_months_ahead <= 0:
        notes.append(
            "Classic parity mode keeps the base scenario parameters unchanged. The snapshot remains available for comparison context, replay data, and map layers."
        )
    else:
        calibration = snapshot.calibration
        donor_summary = snapshot.donor_summary
        params.donor_inter_arrival_h = clamp(
            (params.donor_inter_arrival_h * 0.65)
            + (float(calibration["estimated_donor_inter_arrival_h"]) * 0.35),
            0.20,
            72.0,
        )
        params.donor_show_factor = clamp(
            (params.donor_show_factor * 0.60)
            + (float(calibration["estimated_donor_show_factor"]) * 0.40),
            0.25,
            1.8,
        )
        params.eligible_rate = clamp(
            (params.eligible_rate * 0.70)
            + (float(calibration["estimated_eligible_rate"]) * 0.30),
            0.15,
            0.99,
        )
        params.demand_surge_factor = clamp(
            params.demand_surge_factor
            * (1.0 + (float(calibration["alert_pressure"]) * 0.18)),
            0.5,
            4.0,
        )
        params.reserve_target_days = clamp(
            params.reserve_target_days
            * (1.0 + (float(donor_summary.get("critical_event_count", 0)) * 0.03)),
            0.5,
            14.0,
        )
        horizon_pressure = 1.0 + (timeline_months_ahead * 0.02)
        params.donor_inter_arrival_h = clamp(
            params.donor_inter_arrival_h * (1.0 + (timeline_months_ahead * 0.015)),
            0.20,
            72.0,
        )
        params.demand_rate_h = clamp(
            params.demand_rate_h / horizon_pressure,
            0.05,
            72.0,
        )
        notes.append(
            "Snapshot calibration blended donor show-up, eligibility, alert pressure, and future horizon drift into the base scenario."
        )

    for field_name, factor in universe.parameter_multipliers.items():
        if field_name in RUNTIME_FACTOR_FIELDS and hasattr(params, field_name):
            current = getattr(params, field_name)
            if isinstance(current, (int, float)):
                setattr(params, field_name, float(current) * float(factor))
                notes.append(f"Applied universe multiplier {field_name} x{float(factor):.3f}.")

    for field_name, value in universe.parameter_overrides.items():
        if not hasattr(params, field_name):
            continue
        setattr(params, field_name, value)
        notes.append(f"Applied universe override {field_name}={value}.")

    return params, notes


def replay_event_from_input(event: ReplayEventInput) -> ScheduledReplayEvent:
    return ScheduledReplayEvent(
        key=event.key,
        title=event.title or event.key.replace("_", " ").title(),
        start_hour=float(event.hour),
        end_hour=float(event.hour + event.duration_hours),
        severity=float(event.severity),
        modifiers=dict(event.modifiers),
        forced_weather=event.forced_weather,
        source=event.source,
        description=event.description,
    )


def historical_replay_events(
    snapshot: SimulationSnapshot,
    replay: ReplayConfig,
    *,
    horizon_hours: float,
) -> list[ScheduledReplayEvent]:
    events = snapshot.historical_events[: replay.historical_event_limit]
    if not events:
        return []
    replay_window = replay.divergence_hour
    if replay_window is None:
        replay_window = min(horizon_hours * 0.25, 30.0 * 24.0)
    replay_window = max(float(replay_window), 1.0)
    scheduled: list[ScheduledReplayEvent] = []
    count = len(events)
    for index, event in enumerate(reversed(events)):
        start_hour = replay_window * (index / max(count - 1, 1))
        duration = 12.0 + (event.severity * 36.0)
        scheduled.append(
            ScheduledReplayEvent(
                key=event.key,
                title=event.title,
                start_hour=start_hour,
                end_hour=min(start_hour + duration, horizon_hours),
                severity=event.severity,
                modifiers=dict(event.modifiers),
                forced_weather=event.forced_weather,
                source="historical_alert",
                description=f"Replayed from historical {event.category} event.",
            )
        )
    return scheduled


def universe_replay_schedule(
    universe: UniversePolicyInput,
    request: StudioExperimentRequest,
    snapshot: SimulationSnapshot,
    *,
    horizon_hours: float,
) -> list[ScheduledReplayEvent]:
    scheduled: list[ScheduledReplayEvent] = []
    if request.replay.mode in {"alerts", "hybrid"}:
        scheduled.extend(historical_replay_events(snapshot, request.replay, horizon_hours=horizon_hours))

    if request.replay.mode in {"manual", "hybrid"}:
        for event in request.replay.events:
            if event.universe_keys and universe.key not in event.universe_keys:
                continue
            scheduled.append(replay_event_from_input(event))

    for event in universe.replay_events:
        scheduled.append(replay_event_from_input(event))

    if request.replay.divergence_hour is not None:
        divergence_hour = float(request.replay.divergence_hour)
        filtered: list[ScheduledReplayEvent] = []
        for event in scheduled:
            if event.source == "historical_alert" and event.start_hour > divergence_hour:
                continue
            filtered.append(event)
        scheduled = filtered

    return sorted(scheduled, key=lambda event: (event.start_hour, event.key))


def retention_score(aggregate_summary: dict[str, Any]) -> float:
    service = float(aggregate_summary.get("service_rate_mean", 0.0))
    shortage = float(aggregate_summary.get("shortage_rate_mean", 0.0))
    donated = float(aggregate_summary.get("total_donated_mean", 0.0))
    no_show = float(aggregate_summary.get("total_no_show_mean", 0.0))
    exact_match = float(aggregate_summary.get("exact_match_rate_mean", 0.0))
    return (
        (service * 1.15)
        - (shortage * 0.95)
        + (donated * 0.035)
        - (no_show * 0.015)
        + (exact_match * 0.08)
    )


def shortage_sort_tuple(result: dict[str, Any]) -> tuple[float, float, float, float]:
    summary = result.get("summary", {})
    shortage = float(summary.get("shortage_rate_mean", 1000.0))
    episode_score = float(summary.get("episode_score_mean", 0.0))
    reward_total = float(summary.get("reward_total_mean", 0.0))
    service_rate = float(summary.get("service_rate_mean", 0.0))
    return (
        shortage,
        -episode_score,
        -reward_total,
        -service_rate,
    )


def universe_uses_extended_runtime(
    universe: UniversePolicyInput,
    request: StudioExperimentRequest,
) -> bool:
    return bool(
        request.timeline_months_ahead > 0
        or request.agent_mode.enabled
        or request.replay.mode != "none"
        or universe.parameter_multipliers
        or universe.parameter_overrides
        or universe.replay_events
    )


def selected_dreamer_run(
    universe: UniversePolicyInput,
    request: StudioExperimentRequest,
) -> dict[str, Any] | None:
    if universe.strategy_key == "dreamerv3_official":
        return resolve_dreamerv3_run(request.dreamerv3_run_key)
    return None


def run_universe_replication(
    scenario_key: str,
    snapshot: SimulationSnapshot,
    universe: UniversePolicyInput,
    request: StudioExperimentRequest,
    *,
    seed: int,
) -> dict[str, Any]:
    selected_run = selected_dreamer_run(universe, request)
    params, calibration_notes = apply_snapshot_calibration(
        scenario_key,
        snapshot,
        universe,
        hours_override=request.hours_override,
        timeline_months_ahead=request.timeline_months_ahead,
    )
    if not universe_uses_extended_runtime(universe, request):
        state = execute_simulation(
            params,
            seed=seed,
            include_timeline=request.include_timeline,
            official_dreamerv3_checkpoint=(
                selected_run["checkpoint_input"] if selected_run is not None else None
            ),
        )
        report = build_report(state)
        latest_centers = build_snapshot(state, 1)["centers"]
        return {
            "seed": seed,
            "report": report,
            "timeline": [],
            "agent_timeline": [],
            "runtime_events": [],
            "scheduled_events": [],
            "calibration_notes": calibration_notes,
            "latest_centers": latest_centers,
            "execution_mode": "classic_parity",
            "dreamerv3_run": (
                public_dreamerv3_run_payload(selected_run)
                if selected_run is not None and universe.strategy_key == "dreamerv3_official"
                else None
            ),
        }

    state = initialize_simulation_from_params(
        params,
        seed=seed,
        step_hours=request.step_hours,
        dreamerv3_run_key=request.dreamerv3_run_key,
    )
    scheduled_events = universe_replay_schedule(
        universe,
        request,
        snapshot,
        horizon_hours=float(params.sim_hours),
    )
    runtime = RuntimeAdjustmentController(state, params, scheduled_events)
    runtime.advance_to(0.0)

    agent_runner = (
        DonorAgentCoordinator(
            snapshot.donor_records,
            request.agent_mode,
            policy_strength=universe_policy_strength(universe),
        )
        if request.agent_mode.enabled
        else None
    )
    timeline: list[dict[str, Any]] = []
    step_events: list[dict[str, Any]] = []
    agent_timeline: list[dict[str, Any]] = []

    prev_donated = sum(center.stats["donated"] for center in state.centers)
    prev_shortage = state.total_shortage_units
    step_number = 0

    while state.env.now < params.sim_hours:
        if agent_runner is not None:
            agent_data = agent_runner.adjustments_for_hour(
                hour=float(state.env.now),
                horizon_hours=float(params.sim_hours),
            )
            runtime.set_agent_adjustments(agent_data["parameter_multipliers"])
            agent_timeline.append(
                {
                    "hour": round(float(state.env.now), 2),
                    **agent_data["summary"],
                }
            )

        remaining = float(params.sim_hours - state.env.now)
        if remaining <= 0:
            break
        target_hour = float(state.env.now + min(request.step_hours, remaining))
        checkpoints = {
            target_hour,
            *[
                event.start_hour
                for event in runtime.pending_events
                if state.env.now < event.start_hour < target_hour
            ],
            *[
                event.end_hour
                for event in runtime.active_events
                if state.env.now < event.end_hour < target_hour
            ],
        }
        for checkpoint in sorted(checkpoints):
            delta = checkpoint - float(state.env.now)
            if delta > 1e-6:
                step_simulation(state, delta)
            step_events.extend(runtime.advance_to(checkpoint))

        step_number += 1
        snapshot_row = build_snapshot(
            state,
            step_number,
            prev_donated=prev_donated,
            prev_shortage=prev_shortage,
        )
        snapshot_row["universe_key"] = universe.key
        snapshot_row["active_replay_events"] = [
            event.as_dict() for event in runtime.active_events
        ]
        if agent_timeline:
            snapshot_row["agent_mode"] = agent_timeline[-1]
        timeline.append(snapshot_row)

        prev_donated = sum(center.stats["donated"] for center in state.centers)
        prev_shortage = state.total_shortage_units

    report = build_report(state)
    return {
        "seed": seed,
        "report": report,
        "timeline": timeline if request.include_timeline else [],
        "agent_timeline": agent_timeline,
        "runtime_events": step_events,
        "scheduled_events": [event.as_dict() for event in scheduled_events],
        "calibration_notes": calibration_notes,
        "latest_centers": timeline[-1]["centers"] if timeline else [],
        "execution_mode": "extended_runtime",
        "dreamerv3_run": (
            public_dreamerv3_run_payload(selected_run)
            if selected_run is not None and universe.strategy_key == "dreamerv3_official"
            else None
        ),
    }


def branch_summary(
    universe: UniversePolicyInput,
    reports: list[dict[str, Any]],
    confidence_bands: list[dict[str, Any]],
    snapshot: SimulationSnapshot,
) -> dict[str, Any]:
    aggregate_summary = aggregate_report_summaries(reports)
    score = retention_score(aggregate_summary)
    return {
        "key": universe.key,
        "label": universe.label,
        "description": universe.description,
        "strategy": strategy_payload(universe.strategy_key),
        "summary": aggregate_summary,
        "retention_score": score,
        "confidence_bands": confidence_bands,
        "map_layers": {
            "donor_heatmap": donor_heatmap(snapshot.donor_records),
            "event_count": snapshot.donor_summary.get("critical_event_count", 0),
        },
    }


def build_mission_control(
    universe_results: list[dict[str, Any]],
    snapshot: SimulationSnapshot,
) -> dict[str, Any]:
    return {
        "generated_at": utc_now_iso(),
        "branches": [
            {
                "key": result["key"],
                "label": result["label"],
                "retention_score": round(float(result["retention_score"]), 4),
                "strategy_name": result["strategy"]["name"],
                "summary": result["summary"],
                "primary_metric": {
                    "key": "shortage_rate_mean",
                    "label": "Shortage Rate",
                    "value": round(
                        float(result["summary"].get("shortage_rate_mean", 0.0)), 4
                    ),
                },
                "shortage_rate_mean": round(
                    float(result["summary"].get("shortage_rate_mean", 0.0)), 4
                ),
                "service_rate_mean": round(
                    float(result["summary"].get("service_rate_mean", 0.0)), 4
                ),
                "episode_score_mean": round(
                    float(result["summary"].get("episode_score_mean", 0.0)), 4
                ),
                "budget_spent_mean": round(
                    float(result["summary"].get("budget_spent_mean", 0.0)), 4
                ),
                "budget_remaining_mean": round(
                    float(result["summary"].get("budget_remaining_mean", 0.0)), 4
                ),
            }
            for result in universe_results
        ],
        "confidence_bands": {
            result["key"]: result["confidence_bands"] for result in universe_results
        },
        "map_layers": {
            result["key"]: {
                "centers": result.get("latest_centers", []),
                "donor_heatmap": donor_heatmap(snapshot.donor_records),
            }
            for result in universe_results
        },
    }


def run_studio_experiment(request: StudioExperimentRequest) -> dict[str, Any]:
    snapshot = capture_data_snapshot(request.snapshot)
    universes = experiment_universes(request)

    results: list[dict[str, Any]] = []
    for universe in universes:
        replications: list[dict[str, Any]] = []
        reports: list[dict[str, Any]] = []
        for replication_index in range(request.replications):
            seed = request.seed + replication_index
            replication = run_universe_replication(
                request.scenario_key,
                snapshot,
                universe,
                request,
                seed=seed,
            )
            replications.append(replication)
            reports.append(replication["report"])
        confidence_bands = build_confidence_bands(replications)
        branch = branch_summary(universe, reports, confidence_bands, snapshot)
        branch["replications"] = replications
        branch["latest_centers"] = replications[0]["latest_centers"] if replications else []
        results.append(branch)

    results.sort(key=shortage_sort_tuple)
    return {
        "generated_at": utc_now_iso(),
        "scenario": scenario_payload(
            request.scenario_key,
            resolve_scenario(request.scenario_key),
            source="custom" if request.scenario_key not in SCENARIOS else "built_in",
            default_strategy_key=results[0]["strategy"]["key"] if results else "baseline",
        ),
        "timeline_months_ahead": request.timeline_months_ahead,
        "snapshot": snapshot.as_dict(),
        "universes": results,
        "mission_control": build_mission_control(results, snapshot),
    }


def merge_universe_with_dropout_shock(
    universe: UniversePolicyInput,
    dropout_rise_pct: float,
) -> UniversePolicyInput:
    donor_show_drop = clamp(1.0 - (dropout_rise_pct / 100.0), 0.35, 1.0)
    arrival_penalty = 1.0 + (dropout_rise_pct / 140.0)
    payload = universe.model_dump()
    payload["parameter_multipliers"] = merge_multiplier_dict(
        payload.get("parameter_multipliers", {}),
        {
            "donor_show_factor": donor_show_drop,
            "donor_inter_arrival_h": arrival_penalty,
        },
    )
    payload["description"] = (
        f"{universe.description} Dropout shock of {dropout_rise_pct:.1f}% applied."
    ).strip()
    return UniversePolicyInput(**payload)


def run_strategy_assistant(request: StrategyAssistantRequest) -> dict[str, Any]:
    universes = unique_universe_list(
        request.candidate_universes or default_assistant_universes()
    )
    shocked_universes = [
        merge_universe_with_dropout_shock(universe, request.dropout_rise_pct)
        for universe in universes
    ]
    experiment = run_studio_experiment(
        StudioExperimentRequest(
            scenario_key=request.scenario_key,
            seed=request.seed,
            replications=request.replications,
            hours_override=request.hours_override,
            step_hours=6.0,
            timeline_months_ahead=request.timeline_months_ahead,
            include_timeline=False,
            snapshot=request.snapshot,
            universes=shocked_universes,
            replay=ReplayConfig(mode="none"),
            agent_mode=AgentModeConfig(enabled=True),
            dreamerv3_run_key=request.dreamerv3_run_key,
        )
    )
    rankings = [
        {
            "key": universe["key"],
            "label": universe["label"],
            "strategy_key": universe["strategy"]["key"],
            "strategy_name": universe["strategy"]["name"],
            "summary": universe["summary"],
            "retention_score": round(float(universe["retention_score"]), 4),
            "shortage_rate_mean": round(
                float(universe["summary"].get("shortage_rate_mean", 0.0)), 4
            ),
            "service_rate_mean": round(
                float(universe["summary"].get("service_rate_mean", 0.0)), 4
            ),
            "episode_score_mean": round(
                float(universe["summary"].get("episode_score_mean", 0.0)), 4
            ),
            "total_donated_mean": round(
                float(universe["summary"].get("total_donated_mean", 0.0)), 4
            ),
            "budget_spent_mean": round(
                float(universe["summary"].get("budget_spent_mean", 0.0)), 4
            ),
            "budget_remaining_mean": round(
                float(universe["summary"].get("budget_remaining_mean", 0.0)), 4
            ),
        }
        for universe in experiment["universes"]
    ]
    best = rankings[0] if rankings else None
    return {
        "generated_at": utc_now_iso(),
        "question": (
            f"If donor dropout rises by {request.dropout_rise_pct:.1f}%, which intervention gives the best retention?"
        ),
        "recommendation": {
            "best_universe": best,
            "reasoning": (
                "The recommended branch is the one with the lowest projected shortage rate after replaying the dropout shock across the same donor snapshot. Episode score, reward total, and service rate are used as tie-breakers."
                if best is not None
                else "No valid strategy recommendation could be produced."
            ),
        },
        "rankings": rankings,
        "snapshot_summary": experiment["snapshot"]["donor_summary"],
        "experiment_summary": [
            {
                "key": universe["key"],
                "strategy_name": universe["strategy"]["name"],
                "retention_score": round(float(universe["retention_score"]), 4),
                "shortage_rate_mean": round(
                    float(universe["summary"].get("shortage_rate_mean", 0.0)), 4
                ),
                "episode_score_mean": round(
                    float(universe["summary"].get("episode_score_mean", 0.0)), 4
                ),
            }
            for universe in experiment["universes"]
        ],
    }
