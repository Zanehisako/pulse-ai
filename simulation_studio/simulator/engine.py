"""
engine.py - Simulation processes, runner and report generation.

Changes vs. original
--------------------
* `ppo_shortage_controller` now drives action selection through a **trained**
  PPO agent (``ppo_agent.PPO``) instead of hand-crafted logit weights.
* New helpers:
    - ``train_ppo_agent()``   – train a PPO agent against the simulation env
    - ``load_ppo_agent()``    – re-load a previously trained agent
    - ``run_with_ppo_agent()``– run a scenario using a trained/loaded agent

* The original heuristic controller weights (``PPO_POLICY_WEIGHTS``) and the
  ``softmax_probabilities`` / ``activate_action`` helpers are kept for
  backward compatibility but are no longer called when a trained agent is
  available.

Usage (quick-start)
-------------------
    from engine import train_ppo_agent, run_with_ppo_agent

    # 1. Train
    agent = train_ppo_agent(params, G, north, south, east, west,
                            total_timesteps=100_000, save_path="ppo_model")

    # 2. Run a full episode with the trained agent
    state = run_with_ppo_agent(params, G, north, south, east, west,
                               agent=agent, seed=99)

    # 3. Or load a saved agent and run
    state = run_with_ppo_agent(params, G, north, south, east, west,
                               model_path="ppo_model", seed=99)
"""

from __future__ import annotations

import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import simpy
from calibration import (
    COMPONENT_DEMAND_WEIGHTS,
    QC_BASE_DEMAND_INTERARRIVAL_H,
    QC_BASE_DONOR_INTERARRIVAL_H,
    QC_DAILY_LABILE_PRODUCTS_EST,
)
from core import (
    BLOOD_DISTRIBUTION,
    BLOOD_TYPES,
    CENTER_CONFIGS,
    STEP_TIMES,
    BloodUnit,
    DonationCenter,
    Person,
    WeatherEngine,
    compatible_donor_types,
    demand_surge_multiplier,
    donation_components,
    donor_show_probability,
    hour_in_day,
    is_holiday,
    is_weekend,
    road_travel_hours,
    sim_datetime,
)

# ---------------------------------------------------------------------------
# PPO imports
# ---------------------------------------------------------------------------
# ppo_agent has no dependency on engine so a top-level import is fine.
from ppo_agent import PPO  # NumPy fallback or SB3

# ppo_shared has no dependency on engine either – safe at module level.
from ppo_shared import (  # type: ignore
    ACTION_KEYS,
    COMPONENT_ALLOCATION_CONTROL_MAP,
    DREAMER_ACTION_DIM,
    HOSPITAL_ROUTE_CONTROL_MAP,
    N_ACTIONS,
)
from ppo_shared import (
    PPO_WEATHER_SEVERITY as _PPO_WEATHER_SEVERITY_SHARED,
)
from ppo_shared import (
    build_observation as _build_observation_shared,
)
from ppo_shared import (
    clamp_obs as _clamp_obs,
)
from ppo_shared import (
    continuous_action_levels as decode_continuous_action_levels,
)
from scenarios import (
    ACTION_CATALOG,
    REAL_WORLD_ASSUMPTIONS_TEXT,
    Action,
    ScenarioParams,
)

DREAMER_V3_CONTROLLER_KEY = "dreamerv3_official"
DREAMER_V4_CONTROLLER_KEY = "dreamerv4"
PPO_CONTINUOUS_CONTROLLER_KEY = "ppo_continuous"
SAC_CONTROLLER_KEY = "sac_continuous"
IQL_CONTROLLER_KEY = "iql_offline"
CQL_CONTROLLER_KEY = "cql_offline"
DREAMER_CONTROLLER_KEYS = frozenset(
    {DREAMER_V3_CONTROLLER_KEY, DREAMER_V4_CONTROLLER_KEY}
)
LEARNED_CONTINUOUS_CONTROLLER_KEYS = frozenset(
    {
        PPO_CONTINUOUS_CONTROLLER_KEY,
        SAC_CONTROLLER_KEY,
        IQL_CONTROLLER_KEY,
        CQL_CONTROLLER_KEY,
    }
)
DREAMERV4_CHECKPOINT_CANDIDATE_NAMES = ("dreamerv4_agent.pt", "model.pt")
DREAMER_CHECKPOINT_ENV_VARS = {
    DREAMER_V3_CONTROLLER_KEY: "PIOS_DREAMERV3_CHECKPOINT",
    DREAMER_V4_CONTROLLER_KEY: "PIOS_DREAMERV4_CHECKPOINT",
}
LEARNED_CONTINUOUS_CHECKPOINT_ENV_VARS = {
    PPO_CONTINUOUS_CONTROLLER_KEY: "PIOS_PPO_CONTINUOUS_CHECKPOINT",
    SAC_CONTROLLER_KEY: "PIOS_SAC_CHECKPOINT",
    IQL_CONTROLLER_KEY: "PIOS_IQL_CHECKPOINT",
    CQL_CONTROLLER_KEY: "PIOS_CQL_CHECKPOINT",
}

# ppo_env IS NOT imported at module level (it would circular-import engine).
# Instead every function that needs BloodSupplyEnv / build_sim does:
#   from ppo_env import BloodSupplyEnv, build_sim   (lazy, inside function)


# ===========================================================================
# Data-classes & SimState (unchanged from original)
# ===========================================================================


@dataclass
class RewardBreakdown:
    total: float
    terms: dict[str, float]


@dataclass
class ActivatedAction:
    action: Action
    activated_at_h: float = 0.0
    intensity: float = 1.0
    persistent: bool = False


class SimState:
    def __init__(
        self,
        env,
        params: ScenarioParams,
        centers: list[DonationCenter],
        G,
        north,
        south,
        east,
        west,
        weather: WeatherEngine,
        rng_demand: Optional[random.Random] = None,
        rng_process: Optional[random.Random] = None,
        rng_transport: Optional[random.Random] = None,
        enable_logs: bool = True,
        fast_mode: bool = False,
    ):
        self.env = env
        self.params = params
        self.centers = centers
        self.G = G
        self.north = north
        self.south = south
        self.east = east
        self.west = west
        self.weather = weather
        self.rng_demand = rng_demand or random.Random()
        self.rng_process = rng_process or random.Random()
        self.rng_transport = rng_transport or random.Random()
        self.enable_logs = enable_logs
        self.fast_mode = fast_mode
        self.strategy = params.strategy
        self.record_history = not fast_mode
        self.active_actions: list[ActivatedAction] = [
            ActivatedAction(action=_timing_scaled(action, self.params), activated_at_h=0.0)
            for action in self.strategy.actions
        ]
        self.log: list[str] = []
        self.hourly_inventory: list[dict] = []
        self.shortage_log: list[dict] = []
        self.donation_log: list[dict] = []
        self.demand_log: list[dict] = []
        self.transfer_log: list[dict] = []
        self.replenishment_log: list[dict] = []
        self.conservation_log: list[dict] = []
        self.policy_log: list[dict] = []
        self.forecast_log: list[dict] = []
        self.shipment_log: list[dict] = []
        self.budget_log: list[dict] = []
        self.travel_log: list[float] = []
        self.shortage_by_component = Counter()
        self.shortage_by_hospital = Counter()
        self.total_shortage_units = 0
        self.total_transfer_units = 0
        self.total_external_units = 0
        self.total_conserved_units = 0
        self.total_base_requested_units = 0
        self.total_net_requested_units = 0
        self.total_exact_match_units = 0
        self.total_compatible_substitution_units = 0
        self.total_incompatible_fulfillment_units = 0
        self.priority_weighted_requested_units = 0.0
        self.priority_weighted_shortage_units = 0.0
        self.travel_time_total = 0.0
        self.travel_time_count = 0
        self.wait_time_total = 0.0
        self.wait_time_count = 0
        self.demand_pressure = max(
            0.85, 1.0 + 0.35 * (params.demand_surge_factor - 1.0)
        )
        self.forecast_pressure = self.demand_pressure
        self.forecast_priority_pressure = 0.0
        self.recent_priority_pressure = 0.0
        self.transport_backlog = 0.0
        self.transport_backlog_updated_at = 0.0
        self.budget_total = max(float(params.episode_budget), 0.0)
        self.budget_remaining = self.budget_total
        self.budget_spent = 0.0
        self.budget_exhausted_hours = 0.0
        self.budget_blocked_controls = 0
        self.reward_breakdown: Optional[RewardBreakdown] = None
        self.runtime_controller: Optional[str] = None
        self.runtime_controller_step_hours: Optional[float] = None
        self.episode_score: float = 0.0
        self.episode_reward_deltas: list[float] = []
        self.episode_step_hours: Optional[float] = None
        self.episode_prev_reward_total: Optional[float] = None
        self.episode_prev_transfused: int = 0
        self.continuous_hospital_routing: dict[str, float] = {
            hospital_name: 0.0 for hospital_name in HOSPITAL_ROUTE_CONTROL_MAP.values()
        }
        self.continuous_component_allocation: dict[str, float] = {
            component: 0.0 for component in COMPONENT_ALLOCATION_CONTROL_MAP.values()
        }

    def emit(self, msg: str):
        if not self.enable_logs:
            return
        self.log.append(msg)
        print(msg)

    @property
    def hospitals(self) -> list[DonationCenter]:
        return [center for center in self.centers if center.ctype == "hospital"]

    def nearest_open_center(self, lat, lon):
        open_centers = [
            center
            for center in self.centers
            if is_center_open_for_state(self, center, self.env.now)
        ]
        if not open_centers:
            open_centers = self.centers
        return min(
            open_centers,
            key=lambda center: math.hypot(center.lat - lat, center.lon - lon),
        )


# ===========================================================================
# Constants & helpers (unchanged)
# ===========================================================================

COMPONENT_BUFFER_MULTIPLIER = {"RBC": 1.0, "PLATELETS": 1.35, "PLASMA": 0.85}
REPLENISHMENT_WEATHER_FACTOR = {
    "clear": 1.0,
    "cloudy": 0.97,
    "snow": 0.88,
    "ice_storm": 0.72,
    "blizzard": 0.58,
}
ORDER_PRIORITY_PROFILES = {
    "routine": {"weight": 1.0, "reserve_relief": 0, "delay_factor": 1.05},
    "urgent": {"weight": 2.0, "reserve_relief": 1, "delay_factor": 0.90},
    "emergency": {"weight": 4.0, "reserve_relief": 2, "delay_factor": 0.72},
}


def prestock_centers(
    centers: list[DonationCenter], env, inventory_days: float = 3.0
) -> None:
    total_units = QC_DAILY_LABILE_PRODUCTS_EST * inventory_days
    center_weights = {"hospital": 1.5, "blood_bank": 1.0, "mobile": 0.5}
    total_weight = sum(center_weights.get(center.ctype, 1.0) for center in centers)

    for center in centers:
        center_units = (
            total_units * center_weights.get(center.ctype, 1.0) / total_weight
        )
        for component, component_weight in COMPONENT_DEMAND_WEIGHTS.items():
            component_units = max(
                1,
                round(
                    center_units
                    * component_weight
                    * COMPONENT_BUFFER_MULTIPLIER.get(component, 1.0)
                ),
            )
            remaining = component_units
            for idx, (blood_type, blood_weight) in enumerate(
                BLOOD_DISTRIBUTION.items()
            ):
                if idx == len(BLOOD_DISTRIBUTION) - 1:
                    n_units = max(0, remaining)
                else:
                    n_units = round(component_units * blood_weight)
                    remaining -= n_units
                for _ in range(max(0, n_units)):
                    unit = BloodUnit(env, blood_type, component, center.name)
                    shelf_life = unit.expiry - env.now
                    age_fraction = 0.25 if component == "RBC" else 0.18
                    unit.collection_time = -random.uniform(0, shelf_life * age_fraction)
                    unit.expiry = unit.collection_time + shelf_life
                    center.add_unit(unit)


def city_inventory(state: SimState) -> dict[str, int]:
    totals = {"RBC": 0, "PLATELETS": 0, "PLASMA": 0}
    for center in state.centers:
        for component, units in center.inventory_by_component().items():
            totals[component] += units
    return totals


def initialize_forecast_state(state: SimState) -> None:
    refresh_demand_forecast(state, initial=True)


def refresh_demand_forecast(state: SimState, *, initial: bool = False) -> None:
    baseline_pressure = 1.0 + 0.55 * max(state.params.demand_surge_factor - 1.0, 0.0)
    weather_pressure = 0.08 + (0.15 * PPO_WEATHER_SEVERITY[state.weather.state])
    shock = state.rng_process.gauss(0.0, state.params.demand_shock_scale)
    if initial:
        state.demand_pressure = np.clip(
            baseline_pressure + (0.30 * weather_pressure),
            0.75,
            2.50,
        )
    else:
        blended_target = baseline_pressure + weather_pressure + shock
        state.demand_pressure = np.clip(
            (0.72 * state.demand_pressure) + (0.28 * blended_target),
            0.75,
            2.50,
        )

    forecast_noise = state.rng_process.gauss(0.0, state.params.demand_forecast_noise)
    state.forecast_pressure = float(
        np.clip(state.demand_pressure + forecast_noise, 0.65, 2.60)
    )
    priority_signal = 0.10 + (0.30 * max(state.demand_pressure - 1.0, 0.0))
    if state.weather.state in {"ice_storm", "blizzard"}:
        priority_signal += 0.08
    state.forecast_priority_pressure = float(
        np.clip(priority_signal + (0.25 * forecast_noise), 0.0, 1.0)
    )
    if state.record_history:
        state.forecast_log.append(
            {
                "time": state.env.now,
                "demand_pressure": round(float(state.demand_pressure), 4),
                "forecast_pressure": round(float(state.forecast_pressure), 4),
                "priority_pressure": round(float(state.forecast_priority_pressure), 4),
            }
        )


def demand_forecast_process(state: SimState):
    env = state.env
    interval_h = max(state.params.demand_forecast_interval_h, 1.0)
    while True:
        yield env.timeout(interval_h)
        refresh_demand_forecast(state)


def update_transport_backlog(state: SimState) -> None:
    elapsed_h = max(state.env.now - state.transport_backlog_updated_at, 0.0)
    if elapsed_h <= 0:
        return
    state.transport_backlog *= math.exp(-elapsed_h / 12.0)
    state.transport_backlog_updated_at = state.env.now


def register_transport_load(
    state: SimState, units: float, *, lane_weight: float = 1.0
) -> None:
    update_transport_backlog(state)
    state.transport_backlog += max(float(units), 0.0) * max(float(lane_weight), 0.0)


def congestion_stress(state: SimState) -> float:
    update_transport_backlog(state)
    overload = max(
        state.transport_backlog - state.params.congestion_threshold_units, 0.0
    )
    return float(overload * state.params.congestion_delay_factor)


def transport_multiplier(state: SimState) -> float:
    return effective_transport_penalty(state) * (1.0 + congestion_stress(state))


def budget_available(state: SimState) -> bool:
    return state.budget_total < 0.0 or state.budget_remaining > 1e-6


def budget_pressure(state: SimState) -> float:
    if state.budget_total < 0.0:
        return 0.0
    if state.budget_total == 0.0:
        return 1.0
    return float(np.clip(1.0 - (state.budget_remaining / state.budget_total), 0.0, 1.0))


def budgeted_action_level(state: SimState, action: ActivatedAction) -> float:
    if not budget_available(state):
        return 0.0
    return effective_action_level(action, state.env.now)


def budget_monitor(state: SimState, *, interval_h: float = 6.0):
    env = state.env
    last_h = env.now
    while True:
        remaining_h = float(state.params.sim_hours - env.now)
        if remaining_h <= 0:
            break
        step_h = min(interval_h, remaining_h)
        yield env.timeout(step_h)
        elapsed_h = max(env.now - last_h, 0.0)
        last_h = env.now

        if state.budget_total < 0.0:
            continue

        current_cost = active_action_cost(state)
        spend = current_cost * (elapsed_h / 24.0)
        if spend > 0.0 and state.budget_remaining > 0.0:
            actual_spend = min(spend, state.budget_remaining)
            state.budget_remaining = max(state.budget_remaining - spend, 0.0)
            state.budget_spent += actual_spend
            if state.record_history:
                state.budget_log.append(
                    {
                        "time": env.now,
                        "active_cost": round(current_cost, 4),
                        "spent": round(actual_spend, 4),
                        "remaining": round(state.budget_remaining, 4),
                    }
                )

        if state.budget_remaining <= 0.0:
            state.budget_exhausted_hours += elapsed_h


def priority_profile(priority: str) -> dict[str, float]:
    return ORDER_PRIORITY_PROFILES.get(priority, ORDER_PRIORITY_PROFILES["routine"])


def replenishment_target_units(params: ScenarioParams, component: str) -> float:
    return (
        QC_DAILY_LABILE_PRODUCTS_EST
        * COMPONENT_DEMAND_WEIGHTS[component]
        * params.reserve_target_days
        * COMPONENT_BUFFER_MULTIPLIER.get(component, 1.0)
    )


def _timing_scaled(action, params):
    """Return ``action`` with its activation-delay and ramp-up scaled by the
    scenario's ``delay_scale``/``ramp_scale``. Both default to 1.0, in which
    case the original (frozen) action is returned unchanged so canonical runs
    and validation are byte-identical. Used only by the sensitivity sweep."""
    ds = getattr(params, "delay_scale", 1.0)
    rs = getattr(params, "ramp_scale", 1.0)
    if ds == 1.0 and rs == 1.0:
        return action
    return replace(
        action,
        activation_delay_h=action.activation_delay_h * ds,
        ramp_h=action.ramp_h * rs,
    )


def action_progress(action, env_now: float) -> float:
    elapsed_h = max(env_now - action.activated_at_h, 0.0)
    if elapsed_h <= action.action.activation_delay_h:
        return 0.0
    if action.action.ramp_h <= 0:
        return 1.0
    return min(
        (elapsed_h - action.action.activation_delay_h) / action.action.ramp_h,
        1.0,
    )


def action_effectiveness(action, env_now: float) -> float:
    progress = action_progress(action, env_now)
    if action.persistent:
        return progress

    active_h = action.action.active_h
    if active_h is None or active_h <= 0:
        return progress

    elapsed_h = max(env_now - action.activated_at_h, 0.0)
    peak_start_h = action.action.activation_delay_h + max(action.action.ramp_h, 0.0)
    active_end_h = peak_start_h + active_h
    if elapsed_h <= active_end_h:
        return progress

    decay_h = max(action.action.ramp_h * 0.5, 12.0)
    fade = max(1.0 - (elapsed_h - active_end_h) / decay_h, 0.0)
    return progress * fade


def is_action_expired(action, env_now: float) -> bool:
    if action.persistent:
        return False

    active_h = action.action.active_h
    if active_h is None or active_h <= 0:
        return False

    elapsed_h = max(env_now - action.activated_at_h, 0.0)
    peak_start_h = action.action.activation_delay_h + max(action.action.ramp_h, 0.0)
    decay_h = max(action.action.ramp_h * 0.5, 12.0)
    return elapsed_h > peak_start_h + active_h + decay_h


def prune_expired_actions(state: SimState) -> None:
    state.active_actions = [
        activated
        for activated in state.active_actions
        if not is_action_expired(activated, state.env.now)
    ]


def effective_action_level(action: ActivatedAction, env_now: float) -> float:
    level = max(min(float(action.intensity), 1.0), 0.0)
    return level * action_effectiveness(action, env_now)


def dynamic_strategy_factor(state: SimState, attr: str) -> float:
    prune_expired_actions(state)
    factor = 1.0
    for activated_action in state.active_actions:
        target = getattr(activated_action.action, attr)
        level = budgeted_action_level(state, activated_action)
        factor *= 1.0 + ((target - 1.0) * level)
    return factor


def dynamic_strategy_sum(state: SimState, attr: str) -> float:
    prune_expired_actions(state)
    total = 0.0
    for activated_action in state.active_actions:
        total += getattr(activated_action.action, attr) * budgeted_action_level(
            state, activated_action
        )
    return total


def active_action_cost(state: SimState) -> float:
    prune_expired_actions(state)
    total = 0.0
    for activated_action in state.active_actions:
        level = budgeted_action_level(state, activated_action)
        if level <= 0:
            continue
        if not activated_action.persistent:
            level = max(level, 0.15)
        total += activated_action.action.operational_cost * level
    return total


def strongest_action_level(
    state: SimState,
    predicate,
) -> float:
    prune_expired_actions(state)
    levels = [
        budgeted_action_level(state, activated_action)
        for activated_action in state.active_actions
        if predicate(activated_action.action)
    ]
    return max(levels, default=0.0)


def active_emergency_share(state: SimState) -> bool:
    return strongest_action_level(state, lambda action: action.emergency_share) > 0.05


def active_hours_extension_h(state: SimState) -> float:
    return dynamic_strategy_sum(state, "hours_extension_h")


def is_center_open_for_state(
    state: SimState,
    center: DonationCenter,
    env_now: float,
) -> bool:
    extension = active_hours_extension_h(state) if center.ctype != "hospital" else 0.0
    h = hour_in_day(env_now)
    if center.close_h == 24:
        return True
    start_h = max(center.open_h - extension, 0.0)
    end_h = min(center.close_h + extension, 24.0)
    return start_h <= h < end_h


def active_reserve_threshold(state: SimState) -> int:
    prune_expired_actions(state)
    thresholds = [
        (
            activated_action.action.reserve_threshold,
            budgeted_action_level(state, activated_action),
        )
        for activated_action in state.active_actions
        if activated_action.action.reserve_threshold is not None
        and budgeted_action_level(state, activated_action) > 0.05
    ]
    if not thresholds:
        return 4
    target_threshold = min(threshold for threshold, _level in thresholds)
    level = max(level for _threshold, level in thresholds)
    return int(round((4.0 * (1.0 - level)) + (target_threshold * level)))


def active_replenishment_interval_h(state: SimState) -> float:
    prune_expired_actions(state)
    intervals = [
        (
            activated_action.action.replenishment_interval_h,
            budgeted_action_level(state, activated_action),
        )
        for activated_action in state.active_actions
        if activated_action.action.replenishment_interval_h is not None
        and budgeted_action_level(state, activated_action) > 0.05
    ]
    if not intervals:
        return 12.0
    target_interval = min(interval for interval, _level in intervals)
    level = max(level for _interval, level in intervals)
    return (12.0 * (1.0 - level)) + (target_interval * level)


def active_mobile_release_delay_h(state: SimState) -> float:
    prune_expired_actions(state)
    return sum(
        activated_action.action.mobile_release_delay_h
        * budgeted_action_level(state, activated_action)
        for activated_action in state.active_actions
    )


def effective_transport_penalty(state: SimState) -> float:
    return max(
        state.params.transport_penalty
        * dynamic_strategy_factor(state, "transport_relief_factor"),
        1.0,
    )


def _normalize_control_weights(
    raw_values: dict[str, float],
    baseline_weights: dict[str, float],
) -> dict[str, float]:
    keys = list(baseline_weights.keys())
    raw = np.asarray(
        [max(float(raw_values.get(key, 0.0)), 0.0) for key in keys],
        dtype=np.float32,
    )
    if raw.sum() <= 1e-6:
        weights = np.asarray([baseline_weights[key] for key in keys], dtype=np.float32)
    else:
        weights = raw
    weights = weights / max(float(weights.sum()), 1e-6)
    return {
        key: float(weight) for key, weight in zip(keys, weights.tolist(), strict=False)
    }


def hospital_routing_shares(state: SimState) -> dict[str, float]:
    hospitals = [hospital.name for hospital in state.hospitals]
    if not hospitals:
        return {}
    baseline = {hospital_name: 1.0 for hospital_name in hospitals}
    return _normalize_control_weights(state.continuous_hospital_routing, baseline)


def hospital_priority_bias(state: SimState, hospital_name: str) -> float:
    shares = hospital_routing_shares(state)
    if hospital_name not in shares or not shares:
        return 0.0
    baseline_share = 1.0 / max(len(shares), 1)
    bias = (shares[hospital_name] - baseline_share) / max(baseline_share, 1e-6)
    return float(np.clip(bias, -1.0, 1.0))


def component_allocation_shares(state: SimState) -> dict[str, float]:
    baseline = {
        component: COMPONENT_DEMAND_WEIGHTS[component]
        * COMPONENT_BUFFER_MULTIPLIER.get(component, 1.0)
        for component in ["RBC", "PLATELETS", "PLASMA"]
    }
    return _normalize_control_weights(state.continuous_component_allocation, baseline)


def replenishment_receivers(state: SimState, component: str) -> list[DonationCenter]:
    eligible = [center for center in state.centers if center.ctype != "mobile"]
    routing_shares = hospital_routing_shares(state)
    return sorted(
        eligible,
        key=lambda center: (
            0 if center.ctype == "hospital" else 1,
            -routing_shares.get(center.name, 0.0)
            if center.ctype == "hospital"
            else 0.0,
            center.inventory_by_component().get(component, 0),
            len(center.inventory),
        ),
    )


def add_external_units(
    state: SimState, component: str, units: int, source: str = "provincial_network"
) -> int:
    if units <= 0:
        return 0
    receivers = replenishment_receivers(state, component)
    if not receivers:
        return 0

    added = 0
    for idx in range(units):
        blood_type = random.choices(
            BLOOD_TYPES, weights=list(BLOOD_DISTRIBUTION.values())
        )[0]
        center = receivers[idx % len(receivers)]
        center.add_unit(BloodUnit(state.env, blood_type, component, source))
        added += 1
    return added


def shipment_delay_hours(
    state: SimState,
    units: int,
    *,
    lane: str,
    priority: str = "routine",
) -> float:
    units = max(int(units), 1)
    profile = priority_profile(priority)
    noise = max(
        state.rng_transport.gauss(1.0, state.params.transport_lead_time_noise),
        0.35,
    )
    base_delay_h = 0.35 + (0.05 * units)
    if lane == "replenishment":
        base_delay_h += 1.10 + (0.04 * units)
        lane_weight = 0.75
    else:
        lane_weight = 1.0
    register_transport_load(state, units, lane_weight=lane_weight)
    weather_factor = 1.0 + (0.55 * PPO_WEATHER_SEVERITY[state.weather.state])
    return (
        base_delay_h
        * transport_multiplier(state)
        * weather_factor
        * noise
        * profile["delay_factor"]
    )


def _deliver_external_units(
    state: SimState,
    receiver: DonationCenter,
    component: str,
    blood_types: list[str],
    *,
    source: str,
):
    delay_h = shipment_delay_hours(
        state,
        len(blood_types),
        lane="replenishment",
        priority="urgent",
    )
    yield state.env.timeout(delay_h)
    for blood_type in blood_types:
        receiver.add_unit(BloodUnit(state.env, blood_type, component, source))
    state.total_external_units += len(blood_types)
    if state.record_history:
        state.shipment_log.append(
            {
                "time": state.env.now,
                "kind": "arrival",
                "to": receiver.name,
                "component": component,
                "units": len(blood_types),
                "source": source,
            }
        )


def dispatch_external_units(
    state: SimState,
    component: str,
    units: int,
    *,
    source: str = "provincial_network",
) -> int:
    if units <= 0:
        return 0
    receivers = replenishment_receivers(state, component)
    if not receivers:
        return 0

    grouped_shipments: dict[str, list[str]] = {}
    receiver_lookup = {center.name: center for center in receivers}
    for idx in range(units):
        blood_type = random.choices(
            BLOOD_TYPES, weights=list(BLOOD_DISTRIBUTION.values())
        )[0]
        receiver = receivers[idx % len(receivers)]
        grouped_shipments.setdefault(receiver.name, []).append(blood_type)

    for receiver_name, blood_types in grouped_shipments.items():
        state.env.process(
            _deliver_external_units(
                state,
                receiver_lookup[receiver_name],
                component,
                blood_types,
                source=source,
            )
        )
        if state.record_history:
            state.shipment_log.append(
                {
                    "time": state.env.now,
                    "kind": "dispatch",
                    "to": receiver_name,
                    "component": component,
                    "units": len(blood_types),
                    "source": source,
                }
            )

    return units


def regional_replenishment(state: SimState):
    env = state.env
    params = state.params
    while True:
        cycle_h = active_replenishment_interval_h(state)
        cycle_fraction = cycle_h / 24.0
        yield env.timeout(cycle_h)
        inventory = city_inventory(state)
        cycle_units = 0
        cycle_components: dict[str, int] = {}
        inbound_rate_factor = REPLENISHMENT_WEATHER_FACTOR[state.weather.state]
        inbound_rate_factor /= max(effective_transport_penalty(state) ** 0.35, 1.0)
        replenishment_factor = dynamic_strategy_factor(
            state, "regional_replenishment_factor"
        )
        allocation_shares = component_allocation_shares(state)
        baseline_component_shares = {
            component: COMPONENT_DEMAND_WEIGHTS[component]
            * COMPONENT_BUFFER_MULTIPLIER.get(component, 1.0)
            for component in ["RBC", "PLATELETS", "PLASMA"]
        }
        baseline_total = max(sum(baseline_component_shares.values()), 1e-6)
        baseline_component_shares = {
            component: weight / baseline_total
            for component, weight in baseline_component_shares.items()
        }

        for component in ["RBC", "PLATELETS", "PLASMA"]:
            target = replenishment_target_units(params, component)
            deficit = max(target - inventory[component], 0.0)
            allocation_focus = allocation_shares[component] / max(
                baseline_component_shares[component],
                1e-6,
            )
            capped_inbound = (
                QC_DAILY_LABILE_PRODUCTS_EST
                * COMPONENT_DEMAND_WEIGHTS[component]
                * params.regional_replenishment_rate
                * replenishment_factor
                * cycle_fraction
                * COMPONENT_BUFFER_MULTIPLIER.get(component, 1.0)
                * inbound_rate_factor
                * allocation_focus
            )
            inbound_units = max(0, int(round(min(deficit, capped_inbound))))
            added = dispatch_external_units(state, component, inbound_units)
            if added:
                cycle_units += added
                cycle_components[component] = added

        if cycle_units:
            if state.record_history:
                state.replenishment_log.append(
                    {
                        "time": env.now,
                        "weather": state.weather.state,
                        "units": cycle_units,
                        "components": cycle_components,
                    }
                )
            dt = sim_datetime(env.now)
            component_text = ", ".join(
                f"{component}={count}" for component, count in cycle_components.items()
            )
            state.emit(
                f"[{env.now:6.1f}h | {dt:%d-%b %H:%M}] provincial inbound "
                f"{cycle_units} units ({component_text})"
            )


def donor_process(state: SimState, person: Person):
    env = state.env
    params = state.params

    show_probability = donor_show_probability(
        env.now, state.weather, params.donor_show_factor
    )
    show_probability *= dynamic_strategy_factor(state, "donor_show_up_factor")
    show_probability *= 1 + (0.25 * person.motivation)
    show_probability = min(max(show_probability, 0.02), 0.99)
    if random.random() > show_probability:
        center = state.nearest_open_center(person.lat, person.lon)
        center.stats["no_show"] += 1
        return

    center = state.nearest_open_center(person.lat, person.lon)

    travel_h = road_travel_hours(
        state.G,
        person.lat,
        person.lon,
        center.lat,
        center.lon,
        env.now,
        state.weather,
        transport_penalty=transport_multiplier(state),
    )
    state.travel_time_total += travel_h
    state.travel_time_count += 1
    center.stats["travel_time_total"] += travel_h
    center.stats["travel_time_count"] += 1
    if state.record_history:
        state.travel_log.append(travel_h)
        center.stats["travel_times"].append(travel_h)

    dt = sim_datetime(env.now)
    state.emit(
        f"[{env.now:6.1f}h | {dt:%d-%b %H:%M}] donor->{center.name[:22]} "
        f"travel={travel_h * 60:.0f}m weather={state.weather.state} {person.blood_type}"
    )
    yield env.timeout(travel_h)

    eligible, reasons = person.eligibility()
    yield env.timeout(random.uniform(0.083, 0.25) + random.uniform(0.083, 0.167))
    if not eligible:
        state.emit(
            f"[{env.now:6.1f}h] deferred@{center.name[:22]} {'; '.join(reasons[:2])}"
        )
        center.stats["rejected"] += 1
        return

    t0 = env.now
    with center.nurses.request() as req:
        yield req
        wait_h = env.now - t0
        state.wait_time_total += wait_h
        state.wait_time_count += 1
        center.stats["wait_time_total"] += wait_h
        center.stats["wait_time_count"] += 1
        if state.record_history:
            center.stats["wait_times"].append(wait_h)
        nurse_relief = max(
            1.0 - (0.06 * dynamic_strategy_sum(state, "extra_nurses")), 0.55
        )
        yield env.timeout(random.uniform(0.167, 0.25) * nurse_relief)

    # ── Blood draw complete — count the donation at collection point ──
    # Héma-Québec reports "completed donor events" at the point of
    # collection, not after quarantine.  This avoids the pipeline edge
    # effect that artificially depresses donation counts in short runs.
    center.stats["collected"] += 1
    if state.record_history:
        state.donation_log.append(
            {
                "time": env.now,
                "blood_type": person.blood_type,
                "center": center.name,
                "travel_h": travel_h,
                "committed": person.committed,
            }
        )

    with center.lab.request() as req:
        yield req
        lab_t = random.uniform(1.0, 3.0) * params.lab_time_factor
        lab_t *= dynamic_strategy_factor(state, "lab_speed_factor")
        lab_t *= max(
            1.0 - (0.08 * dynamic_strategy_sum(state, "extra_lab_staff")), 0.45
        )
        if is_holiday(env.now) or is_weekend(env.now):
            lab_t *= 1.20
        yield env.timeout(lab_t)

    if random.random() < 0.005:
        state.emit(f"[{env.now:6.1f}h] lab rejection@{center.name[:22]}")
        center.stats["lab_rejected"] += 1
        return

    with center.processing.request() as req:
        yield req
        processing_t = random.uniform(2.0, 4.0) * params.proc_time_factor
        processing_t *= max(
            1.0 - (0.08 * dynamic_strategy_sum(state, "extra_processing_staff")),
            0.45,
        )
        yield env.timeout(processing_t)

    release_delay_h = random.uniform(
        *STEP_TIMES["quarantine"]
    ) * dynamic_strategy_factor(state, "release_delay_factor")
    if center.ctype == "mobile":
        release_delay_h += active_mobile_release_delay_h(state)
    yield env.timeout(release_delay_h)

    for component in donation_components():
        unit = BloodUnit(env, person.blood_type, component, center.name)
        center.add_unit(unit)

    center.stats["donated"] += 1
    state.emit(
        f"[{env.now:6.1f}h] donation complete -> {center.name[:22]} ({person.blood_type})"
    )


def donor_generator(state: SimState):
    env = state.env
    params = state.params
    while True:
        rate_h = max(
            params.donor_inter_arrival_h
            * dynamic_strategy_factor(state, "donor_arrival_factor"),
            0.08,
        )
        yield env.timeout(random.expovariate(1 / rate_h))
        h = hour_in_day(env.now)
        extension = active_hours_extension_h(state)
        if h < max(7.0 - extension, 0.0) or h > min(21.0 + extension, 24.0):
            continue
        committed = random.random() < min(
            dynamic_strategy_sum(state, "committed_donor_share"), 0.90
        )
        person = Person(
            state.south,
            state.north,
            state.west,
            state.east,
            eligible_rate=min(
                params.eligible_rate
                * dynamic_strategy_factor(state, "donor_eligibility_factor"),
                0.99,
            ),
            committed=committed,
        )
        env.process(donor_process(state, person))


def reserve_floor(state: SimState, center: DonationCenter, component: str) -> int:
    emergency_level = strongest_action_level(
        state, lambda action: action.emergency_share
    )
    if center.ctype == "mobile":
        return int(round((2.0 * (1.0 - emergency_level)) + (1.0 * emergency_level)))
    if center.ctype == "hospital":
        floor = int(round((4.0 * (1.0 - emergency_level)) + (2.0 * emergency_level)))
        floor += int(round(hospital_priority_bias(state, center.name)))
        return max(floor, 1)
    return active_reserve_threshold(state) if emergency_level > 0.05 else 4


def sample_units_requested(params: ScenarioParams, rng: random.Random) -> int:
    upper = max(2.0, params.avg_units_per_order * 2.0)
    return max(1, int(round(rng.triangular(1.0, upper, params.avg_units_per_order))))


def sample_order_priority(
    state: SimState,
    component: str,
    rng: random.Random,
) -> str:
    weights = {"routine": 0.58, "urgent": 0.28, "emergency": 0.14}
    if component == "PLATELETS":
        weights["urgent"] += 0.06
        weights["emergency"] += 0.05
        weights["routine"] -= 0.11
    elif component == "RBC":
        weights["emergency"] += 0.03
        weights["routine"] -= 0.03

    crisis_pressure = max(state.demand_pressure - 1.0, 0.0)
    weights["urgent"] += 0.18 * crisis_pressure
    weights["emergency"] += 0.30 * crisis_pressure
    weights["routine"] = max(weights["routine"] - (0.48 * crisis_pressure), 0.10)

    keys = list(weights.keys())
    values = np.asarray([weights[key] for key in keys], dtype=np.float32)
    values /= max(float(values.sum()), 1e-6)
    return rng.choices(keys, weights=values.tolist())[0]


def managed_units_requested(
    state: SimState, base_units_requested: int
) -> tuple[int, int]:
    demand_factor = dynamic_strategy_factor(state, "demand_management_factor")
    managed_units = max(0, int(round(base_units_requested * demand_factor)))
    conserved_units = max(base_units_requested - managed_units, 0)
    return managed_units, conserved_units


def center_priority(
    state: SimState,
    hospital: DonationCenter,
    center: DonationCenter,
    blood_type: str,
    component: str,
) -> tuple:
    compatible_units = center.compatible_unit_count(blood_type, component)
    component_units = center.component_count(component)
    distance = math.hypot(center.lat - hospital.lat, center.lon - hospital.lon)
    return (
        0 if center is hospital else 1,
        0 if center.ctype == "hospital" else 1,
        0 if center.ctype == "blood_bank" else 1,
        -compatible_units,
        -component_units,
        distance,
    )


def request_one_unit(
    state: SimState,
    hospital: DonationCenter,
    blood_type: str,
    component: str,
    priority: str = "routine",
) -> tuple[Optional[DonationCenter], Optional[BloodUnit]]:
    ordered_centers = sorted(
        state.centers,
        key=lambda center: center_priority(
            state, hospital, center, blood_type, component
        ),
    )
    reserve_relief = int(priority_profile(priority)["reserve_relief"])
    for center in ordered_centers:
        if center is not hospital:
            component_units = center.component_count(component)
            reserve_cutoff = max(
                reserve_floor(state, center, component) - reserve_relief, 0
            )
            if component_units <= reserve_cutoff:
                continue
        unit = center.request_unit(blood_type, component)
        if unit:
            return center, unit
    return None, None


def fulfill_hospital_order(
    state: SimState,
    hospital: DonationCenter,
    blood_type: str,
    component: str,
    units_requested: int,
    base_units_requested: Optional[int] = None,
    conserved_units: int = 0,
    priority: str = "routine",
) -> dict[str, float | int]:
    base_units_requested = (
        base_units_requested if base_units_requested is not None else units_requested
    )
    filled_units = 0
    transfer_units = 0
    transfer_delay = 0.0
    priority_weight = float(priority_profile(priority)["weight"])

    for _ in range(units_requested):
        supplier, unit = request_one_unit(
            state,
            hospital,
            blood_type,
            component,
            priority=priority,
        )
        if unit is None:
            hospital.stats["shortage"] += 1
            state.total_shortage_units += 1
            state.shortage_by_component[component] += 1
            state.shortage_by_hospital[hospital.name] += 1
            if state.record_history:
                state.shortage_log.append(
                    {
                        "time": state.env.now,
                        "hospital": hospital.name,
                        "component": component,
                        "blood_type": blood_type,
                        "priority": priority,
                        "weather": state.weather.state,
                    }
                )
            continue

        hospital.stats["transfused"] += 1
        filled_units += 1
        allowed_donor_types = set(compatible_donor_types(blood_type, component))
        if unit.blood_type == blood_type:
            state.total_exact_match_units += 1
        elif unit.blood_type in allowed_donor_types:
            state.total_compatible_substitution_units += 1
        else:
            state.total_incompatible_fulfillment_units += 1
        if supplier is not hospital:
            hospital.stats["transfers_in"] += 1
            transfer_units += 1
            state.total_transfer_units += 1
            emergency_level = strongest_action_level(
                state, lambda action: action.emergency_share
            )
            transfer_delay += (0.20 * (1.0 - emergency_level)) + (
                0.12 * emergency_level
            )
            if state.record_history:
                state.transfer_log.append(
                    {
                        "time": state.env.now,
                        "from": supplier.name,
                        "to": hospital.name,
                        "component": component,
                        "blood_type": unit.blood_type,
                        "priority": priority,
                    }
                )

    state.total_base_requested_units += base_units_requested
    state.total_net_requested_units += units_requested
    state.priority_weighted_requested_units += base_units_requested * priority_weight
    state.priority_weighted_shortage_units += (
        units_requested - filled_units
    ) * priority_weight
    if state.record_history:
        state.demand_log.append(
            {
                "time": state.env.now,
                "hospital": hospital.name,
                "component": component,
                "blood_type": blood_type,
                "priority": priority,
                "base_requested": base_units_requested,
                "conserved": conserved_units,
                "requested": units_requested,
                "filled": filled_units,
                "shortage": units_requested - filled_units,
            }
        )
    if conserved_units:
        state.total_conserved_units += conserved_units
        if state.record_history:
            state.conservation_log.append(
                {
                    "time": state.env.now,
                    "hospital": hospital.name,
                    "component": component,
                    "blood_type": blood_type,
                    "units": conserved_units,
                }
            )
    return {
        "filled_units": filled_units,
        "transfer_units": transfer_units,
        "transfer_delay": transfer_delay,
        "shortage_units": units_requested - filled_units,
    }


def handle_hospital_order(
    state: SimState,
    hospital: DonationCenter,
    blood_type: str,
    component: str,
    units_requested: int,
    base_units_requested: Optional[int] = None,
    conserved_units: int = 0,
    priority: str = "routine",
):
    if units_requested <= 0:
        priority_weight = float(priority_profile(priority)["weight"])
        state.total_base_requested_units += base_units_requested or 0
        state.total_net_requested_units += 0
        state.priority_weighted_requested_units += (
            base_units_requested or 0
        ) * priority_weight
        if conserved_units:
            state.total_conserved_units += conserved_units
        if state.record_history:
            state.demand_log.append(
                {
                    "time": state.env.now,
                    "hospital": hospital.name,
                    "component": component,
                    "blood_type": blood_type,
                    "priority": priority,
                    "base_requested": base_units_requested or 0,
                    "conserved": conserved_units,
                    "requested": 0,
                    "filled": 0,
                    "shortage": 0,
                }
            )
            if conserved_units:
                state.conservation_log.append(
                    {
                        "time": state.env.now,
                        "hospital": hospital.name,
                        "component": component,
                        "blood_type": blood_type,
                        "units": conserved_units,
                    }
                )
        dt = sim_datetime(state.env.now)
        state.emit(
            f"[{state.env.now:6.1f}h | {dt:%d-%b %H:%M}] conservation "
            f"{hospital.name[:20]} {component} {blood_type} deferred={conserved_units}"
        )
        return

    result = fulfill_hospital_order(
        state,
        hospital,
        blood_type,
        component,
        units_requested,
        base_units_requested=base_units_requested,
        conserved_units=conserved_units,
        priority=priority,
    )
    filled_units = int(result["filled_units"])
    transfer_units = int(result["transfer_units"])
    transfer_delay = float(result["transfer_delay"])
    if transfer_units:
        transfer_delay = shipment_delay_hours(
            state,
            transfer_units,
            lane="hospital_transfer",
            priority=priority,
        )
    if transfer_delay:
        yield state.env.timeout(transfer_delay)

    dt = sim_datetime(state.env.now)
    if filled_units == units_requested:
        state.emit(
            f"[{state.env.now:6.1f}h | {dt:%d-%b %H:%M}] order filled "
            f"{hospital.name[:20]} {priority} {component} {blood_type} x{units_requested}"
        )
    else:
        state.emit(
            f"[{state.env.now:6.1f}h | {dt:%d-%b %H:%M}] shortage "
            f"{hospital.name[:20]} {priority} {component} {blood_type} "
            f"filled={filled_units}/{units_requested}"
        )


def hospital_demand(state: SimState):
    env = state.env
    params = state.params
    rng = state.rng_demand
    while True:
        surge = demand_surge_multiplier(
            env.now, state.weather, params.demand_surge_factor
        )
        rate = max(params.demand_rate_h / (surge * state.demand_pressure), 0.10)
        yield env.timeout(rng.expovariate(1 / rate))

        hospital = rng.choice(state.hospitals)
        hospital.stats["orders"] += 1
        blood_type = rng.choices(
            BLOOD_TYPES, weights=list(BLOOD_DISTRIBUTION.values())
        )[0]
        component = rng.choices(
            ["RBC", "PLATELETS", "PLASMA"],
            weights=[
                COMPONENT_DEMAND_WEIGHTS["RBC"],
                COMPONENT_DEMAND_WEIGHTS["PLATELETS"],
                COMPONENT_DEMAND_WEIGHTS["PLASMA"],
            ],
        )[0]
        priority = sample_order_priority(state, component, rng)
        demand_multiplier = np.clip(
            0.95 + (0.35 * (state.demand_pressure - 1.0)) + rng.gauss(0.0, 0.05),
            0.80,
            1.55,
        )
        base_units_requested = max(
            1,
            int(round(sample_units_requested(params, rng) * demand_multiplier)),
        )
        units_requested, conserved_units = managed_units_requested(
            state, base_units_requested
        )
        state.recent_priority_pressure = float(
            np.clip(
                (0.88 * state.recent_priority_pressure)
                + (0.12 * ((priority_profile(priority)["weight"] - 1.0) / 3.0)),
                0.0,
                1.0,
            )
        )
        env.process(
            handle_hospital_order(
                state,
                hospital,
                blood_type,
                component,
                units_requested,
                base_units_requested=base_units_requested,
                conserved_units=conserved_units,
                priority=priority,
            )
        )


def monitor(state: SimState):
    env = state.env
    _SNAP_INTERVAL = 6  # hours between inventory snapshots
    _EXPIRY_CYCLE = 4  # remove expired every 4 snapshots = 24 h
    tick = 0
    while True:
        yield env.timeout(_SNAP_INTERVAL)
        tick += 1

        # Remove expired units every 24h
        expired = 0
        if tick % _EXPIRY_CYCLE == 0:
            expired = sum(center.remove_expired() for center in state.centers)

        if state.fast_mode:
            continue

        # Take inventory snapshot every 6h for temporal validation
        inv = {"hour": env.now, "RBC": 0, "PLATELETS": 0, "PLASMA": 0, "shortages": 0}
        for center in state.centers:
            for component, units in center.inventory_by_component().items():
                inv[component] += units
        inv["shortages"] = state.total_shortage_units
        state.hourly_inventory.append(inv)

        # Print status every 24h to avoid spam
        if tick % _EXPIRY_CYCLE == 0:
            dt = sim_datetime(env.now)
            state.emit(
                f"\n{'-' * 68}\n"
                f"[{env.now:6.0f}h | {dt:%d-%b-%Y}] "
                f"RBC={inv['RBC']} PLT={inv['PLATELETS']} PLS={inv['PLASMA']} "
                f"expired={expired} cumulative_shortage={inv['shortages']}\n"
                f"{'-' * 68}"
            )


# ===========================================================================
# Legacy heuristic PPO controller (kept for backward compatibility)
# ===========================================================================

PPO_WEATHER_SEVERITY = {
    "clear": 0.0,
    "cloudy": 0.15,
    "snow": 0.45,
    "ice_storm": 0.70,
    "blizzard": 1.0,
}

PPO_POLICY_WEIGHTS = {
    "campaign": {
        "bias": -0.75,
        "donor_stress": 1.55,
        "critical_gap": 0.80,
        "recent_shortage_rate": 0.75,
        "weather_stress": -0.20,
        "active_cost_ratio": -0.10,
    },
    "lab_fast_track": {
        "bias": -0.60,
        "recent_shortage_rate": 1.10,
        "critical_gap": 0.85,
        "demand_stress": 0.45,
        "weather_stress": 0.15,
        "active_cost_ratio": -0.05,
    },
    "emergency_share": {
        "bias": -0.35,
        "recent_shortage_rate": 1.65,
        "critical_gap": 1.10,
        "transport_stress": 0.70,
        "rbc_gap": 0.35,
        "weather_stress": 0.15,
    },
    "clinical_conservation": {
        "bias": -0.70,
        "recent_shortage_rate": 1.75,
        "demand_stress": 1.20,
        "platelet_gap": 0.55,
        "transport_stress": 0.25,
        "active_cost_ratio": -0.10,
    },
    "national_mutual_aid": {
        "bias": -1.00,
        "recent_shortage_rate": 1.45,
        "critical_gap": 1.35,
        "transport_stress": 1.00,
        "weather_stress": 0.65,
        "cumulative_shortage_rate": 0.40,
    },
}


def total_transfused_units(state: SimState) -> int:
    return sum(center.stats["transfused"] for center in state.centers)


def total_donated_units(state: SimState) -> int:
    return sum(center.stats["donated"] for center in state.centers)


def policy_snapshot(state: SimState) -> dict[str, float]:
    return {
        "shortage": float(state.total_shortage_units),
        "requested": float(state.total_net_requested_units),
        "transfused": float(total_transfused_units(state)),
        "donated": float(total_donated_units(state)),
        "platelet_shortage": float(state.shortage_by_component.get("PLATELETS", 0)),
        "rbc_shortage": float(state.shortage_by_component.get("RBC", 0)),
        "plasma_shortage": float(state.shortage_by_component.get("PLASMA", 0)),
    }


def component_coverage_days(state: SimState) -> dict[str, float]:
    inventory = city_inventory(state)
    return {
        component: inventory[component]
        / max(QC_DAILY_LABILE_PRODUCTS_EST * COMPONENT_DEMAND_WEIGHTS[component], 1.0)
        for component in ["RBC", "PLATELETS", "PLASMA"]
    }


def build_ppo_observation(
    state: SimState, previous_snapshot: dict[str, float]
) -> dict[str, float]:
    coverage = component_coverage_days(state)
    return _build_observation_shared(
        total_shortage_units=state.total_shortage_units,
        total_net_requested_units=state.total_net_requested_units,
        total_transfused=total_transfused_units(state),
        coverage=coverage,
        reserve_target_days=state.params.reserve_target_days,
        donor_inter_arrival_h=state.params.donor_inter_arrival_h,
        donor_show_factor=state.params.donor_show_factor,
        demand_rate_h=state.params.demand_rate_h / max(state.demand_pressure, 1e-6),
        demand_surge_factor=state.params.demand_surge_factor * state.demand_pressure,
        effective_transport_penalty=effective_transport_penalty(state),
        weather_state=state.weather.state,
        prev_shortage=previous_snapshot["shortage"],
        prev_requested=previous_snapshot["requested"],
        prev_transfused=previous_snapshot["transfused"],
        active_cost=active_action_cost(state),
        forecast_pressure=state.forecast_pressure,
        priority_pressure=max(
            state.recent_priority_pressure, state.forecast_priority_pressure
        ),
        congestion_stress=min(congestion_stress(state), 1.0),
        budget_pressure=budget_pressure(state),
    )


def softmax_probabilities(logits: dict[str, float]) -> dict[str, float]:
    if not logits:
        return {}
    max_logit = max(logits.values())
    exps = {key: math.exp(value - max_logit) for key, value in logits.items()}
    total = sum(exps.values()) or 1.0
    return {key: value / total for key, value in exps.items()}


def activate_action(
    state: SimState,
    action_key: str,
    policy: dict[str, float],
    logits: dict[str, float],
    observation: dict[str, float],
) -> bool:
    if any(activated.action.key == action_key for activated in state.active_actions):
        return False

    action = ACTION_CATALOG[action_key]
    state.active_actions.append(
        ActivatedAction(action=_timing_scaled(action, state.params), activated_at_h=state.env.now)
    )
    state.policy_log.append(
        {
            "time": state.env.now,
            "kind": "activation",
            "action": action_key,
            "probability": policy.get(action_key, 0.0),
            "logit": logits.get(action_key, 0.0),
            "observation": observation,
        }
    )
    dt = sim_datetime(state.env.now)
    state.emit(
        f"[{state.env.now:6.1f}h | {dt:%d-%b %H:%M}] PPO activates "
        f"{action.name.lower()} (p={policy.get(action_key, 0.0):.2f})"
    )
    return True


def ppo_shortage_controller(state: SimState):
    """
    Legacy heuristic controller – kept for ``controller_key == 'ppo_shortage_minimizer'``
    when no trained agent is provided.  Identical to the original implementation.
    """
    interval_h = max(state.strategy.controller_interval_h, 1.0)
    snapshot = policy_snapshot(state)

    while True:
        yield state.env.timeout(interval_h)
        candidates = [
            action
            for action in state.strategy.action_space
            if all(active.action.key != action.key for active in state.active_actions)
        ]
        observation = build_ppo_observation(state, snapshot)
        snapshot = policy_snapshot(state)

        if not candidates:
            state.policy_log.append(
                {
                    "time": state.env.now,
                    "kind": "idle",
                    "reason": "action_space_exhausted",
                    "observation": observation,
                }
            )
            continue

        logits = {}
        for action in candidates:
            weights = PPO_POLICY_WEIGHTS[action.key]
            logit = weights.get("bias", 0.0)
            for feature_name, feature_weight in weights.items():
                if feature_name == "bias":
                    continue
                logit += observation.get(feature_name, 0.0) * feature_weight
            logits[action.key] = logit

        policy = softmax_probabilities(logits)
        best_action_key = max(policy, key=policy.get)
        severity = max(
            observation["recent_shortage_rate"] * 3.0 + observation["critical_gap"],
            observation["critical_gap"] + 0.85 * observation["donor_stress"],
            observation["critical_gap"]
            + 0.70 * observation["demand_stress"]
            + 0.50 * observation["transport_stress"],
        )
        trigger_probability = 0.33 if severity >= 0.45 else 0.48
        activated = False
        if policy[best_action_key] >= trigger_probability or severity >= 0.95:
            activated = activate_action(
                state,
                best_action_key,
                policy=policy,
                logits=logits,
                observation=observation,
            )

        state.policy_log.append(
            {
                "time": state.env.now,
                "kind": "decision",
                "selected_action": best_action_key,
                "activated": activated,
                "trigger_probability": trigger_probability,
                "severity": severity,
                "policy": policy,
                "observation": observation,
            }
        )


# ===========================================================================
# NEW: Trained-PPO controller (SimPy process)
# ===========================================================================


def _make_obs(state: SimState, prev_snapshot: dict) -> "np.ndarray":
    """Build the observation vector for the PPO agent."""
    import numpy as np  # local import to avoid hard dependency at module level

    raw = build_ppo_observation(state, prev_snapshot)
    return _clamp_obs(raw)


def _make_official_dreamerv3_policy_obs(
    state: SimState,
    reward: float,
    *,
    is_first: bool,
) -> dict[str, np.ndarray]:
    snapshot = policy_snapshot(state)
    vector = _make_obs(state, snapshot).astype(np.float32, copy=False)

    return {
        "vector": vector[None, :],
        "reward": np.asarray([reward], dtype=np.float32),
        "is_first": np.asarray([is_first], dtype=bool),
        "is_last": np.asarray([False], dtype=bool),
        "is_terminal": np.asarray([False], dtype=bool),
    }


def _extract_discrete_action_index(action_value: Any) -> int:
    return int(np.asarray(action_value).reshape(-1)[0])


def normalized_reward_delta(previous_total: float, current_total: float) -> float:
    return (current_total - previous_total) / max(abs(previous_total) + 1.0, 1.0)


def controller_decision_count(policy_log: list[dict[str, Any]]) -> int:
    return sum(
        1
        for entry in policy_log
        if entry.get("kind") in {"decision", "continuous_decision"}
    )


def track_episode_score(
    state: SimState,
    *,
    step_hours: float,
):
    step_hours = max(float(step_hours), 1e-6)
    previous_total = float(
        state.episode_prev_reward_total
        if state.episode_prev_reward_total is not None
        else calculate_reward(
            state,
            prev_transfused=state.episode_prev_transfused,
            update_prev_transfused=False,
        ).total
    )
    previous_transfused = int(state.episode_prev_transfused)

    while state.env.now < state.params.sim_hours:
        remaining = float(state.params.sim_hours - state.env.now)
        if remaining <= 0:
            break

        yield state.env.timeout(min(step_hours, remaining))

        current_total = float(
            calculate_reward(
                state,
                prev_transfused=previous_transfused,
                update_prev_transfused=False,
            ).total
        )
        reward = normalized_reward_delta(previous_total, current_total)
        state.episode_reward_deltas.append(float(reward))
        state.episode_score += float(reward)
        previous_total = current_total
        previous_transfused = total_transfused_units(state)


def apply_continuous_action_levels(
    state: SimState,
    action_levels: dict[str, float],
    *,
    controller: str,
    obs_summary: Optional[dict[str, Any]] = None,
    record_policy: bool = True,
) -> dict[str, Any]:
    upserted: list[dict[str, float | str]] = []
    deactivated: list[str] = []
    routing_updates: list[dict[str, float | str]] = []
    allocation_updates: list[dict[str, float | str]] = []

    for action_key, raw_level in action_levels.items():
        level = max(min(float(raw_level), 1.0), 0.0)
        if action_key in HOSPITAL_ROUTE_CONTROL_MAP:
            hospital_name = HOSPITAL_ROUTE_CONTROL_MAP[action_key]
            previous_level = state.continuous_hospital_routing.get(hospital_name, 0.0)
            if not np.isclose(previous_level, level):
                state.continuous_hospital_routing[hospital_name] = level
                routing_updates.append(
                    {
                        "route_key": action_key,
                        "hospital": hospital_name,
                        "level": round(level, 4),
                    }
                )
            continue

        if action_key in COMPONENT_ALLOCATION_CONTROL_MAP:
            component = COMPONENT_ALLOCATION_CONTROL_MAP[action_key]
            previous_level = state.continuous_component_allocation.get(component, 0.0)
            if not np.isclose(previous_level, level):
                state.continuous_component_allocation[component] = level
                allocation_updates.append(
                    {
                        "allocation_key": action_key,
                        "component": component,
                        "level": round(level, 4),
                    }
                )
            continue

        if action_key not in ACTION_CATALOG:
            continue

        existing_idx = next(
            (
                idx
                for idx, activated in enumerate(state.active_actions)
                if activated.action.key == action_key and activated.persistent
            ),
            None,
        )

        if level > 0.0 and not budget_available(state):
            state.budget_blocked_controls += 1
            if existing_idx is not None:
                state.active_actions.pop(existing_idx)
                deactivated.append(action_key)
            continue

        if level <= 0.0:
            if existing_idx is not None:
                state.active_actions.pop(existing_idx)
                deactivated.append(action_key)
            continue

        if existing_idx is None:
            state.active_actions.append(
                ActivatedAction(
                    action=_timing_scaled(ACTION_CATALOG[action_key], state.params),
                    activated_at_h=state.env.now,
                    intensity=level,
                    persistent=True,
                )
            )
            upserted.append({"action_key": action_key, "level": round(level, 4)})
        else:
            previous_level = state.active_actions[existing_idx].intensity
            if not np.isclose(previous_level, level):
                state.active_actions[existing_idx].intensity = level
                upserted.append({"action_key": action_key, "level": round(level, 4)})

    changed = bool(upserted or deactivated or routing_updates or allocation_updates)
    if record_policy:
        state.policy_log.append(
            {
                "time": state.env.now,
                "kind": "continuous_decision",
                "controller": controller,
                "levels": {
                    key: round(value, 4) for key, value in action_levels.items()
                },
                "upserted": upserted,
                "deactivated": deactivated,
                "routing_updates": routing_updates,
                "allocation_updates": allocation_updates,
                "obs": obs_summary,
            }
        )
        if changed:
            dt = sim_datetime(state.env.now)
            intervention_summary = ", ".join(
                f"{item['action_key']}={item['level']:.2f}" for item in upserted
            )
            if deactivated:
                intervention_summary = ", ".join(
                    part
                    for part in [intervention_summary, f"off={','.join(deactivated)}"]
                    if part
                )
            routing_summary = ", ".join(
                f"{item['route_key']}={item['level']:.2f}" for item in routing_updates
            )
            allocation_summary = ", ".join(
                f"{item['allocation_key']}={item['level']:.2f}"
                for item in allocation_updates
            )
            summary = "; ".join(
                part
                for part in [
                    intervention_summary,
                    routing_summary,
                    allocation_summary,
                ]
                if part
            )
            state.emit(
                f"[{state.env.now:6.1f}h | {dt:%d-%b %H:%M}] "
                f"{controller.replace('_', ' ')} controls {summary}"
            )

    return {
        "levels": action_levels,
        "upserted": upserted,
        "deactivated": deactivated,
        "routing_updates": routing_updates,
        "allocation_updates": allocation_updates,
        "changed": changed,
    }


def _apply_official_dreamerv3_action(
    state: SimState,
    acts: dict[str, Any],
    obs: dict[str, np.ndarray],
    *,
    controller: str = "official_dreamerv3",
) -> None:
    obs_summary = {
        "vector": np.asarray(obs["vector"]).reshape(-1).tolist(),
        "reward": float(np.asarray(obs["reward"]).reshape(-1)[0]),
        "is_first": bool(np.asarray(obs["is_first"]).reshape(-1)[0]),
    }
    try:
        action_levels = decode_continuous_action_levels(
            acts.get("action", np.zeros(DREAMER_ACTION_DIM, dtype=np.float32))
        )
    except ValueError as exc:
        state.policy_log.append(
            {
                "time": state.env.now,
                "kind": "continuous_decision",
                "controller": controller,
                "levels": "invalid",
                "reason": str(exc),
                "obs": obs_summary,
            }
        )
        return

    apply_continuous_action_levels(
        state,
        action_levels,
        controller=controller,
        obs_summary=obs_summary,
        record_policy=True,
    )


def trained_official_dreamerv3_controller(
    state: SimState,
    agent: Any,
    step_hours: float = 6.0,
    *,
    controller: str = "official_dreamerv3",
):
    """
    SimPy process that drives action selection through a Dreamer-compatible
    checkpoint-backed policy.
    """
    env = state.env
    carry = agent.init_policy(1)
    prev_step_snapshot = step_reward_snapshot(state)

    # Added robust try/except to prevent silent crashes in the SimPy background process
    try:
        obs = _make_official_dreamerv3_policy_obs(state, reward=0.0, is_first=True)
        carry, acts, _ = agent.policy(carry, obs, mode="eval")
        _apply_official_dreamerv3_action(state, acts, obs, controller=controller)
    except Exception as e:
        import traceback

        traceback.print_exc()
        state.emit(f"[{state.env.now:6.1f}h] DREAMER FATAL ERROR ON INIT: {e}")
        return

    while True:
        yield env.timeout(step_hours)
        if env.now >= state.params.sim_hours:
            break

        reward = calculate_step_reward(state, prev_step_snapshot)
        prev_step_snapshot = step_reward_snapshot(state)

        try:
            obs = _make_official_dreamerv3_policy_obs(
                state,
                reward=reward,
                is_first=False,
            )
            carry, acts, _ = agent.policy(carry, obs, mode="eval")
            _apply_official_dreamerv3_action(
                state,
                acts,
                obs,
                controller=controller,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            state.emit(f"[{state.env.now:6.1f}h] DREAMER FATAL ERROR IN LOOP: {e}")
            break


def trained_ppo_controller(state: SimState, agent: PPO, step_hours: float = 6.0):
    """
    SimPy process that drives action selection using a **trained** PPO agent.

    Every ``step_hours`` simulation hours the process:
      1. Builds the observation vector.
      2. Calls ``agent.predict(obs, deterministic=True)`` to get the best action.
      3. Translates the discrete action index into an ``ACTION_CATALOG`` key.
      4. Activates that action (if it is not already active) and logs the decision.

    Parameters
    ----------
    state      : live SimState
    agent      : a trained ``PPO`` instance (from ``ppo_agent.py`` or SB3)
    step_hours : how often (sim-hours) the controller queries the agent
    """
    env = state.env
    prev_snapshot = policy_snapshot(state)

    while True:
        yield env.timeout(step_hours)

        obs = _make_obs(state, prev_snapshot)
        prev_snapshot = policy_snapshot(state)

        # ── Query trained agent ────────────────────────────────────────

        action_idx, _ = agent.predict(obs, deterministic=True)
        action_idx = int(action_idx)

        if action_idx == 0:
            # NOOP
            state.policy_log.append(
                {
                    "time": env.now,
                    "kind": "decision",
                    "controller": "trained_ppo",
                    "action_idx": 0,
                    "action_key": "noop",
                    "activated": False,
                    "obs": obs.tolist(),
                }
            )
            continue

        action_key = ACTION_KEYS[action_idx - 1]

        # Skip if already active
        already_active = any(a.action.key == action_key for a in state.active_actions)
        if already_active:
            state.policy_log.append(
                {
                    "time": env.now,
                    "kind": "decision",
                    "controller": "trained_ppo",
                    "action_idx": action_idx,
                    "action_key": action_key,
                    "activated": False,
                    "reason": "already_active",
                    "obs": obs.tolist(),
                }
            )
            continue

        # ── Activate the action ────────────────────────────────────────
        act_obj = ACTION_CATALOG[action_key]
        state.active_actions.append(
            ActivatedAction(action=_timing_scaled(act_obj, state.params), activated_at_h=env.now)
        )
        dt = sim_datetime(env.now)
        state.emit(
            f"[{env.now:6.1f}h | {dt:%d-%b %H:%M}] trained-PPO activates "
            f"{act_obj.name.lower()} (idx={action_idx})"
        )
        state.policy_log.append(
            {
                "time": env.now,
                "kind": "activation",
                "controller": "trained_ppo",
                "action_idx": action_idx,
                "action_key": action_key,
                "activated": True,
                "obs": obs.tolist(),
            }
        )


def trained_ppo_continuous_controller(
    state: SimState, agent: Any, step_hours: float = 6.0
):
    """
    SimPy process that drives action selection using a **trained** PPO agent
    with continuous actions (DreamerV3-style Box(-1,1) control vector).

    Every ``step_hours`` simulation hours the process:
      1. Builds the 16-dim observation vector.
      2. Calls ``agent.predict(obs, deterministic=True)`` to get a continuous
         action vector.
      3. Decodes the vector via ``continuous_action_levels()`` and applies
         it to the simulator through ``apply_continuous_action_levels()``.
    """
    env = state.env
    prev_snapshot = policy_snapshot(state)
    controller_name = getattr(agent, "_controller_name", "trained_ppo_continuous")

    while True:
        yield env.timeout(step_hours)
        if env.now >= state.params.sim_hours:
            break

        obs = _make_obs(state, prev_snapshot)
        prev_snapshot = policy_snapshot(state)

        action, _ = agent.predict(obs, deterministic=True)

        try:
            action_levels = decode_continuous_action_levels(action)
        except Exception as exc:
            state.policy_log.append(
                {
                    "time": env.now,
                    "kind": "continuous_decision",
                    "controller": controller_name,
                    "levels": "invalid",
                    "reason": str(exc),
                }
            )
            continue

        apply_continuous_action_levels(
            state,
            action_levels,
            controller=controller_name,
            obs_summary={"obs": obs.tolist()},
            record_policy=True,
        )


# ===========================================================================
# Training entry-point
# ===========================================================================


def train_ppo_agent(
    params: ScenarioParams,
    G,
    north: float,
    south: float,
    east: float,
    west: float,
    total_timesteps: int = 200_000,
    episode_hours: float = 168.0,
    step_hours: float = 6.0,
    learning_rate: float = 3e-4,
    n_steps: int = 512,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    hidden_dim: int = 64,
    save_path: Optional[str] = None,
    verbose: int = 1,
    seed: int = 42,
    training_scenarios: Optional[list[ScenarioParams]] = None,
    num_envs: int = 1,
    vec_env: str = "auto",
    action_mode: str = "discrete",
    logdir: Optional[str] = None,
) -> PPO:
    """
    Train a PPO agent to manage the blood supply chain.

    The agent observes continuous shortage, inventory, forecast, congestion,
    budget, weather, and demand pressure features and chooses one of
    ``len(ACTION_CATALOG) + 1`` actions (0 = NOOP, 1..N = activate action).

    Parameters
    ----------
    params          : default ScenarioParams used to initialise the training env.
    G               : road-network graph (passed through to the sim).
    north/south/east/west : city bounding box.
    total_timesteps : total environment steps to train for.
    episode_hours   : simulation hours per episode (default 168 = 1 week).
    step_hours      : sim-hours advanced per env.step() call (default 6).
    learning_rate   : Adam LR for the MLP policy.
    n_steps         : rollout buffer length.
    batch_size      : PPO mini-batch size.
    n_epochs        : PPO epochs per rollout.
    gamma           : discount factor.
    gae_lambda      : GAE λ.
    clip_range      : PPO-Clip ε.
    ent_coef        : entropy regularisation.
    hidden_dim      : MLP hidden layer width.
    save_path       : if given, save the trained model to this path.
    verbose         : 0 = silent, 1 = progress.
    seed            : RNG seed.
    training_scenarios : optional episode-level scenario pool for multi-scenario training.

    Returns
    -------
    A trained ``PPO`` instance.
    """
    if verbose >= 1:
        scenario_keys = [
            str(scenario.scenario_key) for scenario in (training_scenarios or [params])
        ]
        resolved_num_envs = _resolve_training_env_count(num_envs)
        print(f"\n{'=' * 68}")
        print("Training PPO agent for blood supply chain management")
        print(f"  total_timesteps = {total_timesteps:,}")
        print(f"  episode_hours   = {episode_hours} h  ({episode_hours / 24:.1f} days)")
        print(f"  step_hours      = {step_hours} h")
        print(f"  num_envs        = {resolved_num_envs}")
        print(f"  train_scenarios = {scenario_keys}")
        print(
            f"  action_space    = {N_ACTIONS}  (0=NOOP, 1..{N_ACTIONS - 1}={ACTION_KEYS})"
        )
        print(f"{'=' * 68}\n")

    from ppo_env import BloodSupplyEnv  # lazy – no circular import

    scenario_templates = list(training_scenarios or [params])
    requested_num_envs = _resolve_training_env_count(num_envs)
    use_sb3 = "stable_baselines3" in getattr(PPO, "__module__", "")
    env_kwargs = dict(
        params=params,
        G=G,
        north=north,
        south=south,
        east=east,
        west=west,
        episode_hours=episode_hours,
        step_hours=step_hours,
        seed=seed,
        scenario_templates=scenario_templates,
        collect_diagnostics=False,
        action_mode=action_mode,
    )

    if use_sb3 and requested_num_envs > 1:
        from stable_baselines3.common.env_util import make_vec_env  # type: ignore
        from stable_baselines3.common.vec_env import (  # type: ignore
            DummyVecEnv,
            SubprocVecEnv,
        )

        vec_env_mode = str(vec_env).strip().lower()
        if vec_env_mode not in {"auto", "dummy", "subproc"}:
            raise ValueError(
                f"Unsupported vec_env={vec_env!r}. Expected 'auto', 'dummy', or 'subproc'."
            )

        selected_vec_env_cls = DummyVecEnv
        selected_vec_env_name = "DummyVecEnv"
        vec_env_kwargs: dict[str, object] = {}

        if vec_env_mode in {"auto", "subproc"}:
            selected_vec_env_cls = SubprocVecEnv
            selected_vec_env_name = "SubprocVecEnv"
            vec_env_kwargs["start_method"] = (
                "spawn" if sys.platform == "darwin" else "forkserver"
            )

        if verbose >= 1:
            print(
                f"[train_ppo_agent] Collecting rollouts with "
                f"{selected_vec_env_name} across {requested_num_envs} envs."
            )

        try:
            env = make_vec_env(
                BloodSupplyEnv,
                n_envs=requested_num_envs,
                seed=seed,
                env_kwargs=env_kwargs,
                vec_env_cls=selected_vec_env_cls,
                vec_env_kwargs=vec_env_kwargs,
            )
        except Exception:
            if vec_env_mode != "auto" or selected_vec_env_cls is DummyVecEnv:
                raise
            if verbose >= 1:
                print(
                    "[train_ppo_agent] Subprocess vector env startup failed; "
                    "falling back to DummyVecEnv."
                )
            env = make_vec_env(
                BloodSupplyEnv,
                n_envs=requested_num_envs,
                seed=seed,
                env_kwargs=env_kwargs,
                vec_env_cls=DummyVecEnv,
            )
    else:
        if requested_num_envs > 1 and verbose >= 1 and not use_sb3:
            print(
                "[train_ppo_agent] Multi-env rollout requires stable-baselines3; "
                "falling back to a single environment."
            )
        env = BloodSupplyEnv(**env_kwargs)
        if use_sb3 and logdir:
            # Monitor wrapping makes rollout/ep_rew_mean available for the
            # learning-curve CSV (the vec-env path is Monitor-wrapped already).
            from stable_baselines3.common.monitor import Monitor  # type: ignore

            env = Monitor(env)

    ppo_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=verbose,
        seed=seed,
    )
    if "stable_baselines3" not in getattr(PPO, "__module__", ""):
        ppo_kwargs["hidden_dim"] = hidden_dim

    agent = PPO(**ppo_kwargs)

    if logdir and use_sb3:
        from stable_baselines3.common.logger import configure as _sb3_configure  # type: ignore

        agent.set_logger(_sb3_configure(str(logdir), ["csv", "stdout"]))

    agent.learn(total_timesteps=total_timesteps)
    # The checkpoint filename (save_path) is the source of truth for whether this
    # is a continuous-action model. The eval loaders honor "_continuous" if set.
    if save_path and "continuous" in str(save_path).lower():
        setattr(agent, "_continuous", True)
        setattr(agent, "_controller_name", "trained_ppo_continuous")
    else:
        setattr(agent, "_controller_name", "trained_ppo")

    if save_path:
        agent.save(save_path)
        if verbose >= 1:
            print(f"[train_ppo_agent] Model saved → {save_path}")

    return agent


def _resolve_training_env_count(requested_num_envs: int) -> int:
    if requested_num_envs > 0:
        return requested_num_envs
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count // 2 or 1))


def load_ppo_agent(
    model_path: str,
    params: ScenarioParams,
    G=None,
    north: float = 0.0,
    south: float = 0.0,
    east: float = 0.0,
    west: float = 0.0,
    hidden_dim: int = 64,
    verbose: int = 1,
) -> PPO:
    """
    Load a previously trained PPO agent from disk.

    If Stable Baselines 3 is available the SB3 ``PPO.load`` is called;
    otherwise the NumPy fallback ``PPO.load`` is used.

    Parameters
    ----------
    model_path : path written by ``train_ppo_agent(save_path=…)``
    params     : ScenarioParams (needed to reconstruct the env for SB3)
    G / north / south / east / west : city graph & bounds (SB3 only)
    hidden_dim : must match the value used during training (NumPy fallback)
    verbose    : verbosity level
    """
    try:
        from ppo_env import BloodSupplyEnv  # lazy – no circular import
        from stable_baselines3 import PPO as SB3_PPO  # type: ignore

        print("successfully imported stable_baselines3", PPO)

        env = BloodSupplyEnv(
            params=params,
            G=G,
            north=north,
            south=south,
            east=east,
            west=west,
        )
        return SB3_PPO.load(model_path, env=env)
    except ModuleNotFoundError:
        return PPO.load(model_path, hidden_dim=hidden_dim, verbose=verbose)


def load_ppo_continuous_agent(
    model_path: str,
    params: ScenarioParams,
    G=None,
    north: float = 0.0,
    south: float = 0.0,
    east: float = 0.0,
    west: float = 0.0,
    hidden_dim: int = 64,
    verbose: int = 1,
) -> PPO:
    """
    Load a previously trained PPO continuous-action agent from disk.

    The environment is configured with ``action_mode="continuous"`` so the
    PPO class auto-detects the continuous Box action space.
    """
    try:
        from ppo_env import BloodSupplyEnv
        from stable_baselines3 import PPO as SB3_PPO  # type: ignore

        env = BloodSupplyEnv(
            params=params,
            G=G,
            north=north,
            south=south,
            east=east,
            west=west,
            action_mode="continuous",
        )
        agent = SB3_PPO.load(model_path, env=env)
    except ModuleNotFoundError:
        from ppo_env import BloodSupplyEnv

        env = BloodSupplyEnv(
            params=params,
            G=G,
            north=north,
            south=south,
            east=east,
            west=west,
            action_mode="continuous",
        )
        agent = PPO.load(model_path, env=env, hidden_dim=hidden_dim, verbose=verbose)

    setattr(agent, "_continuous", True)
    setattr(agent, "_controller_name", "trained_ppo_continuous")
    return agent


def train_sac_agent(
    params: ScenarioParams,
    G,
    north: float,
    south: float,
    east: float,
    west: float,
    total_timesteps: int = 200_000,
    episode_hours: float = 168.0,
    step_hours: float = 6.0,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    buffer_size: int = 100_000,
    learning_starts: int = 1_000,
    tau: float = 0.005,
    gamma: float = 0.99,
    train_freq: int = 1,
    gradient_steps: int = 1,
    hidden_dim: int = 256,
    save_path: Optional[str] = None,
    verbose: int = 1,
    seed: int = 42,
    training_scenarios: Optional[list[ScenarioParams]] = None,
    logdir: Optional[str] = None,
):
    """Train a Soft Actor-Critic agent on the simulator's continuous action space."""
    from ppo_env import BloodSupplyEnv

    try:
        from stable_baselines3 import SAC  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SAC training requires stable-baselines3 with gymnasium installed."
        ) from exc

    scenario_templates = list(training_scenarios or [params])
    env = BloodSupplyEnv(
        params=params,
        G=G,
        north=north,
        south=south,
        east=east,
        west=west,
        episode_hours=episode_hours,
        step_hours=step_hours,
        seed=seed,
        scenario_templates=scenario_templates,
        collect_diagnostics=False,
        action_mode="continuous",
    )
    if logdir:
        # Monitor wrapping records rollout/ep_rew_mean for the learning-curve CSV.
        from stable_baselines3.common.monitor import Monitor  # type: ignore

        env = Monitor(env)
    policy_kwargs = {"net_arch": [hidden_dim, hidden_dim]}
    agent = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=seed,
    )
    if logdir:
        from stable_baselines3.common.logger import configure as _sb3_configure  # type: ignore

        agent.set_logger(_sb3_configure(str(logdir), ["csv", "stdout"]))
    agent.learn(total_timesteps=total_timesteps)
    setattr(agent, "_continuous", True)
    setattr(agent, "_controller_name", "trained_sac_continuous")
    if save_path:
        agent.save(save_path)
        if verbose >= 1:
            print(f"[train_sac_agent] Model saved → {save_path}")
    return agent


def load_sac_agent(
    model_path: str,
    params: ScenarioParams,
    G=None,
    north: float = 0.0,
    south: float = 0.0,
    east: float = 0.0,
    west: float = 0.0,
):
    """Load a trained SAC agent for the simulator's continuous action space."""
    from ppo_env import BloodSupplyEnv

    try:
        from stable_baselines3 import SAC  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading SAC checkpoints requires stable-baselines3."
        ) from exc

    env = BloodSupplyEnv(
        params=params,
        G=G,
        north=north,
        south=south,
        east=east,
        west=west,
        action_mode="continuous",
    )
    agent = SAC.load(model_path, env=env)
    setattr(agent, "_continuous", True)
    setattr(agent, "_controller_name", "trained_sac_continuous")
    return agent


def _learned_continuous_model_candidates(controller_key: str) -> tuple[Path, ...]:
    root = Path(__file__).resolve().parent
    mapping = {
        PPO_CONTINUOUS_CONTROLLER_KEY: (
            root / "ppo_blood_model.zip",
            root / "ppo_blood_model.pkl",
            root / "ppo_blood_model",
        ),
        SAC_CONTROLLER_KEY: (
            root / "sac_blood_model.zip",
            root / "sac_blood_model",
        ),
        IQL_CONTROLLER_KEY: (root / "iql_blood_model.pt",),
        CQL_CONTROLLER_KEY: (root / "cql_blood_model.pt",),
    }
    try:
        return tuple(path.resolve() for path in mapping[controller_key])
    except KeyError as exc:
        raise ValueError(f"Unsupported controller_key={controller_key!r}") from exc


def default_ppo_continuous_model_path() -> str:
    """Return the default path stem for the trained PPO continuous model.

    The actual file may have a ``.zip`` (SB3) or ``.pkl`` (NumPy fallback)
    extension.  Callers should use :func:`resolve_ppo_continuous_model_path`
    to find whichever variant exists on disk.
    """
    return str((Path(__file__).resolve().parent / "ppo_blood_model").resolve())


def resolve_ppo_continuous_model_path() -> str | None:
    """Find the PPO continuous model on disk, checking common extensions."""
    return resolve_learned_continuous_model_path(PPO_CONTINUOUS_CONTROLLER_KEY)


def default_sac_model_path() -> str:
    return str((Path(__file__).resolve().parent / "sac_blood_model").resolve())


def resolve_sac_model_path() -> str | None:
    return resolve_learned_continuous_model_path(SAC_CONTROLLER_KEY)


def resolve_learned_continuous_model_path(controller_key: str) -> str | None:
    env_key = LEARNED_CONTINUOUS_CHECKPOINT_ENV_VARS.get(controller_key)
    env_value = os.environ.get(env_key, "").strip() if env_key else ""
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.exists():
            return str(candidate)
    for candidate in _learned_continuous_model_candidates(controller_key):
        if candidate.exists():
            return str(candidate)
    return None


def load_learned_continuous_agent(
    controller_key: str,
    model_path: str,
    params: ScenarioParams,
    G=None,
    north: float = 0.0,
    south: float = 0.0,
    east: float = 0.0,
    west: float = 0.0,
):
    if controller_key == PPO_CONTINUOUS_CONTROLLER_KEY:
        return load_ppo_continuous_agent(
            model_path,
            params=params,
            G=G,
            north=north,
            south=south,
            east=east,
            west=west,
        )
    if controller_key == SAC_CONTROLLER_KEY:
        return load_sac_agent(
            model_path,
            params=params,
            G=G,
            north=north,
            south=south,
            east=east,
            west=west,
        )
    if controller_key in {IQL_CONTROLLER_KEY, CQL_CONTROLLER_KEY}:
        from offline_rl import load_offline_policy_agent

        return load_offline_policy_agent(
            model_path,
            algorithm=controller_key.split("_", 1)[0],
        )
    raise ValueError(f"Unsupported controller_key={controller_key!r}")


def default_dreamer_checkpoint(controller_key: str = DREAMER_V3_CONTROLLER_KEY) -> str:
    if controller_key == DREAMER_V3_CONTROLLER_KEY:
        discovered = _discover_default_dreamerv3_run_dir()
        if discovered is not None:
            return discovered
    runs_dir = (
        "dreamerv4_runs"
        if controller_key == DREAMER_V4_CONTROLLER_KEY
        else "dreamerv3_runs"
    )
    return str(
        (
            Path(__file__).resolve().parent / runs_dir / "all_scenarios" / "ckpt"
        ).resolve()
    )


def _discover_default_dreamerv3_run_dir() -> str | None:
    runs_root = Path(__file__).resolve().parent / "dreamerv3_runs"
    if not runs_root.exists():
        return None

    candidates: list[tuple[int, float, str, Path]] = []
    for ckpt_dir in runs_root.rglob("ckpt"):
        if not ckpt_dir.is_dir():
            continue

        run_dir = ckpt_dir.parent
        resolved_checkpoint: Path | None = None
        latest = ckpt_dir / "latest"
        if latest.exists():
            try:
                target = (
                    Path(os.readlink(str(latest)))
                    if latest.is_symlink()
                    else Path(latest.read_text(encoding="utf-8").strip())
                )
            except OSError:
                target = None
            if target and str(target):
                candidate = (
                    target if target.is_absolute() else (ckpt_dir / target).resolve()
                )
                if candidate.is_dir() and (candidate / "done").exists():
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

        if resolved_checkpoint is None:
            continue

        run_name = run_dir.name.lower()
        score = 0
        if "aligned" in run_name:
            score += 40
        if "kpi" in run_name:
            score += 15
        if "budget" in run_name:
            score -= 25
        candidates.append(
            (
                score,
                resolved_checkpoint.stat().st_mtime,
                run_dir.name,
                run_dir.resolve(),
            )
        )

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return str(candidates[0][3])


def default_official_dreamerv3_checkpoint() -> str:
    return default_dreamer_checkpoint(DREAMER_V3_CONTROLLER_KEY)


def default_dreamerv4_checkpoint() -> str:
    return default_dreamer_checkpoint(DREAMER_V4_CONTROLLER_KEY)


def resolve_dreamer_checkpoint_path(
    controller_key: str = DREAMER_V3_CONTROLLER_KEY,
    checkpoint_path: Optional[str] = None,
) -> str:
    if checkpoint_path:
        return str(Path(checkpoint_path).expanduser())

    env_key = DREAMER_CHECKPOINT_ENV_VARS.get(controller_key)
    env_value = os.environ.get(env_key, "") if env_key else ""
    if env_value:
        return str(Path(env_value).expanduser())

    default_candidate = Path(default_dreamer_checkpoint(controller_key)).expanduser()
    if default_candidate.exists():
        return str(default_candidate)

    if controller_key == DREAMER_V3_CONTROLLER_KEY:
        discovered_run_dir = _discover_default_dreamerv3_run_dir()
        if discovered_run_dir is not None:
            return discovered_run_dir

    legacy_env = os.environ.get(
        DREAMER_CHECKPOINT_ENV_VARS[DREAMER_V3_CONTROLLER_KEY],
        "",
    )
    if legacy_env:
        return str(Path(legacy_env).expanduser())
    return default_official_dreamerv3_checkpoint()


def load_dreamer_agent(
    controller_key: str = DREAMER_V3_CONTROLLER_KEY,
    checkpoint_path: Optional[str] = None,
    *,
    seed: int = 0,
    step_hours: float = 6.0,
):
    from official_dreamerv3 import load_official_policy_agent

    resolved_path = resolve_dreamer_checkpoint_path(controller_key, checkpoint_path)
    return load_official_policy_agent(
        resolved_path,
        seed=seed,
        step_hours=step_hours,
    )


def load_official_dreamerv3_agent(
    checkpoint_path: Optional[str] = None,
    *,
    seed: int = 0,
    step_hours: float = 6.0,
):
    return load_dreamer_agent(
        DREAMER_V3_CONTROLLER_KEY,
        checkpoint_path,
        seed=seed,
        step_hours=step_hours,
    )


def _find_native_dreamerv4_checkpoints() -> list[str]:
    """Discover native DreamerV4 PyTorch checkpoints in standard locations."""
    simulator_dir = Path(__file__).resolve().parent
    candidates: list[str] = []
    seen: set[str] = set()

    def _safe_sorted_entries(root: Path, *, files_only: bool | None = None) -> list[Path]:
        entries: list[Path] = []
        try:
            for entry in root.iterdir():
                if files_only is True and not entry.is_file():
                    continue
                if files_only is False and not entry.is_dir():
                    continue
                entries.append(entry)
        except OSError:
            return []
        return sorted(
            entries,
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )

    def _resolve_native_checkpoint(path: Path) -> Path | None:
        if path.is_file() and path.suffix == ".pt":
            return path.resolve()
        if not path.is_dir():
            return None

        for filename in DREAMERV4_CHECKPOINT_CANDIDATE_NAMES:
            direct = path / filename
            if direct.is_file():
                return direct.resolve()

        latest = path / "latest"
        if latest.exists():
            try:
                target = (
                    Path(os.readlink(str(latest)))
                    if latest.is_symlink()
                    else Path(latest.read_text(encoding="utf-8").strip())
                )
            except OSError:
                target = None
            if target and str(target):
                latest_candidate = target if target.is_absolute() else (path / target)
                resolved_latest = _resolve_native_checkpoint(latest_candidate.resolve())
                if resolved_latest is not None:
                    return resolved_latest

        for candidate in _safe_sorted_entries(path, files_only=True):
            if candidate.suffix == ".pt":
                return candidate.resolve()

        nested_ckpt_dir = path / "ckpt"
        if nested_ckpt_dir.is_dir():
            resolved_nested = _resolve_native_checkpoint(nested_ckpt_dir)
            if resolved_nested is not None:
                return resolved_nested

        for child_dir in _safe_sorted_entries(path, files_only=False):
            for filename in DREAMERV4_CHECKPOINT_CANDIDATE_NAMES:
                nested = child_dir / filename
                if nested.is_file():
                    return nested.resolve()
            for candidate in _safe_sorted_entries(child_dir, files_only=True):
                if candidate.suffix == ".pt":
                    return candidate.resolve()

        return None

    def _add_candidate(path: Path) -> None:
        resolved = _resolve_native_checkpoint(path)
        if resolved is None:
            return
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        candidates.append(key)

    # Check PIOS_DREAMERV4_CHECKPOINT env var for a file, ckpt dir, or run dir.
    env_path = os.environ.get("PIOS_DREAMERV4_CHECKPOINT", "")
    if env_path:
        _add_candidate(Path(env_path).expanduser())

    runs_root = simulator_dir / "dreamerv4_runs"
    native_dirs = [
        simulator_dir / "dreamerv4_runs" / "native" / "ckpt",
        simulator_dir / "dreamerv4_runs" / "best" / "ckpt",
        simulator_dir / "dreamerv4_runs" / "simulator_compat" / "ckpt",
        simulator_dir / "dreamerv4_runs" / "default" / "ckpt",
    ]
    for d in native_dirs:
        _add_candidate(d)

    if runs_root.exists():
        for run_dir in sorted(runs_root.iterdir()):
            ckpt_dir = run_dir / "ckpt"
            if ckpt_dir.is_dir():
                _add_candidate(ckpt_dir)

    return candidates


def load_dreamerv4_agent(
    checkpoint_path: Optional[str] = None,
    *,
    seed: int = 0,
    step_hours: float = 6.0,
):
    """Load a DreamerV4 agent, preferring a native PyTorch checkpoint.

    Resolution order:
    1. If *checkpoint_path* points to a ``.pt`` file → native PyTorch agent
    2. Try the default native checkpoint location (``dreamerv4_runs/*/ckpt/dreamerv4_agent.pt``)
    3. Fall back to the JAX-based Dreamer loader (``load_dreamer_agent``)
    """
    # --- try explicit native checkpoint path or checkpoint directory ---
    if checkpoint_path:
        try:
            from dreamerv4_agent import (
                load_dreamerv4_pytorch_agent,
                resolve_dreamerv4_checkpoint_artifact,
            )

            resolved = resolve_dreamerv4_checkpoint_artifact(checkpoint_path)
        except (ImportError, FileNotFoundError):
            resolved = None
        if resolved is not None and Path(resolved).exists():
            return load_dreamerv4_pytorch_agent(
                resolved,
                seed=seed,
                step_hours=step_hours,
            )

    # --- try default native checkpoint locations ---
    if checkpoint_path is None:
        native_candidates = _find_native_dreamerv4_checkpoints()
        for candidate in native_candidates:
            if Path(candidate).exists():
                from dreamerv4_agent import load_dreamerv4_pytorch_agent

                return load_dreamerv4_pytorch_agent(
                    candidate, seed=seed, step_hours=step_hours
                )

    # --- fallback to existing JAX-based loader ---
    return load_dreamer_agent(
        DREAMER_V4_CONTROLLER_KEY,
        checkpoint_path,
        seed=seed,
        step_hours=step_hours,
    )


# ===========================================================================
# run_scenario (updated to support both heuristic and trained PPO)
# ===========================================================================


def run_simulation(params, controller, seed=0):
    import simpy

    env = simpy.Environment()

    # initialize state
    state = SimState(params=params, env=env)

    # attach controller
    state.controller = controller

    # run simulation loop
    while env.now < params.episode_hours:
        controller_action = controller(state)

        # apply action (you already have this logic somewhere)
        state.apply_action(controller_action)

        # step simulation
        env.step()

    # build report (adapt to your structure)
    return state.build_report()


def run_scenario(
    params: ScenarioParams,
    G,
    north,
    south,
    east,
    west,
    seed: int = 42,
    enable_logs: bool = True,
    fast_mode: bool = False,
    # ── NEW parameters ────────────────────────────────────────────────
    ppo_agent: Optional[Any] = None,
    ppo_model_path: Optional[str] = None,
    ppo_step_hours: float = 6.0,
    official_dreamerv3_agent: Optional[Any] = None,
    official_dreamerv3_checkpoint: Optional[str] = None,
    official_dreamerv3_step_hours: float = 6.0,
    center_configs: Optional[list[dict[str, Any]]] = None,
) -> SimState:
    """
    Run one complete simulation episode.

    New optional parameters
    -----------------------
    ppo_agent      : a pre-trained PPO instance; if supplied and the strategy
                     is ``ppo_shortage_minimizer``, the *trained* agent drives
                     action selection instead of the hand-crafted heuristic.
    ppo_model_path : alternative to ``ppo_agent`` – path to a saved model that
                     will be loaded automatically.
    ppo_step_hours : how often (sim-hours) the PPO controller fires.
    center_configs : optional list of centre config dicts. If provided, these
                     are used instead of the hardcoded ``CENTER_CONFIGS``.
    """
    learned_continuous_controller_key = (
        params.strategy.controller_key
        if params.strategy.controller_key in LEARNED_CONTINUOUS_CONTROLLER_KEYS
        else None
    )
    ppo_continuous_requested = (
        learned_continuous_controller_key == PPO_CONTINUOUS_CONTROLLER_KEY
    )
    if learned_continuous_controller_key and ppo_agent is None and ppo_model_path is None:
        _resolved = resolve_learned_continuous_model_path(
            learned_continuous_controller_key
        )
        if _resolved is not None:
            ppo_agent = load_learned_continuous_agent(
                learned_continuous_controller_key,
                _resolved,
                params=params,
                G=G,
                north=north,
                south=south,
                east=east,
                west=west,
            )
            if enable_logs:
                print(
                    f"[run_scenario] Loaded {learned_continuous_controller_key} model "
                    f"from {_resolved}"
                )
        else:
            if enable_logs:
                print(
                    f"[run_scenario] {learned_continuous_controller_key} model not found; "
                    "running without controller."
                )

    dreamer_controller_key = (
        params.strategy.controller_key
        if params.strategy.controller_key in DREAMER_CONTROLLER_KEYS
        else DREAMER_V3_CONTROLLER_KEY
    )
    dreamer_requested = (
        official_dreamerv3_agent is not None
        or official_dreamerv3_checkpoint is not None
        or params.strategy.controller_key in DREAMER_CONTROLLER_KEYS
    )
    if (ppo_agent is not None or ppo_model_path is not None) and dreamer_requested:
        raise ValueError(
            "Provide either PPO controller inputs or Dreamer controller inputs, not both."
        )

    # Load agent from disk if a path was given but no live agent provided
    if ppo_agent is None and ppo_model_path is not None:
        if learned_continuous_controller_key:
            ppo_agent = load_learned_continuous_agent(
                learned_continuous_controller_key,
                ppo_model_path,
                params=params,
                G=G,
                north=north,
                south=south,
                east=east,
                west=west,
            )
        else:
            ppo_agent = load_ppo_agent(
                ppo_model_path,
                params=params,
                G=G,
                north=north,
                south=south,
                east=east,
                west=west,
                verbose=1 if enable_logs else 0,
            )
    if official_dreamerv3_agent is None and dreamer_requested:
        official_dreamerv3_agent = load_dreamer_agent(
            dreamer_controller_key,
            official_dreamerv3_checkpoint,
            seed=seed,
            step_hours=official_dreamerv3_step_hours,
        )

    random.seed(seed)
    env = simpy.Environment()
    weather = WeatherEngine(forced_state=params.forced_weather)
    strategy = params.strategy

    centers = []
    base_configs = center_configs if center_configs is not None else CENTER_CONFIGS
    for config in base_configs:
        cfg = dict(config)
        cfg["capacity"] = dict(config.get("capacity") or {})
        cfg["capacity"]["nurses"] = int(cfg["capacity"].get("nurses", 1)) + strategy.extra_nurses
        cfg["capacity"]["lab"] = int(cfg["capacity"].get("lab", 1)) + strategy.extra_lab_staff
        cfg["capacity"]["processing"] = int(cfg["capacity"].get("processing", 1)) + strategy.extra_processing_staff
        if strategy.hours_extension_h and cfg["type"] != "hospital":
            start, end = cfg["hours"]
            extension = strategy.hours_extension_h
            cfg["hours"] = (max(start - extension, 0), min(end + extension, 24))
        centers.append(DonationCenter(env, cfg))

    for idx in range(strategy.mobile_units):
        start_h, end_h = 9.0, 18.0
        if strategy.hours_extension_h:
            start_h = max(start_h - strategy.hours_extension_h, 0)
            end_h = min(end_h + strategy.hours_extension_h, 24)
        extra_cfg = {
            "name": f"Mobile Surge Unit #{idx + 1}",
            "type": "mobile",
            "lat": [46.793, 46.817, 46.846][idx % 3],
            "lon": [-71.262, -71.229, -71.287][idx % 3],
            "capacity": {
                "nurses": 2 + strategy.extra_nurses,
                "lab": 1 + strategy.extra_lab_staff,
                "processing": 1 + strategy.extra_processing_staff,
            },
            "hours": (start_h, end_h),
        }
        centers.append(DonationCenter(env, extra_cfg))

    state = SimState(
        env,
        params,
        centers,
        G,
        north,
        south,
        east,
        west,
        weather,
        rng_demand=random.Random(seed + 10_001),
        rng_process=random.Random(seed + 20_003),
        rng_transport=random.Random(seed + 30_007),
        enable_logs=enable_logs,
        fast_mode=fast_mode,
    )
    initialize_forecast_state(state)
    env.process(donor_generator(state))
    env.process(hospital_demand(state))
    env.process(regional_replenishment(state))
    env.process(demand_forecast_process(state))
    env.process(budget_monitor(state))
    env.process(monitor(state))

    if official_dreamerv3_agent is not None:
        state.runtime_controller = (
            "dreamerv4"
            if dreamer_controller_key == DREAMER_V4_CONTROLLER_KEY
            else "official_dreamerv3"
        )
        state.runtime_controller_step_hours = float(official_dreamerv3_step_hours)
        env.process(
            trained_official_dreamerv3_controller(
                state,
                official_dreamerv3_agent,
                step_hours=official_dreamerv3_step_hours,
                controller=state.runtime_controller,
            )
        )
    elif ppo_agent is not None:
        if ppo_continuous_requested or getattr(ppo_agent, "_continuous", False):
            controller_name = getattr(ppo_agent, "_controller_name", "trained_ppo_continuous")
            state.runtime_controller = controller_name
            state.runtime_controller_step_hours = float(ppo_step_hours)
            if enable_logs:
                print(f"[run_scenario] Using {controller_name} controller.")
            env.process(
                trained_ppo_continuous_controller(
                    state, ppo_agent, step_hours=ppo_step_hours
                )
            )
        else:
            state.runtime_controller = "trained_ppo"
            state.runtime_controller_step_hours = float(ppo_step_hours)
            if enable_logs:
                print("[run_scenario] Using trained PPO discrete controller.")
            env.process(
                trained_ppo_controller(state, ppo_agent, step_hours=ppo_step_hours)
            )
    # ── Register the appropriate controller ─────────────────────────────
    elif strategy.controller_key == "ppo_shortage_minimizer":
        state.runtime_controller = "heuristic_ppo"
        state.runtime_controller_step_hours = float(strategy.controller_interval_h)
        print(">>> USING HEURISTIC PPO <<<")
        env.process(ppo_shortage_controller(state))
        # else:
        #     # Fall back to legacy heuristic when no trained agent is available
        #     env.process(ppo_shortage_controller(state))
        #     if enable_logs:
        #         print(
        #             "[run_scenario] Using legacy heuristic PPO controller (no trained agent supplied)."
        #         )
    elif strategy.controller_key in LEARNED_CONTINUOUS_CONTROLLER_KEYS:
        # If we reach here, the auto-load above didn't find a model
        if enable_logs:
            print(
                f"[run_scenario] {strategy.controller_key} strategy selected but no model loaded."
            )

    if enable_logs:
        print(f"\n{'=' * 68}")
        print(f"Scenario: {params.name}")
        print(f"Strategy: {strategy.name}")
        print(f"{'=' * 68}\n")

    reward_step_hours = 6.0
    if official_dreamerv3_agent is not None:
        reward_step_hours = float(official_dreamerv3_step_hours)
    elif ppo_agent is not None:
        reward_step_hours = float(ppo_step_hours)
    elif strategy.controller_key == "ppo_shortage_minimizer":
        reward_step_hours = float(strategy.controller_interval_h)

    prestock_centers(centers, env, inventory_days=params.initial_inventory_days)
    state.episode_step_hours = reward_step_hours
    state.episode_prev_transfused = total_transfused_units(state)
    state.episode_prev_reward_total = float(
        calculate_reward(
            state,
            prev_transfused=state.episode_prev_transfused,
            update_prev_transfused=False,
        ).total
    )
    env.process(track_episode_score(state, step_hours=reward_step_hours))
    env.run(until=params.sim_hours)
    summarize_state(state)
    return state


def run_with_ppo_agent(
    params: ScenarioParams,
    G,
    north: float,
    south: float,
    east: float,
    west: float,
    agent: Optional[PPO] = None,
    model_path: Optional[str] = None,
    seed: int = 42,
    enable_logs: bool = True,
    fast_mode: bool = False,
    ppo_step_hours: float = 6.0,
) -> SimState:
    """
    Convenience wrapper: run a full episode with a trained PPO agent.

    Provide either ``agent`` (a live PPO object) or ``model_path`` (a path
    written by ``train_ppo_agent``).  Raises ``ValueError`` if neither is given.
    """
    if agent is None and model_path is None:
        raise ValueError("Provide either `agent` or `model_path`.")
    return run_scenario(
        params,
        G,
        north,
        south,
        east,
        west,
        seed=seed,
        enable_logs=enable_logs,
        fast_mode=fast_mode,
        ppo_agent=agent,
        ppo_model_path=model_path,
        ppo_step_hours=ppo_step_hours,
    )


# ===========================================================================
# Reward & summary (unchanged from original)
# ===========================================================================


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _reward_signal_metrics(
    state: SimState,
    *,
    total_transfused: Optional[float] = None,
    total_expired: Optional[float] = None,
) -> dict[str, float]:
    total_transfused = (
        float(total_transfused)
        if total_transfused is not None
        else float(total_transfused_units(state))
    )
    total_expired = (
        float(total_expired)
        if total_expired is not None
        else float(sum(center.stats["expired"] for center in state.centers))
    )
    priority_pressure = max(
        float(state.recent_priority_pressure),
        float(state.forecast_priority_pressure),
    )
    return {
        "shortage_rate": _safe_ratio(
            float(state.total_shortage_units),
            float(state.total_net_requested_units),
        ),
        "service_rate": _safe_ratio(
            total_transfused,
            float(state.total_base_requested_units),
        ),
        "expiry_rate": _safe_ratio(total_expired, total_transfused),
        "priority_shortage_rate": _safe_ratio(
            float(state.priority_weighted_shortage_units),
            float(state.priority_weighted_requested_units),
        ),
        "conservation_rate": _safe_ratio(
            float(state.total_conserved_units),
            float(state.total_base_requested_units),
        ),
        "exact_match_rate": _safe_ratio(
            float(state.total_exact_match_units),
            total_transfused,
        ),
        "compatible_substitution_rate": _safe_ratio(
            float(state.total_compatible_substitution_units),
            total_transfused,
        ),
        "incompatible_fulfillment_rate": _safe_ratio(
            float(state.total_incompatible_fulfillment_units),
            total_transfused,
        ),
        "budget_utilization": budget_pressure(state),
        "priority_pressure": priority_pressure,
    }


def _anticipation_bonus(state: SimState, *, shortage_rate: float) -> float:
    pressure = max(float(state.forecast_pressure) - 1.0, 0.0)
    pressure += 0.8 * max(
        max(
            float(state.recent_priority_pressure),
            float(state.forecast_priority_pressure),
        )
        - 0.18,
        0.0,
    )
    shortage_buffer = max(0.18 - float(shortage_rate), 0.0)
    return pressure * shortage_buffer


def calculate_reward(
    state: SimState,
    *,
    prev_transfused: Optional[int] = None,
    update_prev_transfused: bool = True,
) -> RewardBreakdown:
    weights = state.params.reward_weights

    shortage_by_component = state.shortage_by_component
    total_transfused = sum(center.stats["transfused"] for center in state.centers)
    total_expired = sum(center.stats["expired"] for center in state.centers)

    avg_wait_min = (
        state.wait_time_total / state.wait_time_count * 60.0
        if state.wait_time_count
        else 0.0
    )
    avg_travel_min = (
        state.travel_time_total / state.travel_time_count * 60.0
        if state.travel_time_count
        else 0.0
    )

    active_snapshot = [
        action
        for action in state.active_actions
        if not is_action_expired(action, state.env.now)
    ]
    action_cost = sum(
        budgeted_action_level(state, action) * action.action.operational_cost
        for action in active_snapshot
    )

    terms = {
        "fulfilled_units": total_transfused * weights.fulfilled_unit,
        "shortage_rbc": shortage_by_component.get("RBC", 0) * weights.shortage_rbc,
        "shortage_platelets": shortage_by_component.get("PLATELETS", 0)
        * weights.shortage_platelets,
        "shortage_plasma": shortage_by_component.get("PLASMA", 0)
        * weights.shortage_plasma,
        "priority_shortage": state.priority_weighted_shortage_units
        * weights.priority_shortage_unit,
        "conserved_units": state.total_conserved_units * weights.conserved_unit,
        "expired_units": total_expired * weights.expired_unit,
        "transfer_units": state.total_transfer_units * weights.transfer_unit,
        "exact_match_units": state.total_exact_match_units * weights.exact_match_unit,
        "compatible_substitution_units": state.total_compatible_substitution_units
        * weights.compatible_substitution_unit,
        "incompatible_fulfillment_units": state.total_incompatible_fulfillment_units
        * weights.incompatible_fulfillment_unit,
        "budget_spent": state.budget_spent * weights.budget_spent_unit,
        "budget_blocked_controls": state.budget_blocked_controls
        * weights.budget_blocked_control,
        "budget_exhausted": state.budget_exhausted_hours
        * weights.budget_exhausted_hour,
        "avg_wait": avg_wait_min * weights.avg_wait_minute,
        "avg_travel": avg_travel_min * weights.avg_travel_minute,
        "action_cost": action_cost * weights.action_cost_unit,
    }

    total = float(sum(terms.values()))
    signals = _reward_signal_metrics(
        state,
        total_transfused=float(total_transfused),
        total_expired=float(total_expired),
    )
    shortage_rate = signals["shortage_rate"]
    service_rate = signals["service_rate"]
    expiry_rate = signals["expiry_rate"]
    priority_shortage_rate = signals["priority_shortage_rate"]
    conservation_rate = signals["conservation_rate"]
    exact_match_rate = signals["exact_match_rate"]
    compatible_substitution_rate = signals["compatible_substitution_rate"]
    incompatible_fulfillment_rate = signals["incompatible_fulfillment_rate"]
    budget_utilization = signals["budget_utilization"]

    # ── Override: conservation is beneficial per WHO Patient Blood
    #    Management guidelines — do not penalise it in the total. ──
    total -= terms["conserved_units"]
    terms["conserved_units"] = 0.0

    # ═════════════════════════════════════════════════════════════════
    # KPI-aligned composite bonuses
    #
    # Hierarchy mirrors the per-step training reward and real-world
    # blood-service performance standards (WHO / AABB):
    #   Tier 1 – Patient safety & availability   (~55 %)
    #   Tier 2 – Blood matching quality           (~20 %)
    #   Tier 3 – Waste, logistics & cost          (~25 %)
    #
    # REMOVED (old penalties that punished proactive management):
    #   • conservation_rate penalty  – conservation reduces shortage
    #   • inventory-cap penalty      – safety buffers prevent crises
    #   • emergency-action penalty   – emergency measures save lives
    #   • conservation-excess penalty – penalised good behaviour
    #   • budget_utilization ** 1.35 – over-penalised spending
    # ═════════════════════════════════════════════════════════════════

    # Composite-tier weights. ``reward_tier_weights`` defaults to None → all
    # 1.0, leaving the canonical reward unchanged; the sensitivity sweep sets
    # {"safety","matching","waste"} to reweight the three tiers.
    _tw = getattr(state.params, "reward_tier_weights", None) or {}
    _w_safety = float(_tw.get("safety", 1.0))
    _w_matching = float(_tw.get("matching", 1.0))
    _w_waste = float(_tw.get("waste", 1.0))

    # ── TIER 1 — Patient Safety & Availability ────────────────────
    total += _w_safety * (
        1500.0 * service_rate
        - 4000.0 * shortage_rate
        - 3500.0 * priority_shortage_rate
        - 1500.0 * incompatible_fulfillment_rate
    )

    # ── TIER 2 — Blood Matching Quality ───────────────────────────
    total += _w_matching * (
        800.0 * exact_match_rate
        + 200.0 * compatible_substitution_rate
    )

    # ── TIER 3 — Waste, Logistics & Cost ──────────────────────────
    efficiency = total_transfused / max(total_transfused + total_expired, 1)
    effective_transport = effective_transport_penalty(state)
    _transport_relief_term = 0.0
    if state.params.transport_penalty > 1.15:
        transport_relief = max(
            state.params.transport_penalty - effective_transport, 0.0
        )
        _transport_relief_term = 180.0 * transport_relief
    total += _w_waste * (
        -1200.0 * expiry_rate
        + 700.0 * efficiency
        - 400.0 * budget_utilization
        - 160.0 * congestion_stress(state)
        + _transport_relief_term
    )

    if update_prev_transfused:
        state.prev_transfused = total_transfused
    _ = prev_transfused  # Kept for compatibility with existing reward-delta callers.

    return RewardBreakdown(total=total, terms=terms)


def summarize_state(state: SimState) -> dict:
    total_donated = sum(center.stats["donated"] for center in state.centers)
    total_collected = sum(center.stats.get("collected", 0) for center in state.centers)
    total_rejected = sum(center.stats["rejected"] for center in state.centers)
    total_lab_rejected = sum(
        center.stats.get("lab_rejected", 0) for center in state.centers
    )
    total_no_show = sum(center.stats["no_show"] for center in state.centers)
    total_transfused = sum(center.stats["transfused"] for center in state.centers)
    total_shortage = state.total_shortage_units
    total_expired = sum(center.stats["expired"] for center in state.centers)
    total_inventory = sum(len(center.inventory) for center in state.centers)
    total_external_units = state.total_external_units
    total_conserved = state.total_conserved_units
    total_base_requested = state.total_base_requested_units
    total_net_requested = state.total_net_requested_units
    shortage_rate = total_shortage / max(total_shortage + total_transfused, 1) * 100
    base_shortage_rate = total_shortage / max(total_base_requested, 1) * 100
    # Use collected (blood-draw count) as denominator for rejection rate so
    # that the pipeline edge-effect from quarantine does not inflate the rate.
    # Fall back to donated if collected is not yet tracked (backward compat).
    denom_for_rejection = total_collected if total_collected > 0 else total_donated
    rejection_rate = total_rejected / max(denom_for_rejection + total_rejected, 1) * 100
    avg_travel_min = (
        state.travel_time_total / state.travel_time_count * 60.0
        if state.travel_time_count
        else 0.0
    )
    avg_wait_min = (
        state.wait_time_total / state.wait_time_count * 60.0
        if state.wait_time_count
        else 0.0
    )
    shortage_by_component = state.shortage_by_component.copy()
    shortage_by_hospital = state.shortage_by_hospital.copy()
    state.reward_breakdown = calculate_reward(state)
    return {
        "total_donated": total_collected if total_collected > 0 else total_donated,
        "total_collected": total_collected,
        "total_released": total_donated,
        "total_rejected": total_rejected,
        "total_lab_rejected": total_lab_rejected,
        "total_no_show": total_no_show,
        "total_transfused": total_transfused,
        "total_shortage": total_shortage,
        "total_expired": total_expired,
        "total_inventory": total_inventory,
        "total_external_units": total_external_units,
        "total_base_requested": total_base_requested,
        "total_net_requested": total_net_requested,
        "total_conserved": total_conserved,
        "total_exact_match_units": int(state.total_exact_match_units),
        "total_compatible_substitution_units": int(
            state.total_compatible_substitution_units
        ),
        "total_incompatible_fulfillment_units": int(
            state.total_incompatible_fulfillment_units
        ),
        "shortage_rate": shortage_rate,
        "base_shortage_rate": base_shortage_rate,
        "priority_weighted_shortage": float(state.priority_weighted_shortage_units),
        "priority_weighted_requested": float(state.priority_weighted_requested_units),
        "rejection_rate": rejection_rate,
        "avg_travel_min": avg_travel_min,
        "avg_wait_min": avg_wait_min,
        "forecast_pressure": float(state.forecast_pressure),
        "recent_priority_pressure": float(state.recent_priority_pressure),
        "transport_congestion": float(congestion_stress(state)),
        "budget_total": float(state.budget_total),
        "budget_remaining": float(state.budget_remaining),
        "budget_spent": float(state.budget_spent),
        "budget_exhausted_hours": float(state.budget_exhausted_hours),
        "fast_mode": state.fast_mode,
        "episode_score": float(state.episode_score),
        "active_action_keys": [
            activated.action.key for activated in state.active_actions
        ],
        "active_action_cost": active_action_cost(state),
        "controller_decisions": controller_decision_count(state.policy_log),
        "shortage_by_component": shortage_by_component,
        "shortage_by_hospital": shortage_by_hospital,
        "reward": state.reward_breakdown,
    }


def generate_report(state: SimState, output_path: str = "report.txt"):
    summary = summarize_state(state)
    params = state.params
    strategy = state.strategy
    reward = summary["reward"]
    shortage_by_component = summary["shortage_by_component"]

    lines: list[str] = []

    def add(line: str = ""):
        lines.append(line)

    def fmt_minutes(value: Optional[float]) -> str:
        return f"{value:.1f} min" if value is not None else "n/a (speed mode)"

    add("=" * 72)
    add("QUEBEC CITY BLOOD SUPPLY CHAIN SIMULATION REPORT")
    add("=" * 72)
    add()
    add("SCENARIO")
    add("-" * 72)
    add(f"Name: {params.name}")
    add(f"Duration: {params.sim_hours} h (~{params.sim_hours / 24:.1f} days)")
    add(f"Description: {params.description}")
    if state.fast_mode:
        add("Run mode: fast")
    add()
    add("Narrative:")
    for line in params.narrative.splitlines():
        add(f"  {line}")
    add()
    add("REAL-WORLD CALIBRATION")
    add("-" * 72)
    for line in REAL_WORLD_ASSUMPTIONS_TEXT.splitlines():
        add(line)
    add()
    add("INPUTS")
    add("-" * 72)
    add(f"Donor inter-arrival: {params.donor_inter_arrival_h:.2f} h")
    add(f"Donor show factor: {params.donor_show_factor:.2f}")
    add(f"Eligibility rate: {params.eligible_rate:.0%}")
    add(f"Demand inter-arrival: {params.demand_rate_h:.2f} h")
    add(f"Demand surge factor: {params.demand_surge_factor:.2f}x")
    add(f"Average units per order: {params.avg_units_per_order:.2f}")
    add(f"Initial inventory buffer: {params.initial_inventory_days:.1f} days")
    add(f"Reserve target: {params.reserve_target_days:.1f} days")
    add(
        f"Provincial replenishment rate: {params.regional_replenishment_rate:.0%} of daily demand"
    )
    add(f"Transport penalty: {params.transport_penalty:.2f}x")
    add(f"Effective transport penalty: {effective_transport_penalty(state):.2f}x")
    add(f"Forecast refresh interval: {params.demand_forecast_interval_h:.1f} h")
    add(f"Demand forecast noise: {params.demand_forecast_noise:.2f}")
    add(f"Episode intervention budget: {params.episode_budget:.0f}")
    add(f"Forced weather: {params.forced_weather or 'dynamic winter mix'}")
    add()
    add("ACTIONS")
    add("-" * 72)
    add(f"Selected strategy: {strategy.name}")
    add(f"Description: {strategy.description}")
    add(f"Action cost: {summary['active_action_cost']:.0f}")
    add(
        f"Budget remaining: {summary['budget_remaining']:.0f} / {summary['budget_total']:.0f}"
    )
    runtime_controller = getattr(state, "runtime_controller", None)
    if strategy.is_adaptive or runtime_controller:
        controller_name = runtime_controller or strategy.controller_key
        add(f"Controller: {controller_name}")
        if controller_name == "trained_ppo":
            add("  (driven by a trained NumPy-PPO / SB3-PPO agent)")
        elif controller_name == "heuristic_ppo":
            add("  (driven by legacy heuristic logit controller)")
        elif controller_name == "official_dreamerv3":
            add("  (driven by an official DreamerV3 checkpoint)")
        elif controller_name == "dreamerv4":
            add("  (driven by a DreamerV4-compatible checkpoint)")
        if strategy.is_adaptive:
            add(
                "Controller action space: "
                + ", ".join(action.key for action in strategy.action_space)
            )
    if state.active_actions:
        for activated_action in state.active_actions:
            action = activated_action.action
            activation_note = ""
            if strategy.is_adaptive or runtime_controller:
                activation_note = (
                    f", activated at {activated_action.activated_at_h:.0f}h"
                )
            add(
                f"  - {action.name}: {action.description} "
                f"(cost {action.operational_cost:.0f}{activation_note})"
            )
            if action.activation_delay_h or action.ramp_h:
                add(
                    f"    activation after {action.activation_delay_h:.0f}h from trigger, "
                    f"full effect by "
                    f"{activated_action.activated_at_h + action.activation_delay_h + action.ramp_h:.0f}h"
                )
    else:
        add("  - No intervention actions applied.")
    add(
        f"  - Donor arrival factor: {dynamic_strategy_factor(state, 'donor_arrival_factor'):.2f}x inter-arrival"
    )
    add(
        f"  - Donor show-up factor: {dynamic_strategy_factor(state, 'donor_show_up_factor'):.2f}x"
    )
    add(
        f"  - Donor eligibility factor: {dynamic_strategy_factor(state, 'donor_eligibility_factor'):.2f}x"
    )
    add(
        f"  - Demand management factor: {dynamic_strategy_factor(state, 'demand_management_factor'):.2f}x"
    )
    add(
        f"  - Committed donor share: {dynamic_strategy_sum(state, 'committed_donor_share'):.0%}"
    )
    add(
        f"  - Release delay factor: {dynamic_strategy_factor(state, 'release_delay_factor'):.2f}x"
    )
    add(
        f"  - Extra staffing per site: nurses +"
        f"{dynamic_strategy_sum(state, 'extra_nurses'):.1f}, "
        f"lab +{dynamic_strategy_sum(state, 'extra_lab_staff'):.1f}, "
        f"processing +{dynamic_strategy_sum(state, 'extra_processing_staff'):.1f}"
    )
    add(f"  - Extra mobile units: {dynamic_strategy_sum(state, 'mobile_units'):.1f}")
    add(f"  - Added hours per side: {active_hours_extension_h(state):.1f}")
    add(
        f"  - Replenishment cadence: every {active_replenishment_interval_h(state):.0f}h"
    )
    if dynamic_strategy_factor(state, "transport_relief_factor") < 1.0:
        add(
            f"  - Emergency transport relief: "
            f"{1 - dynamic_strategy_factor(state, 'transport_relief_factor'):.0%}"
        )
    if dynamic_strategy_factor(state, "regional_replenishment_factor") > 1.0:
        add(
            f"  - Provincial replenishment boost: "
            f"{dynamic_strategy_factor(state, 'regional_replenishment_factor') - 1:.0%}"
        )
    if state.policy_log:
        add(
            f"  - Controller decision count: "
            f"{sum(1 for entry in state.policy_log if entry.get('kind') in {'decision', 'continuous_decision'})}"
        )
    add()
    add("OUTPUTS")
    add("-" * 72)
    add(
        f"Donor attempts: {summary['total_donated'] + summary['total_rejected'] + summary['total_no_show']}"
    )
    add(f"Successful donations: {summary['total_donated']}")
    add(
        f"Deferred/rejected: {summary['total_rejected']} ({summary['rejection_rate']:.1f}%)"
    )
    add(f"No-shows: {summary['total_no_show']}")
    add(f"Base requested units: {summary['total_base_requested']}")
    add(f"Deferred/conserved units: {summary['total_conserved']}")
    add(f"Net requested units: {summary['total_net_requested']}")
    add(f"Transfused units: {summary['total_transfused']}")
    add(f"Shortage units: {summary['total_shortage']}")
    add(f"Shortage rate: {summary['shortage_rate']:.1f}%")
    add(f"Shortage vs base demand: {summary['base_shortage_rate']:.1f}%")
    add(f"Expired units: {summary['total_expired']}")
    add(f"Transfer units: {state.total_transfer_units}")
    add(f"Provincial inbound units: {summary['total_external_units']}")
    add(f"Remaining inventory: {summary['total_inventory']}")
    add(f"Average donor travel: {fmt_minutes(summary['avg_travel_min'])}")
    add(f"Average nurse wait: {fmt_minutes(summary['avg_wait_min'])}")
    add()
    add("REWARD BREAKDOWN")
    add("-" * 72)
    for key, value in reward.terms.items():
        add(f"{key:>20}: {value:>8.1f}")
    add(f"{'total_reward':>20}: {reward.total:>8.1f}")
    add()
    add("SHORTAGES BY COMPONENT")
    add("-" * 72)
    for component, count in shortage_by_component.items():
        add(f"{component:>12}: {count}")
    add()
    add("PER-CENTRE PERFORMANCE")
    add("-" * 72)
    for center in state.centers:
        inventory = center.inventory_by_component()
        avg_travel = (
            center.stats["travel_time_total"] / center.stats["travel_time_count"] * 60.0
            if center.stats["travel_time_count"]
            else 0.0
        )
        add(f"{center.name} [{center.ctype}]")
        add(
            f"  donated={center.stats['donated']} rejected={center.stats['rejected']} "
            f"no_show={center.stats['no_show']} orders={center.stats['orders']}"
        )
        add(
            f"  transfused={center.stats['transfused']} shortage={center.stats['shortage']} "
            f"transfers_in={center.stats['transfers_in']} expired={center.stats['expired']}"
        )
        add(
            f"  inventory RBC={inventory['RBC']} PLT={inventory['PLATELETS']} "
            f"PLS={inventory['PLASMA']} avg_travel={avg_travel:.1f}m"
        )
        add()
    add("INVENTORY TIMELINE")
    add("-" * 72)
    if state.hourly_inventory:
        add(f"{'Day':>4} {'RBC':>8} {'PLT':>8} {'PLS':>8} {'CumShort':>10}")
        for snap in state.hourly_inventory:
            add(
                f"{int(snap['hour'] // 24):>4} {snap['RBC']:>8} {snap['PLATELETS']:>8} "
                f"{snap['PLASMA']:>8} {snap['shortages']:>10}"
            )
    else:
        add("Timeline capture disabled in fast mode.")
    add()
    add("RECOMMENDATIONS")
    add("-" * 72)
    if summary["shortage_rate"] > 15:
        add(
            "Escalate action mix. The current intervention still leaves too many unmet units."
        )
    else:
        add(
            "Current action mix keeps shortage pressure within the playable target band."
        )
    if summary["total_conserved"] > max(summary["total_transfused"] * 0.40, 1):
        add(
            "A large share of demand was deferred by blood conservation rules. Check whether that trade-off is acceptable."
        )
    if summary["total_external_units"] == 0:
        add(
            "No provincial replenishment arrived. Check reserve targets and inbound-rate settings."
        )
    if shortage_by_component.get("PLATELETS", 0) > shortage_by_component.get("RBC", 0):
        add(
            "Platelets remain the most fragile component. Prioritise frequent, smaller collections."
        )
    if summary["total_expired"] > max(summary["total_transfused"] * 0.12, 1):
        add(
            "Expiry is too high relative to use. Consider reducing campaign intensity or moving stock earlier."
        )
    if state.total_transfer_units > max(summary["total_transfused"] * 0.30, 1):
        add(
            "Transfers are carrying too much of the load. More local collection capacity would reduce fragility."
        )
    add()
    add(f"Report generated: {datetime.now():%Y-%m-%d %H:%M:%S}")

    report_text = "\n".join(lines)
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)
    print(f"\nReport saved -> {output_path}")
    return report_text


def apply_multi_actions(state, actions, action_keys=None, allowed_action_keys=None):
    if action_keys is None:
        action_keys = ACTION_KEYS
    if allowed_action_keys is None:
        allowed_action_keys = {
            "supply": {"campaign", "lab_fast_track"},
            "hospital": {"clinical_conservation"},
            "logistics": {"emergency_share", "national_mutual_aid"},
        }

    prune_expired_actions(state)
    if len(state.active_actions) >= len(ACTION_CATALOG):
        return

    already_active = {a.action.key for a in state.active_actions}

    for agent_name, a in actions.items():
        if a == 0:
            continue

        if a - 1 >= len(action_keys):
            continue

        choice_key = action_keys[a - 1]
        action_members = [member for member in str(choice_key).split("+") if member]

        allowed = allowed_action_keys.get(agent_name)
        if allowed is not None and not any(
            member in allowed for member in action_members
        ):
            continue

        for action_key in action_members:
            if allowed is not None and action_key not in allowed:
                continue
            if action_key in already_active:
                continue
            if len(state.active_actions) >= len(ACTION_CATALOG):
                return

            action = ACTION_CATALOG[action_key]
            state.active_actions.append(
                ActivatedAction(action=_timing_scaled(action, state.params), activated_at_h=state.env.now)
            )
            already_active.add(action_key)


def step_reward_snapshot(state: SimState) -> dict[str, float]:
    return {
        "transfused": float(sum(c.stats["transfused"] for c in state.centers)),
        "shortage": float(state.total_shortage_units),
        "expired": float(sum(c.stats["expired"] for c in state.centers)),
        "exact_match": float(state.total_exact_match_units),
        "compatible_substitution": float(state.total_compatible_substitution_units),
        "incompatible_fulfillment": float(state.total_incompatible_fulfillment_units),
        "conserved": float(state.total_conserved_units),
        "priority_shortage": float(state.priority_weighted_shortage_units),
        "budget_spent": float(state.budget_spent),
        "budget_blocked_controls": float(state.budget_blocked_controls),
        "congestion": float(congestion_stress(state)),
    }


def calculate_step_reward(state: SimState, prev_snapshot: dict) -> float:
    """
    Per-step reward for DreamerV3 aligned with real-world blood supply chain KPIs.

    Design principles
    -----------------
    1. **Pure delta-based (Markovian):** only what changed THIS step matters.
    2. **Hierarchical weights:** Safety > Matching quality > Waste & cost,
       mirroring WHO / AABB blood-service performance standards.
    3. **Smoothly bounded via tanh:** no hard clip, gradient alive everywhere.
    4. **Few, clear terms** the world model can learn to predict.

    Real-world KPIs targeted
    ------------------------
    - Fill Rate  (shortage minimisation)   — WHO primary supply metric
    - Priority Patient Coverage            — emergency / surgical readiness
    - Transfusion Safety                   — ABO/Rh compatibility
    - Exact-Match Rate                     — best clinical outcomes
    - Wastage Rate  (expiry minimisation)  — AABB quality standard
    - Cost Efficiency                      — sustainable operations
    """
    total_transfused = sum(c.stats["transfused"] for c in state.centers)
    total_expired = sum(c.stats["expired"] for c in state.centers)

    # ── per-step deltas (clamped ≥ 0 for rate computation) ────────────
    d_trans = max(total_transfused - prev_snapshot["transfused"], 0.0)
    d_short = max(state.total_shortage_units - prev_snapshot["shortage"], 0.0)
    d_exp = max(total_expired - prev_snapshot.get("expired", 0.0), 0.0)
    d_exact = max(
        state.total_exact_match_units - prev_snapshot.get("exact_match", 0.0), 0.0
    )
    d_compat = max(
        state.total_compatible_substitution_units
        - prev_snapshot.get("compatible_substitution", 0.0),
        0.0,
    )
    d_incompat = max(
        state.total_incompatible_fulfillment_units
        - prev_snapshot.get("incompatible_fulfillment", 0.0),
        0.0,
    )
    d_pri_short = max(
        state.priority_weighted_shortage_units
        - prev_snapshot.get("priority_shortage", 0.0),
        0.0,
    )

    # ── this-step demand & supply totals ──────────────────────────────
    step_demand = d_trans + d_short  # units demanded this step
    step_supply = d_trans + d_exp  # units consumed or wasted

    # Quiet step (no meaningful activity) → neutral reward
    if step_demand < 0.5 and step_supply < 0.5:
        return 0.0

    # ── bounded per-step rates [0, 1] ─────────────────────────────────
    fill_rate = d_trans / max(step_demand, 1.0)
    match_rate = d_exact / max(d_trans, 1.0)
    compat_rate = d_compat / max(d_trans, 1.0)
    incompat_rate = d_incompat / max(d_trans, 1.0)
    wastage_rate = d_exp / max(step_supply, 1.0)
    pri_short_rate = d_pri_short / max(step_demand, 1.0)

    # ═════════════════════════════════════════════════════════════════
    #  TIER 1 — Patient Safety & Availability            (weight ≈55%)
    # ═════════════════════════════════════════════════════════════════
    r = 1.00 * fill_rate  # meeting demand — the #1 KPI
    r -= 0.60 * pri_short_rate  # critical-patient shortage
    r -= 0.40 * incompat_rate  # ABO/Rh mismatch risk

    # ═════════════════════════════════════════════════════════════════
    #  TIER 2 — Blood Matching Quality                   (weight ≈20%)
    # ═════════════════════════════════════════════════════════════════
    r += 0.25 * match_rate  # exact type match preferred
    r += 0.05 * compat_rate  # acceptable substitution

    # ═════════════════════════════════════════════════════════════════
    #  TIER 3 — Waste & Operational Cost                 (weight ≈25%)
    # ═════════════════════════════════════════════════════════════════
    r -= 0.35 * wastage_rate  # expired units = wasted donations

    # Budget: penalise only overspend vs expected per-step budget
    budget_delta = max(state.budget_spent - prev_snapshot.get("budget_spent", 0.0), 0.0)
    expected_step_budget = state.params.episode_budget / max(
        state.params.sim_hours / 6.0, 1.0
    )
    budget_overspend = max(budget_delta - expected_step_budget, 0.0) / max(
        expected_step_budget, 1.0
    )
    r -= 0.08 * min(budget_overspend, 1.0)

    # Transport / logistics congestion
    r -= 0.07 * min(congestion_stress(state), 1.0)

    # ── centre so "good performance" ≈ 0 ─────────────────────────────
    # Perfect  step ≈ 1.30  →  after shift 0.75  → tanh(1.88) ≈  0.95
    # Good     step ≈ 1.10  →  after shift 0.55  → tanh(1.38) ≈  0.88
    # Average  step ≈ 0.83  →  after shift 0.28  → tanh(0.70) ≈  0.60
    # Poor     step ≈ 0.25  →  after shift –0.30 → tanh(–0.75)≈ –0.64
    # Terrible step ≈ –0.57 →  after shift –1.12 → tanh(–2.80)≈ –0.99
    r -= 0.55

    # Smooth bounding — tanh keeps gradient alive at all performance levels
    return float(np.tanh(r * 2.5))


def calculate_step_reward_budget(state: SimState, prev_snapshot: dict) -> float:
    """
    Budget-aware per-step reward for DreamerV3.

    Same core supply-chain KPIs as :func:`calculate_step_reward`, but with
    budget spent and action cost as **first-class** optimisation targets.
    The agent must balance supply-chain performance against frugal budget use.

    Key differences from ``calculate_step_reward``
    -----------------------------------------------
    - Action cost incurs a proportional per-step penalty.
    - Budget overspend weight is ~3× heavier (0.25 vs 0.08).
    - A *progressive* budget-exhaustion penalty grows with the fraction of
      remaining simulation time — running out early hurts much more.
    - Quiet steps still incur a cost penalty if the agent is burning budget.
    - Tier 3 (Waste & Cost) receives ~45 % of the reward weight (up from ~25 %).

    Real-world motivation
    ---------------------
    Blood-service networks operate under constrained budgets.  Activating
    expensive interventions (mobile units, national mutual aid, surge staff)
    should only happen when the marginal patient-safety benefit justifies the
    spend.  This reward trains an agent that is both *effective* and *frugal*.
    """
    total_transfused = sum(c.stats["transfused"] for c in state.centers)
    total_expired = sum(c.stats["expired"] for c in state.centers)

    # ── per-step deltas ───────────────────────────────────────────────
    d_trans = max(total_transfused - prev_snapshot["transfused"], 0.0)
    d_short = max(state.total_shortage_units - prev_snapshot["shortage"], 0.0)
    d_exp = max(total_expired - prev_snapshot.get("expired", 0.0), 0.0)
    d_exact = max(
        state.total_exact_match_units - prev_snapshot.get("exact_match", 0.0), 0.0
    )
    d_compat = max(
        state.total_compatible_substitution_units
        - prev_snapshot.get("compatible_substitution", 0.0),
        0.0,
    )
    d_incompat = max(
        state.total_incompatible_fulfillment_units
        - prev_snapshot.get("incompatible_fulfillment", 0.0),
        0.0,
    )
    d_pri_short = max(
        state.priority_weighted_shortage_units
        - prev_snapshot.get("priority_shortage", 0.0),
        0.0,
    )

    step_demand = d_trans + d_short
    step_supply = d_trans + d_exp

    # -- budget book-keeping (shared by quiet and active paths) --------
    budget_delta = max(state.budget_spent - prev_snapshot.get("budget_spent", 0.0), 0.0)
    expected_step_budget = state.params.episode_budget / max(
        state.params.sim_hours / 6.0, 1.0
    )

    # Quiet step — still penalise ongoing budget burn
    if step_demand < 0.5 and step_supply < 0.5:
        if budget_delta > 0.0:
            cost_rate = budget_delta / max(expected_step_budget * 2.0, 1.0)
            return float(np.tanh(-0.15 * min(cost_rate, 1.0) * 2.5))
        return 0.0

    # ── bounded per-step rates [0, 1] ─────────────────────────────────
    fill_rate = d_trans / max(step_demand, 1.0)
    match_rate = d_exact / max(d_trans, 1.0)
    compat_rate = d_compat / max(d_trans, 1.0)
    incompat_rate = d_incompat / max(d_trans, 1.0)
    wastage_rate = d_exp / max(step_supply, 1.0)
    pri_short_rate = d_pri_short / max(step_demand, 1.0)

    # ═════════════════════════════════════════════════════════════════
    #  TIER 1 — Patient Safety & Availability           (weight ≈40 %)
    # ═════════════════════════════════════════════════════════════════
    r = 0.80 * fill_rate
    r -= 0.50 * pri_short_rate
    r -= 0.30 * incompat_rate

    # ═════════════════════════════════════════════════════════════════
    #  TIER 2 — Blood Matching Quality                  (weight ≈15 %)
    # ═════════════════════════════════════════════════════════════════
    r += 0.20 * match_rate
    r += 0.04 * compat_rate

    # ═════════════════════════════════════════════════════════════════
    #  TIER 3 — Waste & Budget / Cost Efficiency        (weight ≈45 %)
    # ═════════════════════════════════════════════════════════════════
    r -= 0.25 * wastage_rate

    # Budget overspend — 3× heavier than the standard reward (0.25 vs 0.08)
    budget_overspend = max(budget_delta - expected_step_budget, 0.0) / max(
        expected_step_budget, 1.0
    )
    r -= 0.25 * min(budget_overspend, 1.0)

    # Action cost proportional penalty — every dollar spent hurts
    action_cost_rate = budget_delta / max(expected_step_budget * 2.0, 1.0)
    r -= 0.15 * min(action_cost_rate, 1.0)

    # Budget exhaustion progressive penalty — running out early is very bad
    if state.budget_remaining <= 1e-6 and state.budget_total > 0:
        remaining_time_frac = max(
            (state.params.sim_hours - state.env.now) / max(state.params.sim_hours, 1.0),
            0.0,
        )
        # The earlier the budget runs out, the bigger the penalty
        r -= 0.40 * remaining_time_frac

    # Transport / logistics congestion
    r -= 0.07 * min(congestion_stress(state), 1.0)

    # ── centre so "good frugal performance" ≈ 0 ──────────────────────
    r -= 0.45

    return float(np.tanh(r * 2.5))


def step_simulation(state, step_hours=6.0):
    target = state.env.now + step_hours
    state.env.run(until=target)
