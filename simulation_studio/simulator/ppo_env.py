"""
ppo_env.py  –  Gymnasium-compatible environment wrapping the blood-supply
               chain simulation for PPO training.

Circular-import fix
-------------------
This file no longer imports from engine.py at module level.
All shared constants/helpers live in ppo_shared.py.
engine.py symbols are imported lazily inside functions (after engine is
fully initialised) so Python never sees a partially-initialised module.

Observation space  (16 continuous features, all clamped to [0, 1]):
    0  recent_shortage_rate
    1  cumulative_shortage_rate
    2  critical_gap
    3  rbc_gap
    4  platelet_gap
    5  plasma_gap
    6  transport_stress   / 3
    7  weather_stress
    8  donor_stress       / 3
    9  demand_stress      / 3
   10  recent_service_pressure / 5
   11  active_cost_ratio
   12  noisy demand forecast signal
   13  priority pressure
   14  congestion stress
   15  budget pressure

Action space
    discrete     -> Discrete(len(ACTION_CATALOG) + 1)
                    0      … NOOP
                    1 … N  … activate the i-th action from ACTION_CATALOG
    continuous   -> Box(-1, 1, shape=(DREAMER_ACTION_DIM,))
                    each dimension is the intensity of one intervention dial;
                    -1 means off, 0 means medium intensity, +1 means max.

Reward
    discrete     -> delta in calculate_reward().total between steps, normalised
    continuous   -> per-step Dreamer reward from simulator deltas only
"""

from __future__ import annotations

import copy
import random
from typing import Any, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Optional gymnasium shim
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ModuleNotFoundError:
    _GYM_AVAILABLE = False

    class _FakeSpace:
        pass

    class spaces:  # type: ignore[no-redef]
        Box = _FakeSpace
        Discrete = _FakeSpace

    class gym:  # type: ignore[no-redef]
        class Env:
            pass


import simpy
from calibration import (  # type: ignore
    COMPONENT_DEMAND_WEIGHTS,
    QC_DAILY_LABILE_PRODUCTS_EST,
)

# Only import modules that do NOT depend on engine or ppo_env
from core import CENTER_CONFIGS, DonationCenter, WeatherEngine  # type: ignore
from ppo_shared import (  # type: ignore
    ACTION_KEYS,
    CONTINUOUS_ACTION_HIGH,
    CONTINUOUS_ACTION_LOW,
    DREAMER_ACTION_DIM,
    N_ACTIONS,
    OBS_DIM,
    build_observation,
    clamp_obs,
    continuous_action_levels,
)
from scenarios import ACTION_CATALOG, ScenarioParams  # type: ignore

__all__ = [
    "ACTION_KEYS",
    "N_ACTIONS",
    "OBS_DIM",
    "clamp_obs",
    "BloodSupplyEnv",
    "build_sim",
]


# ---------------------------------------------------------------------------
# Lazy engine helpers
# ---------------------------------------------------------------------------


def _lazy_engine():
    """Return the engine module, importing it lazily to avoid circular deps."""
    import engine as _eng  # noqa: PLC0415

    return _eng


# ---------------------------------------------------------------------------
# build_sim: constructs a fresh SimState without running it
# ---------------------------------------------------------------------------


def build_sim(
    params: ScenarioParams,
    G,
    north: float,
    south: float,
    east: float,
    west: float,
    seed: int,
):
    """
    Construct and pre-stock a new SimState without running it.
    Imports engine lazily so this module can be imported before engine is
    fully initialised.
    """
    eng = _lazy_engine()

    random.seed(seed)
    env = simpy.Environment()
    weather = WeatherEngine(forced_state=params.forced_weather)
    strategy = params.strategy

    centers: list[DonationCenter] = []
    for config in CENTER_CONFIGS:
        cfg = dict(config)
        cfg["capacity"] = dict(config["capacity"])
        cfg["capacity"]["nurses"] += strategy.extra_nurses
        cfg["capacity"]["lab"] += strategy.extra_lab_staff
        cfg["capacity"]["processing"] += strategy.extra_processing_staff
        if strategy.hours_extension_h and cfg["type"] != "hospital":
            start, end = cfg["hours"]
            ext = strategy.hours_extension_h
            cfg["hours"] = (max(start - ext, 0), min(end + ext, 24))
        centers.append(DonationCenter(env, cfg))

    for idx in range(strategy.mobile_units):
        start_h, end_h = 9.0, 18.0
        if strategy.hours_extension_h:
            start_h = max(start_h - strategy.hours_extension_h, 0)
            end_h = min(end_h + strategy.hours_extension_h, 24)
        centers.append(
            DonationCenter(
                env,
                {
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
                },
            )
        )

    state = eng.SimState(
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
        enable_logs=False,
        fast_mode=True,
    )
    eng.initialize_forecast_state(state)

    env.process(eng.donor_generator(state))
    env.process(eng.hospital_demand(state))
    env.process(eng.regional_replenishment(state))
    env.process(eng.demand_forecast_process(state))
    env.process(eng.budget_monitor(state))
    env.process(eng.monitor(state))

    eng.prestock_centers(centers, env, inventory_days=params.initial_inventory_days)
    return state


# ---------------------------------------------------------------------------
# Observation / snapshot helpers (no engine import at call time)
# ---------------------------------------------------------------------------


def _policy_snapshot(state) -> dict[str, float]:
    total_transfused = sum(c.stats["transfused"] for c in state.centers)
    return {
        "shortage": float(state.total_shortage_units),
        "requested": float(state.total_net_requested_units),
        "transfused": float(total_transfused),
    }


def _get_coverage(state) -> dict[str, float]:
    totals: dict[str, int] = {"RBC": 0, "PLATELETS": 0, "PLASMA": 0}
    for center in state.centers:
        for comp, units in center.inventory_by_component().items():
            totals[comp] += units
    return {
        comp: totals[comp]
        / max(QC_DAILY_LABILE_PRODUCTS_EST * COMPONENT_DEMAND_WEIGHTS[comp], 1.0)
        for comp in ["RBC", "PLATELETS", "PLASMA"]
    }


def _active_cost(state) -> float:
    return _lazy_engine().active_action_cost(state)


def _effective_transport_penalty(state) -> float:
    return _lazy_engine().effective_transport_penalty(state)


def _obs_from_state(state, prev_snapshot: dict) -> np.ndarray:
    eng = _lazy_engine()
    total_transfused = sum(c.stats["transfused"] for c in state.centers)
    raw = build_observation(
        total_shortage_units=state.total_shortage_units,
        total_net_requested_units=state.total_net_requested_units,
        total_transfused=total_transfused,
        coverage=_get_coverage(state),
        reserve_target_days=state.params.reserve_target_days,
        donor_inter_arrival_h=state.params.donor_inter_arrival_h,
        donor_show_factor=state.params.donor_show_factor,
        demand_rate_h=state.params.demand_rate_h / max(state.demand_pressure, 1e-6),
        demand_surge_factor=state.params.demand_surge_factor * state.demand_pressure,
        effective_transport_penalty=_effective_transport_penalty(state),
        weather_state=state.weather.state,
        prev_shortage=prev_snapshot["shortage"],
        prev_requested=prev_snapshot["requested"],
        prev_transfused=prev_snapshot["transfused"],
        active_cost=_active_cost(state),
        forecast_pressure=state.forecast_pressure,
        priority_pressure=max(state.recent_priority_pressure, state.forecast_priority_pressure),
        congestion_stress=min(eng.congestion_stress(state), 1.0),
        budget_pressure=eng.budget_pressure(state),
    )
    return clamp_obs(raw)


# ---------------------------------------------------------------------------
# The Gym environment
# ---------------------------------------------------------------------------


class BloodSupplyEnv(gym.Env if _GYM_AVAILABLE else object):  # type: ignore[misc]
    """
    Gymnasium environment for PPO training on the blood supply chain.

    Parameters
    ----------
    params        : ScenarioParams
    scenario_templates : optional list of ScenarioParams sampled across episodes
    G             : networkx road graph
    north/south/east/west : city bounding box
    episode_hours : total sim-hours per episode  (default 168 = 1 week)
    step_hours    : sim-hours advanced per step   (default 6)
    seed          : base RNG seed
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: ScenarioParams,
        G,
        north: float,
        south: float,
        east: float,
        west: float,
        episode_hours: float = 168.0,
        step_hours: float = 6.0,
        seed: int = 42,
        scenario_templates: Optional[Sequence[ScenarioParams]] = None,
        action_mode: str = "discrete",
        collect_diagnostics: bool = True,
    ) -> None:
        super().__init__()
        templates = list(scenario_templates or [params])
        if not templates:
            raise ValueError("BloodSupplyEnv requires at least one scenario template.")
        self._scenario_templates = [copy.deepcopy(template) for template in templates]
        self._scenario_index = 0
        self._scenario_key = str(self._scenario_templates[0].scenario_key)
        self._episodes_started = 0
        self.params = copy.deepcopy(self._scenario_templates[0])
        self.G = G
        self.north = north
        self.south = south
        self.east = east
        self.west = west
        self.episode_hours = episode_hours
        self.step_hours = step_hours
        self._base_seed = seed
        self._episode_seed = seed
        self.action_mode = str(action_mode).strip().lower()
        self.collect_diagnostics = bool(collect_diagnostics)
        if self.action_mode not in {"discrete", "continuous"}:
            raise ValueError(
                f"Unsupported action_mode={action_mode!r}. Expected 'discrete' or 'continuous'."
            )

        if _GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
            )
            if self.action_mode == "continuous":
                self.action_space = spaces.Box(
                    low=CONTINUOUS_ACTION_LOW,
                    high=CONTINUOUS_ACTION_HIGH,
                    shape=(DREAMER_ACTION_DIM,),
                    dtype=np.float32,
                )
            else:
                self.action_space = spaces.Discrete(N_ACTIONS)

        self._state: Optional[object] = None
        self._prev_reward: float = 0.0
        self._prev_snapshot: dict = {
            "shortage": 0.0,
            "requested": 0.0,
            "transfused": 0.0,
        }
        self._elapsed_hours: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._episode_seed = seed
        else:
            self._episode_seed += 1

        self._scenario_index = self._select_scenario_index()
        self.params = copy.deepcopy(self._scenario_templates[self._scenario_index])
        self._scenario_key = str(self.params.scenario_key)
        self._episodes_started += 1

        self._state = build_sim(
            self.params,
            self.G,
            self.north,
            self.south,
            self.east,
            self.west,
            seed=self._episode_seed,
        )
        self._prev_snapshot = _policy_snapshot(self._state)
        self._prev_reward = _lazy_engine().calculate_reward(self._state).total
        self._elapsed_hours = 0.0

        obs = _obs_from_state(self._state, self._prev_snapshot)
        info = self._info()
        if _GYM_AVAILABLE:
            return obs, info
        return obs

    def step(self, action):
        assert self._state is not None, "Call reset() before step()."
        state = self._state
        env = state.env
        prev_step_snapshot = None


        # 1. Apply action
        action_activated = False
        action_levels = None
        eng = _lazy_engine()
        if self.action_mode == "continuous":
            prev_step_snapshot = eng.step_reward_snapshot(state)
            action_levels = continuous_action_levels(action)
            result = eng.apply_continuous_action_levels(
                state,
                action_levels,
                controller="dreamer_env",
                record_policy=False,
            )
            action_activated = bool(result["changed"])
        elif action > 0:
            action_key = ACTION_KEYS[int(action) - 1]
            already_active = any(
                a.action.key == action_key for a in state.active_actions
            )
            if not already_active:
                act_obj = ACTION_CATALOG[action_key]
                state.active_actions.append(
                    eng.ActivatedAction(action=act_obj, activated_at_h=env.now)
                )
                action_activated = True

        # 2. Advance clock
        target = min(env.now + self.step_hours, self.episode_hours)
        env.run(until=target)
        self._elapsed_hours = env.now

        # 3. Reward delta
        previous_snapshot = dict(self._prev_snapshot)
        if self.action_mode == "continuous":
            reward = eng.calculate_step_reward(state, prev_step_snapshot)
        else:
            current_reward = eng.calculate_reward(state).total
            reward = (current_reward - self._prev_reward) / max(
                abs(self._prev_reward) + 1.0, 1.0
            )
            self._prev_reward = current_reward
        current_snapshot = _policy_snapshot(state)

        # 4. Termination
        terminated = False
        truncated = env.now >= self.episode_hours

        obs = _obs_from_state(state, previous_snapshot)
        self._prev_snapshot = current_snapshot
        info = self._info(
            action_activated=action_activated,
            action_levels=action_levels,
        )

        if _GYM_AVAILABLE:
            return obs, float(reward), terminated, truncated, info
        return obs, float(reward), truncated, info

    def render(self):
        pass

    def close(self):
        self._state = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _info(
        self,
        *,
        action_activated: bool = False,
        action_levels: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        info = {
            "scenario_index": self._scenario_index,
            "scenario_key": self._scenario_key,
            "elapsed_hours": self._elapsed_hours,
            "action_activated": action_activated,
            "action_mode": self.action_mode,
            "action_levels": action_levels or {},
        }
        if self._state is None or not self.collect_diagnostics:
            return info
        eng = _lazy_engine()
        info.update(
            {
                "total_shortage": self._state.total_shortage_units,
                "active_actions": [a.action.key for a in self._state.active_actions],
                "shortage_rate": self._state.total_shortage_units
                / max(self._state.total_net_requested_units, 1),
                "expiry_rate": sum(
                    center.stats["expired"] for center in self._state.centers
                )
                / max(
                    sum(center.stats["transfused"] for center in self._state.centers),
                    1,
                ),
                "budget_remaining": float(self._state.budget_remaining),
                "congestion_stress": float(eng.congestion_stress(self._state)),
            }
        )
        return info

    def _select_scenario_index(self) -> int:
        if len(self._scenario_templates) == 1:
            return 0
        return self._episodes_started % len(self._scenario_templates)
