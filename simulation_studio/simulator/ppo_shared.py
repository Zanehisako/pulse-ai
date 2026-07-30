"""
ppo_shared.py  –  Constants and pure functions shared between engine.py and
                  ppo_env.py.  Importing this module never triggers a circular
                  dependency because it only depends on calibration / core /
                  scenarios – never on engine or ppo_env.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Optional

import numpy as np
from calibration import (
    COMPONENT_DEMAND_WEIGHTS,
    QC_BASE_DEMAND_INTERARRIVAL_H,
    QC_BASE_DONOR_INTERARRIVAL_H,
    QC_DAILY_LABILE_PRODUCTS_EST,
)
from core import CENTER_CONFIGS
from scenarios import ACTION_CATALOG


def _slugify_control_key(value: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")

# ---------------------------------------------------------------------------
# Stable action ordering shared by both discrete PPO and continuous Dreamer
# ---------------------------------------------------------------------------
ACTION_KEYS: list[str] = sorted(ACTION_CATALOG.keys())
HOSPITAL_ROUTE_CONTROL_MAP: dict[str, str] = {
    f"route_{_slugify_control_key(config['name'])}": config["name"]
    for config in CENTER_CONFIGS
    if config["type"] == "hospital"
}
HOSPITAL_ROUTE_KEYS: list[str] = list(HOSPITAL_ROUTE_CONTROL_MAP.keys())
COMPONENT_ALLOCATION_CONTROL_MAP: dict[str, str] = {
    "component_rbc": "RBC",
    "component_platelets": "PLATELETS",
    "component_plasma": "PLASMA",
}
COMPONENT_ALLOCATION_KEYS: list[str] = list(COMPONENT_ALLOCATION_CONTROL_MAP.keys())
DREAMER_CONTROL_KEYS: list[str] = (
    ACTION_KEYS + HOSPITAL_ROUTE_KEYS + COMPONENT_ALLOCATION_KEYS
)
N_ACTIONS: int = len(ACTION_KEYS) + 1  # +1 for NOOP (index 0)
DREAMER_ACTION_DIM: int = len(DREAMER_CONTROL_KEYS)
OBS_DIM: int = 16
CONTINUOUS_ACTION_LOW: float = -1.0
CONTINUOUS_ACTION_HIGH: float = 1.0
CONTINUOUS_ACTION_THRESHOLD: float = 0.05


def clip_continuous_action(
    action_value,
    *,
    expected_dim: int = DREAMER_ACTION_DIM,
) -> np.ndarray:
    action = np.asarray(action_value, dtype=np.float32).reshape(-1)
    if action.size != expected_dim:
        raise ValueError(
            f"Expected continuous action with {expected_dim} values, got {action.size}."
        )
    return np.clip(action, CONTINUOUS_ACTION_LOW, CONTINUOUS_ACTION_HIGH)


def continuous_action_levels(
    action_value,
    *,
    activation_threshold: float = CONTINUOUS_ACTION_THRESHOLD,
) -> dict[str, float]:
    """
    Translate DreamerV3's continuous control vector into per-action intensities.

    The simulator accepts a symmetric ``Box(-1, 1)`` so the policy can use the
    standard Dreamer continuous-control interface. We map the full range onto
    usable action intensity so a zero-centered policy does not collapse into a
    permanent NOOP policy:
      - ``-1`` -> action is off
      - `` 0`` -> medium intensity
      - ``+1`` -> full intensity
    """
    raw = clip_continuous_action(action_value)
    levels = 0.5 * (raw + 1.0)
    if activation_threshold > 0:
        levels = np.where(levels >= activation_threshold, levels, 0.0)
    return {
        key: float(level)
        for key, level in zip(DREAMER_CONTROL_KEYS, levels.tolist(), strict=False)
    }

# ---------------------------------------------------------------------------
# Weather severity mapping used by the PPO observation builder
# ---------------------------------------------------------------------------
PPO_WEATHER_SEVERITY: dict[str, float] = {
    "clear": 0.0,
    "cloudy": 0.15,
    "snow": 0.45,
    "ice_storm": 0.70,
    "blizzard": 1.0,
}


# ---------------------------------------------------------------------------
# Observation vector helpers
# ---------------------------------------------------------------------------


def get_multi_obs(state):
    base = state_to_obs(state)

    # 🔹 Extract signals
    donations = sum(center.stats["donated"] for center in state.centers)
    shortage = state.total_shortage_units
    transfers = state.total_transfer_units

    # 🔹 Normalize (VERY IMPORTANT)
    donations /= 1000.0
    shortage /= 1000.0
    transfers /= 1000.0

    # 🔹 Global coordination signal
    global_signal = state.total_shortage_units / max(state.total_net_requested_units, 1)

    return {
        "supply": np.concatenate([base, [donations, global_signal]]),
        "hospital": np.concatenate([base, [shortage, global_signal]]),
        "logistics": np.concatenate([base, [transfers, global_signal]]),
    }


def clamp_obs(raw: dict[str, float]) -> np.ndarray:
    """Convert the raw observation dict to a fixed-length float32 array."""
    obs = np.array(
        [
            raw["recent_shortage_rate"],
            raw["cumulative_shortage_rate"],
            raw["critical_gap"],
            raw["rbc_gap"],
            raw["platelet_gap"],
            raw["plasma_gap"],
            min(raw["transport_stress"] / 3.0, 1.0),
            raw["weather_stress"],
            min(raw["donor_stress"] / 3.0, 1.0),
            min(raw["demand_stress"] / 3.0, 1.0),
            min(raw["recent_service_pressure"] / 5.0, 1.0),
            min(raw["active_cost_ratio"], 1.0),
            min(raw["forecast_signal"], 1.0),
            min(raw["priority_pressure"], 1.0),
            min(raw["congestion_stress"], 1.0),
            min(raw["budget_pressure"], 1.0),
        ],
        dtype=np.float32,
    )
    return np.clip(obs, 0.0, 1.0)


def build_observation(
    *,
    total_shortage_units: int,
    total_net_requested_units: int,
    total_transfused: int,
    coverage: dict[str, float],
    reserve_target_days: float,
    donor_inter_arrival_h: float,
    donor_show_factor: float,
    demand_rate_h: float,
    demand_surge_factor: float,
    effective_transport_penalty: float,
    weather_state: str,
    prev_shortage: float,
    prev_requested: float,
    prev_transfused: float,
    active_cost: float,
    forecast_pressure: float,
    priority_pressure: float,
    congestion_stress: float,
    budget_pressure: float,
) -> dict[str, float]:
    """
    Pure-function version of engine.build_ppo_observation().
    All inputs are plain scalars / dicts – no SimState required.
    """
    reserve_target = max(reserve_target_days, 1.0)

    recent_shortage = max(total_shortage_units - prev_shortage, 0.0)
    recent_requested = max(total_net_requested_units - prev_requested, 0.0)
    recent_transfused = max(total_transfused - prev_transfused, 0.0)

    rbc_gap = max(reserve_target - coverage["RBC"], 0.0) / reserve_target
    platelet_gap = max(reserve_target - coverage["PLATELETS"], 0.0) / reserve_target
    plasma_gap = max(reserve_target - coverage["PLASMA"], 0.0) / reserve_target

    donor_stress = max(
        donor_inter_arrival_h / max(QC_BASE_DONOR_INTERARRIVAL_H, 1e-6) - 1.0, 0.0
    ) + max(1.0 - donor_show_factor, 0.0)
    demand_stress = max(
        QC_BASE_DEMAND_INTERARRIVAL_H / max(demand_rate_h, 1e-6) - 1.0, 0.0
    ) + max(demand_surge_factor - 1.0, 0.0)

    return {
        "recent_shortage_rate": recent_shortage / max(recent_requested, 1.0),
        "cumulative_shortage_rate": total_shortage_units
        / max(total_net_requested_units, 1.0),
        "critical_gap": max(rbc_gap, platelet_gap * 1.15, plasma_gap * 0.65),
        "rbc_gap": rbc_gap,
        "platelet_gap": platelet_gap,
        "plasma_gap": plasma_gap,
        "transport_stress": max(effective_transport_penalty - 1.0, 0.0),
        "weather_stress": PPO_WEATHER_SEVERITY[weather_state],
        "donor_stress": donor_stress,
        "demand_stress": demand_stress,
        "recent_service_pressure": recent_requested / max(recent_transfused + 1.0, 1.0),
        "active_cost_ratio": active_cost / 500.0,
        "forecast_signal": np.clip((forecast_pressure - 0.75) / 1.5, 0.0, 1.0),
        "priority_pressure": priority_pressure,
        "congestion_stress": congestion_stress,
        "budget_pressure": budget_pressure,
    }


# Global State
def get_global_state(obs_dict):
    return np.concatenate(
        [
            obs_dict["supply"],
            obs_dict["hospital"],
            obs_dict["logistics"],
        ]
    )


def state_to_obs(state):
    """
    Convert full simulator state → flat vector
    (keep it simple but informative)
    """

    total_inventory = 0
    total_rbc = 0
    total_platelets = 0
    total_plasma = 0

    total_capacity = 0

    for c in state.centers:
        inv = c.inventory_by_component()

        total_rbc += inv.get("RBC", 0)
        total_platelets += inv.get("PLATELETS", 0)
        total_plasma += inv.get("PLASMA", 0)

        total_inventory += len(c.inventory)

        total_capacity += c.capacity["nurses"]

    n_centers = len(state.centers)

    avg_inventory = total_inventory / max(n_centers, 1)

    shortage = state.total_shortage_units
    transfers = state.total_transfer_units
    expired = sum(c.stats["expired"] for c in state.centers)
    donated = sum(c.stats["donated"] for c in state.centers)

    time_norm = state.env.now / max(state.params.sim_hours, 1)

    obs = np.array(
        [
            total_rbc,
            total_platelets,
            total_plasma,
            avg_inventory,
            shortage,
            transfers,
            expired,
            donated,
            total_capacity,
            len(state.active_actions),
            time_norm,
            n_centers,
        ],
        dtype=np.float32,
    )

    return obs
