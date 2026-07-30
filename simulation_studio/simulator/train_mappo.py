from __future__ import annotations

import argparse
import copy
import csv
import math
import numpy as np
import os
import tempfile
from pathlib import Path

CACHE_ROOT = Path(tempfile.gettempdir()) / "pios_mappo_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

from calibration import (
    QC_BASE_DEMAND_INTERARRIVAL_H,
    QC_BASE_DONOR_INTERARRIVAL_H,
    QC_DAILY_LABILE_PRODUCTS_EST,
)
from core import load_city_graph
from engine import (
    PPO_POLICY_WEIGHTS,
    apply_multi_actions,
    action_progress,
    build_ppo_observation,
    calculate_reward,
    effective_transport_penalty,
    policy_snapshot,
    run_scenario,
    step_simulation,
    total_donated_units,
    total_transfused_units,
)
from mappo import MAPPO
from ppo_shared import clamp_obs
from scenarios import ACTION_CATALOG, SCENARIOS, TRAINING_SCENARIO_KEYS

DEFAULT_BOUNDS = (46.90, 46.70, -71.10, -71.35)
ROLE_AGENT_NAMES = ["supply", "hospital", "logistics"]
MANAGER_AGENT_NAME = "manager"
ALL_AGENT_NAMES = [MANAGER_AGENT_NAME, *ROLE_AGENT_NAMES]
N_AGENTS = len(ALL_AGENT_NAMES)
BUNDLE_SEPARATOR = "+"
BASE_ROLE_ACTION_KEYS = {
    "supply": (
        "campaign",
        "extend_hours",
        "mobile_unit",
        "surge_staff",
        "lab_fast_track",
    ),
    "hospital": ("clinical_conservation",),
    "logistics": ("emergency_share", "national_mutual_aid", "rapid_courier"),
}
ROLE_ACTION_KEYS = {
    "supply": (
        *BASE_ROLE_ACTION_KEYS["supply"],
        "campaign+extend_hours",
        "mobile_unit+surge_staff",
        "lab_fast_track+surge_staff",
        "campaign+extend_hours+mobile_unit+surge_staff",
    ),
    "hospital": BASE_ROLE_ACTION_KEYS["hospital"],
    "logistics": (
        *BASE_ROLE_ACTION_KEYS["logistics"],
        "emergency_share+rapid_courier",
        "rapid_courier+national_mutual_aid",
        "emergency_share+rapid_courier+national_mutual_aid",
    ),
}
MAPPO_ACTION_KEYS = [
    *ROLE_ACTION_KEYS["supply"],
    *ROLE_ACTION_KEYS["hospital"],
    *ROLE_ACTION_KEYS["logistics"],
]
MANAGER_MODE_KEYS = [
    "steady",
    "supply_push",
    "hospital_protect",
    "logistics_rescue",
    "balanced_surge",
    "compound_stabilization",
    "full_emergency",
]
ACTION_OWNER = {
    action_key: role_name
    for role_name, action_keys in ROLE_ACTION_KEYS.items()
    for action_key in action_keys
}
MANAGER_MODE_SPECS = {
    0: {
        "name": "steady",
        "budget": 0,
        "priorities": {"supply": 0.45, "hospital": 0.45, "logistics": 0.45},
        "action_bonus": {},
    },
    1: {
        "name": "supply_push",
        "budget": 2,
        "priorities": {"supply": 1.20, "hospital": 0.55, "logistics": 0.70},
        "action_bonus": {
            "campaign": 0.55,
            "extend_hours": 0.35,
            "mobile_unit": 0.45,
            "surge_staff": 0.30,
            "lab_fast_track": 0.20,
        },
    },
    2: {
        "name": "hospital_protect",
        "budget": 2,
        "priorities": {"supply": 0.60, "hospital": 1.25, "logistics": 0.90},
        "action_bonus": {
            "clinical_conservation": 0.75,
            "lab_fast_track": 0.30,
            "emergency_share": 0.30,
        },
    },
    3: {
        "name": "logistics_rescue",
        "budget": 2,
        "priorities": {"supply": 0.55, "hospital": 0.65, "logistics": 1.35},
        "action_bonus": {
            "rapid_courier": 0.80,
            "emergency_share": 0.45,
            "national_mutual_aid": 0.35,
        },
    },
    4: {
        "name": "balanced_surge",
        "budget": 3,
        "priorities": {"supply": 1.0, "hospital": 1.0, "logistics": 1.0},
        "action_bonus": {
            "surge_staff": 0.35,
            "lab_fast_track": 0.35,
            "rapid_courier": 0.35,
            "clinical_conservation": 0.35,
        },
    },
    5: {
        "name": "compound_stabilization",
        "budget": 3,
        "priorities": {"supply": 1.05, "hospital": 1.25, "logistics": 1.25},
        "action_bonus": {
            "campaign": 0.25,
            "surge_staff": 0.40,
            "lab_fast_track": 0.40,
            "clinical_conservation": 0.75,
            "emergency_share": 0.40,
            "national_mutual_aid": 0.55,
            "rapid_courier": 0.65,
        },
    },
    6: {
        "name": "full_emergency",
        "budget": 3,
        "priorities": {"supply": 1.05, "hospital": 1.15, "logistics": 1.35},
        "action_bonus": {
            "national_mutual_aid": 0.75,
            "emergency_share": 0.55,
            "rapid_courier": 0.55,
            "clinical_conservation": 0.45,
            "surge_staff": 0.25,
        },
    },
}
MAPPO_ACTION_MASKS = {
    agent_name: [1.0]
    + [1.0 if key in ROLE_ACTION_KEYS[agent_name] else 0.0 for key in MAPPO_ACTION_KEYS]
    for agent_name in ROLE_AGENT_NAMES
}
N_ACTIONS = max(len(MANAGER_MODE_KEYS), len(MAPPO_ACTION_KEYS) + 1)
ACTION_INDEX = {key: idx + 1 for idx, key in enumerate(MAPPO_ACTION_KEYS)}
MAPPO_ACTION_MASKS[MANAGER_AGENT_NAME] = [
    1.0 if idx < len(MANAGER_MODE_KEYS) else 0.0 for idx in range(N_ACTIONS)
]

HEURISTIC_ACTION_PRIORS = {
    **PPO_POLICY_WEIGHTS,
    "extend_hours": {
        "bias": -0.55,
        "donor_stress": 1.10,
        "recent_service_pressure": 0.80,
        "critical_gap": 0.35,
        "active_cost_ratio": -0.08,
    },
    "mobile_unit": {
        "bias": -0.80,
        "donor_stress": 0.95,
        "demand_stress": 0.65,
        "critical_gap": 0.75,
        "weather_stress": -0.30,
        "active_cost_ratio": -0.05,
    },
    "surge_staff": {
        "bias": -0.65,
        "recent_service_pressure": 1.15,
        "critical_gap": 0.70,
        "recent_shortage_rate": 0.55,
        "active_cost_ratio": -0.08,
    },
    "rapid_courier": {
        "bias": -0.45,
        "transport_stress": 1.55,
        "weather_stress": 0.80,
        "recent_shortage_rate": 0.65,
        "critical_gap": 0.30,
    },
}

SCENARIO_WEIGHTS = {
    "baseline": 1.0,
    "donor_decrease": 3.0,
    "demand_surge": 1.0,
    "transport_disruption": 4.0,
    "combined_crisis": 3.0,
}
WARMUP_SCENARIOS = [
    "transport_disruption",
    "combined_crisis",
    "donor_decrease",
    "baseline",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MAPPO for the blood supply simulator and save training plots."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to train. Defaults to 200 if no limit is set.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Target number of environment steps across episodes.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Checkpoint path or prefix for the trained MAPPO model.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="PNG path for the training curves.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=None,
        help="CSV path for per-episode training metrics.",
    )
    parser.add_argument(
        "--plot-window",
        type=int,
        default=10,
        help="Moving-average window used in the saved plots.",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        default=None,
        help="Train on a fixed scenario instead of sampling scenarios each episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable training runs.",
    )
    parser.add_argument(
        "--rollout-episodes",
        type=int,
        default=4,
        help="How many episodes to accumulate before each MAPPO update.",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=8,
        help="Number of PPO epochs per MAPPO update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size used inside each MAPPO update.",
    )
    return parser.parse_args()


def load_training_context():
    try:
        return load_city_graph()
    except Exception as exc:
        print(
            f"[train_mappo] Falling back to G=None because city graph loading failed: {exc}"
        )
        return (None, *DEFAULT_BOUNDS)


def infer_obs_dim(sim_context) -> int:
    G, NORTH, SOUTH, EAST, WEST = sim_context
    dummy_state = run_scenario(
        prepare_mappo_scenario(SCENARIOS[TRAINING_SCENARIO_KEYS[0]]),
        G,
        NORTH,
        SOUTH,
        EAST,
        WEST,
        enable_logs=False,
        fast_mode=True,
    )
    snapshot = policy_snapshot(dummy_state)
    return len(build_mappo_obs(dummy_state, snapshot)["supply"])


def prepare_mappo_scenario(scenario):
    prepared = copy.deepcopy(scenario)
    prepared.strategy_key = "baseline"
    return prepared


def default_manager_context():
    return {
        "mode_index": 0,
        "budget": 0,
        "mode_age": 0,
        "priorities": {
            "supply": 0.45,
            "hospital": 0.45,
            "logistics": 0.45,
        },
        "action_bonus": {},
    }


def manager_context_features(manager_context) -> np.ndarray:
    context = default_manager_context() if manager_context is None else manager_context
    mode_one_hot = np.zeros(len(MANAGER_MODE_KEYS), dtype=np.float32)
    mode_index = int(context.get("mode_index", 0))
    if 0 <= mode_index < len(MANAGER_MODE_KEYS):
        mode_one_hot[mode_index] = 1.0
    return np.concatenate(
        [
            mode_one_hot,
            np.array(
                [
                    float(context.get("budget", 0)) / 3.0,
                    min(float(context.get("mode_age", 0)) / 8.0, 1.0),
                    float(context["priorities"]["supply"]),
                    float(context["priorities"]["hospital"]),
                    float(context["priorities"]["logistics"]),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def split_choice_key(action_key: str) -> tuple[str, ...]:
    return tuple(part for part in action_key.split(BUNDLE_SEPARATOR) if part)


def is_compound_choice(action_key: str) -> bool:
    return len(split_choice_key(action_key)) > 1


def available_choice_members(action_key: str, active_action_keys: set[str]) -> tuple[str, ...]:
    return tuple(
        member for member in split_choice_key(action_key) if member not in active_action_keys
    )


def choice_readiness(action_key: str, active_action_keys: set[str]) -> float:
    members = split_choice_key(action_key)
    if not members:
        return 0.0
    return float(len(available_choice_members(action_key, active_action_keys))) / float(
        len(members)
    )


def compute_system_signals(state, raw, prev_snapshot):
    active_action_keys = {activated.action.key for activated in state.active_actions}
    action_load = min(len(state.active_actions) / max(len(ACTION_CATALOG), 1), 1.0)
    response_progress = 0.0
    if state.active_actions:
        response_progress = min(
            float(
                np.mean(
                    [
                        action_progress(activated, state.env.now)
                        for activated in state.active_actions
                    ]
                )
            ),
            1.0,
        )

    shortage_recovery = min(
        max(prev_snapshot["shortage"] - state.total_shortage_units, 0.0)
        / max(prev_snapshot["requested"] + 1.0, 1.0),
        1.0,
    )
    role_urgency = {
        "supply": min(
            0.58 * raw["donor_stress"]
            + 0.32 * raw["critical_gap"]
            + 0.20 * raw["recent_shortage_rate"],
            1.0,
        ),
        "hospital": min(
            0.75 * raw["recent_shortage_rate"]
            + 0.45 * raw["demand_stress"]
            + 0.35 * raw["critical_gap"]
            + 0.20 * raw["recent_service_pressure"],
            1.0,
        ),
        "logistics": min(
            0.80 * raw["transport_stress"]
            + 0.45 * raw["weather_stress"]
            + 0.20 * raw["critical_gap"],
            1.0,
        ),
    }
    role_active_ratio = {
        name: min(
            sum(1 for key in BASE_ROLE_ACTION_KEYS[name] if key in active_action_keys)
            / max(len(BASE_ROLE_ACTION_KEYS[name]), 1),
            1.0,
        )
        for name in BASE_ROLE_ACTION_KEYS
    }
    global_severity = min(
        max(
            role_urgency["supply"],
            role_urgency["hospital"],
            role_urgency["logistics"],
            raw["cumulative_shortage_rate"] * 2.0,
        ),
        1.0,
    )
    stable_system = (
        global_severity < 0.18
        and raw["cumulative_shortage_rate"] < 0.05
        and raw["recent_service_pressure"] < 0.18
    )

    return {
        "active_action_keys": active_action_keys,
        "action_load": action_load,
        "response_progress": response_progress,
        "shortage_recovery": shortage_recovery,
        "role_urgency": role_urgency,
        "role_active_ratio": role_active_ratio,
        "global_severity": global_severity,
        "stable_system": stable_system,
    }


def compound_crisis_score(raw, signals) -> float:
    donor_axis = min(
        0.75 * raw.get("donor_stress", 0.0)
        + 0.25 * raw.get("critical_gap", 0.0)
        + 0.20 * raw.get("cumulative_shortage_rate", 0.0),
        1.25,
    )
    hospital_axis = min(
        0.70 * raw.get("recent_shortage_rate", 0.0)
        + 0.55 * raw.get("demand_stress", 0.0)
        + 0.35 * raw.get("critical_gap", 0.0),
        1.25,
    )
    logistics_axis = min(
        0.82 * raw.get("transport_stress", 0.0)
        + 0.55 * raw.get("weather_stress", 0.0)
        + 0.18 * raw.get("critical_gap", 0.0),
        1.25,
    )
    axes = np.array([donor_axis, hospital_axis, logistics_axis], dtype=np.float32)
    stressed_axes = float(np.count_nonzero(axes >= 0.42)) / 3.0
    score = (
        0.34 * float(axes.mean())
        + 0.26 * float(axes.min())
        + 0.18 * float(axes.max())
        + 0.12 * float(signals["global_severity"])
        + 0.10 * stressed_axes
    )
    if raw.get("cumulative_shortage_rate", 0.0) >= 0.10:
        score += 0.08
    return float(np.clip(score, 0.0, 1.0))


def build_manager_prior_bias(state, raw, signals, manager_context=None):
    bias = np.full(N_ACTIONS, -2.5, dtype=np.float32)
    prev_context = default_manager_context() if manager_context is None else manager_context
    prev_mode = int(prev_context.get("mode_index", 0))
    persistence = min(float(prev_context.get("mode_age", 0)) / 6.0, 1.0)
    prev_priorities = prev_context["priorities"]
    compound_score = compound_crisis_score(raw, signals)
    crisis_vector = np.array(
        [
            signals["role_urgency"]["supply"] + 0.55 * raw["donor_stress"],
            signals["role_urgency"]["hospital"]
            + 0.45 * raw["recent_shortage_rate"]
            + 0.25 * raw["demand_stress"],
            signals["role_urgency"]["logistics"]
            + 0.75 * raw["transport_stress"]
            + 0.35 * raw["weather_stress"],
        ],
        dtype=np.float32,
    )

    bias[0] = (
        0.50
        - 1.40 * signals["global_severity"]
        - 0.85 * compound_score
        - 0.35 * raw["cumulative_shortage_rate"]
        - 0.15 * signals["action_load"]
    )

    for action_idx in range(1, len(MANAGER_MODE_KEYS)):
        template = MANAGER_MODE_SPECS[action_idx]
        mode_name = template["name"]
        template_vec = np.array(
            [template["priorities"][name] for name in ROLE_AGENT_NAMES], dtype=np.float32
        )
        score = float(np.dot(crisis_vector, template_vec) / max(template_vec.sum(), 1e-6))
        score += 0.10 * template["budget"]
        if action_idx == prev_mode:
            score += 0.25 * persistence
        if action_idx == 1:
            score += 0.35 * raw["donor_stress"]
        elif action_idx == 2:
            score += 0.30 * raw["recent_shortage_rate"] + 0.20 * raw["demand_stress"]
        elif action_idx == 3:
            score += 0.55 * raw["transport_stress"] + 0.20 * raw["weather_stress"]
        elif action_idx == 4:
            score += 0.20 * signals["global_severity"]
        elif mode_name == "compound_stabilization":
            score += 0.95 * compound_score
            score += 0.35 * float(crisis_vector.min())
            score += 0.20 * float(np.count_nonzero(crisis_vector >= 0.50)) / 3.0
        elif mode_name == "full_emergency":
            score += 0.45 * signals["global_severity"] + 0.20 * raw["cumulative_shortage_rate"]
            score += 0.40 * compound_score
        score += 0.08 * (
            prev_priorities["supply"] * template["priorities"]["supply"]
            + prev_priorities["hospital"] * template["priorities"]["hospital"]
            + prev_priorities["logistics"] * template["priorities"]["logistics"]
        )
        if action_idx < N_ACTIONS:
            bias[action_idx] = score

    valid = np.asarray(MAPPO_ACTION_MASKS[MANAGER_AGENT_NAME], dtype=np.float32) > 0
    centered = bias.copy()
    centered[valid] -= float(centered[valid].mean())
    scale = max(float(centered[valid].std()), 1.0)
    centered[valid] = np.clip(centered[valid] / scale, -2.5, 2.5)
    return centered.astype(np.float32)


def manager_signal_from_action(action: int, state, raw, signals, manager_context=None):
    prev_context = default_manager_context() if manager_context is None else manager_context
    template = MANAGER_MODE_SPECS.get(action, MANAGER_MODE_SPECS[0])
    mode_name = template["name"]
    prev_mode = int(prev_context.get("mode_index", 0))
    same_mode = action == prev_mode
    persistence = min(float(prev_context.get("mode_age", 0)) / 6.0, 1.0)
    prev_priorities = np.array(
        [prev_context["priorities"][name] for name in ROLE_AGENT_NAMES], dtype=np.float32
    )
    compound_score = compound_crisis_score(raw, signals)
    crisis_vector = np.array(
        [
            signals["role_urgency"]["supply"]
            + 0.55 * raw.get("donor_stress", 0.0)
            + 0.15 * raw.get("critical_gap", 0.0),
            signals["role_urgency"]["hospital"]
            + 0.45 * raw.get("recent_shortage_rate", 0.0)
            + 0.35 * raw.get("demand_stress", 0.0),
            signals["role_urgency"]["logistics"]
            + 0.80 * raw.get("transport_stress", 0.0)
            + 0.45 * raw.get("weather_stress", 0.0),
        ],
        dtype=np.float32,
    )
    template_delta = np.array(
        [template["priorities"][name] - 1.0 for name in ROLE_AGENT_NAMES], dtype=np.float32
    )

    raw_priority = 0.95 * crisis_vector + 0.45 * template_delta + 0.30 * prev_priorities
    if same_mode:
        raw_priority += 0.20 * persistence * prev_priorities
    if signals["shortage_recovery"] < 0.03 and prev_context.get("budget", 0) > 0:
        raw_priority += 0.15 * prev_priorities
    if compound_score >= 0.35:
        raw_priority += 0.22 * np.clip(crisis_vector - 0.35, 0.0, None)
        raw_priority += np.array(
            [0.08 * compound_score, 0.12 * compound_score, 0.12 * compound_score],
            dtype=np.float32,
        )
    if state.params.transport_penalty >= 1.8:
        raw_priority[2] += 0.25
    if state.params.donor_show_factor < 0.75:
        raw_priority[0] += 0.20
    if raw["recent_shortage_rate"] >= 0.10:
        raw_priority[1] += 0.20

    shifted = raw_priority - float(raw_priority.max())
    weights = np.exp(np.clip(shifted, -8.0, 8.0))
    weights = weights / max(float(weights.sum()), 1e-6)
    priority_values = 0.40 + 1.95 * weights
    priorities = {
        role_name: float(priority_values[idx]) for idx, role_name in enumerate(ROLE_AGENT_NAMES)
    }

    budget_score = (
        2.4 * signals["global_severity"]
        + 1.1 * raw.get("cumulative_shortage_rate", 0.0)
        + 0.65 * raw.get("transport_stress", 0.0)
        + 1.10 * compound_score
        + 0.35 * template["budget"]
        + 0.30 * max(0.0, 0.04 - signals["shortage_recovery"]) * 10.0
    )
    if same_mode:
        budget_score += 0.20 * persistence
    budget = int(np.clip(np.round(budget_score), 0, len(ROLE_AGENT_NAMES)))
    if signals["stable_system"] and action == 0:
        budget = 0
    elif signals["global_severity"] >= 0.88:
        budget = len(ROLE_AGENT_NAMES)
    elif signals["global_severity"] >= 0.70:
        budget = max(budget, 2)
    if compound_score >= 0.55 or mode_name in {"compound_stabilization", "full_emergency"}:
        budget = len(ROLE_AGENT_NAMES)
    elif compound_score >= 0.38:
        budget = max(budget, 2)

    action_bonus = {}
    for action_key in MAPPO_ACTION_KEYS:
        owner = ACTION_OWNER[action_key]
        owner_priority = priorities[owner]
        owner_urgency = signals["role_urgency"][owner]
        bonus = 0.22 * owner_priority + 0.35 * owner_urgency
        bonus += 0.35 * template["action_bonus"].get(action_key, 0.0)

        if action_key == "campaign":
            bonus += 0.55 * raw.get("donor_stress", 0.0) + 0.15 * signals["shortage_recovery"]
        elif action_key == "extend_hours":
            bonus += 0.40 * raw.get("donor_stress", 0.0) + 0.20 * raw.get("recent_service_pressure", 0.0)
        elif action_key == "mobile_unit":
            bonus += 0.30 * raw.get("donor_stress", 0.0) + 0.30 * raw.get("demand_stress", 0.0) - 0.25 * raw.get("weather_stress", 0.0)
        elif action_key == "surge_staff":
            bonus += 0.35 * raw.get("recent_service_pressure", 0.0) + 0.25 * raw.get("critical_gap", 0.0)
        elif action_key == "lab_fast_track":
            bonus += 0.30 * raw.get("critical_gap", 0.0) + 0.25 * raw.get("recent_shortage_rate", 0.0)
        elif action_key == "clinical_conservation":
            bonus += 0.50 * raw.get("recent_shortage_rate", 0.0) + 0.35 * raw.get("demand_stress", 0.0)
        elif action_key == "rapid_courier":
            bonus += 0.75 * raw.get("transport_stress", 0.0) + 0.35 * raw.get("weather_stress", 0.0)
        elif action_key == "emergency_share":
            bonus += 0.45 * raw.get("transport_stress", 0.0) + 0.35 * raw.get("recent_shortage_rate", 0.0)
        elif action_key == "national_mutual_aid":
            bonus += 0.55 * signals["global_severity"] + 0.25 * raw.get("cumulative_shortage_rate", 0.0)

        if compound_score >= 0.40:
            if action_key == "campaign":
                bonus += 0.18 * compound_score
            elif action_key == "surge_staff":
                bonus += 0.28 * compound_score
            elif action_key == "lab_fast_track":
                bonus += 0.22 * compound_score
            elif action_key == "clinical_conservation":
                bonus += 0.40 * compound_score
            elif action_key == "emergency_share":
                bonus += 0.28 * compound_score
            elif action_key == "rapid_courier":
                bonus += 0.45 * compound_score
            elif action_key == "national_mutual_aid":
                bonus += 0.35 * compound_score

        if same_mode and mode_name in {"logistics_rescue", "compound_stabilization", "full_emergency"} and owner == "logistics":
            bonus += 0.15 * persistence
        action_bonus[action_key] = float(np.clip(bonus, -0.6, 2.2))

    return {
        "mode_index": int(action),
        "mode_age": int(prev_context.get("mode_age", 0) + 1 if same_mode else 1),
        "budget": budget,
        "priorities": priorities,
        "action_bonus": action_bonus,
    }


def _single_action_prior_score(action_key: str, state, raw, signals, manager_signal=None) -> float:
    weights = HEURISTIC_ACTION_PRIORS.get(action_key, {"bias": -0.6})
    score = float(weights.get("bias", 0.0))
    compound_score = compound_crisis_score(raw, signals)
    for feature_name, feature_weight in weights.items():
        if feature_name == "bias":
            continue
        score += raw.get(feature_name, 0.0) * feature_weight

    active_keys = signals["active_action_keys"]
    role_urgency = signals["role_urgency"]
    owner = ACTION_OWNER[action_key]

    if action_key in {"campaign", "extend_hours", "mobile_unit", "surge_staff", "lab_fast_track"}:
        score += 0.30 * role_urgency["supply"] - 0.15 * signals["action_load"]
    if action_key == "surge_staff" and active_keys.intersection(
        {"campaign", "extend_hours", "mobile_unit"}
    ):
        score += 0.35
    if action_key == "mobile_unit" and raw["weather_stress"] >= 0.45:
        score -= 0.35
    if action_key == "clinical_conservation":
        score += 0.35 * role_urgency["hospital"]
    if action_key in {"rapid_courier", "emergency_share", "national_mutual_aid"}:
        score += 0.40 * role_urgency["logistics"]
    if action_key == "national_mutual_aid":
        if signals["global_severity"] < 0.65 and len(active_keys) < 2:
            score -= 0.45
        if raw["weather_stress"] >= 0.65:
            score += 0.20
    if action_key == "rapid_courier" and raw["weather_stress"] >= 0.45:
        score += 0.25
    if action_key == "rapid_courier" and state.params.donor_show_factor < 0.80:
        score += 0.30
    if action_key == "rapid_courier" and state.params.transport_penalty >= 1.8:
        score += 0.45
    if compound_score >= 0.35:
        if action_key == "campaign":
            score += 0.18 * compound_score
        elif action_key == "surge_staff":
            score += 0.30 * compound_score
        elif action_key == "lab_fast_track":
            score += 0.24 * compound_score
        elif action_key == "clinical_conservation":
            score += 0.45 * compound_score
        elif action_key == "emergency_share":
            score += 0.28 * compound_score
        elif action_key == "rapid_courier":
            score += 0.50 * compound_score
        elif action_key == "national_mutual_aid":
            score += 0.40 * compound_score
    if manager_signal is not None:
        score += 0.30 * manager_signal["priorities"][owner]
        score += manager_signal["action_bonus"].get(action_key, 0.0)

    return float(score)


def action_prior_score(action_key: str, state, raw, signals, manager_signal=None) -> float:
    active_keys = signals["active_action_keys"]
    if not is_compound_choice(action_key):
        return _single_action_prior_score(
            action_key, state, raw, signals, manager_signal=manager_signal
        )

    members = available_choice_members(action_key, active_keys)
    if not members:
        return -4.0

    member_scores = [
        _single_action_prior_score(member, state, raw, signals, manager_signal=manager_signal)
        for member in members
    ]
    top_scores = sorted(member_scores, reverse=True)[: min(len(member_scores), 2)]
    owner = ACTION_OWNER[action_key]
    compound_score = compound_crisis_score(raw, signals)
    synergy = 0.18 * (len(members) - 1)
    if {"campaign", "extend_hours"}.issubset(members):
        synergy += 0.22 * raw.get("donor_stress", 0.0)
    if {"mobile_unit", "surge_staff"}.issubset(members):
        synergy += 0.18 * raw.get("critical_gap", 0.0)
    if {"lab_fast_track", "surge_staff"}.issubset(members):
        synergy += 0.18 * raw.get("recent_shortage_rate", 0.0)
    if {"emergency_share", "rapid_courier"}.issubset(members):
        synergy += 0.28 * raw.get("transport_stress", 0.0)
    if {"rapid_courier", "national_mutual_aid"}.issubset(members):
        synergy += 0.18 * signals["global_severity"]
    if {"emergency_share", "rapid_courier", "national_mutual_aid"}.issubset(members):
        synergy += 0.22 * compound_score
    if manager_signal is not None:
        synergy += 0.10 * manager_signal["priorities"][owner]
        synergy += 0.12 * float(
            np.mean([manager_signal["action_bonus"].get(member, 0.0) for member in members])
        )
    load_penalty = 0.28 * signals["action_load"] * len(members)
    complexity_penalty = 0.16 * (len(members) - 1)
    return float(
        np.mean(top_scores)
        + synergy
        + 0.12 * choice_readiness(action_key, active_keys)
        - load_penalty
        - complexity_penalty
    )


def _single_action_trigger(action_key: str, state, raw, signals, manager_signal=None) -> bool:
    active_keys = signals["active_action_keys"]
    severity = signals["global_severity"]
    compound_score = compound_crisis_score(raw, signals)

    if action_key in active_keys:
        return False
    if signals["stable_system"]:
        return False

    if action_key == "campaign":
        allow = (
            raw["donor_stress"] >= 0.16
            or raw["critical_gap"] >= 0.20
            or raw["recent_shortage_rate"] >= 0.08
        )
    elif action_key == "extend_hours":
        allow = (
            raw["donor_stress"] >= 0.12
            or raw["recent_service_pressure"] >= 0.35
            or (state.env.now < 96 and raw["critical_gap"] >= 0.20)
        )
    elif action_key == "mobile_unit":
        allow = (
            raw["donor_stress"] >= 0.22
            or raw["demand_stress"] >= 0.28
            or raw["cumulative_shortage_rate"] >= 0.10
        )
    elif action_key == "surge_staff":
        allow = (
            raw["recent_service_pressure"] >= 0.40
            or raw["critical_gap"] >= 0.28
            or active_keys.intersection({"campaign", "extend_hours", "mobile_unit"})
        )
    elif action_key == "lab_fast_track":
        allow = (
            raw["recent_shortage_rate"] >= 0.08
            or raw["critical_gap"] >= 0.18
            or raw["platelet_gap"] >= 0.20
        )
    elif action_key == "clinical_conservation":
        allow = (
            raw["demand_stress"] >= 0.20
            or raw["recent_shortage_rate"] >= 0.12
            or raw["platelet_gap"] >= 0.18
            or raw["cumulative_shortage_rate"] >= 0.09
        )
    elif action_key == "rapid_courier":
        allow = (
            raw["transport_stress"] >= 0.15
            or raw["weather_stress"] >= 0.45
            or state.params.transport_penalty > 1.25
            or state.params.donor_show_factor < 0.80
        )
    elif action_key == "emergency_share":
        allow = (
            raw["recent_shortage_rate"] >= 0.16
            or raw["critical_gap"] >= 0.22
            or raw["transport_stress"] >= 0.18
        )
    elif action_key == "national_mutual_aid":
        allow = (
            raw["cumulative_shortage_rate"] >= 0.14
            or severity >= 0.75
            or (raw["transport_stress"] >= 0.30 and raw["weather_stress"] >= 0.45)
        )
    else:
        allow = True

    if (
        len(active_keys) >= 3
        and action_key in {"campaign", "extend_hours", "mobile_unit", "surge_staff"}
        and severity < 0.50
        and raw["donor_stress"] < 0.35
    ):
        return False
    if (
        manager_signal is not None
        and manager_signal["mode_index"] == 0
        and severity < 0.70
    ):
        return False
    if compound_score >= 0.50:
        if action_key in {"campaign", "surge_staff", "lab_fast_track", "clinical_conservation", "rapid_courier", "emergency_share"}:
            allow = allow or severity >= 0.45
        if action_key == "national_mutual_aid":
            allow = allow or severity >= 0.60 or raw["cumulative_shortage_rate"] >= 0.08

    return bool(allow)


def action_trigger(action_key: str, state, raw, signals, manager_signal=None) -> bool:
    if not is_compound_choice(action_key):
        return _single_action_trigger(
            action_key, state, raw, signals, manager_signal=manager_signal
        )

    active_keys = signals["active_action_keys"]
    members = available_choice_members(action_key, active_keys)
    if not members or signals["stable_system"]:
        return False

    member_allows = [
        member
        for member in members
        if _single_action_trigger(member, state, raw, signals, manager_signal=manager_signal)
    ]
    if not member_allows:
        return False

    owner = ACTION_OWNER[action_key]
    severity = signals["global_severity"]
    compound_score = compound_crisis_score(raw, signals)
    urgency = signals["role_urgency"][owner]

    if manager_signal is not None and manager_signal["mode_index"] == 0 and severity < 0.78:
        return False
    if signals["action_load"] >= 0.45 and compound_score < 0.72:
        return False

    if len(members) >= 3:
        return bool(
            state.env.now >= 12.0
            and (
                compound_score >= 0.62
                or severity >= 0.76
                or (urgency >= 0.82 and len(member_allows) == len(members))
            )
            and len(member_allows) >= max(2, len(members) - 1)
        )

    if len(member_allows) == len(members):
        return bool(compound_score >= 0.30 or severity >= 0.45 or urgency >= 0.55)

    return bool(
        state.env.now >= 6.0 and (compound_score >= 0.58 or severity >= 0.70 or urgency >= 0.78)
    )


def build_dynamic_action_masks(state, raw, signals, manager_signal=None):
    masks = {name: np.zeros(N_ACTIONS, dtype=np.float32) for name in ALL_AGENT_NAMES}
    masks[MANAGER_AGENT_NAME] = np.asarray(
        MAPPO_ACTION_MASKS[MANAGER_AGENT_NAME], dtype=np.float32
    )

    allowed_roles = set(ROLE_AGENT_NAMES)
    compound_score = compound_crisis_score(raw, signals)
    if manager_signal is not None:
        budget = int(manager_signal["budget"])
        if compound_score >= 0.50:
            allowed_roles = set(ROLE_AGENT_NAMES)
        elif budget <= 0:
            allowed_roles = set()
        else:
            ranked_roles = sorted(
                ROLE_AGENT_NAMES,
                key=lambda name: (
                    manager_signal["priorities"][name] * max(signals["role_urgency"][name], 0.05)
                ),
                reverse=True,
            )
            allowed_roles = set(ranked_roles[:budget])

    for agent_name in ROLE_AGENT_NAMES:
        mask = masks[agent_name]
        mask[0] = 1.0
        if agent_name not in allowed_roles:
            continue
        for action_key in ROLE_ACTION_KEYS[agent_name]:
            if action_trigger(action_key, state, raw, signals, manager_signal=manager_signal):
                mask[ACTION_INDEX[action_key]] = 1.0

        if mask.sum() <= 1.0 and not signals["stable_system"]:
            fallback_key = None
            fallback_score = -float("inf")
            for action_key in ROLE_ACTION_KEYS[agent_name]:
                if action_key in signals["active_action_keys"]:
                    continue
                score = action_prior_score(
                    action_key,
                    state,
                    raw,
                    signals,
                    manager_signal=manager_signal,
                )
                if score > fallback_score:
                    fallback_key = action_key
                    fallback_score = score
            if fallback_key is not None:
                mask[ACTION_INDEX[fallback_key]] = 1.0

    return masks


def build_action_prior_biases(state, raw, signals, action_masks, manager_signal=None):
    biases = {}
    biases[MANAGER_AGENT_NAME] = build_manager_prior_bias(
        state,
        raw,
        signals,
        manager_context=manager_signal,
    )
    for agent_name in ROLE_AGENT_NAMES:
        urgency = signals["role_urgency"][agent_name]
        bias = np.zeros(N_ACTIONS, dtype=np.float32)
        bias[0] = 0.40 - 1.35 * urgency - 0.20 * signals["action_load"]

        for action_key in ROLE_ACTION_KEYS[agent_name]:
            idx = ACTION_INDEX[action_key]
            bias[idx] = action_prior_score(
                action_key,
                state,
                raw,
                signals,
                manager_signal=manager_signal,
            )

        valid = action_masks[agent_name] > 0
        valid_scores = bias[valid]
        if valid_scores.size > 1:
            centered = bias.copy()
            centered[valid] -= float(valid_scores.mean())
            scale = max(float(valid_scores.std()), 1.0)
            centered[valid] = np.clip(centered[valid] / scale, -2.5, 2.5)
            bias = centered
        else:
            bias[:] = 0.0

        biases[agent_name] = bias.astype(np.float32)

    return biases


def scenario_context_features(state) -> np.ndarray:
    return np.array(
        [
            min(
                max(state.params.donor_inter_arrival_h / max(QC_BASE_DONOR_INTERARRIVAL_H, 1e-6) - 1.0, 0.0)
                / 2.0,
                1.0,
            ),
            min(max(1.0 - state.params.donor_show_factor, 0.0), 1.0),
            min(
                max(QC_BASE_DEMAND_INTERARRIVAL_H / max(state.params.demand_rate_h, 1e-6) - 1.0, 0.0)
                / 2.0,
                1.0,
            ),
            min(max(state.params.demand_surge_factor - 1.0, 0.0) / 1.5, 1.0),
            min(max(state.params.transport_penalty - 1.0, 0.0) / 2.5, 1.0),
            min(max(0.45 - state.params.regional_replenishment_rate, 0.0) / 0.45, 1.0),
            min(state.params.reserve_target_days / 6.0, 1.0),
            min(state.params.sim_hours / 720.0, 1.0),
        ],
        dtype=np.float32,
    )


def action_tempo_features(state) -> np.ndarray:
    role_progress: dict[str, list[float]] = {name: [] for name in ROLE_AGENT_NAMES}
    ages: list[float] = []
    pending = 0
    mature = 0
    for activated in state.active_actions:
        progress = float(action_progress(activated, state.env.now))
        owner = ACTION_OWNER.get(activated.action.key)
        if owner in role_progress:
            role_progress[owner].append(progress)
        ages.append(max(state.env.now - activated.activated_at_h, 0.0))
        pending += int(progress < 0.35)
        mature += int(progress >= 0.80)

    active_count = len(state.active_actions)
    role_means = [
        float(np.mean(role_progress[name])) if role_progress[name] else 0.0
        for name in ROLE_AGENT_NAMES
    ]
    return np.array(
        [
            *role_means,
            min(pending / max(active_count, 1), 1.0),
            min(mature / max(active_count, 1), 1.0),
            min((max(ages) if ages else 0.0) / 120.0, 1.0),
        ],
        dtype=np.float32,
    )


def guardrail_features(state) -> np.ndarray:
    total_inventory = float(sum(len(center.inventory) for center in state.centers))
    total_transfused = float(total_transfused_units(state))
    total_expired = float(sum(center.stats["expired"] for center in state.centers))
    target_inventory_units = max(
        QC_DAILY_LABILE_PRODUCTS_EST * max(state.params.reserve_target_days, 1.0),
        1.0,
    )
    inventory_bloat = min(max(total_inventory / target_inventory_units - 1.0, 0.0) / 1.5, 1.0)
    conservation_ratio = min(
        state.total_conserved_units / max(state.total_base_requested_units, 1.0),
        1.0,
    )
    expiry_ratio = min(total_expired / max(total_transfused + 1.0, 1.0), 1.0)
    external_dependency = min(state.total_external_units / max(total_transfused + 1.0, 1.0), 1.0)
    return np.array(
        [
            inventory_bloat,
            conservation_ratio,
            expiry_ratio,
            external_dependency,
        ],
        dtype=np.float32,
    )


def build_mappo_obs(state, prev_snapshot, manager_signal=None):
    raw = build_ppo_observation(state, prev_snapshot)
    base = clamp_obs(raw)
    signals = compute_system_signals(state, raw, prev_snapshot)

    recent_donations = max(total_donated_units(state) - prev_snapshot["donated"], 0.0)
    global_shortage_ratio = min(
        state.total_shortage_units / max(state.total_net_requested_units, 1.0),
        1.0,
    )
    time_norm = min(state.env.now / max(state.params.sim_hours, 1.0), 1.0)
    donor_throughput = min(recent_donations / 25.0, 1.0)
    hospital_pressure = min(raw["recent_shortage_rate"] + raw["critical_gap"], 1.0)
    transfer_load = min(
        state.total_transfer_units / max(total_transfused_units(state) + 1.0, 1.0),
        1.0,
    )
    logistics_pressure = min(
        max(effective_transport_penalty(state) - 1.0, 0.0) / 3.0 + transfer_load,
        1.0,
    )
    manager_context = default_manager_context() if manager_signal is None else manager_signal
    manager_features = manager_context_features(manager_context)
    shared_context = np.concatenate(
        [
            scenario_context_features(state),
            action_tempo_features(state),
            guardrail_features(state),
        ]
    ).astype(np.float32)

    return {
        MANAGER_AGENT_NAME: np.concatenate(
            [
                base,
                shared_context,
                [
                    time_norm,
                    global_shortage_ratio,
                    signals["action_load"],
                    signals["response_progress"],
                    signals["shortage_recovery"],
                    donor_throughput,
                    logistics_pressure,
                ],
                manager_features,
            ]
        ).astype(np.float32),
        "supply": np.concatenate(
            [
                base,
                shared_context,
                [
                    donor_throughput,
                    time_norm,
                    signals["action_load"],
                    signals["response_progress"],
                    signals["shortage_recovery"],
                    signals["role_urgency"]["supply"],
                    signals["role_active_ratio"]["supply"],
                ],
                manager_features,
            ]
        ).astype(np.float32),
        "hospital": np.concatenate(
            [
                base,
                shared_context,
                [
                    hospital_pressure,
                    global_shortage_ratio,
                    signals["action_load"],
                    signals["response_progress"],
                    signals["shortage_recovery"],
                    signals["role_urgency"]["hospital"],
                    signals["role_active_ratio"]["hospital"],
                ],
                manager_features,
            ]
        ).astype(np.float32),
        "logistics": np.concatenate(
            [
                base,
                shared_context,
                [
                    logistics_pressure,
                    time_norm,
                    signals["action_load"],
                    signals["response_progress"],
                    signals["shortage_recovery"],
                    signals["role_urgency"]["logistics"],
                    signals["role_active_ratio"]["logistics"],
                ],
                manager_features,
            ]
        ).astype(np.float32),
    }


def build_mappo_step_inputs(state, prev_snapshot, manager_context=None):
    raw = build_ppo_observation(state, prev_snapshot)
    signals = compute_system_signals(state, raw, prev_snapshot)
    manager_obs = build_mappo_obs(
        state,
        prev_snapshot,
        manager_signal=manager_context,
    )[MANAGER_AGENT_NAME]
    manager_mask = np.asarray(MAPPO_ACTION_MASKS[MANAGER_AGENT_NAME], dtype=np.float32)
    manager_bias = build_manager_prior_bias(
        state,
        raw,
        signals,
        manager_context=manager_context,
    )
    return raw, signals, manager_obs, manager_mask, manager_bias


def capture_reward_metrics(state, prev_snapshot=None) -> dict[str, float]:
    total_transfused = float(total_transfused_units(state))
    total_requested = float(max(state.total_base_requested_units, 1.0))
    total_expired = float(sum(center.stats["expired"] for center in state.centers))
    total_donations = float(total_donated_units(state))
    active_cost = float(sum(action.action.operational_cost for action in state.active_actions))
    total_inventory = float(sum(len(center.inventory) for center in state.centers))
    inventory_target = max(
        QC_DAILY_LABILE_PRODUCTS_EST * max(state.params.reserve_target_days, 1.0),
        1.0,
    )
    raw = None
    signals = None
    if prev_snapshot is not None:
        raw = build_ppo_observation(state, prev_snapshot)
        signals = compute_system_signals(state, raw, prev_snapshot)

    return {
        "shortage_rate": float(state.total_shortage_units / total_requested),
        "service_rate": float(total_transfused / total_requested),
        "transport_stress": float(
            min(max(effective_transport_penalty(state) - 1.0, 0.0) / 2.5, 1.0)
        ),
        "expiry_rate": float(total_expired / max(total_transfused + 1.0, 1.0)),
        "transfer_load": float(state.total_transfer_units / max(total_transfused + 1.0, 1.0)),
        "donor_flow": float(min(total_donations / max(total_requested, 1.0), 1.5)),
        "action_cost_ratio": float(min(active_cost / 320.0, 1.5)),
        "conservation_rate": float(
            state.total_conserved_units / max(state.total_base_requested_units, 1.0)
        ),
        "inventory_bloat": float(max(total_inventory / inventory_target - 1.0, 0.0)),
        "action_pressure": float(len(state.active_actions) / max(len(ACTION_CATALOG), 1)),
        "compound_score": float(
            compound_crisis_score(raw, signals) if raw is not None and signals is not None else 0.0
        ),
    }


def reward_weight_profile(state, metrics: dict[str, float]) -> dict[str, float]:
    phase = min(state.env.now / max(state.params.sim_hours, 1.0), 1.0)
    shortage_need = min(
        metrics["shortage_rate"] * 2.2 + max(1.0 - metrics["service_rate"], 0.0),
        1.6,
    )
    transport_need = min(
        metrics["transport_stress"] + max(state.params.transport_penalty - 1.0, 0.0) / 1.5,
        1.6,
    )
    donor_need = min(
        max(1.0 - metrics["donor_flow"], 0.0)
        + max(0.0, state.params.transport_penalty - 1.0) / 2.0,
        1.5,
    )
    conservation_need = min(metrics["conservation_rate"] * 2.2 + max(phase - 0.15, 0.0), 1.6)
    inventory_need = min(metrics["inventory_bloat"] * (1.2 + phase), 1.6)
    compound_need = min(
        0.40 * shortage_need
        + 0.30 * transport_need
        + 0.22 * donor_need
        + 0.20 * max(1.0 - metrics["service_rate"], 0.0),
        0.20 * conservation_need
        + 0.18 * inventory_need
        + 0.40 * metrics["compound_score"],
        2.2,
    )
    compound_axes = float(
        sum(
            [
                shortage_need >= 0.55,
                transport_need >= 0.45,
                donor_need >= 0.40,
                metrics["compound_score"] >= 0.45,
            ]
        )
    ) / 4.0
    compound_score = min(
        0.48 * (compound_need / 2.2) + 0.22 * compound_axes + 0.30 * metrics["compound_score"],
        1.0,
    )
    stability = max(
        1.0
        - max(
            shortage_need / 1.6,
            transport_need / 1.6,
            metrics["shortage_rate"] * 1.8,
            metrics["compound_score"],
        ),
        0.0,
    )
    return {
        "service": 2.0 + 2.4 * shortage_need + 1.4 * compound_score,
        "shortage": 4.4 + 3.8 * shortage_need + 1.8 * transport_need + 2.4 * compound_score,
        "transport": 0.6 + 2.8 * transport_need + 1.6 * compound_score,
        "expiry": 0.45 + 1.45 * phase + 0.75 * inventory_need,
        "transfer": 0.30 + 1.10 * transport_need,
        "cost": max(0.04, 0.10 + 0.95 * stability * (0.35 + phase) - 0.22 * compound_score),
        "donor": 0.25 + 1.20 * donor_need + 1.05 * compound_score,
        "conservation": 0.55 + 2.15 * conservation_need + 1.10 * compound_score,
        "inventory": 0.30 + 1.85 * inventory_need + 0.65 * phase,
        "action": 0.10 + 0.80 * stability,
        "compound": 0.35 + 2.05 * compound_score,
    }


def reward_score(state, metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
    weights = reward_weight_profile(state, metrics) if weights is None else weights
    return (
        weights["service"] * metrics["service_rate"]
        - weights["shortage"] * metrics["shortage_rate"]
        - weights["transport"] * metrics["transport_stress"]
        - weights["expiry"] * metrics["expiry_rate"]
        - weights["transfer"] * metrics["transfer_load"]
        - weights["cost"] * metrics["action_cost_ratio"]
        - weights["conservation"] * metrics["conservation_rate"]
        - weights["inventory"] * metrics["inventory_bloat"]
        - weights["action"] * metrics["action_pressure"]
        + weights["donor"] * metrics["donor_flow"]
    )


def compute_dynamic_step_reward(state, prev_metrics: dict[str, float], current_metrics: dict[str, float]) -> float:
    current_weights = reward_weight_profile(state, current_metrics)
    previous_weights = reward_weight_profile(state, prev_metrics)
    current_score = reward_score(state, current_metrics, current_weights)
    previous_score = reward_score(state, prev_metrics, previous_weights)

    delta_shortage = prev_metrics["shortage_rate"] - current_metrics["shortage_rate"]
    delta_service = current_metrics["service_rate"] - prev_metrics["service_rate"]
    delta_transport = prev_metrics["transport_stress"] - current_metrics["transport_stress"]
    delta_expiry = prev_metrics["expiry_rate"] - current_metrics["expiry_rate"]
    delta_transfer = prev_metrics["transfer_load"] - current_metrics["transfer_load"]
    delta_donor = current_metrics["donor_flow"] - prev_metrics["donor_flow"]
    delta_conservation = prev_metrics["conservation_rate"] - current_metrics["conservation_rate"]
    delta_inventory = prev_metrics["inventory_bloat"] - current_metrics["inventory_bloat"]
    delta_action_pressure = prev_metrics["action_pressure"] - current_metrics["action_pressure"]
    delta_compound = prev_metrics["compound_score"] - current_metrics["compound_score"]

    reward = current_score - previous_score
    reward += 0.55 * current_weights["shortage"] * delta_shortage
    reward += 0.45 * current_weights["service"] * delta_service
    reward += 0.35 * current_weights["transport"] * delta_transport
    reward += 0.18 * current_weights["expiry"] * delta_expiry
    reward += 0.15 * current_weights["transfer"] * delta_transfer
    reward += 0.20 * current_weights["donor"] * delta_donor
    reward += 0.26 * current_weights["conservation"] * delta_conservation
    reward += 0.18 * current_weights["inventory"] * delta_inventory
    reward += 0.08 * current_weights["action"] * delta_action_pressure
    compound_progress = (
        0.85 * max(delta_shortage, 0.0)
        + 0.65 * max(delta_service, 0.0)
        + 0.80 * max(delta_transport, 0.0)
        + 0.55 * max(delta_donor, 0.0)
        + 0.45 * max(delta_conservation, 0.0)
        + 0.30 * max(delta_inventory, 0.0)
        + 0.40 * max(delta_compound, 0.0)
    )
    compound_regress = (
        0.90 * max(-delta_shortage, 0.0)
        + 0.55 * max(-delta_service, 0.0)
        + 0.70 * max(-delta_transport, 0.0)
        + 0.40 * max(-delta_conservation, 0.0)
        + 0.35 * max(-delta_compound, 0.0)
    )
    reward += 0.18 * current_weights["compound"] * compound_progress
    reward -= 0.10 * current_weights["compound"] * compound_regress

    if state.env.now >= state.params.sim_hours:
        reward += 1.00 * current_weights["service"] * current_metrics["service_rate"]
        reward -= 1.35 * current_weights["shortage"] * current_metrics["shortage_rate"]
        reward -= 0.70 * current_weights["conservation"] * current_metrics["conservation_rate"]
        reward -= 0.45 * current_weights["inventory"] * current_metrics["inventory_bloat"]

    return float(np.clip(reward, -3.0, 3.0))


def run_episode(mappo: MAPPO, scenario, sim_context, prior_scale: float = 1.0):
    G, NORTH, SOUTH, EAST, WEST = sim_context
    prepared_scenario = prepare_mappo_scenario(scenario)

    state = run_scenario(
        prepared_scenario,
        G,
        NORTH,
        SOUTH,
        EAST,
        WEST,
        seed=np.random.randint(0, 10000),
        enable_logs=False,
        fast_mode=True,
    )

    traj = {
        "global_states": [],
        "rewards": [],
        "dones": [],
        "normalized_obs": True,
        "log_probs": {name: [] for name in mappo.agent_names},
        "actions": {name: [] for name in mappo.agent_names},
        "obs": {name: [] for name in mappo.agent_names},
        "action_masks": {name: [] for name in mappo.agent_names},
        "action_biases": {name: [] for name in mappo.agent_names},
    }
    prev_snapshot = policy_snapshot(state)
    manager_context = default_manager_context()
    prev_metrics = capture_reward_metrics(state, prev_snapshot=prev_snapshot)

    while state.env.now < state.params.sim_hours:
        raw, signals, manager_obs, manager_mask, manager_bias = build_mappo_step_inputs(
            state,
            prev_snapshot,
            manager_context=manager_context,
        )
        scaled_manager_bias = np.asarray(manager_bias, dtype=np.float32) * float(prior_scale)
        manager_action, manager_log_prob, processed_manager_obs = mappo.act_single(
            MANAGER_AGENT_NAME,
            manager_obs,
            action_mask=manager_mask,
            action_bias=scaled_manager_bias,
        )
        manager_signal = manager_signal_from_action(
            manager_action,
            state,
            raw,
            signals,
            manager_context=manager_context,
        )
        obs_dict = build_mappo_obs(state, prev_snapshot, manager_signal=manager_signal)
        action_masks = build_dynamic_action_masks(
            state, raw, signals, manager_signal=manager_signal
        )
        action_biases = build_action_prior_biases(
            state, raw, signals, action_masks, manager_signal=manager_signal
        )
        step_snapshot = policy_snapshot(state)
        scaled_biases = {
            name: np.asarray(action_biases[name], dtype=np.float32) * float(prior_scale)
            for name in ROLE_AGENT_NAMES
        }
        actions, log_probs, processed_obs = mappo.act(
            {name: obs_dict[name] for name in ROLE_AGENT_NAMES},
            action_masks=action_masks,
            action_biases=scaled_biases,
        )
        processed_obs[MANAGER_AGENT_NAME] = processed_manager_obs
        actions[MANAGER_AGENT_NAME] = manager_action
        log_probs[MANAGER_AGENT_NAME] = manager_log_prob

        apply_multi_actions(
            state,
            {name: actions[name] for name in ROLE_AGENT_NAMES},
            action_keys=MAPPO_ACTION_KEYS,
            allowed_action_keys=ROLE_ACTION_KEYS,
        )
        step_simulation(state)

        current_metrics = capture_reward_metrics(state, prev_snapshot=step_snapshot)
        done = state.env.now >= state.params.sim_hours
        reward = compute_dynamic_step_reward(state, prev_metrics, current_metrics)
        prev_metrics = current_metrics
        manager_context = manager_signal
        prev_snapshot = step_snapshot

        traj["global_states"].append(mappo.build_global_state(processed_obs))
        traj["rewards"].append(reward)
        traj["dones"].append(float(done))

        for name in mappo.agent_names:
            traj["actions"][name].append(actions[name])
            traj["obs"][name].append(processed_obs[name])
            traj["log_probs"][name].append(log_probs[name])
            traj["action_masks"][name].append(np.asarray(action_masks[name], dtype=np.float32))
            stored_bias = (
                scaled_manager_bias if name == MANAGER_AGENT_NAME else scaled_biases[name]
            )
            traj["action_biases"][name].append(np.asarray(stored_bias, dtype=np.float32))

    final_reward = calculate_reward(state)
    traj["episode_metrics"] = {
        **prev_metrics,
        "manager_budget": float(manager_context["budget"]),
        "active_actions": float(len(state.active_actions)),
        "reward_total": float(final_reward.total),
    }
    return traj


def moving_average(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr

    smoothed = np.empty_like(arr, dtype=np.float32)
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        smoothed[idx] = float(arr[start : idx + 1].mean())
    return smoothed


def recent_mean_delta(values: list[float], window: int) -> float:
    if len(values) < window * 2:
        return 0.0
    recent = np.asarray(values[-window:], dtype=np.float32)
    previous = np.asarray(values[-2 * window : -window], dtype=np.float32)
    return float(recent.mean() - previous.mean())


def compute_plateau_level(history: list[dict], window: int = 16) -> float:
    if len(history) < window * 2:
        return 0.0

    avg_rewards = [float(row["avg_reward"]) for row in history]
    improvement = recent_mean_delta(avg_rewards, window)
    recent = np.asarray(avg_rewards[-window:], dtype=np.float32)
    plateau = 0.0
    if improvement < 0.008:
        plateau += 0.40
    if improvement < 0.003:
        plateau += 0.35
    if recent.std() > 0.03:
        plateau += 0.20
    return float(min(max(plateau, 0.0), 1.0))


def scenario_focus_pressure(history: list[dict], scenario_name: str, window: int = 10) -> float:
    scenario_rewards = [
        float(row["reward"]) for row in history if row["scenario"] == scenario_name
    ]
    if len(scenario_rewards) < max(4, window // 2):
        return 0.0

    recent = np.asarray(scenario_rewards[-window:], dtype=np.float32)
    previous = (
        np.asarray(scenario_rewards[-2 * window : -window], dtype=np.float32)
        if len(scenario_rewards) >= window * 2
        else None
    )
    recent_mean = float(recent.mean())
    improvement = (
        recent_mean - float(previous.mean()) if previous is not None and len(previous) else 0.0
    )

    pressure = 0.0
    if recent_mean < 0.5:
        pressure += min(0.9, (0.5 - recent_mean) / 2.5)
    if improvement < 0.15:
        pressure += min(0.7, 0.20 + (0.15 - improvement) / 2.5)
    if recent.std() > 1.25:
        pressure += 0.10
    return float(min(max(pressure, 0.0), 1.5))


def merge_trajectories(trajs: list[dict], agent_names: list[str]) -> dict:
    merged = {
        "global_states": [],
        "rewards": [],
        "dones": [],
        "normalized_obs": True,
        "log_probs": {name: [] for name in agent_names},
        "actions": {name: [] for name in agent_names},
        "obs": {name: [] for name in agent_names},
        "action_masks": {name: [] for name in agent_names},
        "action_biases": {name: [] for name in agent_names},
    }
    for traj in trajs:
        merged["global_states"].extend(traj["global_states"])
        merged["rewards"].extend(traj["rewards"])
        merged["dones"].extend(traj["dones"])
        for name in agent_names:
            merged["log_probs"][name].extend(traj["log_probs"][name])
            merged["actions"][name].extend(traj["actions"][name])
            merged["obs"][name].extend(traj["obs"][name])
            merged["action_masks"][name].extend(traj["action_masks"][name])
            merged["action_biases"][name].extend(traj["action_biases"][name])
    return merged


def resolve_artifact_paths(args) -> tuple[Path, Path, Path]:
    if args.save:
        base = Path(args.save)
        stem = base if base.suffix == "" else base.with_suffix("")
    else:
        stem = Path("mappo_training")

    plot_path = (
        Path(args.plot) if args.plot else stem.with_name(f"{stem.name}_training.png")
    )
    metrics_path = (
        Path(args.metrics_csv)
        if args.metrics_csv
        else stem.with_name(f"{stem.name}_history.csv")
    )
    scenario_plot_path = stem.with_name(f"{stem.name}_scenario_bands.png")
    return plot_path, metrics_path, scenario_plot_path


def save_metrics_csv(history: list[dict], output_path: Path) -> None:
    if not history:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_training_plot(history: list[dict], output_path: Path, window: int = 10) -> None:
    if not history:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["total_steps"] for row in history]
    reward = [row["reward"] for row in history]
    avg_reward = [row["avg_reward"] for row in history]
    episode_steps = [row["episode_steps"] for row in history]
    critic_loss = [row["critic_loss"] for row in history]
    policy_loss = [row["policy_loss"] for row in history]
    entropy = [row["entropy"] for row in history]
    adv_std = [row["adv_std"] for row in history]

    reward_ma = moving_average(reward, window)
    avg_reward_ma = moving_average(avg_reward, window)
    critic_ma = moving_average(critic_loss, window)
    entropy_ma = moving_average(entropy, window)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_reward, ax_avg, ax_critic, ax_policy = axes.flatten()

    ax_reward.plot(steps, reward, alpha=0.35, label="Episode reward", color="#1f77b4")
    ax_reward.plot(
        steps,
        reward_ma,
        linewidth=2.2,
        label=f"Reward MA ({window})",
        color="#0d3b66",
    )
    ax_reward.set_title("Episode Reward")
    ax_reward.set_xlabel("Cumulative timesteps")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(alpha=0.25)
    ax_reward.legend()

    ax_avg.plot(
        steps,
        avg_reward,
        alpha=0.35,
        label="Average step reward",
        color="#2a9d8f",
    )
    ax_avg.plot(
        steps,
        avg_reward_ma,
        linewidth=2.2,
        label=f"Avg reward MA ({window})",
        color="#1d7874",
    )
    ax_avg.set_title("Reward Quality")
    ax_avg.set_xlabel("Cumulative timesteps")
    ax_avg.set_ylabel("Average reward")
    ax_avg.grid(alpha=0.25)
    episode_ax = ax_avg.twinx()
    episode_ax.plot(
        steps,
        episode_steps,
        linestyle="--",
        alpha=0.5,
        label="Episode length",
        color="#f4a261",
    )
    episode_ax.set_ylabel("Episode steps")
    lines, labels = ax_avg.get_legend_handles_labels()
    lines2, labels2 = episode_ax.get_legend_handles_labels()
    ax_avg.legend(lines + lines2, labels + labels2, loc="best")

    ax_critic.plot(
        steps,
        critic_loss,
        alpha=0.3,
        label="Critic loss",
        color="#d62828",
    )
    ax_critic.plot(
        steps,
        critic_ma,
        linewidth=2.2,
        label=f"Critic MA ({window})",
        color="#9d0208",
    )
    ax_critic.plot(
        steps,
        adv_std,
        linewidth=1.3,
        alpha=0.8,
        label="Advantage std",
        color="#6a4c93",
    )
    ax_critic.set_title("Critic Stability")
    ax_critic.set_xlabel("Cumulative timesteps")
    ax_critic.set_ylabel("Loss / std")
    ax_critic.set_yscale("symlog", linthresh=1.0)
    ax_critic.grid(alpha=0.25)
    ax_critic.legend()

    ax_policy.plot(
        steps,
        policy_loss,
        linewidth=1.3,
        label="Policy loss",
        color="#264653",
    )
    ax_policy.set_title("Policy Update Signal")
    ax_policy.set_xlabel("Cumulative timesteps")
    ax_policy.set_ylabel("Policy loss")
    ax_policy.grid(alpha=0.25)
    entropy_ax = ax_policy.twinx()
    entropy_ax.plot(
        steps,
        entropy,
        alpha=0.35,
        label="Entropy",
        color="#e76f51",
    )
    entropy_ax.plot(
        steps,
        entropy_ma,
        linewidth=2.0,
        label=f"Entropy MA ({window})",
        color="#bc4749",
    )
    entropy_ax.set_ylabel("Entropy")
    lines, labels = ax_policy.get_legend_handles_labels()
    lines2, labels2 = entropy_ax.get_legend_handles_labels()
    ax_policy.legend(lines + lines2, labels + labels2, loc="best")

    fig.suptitle("MAPPO Training Diagnostics", fontsize=16)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_scenario_plot(history: list[dict], output_path: Path) -> None:
    if not history:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario_names = sorted({row["scenario"] for row in history})
    cmap = plt.get_cmap("tab10")
    scenario_colors = {
        name: cmap(idx % max(len(scenario_names), 1)) for idx, name in enumerate(scenario_names)
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[1.0, 1.15])
    ax_episode, ax_band = axes

    episodes = np.array([row["episode"] for row in history], dtype=np.int32)
    rewards = np.array([row["reward"] for row in history], dtype=np.float32)

    ax_episode.plot(
        episodes,
        rewards,
        color="#8d99ae",
        linewidth=1.0,
        alpha=0.5,
        label="Episode reward",
    )
    for name in scenario_names:
        indices = [idx for idx, row in enumerate(history) if row["scenario"] == name]
        scenario_episodes = np.array([history[idx]["episode"] for idx in indices], dtype=np.int32)
        scenario_rewards = np.array([history[idx]["reward"] for idx in indices], dtype=np.float32)
        ax_episode.scatter(
            scenario_episodes,
            scenario_rewards,
            s=28,
            alpha=0.9,
            color=scenario_colors[name],
            label=name,
        )

    ax_episode.set_title("Episode Reward by Scenario")
    ax_episode.set_xlabel("Episode")
    ax_episode.set_ylabel("Reward")
    ax_episode.grid(alpha=0.25)
    ax_episode.legend(ncol=2, fontsize=9)

    for name in scenario_names:
        scenario_rewards = np.array(
            [row["reward"] for row in history if row["scenario"] == name],
            dtype=np.float32,
        )
        if scenario_rewards.size == 0:
            continue

        occurrences = np.arange(1, scenario_rewards.size + 1, dtype=np.int32)
        expanding_mean = np.array(
            [scenario_rewards[:idx].mean() for idx in occurrences], dtype=np.float32
        )
        expanding_std = np.array(
            [scenario_rewards[:idx].std() for idx in occurrences], dtype=np.float32
        )
        lower = expanding_mean - expanding_std
        upper = expanding_mean + expanding_std

        color = scenario_colors[name]
        ax_band.plot(
            occurrences,
            expanding_mean,
            color=color,
            linewidth=2.0,
            label=f"{name} mean",
        )
        ax_band.fill_between(
            occurrences,
            lower,
            upper,
            color=color,
            alpha=0.18,
            label=f"{name} +/- 1 std",
        )

    ax_band.set_title("Scenario Reward Bands Across Occurrences")
    ax_band.set_xlabel("Scenario occurrence count")
    ax_band.set_ylabel("Reward")
    ax_band.grid(alpha=0.25)
    ax_band.legend(ncol=2, fontsize=9)

    fig.suptitle("MAPPO Scenario Diagnostics", fontsize=16)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def choose_scenario_key(args, episode: int, history: list[dict]) -> str:
    if args.scenario:
        return args.scenario

    if episode < len(WARMUP_SCENARIOS) * 3:
        return WARMUP_SCENARIOS[episode % len(WARMUP_SCENARIOS)]

    combined_focus = scenario_focus_pressure(history, "combined_crisis")
    if combined_focus >= 1.0 and episode % 3 != 0:
        return "combined_crisis"

    weights = []
    recent_by_scenario = {name: [] for name in TRAINING_SCENARIO_KEYS}
    previous_by_scenario = {name: [] for name in TRAINING_SCENARIO_KEYS}
    for row in reversed(history):
        scenario = row["scenario"]
        if scenario not in recent_by_scenario:
            continue
        recent_bucket = recent_by_scenario[scenario]
        previous_bucket = previous_by_scenario[scenario]
        if len(recent_bucket) < 8:
            recent_bucket.append(float(row["reward"]))
        elif len(previous_bucket) < 8:
            previous_bucket.append(float(row["reward"]))
        if all(
            len(recent_by_scenario[name]) >= 8 and len(previous_by_scenario[name]) >= 8
            for name in recent_by_scenario
        ):
            break

    scenario_names = list(TRAINING_SCENARIO_KEYS)
    for name in scenario_names:
        weight = SCENARIO_WEIGHTS.get(name, 1.0)
        weight *= 1.0 + scenario_focus_pressure(history, name)
        recent = recent_by_scenario[name]
        previous = previous_by_scenario[name]
        if recent:
            recent_mean = sum(recent) / len(recent)
            if recent_mean < 0.75:
                weight *= 1.0 + min(2.2, (0.75 - recent_mean) * 0.55)
        if recent and previous:
            improvement = (sum(recent) / len(recent)) - (sum(previous) / len(previous))
            if improvement < 0.12:
                weight *= 1.0 + min(1.4, 0.35 + (0.12 - improvement) * 1.6)
        if name == "combined_crisis":
            weight *= 1.35 + 0.45 * combined_focus
        weights.append(weight)

    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return str(np.random.choice(scenario_names, p=weights))


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    max_episodes = args.episodes
    if max_episodes is None and args.timesteps is None:
        max_episodes = 200

    sim_context = load_training_context()
    obs_dim = infer_obs_dim(sim_context)
    print("OBS_DIM =", obs_dim)

    mappo = MAPPO(
        obs_dim=obs_dim,
        action_dim=N_ACTIONS,
        n_agents=N_AGENTS,
        agent_names=ALL_AGENT_NAMES,
        lr=1.0e-4,
        action_masks=MAPPO_ACTION_MASKS,
        batch_size=args.batch_size,
    )

    history: list[dict] = []
    total_steps = 0
    episode = 0
    plot_path, metrics_path, scenario_plot_path = resolve_artifact_paths(args)
    checkpoint_path = Path(args.save) if args.save else None
    pending_trajs: list[dict] = []
    last_metrics = {
        "policy_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "elite_loss": 0.0,
        "adv_mean": 0.0,
        "adv_std": 0.0,
    }
    prior_scale = 0.0
    current_update_epochs = max(1, args.update_epochs)
    plateau_level = 0.0
    combined_focus = 0.0

    try:
        while True:
            if max_episodes is not None and episode >= max_episodes:
                break
            if args.timesteps is not None and total_steps >= args.timesteps:
                break

            plateau_level = compute_plateau_level(history)
            combined_focus = scenario_focus_pressure(history, "combined_crisis")
            entropy_base = max(0.008 * (0.989**episode), 0.00015)
            entropy_scale = 1.0 - 0.75 * plateau_level - 0.20 * min(combined_focus, 1.0)
            mappo.entropy_coef = max(entropy_base * max(entropy_scale, 0.12), 0.00004)
            prior_floor = 0.08 + 0.05 * plateau_level + 0.06 * min(combined_focus, 1.0)
            prior_scale = max(0.65 * (0.992**episode), prior_floor)
            current_update_epochs = max(
                1,
                args.update_epochs
                + int(round(2 * plateau_level + 2 * min(combined_focus, 1.0))),
            )
            scenario_key = choose_scenario_key(args, episode, history)
            scenario = copy.deepcopy(SCENARIOS[scenario_key])

            traj = run_episode(mappo, scenario, sim_context, prior_scale=prior_scale)
            pending_trajs.append(traj)

            episode_steps = len(traj["rewards"])
            total_steps += episode_steps
            total_reward = float(sum(traj["rewards"]))
            avg_reward = total_reward / max(episode_steps, 1)
            episode_metrics = dict(traj.get("episode_metrics", {}))

            update_applied = False
            if len(pending_trajs) >= max(1, args.rollout_episodes):
                merged = merge_trajectories(pending_trajs, mappo.agent_names)
                last_metrics = mappo.update(merged, epochs=current_update_epochs)
                pending_trajs.clear()
                update_applied = True

            history.append(
                {
                    "episode": episode,
                    "scenario": scenario_key,
                    "total_steps": total_steps,
                    "episode_steps": episode_steps,
                    "reward": total_reward,
                    "avg_reward": avg_reward,
                    "policy_loss": float(last_metrics["policy_loss"]),
                    "critic_loss": float(last_metrics["critic_loss"]),
                    "entropy": float(last_metrics["entropy"]),
                    "elite_loss": float(last_metrics["elite_loss"]),
                    "adv_mean": float(last_metrics["adv_mean"]),
                    "adv_std": float(last_metrics["adv_std"]),
                    "entropy_coef": float(mappo.entropy_coef),
                    "prior_scale": float(prior_scale),
                    "plateau_level": float(plateau_level),
                    "update_epochs": int(current_update_epochs),
                    "update_applied": int(update_applied),
                    **episode_metrics,
                }
            )

            print(
                f"Episode {episode} | steps={episode_steps} | total_steps={total_steps} "
                f"| scenario={scenario_key} | reward={total_reward:.2f} | avg={avg_reward:.2f} "
                f"| policy={last_metrics['policy_loss']:.4f} | critic={last_metrics['critic_loss']:.4f}"
            )
            episode += 1
    finally:
        if pending_trajs:
            merged = merge_trajectories(pending_trajs, mappo.agent_names)
            last_metrics = mappo.update(merged, epochs=current_update_epochs)
            pending_count = len(pending_trajs)
            for row in history[-pending_count:]:
                row["policy_loss"] = float(last_metrics["policy_loss"])
                row["critic_loss"] = float(last_metrics["critic_loss"])
                row["entropy"] = float(last_metrics["entropy"])
                row["elite_loss"] = float(last_metrics["elite_loss"])
                row["adv_mean"] = float(last_metrics["adv_mean"])
                row["adv_std"] = float(last_metrics["adv_std"])
                row["update_applied"] = int(row["update_applied"]) or 1

        if checkpoint_path is not None and history:
            saved_model = mappo.save(str(checkpoint_path))
            print(f"Saved MAPPO checkpoint to {saved_model}")

        if history:
            save_metrics_csv(history, metrics_path)
            save_training_plot(history, plot_path, window=max(args.plot_window, 1))
            save_scenario_plot(history, scenario_plot_path)
            print(f"Saved training metrics to {metrics_path}")
            print(f"Saved training plot to {plot_path}")
            print(f"Saved scenario plot to {scenario_plot_path}")


if __name__ == "__main__":
    main()
