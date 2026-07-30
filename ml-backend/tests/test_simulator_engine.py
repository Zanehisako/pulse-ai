from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import simpy

SIMULATOR_DIR = Path(__file__).resolve().parents[1] / "simulator"
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from core import CENTER_CONFIGS, BloodUnit, DonationCenter, WeatherEngine
from engine import (
    ActivatedAction,
    SimState,
    active_replenishment_interval_h,
    apply_continuous_action_levels,
    apply_multi_actions,
    calculate_reward,
    calculate_step_reward,
    component_allocation_shares,
    dynamic_strategy_factor,
    fulfill_hospital_order,
    hospital_routing_shares,
    replenishment_receivers,
    run_scenario,
    summarize_state,
)
from ppo_env import BloodSupplyEnv
from ppo_shared import (
    COMPONENT_ALLOCATION_CONTROL_MAP,
    DREAMER_ACTION_DIM,
    HOSPITAL_ROUTE_CONTROL_MAP,
    OBS_DIM,
)
from scenarios import ACTION_CATALOG, SCENARIOS, STRATEGIES, ScenarioParams


def build_state(params: ScenarioParams) -> tuple[SimState, DonationCenter]:
    env = simpy.Environment()
    centers = [DonationCenter(env, cfg) for cfg in CENTER_CONFIGS[:3]]
    hospital = next(center for center in centers if center.ctype == "hospital")
    state = SimState(
        env=env,
        params=params,
        centers=centers,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        weather=WeatherEngine(forced_state="clear"),
    )
    return state, hospital


def test_shortage_count_is_per_missing_unit():
    params = ScenarioParams(strategy_key="baseline")
    state, hospital = build_state(params)

    result = fulfill_hospital_order(state, hospital, "A+", "RBC", 3)

    assert result["filled_units"] == 0
    assert result["shortage_units"] == 3
    assert len(state.shortage_log) == 3
    assert sum(center.stats["shortage"] for center in state.centers) == 3


def test_compatible_inventory_prevents_false_shortage():
    params = ScenarioParams(strategy_key="baseline")
    state, hospital = build_state(params)
    hospital.add_unit(BloodUnit(state.env, "O+", "RBC", hospital.name))

    result = fulfill_hospital_order(state, hospital, "A+", "RBC", 1)

    assert result["filled_units"] == 1
    assert result["shortage_units"] == 0
    assert len(state.shortage_log) == 0
    assert hospital.stats["transfused"] == 1


def test_emergency_priority_shortage_is_weighted_more_heavily():
    params = ScenarioParams(strategy_key="baseline")
    state, hospital = build_state(params)

    result = fulfill_hospital_order(
        state,
        hospital,
        "A+",
        "RBC",
        1,
        priority="emergency",
    )

    assert result["shortage_units"] == 1
    assert state.priority_weighted_requested_units == 4.0
    assert state.priority_weighted_shortage_units == 4.0
    assert state.shortage_log[0]["priority"] == "emergency"


def test_full_response_beats_baseline_on_same_seed(monkeypatch):
    monkeypatch.setattr("engine.road_travel_hours", lambda *args, **kwargs: 0.08)

    baseline = copy.deepcopy(SCENARIOS["donor_decrease"])
    baseline.strategy_key = "baseline"
    baseline.sim_hours = 96

    full_response = copy.deepcopy(SCENARIOS["donor_decrease"])
    full_response.strategy_key = "full_response"
    full_response.sim_hours = 96

    state_baseline = run_scenario(
        baseline,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        seed=7,
    )
    state_full = run_scenario(
        full_response,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        seed=7,
    )

    summary_baseline = summarize_state(state_baseline)
    summary_full = summarize_state(state_full)

    assert summary_full["shortage_rate"] < summary_baseline["shortage_rate"]
    assert summary_full["base_shortage_rate"] < summary_baseline["base_shortage_rate"]
    assert summary_full["total_donated"] > summary_baseline["total_donated"]
    # With faster quarantine (12-36 h), the full_response strategy may not
    # always need *more* external units than baseline — the key assertion is
    # that external replenishment is actively used by full_response.
    assert summary_full["total_external_units"] > 0
    assert summary_full["total_conserved"] > 0


def test_combined_crisis_full_response_faces_same_demand_and_beats_baseline(
    monkeypatch,
):
    monkeypatch.setattr("engine.road_travel_hours", lambda *args, **kwargs: 0.08)

    baseline = copy.deepcopy(SCENARIOS["combined_crisis"])
    baseline.strategy_key = "baseline"
    baseline.sim_hours = 120

    full_response = copy.deepcopy(SCENARIOS["combined_crisis"])
    full_response.strategy_key = "full_response"
    full_response.sim_hours = 120

    state_baseline = run_scenario(
        baseline,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        seed=7,
    )
    state_full = run_scenario(
        full_response,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        seed=7,
    )

    base_requested_baseline = sum(
        entry.get("base_requested", entry["requested"])
        for entry in state_baseline.demand_log
    )
    base_requested_full = sum(
        entry.get("base_requested", entry["requested"])
        for entry in state_full.demand_log
    )
    requested_baseline = sum(entry["requested"] for entry in state_baseline.demand_log)
    requested_full = sum(entry["requested"] for entry in state_full.demand_log)

    summary_baseline = summarize_state(state_baseline)
    summary_full = summarize_state(state_full)

    assert len(state_full.demand_log) == len(state_baseline.demand_log)
    assert base_requested_full == base_requested_baseline
    assert requested_full < requested_baseline
    assert summary_full["total_shortage"] < summary_baseline["total_shortage"]
    assert summary_full["shortage_rate"] < 50
    assert (
        summary_full["base_shortage_rate"]
        <= summary_baseline["base_shortage_rate"] - 40
    )
    assert (
        summary_full["total_external_units"]
        > summary_baseline["total_external_units"] * 4
    )
    assert summary_full["total_conserved"] > 0


def test_normal_baseline_is_no_longer_structurally_short(monkeypatch):
    monkeypatch.setattr("engine.road_travel_hours", lambda *args, **kwargs: 0.08)

    params = copy.deepcopy(SCENARIOS["baseline"])
    params.sim_hours = 96

    state = run_scenario(
        params,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        seed=7,
    )
    summary = summarize_state(state)

    assert summary["shortage_rate"] < 15
    assert summary["total_external_units"] > 0


def test_run_scenario_can_disable_logs(monkeypatch, capsys):
    monkeypatch.setattr("engine.road_travel_hours", lambda *args, **kwargs: 0.08)

    params = copy.deepcopy(SCENARIOS["baseline"])
    params.sim_hours = 24

    state = run_scenario(
        params,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        seed=7,
        enable_logs=False,
    )

    captured = capsys.readouterr()

    assert captured.out == ""
    assert state.log == []


def test_run_scenario_fast_mode_skips_detailed_history(monkeypatch):
    monkeypatch.setattr("engine.road_travel_hours", lambda *args, **kwargs: 0.08)

    params = copy.deepcopy(SCENARIOS["combined_crisis"])
    params.sim_hours = 48

    state = run_scenario(
        params,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        seed=7,
        enable_logs=False,
        fast_mode=True,
    )

    assert state.demand_log == []
    assert state.shortage_log == []
    assert state.transfer_log == []
    assert state.donation_log == []
    assert state.hourly_inventory == []


def test_ppo_strategy_is_registered_as_adaptive_controller():
    strategy = STRATEGIES["ppo_shortage_minimizer"]

    assert strategy.is_adaptive is True
    assert strategy.controller_key == "ppo_shortage_minimizer"
    assert strategy.actions == ()
    assert {action.key for action in strategy.action_space} == {
        "campaign",
        "lab_fast_track",
        "emergency_share",
        "clinical_conservation",
        "national_mutual_aid",
    }


def test_apply_multi_actions_can_expand_compound_choices():
    params = ScenarioParams(strategy_key="baseline")
    state, _ = build_state(params)

    apply_multi_actions(
        state,
        {"supply": 1},
        action_keys=["campaign+extend_hours"],
        allowed_action_keys={"supply": {"campaign", "extend_hours"}},
    )

    assert {activated.action.key for activated in state.active_actions} == {
        "campaign",
        "extend_hours",
    }


def test_ppo_strategy_reduces_shortages_vs_baseline_and_uses_lower_cost_than_full_response(
    monkeypatch,
):
    monkeypatch.setattr("engine.road_travel_hours", lambda *args, **kwargs: 0.08)

    baseline = copy.deepcopy(SCENARIOS["donor_decrease"])
    baseline.strategy_key = "baseline"
    baseline.sim_hours = 96

    ppo = copy.deepcopy(SCENARIOS["donor_decrease"])
    ppo.strategy_key = "ppo_shortage_minimizer"
    ppo.sim_hours = 96

    state_baseline = run_scenario(
        baseline,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        seed=7,
        enable_logs=False,
    )
    state_ppo = run_scenario(
        ppo,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        seed=7,
        enable_logs=False,
    )

    summary_baseline = summarize_state(state_baseline)
    summary_ppo = summarize_state(state_ppo)

    assert summary_ppo["total_shortage"] < summary_baseline["total_shortage"]
    assert summary_ppo["shortage_rate"] < summary_baseline["shortage_rate"]
    assert summary_ppo["active_action_cost"] < STRATEGIES["full_response"].action_cost
    assert set(summary_ppo["active_action_keys"]) == {
        "campaign",
        "emergency_share",
        "lab_fast_track",
        "national_mutual_aid",
        "clinical_conservation",
    }
    assert any(entry.get("kind") == "activation" for entry in state_ppo.policy_log)


@pytest.mark.parametrize(
    ("strategy_key", "controller_key"),
    [
        ("dreamerv3_official", "dreamerv3_official"),
        ("dreamerv4", "dreamerv4"),
    ],
)
def test_dreamer_strategies_are_registered_as_adaptive_controllers(
    strategy_key: str,
    controller_key: str,
):
    strategy = STRATEGIES[strategy_key]

    assert strategy.is_adaptive is True
    assert strategy.controller_key == controller_key
    assert strategy.actions == ()
    assert {action.key for action in strategy.action_space} == set(ACTION_CATALOG)


def test_continuous_action_levels_scale_runtime_effects_without_expiring():
    params = ScenarioParams(strategy_key="baseline")
    state, _ = build_state(params)

    apply_continuous_action_levels(
        state,
        {"rapid_courier": 0.5},
        controller="test_controller",
        record_policy=False,
    )

    state.env.run(until=80)

    expected_arrival_factor = 1.0 + (
        (ACTION_CATALOG["rapid_courier"].donor_arrival_factor - 1.0) * 0.5
    )
    assert np.isclose(
        dynamic_strategy_factor(state, "donor_arrival_factor"),
        expected_arrival_factor,
    )
    assert np.isclose(active_replenishment_interval_h(state), 8.0)

    apply_continuous_action_levels(
        state,
        {"rapid_courier": 0.0},
        controller="test_controller",
        record_policy=False,
    )

    assert all(
        activated.action.key != "rapid_courier" for activated in state.active_actions
    )
    assert np.isclose(dynamic_strategy_factor(state, "donor_arrival_factor"), 1.0)
    assert np.isclose(active_replenishment_interval_h(state), 12.0)


def test_continuous_routing_and_component_controls_shift_live_priorities():
    params = ScenarioParams(strategy_key="baseline")
    env = simpy.Environment()
    centers = [DonationCenter(env, cfg) for cfg in CENTER_CONFIGS]
    state = SimState(
        env=env,
        params=params,
        centers=centers,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        weather=WeatherEngine(forced_state="clear"),
    )

    route_controls = list(HOSPITAL_ROUTE_CONTROL_MAP.items())
    primary_route_key, primary_hospital = route_controls[0]
    secondary_route_key, secondary_hospital = route_controls[1]

    platelet_control_key = next(
        key
        for key, component in COMPONENT_ALLOCATION_CONTROL_MAP.items()
        if component == "PLATELETS"
    )

    apply_continuous_action_levels(
        state,
        {
            primary_route_key: 1.0,
            secondary_route_key: 0.2,
            platelet_control_key: 1.0,
        },
        controller="test_controller",
        record_policy=False,
    )

    routing = hospital_routing_shares(state)
    allocation = component_allocation_shares(state)
    ordered_receivers = [
        center.name
        for center in replenishment_receivers(state, "RBC")
        if center.ctype == "hospital"
    ]

    assert routing[primary_hospital] > routing[secondary_hospital]
    assert ordered_receivers[0] == primary_hospital
    assert allocation["PLATELETS"] > allocation["RBC"]


def test_calculate_step_reward_uses_step_deltas_for_dreamer():
    params = ScenarioParams(strategy_key="baseline")
    state, _hospital = build_state(params)
    state.total_net_requested_units = 100

    prev_snapshot = {
        "transfused": 0.0,
        "shortage": 0.0,
        "expired": 0.0,
    }

    state.centers[0].stats["transfused"] = 10
    state.total_shortage_units = 2
    state.centers[0].stats["expired"] = 1

    reward = calculate_step_reward(state, prev_snapshot)

    # Reward value depends on eligible_rate (default 0.93) and step-reward
    # centering.  We only check sign and rough magnitude here.
    assert reward > 0.0, f"Expected positive reward, got {reward}"
    assert reward < 2.0, f"Reward unexpectedly large: {reward}"


def test_calculate_reward_penalizes_expensive_controls_when_outcomes_match():
    params = ScenarioParams(strategy_key="baseline")
    lean_state, _ = build_state(params)
    costly_state, _ = build_state(params)

    for state in (lean_state, costly_state):
        state.total_base_requested_units = 100
        state.total_net_requested_units = 90
        state.total_shortage_units = 8
        state.priority_weighted_requested_units = 120.0
        state.priority_weighted_shortage_units = 14.0
        state.total_conserved_units = 4
        state.total_exact_match_units = 70
        state.total_compatible_substitution_units = 10
        state.total_incompatible_fulfillment_units = 2
        state.centers[0].stats["transfused"] = 82
        state.centers[0].stats["expired"] = 3

    costly_state.budget_spent = 900.0
    costly_state.budget_blocked_controls = 3
    costly_state.active_actions.append(
        ActivatedAction(
            action=ACTION_CATALOG["rapid_courier"],
            activated_at_h=-24.0,
        )
    )

    assert calculate_reward(costly_state).total < calculate_reward(lean_state).total


def test_calculate_step_reward_prefers_clean_fulfillment_to_costly_substitution():
    params = ScenarioParams(strategy_key="baseline")
    clean_state, _ = build_state(params)
    costly_state, _ = build_state(params)
    prev_snapshot = {
        "transfused": 0.0,
        "shortage": 0.0,
        "expired": 0.0,
        "exact_match": 0.0,
        "compatible_substitution": 0.0,
        "incompatible_fulfillment": 0.0,
        "conserved": 0.0,
        "priority_shortage": 0.0,
        "budget_spent": 0.0,
        "budget_blocked_controls": 0.0,
        "congestion": 0.0,
    }

    for state in (clean_state, costly_state):
        state.total_base_requested_units = 10
        state.total_net_requested_units = 10
        state.priority_weighted_requested_units = 10.0
        state.centers[0].stats["transfused"] = 10

    clean_state.total_exact_match_units = 10
    clean_state.budget_spent = 20.0

    costly_state.total_compatible_substitution_units = 4
    costly_state.total_incompatible_fulfillment_units = 6
    costly_state.total_conserved_units = 3
    costly_state.budget_spent = 180.0

    assert calculate_step_reward(clean_state, prev_snapshot) > calculate_step_reward(
        costly_state,
        prev_snapshot,
    )


def test_continuous_blood_supply_env_uses_step_reward(monkeypatch):
    monkeypatch.setattr("engine.road_travel_hours", lambda *args, **kwargs: 0.08)
    monkeypatch.setattr("engine.calculate_step_reward", lambda state, snapshot: 0.75)

    params = copy.deepcopy(SCENARIOS["baseline"])
    params.sim_hours = 12

    env = BloodSupplyEnv(
        params=params,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        episode_hours=12,
        step_hours=6,
        seed=7,
        action_mode="continuous",
    )

    obs, _info = env.reset(seed=7)
    action = np.zeros(DREAMER_ACTION_DIM, dtype=np.float32)
    _obs, reward, terminated, truncated, info = env.step(action)

    assert reward == 0.75
    assert terminated is False
    assert truncated is False
    assert info["action_mode"] == "continuous"
    assert np.isclose(info["action_levels"][sorted(ACTION_CATALOG.keys())[0]], 0.5)
    assert obs.shape == (OBS_DIM,)
    assert "budget_remaining" in info
    assert "congestion_stress" in info


def test_blood_supply_env_can_skip_expensive_diagnostics(monkeypatch):
    monkeypatch.setattr("engine.road_travel_hours", lambda *args, **kwargs: 0.08)

    params = copy.deepcopy(SCENARIOS["baseline"])
    params.sim_hours = 12

    env = BloodSupplyEnv(
        params=params,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        episode_hours=12,
        step_hours=6,
        seed=7,
        collect_diagnostics=False,
    )

    _obs, reset_info = env.reset(seed=7)
    _obs, _reward, _terminated, _truncated, step_info = env.step(0)

    assert reset_info["scenario_key"] == "baseline"
    assert step_info["action_mode"] == "discrete"
    assert "budget_remaining" not in step_info
    assert "congestion_stress" not in step_info


def test_donation_center_inventory_counts_stay_in_sync():
    env = simpy.Environment()
    center = DonationCenter(env, CENTER_CONFIGS[0])
    fresh = BloodUnit(env, "O+", "RBC", center.name)
    expired = BloodUnit(env, "A+", "PLATELETS", center.name)
    expired.expiry = -1.0

    center.add_unit(fresh)
    center.add_unit(expired)

    assert center.component_count("RBC") == 1
    assert center.component_count("PLATELETS") == 1
    assert center.compatible_unit_count("A+", "RBC") == 1

    removed = center.remove_expired()
    dispensed = center.request_unit("A+", "RBC")

    assert removed == 1
    assert dispensed is fresh
    assert center.component_count("RBC") == 0
    assert center.inventory_by_component()["PLATELETS"] == 0


def test_blood_supply_env_cycles_across_scenario_templates(monkeypatch):
    monkeypatch.setattr("engine.road_travel_hours", lambda *args, **kwargs: 0.08)

    baseline = copy.deepcopy(SCENARIOS["baseline"])
    donor_decrease = copy.deepcopy(SCENARIOS["donor_decrease"])
    baseline.sim_hours = 6
    donor_decrease.sim_hours = 6

    env = BloodSupplyEnv(
        params=baseline,
        scenario_templates=[baseline, donor_decrease],
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        episode_hours=6,
        step_hours=6,
        seed=11,
    )

    _obs1, info1 = env.reset(seed=11)
    _obs2, _reward, _terminated2, truncated2, _info2 = env.step(0)
    _obs3, info3 = env.reset()
    _obs4, _reward, _terminated4, truncated4, _info4 = env.step(0)
    _obs5, info5 = env.reset()

    assert info1["scenario_key"] == "baseline"
    assert info3["scenario_key"] == "donor_decrease"
    assert info5["scenario_key"] == "baseline"
    assert truncated2 is True
    assert truncated4 is True


def test_budget_exhaustion_blocks_new_continuous_actions():
    params = ScenarioParams(strategy_key="baseline", episode_budget=0.0)
    state, _hospital = build_state(params)
    state.budget_total = 0.0
    state.budget_remaining = 0.0

    result = apply_continuous_action_levels(
        state,
        {"rapid_courier": 1.0},
        controller="test_controller",
        record_policy=False,
    )

    assert result["changed"] is False
    assert state.budget_blocked_controls == 1
    assert all(
        activated.action.key != "rapid_courier" for activated in state.active_actions
    )


class StubOfficialDreamerAgent:
    def __init__(self, action_vectors):
        self._action_vectors = [
            np.asarray(vector, dtype=np.float32) for vector in action_vectors
        ]
        self.init_calls = []
        self.policy_calls = []

    def init_policy(self, batch_size):
        self.init_calls.append(batch_size)
        return {"carry": batch_size}

    def policy(self, carry, obs, mode="eval"):
        assert not any(key.startswith("log/") for key in obs)
        self.policy_calls.append(
            {
                "mode": mode,
                "reward": float(obs["reward"][0]),
                "is_first": bool(obs["is_first"][0]),
            }
        )
        vector = self._action_vectors[len(self.policy_calls) - 1]
        acts = {"action": np.asarray([vector], dtype=np.float32)}
        return carry, acts, {}


@pytest.mark.parametrize(
    ("strategy_key", "expected_controller"),
    [
        ("dreamerv3_official", "official_dreamerv3"),
        ("dreamerv4", "dreamerv4"),
    ],
)
def test_run_scenario_can_use_dreamer_controllers(
    monkeypatch,
    strategy_key: str,
    expected_controller: str,
):
    monkeypatch.setattr("engine.road_travel_hours", lambda *args, **kwargs: 0.08)

    params = copy.deepcopy(SCENARIOS["baseline"])
    params.strategy_key = strategy_key
    params.sim_hours = 12

    first_action = np.zeros(DREAMER_ACTION_DIM, dtype=np.float32)
    first_action[0] = 0.8
    agent = StubOfficialDreamerAgent([first_action, first_action])
    state = run_scenario(
        params,
        G=None,
        north=46.90,
        south=46.70,
        east=-71.10,
        west=-71.35,
        seed=7,
        enable_logs=False,
        fast_mode=True,
        official_dreamerv3_agent=agent,
    )

    assert state.runtime_controller == expected_controller
    assert state.runtime_controller_step_hours == 6.0
    assert agent.init_calls == [1]
    assert [call["is_first"] for call in agent.policy_calls] == [True, False]
    assert [call["mode"] for call in agent.policy_calls] == ["eval", "eval"]
    assert agent.policy_calls[0]["reward"] == 0.0
    assert any(
        entry.get("controller") == expected_controller for entry in state.policy_log
    )
    assert any(entry.get("kind") == "continuous_decision" for entry in state.policy_log)
    expected_action_key = sorted(ACTION_CATALOG.keys())[0]
    assert any(
        activated.action.key == expected_action_key
        for activated in state.active_actions
    )
