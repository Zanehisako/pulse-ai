# fast_env_budget.py
"""
Budget-aware vectorized approximation of the blood supply simulator.
Extends FastBloodEnv with explicit budget tracking, budget-proportional
cost penalties, and early termination on budget exhaustion.

Drop-in replacement for FastBloodEnv in PPO training loops — same action
space, observation space (OBS_DIM=16), and return signature.
"""

import gymnasium as gym
import numpy as np
from calibration import (
    COMPONENT_DEMAND_WEIGHTS,
    QC_BASE_BUFFER_DAYS,
    QC_DAILY_COMPLETED_DONATIONS_EST,
    QC_DAILY_LABILE_PRODUCTS_EST,
)
from gymnasium import spaces
from ppo_shared import ACTION_KEYS, OBS_DIM
from scenarios import ACTION_CATALOG

COMPONENTS = ["RBC", "PLATELETS", "PLASMA"]
N_COMP = 3


class FastBloodBudgetEnv(gym.Env):
    """
    Budget-aware vectorized approximation of the blood supply simulator.
    Extends FastBloodEnv with explicit budget tracking, budget-proportional
    cost penalties, and early termination on budget exhaustion.

    Reward philosophy (mirrors ``calculate_step_reward_budget`` in engine.py):
        - Tier 1 — Shortage / patient safety           ~40 %
        - Tier 2 — Low-inventory anticipation            ~5 %
        - Tier 3 — Budget / cost efficiency             ~45 %
        - Smoothing / normalisation                     ~10 %

    The action cost uses the *real* ``operational_cost`` from ACTION_CATALOG
    (NOT scaled down by 0.1 as in FastBloodEnv) so that budget tracking is
    consistent and the agent feels the true fiscal impact of each decision.
    """

    def __init__(self, n_envs=32, episode_steps=28, episode_budget=6500.0):
        super().__init__()

        self.n_envs = n_envs
        self.episode_steps = episode_steps
        self.episode_budget = float(episode_budget)
        self.step_count = 0

        # Expected per-step budget (for overspend penalty)
        self._expected_step_budget = self.episode_budget / max(self.episode_steps, 1)

        self.action_space = spaces.Discrete(len(ACTION_KEYS) + 1)
        self.observation_space = spaces.Box(0.0, 1.0, (OBS_DIM,), dtype=np.float32)

        self.reset()

    # ─────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0

        # Budget tracking
        self.budget_remaining = np.full(self.n_envs, self.episode_budget)
        self.budget_spent = np.zeros(self.n_envs)

        # Initial inventory: buffer_days * daily_products * component weights
        _daily = QC_DAILY_LABILE_PRODUCTS_EST
        _buffer = QC_BASE_BUFFER_DAYS
        _comp_weights = np.array(
            [
                COMPONENT_DEMAND_WEIGHTS["RBC"],
                COMPONENT_DEMAND_WEIGHTS["PLATELETS"],
                COMPONENT_DEMAND_WEIGHTS["PLASMA"],
            ]
        )
        base_inv = _daily * _buffer * _comp_weights
        self.inventory = np.tile(base_inv, (self.n_envs, 1)) * np.random.uniform(
            0.8, 1.2, (self.n_envs, N_COMP)
        )
        self.pipeline = np.zeros((self.n_envs, 4, N_COMP))

        # Demand rate per step (daily products distributed by component weights)
        self.demand_rate = np.tile(_daily * _comp_weights, (self.n_envs, 1))

        # Donor supply rate (daily donations producing ~2.3 components each)
        _donation_components = QC_DAILY_COMPLETED_DONATIONS_EST * 2.3
        self.donor_rate = np.tile(
            _donation_components * _comp_weights, (self.n_envs, 1)
        )

        self.total_shortage = np.zeros((self.n_envs, N_COMP))
        self.total_demand = np.zeros((self.n_envs, N_COMP))

        self.prev_reward = 0.0

        return self._obs(), {}

    # ─────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────
    def step(self, actions):
        if np.isscalar(actions):
            actions = np.full(self.n_envs, actions)

        # ─────────────────────────────────────────
        # 1. Initialize modifiers
        # ─────────────────────────────────────────
        donor_boost = np.ones((self.n_envs, 3))
        demand_scale = np.ones((self.n_envs, 3))
        action_cost = np.zeros(self.n_envs)

        # ─────────────────────────────────────────
        # 2. Apply actions (STRONG effects)
        # ─────────────────────────────────────────
        for i, a in enumerate(actions):
            if a == 0:
                continue

            action = ACTION_CATALOG[ACTION_KEYS[a - 1]]

            # ── Cost: use REAL operational_cost (no 0.1 scaling) ──
            action_cost[i] = action.operational_cost

            # ── Supply boost (strong nonlinear effect)
            donor_boost[i] *= 1.0 + 2.0 * (action.donor_arrival_factor - 1.0)

            # ── Demand reduction (bounded)
            demand_scale[i] *= 1.0 - 0.7 * (1.0 - action.demand_management_factor)

            # ── LAB SPEED / RELEASE SPEED → reduce delay
            if action.lab_speed_factor < 1.0 or action.release_delay_factor < 1.0:
                self.pipeline[i] = np.roll(self.pipeline[i], shift=-1, axis=0)

            # ── MOBILE UNITS → immediate supply injection
            if action.mobile_units > 0:
                self.inventory[i] += np.array([15, 5, 8]) * action.mobile_units

            # ── EMERGENCY SHARE → strong immediate rescue
            if action.emergency_share:
                self.inventory[i] += np.array([25, 10, 10])

            # ── NATIONAL / REGIONAL AID → moderate boost
            if action.regional_replenishment_factor > 1.0:
                self.inventory[i] += np.array([10, 5, 5])

            # ── STAFF → improves throughput
            staff_boost = (
                action.extra_nurses
                + action.extra_lab_staff
                + action.extra_processing_staff
            )
            if staff_boost > 0:
                donor_boost[i] *= 1.0 + 0.1 * staff_boost

            # ── HOURS EXTENSION → more donors
            if action.hours_extension_h > 0:
                donor_boost[i] *= 1.3

        # ─────────────────────────────────────────
        # 3. Budget tracking: deduct action costs
        # ─────────────────────────────────────────
        self.budget_spent += action_cost
        self.budget_remaining -= action_cost
        # Clamp remaining to zero (don't go negative for accounting)
        self.budget_remaining = np.maximum(self.budget_remaining, 0.0)

        # ─────────────────────────────────────────
        # 4. Demand (stochastic)
        # ─────────────────────────────────────────
        demand = np.random.poisson(self.demand_rate * demand_scale)

        # ─────────────────────────────────────────
        # 5. Supply (stochastic)
        # ─────────────────────────────────────────
        donations = np.random.poisson(self.donor_rate * donor_boost)

        # ─────────────────────────────────────────
        # 6. Pipeline (delays)
        # ─────────────────────────────────────────
        arrivals = self.pipeline[:, -1]
        self.pipeline = np.roll(self.pipeline, shift=1, axis=1)
        self.pipeline[:, 0] = donations

        # ─────────────────────────────────────────
        # 7. Inventory update
        # ─────────────────────────────────────────
        self.inventory += arrivals

        served = np.minimum(self.inventory, demand)
        self.inventory -= served

        shortage = demand - served

        # Track totals
        self.total_shortage += shortage
        self.total_demand += demand

        # ─────────────────────────────────────────
        # 8. Reward (BUDGET-AWARE — CRITICAL PART)
        # ─────────────────────────────────────────

        # ── Component criticality weights (proportional to demand) ──
        _crit = np.array(
            [
                COMPONENT_DEMAND_WEIGHTS["RBC"],
                COMPONENT_DEMAND_WEIGHTS["PLATELETS"],
                COMPONENT_DEMAND_WEIGHTS["PLASMA"],
            ]
        )
        _crit = _crit / _crit.max() * 200  # scale so max penalty = 200

        # ── TIER 1: Shortage penalty (~40% of reward weight) ──
        shortage_penalty = (
            _crit[0] * shortage[:, 0]  # RBC
            + _crit[1] * shortage[:, 1]  # PLATELETS
            + _crit[2] * shortage[:, 2]  # PLASMA
        )

        # ── Low-inventory anticipation penalty (~5%) ──
        base_inv = (
            QC_DAILY_LABILE_PRODUCTS_EST
            * QC_BASE_BUFFER_DAYS
            * np.array(
                [
                    COMPONENT_DEMAND_WEIGHTS["RBC"],
                    COMPONENT_DEMAND_WEIGHTS["PLATELETS"],
                    COMPONENT_DEMAND_WEIGHTS["PLASMA"],
                ]
            )
        )
        low_inventory_penalty = (
            np.maximum(0, base_inv[0] * 0.3 - self.inventory[:, 0]) * 5
        )

        # ══════════════════════════════════════════════════════════════
        # TIER 3: Budget / Cost Efficiency              (weight ~45 %)
        # ══════════════════════════════════════════════════════════════

        # (a) Raw action cost penalty — 3× multiplier vs standard env
        #     Every dollar hurts proportionally.
        action_cost_penalty = action_cost * 3.0

        # (b) Budget overspend penalty — extra sting when step cost
        #     exceeds the expected per-step budget.
        overspend = np.maximum(action_cost - self._expected_step_budget, 0.0)
        overspend_rate = overspend / max(self._expected_step_budget, 1.0)
        budget_overspend_penalty = np.minimum(overspend_rate, 1.0) * 150.0

        # (c) Budget exhaustion penalty — running out early is very bad.
        #     Penalty scales with how much episode time remains.
        budget_exhaustion_penalty = np.zeros(self.n_envs)
        exhausted_mask = self.budget_remaining <= 1e-6
        if np.any(exhausted_mask) and self.step_count < self.episode_steps - 1:
            remaining_frac = (self.episode_steps - self.step_count) / max(
                self.episode_steps, 1
            )
            # The earlier the budget is exhausted, the bigger the penalty
            budget_exhaustion_penalty[exhausted_mask] = 300.0 * remaining_frac

        # ── Combine all reward terms ──
        raw_reward = -(
            shortage_penalty
            + low_inventory_penalty
            + action_cost_penalty
            + budget_overspend_penalty
            + budget_exhaustion_penalty
        )

        # Normalize (VERY IMPORTANT — same scale as FastBloodEnv)
        reward = raw_reward / 100.0
        reward = np.clip(reward, -10, 10)
        reward = reward.astype(np.float32)

        # Smooth reward to help critic
        reward = 0.9 * reward + 0.1 * self.prev_reward
        self.prev_reward = reward

        # ─────────────────────────────────────────
        # 9. Step update + early termination
        # ─────────────────────────────────────────
        self.step_count += 1
        done = self.step_count >= self.episode_steps

        # Early termination: if budget exhausted with > 2 steps remaining
        if not done:
            budget_exhausted_early = (
                self.budget_remaining[0] <= 1e-6
                and self.step_count < self.episode_steps - 2
            )
            if budget_exhausted_early:
                done = True

        obs = self._obs()

        return obs, reward[0], done, False, {}

    # ─────────────────────────────────────────
    # OBSERVATION
    # ─────────────────────────────────────────
    def _obs(self):
        inv = self.inventory[0]

        shortage_rate = self.total_shortage[0].sum() / (
            self.total_demand[0].sum() + 1e-6
        )

        # active_cost_ratio: fraction of total budget spent so far
        active_cost_ratio = float(self.budget_spent[0] / max(self.episode_budget, 1.0))

        # budget_pressure: how close to 80 % of budget have we burned?
        # Saturates at 1.0 once 80 % is spent, giving the agent early warning.
        budget_pressure = float(
            min(self.budget_spent[0] / max(self.episode_budget * 0.8, 1.0), 1.0)
        )

        obs = np.array(
            [
                shortage_rate,  # 0  – overall shortage rate
                inv[0] / 100.0,  # 1  – RBC inventory (normalised)
                inv[1] / 100.0,  # 2  – PLT inventory (normalised)
                inv[2] / 100.0,  # 3  – PLS inventory (normalised)
                min(inv[0] / 40.0, 1.0),  # 4  – RBC days-of-stock proxy
                self.donor_rate[0, 0] / 10.0,  # 5  – donor rate (normalised)
                self.demand_rate[0, 0] / 10.0,  # 6  – demand rate (normalised)
                0,  # 7  – (reserved)
                0,  # 8  – (reserved)
                0,  # 9  – (reserved)
                0,  # 10 – (reserved)
                active_cost_ratio,  # 11 – budget spent / total budget
                0,  # 12 – (reserved)
                0,  # 13 – (reserved)
                0,  # 14 – (reserved)
                budget_pressure,  # 15 – budget pressure signal
            ],
            dtype=np.float32,
        )

        return np.clip(obs, 0, 1)
