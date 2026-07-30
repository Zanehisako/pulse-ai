# fast_env.py
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


class FastBloodEnv(gym.Env):
    """
    Vectorized approximation of the blood supply simulator.
    No SimPy, pure NumPy → FAST.
    """

    def __init__(self, n_envs=32, episode_steps=28):
        super().__init__()

        self.n_envs = n_envs
        self.episode_steps = episode_steps
        self.step_count = 0

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

        # Initial inventory: buffer_days * daily_products, distributed by component demand weights
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

        # Donor supply rate (daily donations producing ~2.3 components each, by collection weights)
        _donation_components = (
            QC_DAILY_COMPLETED_DONATIONS_EST * 2.3
        )  # avg components per donation
        self.donor_rate = np.tile(
            _donation_components * _comp_weights, (self.n_envs, 1)
        )

        self.total_shortage = np.zeros((self.n_envs, N_COMP))
        self.total_demand = np.zeros((self.n_envs, N_COMP))

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

            # ── Cost (scaled down for training stability)
            action_cost[i] = action.operational_cost * 0.1

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
        # 3. Demand (stochastic)
        # ─────────────────────────────────────────
        demand = np.random.poisson(self.demand_rate * demand_scale)

        # ─────────────────────────────────────────
        # 4. Supply (stochastic)
        # ─────────────────────────────────────────
        donations = np.random.poisson(self.donor_rate * donor_boost)

        # ─────────────────────────────────────────
        # 5. Pipeline (delays)
        # ─────────────────────────────────────────
        arrivals = self.pipeline[:, -1]
        self.pipeline = np.roll(self.pipeline, shift=1, axis=1)
        self.pipeline[:, 0] = donations

        # ─────────────────────────────────────────
        # 6. Inventory update
        # ─────────────────────────────────────────
        self.inventory += arrivals

        served = np.minimum(self.inventory, demand)
        self.inventory -= served

        shortage = demand - served

        # Track totals
        self.total_shortage += shortage
        self.total_demand += demand

        # ─────────────────────────────────────────
        # 7. Reward (CRITICAL PART)
        # ─────────────────────────────────────────

        # Component criticality weights (proportional to demand)
        _crit = np.array(
            [
                COMPONENT_DEMAND_WEIGHTS["RBC"],
                COMPONENT_DEMAND_WEIGHTS["PLATELETS"],
                COMPONENT_DEMAND_WEIGHTS["PLASMA"],
            ]
        )
        _crit = _crit / _crit.max() * 200  # scale so max penalty = 200

        # Strong shortage penalty (RBC is critical)
        shortage_penalty = (
            _crit[0] * shortage[:, 0]  # RBC
            + _crit[1] * shortage[:, 1]  # PLATELETS
            + _crit[2] * shortage[:, 2]  # PLASMA
        )

        # Anticipation penalty (LOW inventory BEFORE shortage)
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

        # Final reward
        raw_reward = -(shortage_penalty + low_inventory_penalty + action_cost)

        # normalize (VERY IMPORTANT)
        reward = raw_reward / 100.0
        reward = np.clip(reward, -10, 10)
        reward = reward.astype(np.float32)
        # smooth reward to help critic
        reward = 0.9 * reward + 0.1 * getattr(self, "prev_reward", 0.0)
        self.prev_reward = reward

        # ─────────────────────────────────────────
        # 8. Step update
        # ─────────────────────────────────────────
        self.step_count += 1
        done = self.step_count >= self.episode_steps

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

        obs = np.array(
            [
                shortage_rate,
                inv[0] / 100.0,  # normalize better
                inv[1] / 100.0,
                inv[2] / 100.0,
                min(inv[0] / 40.0, 1.0),
                self.donor_rate[0, 0] / 10.0,
                self.demand_rate[0, 0] / 10.0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=np.float32,
        )

        return np.clip(obs, 0, 1)
