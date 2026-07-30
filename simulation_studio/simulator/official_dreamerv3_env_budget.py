"""
Budget-aware DreamerV3 environment variant.

Wraps the blood-supply simulator with a reward function that heavily penalises
budget overspend and action cost.  When the agent exhausts its fixed budget
before the simulation ends the episode is **terminated early** with a large
negative terminal penalty — teaching the agent that blowing the budget has
catastrophic consequences.

Usage
-----
Drop-in replacement for ``OfficialDreamerBloodEnv`` in DreamerV3 training::

    env = OfficialDreamerBloodBudgetEnv(
        scenario_keys, sim_context,
        step_hours=6.0,
        seed=0,
        fixed_budget=5000.0,   # override per-scenario budgets
    )
"""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

from dreamerv3_runtime_paths import inject_official_runtime_paths

inject_official_runtime_paths()

import elements
import embodied
import numpy as np

CACHE_ROOT = Path(tempfile.gettempdir()) / "pios_dreamer_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

from ppo_env import BloodSupplyEnv
from ppo_shared import (
    CONTINUOUS_ACTION_HIGH,
    CONTINUOUS_ACTION_LOW,
    DREAMER_ACTION_DIM,
    OBS_DIM,
)
from scenarios import SCENARIOS
from train_mappo import prepare_mappo_scenario


class OfficialDreamerBloodBudgetEnv(embodied.Env):
    """
    Budget-aware variant of ``OfficialDreamerBloodEnv``.

    Key differences from the standard environment
    ----------------------------------------------
    * Uses ``engine.calculate_step_reward_budget`` instead of the regular
      step reward — budget overspend and action cost are first-class
      optimisation targets.
    * **Early termination on budget exhaustion.**  If the agent spends all of
      its budget before the episode ends (with > 2 % simulation time left),
      the episode is terminated immediately with a harsh negative reward of
      ``BUDGET_EXHAUSTION_PENALTY``.
    * Optionally overrides each scenario's ``episode_budget`` with a single
      ``fixed_budget`` value, making the constraint uniform across scenarios.

    Parameters
    ----------
    scenario_keys : list[str]
        Scenario names from ``SCENARIOS`` to train on.
    sim_context : tuple
        ``(G, north, south, east, west)`` as returned by
        ``load_training_context()``.
    step_hours : float
        Simulator hours advanced per agent action (default 6).
    seed : int
        Base RNG seed.
    fixed_budget : float | None
        When set, *every* episode uses this budget instead of the per-scenario
        ``episode_budget``.  Useful for ablation experiments.
    """

    BUDGET_EXHAUSTION_PENALTY: float = -0.95

    def __init__(
        self,
        scenario_keys: list[str] | tuple[str, ...],
        sim_context,
        *,
        step_hours: float = 6.0,
        seed: int = 0,
        fixed_budget: float | None = None,
    ):
        self._scenario_keys = list(scenario_keys)
        if not self._scenario_keys:
            raise ValueError(
                "OfficialDreamerBloodBudgetEnv requires at least one scenario key."
            )

        self._sim_context = sim_context
        self._step_hours = step_hours
        self._scenario_templates = {
            key: prepare_mappo_scenario(SCENARIOS[key]) for key in self._scenario_keys
        }
        self._base_seed = int(seed)
        self._episode_seed = int(seed)
        self._done = True
        self._env: BloodSupplyEnv | None = None
        self._scenario_key = self._scenario_keys[0]
        self._scenario_index = 0
        self._fixed_budget = fixed_budget

    # ------------------------------------------------------------------
    # embodied.Env interface
    # ------------------------------------------------------------------

    @property
    def obs_space(self):
        return {
            "vector": elements.Space(np.float32, (OBS_DIM,), 0.0, 1.0),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "log/scenario_index": elements.Space(
                np.int32,
                (),
                0,
                len(self._scenario_keys),
            ),
        }

    @property
    def act_space(self):
        return {
            "action": elements.Space(
                np.float32,
                (DREAMER_ACTION_DIM,),
                CONTINUOUS_ACTION_LOW,
                CONTINUOUS_ACTION_HIGH,
            ),
            "reset": elements.Space(bool),
        }

    def step(self, action):
        if bool(action["reset"]) or self._done or self._env is None:
            obs = self._reset_env()
            return self._obs(
                obs,
                reward=0.0,
                is_first=True,
                is_last=False,
                is_terminal=False,
            )

        # Lazy import avoids circular dependency
        import engine as eng

        # ── snapshot *before* the step ────────────────────────────────
        state = self._env._state
        prev_snapshot = eng.step_reward_snapshot(state)

        # ── execute the regular environment step ──────────────────────
        obs_raw, _orig_reward, terminated, truncated, _info = self._env.step(
            np.asarray(action["action"], dtype=np.float32)
        )

        # ── compute budget-aware reward ───────────────────────────────
        reward = eng.calculate_step_reward_budget(state, prev_snapshot)

        # ── budget exhaustion → early termination + punishment ────────
        if (
            state.budget_total > 0
            and state.budget_remaining <= 1e-6
            and not terminated
            and not truncated
        ):
            remaining_frac = max(
                (state.params.sim_hours - state.env.now)
                / max(state.params.sim_hours, 1.0),
                0.0,
            )
            # Only terminate if there is meaningful time left (> 2 %)
            if remaining_frac > 0.02:
                reward = self.BUDGET_EXHAUSTION_PENALTY
                terminated = True

        self._done = bool(terminated or truncated)
        return self._obs(
            obs_raw,
            reward=reward,
            is_first=False,
            is_last=self._done,
            is_terminal=bool(terminated),
        )

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _obs(
        self,
        obs: np.ndarray,
        *,
        reward: float,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
    ):
        return {
            "vector": np.asarray(obs, dtype=np.float32),
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            "log/scenario_index": np.int32(self._scenario_index),
        }

    def _reset_env(self) -> np.ndarray:
        if self._env is not None:
            self._env.close()

        self._episode_seed += 1

        if len(self._scenario_keys) == 1:
            self._scenario_key = self._scenario_keys[0]
            self._scenario_index = 0
        else:
            # ── Curriculum-aware scenario sampling (mirrors standard env) ──
            episodes_completed = max(self._episode_seed - self._base_seed - 1, 0)
            progress = min(episodes_completed / 800.0, 1.0)

            EASY_WEIGHTS = {
                "baseline": 0.35,
                "donor_decrease": 0.25,
                "demand_surge": 0.20,
                "transport_disruption": 0.12,
                "combined_crisis": 0.08,
            }
            MATURE_WEIGHTS = {
                "baseline": 0.18,
                "donor_decrease": 0.22,
                "demand_surge": 0.22,
                "transport_disruption": 0.18,
                "combined_crisis": 0.20,
            }
            blended = {
                k: (
                    EASY_WEIGHTS.get(k, 0.0) * (1.0 - progress)
                    + MATURE_WEIGHTS.get(k, 1.0 / len(self._scenario_keys)) * progress
                )
                for k in self._scenario_keys
            }
            weights = [
                blended.get(k, 1.0 / len(self._scenario_keys))
                for k in self._scenario_keys
            ]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            rng = np.random.default_rng(self._episode_seed)
            self._scenario_index = int(rng.choice(len(self._scenario_keys), p=weights))
            self._scenario_key = self._scenario_keys[self._scenario_index]

        params = copy.deepcopy(self._scenario_templates[self._scenario_key])

        # ── Override budget when a fixed budget is requested ──────────
        if self._fixed_budget is not None:
            params.episode_budget = float(self._fixed_budget)

        G, north, south, east, west = self._sim_context
        self._env = BloodSupplyEnv(
            params=params,
            G=G,
            north=north,
            south=south,
            east=east,
            west=west,
            episode_hours=float(params.sim_hours),
            step_hours=float(self._step_hours),
            seed=int(self._episode_seed),
            action_mode="continuous",
        )
        self._done = False
        obs, _info = self._env.reset(seed=int(self._episode_seed))
        return np.asarray(obs, dtype=np.float32)
