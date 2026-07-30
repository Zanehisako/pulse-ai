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


class OfficialDreamerBloodEnv(embodied.Env):
    def __init__(
        self,
        scenario_keys: list[str] | tuple[str, ...],
        sim_context,
        *,
        step_hours: float = 6.0,
        seed: int = 0,
    ):
        self._scenario_keys = list(scenario_keys)
        if not self._scenario_keys:
            raise ValueError(
                "OfficialDreamerBloodEnv requires at least one scenario key."
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

        obs, reward, terminated, truncated, _info = self._env.step(
            np.asarray(action["action"], dtype=np.float32)
        )
        self._done = bool(terminated or truncated)
        return self._obs(
            obs,
            reward=reward,
            is_first=False,
            is_last=self._done,
            is_terminal=bool(terminated),
        )

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

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
            # ── Curriculum-aware scenario sampling ──────────────────────
            # Early episodes favor easier scenarios to build basic
            # competence; the mix shifts toward harder scenarios over time.
            episodes_completed = max(self._episode_seed - self._base_seed - 1, 0)
            # progress ramps from 0 → 1 over ~800 episodes
            progress = min(episodes_completed / 800.0, 1.0)

            # Weights interpolate from easy-heavy start to balanced target
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
