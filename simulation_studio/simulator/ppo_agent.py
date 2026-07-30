"""
ppo_agent.py  –  Self-contained PPO (Proximal Policy Optimisation) agent.

Architecture
------------
• Works entirely in **NumPy** – no PyTorch / TensorFlow required.
• Implements the canonical PPO-Clip algorithm (Schulman et al., 2017) with
  Generalised Advantage Estimation (GAE-λ).
• Exposes an interface that is deliberately close to Stable Baselines 3's
  ``PPO`` class so swapping is trivial once SB3 is available:

      agent = PPO("MlpPolicy", env, verbose=1)
      agent.learn(total_timesteps=50_000)
      obs, _ = env.reset()
      action, _ = agent.predict(obs)

Internal policy/value network
------------------------------
Both the policy (actor) and value function (critic) are two-hidden-layer MLPs.
Weights live in plain NumPy arrays; SGD-style updates use a vanilla gradient
descent step derived from the log-probabilities and advantage estimates.

Because NumPy has no auto-diff, we compute gradients analytically for the
softmax + cross-entropy actor head and an MSE critic head, then back-prop
through the shared two-layer MLP trunk with tanh activations.

SB3 compatibility shim
-----------------------
If Stable Baselines 3 **is** available at import time, ``PPO`` is re-exported
directly from ``stable_baselines3``, so the rest of the codebase can always do:

    from ppo_agent import PPO
"""

from __future__ import annotations

import math
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Try to delegate to SB3 if it is installed
# ---------------------------------------------------------------------------
try:
    from stable_baselines3 import PPO  # type: ignore
    from stable_baselines3.common.env_util import make_vec_env  # type: ignore

    _SB3_AVAILABLE = True
    print("[ppo_agent] stable-baselines3 found – using native SB3 PPO.")

except ModuleNotFoundError:
    print("Stable baseline 3 is not availble")
    _SB3_AVAILABLE = False

    # -----------------------------------------------------------------------
    # ── NumPy-only PPO implementation ──────────────────────────────────────
    # -----------------------------------------------------------------------

    RNG = np.random.default_rng(0)  # module-level RNG, re-seeded in __init__

    # ── Activation helpers ─────────────────────────────────────────────────

    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _dtanh(y: np.ndarray) -> np.ndarray:
        """Derivative of tanh given its *output* y."""
        return 1.0 - y**2

    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    # ── Two-layer MLP with separate actor/critic heads ─────────────────────

    class _MLP:
        """
        Shared-trunk MLP:
            input (obs_dim)
            → hidden1 (hidden_dim, tanh)
            → hidden2 (hidden_dim, tanh)
            → [policy logits (n_actions) | value scalar]
        """

        def __init__(
            self,
            obs_dim: int,
            n_actions: int,
            hidden_dim: int = 64,
            lr: float = 3e-4,
            rng: np.random.Generator = None,
        ) -> None:
            rng = rng or RNG
            scale1 = math.sqrt(2.0 / obs_dim)
            scale2 = math.sqrt(2.0 / hidden_dim)

            # Trunk weights
            self.W1 = (
                rng.standard_normal((obs_dim, hidden_dim)).astype(np.float32) * scale1
            )
            self.b1 = np.zeros(hidden_dim, dtype=np.float32)
            self.W2 = (
                rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32)
                * scale2
            )
            self.b2 = np.zeros(hidden_dim, dtype=np.float32)

            # Actor head
            self.Wa = (
                rng.standard_normal((hidden_dim, n_actions)).astype(np.float32) * 0.01
            )
            self.ba = np.zeros(n_actions, dtype=np.float32)

            # Critic head
            self.Wv = rng.standard_normal((hidden_dim, 1)).astype(np.float32) * scale2
            self.bv = np.zeros(1, dtype=np.float32)

            self.lr = lr

            # Adam state for every parameter
            self._params = ["W1", "b1", "W2", "b2", "Wa", "ba", "Wv", "bv"]
            self._m: dict[str, np.ndarray] = {
                p: np.zeros_like(getattr(self, p)) for p in self._params
            }
            self._v: dict[str, np.ndarray] = {
                p: np.zeros_like(getattr(self, p)) for p in self._params
            }
            self._t: int = 0

        # ── Forward pass ────────────────────────────────────────────────

        def forward(self, obs: np.ndarray):
            """
            obs: (B, obs_dim)
            Returns (logits (B, n_actions), values (B,), cache for backward)
            """
            z1 = obs @ self.W1 + self.b1  # (B, H)
            h1 = _tanh(z1)
            z2 = h1 @ self.W2 + self.b2  # (B, H)
            h2 = _tanh(z2)

            logits = h2 @ self.Wa + self.ba  # (B, A)
            values = (h2 @ self.Wv + self.bv).squeeze(-1)  # (B,)

            cache = dict(obs=obs, z1=z1, h1=h1, z2=z2, h2=h2)
            return logits, values, cache

        def predict(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Single or batched inference. Returns (probs, values)."""
            logits, values, _ = self.forward(obs)
            probs = _softmax(logits)
            return probs, values

        # ── Backward + Adam update ───────────────────────────────────────

        def update(
            self,
            obs: np.ndarray,  # (B, obs_dim)
            actions: np.ndarray,  # (B,) int
            old_log_p: np.ndarray,  # (B,)
            advantages: np.ndarray,  # (B,)
            returns: np.ndarray,  # (B,)
            clip_eps: float = 0.2,
            ent_coef: float = 0.01,
            vf_coef: float = 0.5,
            max_grad: float = 0.5,
        ) -> dict[str, float]:
            B = obs.shape[0]
            logits, values, cache = self.forward(obs)
            h2 = cache["h2"]
            h1 = cache["h1"]

            probs = _softmax(logits)  # (B, A)
            log_probs = np.log(probs + 1e-8)

            # Gather log-probs for chosen actions
            idx = np.arange(B)
            lp = log_probs[idx, actions]  # (B,)
            ratio = np.exp(lp - old_log_p)  # (B,)

            # PPO-Clip actor loss
            surr1 = ratio * advantages
            surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            actor_loss = -np.mean(np.minimum(surr1, surr2))

            # Entropy bonus
            entropy = -np.sum(probs * log_probs, axis=-1).mean()

            # Critic (value) loss
            vf_loss = 0.5 * np.mean((values - returns) ** 2)

            total_loss = actor_loss - ent_coef * entropy + vf_coef * vf_loss

            # ── Manual back-prop ─────────────────────────────────────────
            # 1. Gradient of total loss w.r.t. logits
            #    Actor gradient via REINFORCE-style PG with clipped ratio
            clip_mask = ((ratio > 1 + clip_eps) & (advantages > 0)) | (
                (ratio < 1 - clip_eps) & (advantages < 0)
            )
            pg_weight = np.where(clip_mask, 0.0, -advantages)  # (B,)

            # dL/d_logits  (softmax cross-entropy gradient)
            d_logits = probs.copy()  # (B, A)
            d_logits[idx, actions] -= 1.0  # one-hot
            d_logits *= pg_weight[:, None] / B
            # subtract entropy gradient
            d_logits -= ent_coef * (-log_probs - 1.0) * probs / B

            # 2. Gradient of value loss w.r.t. values
            d_values = vf_coef * (values - returns) / B  # (B,)

            # 3. Gradients of actor head
            dWa = h2.T @ d_logits  # (H, A)
            dba = d_logits.sum(axis=0)

            # 4. Gradients of critic head
            dWv = (h2 * d_values[:, None]).T @ np.ones((B, 1))  # (H, 1)
            dWv = h2.T @ d_values[:, None]
            dbv = d_values.sum(keepdims=True)

            # 5. Back through h2
            d_h2 = d_logits @ self.Wa.T + d_values[:, None] * self.Wv.T  # (B, H)
            d_z2 = d_h2 * _dtanh(cache["h2"])

            dW2 = h1.T @ d_z2
            db2 = d_z2.sum(axis=0)

            # 6. Back through h1
            d_h1 = d_z2 @ self.W2.T
            d_z1 = d_h1 * _dtanh(cache["h1"])

            dW1 = obs.T @ d_z1
            db1 = d_z1.sum(axis=0)

            grads = dict(W1=dW1, b1=db1, W2=dW2, b2=db2, Wa=dWa, ba=dba, Wv=dWv, bv=dbv)

            # 7. Gradient clipping
            all_grads = np.concatenate([g.ravel() for g in grads.values()])
            grad_norm = np.linalg.norm(all_grads)
            if grad_norm > max_grad:
                scale = max_grad / (grad_norm + 1e-8)
                grads = {k: v * scale for k, v in grads.items()}

            # 8. Adam step
            self._t += 1
            beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
            for p in self._params:
                g = grads[p]
                self._m[p] = beta1 * self._m[p] + (1 - beta1) * g
                self._v[p] = beta2 * self._v[p] + (1 - beta2) * g**2
                m_hat = self._m[p] / (1 - beta1**self._t)
                v_hat = self._v[p] / (1 - beta2**self._t)
                setattr(
                    self,
                    p,
                    getattr(self, p) - self.lr * m_hat / (np.sqrt(v_hat) + eps_adam),
                )

            return {
                "actor_loss": float(actor_loss),
                "vf_loss": float(vf_loss),
                "entropy": float(entropy),
                "total_loss": float(total_loss),
                "grad_norm": float(grad_norm),
            }

        # ── Persistence ─────────────────────────────────────────────────

        def save(self, path: str) -> None:
            data = {p: getattr(self, p) for p in self._params}
            data["_t"] = self._t
            with open(path, "wb") as f:
                pickle.dump(data, f)

        def load(self, path: str) -> None:
            with open(path, "rb") as f:
                data = pickle.load(f)
            for p in self._params:
                setattr(self, p, data[p])
            self._t = data.get("_t", 0)

    # ── Rollout buffer ─────────────────────────────────────────────────────

    class _RolloutBuffer:
        """Stores one rollout worth of transitions then yields mini-batches."""

        def __init__(
            self, capacity: int, obs_dim: int, gamma: float, lam: float
        ) -> None:
            self.capacity = capacity
            self.obs_dim = obs_dim
            self.gamma = gamma
            self.lam = lam
            self._ptr = 0
            self._full = False

            self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
            self.actions = np.zeros(capacity, dtype=np.int32)
            self.rewards = np.zeros(capacity, dtype=np.float32)
            self.values = np.zeros(capacity, dtype=np.float32)
            self.log_probs = np.zeros(capacity, dtype=np.float32)
            self.dones = np.zeros(capacity, dtype=np.float32)
            self.advantages = np.zeros(capacity, dtype=np.float32)
            self.returns = np.zeros(capacity, dtype=np.float32)

        def add(
            self,
            obs: np.ndarray,
            action: int,
            reward: float,
            value: float,
            log_prob: float,
            done: bool,
        ) -> None:
            i = self._ptr % self.capacity
            self.obs[i] = obs
            self.actions[i] = action
            self.rewards[i] = reward
            self.values[i] = value
            self.log_probs[i] = log_prob
            self.dones[i] = float(done)
            self._ptr += 1
            if self._ptr >= self.capacity:
                self._full = True

        @property
        def full(self) -> bool:
            return self._full

        def compute_returns_and_advantages(self, last_value: float) -> None:
            """GAE-λ advantage estimation."""
            gae = 0.0
            n = min(self._ptr, self.capacity)
            for step in reversed(range(n)):
                next_value = last_value if step == n - 1 else self.values[step + 1]
                next_done = self.dones[step]
                delta = (
                    self.rewards[step]
                    + self.gamma * next_value * (1.0 - next_done)
                    - self.values[step]
                )
                gae = delta + self.gamma * self.lam * (1.0 - next_done) * gae
                self.advantages[step] = gae
                self.returns[step] = gae + self.values[step]

        def get_batches(self, batch_size: int):
            """Yield shuffled mini-batches of (obs, actions, old_lp, adv, ret)."""
            n = min(self._ptr, self.capacity)
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                sl = idx[start : start + batch_size]
                yield (
                    self.obs[sl],
                    self.actions[sl],
                    self.log_probs[sl],
                    self.advantages[sl],
                    self.returns[sl],
                )

        def reset(self) -> None:
            self._ptr = 0
            self._full = False

    # ─────────────────────────────────────────────────────────────────────────
    # Public PPO class (NumPy fallback)
    # ─────────────────────────────────────────────────────────────────────────

    class PPO:
        """
        PPO-Clip with GAE-λ, implemented in pure NumPy.

        Parameters mirror a subset of Stable Baselines 3's PPO for easy swap.

        Parameters
        ----------
        policy          : ignored (always 'MlpPolicy'), kept for SB3 compat.
        env             : a Gymnasium-compatible environment.
        learning_rate   : Adam learning rate.
        n_steps         : rollout length before each update.
        batch_size      : mini-batch size for the PPO update epochs.
        n_epochs        : number of epochs over the rollout per update.
        gamma           : discount factor.
        gae_lambda      : GAE λ.
        clip_range      : ε for the PPO-Clip surrogate.
        ent_coef        : entropy regularisation coefficient.
        vf_coef         : value-function loss coefficient.
        max_grad_norm   : gradient clipping threshold.
        hidden_dim      : width of each MLP hidden layer.
        verbose         : 0 = silent, 1 = progress, 2 = all losses.
        seed            : RNG seed.
        """

        def __init__(
            self,
            policy: str = "MlpPolicy",
            env=None,
            learning_rate: float = 3e-4,
            n_steps: int = 512,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            ent_coef: float = 0.01,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            hidden_dim: int = 64,
            verbose: int = 1,
            seed: int = 42,
        ) -> None:
            global RNG
            RNG = np.random.default_rng(seed)
            np.random.seed(seed)

            self.env = env
            self.lr = learning_rate
            self.n_steps = n_steps
            self.batch_size = batch_size
            self.n_epochs = n_epochs
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_range = clip_range
            self.ent_coef = ent_coef
            self.vf_coef = vf_coef
            self.max_grad_norm = max_grad_norm
            self.verbose = verbose
            self.seed = seed
            self._hidden_dim = hidden_dim

            obs_dim = (
                env.observation_space.shape[0]
                if hasattr(env, "observation_space")
                else 12
            )
            n_actions = env.action_space.n if hasattr(env, "action_space") else 6

            self.net = _MLP(
                obs_dim, n_actions, hidden_dim=hidden_dim, lr=learning_rate, rng=RNG
            )
            self.buffer = _RolloutBuffer(n_steps, obs_dim, gamma, gae_lambda)

            # Logging
            self.num_timesteps: int = 0
            self.ep_reward_buf: deque = deque(maxlen=100)
            self.ep_len_buf: deque = deque(maxlen=100)
            self._start_time: float = time.time()

        # ── Prediction ─────────────────────────────────────────────────

        def predict(
            self,
            observation: np.ndarray,
            deterministic: bool = False,
        ) -> Tuple[int, None]:
            """
            Given an observation, return (action, state).
            ``state`` is always None (no recurrent network).
            """
            obs = np.atleast_2d(observation).astype(np.float32)
            probs, _ = self.net.predict(obs)
            probs = probs[0]
            if deterministic:
                action = int(np.argmax(probs))
            else:
                action = int(np.random.choice(len(probs), p=probs))
            return action, None

        # ── Training ───────────────────────────────────────────────────

        def learn(
            self,
            total_timesteps: int,
            callback=None,
            log_interval: int = 1,
            reset_num_timesteps: bool = True,
        ) -> "PPO":
            """Train for ``total_timesteps`` environment steps."""
            if reset_num_timesteps:
                self.num_timesteps = 0

            def _reset_env(seed: Optional[int] = None) -> np.ndarray:
                result = self.env.reset(seed=seed) if seed is not None else self.env.reset()
                if isinstance(result, tuple) and len(result) == 2:
                    obs, _info = result
                else:
                    obs = result
                return np.asarray(obs, dtype=np.float32)

            env = self.env
            obs = _reset_env(seed=self.seed)
            ep_reward = 0.0
            ep_len = 0
            update_n = 0

            while self.num_timesteps < total_timesteps:
                # ── Collect rollout ─────────────────────────────────────
                self.buffer.reset()
                for _ in range(self.n_steps):
                    probs, values = self.net.predict(obs[None])
                    probs = probs[0]
                    value = float(values[0])

                    action = int(np.random.choice(len(probs), p=probs))
                    log_prob = float(np.log(probs[action] + 1e-8))

                    result = env.step(action)
                    if len(result) == 5:
                        next_obs, reward, terminated, truncated, info = result
                        done = terminated or truncated
                    else:
                        next_obs, reward, done, info = result

                    self.buffer.add(obs, action, float(reward), value, log_prob, done)
                    ep_reward += reward
                    ep_len += 1
                    self.num_timesteps += 1

                    if done:
                        self.ep_reward_buf.append(ep_reward)
                        self.ep_len_buf.append(ep_len)
                        ep_reward = 0.0
                        ep_len = 0
                        obs = _reset_env()
                    else:
                        obs = np.array(next_obs, dtype=np.float32)

                    if self.num_timesteps >= total_timesteps:
                        break

                # Bootstrap value for last step
                _, last_values = self.net.predict(obs[None])
                last_value = float(last_values[0])
                self.buffer.compute_returns_and_advantages(last_value)

                # Normalise advantages
                n = min(self.buffer._ptr, self.n_steps)
                adv_mean = self.buffer.advantages[:n].mean()
                adv_std = self.buffer.advantages[:n].std() + 1e-8
                self.buffer.advantages[:n] = (
                    self.buffer.advantages[:n] - adv_mean
                ) / adv_std

                # ── PPO update epochs ───────────────────────────────────
                update_n += 1
                loss_acc: dict[str, list] = {
                    "actor_loss": [],
                    "vf_loss": [],
                    "entropy": [],
                    "grad_norm": [],
                }
                for _ in range(self.n_epochs):
                    for batch in self.buffer.get_batches(self.batch_size):
                        obs_b, act_b, lp_b, adv_b, ret_b = batch
                        info_d = self.net.update(
                            obs_b,
                            act_b,
                            lp_b,
                            adv_b,
                            ret_b,
                            clip_eps=self.clip_range,
                            ent_coef=self.ent_coef,
                            vf_coef=self.vf_coef,
                            max_grad=self.max_grad_norm,
                        )
                        for k in loss_acc:
                            loss_acc[k].append(info_d[k])

                # ── Logging ─────────────────────────────────────────────
                if self.verbose >= 1 and update_n % log_interval == 0:
                    elapsed = time.time() - self._start_time
                    fps = max(self.num_timesteps / (elapsed + 1e-6), 0)
                    mean_ep = (
                        np.mean(self.ep_reward_buf)
                        if self.ep_reward_buf
                        else float("nan")
                    )
                    mean_len = (
                        np.mean(self.ep_len_buf) if self.ep_len_buf else float("nan")
                    )
                    print(
                        f"  timestep={self.num_timesteps:>8,d} | "
                        f"update={update_n:>4d} | "
                        f"fps={fps:>5.0f} | "
                        f"ep_rew={mean_ep:>7.3f} | "
                        f"ep_len={mean_len:>5.0f} | "
                        f"actor={np.mean(loss_acc['actor_loss']):>7.4f} | "
                        f"vf={np.mean(loss_acc['vf_loss']):>7.4f} | "
                        f"ent={np.mean(loss_acc['entropy']):>5.3f}"
                    )

            return self

        # ── Persistence ─────────────────────────────────────────────────

        def save(self, path: str) -> None:
            """Save weights to a .pkl file (mirrors SB3's .save())."""
            p = Path(path)
            if p.suffix == "":
                p = p.with_suffix(".pkl")
            self.net.save(str(p))
            if self.verbose >= 1:
                print(f"[ppo_agent] Model saved → {p}")

        @classmethod
        def load(
            cls,
            path: str,
            env=None,
            **kwargs,
        ) -> "PPO":
            """Load a previously saved model (mirrors SB3's PPO.load())."""
            agent = cls(env=env, **kwargs)
            p = Path(path)
            if p.suffix == "":
                p = p.with_suffix(".pkl")
            agent.net.load(str(p))
            if agent.verbose >= 1:
                print(f"[ppo_agent] Model loaded ← {p}")
            return agent

        # ── SB3-compat helpers ───────────────────────────────────────────

        def set_env(self, env) -> None:
            self.env = env

        def get_env(self):
            return self.env
