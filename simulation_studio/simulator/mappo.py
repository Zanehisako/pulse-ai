from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


def _init_linear(layer: nn.Linear, gain: float = np.sqrt(2.0)) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class RunningNormalizer:
    def __init__(self, shape: int, clip: float = 5.0, eps: float = 1e-6):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = eps
        self.clip = clip
        self.eps = eps

    def update(self, batch: torch.Tensor | np.ndarray) -> None:
        values = torch.as_tensor(batch, dtype=torch.float32)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape[0] == 0:
            return

        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        batch_count = float(values.shape[0])

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = torch.clamp(m2 / total_count, min=self.eps)
        self.count = total_count

    def normalize(self, values: torch.Tensor | np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        normalized = (tensor - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(normalized, -self.clip, self.clip)


# =========================
# Actor (per agent)
# =========================
class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, action_dim)

        _init_linear(self.fc1)
        _init_linear(self.fc2)
        _init_linear(self.fc_out, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        hidden = torch.tanh(self.fc1(x))
        hidden = torch.tanh(self.fc2(hidden))
        return self.fc_out(hidden)


# =========================
# Central Critic
# =========================
class Critic(nn.Module):
    def __init__(self, global_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(global_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, 1)

        _init_linear(self.fc1)
        _init_linear(self.fc2)
        _init_linear(self.fc_out, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        hidden = torch.tanh(self.fc1(x))
        hidden = torch.tanh(self.fc2(hidden))
        return self.fc_out(hidden)


# =========================
# MAPPO
# =========================
class MAPPO:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        agent_names: list[str] | tuple[str, ...] | None = None,
        lr: float = 5e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        max_grad_norm: float = 0.5,
        obs_clip: float = 5.0,
        action_masks: dict[str, list[float] | np.ndarray] | None = None,
        batch_size: int = 256,
        elite_imitation_coef: float = 0.08,
        elite_quantile: float = 0.25,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.lr = lr
        if agent_names is None:
            default_names = ["manager", "supply", "hospital", "logistics"]
            if n_agents == 3:
                default_names = ["supply", "hospital", "logistics"]
            self.agent_names = default_names[:n_agents]
        else:
            self.agent_names = list(agent_names)
        if len(self.agent_names) != n_agents:
            raise ValueError(
                f"MAPPO expected {n_agents} agent names, got {len(self.agent_names)}."
            )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm
        self.obs_clip = obs_clip
        self.entropy_coef = 0.01
        self.batch_size = batch_size
        self.elite_imitation_coef = elite_imitation_coef
        self.elite_quantile = elite_quantile
        self.action_masks = self._build_action_masks(action_masks, action_dim)

        self.actors = {name: Actor(obs_dim, action_dim) for name in self.agent_names}
        self.actor_opts = {
            name: optim.Adam(self.actors[name].parameters(), lr=lr, eps=1e-5)
            for name in self.agent_names
        }

        self.critic = Critic(obs_dim * n_agents)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)

        self.obs_normalizers = {
            name: RunningNormalizer(obs_dim, clip=obs_clip) for name in self.agent_names
        }

    def _build_action_masks(
        self,
        action_masks: dict[str, list[float] | np.ndarray] | None,
        action_dim: int,
    ) -> dict[str, torch.Tensor]:
        masks: dict[str, torch.Tensor] = {}
        for name in self.agent_names:
            raw_mask = None if action_masks is None else action_masks.get(name)
            if raw_mask is None:
                mask = torch.ones(action_dim, dtype=torch.float32)
            else:
                mask = torch.as_tensor(raw_mask, dtype=torch.float32)
            if mask.numel() != action_dim:
                raise ValueError(
                    f"Action mask for {name} has size {mask.numel()}, expected {action_dim}."
                )
            if mask.max().item() <= 0:
                raise ValueError(f"Action mask for {name} must allow at least one action.")
            masks[name] = mask
        return masks

    def _distribution(
        self,
        name: str,
        obs: torch.Tensor,
        action_mask: torch.Tensor | np.ndarray | None = None,
        action_bias: torch.Tensor | np.ndarray | None = None,
    ) -> torch.distributions.Categorical:
        logits = self.actors[name](obs)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        mask = self.action_masks[name].to(logits.device).unsqueeze(0)
        if action_mask is not None:
            step_mask = torch.as_tensor(action_mask, dtype=torch.float32, device=logits.device)
            if step_mask.ndim == 1:
                step_mask = step_mask.unsqueeze(0)
            mask = mask * step_mask
        else:
            mask = mask.expand(logits.shape[0], -1)

        if action_bias is not None:
            bias = torch.as_tensor(action_bias, dtype=torch.float32, device=logits.device)
            if bias.ndim == 1:
                bias = bias.unsqueeze(0)
            logits = logits + bias

        empty_rows = mask.sum(dim=-1) <= 0
        if empty_rows.any():
            mask = mask.clone()
            mask[empty_rows, 0] = 1.0

        masked_logits = logits.masked_fill(mask <= 0, -1e9)
        return torch.distributions.Categorical(logits=masked_logits)

    def _normalize_obs(
        self,
        name: str,
        obs: torch.Tensor | np.ndarray,
        *,
        update_stats: bool,
    ) -> torch.Tensor:
        if update_stats:
            self.obs_normalizers[name].update(obs)
        return self.obs_normalizers[name].normalize(obs)

    def build_global_state(self, obs_dict: dict[str, np.ndarray | torch.Tensor]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(obs_dict[name], dtype=np.float32) for name in self.agent_names]
        ).astype(np.float32)

    def preprocess_obs(
        self,
        obs_dict: dict[str, np.ndarray | torch.Tensor],
        *,
        update_stats: bool = False,
    ) -> dict[str, np.ndarray]:
        processed: dict[str, np.ndarray] = {}
        for name, obs in obs_dict.items():
            if name not in self.agent_names:
                raise KeyError(f"Unknown MAPPO agent '{name}'.")
            normalized = self._normalize_obs(name, obs_dict[name], update_stats=update_stats)
            processed[name] = normalized.detach().cpu().numpy().astype(np.float32)
        return processed

    # ----------------------
    def act(
        self,
        obs_dict: dict[str, np.ndarray | torch.Tensor],
        *,
        update_stats: bool = True,
        action_masks: dict[str, np.ndarray | torch.Tensor] | None = None,
        action_biases: dict[str, np.ndarray | torch.Tensor] | None = None,
        deterministic: bool = False,
    ):
        actions: dict[str, int] = {}
        log_probs: dict[str, float] = {}
        processed_obs = self.preprocess_obs(obs_dict, update_stats=update_stats)

        for name in obs_dict:
            obs = torch.as_tensor(processed_obs[name], dtype=torch.float32)
            step_mask = None if action_masks is None else action_masks.get(name)
            step_bias = None if action_biases is None else action_biases.get(name)
            dist = self._distribution(name, obs, action_mask=step_mask, action_bias=step_bias)
            action = (
                torch.argmax(dist.logits, dim=-1)
                if deterministic
                else dist.sample()
            )

            actions[name] = int(action.item())
            log_probs[name] = float(dist.log_prob(action).item())

        return actions, log_probs, processed_obs

    def act_single(
        self,
        name: str,
        obs: np.ndarray | torch.Tensor,
        *,
        update_stats: bool = True,
        action_mask: np.ndarray | torch.Tensor | None = None,
        action_bias: np.ndarray | torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[int, float, np.ndarray]:
        if name not in self.agent_names:
            raise KeyError(f"Unknown MAPPO agent '{name}'.")

        normalized = self._normalize_obs(name, obs, update_stats=update_stats)
        processed_obs = normalized.detach().cpu().numpy().astype(np.float32)
        obs_tensor = torch.as_tensor(processed_obs, dtype=torch.float32)
        dist = self._distribution(
            name,
            obs_tensor,
            action_mask=action_mask,
            action_bias=action_bias,
        )
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            processed_obs,
        )

    # ----------------------
    def compute_gae(
        self,
        rewards: torch.Tensor | np.ndarray,
        values: torch.Tensor | np.ndarray,
        dones: torch.Tensor | np.ndarray,
        last_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
        values_t = torch.as_tensor(values, dtype=torch.float32)
        dones_t = torch.as_tensor(dones, dtype=torch.float32)

        advantages = torch.zeros_like(rewards_t)
        gae = torch.tensor(0.0, dtype=torch.float32)
        next_value = torch.tensor(float(last_value), dtype=torch.float32)

        for t in reversed(range(len(rewards_t))):
            non_terminal = 1.0 - dones_t[t]
            delta = rewards_t[t] + self.gamma * next_value * non_terminal - values_t[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae
            next_value = values_t[t]

        returns = advantages + values_t
        return advantages, returns

    # ----------------------
    def update(self, traj, clip_eps: float | None = None, epochs: int = 5):
        if clip_eps is None:
            clip_eps = self.clip_eps

        rewards = torch.as_tensor(traj["rewards"], dtype=torch.float32)
        if rewards.numel() == 0:
            return {
                "critic_loss": 0.0,
                "policy_loss": 0.0,
                "entropy": 0.0,
                "elite_loss": 0.0,
                "adv_mean": 0.0,
                "adv_std": 0.0,
            }

        if traj.get("normalized_obs"):
            processed_obs = {
                name: np.asarray(traj["obs"][name], dtype=np.float32)
                for name in self.agent_names
            }
            global_states_np = np.asarray(traj["global_states"], dtype=np.float32)
        else:
            processed_obs = {}
            for name in self.agent_names:
                obs_batch = np.asarray(traj["obs"][name], dtype=np.float32)
                processed = self._normalize_obs(name, obs_batch, update_stats=False)
                processed_obs[name] = processed.detach().cpu().numpy().astype(np.float32)

            global_states_np = np.stack(
                [
                    self.build_global_state(
                        {name: processed_obs[name][idx] for name in self.agent_names}
                    )
                    for idx in range(len(rewards))
                ],
                axis=0,
            ).astype(np.float32)

        global_states = torch.as_tensor(global_states_np, dtype=torch.float32)
        dones = torch.as_tensor(
            traj.get("dones", [0.0] * (len(rewards) - 1) + [1.0]), dtype=torch.float32
        )

        with torch.no_grad():
            old_values = self.critic(global_states).squeeze(-1)
            last_value = 0.0
            if len(dones) > 0 and dones[-1].item() < 0.5 and "last_global_state" in traj:
                last_state = torch.as_tensor(traj["last_global_state"], dtype=torch.float32)
                last_value = float(self.critic(last_state).squeeze().item())

            raw_advantages, returns = self.compute_gae(
                rewards, old_values, dones, last_value=last_value
            )
            adv_std = raw_advantages.std(unbiased=False)
            advantages = (raw_advantages - raw_advantages.mean()) / (adv_std + 1e-8)

        metrics = {
            "critic_loss": [],
            "policy_loss": [],
            "entropy": [],
            "elite_loss": [],
            "adv_mean": float(raw_advantages.mean().item()),
            "adv_std": float(raw_advantages.std(unbiased=False).item()),
        }
        action_masks = traj.get("action_masks", {})
        action_biases = traj.get("action_biases", {})
        elite_threshold = None
        if len(raw_advantages) >= 4 and self.elite_imitation_coef > 0:
            elite_quantile = float(np.clip(self.elite_quantile, 0.0, 1.0))
            elite_threshold = torch.quantile(
                raw_advantages.detach(), max(0.0, 1.0 - elite_quantile)
            )

        batch_size = max(1, min(self.batch_size, len(rewards)))

        for _ in range(epochs):
            permutation = torch.randperm(len(rewards))
            for start in range(0, len(rewards), batch_size):
                batch_idx = permutation[start : start + batch_size]

                global_states_b = global_states[batch_idx]
                old_values_b = old_values[batch_idx]
                returns_b = returns[batch_idx]
                advantages_b = advantages[batch_idx]
                raw_advantages_b = raw_advantages[batch_idx]

                values = self.critic(global_states_b).squeeze(-1)
                value_clipped = old_values_b + (values - old_values_b).clamp(
                    -clip_eps, clip_eps
                )
                critic_loss = 0.5 * torch.max(
                    (values - returns_b).pow(2),
                    (value_clipped - returns_b).pow(2),
                ).mean()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_opt.step()
                metrics["critic_loss"].append(float(critic_loss.item()))

                for name in self.agent_names:
                    obs = torch.as_tensor(processed_obs[name], dtype=torch.float32)[batch_idx]
                    actions = torch.as_tensor(traj["actions"][name], dtype=torch.long)[
                        batch_idx
                    ]
                    old_log_probs = torch.as_tensor(
                        traj["log_probs"][name], dtype=torch.float32
                    )[batch_idx]
                    batch_masks = None
                    if name in action_masks:
                        batch_masks = torch.as_tensor(
                            np.asarray(action_masks[name], dtype=np.float32),
                            dtype=torch.float32,
                        )[batch_idx]
                    batch_biases = None
                    if name in action_biases:
                        batch_biases = torch.as_tensor(
                            np.asarray(action_biases[name], dtype=np.float32),
                            dtype=torch.float32,
                        )[batch_idx]

                    dist = self._distribution(
                        name,
                        obs,
                        action_mask=batch_masks,
                        action_bias=batch_biases,
                    )
                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages_b
                    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_b

                    policy_loss = -torch.min(surr1, surr2).mean()
                    elite_loss = torch.tensor(0.0, dtype=torch.float32)
                    if elite_threshold is not None:
                        elite_mask = raw_advantages_b >= elite_threshold
                        if elite_mask.any():
                            elite_weights = torch.clamp(
                                raw_advantages_b[elite_mask], min=0.0
                            )
                            elite_weights = elite_weights / (
                                elite_weights.mean().detach() + 1e-6
                            )
                            elite_loss = -(
                                elite_weights.detach() * new_log_probs[elite_mask]
                            ).mean()

                    loss = (
                        policy_loss
                        + self.elite_imitation_coef * elite_loss
                        - self.entropy_coef * entropy
                    )

                    self.actor_opts[name].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actors[name].parameters(), self.max_grad_norm
                    )
                    self.actor_opts[name].step()

                    metrics["policy_loss"].append(float(policy_loss.item()))
                    metrics["entropy"].append(float(entropy.item()))
                    metrics["elite_loss"].append(float(elite_loss.item()))

        return {
            "critic_loss": float(np.mean(metrics["critic_loss"])),
            "policy_loss": float(np.mean(metrics["policy_loss"])),
            "entropy": float(np.mean(metrics["entropy"])),
            "elite_loss": float(np.mean(metrics["elite_loss"])) if metrics["elite_loss"] else 0.0,
            "adv_mean": metrics["adv_mean"],
            "adv_std": metrics["adv_std"],
        }

    def save(self, path: str) -> Path:
        target = Path(path)
        if target.suffix == "":
            target = target.with_suffix(".pt")

        payload = {
            "config": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "n_agents": self.n_agents,
                "agent_names": list(self.agent_names),
                "lr": self.lr,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_eps": self.clip_eps,
                "max_grad_norm": self.max_grad_norm,
                "obs_clip": self.obs_clip,
                "action_masks": {
                    name: mask.tolist() for name, mask in self.action_masks.items()
                },
                "batch_size": self.batch_size,
                "elite_imitation_coef": self.elite_imitation_coef,
                "elite_quantile": self.elite_quantile,
            },
            "entropy_coef": self.entropy_coef,
            "actors": {
                name: actor.state_dict() for name, actor in self.actors.items()
            },
            "critic": self.critic.state_dict(),
            "actor_opts": {
                name: optimizer.state_dict()
                for name, optimizer in self.actor_opts.items()
            },
            "critic_opt": self.critic_opt.state_dict(),
            "obs_normalizers": {
                name: {
                    "mean": normalizer.mean,
                    "var": normalizer.var,
                    "count": normalizer.count,
                }
                for name, normalizer in self.obs_normalizers.items()
            },
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> "MAPPO":
        payload = torch.load(path, map_location=map_location)
        agent = cls(**payload["config"])
        agent.entropy_coef = payload.get("entropy_coef", agent.entropy_coef)

        for name, state_dict in payload["actors"].items():
            agent.actors[name].load_state_dict(state_dict)
        agent.critic.load_state_dict(payload["critic"])

        for name, state_dict in payload.get("actor_opts", {}).items():
            agent.actor_opts[name].load_state_dict(state_dict)
        if "critic_opt" in payload:
            agent.critic_opt.load_state_dict(payload["critic_opt"])

        for name, stats in payload.get("obs_normalizers", {}).items():
            normalizer = agent.obs_normalizers[name]
            normalizer.mean = stats["mean"].detach().clone().to(torch.float32)
            normalizer.var = stats["var"].detach().clone().to(torch.float32)
            normalizer.count = float(stats["count"])

        return agent
