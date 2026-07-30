from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _as_float_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def load_offline_dataset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        required = ("obs", "actions", "rewards", "next_obs", "dones")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Offline dataset is missing keys: {missing}")
        dataset = {
            "obs": np.asarray(data["obs"], dtype=np.float32),
            "actions": np.asarray(data["actions"], dtype=np.float32),
            "rewards": np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1),
            "next_obs": np.asarray(data["next_obs"], dtype=np.float32),
            "dones": np.asarray(data["dones"], dtype=np.float32).reshape(-1, 1),
        }
        if "timeouts" in data:
            dataset["timeouts"] = np.asarray(data["timeouts"], dtype=np.float32).reshape(
                -1, 1
            )
    if dataset["obs"].ndim != 2 or dataset["actions"].ndim != 2:
        raise ValueError("Expected 2D `obs` and `actions` arrays in offline dataset.")
    if dataset["obs"].shape[0] != dataset["actions"].shape[0]:
        raise ValueError("Offline dataset arrays must all share the same first dimension.")
    return dataset


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.trunk = MLP(obs_dim, hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _dist(self, obs: torch.Tensor) -> Normal:
        feat = self.trunk(obs)
        mean = self.mean_head(feat)
        log_std = torch.clamp(self.log_std_head(feat), LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def sample_n(
        self,
        obs: torch.Tensor,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        raw = dist.rsample((num_samples,)).permute(1, 0, 2)
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw.permute(1, 0, 2)).permute(1, 0, 2)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        bounded_actions = torch.clamp(actions, -0.999999, 0.999999)
        raw = 0.5 * (
            torch.log1p(bounded_actions) - torch.log1p(-bounded_actions)
        )
        dist = self._dist(obs)
        log_prob = dist.log_prob(raw) - torch.log(
            1.0 - bounded_actions.pow(2) + 1e-6
        )
        return log_prob.sum(dim=-1, keepdim=True)

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        dist = self._dist(obs)
        raw = dist.mean if deterministic else dist.rsample()
        return torch.tanh(raw)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = MLP(obs_dim + action_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = MLP(obs_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def _sample_batch(
    dataset: dict[str, torch.Tensor],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    count = dataset["obs"].shape[0]
    indices = torch.randint(0, count, (batch_size,), device=dataset["obs"].device)
    return (
        dataset["obs"][indices],
        dataset["actions"][indices],
        dataset["rewards"][indices],
        dataset["next_obs"][indices],
        dataset["dones"][indices],
    )


def _prepare_tensor_dataset(
    dataset_path: str | Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, int]:
    dataset = load_offline_dataset(dataset_path)
    tensors = {key: _as_float_tensor(value, device) for key, value in dataset.items()}
    obs_dim = int(dataset["obs"].shape[1])
    action_dim = int(dataset["actions"].shape[1])
    return tensors, obs_dim, action_dim


def _save_checkpoint(
    path: str | Path,
    *,
    algorithm: str,
    actor: SquashedGaussianActor,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
    stats: dict[str, float],
    config: dict[str, float | int],
) -> Path:
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": algorithm,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "actor_state_dict": actor.state_dict(),
            "stats": stats,
            "config": config,
        },
        resolved,
    )
    return resolved


def train_iql(
    dataset_path: str | Path,
    save_path: str | Path,
    *,
    updates: int = 2_000,
    batch_size: int = 256,
    hidden_dim: int = 256,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    expectile: float = 0.7,
    beta: float = 3.0,
    seed: int = 42,
    device: str | None = None,
    log_every: int = 200,
) -> tuple[Path, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = _resolve_device(device)
    dataset, obs_dim, action_dim = _prepare_tensor_dataset(dataset_path, torch_device)

    actor = SquashedGaussianActor(obs_dim, action_dim, hidden_dim).to(torch_device)
    q1 = QNetwork(obs_dim, action_dim, hidden_dim).to(torch_device)
    q2 = QNetwork(obs_dim, action_dim, hidden_dim).to(torch_device)
    value_net = ValueNetwork(obs_dim, hidden_dim).to(torch_device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    q_optim = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()),
        lr=learning_rate,
    )
    value_optim = torch.optim.Adam(value_net.parameters(), lr=learning_rate)

    stats: dict[str, float] = {}
    for step in range(1, updates + 1):
        obs, actions, rewards, next_obs, dones = _sample_batch(dataset, batch_size)

        with torch.no_grad():
            next_v = value_net(next_obs)
            target_q = rewards + gamma * (1.0 - dones) * next_v

        q1_pred = q1(obs, actions)
        q2_pred = q2(obs, actions)
        q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)
        q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        q_optim.step()

        with torch.no_grad():
            q_min = torch.min(q1(obs, actions), q2(obs, actions))
        value_pred = value_net(obs)
        value_diff = q_min - value_pred
        value_weight = torch.where(
            value_diff > 0.0,
            torch.full_like(value_diff, expectile),
            torch.full_like(value_diff, 1.0 - expectile),
        )
        value_loss = (value_weight * value_diff.pow(2)).mean()
        value_optim.zero_grad(set_to_none=True)
        value_loss.backward()
        value_optim.step()

        with torch.no_grad():
            adv = torch.min(q1(obs, actions), q2(obs, actions)) - value_net(obs)
            exp_adv = torch.exp(beta * adv).clamp(max=100.0)
        actor_loss = -(exp_adv * actor.log_prob(obs, actions)).mean()
        actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optim.step()

        stats = {
            "q_loss": float(q_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "adv_mean": float(adv.detach().mean().cpu().item()),
        }
        if log_every > 0 and (step == 1 or step % log_every == 0 or step == updates):
            print(
                "[train_iql] "
                f"step={step:>5d} q_loss={stats['q_loss']:.4f} "
                f"value_loss={stats['value_loss']:.4f} "
                f"actor_loss={stats['actor_loss']:.4f} adv={stats['adv_mean']:.4f}"
            )

    checkpoint = _save_checkpoint(
        save_path,
        algorithm="iql",
        actor=actor,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        stats=stats,
        config={
            "updates": updates,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "expectile": expectile,
            "beta": beta,
            "seed": seed,
        },
    )
    return checkpoint, stats


def train_cql(
    dataset_path: str | Path,
    save_path: str | Path,
    *,
    updates: int = 2_000,
    batch_size: int = 256,
    hidden_dim: int = 256,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    alpha: float = 0.2,
    cql_alpha: float = 1.0,
    action_samples: int = 10,
    seed: int = 42,
    device: str | None = None,
    log_every: int = 200,
) -> tuple[Path, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = _resolve_device(device)
    dataset, obs_dim, action_dim = _prepare_tensor_dataset(dataset_path, torch_device)

    actor = SquashedGaussianActor(obs_dim, action_dim, hidden_dim).to(torch_device)
    q1 = QNetwork(obs_dim, action_dim, hidden_dim).to(torch_device)
    q2 = QNetwork(obs_dim, action_dim, hidden_dim).to(torch_device)
    q1_target = QNetwork(obs_dim, action_dim, hidden_dim).to(torch_device)
    q2_target = QNetwork(obs_dim, action_dim, hidden_dim).to(torch_device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    actor_optim = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    q_optim = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()),
        lr=learning_rate,
    )

    stats: dict[str, float] = {}
    for step in range(1, updates + 1):
        obs, actions, rewards, next_obs, dones = _sample_batch(dataset, batch_size)

        with torch.no_grad():
            next_actions, next_log_prob = actor.sample(next_obs)
            next_q = torch.min(
                q1_target(next_obs, next_actions),
                q2_target(next_obs, next_actions),
            )
            target_q = rewards + gamma * (1.0 - dones) * (next_q - alpha * next_log_prob)

        q1_pred = q1(obs, actions)
        q2_pred = q2(obs, actions)
        bellman_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        repeated_obs = obs.unsqueeze(1).expand(-1, action_samples, -1).reshape(
            -1, obs_dim
        )
        random_actions = torch.empty(
            batch_size,
            action_samples,
            action_dim,
            device=torch_device,
        ).uniform_(-1.0, 1.0)
        policy_actions, policy_log_prob = actor.sample_n(obs, action_samples)

        q1_rand = q1(repeated_obs, random_actions.reshape(-1, action_dim)).view(
            batch_size, action_samples
        )
        q2_rand = q2(repeated_obs, random_actions.reshape(-1, action_dim)).view(
            batch_size, action_samples
        )
        q1_policy = q1(repeated_obs, policy_actions.reshape(-1, action_dim)).view(
            batch_size, action_samples
        ) - policy_log_prob
        q2_policy = q2(repeated_obs, policy_actions.reshape(-1, action_dim)).view(
            batch_size, action_samples
        ) - policy_log_prob

        cql1_loss = (
            torch.logsumexp(torch.cat([q1_rand, q1_policy], dim=1), dim=1).mean()
            - q1_pred.mean()
        ) * cql_alpha
        cql2_loss = (
            torch.logsumexp(torch.cat([q2_rand, q2_policy], dim=1), dim=1).mean()
            - q2_pred.mean()
        ) * cql_alpha
        q_loss = bellman_loss + cql1_loss + cql2_loss
        q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        q_optim.step()

        actor_actions, actor_log_prob = actor.sample(obs)
        actor_loss = (
            alpha * actor_log_prob
            - torch.min(q1(obs, actor_actions), q2(obs, actor_actions))
        ).mean()
        actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optim.step()

        _soft_update(q1_target, q1, tau)
        _soft_update(q2_target, q2, tau)

        stats = {
            "bellman_loss": float(bellman_loss.detach().cpu().item()),
            "cql1_loss": float(cql1_loss.detach().cpu().item()),
            "cql2_loss": float(cql2_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
        }
        if log_every > 0 and (step == 1 or step % log_every == 0 or step == updates):
            print(
                "[train_cql] "
                f"step={step:>5d} bellman={stats['bellman_loss']:.4f} "
                f"cql1={stats['cql1_loss']:.4f} cql2={stats['cql2_loss']:.4f} "
                f"actor={stats['actor_loss']:.4f}"
            )

    checkpoint = _save_checkpoint(
        save_path,
        algorithm="cql",
        actor=actor,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        stats=stats,
        config={
            "updates": updates,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "tau": tau,
            "alpha": alpha,
            "cql_alpha": cql_alpha,
            "action_samples": action_samples,
            "seed": seed,
        },
    )
    return checkpoint, stats


class OfflinePolicyAdapter:
    def __init__(
        self,
        actor: SquashedGaussianActor,
        algorithm: str,
        device: torch.device,
    ) -> None:
        self.actor = actor.eval()
        self.algorithm = algorithm
        self.device = device
        self._continuous = True
        self._controller_name = f"trained_{algorithm}_offline"

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True) -> tuple[np.ndarray, None]:
        obs_array = np.asarray(obs, dtype=np.float32)
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        action = self.actor.act(obs_tensor, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy().astype(np.float32), None


def load_offline_policy_agent(
    checkpoint_path: str | Path,
    *,
    algorithm: str | None = None,
    device: str | None = None,
) -> OfflinePolicyAdapter:
    torch_device = _resolve_device(device)
    payload: dict[str, Any] = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location=torch_device,
        weights_only=False,
    )
    resolved_algorithm = str(algorithm or payload.get("algorithm", "")).strip().lower()
    if resolved_algorithm not in {"iql", "cql"}:
        raise ValueError(
            f"Unsupported offline policy algorithm={resolved_algorithm!r} in checkpoint."
        )
    actor = SquashedGaussianActor(
        int(payload["obs_dim"]),
        int(payload["action_dim"]),
        int(payload.get("hidden_dim", 256)),
    ).to(torch_device)
    actor.load_state_dict(payload["actor_state_dict"])
    return OfflinePolicyAdapter(actor, resolved_algorithm, torch_device)


def summarize_offline_dataset(path: str | Path) -> dict[str, float]:
    dataset = load_offline_dataset(path)
    rewards = dataset["rewards"].reshape(-1)
    dones = dataset["dones"].reshape(-1)
    action_norm = np.linalg.norm(dataset["actions"], axis=1)
    return {
        "transitions": float(dataset["obs"].shape[0]),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "terminal_rate": float(dones.mean()),
        "action_norm_mean": float(action_norm.mean()),
        "action_norm_std": float(action_norm.std()),
    }
