"""Pure-PyTorch TQC matching the project's SB3-Contrib training semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
SQUASH_EPSILON = 1e-6


def _hidden_mlp(input_dim: int, hidden_dims: Sequence[int]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    width = int(input_dim)
    for hidden_dim in hidden_dims:
        hidden_dim = int(hidden_dim)
        if hidden_dim < 1:
            raise ValueError("hidden layer widths must be positive")
        layers.extend((nn.Linear(width, hidden_dim), nn.ReLU()))
        width = hidden_dim
    return nn.Sequential(*layers), width


def _output_mlp(input_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> nn.Sequential:
    hidden, width = _hidden_mlp(input_dim, hidden_dims)
    return nn.Sequential(*hidden, nn.Linear(width, output_dim))


class SquashedGaussianActor(nn.Module):
    """Tanh-squashed diagonal Gaussian actor with SB3-compatible log probability."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 512),
    ) -> None:
        super().__init__()
        if observation_dim < 1 or action_dim < 1:
            raise ValueError("observation_dim and action_dim must be positive")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.latent, last_dim = _hidden_mlp(self.observation_dim, hidden_dims)
        self.mean = nn.Linear(last_dim, self.action_dim)
        self.log_std = nn.Linear(last_dim, self.action_dim)

    def distribution_parameters(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.latent(observations)
        mean = self.mean(latent)
        log_std = self.log_std(latent).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def action_log_prob(
        self,
        observations: Tensor,
        *,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return reparameterized actions and column-vector log probabilities."""
        mean, log_std = self.distribution_parameters(observations)
        std = log_std.exp()
        if noise is None:
            gaussian_actions = torch.distributions.Normal(mean, std).rsample()
        else:
            if noise.shape != mean.shape:
                raise ValueError(f"noise shape {tuple(noise.shape)} != {tuple(mean.shape)}")
            gaussian_actions = mean + std * noise
        actions = torch.tanh(gaussian_actions)
        log_prob = torch.distributions.Normal(mean, std).log_prob(gaussian_actions).sum(dim=1)
        log_prob -= torch.log(1.0 - actions.square() + SQUASH_EPSILON).sum(dim=1)
        return actions, log_prob.reshape(-1, 1)

    def forward(self, observations: Tensor, deterministic: bool = False) -> Tensor:
        if deterministic:
            mean, _ = self.distribution_parameters(observations)
            return torch.tanh(mean)
        actions, _ = self.action_log_prob(observations)
        return actions


class QuantileCritic(nn.Module):
    """Ensemble of quantile critics returning ``[batch, critics, quantiles]``."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 512),
        *,
        n_critics: int = 2,
        n_quantiles: int = 25,
    ) -> None:
        super().__init__()
        if observation_dim < 1 or action_dim < 1:
            raise ValueError("observation_dim and action_dim must be positive")
        if n_critics < 1 or n_quantiles < 1:
            raise ValueError("n_critics and n_quantiles must be positive")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.n_critics = int(n_critics)
        self.n_quantiles = int(n_quantiles)
        self.quantiles_total = self.n_critics * self.n_quantiles
        self.q_networks = nn.ModuleList(
            _output_mlp(
                self.observation_dim + self.action_dim,
                self.n_quantiles,
                hidden_dims,
            )
            for _ in range(self.n_critics)
        )

    def forward(self, observations: Tensor, actions: Tensor) -> Tensor:
        inputs = torch.cat((observations, actions), dim=1)
        return torch.stack(tuple(network(inputs) for network in self.q_networks), dim=1)


def quantile_huber_loss(
    current_quantiles: Tensor,
    target_quantiles: Tensor,
    cumulative_probabilities: Tensor | None = None,
    *,
    sum_over_quantiles: bool = True,
) -> Tensor:
    """Quantile-regression Huber loss, equivalent to SB3-Contrib's implementation."""
    if current_quantiles.ndim != target_quantiles.ndim:
        raise ValueError(
            "current_quantiles and target_quantiles must have the same number of dimensions"
        )
    if current_quantiles.shape[0] != target_quantiles.shape[0]:
        raise ValueError("current_quantiles and target_quantiles batch sizes must match")
    if current_quantiles.ndim not in (2, 3):
        raise ValueError("quantile tensors must have two or three dimensions")

    if cumulative_probabilities is None:
        n_quantiles = current_quantiles.shape[-1]
        cumulative_probabilities = (
            torch.arange(n_quantiles, device=current_quantiles.device, dtype=torch.float) + 0.5
        ) / n_quantiles
        view_shape = (1, -1, 1) if current_quantiles.ndim == 2 else (1, 1, -1, 1)
        cumulative_probabilities = cumulative_probabilities.view(*view_shape)

    pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
    absolute_delta = pairwise_delta.abs()
    huber = torch.where(
        absolute_delta > 1.0,
        absolute_delta - 0.5,
        0.5 * pairwise_delta.square(),
    )
    loss = (
        cumulative_probabilities - (pairwise_delta.detach() < 0).float()
    ).abs() * huber
    if sum_over_quantiles:
        return loss.sum(dim=-2).mean()
    return loss.mean()


def truncate_target_quantiles(next_quantiles: Tensor, top_drop_per_critic: int) -> Tensor:
    """Sort the critic mixture and remove the configured upper-tail quantiles."""
    if next_quantiles.ndim != 3:
        raise ValueError("next_quantiles must have shape [batch, critics, quantiles]")
    if top_drop_per_critic < 0:
        raise ValueError("top_drop_per_critic cannot be negative")
    n_critics = next_quantiles.shape[1]
    keep = next_quantiles.shape[1] * next_quantiles.shape[2] - top_drop_per_critic * n_critics
    if keep < 1:
        raise ValueError("quantile truncation must retain at least one target quantile")
    sorted_quantiles = torch.sort(next_quantiles.reshape(next_quantiles.shape[0], -1), dim=1).values
    return sorted_quantiles[:, :keep]


def _column(value: Tensor, *, name: str) -> Tensor:
    if value.ndim == 1:
        return value.reshape(-1, 1)
    if value.ndim == 2 and value.shape[1] == 1:
        return value
    raise ValueError(f"{name} must have shape [batch] or [batch, 1]")


def build_target_quantiles(
    *,
    next_quantiles: Tensor,
    next_log_prob: Tensor,
    rewards: Tensor,
    dones: Tensor,
    discount: float | Tensor,
    entropy_coefficient: Tensor,
    top_drop_per_critic: int,
) -> Tensor:
    """Build SB3 TQC's entropy-adjusted, truncated Bellman quantile target."""
    truncated = truncate_target_quantiles(next_quantiles, top_drop_per_critic)
    next_log_prob = _column(next_log_prob, name="next_log_prob")
    rewards = _column(rewards, name="rewards")
    dones = _column(dones, name="dones")
    entropy_adjusted = truncated - entropy_coefficient * next_log_prob
    if isinstance(discount, Tensor):
        discount_value: float | Tensor = _column(discount, name="discount")
    else:
        discount_value = float(discount)
    targets = rewards + (1.0 - dones) * discount_value * entropy_adjusted
    return targets.unsqueeze(1)


@dataclass(frozen=True)
class TQCConfig:
    observation_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    n_critics: int = 2
    n_quantiles: int = 25
    top_quantiles_to_drop_per_critic: int = 2
    gamma: float = 0.95
    tau: float = 0.05
    learning_rate: float = 1e-3
    target_entropy: float | None = None
    adam_epsilon: float = 1e-5
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.observation_dim < 1 or self.action_dim < 1:
            raise ValueError("observation_dim and action_dim must be positive")
        if self.n_critics < 1 or self.n_quantiles < 1:
            raise ValueError("n_critics and n_quantiles must be positive")
        if self.top_quantiles_to_drop_per_critic < 0:
            raise ValueError("top quantiles to drop cannot be negative")
        if self.top_quantiles_to_drop_per_critic >= self.n_quantiles:
            raise ValueError("top quantiles to drop must be less than n_quantiles")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be between zero and one")
        if self.learning_rate <= 0.0 or self.adam_epsilon <= 0.0:
            raise ValueError("optimizer settings must be positive")

    @property
    def resolved_target_entropy(self) -> float:
        return -float(self.action_dim) if self.target_entropy is None else float(self.target_entropy)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["resolved_target_entropy"] = self.resolved_target_entropy
        return result


@dataclass(frozen=True)
class TQCBatch:
    observations: Tensor
    actions: Tensor
    next_observations: Tensor
    rewards: Tensor
    dones: Tensor
    discounts: Tensor | None = None


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters(), strict=True):
        target_parameter.mul_(1.0 - tau)
        target_parameter.add_(source_parameter, alpha=tau)


class TQCLearner:
    """Own TQC networks and perform one SB3-compatible CUDA update at a time."""

    def __init__(self, config: TQCConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        actor_kwargs = (config.observation_dim, config.action_dim, config.hidden_dims)
        critic_kwargs = {
            "hidden_dims": config.hidden_dims,
            "n_critics": config.n_critics,
            "n_quantiles": config.n_quantiles,
        }
        self.actor = SquashedGaussianActor(*actor_kwargs).to(self.device)
        self.critic = QuantileCritic(
            config.observation_dim,
            config.action_dim,
            **critic_kwargs,
        ).to(self.device)
        self.critic_target = QuantileCritic(
            config.observation_dim,
            config.action_dim,
            **critic_kwargs,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.train(False)

        optimizer_kwargs = {"lr": config.learning_rate, "eps": config.adam_epsilon}
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), **optimizer_kwargs)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), **optimizer_kwargs)
        self.log_entropy_coefficient = nn.Parameter(torch.zeros(1, device=self.device))
        self.entropy_optimizer = torch.optim.Adam(
            (self.log_entropy_coefficient,),
            **optimizer_kwargs,
        )
        self.updates = 0

    def update(self, batch: TQCBatch) -> dict[str, Tensor | int]:
        actions_pi, log_prob = self.actor.action_log_prob(batch.observations)
        entropy_coefficient = self.log_entropy_coefficient.detach().exp()
        entropy_loss = -(
            self.log_entropy_coefficient
            * (log_prob + self.config.resolved_target_entropy).detach()
        ).mean()

        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.action_log_prob(batch.next_observations)
            next_quantiles = self.critic_target(batch.next_observations, next_actions)
            targets = build_target_quantiles(
                next_quantiles=next_quantiles,
                next_log_prob=next_log_prob,
                rewards=batch.rewards,
                dones=batch.dones,
                discount=self.config.gamma if batch.discounts is None else batch.discounts,
                entropy_coefficient=entropy_coefficient,
                top_drop_per_critic=self.config.top_quantiles_to_drop_per_critic,
            )

        current_quantiles = self.critic(batch.observations, batch.actions)
        critic_loss = quantile_huber_loss(
            current_quantiles,
            targets,
            sum_over_quantiles=False,
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        policy_quantiles = self.critic(batch.observations, actions_pi)
        q_policy = policy_quantiles.mean(dim=2).mean(dim=1, keepdim=True)
        actor_loss = (entropy_coefficient * log_prob - q_policy).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        polyak_update(self.critic, self.critic_target, self.config.tau)
        self.updates += 1
        return {
            # Keep update metrics device-resident. The trainer may transfer them only
            # at its lower-frequency logging boundary instead of synchronizing here.
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach(),
            "entropy_coefficient": entropy_coefficient.detach(),
            "entropy_coefficient_loss": entropy_loss.detach(),
            "updates": self.updates,
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "updates": self.updates,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_entropy_coefficient": self.log_entropy_coefficient.detach().clone(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "entropy_optimizer": self.entropy_optimizer.state_dict(),
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        with torch.no_grad():
            self.log_entropy_coefficient.copy_(checkpoint["log_entropy_coefficient"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.entropy_optimizer.load_state_dict(checkpoint["entropy_optimizer"])
        self.updates = int(checkpoint["updates"])
