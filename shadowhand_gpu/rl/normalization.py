"""CUDA-resident running normalization matching SB3 VecNormalize semantics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


SB3_OBSERVATION_KEY_ORDER = ("achieved_goal", "desired_goal", "observation")


class TorchRunningMeanStd(nn.Module):
    """Parallel running moments with SB3's float64 state and epsilon initialization."""

    def __init__(
        self,
        shape: Sequence[int] = (),
        *,
        epsilon: float = 1e-4,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        normalized_shape = tuple(int(item) for item in shape)
        self.register_buffer("mean", torch.zeros(normalized_shape, dtype=torch.float64, device=device))
        self.register_buffer("var", torch.ones(normalized_shape, dtype=torch.float64, device=device))
        self.register_buffer("count", torch.tensor(float(epsilon), dtype=torch.float64, device=device))

    @torch.no_grad()
    def update(self, values: Tensor) -> None:
        # SB3/NumPy computes batch moments in the incoming array dtype and then
        # combines them with float64 running state.
        batch_mean = values.mean(dim=0).to(torch.float64)
        batch_var = values.var(dim=0, correction=0).to(torch.float64)
        self.update_from_moments(batch_mean, batch_var, values.shape[0])

    @torch.no_grad()
    def update_from_moments(
        self,
        batch_mean: Tensor,
        batch_var: Tensor,
        batch_count: int | Tensor,
    ) -> None:
        batch_count_tensor = torch.as_tensor(
            batch_count,
            dtype=torch.float64,
            device=self.count.device,
        )
        delta = batch_mean - self.mean
        total_count = self.count + batch_count_tensor
        new_mean = self.mean + delta * batch_count_tensor / total_count
        first_moment = self.var * self.count
        second_moment = batch_var * batch_count_tensor
        combined_moment = (
            first_moment
            + second_moment
            + delta.square() * self.count * batch_count_tensor / total_count
        )
        self.mean.copy_(new_mean)
        self.var.copy_(combined_moment / total_count)
        self.count.copy_(total_count)


def flatten_observations(
    observations: Mapping[str, Tensor],
    key_order: Sequence[str] = SB3_OBSERVATION_KEY_ORDER,
) -> Tensor:
    """Flatten a goal dictionary in SB3 CombinedExtractor/Gym Dict key order."""
    missing = [key for key in key_order if key not in observations]
    extras = [key for key in observations if key not in key_order]
    if missing or extras:
        raise ValueError(f"observation keys mismatch: missing={missing}, extras={extras}")
    return torch.cat(
        tuple(observations[key].reshape(observations[key].shape[0], -1) for key in key_order),
        dim=1,
    )


class CudaVecNormalize(nn.Module):
    """Device-side Dict observation and discounted-return normalization.

    Raw observations and rewards should be stored in replay. Sampling calls
    :meth:`normalize_observations` and :meth:`normalize_rewards` with the current
    statistics, matching SB3's VecNormalize/HerReplayBuffer interaction.
    """

    def __init__(
        self,
        observation_shapes: Mapping[str, Sequence[int]],
        *,
        num_envs: int,
        gamma: float = 0.95,
        epsilon: float = 1e-8,
        clip_observation: float = 10.0,
        clip_reward: float = 10.0,
        training: bool = True,
        normalize_observation: bool = True,
        normalize_reward: bool = True,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if epsilon <= 0.0 or clip_observation <= 0.0 or clip_reward <= 0.0:
            raise ValueError("epsilon and clipping limits must be positive")
        self.num_envs = int(num_envs)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.clip_observation = float(clip_observation)
        self.clip_reward = float(clip_reward)
        self.training = bool(training)
        self.normalize_observation = bool(normalize_observation)
        self.normalize_reward = bool(normalize_reward)
        self.observation_shapes = {
            key: tuple(int(item) for item in shape) for key, shape in observation_shapes.items()
        }
        self.observation_rms = nn.ModuleDict(
            {
                key: TorchRunningMeanStd(shape, device=device)
                for key, shape in self.observation_shapes.items()
            }
        )
        self.return_rms = TorchRunningMeanStd((), device=device)
        self.register_buffer("returns", torch.zeros(self.num_envs, dtype=torch.float64, device=device))

    @property
    def device(self) -> torch.device:
        return self.returns.device

    @torch.no_grad()
    def _update_observations(self, observations: Mapping[str, Tensor]) -> None:
        for key, running in self.observation_rms.items():
            running.update(observations[key])

    @torch.no_grad()
    def reset(self, observations: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self.returns.zero_()
        if self.training and self.normalize_observation:
            self._update_observations(observations)
        return self.normalize_observations(observations)

    @torch.no_grad()
    def step(
        self,
        observations: Mapping[str, Tensor],
        rewards: Tensor,
        dones: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Update stats and normalize one vectorized environment step."""
        if self.training and self.normalize_observation:
            self._update_observations(observations)
        if self.training:
            reward_vector = rewards.reshape(self.num_envs).to(torch.float64)
            self.returns.mul_(self.gamma).add_(reward_vector)
            self.return_rms.update(self.returns)
        normalized_observations = self.normalize_observations(observations)
        normalized_rewards = self.normalize_rewards(rewards)
        self.returns.masked_fill_(dones.reshape(self.num_envs).bool(), 0.0)
        return normalized_observations, normalized_rewards

    @torch.no_grad()
    def normalize_observations(self, observations: Mapping[str, Tensor]) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        for key, values in observations.items():
            if self.normalize_observation:
                running = self.observation_rms[key]
                normalized = (values.to(torch.float64) - running.mean) / torch.sqrt(
                    running.var + self.epsilon
                )
                result[key] = normalized.clamp(
                    -self.clip_observation,
                    self.clip_observation,
                ).to(torch.float32)
            else:
                result[key] = values.to(torch.float32)
        return result

    @torch.no_grad()
    def normalize_rewards(self, rewards: Tensor) -> Tensor:
        rewards = rewards.to(torch.float64)
        if self.normalize_reward:
            rewards = rewards / torch.sqrt(self.return_rms.var + self.epsilon)
            rewards = rewards.clamp(-self.clip_reward, self.clip_reward)
        return rewards.to(torch.float32)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "state_dict": self.state_dict(),
            "training": self.training,
            "normalize_observation": self.normalize_observation,
            "normalize_reward": self.normalize_reward,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "clip_observation": self.clip_observation,
            "clip_reward": self.clip_reward,
            "num_envs": self.num_envs,
            "observation_shapes": self.observation_shapes,
        }

    def load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if int(checkpoint["num_envs"]) != self.num_envs:
            raise ValueError("normalizer checkpoint num_envs does not match")
        checkpoint_shapes = {
            key: tuple(int(item) for item in shape)
            for key, shape in checkpoint["observation_shapes"].items()
        }
        if checkpoint_shapes != self.observation_shapes:
            raise ValueError("normalizer checkpoint observation shapes do not match")
        if float(checkpoint["gamma"]) != self.gamma:
            raise ValueError("normalizer checkpoint gamma does not match")
        self.load_state_dict(checkpoint["state_dict"])
        self.training = bool(checkpoint["training"])
        self.normalize_observation = bool(checkpoint["normalize_observation"])
        self.normalize_reward = bool(checkpoint["normalize_reward"])
