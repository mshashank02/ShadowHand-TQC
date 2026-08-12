"""Memory-planned, episode-aware CUDA replay with vectorized future HER."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from .normalization import CudaVecNormalize, SB3_OBSERVATION_KEY_ORDER, flatten_observations
from .tqc import TQCBatch


RewardFunction = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class ReplayMemoryPlan:
    requested_capacity: int
    effective_capacity: int
    rows: int
    num_envs: int
    bytes_per_transition: int
    fixed_overhead_bytes: int
    storage_bytes: int
    memory_budget_bytes: int | None
    available_device_bytes: int | None
    auto_capacity: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shape_elements(shape: Sequence[int]) -> int:
    elements = 1
    for width in shape:
        elements *= int(width)
    return elements


def plan_replay_memory(
    *,
    requested_capacity: int,
    num_envs: int,
    observation_shapes: Mapping[str, Sequence[int]],
    action_dim: int,
    max_episode_steps: int,
    device: str | torch.device = "cuda",
    memory_budget_bytes: int | None = None,
    memory_fraction: float = 0.5,
    auto_capacity: bool = False,
) -> ReplayMemoryPlan:
    """Resolve replay capacity before allocating any large device tensor."""
    if requested_capacity < 1 or num_envs < 1 or action_dim < 1 or max_episode_steps < 1:
        raise ValueError("capacity, environment/action counts, and episode horizon must be positive")
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    observation_elements = sum(_shape_elements(shape) for shape in observation_shapes.values())
    # Current/next float32 observations, action/reward, bool done/timeout, and
    # int64 episode start/length per stored transition.
    bytes_per_transition = (
        2 * observation_elements * torch.tensor([], dtype=torch.float32).element_size()
        + action_dim * torch.tensor([], dtype=torch.float32).element_size()
        + torch.tensor([], dtype=torch.float32).element_size()
        + 2 * torch.tensor([], dtype=torch.bool).element_size()
        + 2 * torch.tensor([], dtype=torch.int64).element_size()
    )
    fixed_overhead_bytes = (
        2 * num_envs * torch.tensor([], dtype=torch.int64).element_size()
        + max_episode_steps * torch.tensor([], dtype=torch.int64).element_size()
    )

    resolved_device = torch.device(device)
    available_device_bytes: int | None = None
    if resolved_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA replay was requested but torch.cuda.is_available() is false")
        device_index = (
            torch.cuda.current_device() if resolved_device.index is None else resolved_device.index
        )
        available_device_bytes = int(torch.cuda.mem_get_info(device_index)[0])
        if memory_budget_bytes is None:
            memory_budget_bytes = int(available_device_bytes * memory_fraction)
    elif memory_budget_bytes is not None and memory_budget_bytes < 1:
        raise ValueError("memory_budget_bytes must be positive")

    requested_rows = max(requested_capacity // num_envs, 1)
    if requested_rows < max_episode_steps:
        minimum_capacity = max_episode_steps * num_envs
        raise ValueError(
            "CUDA HER replay must hold at least one complete episode per world: "
            f"requested_capacity={requested_capacity:,}, minimum_capacity={minimum_capacity:,}"
        )
    rows = requested_rows
    requested_storage = rows * num_envs * bytes_per_transition + fixed_overhead_bytes
    if memory_budget_bytes is not None and requested_storage > memory_budget_bytes:
        if not auto_capacity:
            raise MemoryError(
                "Requested CUDA HER replay does not fit its memory budget: "
                f"capacity={requested_capacity:,}, required={requested_storage:,} bytes, "
                f"budget={memory_budget_bytes:,} bytes. Reduce capacity or explicitly enable "
                "auto_capacity so the selected value is recorded."
            )
        usable_bytes = max(0, memory_budget_bytes - fixed_overhead_bytes)
        rows = usable_bytes // (bytes_per_transition * num_envs)
        rows = min(rows, requested_rows)
        if rows < max_episode_steps:
            minimum_bytes = (
                max_episode_steps * num_envs * bytes_per_transition + fixed_overhead_bytes
            )
            raise MemoryError(
                "The replay memory budget cannot hold one complete vectorized episode: "
                f"required={minimum_bytes:,} bytes, budget={memory_budget_bytes:,} bytes"
            )

    effective_capacity = rows * num_envs
    storage_bytes = effective_capacity * bytes_per_transition + fixed_overhead_bytes
    return ReplayMemoryPlan(
        requested_capacity=int(requested_capacity),
        effective_capacity=int(effective_capacity),
        rows=int(rows),
        num_envs=int(num_envs),
        bytes_per_transition=int(bytes_per_transition),
        fixed_overhead_bytes=int(fixed_overhead_bytes),
        storage_bytes=int(storage_bytes),
        memory_budget_bytes=memory_budget_bytes,
        available_device_bytes=available_device_bytes,
        auto_capacity=bool(auto_capacity),
    )


def _quaternion_conjugate(quaternion: Tensor) -> Tensor:
    result = -quaternion
    result = result.clone()
    result[..., 0] = quaternion[..., 0]
    return result


def _quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    w0, x0, y0, z0 = left.unbind(dim=-1)
    w1, x1, y1, z1 = right.unbind(dim=-1)
    return torch.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1,
            w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1,
        ),
        dim=-1,
    )


def _quaternion_to_euler(quaternion: Tensor) -> Tensor:
    """Torch equivalent of gymnasium_robotics.utils.rotations.quat2euler."""
    quaternion = quaternion.to(torch.float64)
    w, x, y, z = quaternion.unbind(dim=-1)
    norm_squared = quaternion.square().sum(dim=-1)
    scale = 2.0 / norm_squared
    x_scale, y_scale, z_scale = x * scale, y * scale, z * scale
    wx, wy, wz = w * x_scale, w * y_scale, w * z_scale
    xx, xy, xz = x * x_scale, x * y_scale, x * z_scale
    yy, yz, zz = y * y_scale, y * z_scale, z * z_scale
    matrix = torch.stack(
        (
            1.0 - (yy + zz),
            xy - wz,
            xz + wy,
            xy + wz,
            1.0 - (xx + zz),
            yz - wx,
            xz - wy,
            yz + wx,
            1.0 - (xx + yy),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    valid = norm_squared > np_finfo_float64_epsilon()
    identity = torch.eye(3, dtype=torch.float64, device=quaternion.device)
    matrix = torch.where(valid[..., None, None], matrix, identity)

    cosine_y = torch.sqrt(matrix[..., 2, 2].square() + matrix[..., 1, 2].square())
    condition = cosine_y > 4.0 * np_finfo_float64_epsilon()
    x_angle = torch.where(
        condition,
        -torch.atan2(matrix[..., 1, 2], matrix[..., 2, 2]),
        torch.zeros_like(cosine_y),
    )
    y_angle = -torch.atan2(-matrix[..., 0, 2], cosine_y)
    z_angle = torch.where(
        condition,
        -torch.atan2(matrix[..., 0, 1], matrix[..., 0, 0]),
        -torch.atan2(-matrix[..., 1, 0], matrix[..., 1, 1]),
    )
    return torch.stack((x_angle, y_angle, z_angle), dim=-1)


def np_finfo_float64_epsilon() -> float:
    # Kept local to avoid importing NumPy into the CUDA hot path.
    return 2.220446049250313e-16


def _euler_to_quaternion(euler: Tensor) -> Tensor:
    euler = euler.to(torch.float64)
    first = euler[..., 2] / 2.0
    second = -euler[..., 1] / 2.0
    third = euler[..., 0] / 2.0
    sin_first, sin_second, sin_third = torch.sin(first), torch.sin(second), torch.sin(third)
    cos_first, cos_second, cos_third = torch.cos(first), torch.cos(second), torch.cos(third)
    cosine_cosine = cos_first * cos_third
    cosine_sine = cos_first * sin_third
    sine_cosine = sin_first * cos_third
    sine_sine = sin_first * sin_third
    return torch.stack(
        (
            cos_second * cosine_cosine + sin_second * sine_sine,
            cos_second * cosine_sine - sin_second * sine_cosine,
            -(cos_second * sine_sine + sin_second * cosine_cosine),
            cos_second * sine_cosine - sin_second * cosine_sine,
        ),
        dim=-1,
    )


def shadowhand_sparse_reward(
    achieved_goal: Tensor,
    desired_goal: Tensor,
    *,
    distance_threshold: float = 0.01,
    rotation_threshold: float = 0.1,
    ignore_position: bool = False,
    ignore_rotation: bool = False,
    ignore_z_rotation: bool = False,
    legacy_ignore_z_batch_semantics: bool = False,
) -> Tensor:
    """Gymnasium-Robotics ShadowHand sparse reward on CUDA.

    The installed reference has a batch-indexing quirk in its ignore-Z path
    (``euler_a[2] = euler_b[2]``). HER invokes that function with a batch while
    online vector environments invoke it once per world. The explicit legacy flag
    reproduces the HER behavior; task stepping uses the per-world interpretation.
    """
    if achieved_goal.shape != desired_goal.shape or achieved_goal.shape[-1] != 7:
        raise ValueError("achieved and desired goals must have matching [..., 7] shapes")
    if ignore_position:
        position_distance = torch.zeros_like(achieved_goal[..., 0])
    else:
        position_distance = torch.linalg.vector_norm(
            achieved_goal[..., :3] - desired_goal[..., :3],
            dim=-1,
        )
    if ignore_rotation:
        rotation_distance = torch.zeros_like(desired_goal[..., 0])
    else:
        achieved_quaternion = achieved_goal[..., 3:]
        desired_quaternion = desired_goal[..., 3:]
        if ignore_z_rotation:
            achieved_euler = _quaternion_to_euler(achieved_quaternion).clone()
            desired_euler = _quaternion_to_euler(desired_quaternion)
            if legacy_ignore_z_batch_semantics and achieved_euler.ndim > 1:
                achieved_euler[2] = desired_euler[2]
            else:
                achieved_euler[..., 2] = desired_euler[..., 2]
            achieved_quaternion = _euler_to_quaternion(achieved_euler)
        difference = _quaternion_multiply(
            achieved_quaternion,
            _quaternion_conjugate(desired_quaternion),
        )
        rotation_distance = 2.0 * torch.acos(difference[..., 0].clamp(-1.0, 1.0))
    success = (position_distance < distance_threshold) & (rotation_distance < rotation_threshold)
    return success.to(torch.float32) - 1.0


@dataclass(frozen=True)
class HERReplaySample:
    observations: dict[str, Tensor]
    actions: Tensor
    next_observations: dict[str, Tensor]
    dones: Tensor
    rewards: Tensor
    source_rows: Tensor
    source_envs: Tensor
    virtual: Tensor
    future_rows: Tensor

    def to_tqc_batch(
        self,
        key_order: Sequence[str] = SB3_OBSERVATION_KEY_ORDER,
    ) -> TQCBatch:
        return TQCBatch(
            observations=flatten_observations(self.observations, key_order),
            actions=self.actions,
            next_observations=flatten_observations(self.next_observations, key_order),
            rewards=self.rewards,
            dones=self.dones,
        )


class CudaHERReplayBuffer:
    """Raw-transition CUDA ring buffer with complete-episode future HER sampling."""

    def __init__(
        self,
        *,
        requested_capacity: int,
        num_envs: int,
        observation_shapes: Mapping[str, Sequence[int]],
        action_dim: int,
        max_episode_steps: int = 100,
        n_sampled_goal: int = 4,
        reward_function: RewardFunction = shadowhand_sparse_reward,
        device: str | torch.device = "cuda",
        memory_budget_bytes: int | None = None,
        memory_fraction: float = 0.5,
        auto_capacity: bool = False,
    ) -> None:
        if n_sampled_goal < 0:
            raise ValueError("n_sampled_goal cannot be negative")
        required_keys = set(SB3_OBSERVATION_KEY_ORDER)
        if set(observation_shapes) != required_keys:
            raise ValueError(
                f"observation_shapes keys must be {sorted(required_keys)}, "
                f"found {sorted(observation_shapes)}"
            )
        if tuple(observation_shapes["achieved_goal"]) != (7,) or tuple(
            observation_shapes["desired_goal"]
        ) != (7,):
            raise ValueError("ShadowHand achieved_goal and desired_goal must both have shape (7,)")
        self.device = torch.device(device)
        self.plan = plan_replay_memory(
            requested_capacity=requested_capacity,
            num_envs=num_envs,
            observation_shapes=observation_shapes,
            action_dim=action_dim,
            max_episode_steps=max_episode_steps,
            device=self.device,
            memory_budget_bytes=memory_budget_bytes,
            memory_fraction=memory_fraction,
            auto_capacity=auto_capacity,
        )
        self.num_envs = int(num_envs)
        self.rows = self.plan.rows
        self.action_dim = int(action_dim)
        self.max_episode_steps = int(max_episode_steps)
        self.n_sampled_goal = int(n_sampled_goal)
        self.her_ratio = 1.0 - 1.0 / (self.n_sampled_goal + 1)
        self.reward_function = reward_function
        self.observation_shapes = {
            key: tuple(int(item) for item in shape) for key, shape in observation_shapes.items()
        }

        storage_shape = (self.rows, self.num_envs)
        self.observations = {
            key: torch.empty((*storage_shape, *shape), dtype=torch.float32, device=self.device)
            for key, shape in self.observation_shapes.items()
        }
        self.next_observations = {
            key: torch.empty((*storage_shape, *shape), dtype=torch.float32, device=self.device)
            for key, shape in self.observation_shapes.items()
        }
        self.actions = torch.empty(
            (*storage_shape, self.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.rewards = torch.empty(storage_shape, dtype=torch.float32, device=self.device)
        self.dones = torch.empty(storage_shape, dtype=torch.bool, device=self.device)
        self.timeouts = torch.empty(storage_shape, dtype=torch.bool, device=self.device)
        self.episode_start = torch.zeros(storage_shape, dtype=torch.int64, device=self.device)
        self.episode_length = torch.zeros(storage_shape, dtype=torch.int64, device=self.device)
        self.current_episode_start = torch.zeros(
            self.num_envs,
            dtype=torch.int64,
            device=self.device,
        )
        self.current_episode_steps = torch.zeros_like(self.current_episode_start)
        self._episode_offsets = torch.arange(
            self.max_episode_steps,
            dtype=torch.int64,
            device=self.device,
        )
        self.pos = 0
        self.full = False

    @torch.no_grad()
    def clear(self) -> None:
        """Discard all episode metadata without rewriting unused payload storage."""
        self.episode_start.zero_()
        self.episode_length.zero_()
        self.current_episode_start.zero_()
        self.current_episode_steps.zero_()
        self.pos = 0
        self.full = False

    def _copy_step_tensor(self, target: Tensor, value: Tensor, *, name: str) -> None:
        if value.shape != target.shape:
            raise ValueError(f"{name} shape {tuple(value.shape)} != {tuple(target.shape)}")
        target.copy_(value.to(device=self.device, dtype=target.dtype))

    @torch.no_grad()
    def add(
        self,
        *,
        observations: Mapping[str, Tensor],
        next_observations: Mapping[str, Tensor],
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        timeouts: Tensor,
    ) -> None:
        """Store one transition for every world and finalize masked episodes."""
        world_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        offsets = self._episode_offsets.unsqueeze(0)

        # Invalidate every transition from an old completed episode before any
        # member of that episode is overwritten, preventing partial sampling.
        old_starts = self.episode_start[self.pos]
        old_lengths = self.episode_length[self.pos]
        old_rows = (old_starts.unsqueeze(1) + offsets) % self.rows
        old_mask = offsets < old_lengths.unsqueeze(1)
        expanded_worlds = world_indices.expand(-1, self.max_episode_steps)
        self.episode_length[old_rows[old_mask], expanded_worlds[old_mask]] = 0

        self.episode_start[self.pos].copy_(self.current_episode_start)
        self.episode_length[self.pos].zero_()
        for key in self.observations:
            self._copy_step_tensor(
                self.observations[key][self.pos],
                observations[key],
                name=f"observations[{key!r}]",
            )
            self._copy_step_tensor(
                self.next_observations[key][self.pos],
                next_observations[key],
                name=f"next_observations[{key!r}]",
            )
        self._copy_step_tensor(self.actions[self.pos], actions, name="actions")
        self._copy_step_tensor(
            self.rewards[self.pos],
            rewards.reshape(self.num_envs),
            name="rewards",
        )
        done_values = dones.reshape(self.num_envs).to(device=self.device, dtype=torch.bool)
        timeout_values = timeouts.reshape(self.num_envs).to(device=self.device, dtype=torch.bool)
        self.dones[self.pos].copy_(done_values)
        self.timeouts[self.pos].copy_(timeout_values)

        self.pos += 1
        if self.pos == self.rows:
            self.pos = 0
            self.full = True

        self.current_episode_steps.add_(1)
        completed_lengths = self.current_episode_steps.clone()
        completed_rows = (self.current_episode_start.unsqueeze(1) + offsets) % self.rows
        completed_mask = done_values.unsqueeze(1) & (offsets < completed_lengths.unsqueeze(1))
        expanded_lengths = completed_lengths.unsqueeze(1).expand(-1, self.max_episode_steps)
        self.episode_length[
            completed_rows[completed_mask],
            expanded_worlds[completed_mask],
        ] = expanded_lengths[completed_mask]
        new_start = torch.full_like(self.current_episode_start, self.pos)
        self.current_episode_start.copy_(
            torch.where(done_values, new_start, self.current_episode_start)
        )
        self.current_episode_steps.masked_fill_(done_values, 0)

    def _gather(
        self,
        rows: Tensor,
        envs: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor, dict[str, Tensor], Tensor, Tensor]:
        observations = {key: values[rows, envs] for key, values in self.observations.items()}
        next_observations = {
            key: values[rows, envs] for key, values in self.next_observations.items()
        }
        actions = self.actions[rows, envs]
        dones = (self.dones[rows, envs] & ~self.timeouts[rows, envs]).to(torch.float32).reshape(-1, 1)
        rewards = self.rewards[rows, envs].reshape(-1, 1)
        return observations, actions, next_observations, dones, rewards

    @torch.no_grad()
    def sample_from_indices(
        self,
        rows: Tensor,
        envs: Tensor,
        *,
        virtual_count: int,
        future_rows: Tensor | None = None,
        future_uniform: Tensor | None = None,
        normalizer: CudaVecNormalize | None = None,
    ) -> HERReplaySample:
        """Assemble an SB3-ordered batch from explicit source indices for parity tests."""
        rows = rows.to(device=self.device, dtype=torch.int64)
        envs = envs.to(device=self.device, dtype=torch.int64)
        if rows.ndim != 1 or rows.shape != envs.shape:
            raise ValueError("rows and envs must be same-shaped vectors")
        batch_size = rows.shape[0]
        if not 0 <= virtual_count <= batch_size:
            raise ValueError("virtual_count must be between zero and batch size")
        virtual_rows = rows[:virtual_count]
        virtual_envs = envs[:virtual_count]
        real_rows = rows[virtual_count:]
        real_envs = envs[virtual_count:]

        real_obs, real_actions, real_next, real_dones, real_rewards = self._gather(
            real_rows,
            real_envs,
        )
        virtual_obs, virtual_actions, virtual_next, virtual_dones, _ = self._gather(
            virtual_rows,
            virtual_envs,
        )
        if future_rows is None:
            starts = self.episode_start[virtual_rows, virtual_envs]
            lengths = self.episode_length[virtual_rows, virtual_envs]
            current_in_episode = (virtual_rows - starts) % self.rows
            choices = lengths - current_in_episode
            if future_uniform is None:
                future_uniform = torch.rand(virtual_count, device=self.device)
            else:
                future_uniform = future_uniform.to(device=self.device, dtype=torch.float32)
                if future_uniform.shape != (virtual_count,):
                    raise ValueError("future_uniform must have shape [virtual_count]")
            future_in_episode = current_in_episode + torch.floor(
                future_uniform * choices.to(torch.float32)
            ).to(torch.int64)
            future_rows = (starts + future_in_episode) % self.rows
        else:
            future_rows = future_rows.to(device=self.device, dtype=torch.int64)
            if future_rows.shape != (virtual_count,):
                raise ValueError("future_rows must have shape [virtual_count]")

        new_goals = self.next_observations["achieved_goal"][future_rows, virtual_envs]
        virtual_obs["desired_goal"] = new_goals
        virtual_next["desired_goal"] = new_goals
        virtual_rewards = self.reward_function(
            virtual_next["achieved_goal"],
            virtual_obs["desired_goal"],
        ).reshape(-1, 1)

        combined_observations = {
            key: torch.cat((real_obs[key], virtual_obs[key]), dim=0)
            for key in self.observations
        }
        combined_next = {
            key: torch.cat((real_next[key], virtual_next[key]), dim=0)
            for key in self.next_observations
        }
        combined_rewards = torch.cat((real_rewards, virtual_rewards), dim=0)
        if normalizer is not None:
            combined_observations = normalizer.normalize_observations(combined_observations)
            combined_next = normalizer.normalize_observations(combined_next)
            combined_rewards = normalizer.normalize_rewards(combined_rewards)
        real_count = batch_size - virtual_count
        return HERReplaySample(
            observations=combined_observations,
            actions=torch.cat((real_actions, virtual_actions), dim=0),
            next_observations=combined_next,
            dones=torch.cat((real_dones, virtual_dones), dim=0),
            rewards=combined_rewards,
            source_rows=torch.cat((real_rows, virtual_rows), dim=0),
            source_envs=torch.cat((real_envs, virtual_envs), dim=0),
            virtual=torch.cat(
                (
                    torch.zeros(real_count, dtype=torch.bool, device=self.device),
                    torch.ones(virtual_count, dtype=torch.bool, device=self.device),
                )
            ),
            future_rows=future_rows,
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        *,
        normalizer: CudaVecNormalize | None = None,
        generator: torch.Generator | None = None,
    ) -> HERReplaySample:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        # Completed transitions have equal unit sampling weight. torch.multinomial
        # keeps index selection on-device and never exposes a valid-count scalar.
        valid_weights = self.episode_length.reshape(-1).gt(0).to(torch.float32)
        flat_indices = torch.multinomial(
            valid_weights,
            batch_size,
            replacement=True,
            generator=generator,
        )
        rows = torch.div(flat_indices, self.num_envs, rounding_mode="floor")
        envs = flat_indices % self.num_envs
        virtual_count = int(self.her_ratio * batch_size)
        future_uniform = torch.rand(
            virtual_count,
            device=self.device,
            generator=generator,
        )
        return self.sample_from_indices(
            rows,
            envs,
            virtual_count=virtual_count,
            future_uniform=future_uniform,
            normalizer=normalizer,
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "pos": self.pos,
            "full": self.full,
            "observations": self.observations,
            "next_observations": self.next_observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "timeouts": self.timeouts,
            "episode_start": self.episode_start,
            "episode_length": self.episode_length,
            "current_episode_start": self.current_episode_start,
            "current_episode_steps": self.current_episode_steps,
        }

    def load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        checkpoint_plan = checkpoint["plan"]
        if int(checkpoint_plan["effective_capacity"]) != self.plan.effective_capacity:
            raise ValueError("replay checkpoint capacity does not match allocated storage")
        for key in self.observations:
            self.observations[key].copy_(checkpoint["observations"][key])
            self.next_observations[key].copy_(checkpoint["next_observations"][key])
        for name in (
            "actions",
            "rewards",
            "dones",
            "timeouts",
            "episode_start",
            "episode_length",
            "current_episode_start",
            "current_episode_steps",
        ):
            getattr(self, name).copy_(checkpoint[name])
        self.pos = int(checkpoint["pos"])
        self.full = bool(checkpoint["full"])
