"""Native-rigid batched ShadowHand task logic on direct MuJoCo Warp data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .rl.replay import shadowhand_sparse_reward
from .warp_backend import MujocoWarpBackend


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


@dataclass(frozen=True)
class TaskModelLayout:
    robot_qpos_indices: tuple[int, ...]
    robot_qvel_indices: tuple[int, ...]
    object_qpos_start: int
    object_qvel_start: int
    target_qpos_start: int
    control_centers: tuple[float, ...]
    control_half_ranges: tuple[float, ...]
    control_lows: tuple[float, ...]
    control_highs: tuple[float, ...]

    @classmethod
    def from_model(cls, model: Any) -> "TaskModelLayout":
        import mujoco

        robot_qpos_indices: list[int] = []
        robot_qvel_indices: list[int] = []
        for joint_id in range(int(model.njnt)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if not name or not name.startswith("robot"):
                continue
            joint_type = int(model.jnt_type[joint_id])
            if joint_type not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise NotImplementedError(f"robot joint {name!r} is not scalar")
            robot_qpos_indices.append(int(model.jnt_qposadr[joint_id]))
            robot_qvel_indices.append(int(model.jnt_dofadr[joint_id]))

        def joint_addresses(name: str) -> tuple[int, int]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"required free joint {name!r} was not found")
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError(f"required joint {name!r} is not free")
            return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])

        object_qpos_start, object_qvel_start = joint_addresses("object:joint")
        target_qpos_start, _ = joint_addresses("target:joint")
        control_ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
        if tuple(control_ranges.shape) != (20, 2):
            raise ValueError(f"ShadowHand task requires 20 actuators, found {int(model.nu)}")
        centers = 0.5 * (control_ranges[:, 0] + control_ranges[:, 1])
        half_ranges = 0.5 * (control_ranges[:, 1] - control_ranges[:, 0])
        if len(robot_qpos_indices) != 24 or len(robot_qvel_indices) != 24:
            raise ValueError(
                "ShadowHand task requires 24 scalar robot joints, found "
                f"{len(robot_qpos_indices)}"
            )
        return cls(
            robot_qpos_indices=tuple(robot_qpos_indices),
            robot_qvel_indices=tuple(robot_qvel_indices),
            object_qpos_start=object_qpos_start,
            object_qvel_start=object_qvel_start,
            target_qpos_start=target_qpos_start,
            control_centers=tuple(float(value) for value in centers),
            control_half_ranges=tuple(float(value) for value in half_ranges),
            control_lows=tuple(float(value) for value in control_ranges[:, 0]),
            control_highs=tuple(float(value) for value in control_ranges[:, 1]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShadowHandTaskConfig:
    max_episode_steps: int = 100
    physics_steps_per_action: int = 20
    base_reset_settle_policy_steps: int = 10
    additional_reset_settle_policy_steps: int = 0
    action_scale: float = 1.0
    action_clip: float | None = None
    action_smoothing: float = 0.0
    target_position: str = "random"
    target_rotation: str = "xyz"
    ignore_z_rotation: bool = False
    distance_threshold: float = 0.01
    rotation_threshold: float = 0.1
    max_reset_attempts: int = 5

    def __post_init__(self) -> None:
        if self.max_episode_steps < 1 or self.physics_steps_per_action < 1:
            raise ValueError("episode and physics step counts must be positive")
        if self.base_reset_settle_policy_steps < 1:
            raise ValueError("base reset settling must contain at least one policy step")
        if self.additional_reset_settle_policy_steps < 0 or self.max_reset_attempts < 1:
            raise ValueError("additional settling cannot be negative and attempts must be positive")
        if not 0.0 <= self.action_smoothing <= 0.98:
            raise ValueError("action_smoothing must be in [0, 0.98]")
        if self.action_clip is not None and self.action_clip <= 0.0:
            object.__setattr__(self, "action_clip", None)
        if self.target_position not in ("random", "ignore"):
            raise ValueError("production target_position must be 'random' or 'ignore'")
        if self.target_rotation != "xyz":
            raise ValueError("production target_rotation currently must be 'xyz'")


@dataclass(frozen=True)
class ShadowHandTaskStep:
    observations: dict[str, Tensor]
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    success: Tensor
    conditioned_actions: Tensor


class PerWorldRandom:
    """Small device-side independent RNG stream for every simulated world."""

    _MODULUS = 2**31
    _MASK = _MODULUS - 1
    _MULTIPLIER = 1_103_515_245
    _INCREMENT = 12_345

    def __init__(self, num_worlds: int, seed: int, device: torch.device) -> None:
        world_ids = torch.arange(num_worlds, dtype=torch.int64, device=device)
        self.state = (int(seed) + 1 + world_ids * 1_000_003) & self._MASK

    def uniform(
        self,
        width: int,
        *,
        mask: Tensor | None = None,
        low: float = 0.0,
        high: float = 1.0,
    ) -> Tensor:
        if width < 1:
            raise ValueError("random width must be positive")
        if mask is None:
            mask = torch.ones_like(self.state, dtype=torch.bool)
        values: list[Tensor] = []
        for _ in range(width):
            candidate = (self._MULTIPLIER * self.state + self._INCREMENT) & self._MASK
            self.state = torch.where(mask, candidate, self.state)
            values.append(candidate.to(torch.float32) / float(self._MODULUS))
        result = torch.stack(values, dim=1)
        return low + (high - low) * result

    def normal(self, width: int, *, mask: Tensor | None = None) -> Tensor:
        pairs = (width + 1) // 2
        uniforms = self.uniform(2 * pairs, mask=mask).clamp_min_(1e-7)
        radius = torch.sqrt(-2.0 * torch.log(uniforms[:, 0::2]))
        angle = 2.0 * math.pi * uniforms[:, 1::2]
        values = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=2)
        return values.reshape(values.shape[0], -1)[:, :width]


def condition_actions(
    actions: Tensor,
    previous_actions: Tensor,
    *,
    action_scale: float,
    action_clip: float | None,
    action_smoothing: float,
) -> Tensor:
    """Apply the legacy wrapper's clip/scale/smooth order entirely on device."""
    conditioned = actions.clamp(-1.0, 1.0)
    if action_scale != 1.0:
        conditioned = conditioned * action_scale
    if action_clip is not None:
        conditioned = conditioned.clamp(-action_clip, action_clip)
    if action_smoothing > 0.0:
        conditioned = action_smoothing * previous_actions + (1.0 - action_smoothing) * conditioned
        if action_clip is not None:
            conditioned = conditioned.clamp(-action_clip, action_clip)
    return conditioned


def build_task_observations(
    *,
    qpos: Tensor,
    qvel: Tensor,
    touch: Tensor,
    desired_goals: Tensor,
    episode_steps: Tensor,
    max_episode_steps: int,
    robot_qpos_indices: Tensor,
    robot_qvel_indices: Tensor,
    object_qpos_start: int,
    object_qvel_start: int,
) -> dict[str, Tensor]:
    achieved_goals = qpos[:, object_qpos_start : object_qpos_start + 7]
    time_feature = 1.0 - episode_steps.to(qpos.dtype) / float(max_episode_steps)
    observation = torch.cat(
        (
            qpos.index_select(1, robot_qpos_indices),
            qvel.index_select(1, robot_qvel_indices),
            qvel[:, object_qvel_start : object_qvel_start + 6],
            achieved_goals,
            touch,
            time_feature.unsqueeze(1),
        ),
        dim=1,
    )
    return {
        "observation": observation,
        "achieved_goal": achieved_goals,
        "desired_goal": desired_goals,
    }


class ShadowHandWarpTask:
    """Batched task state and reset/step orchestration over one MJWarp backend."""

    _TARGET_POSITION_RANGE = ((-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06))

    def __init__(
        self,
        backend: MujocoWarpBackend,
        *,
        config: ShadowHandTaskConfig = ShadowHandTaskConfig(),
        seed: int = 0,
    ) -> None:
        if int(backend.model.nflex) != 0:
            raise NotImplementedError("ShadowHandWarpTask supports native-rigid models only")
        if int(backend.model.na) != 0:
            raise NotImplementedError("actuator activation state is not implemented")
        self.backend = backend
        self.config = config
        self.layout = TaskModelLayout.from_model(backend.model)
        self.device = backend.torch_device
        self.worlds = backend.worlds
        self.dtype = backend.qpos.dtype
        self.rng = PerWorldRandom(self.worlds, seed, self.device)
        self.robot_qpos_indices = torch.tensor(
            self.layout.robot_qpos_indices, dtype=torch.int64, device=self.device
        )
        self.robot_qvel_indices = torch.tensor(
            self.layout.robot_qvel_indices, dtype=torch.int64, device=self.device
        )
        self.control_centers = torch.tensor(
            self.layout.control_centers, dtype=self.dtype, device=self.device
        )
        self.control_half_ranges = torch.tensor(
            self.layout.control_half_ranges, dtype=self.dtype, device=self.device
        )
        self.control_lows = torch.tensor(
            self.layout.control_lows, dtype=self.dtype, device=self.device
        )
        self.control_highs = torch.tensor(
            self.layout.control_highs, dtype=self.dtype, device=self.device
        )
        self.initial_qpos = torch.tensor(
            np.asarray(backend.model.qpos0), dtype=self.dtype, device=self.device
        )
        self.goals = torch.zeros(self.worlds, 7, dtype=self.dtype, device=self.device)
        self.last_actions = torch.zeros(
            self.worlds, int(backend.model.nu), dtype=self.dtype, device=self.device
        )
        self.episode_steps = torch.zeros(
            self.worlds, dtype=torch.int64, device=self.device
        )
        self.episode_ids = torch.full(
            (self.worlds,), -1, dtype=torch.int64, device=self.device
        )

    @property
    def observation_width(self) -> int:
        return 62 + len(self.backend.sensor_layout.touch_data_indices)

    def _absolute_controls(self, conditioned_actions: Tensor) -> Tensor:
        return (self.control_centers + conditioned_actions * self.control_half_ranges).clamp(
            self.control_lows, self.control_highs
        )

    @torch.no_grad()
    def apply_actions(self, actions: Tensor) -> Tensor:
        if tuple(actions.shape) != (self.worlds, 20):
            raise ValueError(f"actions must have shape ({self.worlds}, 20)")
        actions = actions.to(device=self.device, dtype=self.dtype)
        conditioned = condition_actions(
            actions,
            self.last_actions,
            action_scale=self.config.action_scale,
            action_clip=self.config.action_clip,
            action_smoothing=self.config.action_smoothing,
        )
        self.last_actions.copy_(conditioned)
        self.backend.ctrl.copy_(self._absolute_controls(conditioned))
        return conditioned

    def observations(self) -> dict[str, Tensor]:
        return build_task_observations(
            qpos=self.backend.qpos,
            qvel=self.backend.qvel,
            touch=self.backend.touch,
            desired_goals=self.goals,
            episode_steps=self.episode_steps,
            max_episode_steps=self.config.max_episode_steps,
            robot_qpos_indices=self.robot_qpos_indices,
            robot_qvel_indices=self.robot_qvel_indices,
            object_qpos_start=self.layout.object_qpos_start,
            object_qvel_start=self.layout.object_qvel_start,
        )

    def compute_rewards(self, achieved_goals: Tensor, desired_goals: Tensor) -> Tensor:
        """Task reward callback shared by online stepping and future HER."""
        return shadowhand_sparse_reward(
            achieved_goals,
            desired_goals,
            distance_threshold=self.config.distance_threshold,
            rotation_threshold=self.config.rotation_threshold,
            ignore_position=self.config.target_position == "ignore",
            ignore_z_rotation=self.config.ignore_z_rotation,
            legacy_ignore_z_batch_semantics=False,
        )

    def compute_her_rewards(self, achieved_goals: Tensor, desired_goals: Tensor) -> Tensor:
        """Reward callback reproducing the reference's batched ignore-Z behavior."""
        return shadowhand_sparse_reward(
            achieved_goals,
            desired_goals,
            distance_threshold=self.config.distance_threshold,
            rotation_threshold=self.config.rotation_threshold,
            ignore_position=self.config.target_position == "ignore",
            ignore_z_rotation=self.config.ignore_z_rotation,
            legacy_ignore_z_batch_semantics=True,
        )

    def _sample_xyz_quaternion(self, mask: Tensor) -> Tensor:
        angle = self.rng.uniform(1, mask=mask, low=-math.pi, high=math.pi)[:, 0]
        axis = self.rng.uniform(3, mask=mask, low=-1.0, high=1.0)
        axis = axis / torch.linalg.vector_norm(axis, dim=1, keepdim=True).clamp_min(1e-12)
        half_angle = 0.5 * angle
        return torch.cat(
            (torch.cos(half_angle).unsqueeze(1), torch.sin(half_angle).unsqueeze(1) * axis),
            dim=1,
        )

    def _randomized_initial_qpos(self, mask: Tensor) -> Tensor:
        qpos = self.initial_qpos.unsqueeze(0).expand(self.worlds, -1).clone()
        start = self.layout.object_qpos_start
        qpos[:, start : start + 3] += 0.005 * self.rng.normal(3, mask=mask)
        base_quaternion = qpos[:, start + 3 : start + 7]
        randomized = _quaternion_multiply(base_quaternion, self._sample_xyz_quaternion(mask))
        qpos[:, start + 3 : start + 7] = randomized / torch.linalg.vector_norm(
            randomized, dim=1, keepdim=True
        ).clamp_min(1e-12)
        return qpos

    def _sample_goals(self, mask: Tensor) -> Tensor:
        achieved = self.backend.qpos[
            :, self.layout.object_qpos_start : self.layout.object_qpos_start + 7
        ]
        goals = achieved.clone()
        if self.config.target_position == "random":
            uniforms = self.rng.uniform(3, mask=mask)
            ranges = torch.tensor(
                self._TARGET_POSITION_RANGE, dtype=self.dtype, device=self.device
            )
            goals[:, :3] += ranges[:, 0] + uniforms * (ranges[:, 1] - ranges[:, 0])
        goals[:, 3:] = self._sample_xyz_quaternion(mask)
        goals[:, 3:] /= torch.linalg.vector_norm(
            goals[:, 3:], dim=1, keepdim=True
        ).clamp_min(1e-12)
        return goals

    def _state_snapshot(self) -> dict[str, Tensor]:
        return {
            "qpos": self.backend.qpos.clone(),
            "qvel": self.backend.qvel.clone(),
            "ctrl": self.backend.ctrl.clone(),
            "time": self.backend.time.clone(),
            "qacc_warmstart": self.backend.qacc_warmstart.clone(),
            "sensordata": self.backend.sensordata.clone(),
            "constraint_counts": self.backend.constraint_counts.clone(),
        }

    def _restore_protected(self, snapshot: Mapping[str, Tensor], protected: Tensor) -> None:
        for name, values in snapshot.items():
            target = getattr(self.backend, name)
            target[protected] = values[protected]

    @torch.no_grad()
    def reset(self, mask: Tensor | None = None) -> dict[str, Tensor]:
        """Randomize and settle selected worlds without advancing protected worlds."""
        if mask is None:
            mask = torch.ones(self.worlds, dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(device=self.device, dtype=torch.bool).reshape(self.worlds)
        pending = mask.clone()
        settle_physics_steps = (
            self.config.base_reset_settle_policy_steps
            + self.config.additional_reset_settle_policy_steps
        ) * self.config.physics_steps_per_action

        for _ in range(self.config.max_reset_attempts):
            snapshot = self._state_snapshot()
            protected = ~pending
            randomized_qpos = self._randomized_initial_qpos(pending)
            self.backend.qpos[pending] = randomized_qpos[pending]
            self.backend.qvel[pending] = 0.0
            self.backend.qacc_warmstart[pending] = 0.0
            self.backend.time[pending] = 0.0
            self.backend.ctrl[pending] = self.control_centers
            self.backend.step(settle_physics_steps)
            self._restore_protected(snapshot, protected)
            self.backend.synchronize()

            object_height = self.backend.qpos[:, self.layout.object_qpos_start + 2]
            pending = mask & (object_height <= 0.04)
            if not bool(pending.any().item()):
                break
        else:
            raise RuntimeError(
                f"failed to produce a valid on-palm reset in {self.config.max_reset_attempts} attempts"
            )

        sampled_goals = self._sample_goals(mask)
        self.goals[mask] = sampled_goals[mask]
        self.last_actions[mask] = 0.0
        self.episode_steps[mask] = 0
        self.episode_ids[mask] += 1
        return self.observations()

    @torch.no_grad()
    def step(self, actions: Tensor) -> ShadowHandTaskStep:
        conditioned = self.apply_actions(actions)
        self.backend.step(self.config.physics_steps_per_action)
        self.episode_steps.add_(1)
        observations = self.observations()
        rewards = self.compute_rewards(
            observations["achieved_goal"], observations["desired_goal"]
        )
        success = rewards + 1.0
        truncated = self.episode_steps >= self.config.max_episode_steps
        terminated = torch.zeros_like(truncated)
        return ShadowHandTaskStep(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            success=success,
            conditioned_actions=conditioned,
        )

    def report(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "layout": self.layout.to_dict(),
            "worlds": self.worlds,
            "observation_width": self.observation_width,
            "backend": self.backend.report(),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "qpos": self.backend.qpos.clone(),
            "qvel": self.backend.qvel.clone(),
            "ctrl": self.backend.ctrl.clone(),
            "time": self.backend.time.clone(),
            "qacc_warmstart": self.backend.qacc_warmstart.clone(),
            "sensordata": self.backend.sensordata.clone(),
            "goals": self.goals.clone(),
            "last_actions": self.last_actions.clone(),
            "episode_steps": self.episode_steps.clone(),
            "episode_ids": self.episode_ids.clone(),
            "rng_state": self.rng.state.clone(),
        }

    @torch.no_grad()
    def load_checkpoint(self, checkpoint: Mapping[str, Tensor]) -> None:
        for name in ("qpos", "qvel", "ctrl", "time", "qacc_warmstart", "sensordata"):
            getattr(self.backend, name).copy_(checkpoint[name])
        for name in ("goals", "last_actions", "episode_steps", "episode_ids"):
            getattr(self, name).copy_(checkpoint[name])
        self.rng.state.copy_(checkpoint["rng_state"])
