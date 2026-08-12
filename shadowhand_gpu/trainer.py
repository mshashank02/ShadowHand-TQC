"""Explicit CUDA-resident rollout, HER, and TQC training orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch
from torch import Tensor

from .rl.normalization import CudaVecNormalize, flatten_observations
from .rl.replay import CudaHERReplayBuffer
from .rl.tqc import TQCLearner
from .task import PerWorldRandom, ShadowHandWarpTask


CHECKPOINT_VERSION = 1
REFERENCE_TRANSITIONS_PER_GRADIENT_UPDATE = 6


@dataclass(frozen=True)
class TrainerConfig:
    batch_size: int = 2048
    learning_starts: int = 8000
    gradient_steps: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.gradient_steps < 1:
            raise ValueError("batch_size and gradient_steps must be positive")
        if self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")


@dataclass(frozen=True)
class TrainerStep:
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    update_metrics: dict[str, Tensor | int] | None
    global_steps: int
    gradient_updates: int
    phase_timings_ms: dict[str, float] | None = None


@dataclass(frozen=True)
class EvaluationResult:
    timestep: int
    episodes: int
    mean_reward: float
    std_reward: float
    success_rate: float
    elapsed_seconds: float

    def to_metrics(self) -> dict[str, float | int]:
        return {
            "training_step": self.timestep,
            "eval/episodes": self.episodes,
            "eval/mean_reward": self.mean_reward,
            "eval/std_reward": self.std_reward,
            "eval/success_rate": self.success_rate,
            "eval/elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class AutoTuneRecommendation:
    """A measured-safe complete-loop backend allocation."""

    num_envs: int
    contacts_per_world: int
    constraints_per_world: int


def reference_gradient_steps(num_envs: int) -> int:
    """Match the reference six-env SB3 rate of one update per vector step."""
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    return max(
        1,
        (int(num_envs) + REFERENCE_TRANSITIONS_PER_GRADIENT_UPDATE // 2)
        // REFERENCE_TRANSITIONS_PER_GRADIENT_UPDATE,
    )


def seed_torch(seed: int) -> None:
    """Seed network, action, replay, and learner randomness."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class CudaTQCTrainer:
    """Connect one batched task directly to CUDA normalization, HER, and TQC.

    The hot path contains no NumPy conversion, per-world Python loop, or routine
    device synchronization. The only host-side episode decision is a scalar horizon
    counter because the current task has timeout-only, synchronized episodes.
    """

    def __init__(
        self,
        *,
        task: ShadowHandWarpTask,
        normalizer: CudaVecNormalize,
        replay: CudaHERReplayBuffer,
        learner: TQCLearner,
        config: TrainerConfig = TrainerConfig(),
        seed: int = 0,
    ) -> None:
        if normalizer.num_envs != task.worlds or replay.num_envs != task.worlds:
            raise ValueError("task, normalizer, and replay num_envs must match")
        if learner.config.action_dim != int(task.backend.model.nu):
            raise ValueError("learner action dimension does not match the task")
        expected_policy_width = task.observation_width + 14
        if learner.config.observation_dim != expected_policy_width:
            raise ValueError(
                "learner observation dimension does not match the task policy input: "
                f"{learner.config.observation_dim} != {expected_policy_width}"
            )
        if config.batch_size > replay.plan.effective_capacity:
            raise ValueError("training batch size exceeds replay capacity")
        self.task = task
        self.normalizer = normalizer
        self.replay = replay
        self.learner = learner
        self.config = config
        self.seed = int(seed)
        self.device = task.device
        self.global_steps = 0
        self.vector_steps = 0
        self.gradient_updates = int(learner.updates)
        self.completed_episodes = 0
        self.replay_completed_episodes = 0
        self.rollout_episode_step = 0
        self.learning_block_until = int(config.learning_starts)
        self.current_observations: dict[str, Tensor] | None = None
        self.normalized_observations: dict[str, Tensor] | None = None
        self.episode_returns = torch.zeros(task.worlds, dtype=torch.float32, device=self.device)
        self.recent_return_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self.recent_success_sum = torch.zeros((), dtype=torch.float32, device=self.device)
        self.recent_episode_count = 0
        self.started_at = time.perf_counter()

    @property
    def initialized(self) -> bool:
        return self.current_observations is not None

    @torch.no_grad()
    def initialize(self) -> dict[str, Tensor]:
        if self.initialized:
            raise RuntimeError("trainer is already initialized")
        observations = self.task.reset()
        self.current_observations = observations
        self.normalized_observations = self.normalizer.reset(observations)
        return observations

    @torch.no_grad()
    def _select_actions(self) -> Tensor:
        if self.normalized_observations is None:
            raise RuntimeError("trainer must be initialized before collecting steps")
        if self.global_steps < self.config.learning_starts:
            return torch.rand(
                self.task.worlds,
                self.learner.config.action_dim,
                dtype=torch.float32,
                device=self.device,
            ).mul_(2.0).sub_(1.0)
        policy_input = flatten_observations(self.normalized_observations)
        return self.learner.actor(policy_input, deterministic=False)

    def _ready_to_update(self) -> bool:
        return (
            self.global_steps > max(self.config.learning_starts, self.learning_block_until)
            and self.replay_completed_episodes > 0
        )

    def collect_and_update(self, *, profile: bool = False) -> TrainerStep:
        """Collect one transition row and perform the configured learner updates."""
        if self.current_observations is None:
            self.initialize()
        assert self.current_observations is not None

        timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        active_timing_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

        def begin(name: str) -> None:
            if profile and self.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                active_timing_events[name] = (start, stop)

        def end(name: str) -> None:
            event_pair = active_timing_events.pop(name, None)
            if event_pair is not None:
                event_pair[1].record()
                timing_events.setdefault(name, []).append(event_pair)

        begin("complete_step")
        begin("policy_inference")
        actions = self._select_actions()
        end("policy_inference")
        begin("simulation_task")
        task_step = self.task.step(actions)
        end("simulation_task")
        self.rollout_episode_step += 1
        at_horizon = self.rollout_episode_step >= self.task.config.max_episode_steps
        dones = task_step.terminated | task_step.truncated

        # Store terminal observations before a reset mutates qpos-backed goal views.
        begin("replay_insert")
        self.replay.add(
            observations=self.current_observations,
            next_observations=task_step.observations,
            actions=actions,
            rewards=task_step.rewards,
            dones=dones,
            timeouts=task_step.truncated,
        )
        end("replay_insert")
        self.episode_returns.add_(task_step.rewards)

        begin("reset_and_normalization")
        if at_horizon:
            # The production task has no early termination. Every world starts at
            # the same horizon offset, so this branch needs no CUDA scalar read.
            next_observations = self.task.reset()
            self.completed_episodes += self.task.worlds
            self.replay_completed_episodes += self.task.worlds
            self.recent_episode_count += self.task.worlds
            self.recent_return_sum.add_(self.episode_returns.sum())
            self.recent_success_sum.add_(task_step.success.sum())
            self.episode_returns.zero_()
            self.rollout_episode_step = 0
        else:
            next_observations = task_step.observations

        self.current_observations = next_observations
        self.normalized_observations, _ = self.normalizer.step(
            next_observations,
            task_step.rewards,
            dones,
        )
        end("reset_and_normalization")
        self.vector_steps += 1
        self.global_steps += self.task.worlds

        update_metrics: dict[str, Tensor | int] | None = None
        if self._ready_to_update():
            for _ in range(self.config.gradient_steps):
                begin("her_sample")
                sample = self.replay.sample(
                    self.config.batch_size,
                    normalizer=self.normalizer,
                )
                end("her_sample")
                begin("tqc_update")
                update_metrics = self.learner.update(sample.to_tqc_batch())
                end("tqc_update")
            self.gradient_updates = int(self.learner.updates)

        end("complete_step")
        phase_timings_ms: dict[str, float] | None = None
        if timing_events:
            self.task.backend.synchronize()
            phase_timings_ms = {
                name: sum(float(start.elapsed_time(stop)) for start, stop in pairs)
                for name, pairs in timing_events.items()
            }

        return TrainerStep(
            actions=actions,
            rewards=task_step.rewards,
            dones=dones,
            update_metrics=update_metrics,
            global_steps=self.global_steps,
            gradient_updates=self.gradient_updates,
            phase_timings_ms=phase_timings_ms,
        )

    def train_until(self, total_timesteps: int) -> None:
        if total_timesteps < self.global_steps:
            raise ValueError("total_timesteps is behind the restored global step")
        while self.global_steps < total_timesteps:
            self.collect_and_update()

    @torch.no_grad()
    def evaluate(self, episodes: int, *, seed: int | None = None) -> EvaluationResult:
        """Run fixed-seed deterministic evaluation using the existing batched worlds."""
        if episodes < 1:
            raise ValueError("evaluation episodes must be positive")
        if not self.initialized:
            self.initialize()
        training_state = self.task.checkpoint()
        evaluation_seed = self.seed + 1000 if seed is None else int(seed)
        self.task.rng = PerWorldRandom(self.task.worlds, evaluation_seed, self.device)
        reward_batches: list[Tensor] = []
        success_batches: list[Tensor] = []
        remaining = int(episodes)
        started = time.perf_counter()
        try:
            while remaining > 0:
                observations = self.task.reset()
                returns = torch.zeros(
                    self.task.worlds,
                    dtype=torch.float32,
                    device=self.device,
                )
                final_success = torch.zeros_like(returns)
                for _ in range(self.task.config.max_episode_steps):
                    normalized = self.normalizer.normalize_observations(observations)
                    actions = self.learner.actor(
                        flatten_observations(normalized),
                        deterministic=True,
                    )
                    task_step = self.task.step(actions)
                    returns.add_(task_step.rewards)
                    final_success = task_step.success
                    observations = task_step.observations
                take = min(remaining, self.task.worlds)
                reward_batches.append(returns[:take])
                success_batches.append(final_success[:take])
                remaining -= take
            self.task.backend.synchronize()
            rewards = torch.cat(reward_batches)
            successes = torch.cat(success_batches)
            values = torch.stack(
                (rewards.mean(), rewards.std(correction=0), successes.mean())
            ).cpu().tolist()
        finally:
            self.task.load_checkpoint(training_state)
            self.task.backend.synchronize()
        return EvaluationResult(
            timestep=self.global_steps,
            episodes=int(episodes),
            mean_reward=float(values[0]),
            std_reward=float(values[1]),
            success_rate=float(values[2]),
            elapsed_seconds=time.perf_counter() - started,
        )

    def low_frequency_metrics(self, *, reset_episode_window: bool = False) -> dict[str, Any]:
        """Synchronize once and transfer logging scalars at an explicit boundary."""
        self.task.backend.synchronize()
        elapsed = max(time.perf_counter() - self.started_at, 1e-12)
        metrics: dict[str, Any] = {
            "training_step": self.global_steps,
            "train/vector_steps": self.vector_steps,
            "train/gradient_updates": self.gradient_updates,
            "train/completed_episodes": self.completed_episodes,
            "performance/wall_clock_elapsed": elapsed,
            "performance/transitions_per_second": self.global_steps / elapsed,
            "performance/environment_steps_per_second": self.global_steps / elapsed,
            "performance/gradient_updates_per_second": self.gradient_updates / elapsed,
            "performance/physics_steps_per_second": (
                self.global_steps * self.task.config.physics_steps_per_action / elapsed
            ),
            "gpu/memory_allocated_bytes": (
                torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
            ),
            "gpu/memory_reserved_bytes": (
                torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0
            ),
        }
        if self.recent_episode_count:
            summary = torch.stack((self.recent_return_sum, self.recent_success_sum)).cpu()
            metrics["rollout/mean_episode_reward"] = float(
                summary[0] / self.recent_episode_count
            )
            metrics["rollout/success_rate"] = float(summary[1] / self.recent_episode_count)
        if reset_episode_window:
            self.recent_return_sum.zero_()
            self.recent_success_sum.zero_()
            self.recent_episode_count = 0
        return metrics

    def checkpoint(self, *, include_replay: bool = True) -> dict[str, Any]:
        if not self.initialized:
            self.initialize()
        state: dict[str, Any] = {
            "version": CHECKPOINT_VERSION,
            "trainer_config": asdict(self.config),
            "seed": self.seed,
            "global_steps": self.global_steps,
            "vector_steps": self.vector_steps,
            "gradient_updates": self.gradient_updates,
            "completed_episodes": self.completed_episodes,
            "replay_completed_episodes": self.replay_completed_episodes,
            "rollout_episode_step": self.rollout_episode_step,
            "learning_block_until": self.learning_block_until,
            "episode_returns": self.episode_returns,
            "recent_return_sum": self.recent_return_sum,
            "recent_success_sum": self.recent_success_sum,
            "recent_episode_count": self.recent_episode_count,
            "task": self.task.checkpoint(),
            "normalizer": self.normalizer.checkpoint(),
            "learner": self.learner.checkpoint(),
            "replay_included": bool(include_replay),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
            ),
        }
        if include_replay:
            state["replay"] = self.replay.checkpoint()
        return state

    def load_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        *,
        resume_warmup_steps: int | None = None,
    ) -> bool:
        """Restore state and return whether replay history was restored."""
        if int(checkpoint["version"]) != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported trainer checkpoint version {checkpoint['version']}")
        checkpoint_config = TrainerConfig(**checkpoint["trainer_config"])
        if checkpoint_config != self.config:
            raise ValueError("trainer configuration does not match checkpoint")
        self.normalizer.load_checkpoint(checkpoint["normalizer"])
        self.learner.load_checkpoint(dict(checkpoint["learner"]))
        self.global_steps = int(checkpoint["global_steps"])
        self.vector_steps = int(checkpoint["vector_steps"])
        self.gradient_updates = int(checkpoint["gradient_updates"])
        self.completed_episodes = int(checkpoint["completed_episodes"])
        self.recent_return_sum.copy_(checkpoint["recent_return_sum"])
        self.recent_success_sum.copy_(checkpoint["recent_success_sum"])
        self.recent_episode_count = int(checkpoint["recent_episode_count"])

        replay_restored = bool(checkpoint.get("replay_included", False))
        if replay_restored:
            self.replay.load_checkpoint(checkpoint["replay"])
            self.task.load_checkpoint(checkpoint["task"])
            self.replay_completed_episodes = int(checkpoint["replay_completed_episodes"])
            self.rollout_episode_step = int(checkpoint["rollout_episode_step"])
            self.learning_block_until = int(checkpoint["learning_block_until"])
            self.episode_returns.copy_(checkpoint["episode_returns"])
            observations = self.task.observations()
            self.current_observations = observations
            self.normalized_observations = self.normalizer.normalize_observations(observations)
        else:
            self.replay.clear()
            observations = self.task.reset()
            self.current_observations = observations
            self.normalized_observations = self.normalizer.reset(observations)
            self.replay_completed_episodes = 0
            self.rollout_episode_step = 0
            self.episode_returns.zero_()
            minimum_warmup = self.task.config.max_episode_steps * self.task.worlds
            requested_warmup = (
                self.config.learning_starts
                if resume_warmup_steps is None
                else int(resume_warmup_steps)
            )
            if requested_warmup < 0:
                raise ValueError("resume_warmup_steps cannot be negative")
            self.learning_block_until = self.global_steps + max(
                requested_warmup,
                minimum_warmup,
            )

        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if self.device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), self.device)
        self.started_at = time.perf_counter()
        return replay_restored

    def save_checkpoint(self, path: str | Path, *, include_replay: bool = True) -> Path:
        """Synchronize and atomically serialize a complete training checkpoint."""
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.tmp-{os.getpid()}"
        )
        self.task.backend.synchronize()
        try:
            torch.save(self.checkpoint(include_replay=include_replay), temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def load_checkpoint_file(
        self,
        path: str | Path,
        *,
        resume_warmup_steps: int | None = None,
    ) -> bool:
        checkpoint = torch.load(
            Path(path).expanduser().resolve(),
            map_location=self.device,
            weights_only=False,
        )
        return self.load_checkpoint(
            checkpoint,
            resume_warmup_steps=resume_warmup_steps,
        )


def study_metrics_payload(
    *,
    task_name: str,
    total_timesteps: int,
    checkpoint_steps: list[int],
    success_curve: list[float],
    seed: int,
    object_id: str | None = None,
    candidate_id: str | None = None,
    physics_mode: str | None = None,
    backend: str = "mujoco_warp",
) -> dict[str, Any]:
    if not checkpoint_steps or len(checkpoint_steps) != len(success_curve):
        raise ValueError("checkpoint steps and success curve must be non-empty and aligned")
    denominator = max(int(total_timesteps), 1)
    payload: dict[str, Any] = {
        "tasks": [task_name],
        "checkpoints": [min(1.0, int(step) / denominator) for step in checkpoint_steps],
        "success": {task_name: [float(value) for value in success_curve]},
        "final_success": {task_name: float(success_curve[-1])},
        "seed": int(seed),
        "backend": backend,
    }
    if object_id is not None:
        payload["object_id"] = object_id
    if candidate_id is not None:
        payload["candidate_id"] = candidate_id
    if physics_mode is not None:
        payload["physics_mode"] = physics_mode
    return payload


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    import json

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_complete_loop_recommendation(
    path: str | Path,
    *,
    xml_path: str | Path | None = None,
) -> AutoTuneRecommendation:
    """Load a measured-safe complete-loop allocation for production auto-tuning."""
    import json

    report_path = Path(path).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("recommendation_basis") != (
        "maximum measured complete-loop transitions_per_second"
    ):
        raise ValueError("auto-num-envs report was not produced by the complete-loop benchmark")
    if report.get("update_ratio_basis") != (
        "one gradient update per 6 collected transitions (SB3 six-env reference)"
    ):
        raise ValueError("auto-num-envs report does not preserve the reference update ratio")
    if xml_path is not None:
        expected_xml = Path(xml_path).expanduser().resolve()
        measured_xml = Path(report["xml"]).expanduser().resolve()
        if measured_xml != expected_xml:
            raise ValueError(
                f"auto-num-envs report XML {measured_xml} does not match training XML {expected_xml}"
            )
    recommendation = int(report["recommended_num_envs"])
    successful_results = {
        int(result["worlds"]): result
        for result in report.get("results", ())
        if bool(result.get("ok", False))
    }
    if recommendation < 1 or recommendation not in successful_results:
        raise ValueError("auto-num-envs recommendation is not a successful measured world count")
    selected = successful_results[recommendation]
    high_water = selected.get("capacity_high_water")
    if not isinstance(high_water, Mapping):
        raise ValueError("recommended result has no capacity high-water measurements")
    if int(high_water.get("overflow_flags", -1)) != 0:
        raise ValueError("recommended result reported a MuJoCo Warp capacity overflow")

    contacts_per_world = int(high_water["contacts_per_world"])
    constraints_per_world = int(high_water["constraints_per_world"])
    if contacts_per_world < 1 or constraints_per_world < 1:
        raise ValueError("recommended backend capacities must be positive")
    active_contacts = int(high_water.get("batch_global_active_contacts", -1))
    max_constraints = int(high_water.get("max_constraints_per_world", -1))
    if active_contacts < 0 or active_contacts > recommendation * contacts_per_world:
        raise ValueError("recommended contact capacity is inconsistent with its high-water mark")
    if max_constraints < 0 or max_constraints > constraints_per_world:
        raise ValueError("recommended constraint capacity is inconsistent with its high-water mark")
    return AutoTuneRecommendation(
        num_envs=recommendation,
        contacts_per_world=contacts_per_world,
        constraints_per_world=constraints_per_world,
    )


def load_num_envs_recommendation(
    path: str | Path,
    *,
    xml_path: str | Path | None = None,
) -> int:
    """Compatibility helper returning only the measured-safe world count."""
    return load_complete_loop_recommendation(path, xml_path=xml_path).num_envs
