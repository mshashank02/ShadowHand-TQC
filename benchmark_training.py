#!/usr/bin/env python3
"""Benchmark the complete direct-MJWarp + CUDA HER + TQC training loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
from typing import Any

import torch

from shadowhand_gpu.rl.normalization import CudaVecNormalize
from shadowhand_gpu.rl.replay import CudaHERReplayBuffer
from shadowhand_gpu.rl.tqc import TQCConfig, TQCLearner
from shadowhand_gpu.task import ShadowHandTaskConfig, ShadowHandWarpTask
from shadowhand_gpu.trainer import (
    CudaTQCTrainer,
    TrainerConfig,
    reference_gradient_steps,
    seed_torch,
)
from shadowhand_gpu.warp_backend import MujocoWarpBackend


def _positive_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def benchmark_one(
    *,
    xml_path: Path,
    worlds: int,
    device: str,
    seed: int,
    batch_size: int,
    hidden_dims: tuple[int, ...],
    horizon: int,
    profile_steps: int,
    measured_steps: int,
    contacts_per_world: int,
    constraints_per_world: int,
    use_cuda_graphs: bool,
    gradient_steps: int,
) -> dict[str, Any]:
    seed_torch(seed)
    backend = MujocoWarpBackend(
        xml_path,
        worlds=worlds,
        device=device,
        contacts_per_world=contacts_per_world,
        constraints_per_world=constraints_per_world,
        use_cuda_graphs=use_cuda_graphs,
    )
    task = ShadowHandWarpTask(
        backend,
        config=ShadowHandTaskConfig(max_episode_steps=horizon),
        seed=seed,
    )
    observation_shapes = {
        "achieved_goal": (7,),
        "desired_goal": (7,),
        "observation": (task.observation_width,),
    }
    normalizer = CudaVecNormalize(
        observation_shapes,
        num_envs=worlds,
        gamma=0.95,
        device=device,
    )
    learner = TQCLearner(
        TQCConfig(
            observation_dim=task.observation_width + 14,
            action_dim=20,
            hidden_dims=hidden_dims,
            n_critics=2,
            n_quantiles=25,
            top_quantiles_to_drop_per_critic=2,
            gamma=0.95,
            tau=0.05,
            learning_rate=1e-3,
            target_entropy=-20.0,
            device=device,
        )
    )
    minimum_capacity = horizon * worlds
    replay_capacity = max(2 * minimum_capacity, batch_size)
    replay = CudaHERReplayBuffer(
        requested_capacity=replay_capacity,
        num_envs=worlds,
        observation_shapes=observation_shapes,
        action_dim=20,
        max_episode_steps=horizon,
        n_sampled_goal=4,
        reward_function=task.compute_her_rewards,
        device=device,
    )
    trainer = CudaTQCTrainer(
        task=task,
        normalizer=normalizer,
        replay=replay,
        learner=learner,
        config=TrainerConfig(
            batch_size=batch_size,
            learning_starts=0,
            gradient_steps=gradient_steps,
        ),
        seed=seed,
    )
    max_contacts = torch.zeros((), dtype=torch.int32, device=device)
    max_collisions = torch.zeros((), dtype=torch.int32, device=device)
    max_constraints = torch.zeros((), dtype=torch.int32, device=device)
    overflow_flags = torch.zeros((), dtype=torch.int32, device=device)

    @torch.no_grad()
    def record_capacity() -> None:
        max_contacts.copy_(torch.maximum(max_contacts, backend.active_contact_counts.max()))
        max_collisions.copy_(torch.maximum(max_collisions, backend.collision_counts.max()))
        max_constraints.copy_(torch.maximum(max_constraints, backend.constraint_counts.max()))
        overflow_flags.copy_(torch.bitwise_or(overflow_flags, backend.overflow_flags.max()))

    reset_started = time.perf_counter()
    trainer.initialize()
    backend.synchronize()
    reset_seconds = time.perf_counter() - reset_started

    warmup_started = time.perf_counter()
    while trainer.replay_completed_episodes == 0:
        trainer.collect_and_update()
        record_capacity()
    # Warm all sample/learner kernels outside both profiling and throughput windows.
    trainer.collect_and_update()
    record_capacity()
    backend.synchronize()
    warmup_seconds = time.perf_counter() - warmup_started

    phase_samples: dict[str, list[float]] = {}
    for _ in range(profile_steps):
        step = trainer.collect_and_update(profile=True)
        record_capacity()
        for name, milliseconds in (step.phase_timings_ms or {}).items():
            phase_samples.setdefault(name, []).append(milliseconds)
    phase_mean_ms = {
        name: sum(values) / len(values) for name, values in phase_samples.items()
    }

    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    updates_before = trainer.gradient_updates
    backend.synchronize()
    measured_started = time.perf_counter()
    for _ in range(measured_steps):
        trainer.collect_and_update()
        record_capacity()
    backend.synchronize()
    measured_seconds = time.perf_counter() - measured_started
    transitions = measured_steps * worlds
    update_count = trainer.gradient_updates - updates_before
    capacity_values = torch.stack(
        (max_contacts, max_collisions, max_constraints, overflow_flags)
    ).cpu().tolist()
    capacity_safe = (
        int(capacity_values[3]) == 0
        and int(capacity_values[0]) <= worlds * contacts_per_world
        and int(capacity_values[2]) <= constraints_per_world
    )
    result = {
        "ok": capacity_safe,
        "worlds": worlds,
        "tactile_dimension": len(backend.sensor_layout.touch_data_indices),
        "raw_observation_dimension": task.observation_width,
        "policy_input_dimension": task.observation_width + 14,
        "batch_size": batch_size,
        "gradient_steps_per_vector_step": gradient_steps,
        "target_transitions_per_gradient_update": worlds / gradient_steps,
        "hidden_dims": list(hidden_dims),
        "episode_horizon": horizon,
        "physics_steps_per_action": task.config.physics_steps_per_action,
        "profile_steps": profile_steps,
        "measured_steps": measured_steps,
        "reset_seconds": reset_seconds,
        "warmup_complete_episode_seconds": warmup_seconds,
        "warmup_transitions": (horizon + 1) * worlds,
        "measured_seconds": measured_seconds,
        "transitions_per_second": transitions / measured_seconds,
        "environment_steps_per_second": transitions / measured_seconds,
        "physics_world_steps_per_second": (
            transitions * task.config.physics_steps_per_action / measured_seconds
        ),
        "gradient_updates_per_second": update_count / measured_seconds,
        "seconds_per_100k_transitions": measured_seconds * 100_000 / transitions,
        "phase_mean_milliseconds": phase_mean_ms,
        "replay": replay.plan.to_dict(),
        "allocated_before_measurement_bytes": allocated_before,
        "reserved_before_measurement_bytes": reserved_before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "capacity_high_water": {
            "batch_global_active_contacts": int(capacity_values[0]),
            "batch_global_collision_candidates": int(capacity_values[1]),
            "max_constraints_per_world": int(capacity_values[2]),
            "overflow_flags": int(capacity_values[3]),
            "contacts_per_world": contacts_per_world,
            "constraints_per_world": constraints_per_world,
        },
        "backend": backend.report(),
    }
    if not capacity_safe:
        result["error"] = "MuJoCo Warp capacity high-water check failed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--worlds", type=_positive_ints, default=[64])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--arch", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument(
        "--gradient-steps",
        type=int,
        help=(
            "Explicit updates per batched policy step. Default rounds worlds/6 to "
            "match the six-env SB3 transition/update ratio."
        ),
    )
    parser.add_argument("--contacts-per-world", type=int, default=2048)
    parser.add_argument("--constraints-per-world", type=int, default=2048)
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.gradient_steps is not None and args.gradient_steps < 1:
        parser.error("--gradient-steps must be positive")
    for name in (
        "batch_size",
        "horizon",
        "profile_steps",
        "measured_steps",
        "contacts_per_world",
        "constraints_per_world",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.xml = args.xml.expanduser().resolve()
    if not args.xml.is_file():
        parser.error(f"XML does not exist: {args.xml}")

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "xml": str(args.xml),
        "device": args.device,
        "update_ratio_basis": (
            "one gradient update per 6 collected transitions (SB3 six-env reference)"
            if args.gradient_steps is None
            else "explicit gradient_steps override"
        ),
        "results": [],
    }
    for worlds in args.worlds:
        try:
            result = benchmark_one(
                xml_path=args.xml,
                worlds=worlds,
                device=args.device,
                seed=args.seed,
                batch_size=args.batch_size,
                hidden_dims=tuple(args.arch),
                horizon=args.horizon,
                profile_steps=args.profile_steps,
                measured_steps=args.measured_steps,
                contacts_per_world=args.contacts_per_world,
                constraints_per_world=args.constraints_per_world,
                use_cuda_graphs=not args.no_cuda_graphs,
                gradient_steps=(
                    reference_gradient_steps(worlds)
                    if args.gradient_steps is None
                    else args.gradient_steps
                ),
            )
        except Exception as exc:
            result = {
                "ok": False,
                "worlds": worlds,
                "error": f"{type(exc).__name__}: {exc}",
            }
        report["results"].append(result)
        gc.collect()
        torch.cuda.empty_cache()
        if not result["ok"]:
            break
    successful = [result for result in report["results"] if result["ok"]]
    if successful:
        recommended = max(successful, key=lambda result: result["transitions_per_second"])
        report["recommended_num_envs"] = recommended["worlds"]
        report["recommendation_basis"] = "maximum measured complete-loop transitions_per_second"

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if len(successful) == len(report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
