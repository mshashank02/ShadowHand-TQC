#!/usr/bin/env python3
"""Train ShadowHand TQC/HER entirely through direct MuJoCo Warp and PyTorch CUDA."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Mapping

import torch
from torch import Tensor

from shadowhand_gpu.rl.normalization import CudaVecNormalize
from shadowhand_gpu.rl.replay import CudaHERReplayBuffer
from shadowhand_gpu.rl.tqc import TQCConfig, TQCLearner
from shadowhand_gpu.task import ShadowHandTaskConfig, ShadowHandWarpTask
from shadowhand_gpu.trainer import (
    CudaTQCTrainer,
    TrainerConfig,
    load_complete_loop_recommendation,
    reference_gradient_steps,
    seed_torch,
    study_metrics_payload,
    write_json_atomic,
)
from shadowhand_gpu.warp_backend import MujocoWarpBackend


class _MetricSink:
    def log(self, metrics: Mapping[str, Any]) -> None:
        del metrics

    def finish(self) -> None:
        pass


class _WandbSink(_MetricSink):
    def __init__(self, args: argparse.Namespace, config: Mapping[str, Any]) -> None:
        import wandb

        self._run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            config=dict(config),
            name=args.wandb_name,
            id=args.wandb_id,
            resume=args.wandb_resume,
            mode=args.wandb_mode,
            dir=str(Path(args.artifact_root) / "wandb"),
        )
        self._run.define_metric("training_step")
        self._run.define_metric("eval/*", step_metric="training_step")
        self._run.define_metric("train/*", step_metric="training_step")
        self._run.define_metric("performance/*", step_metric="training_step")

    def log(self, metrics: Mapping[str, Any]) -> None:
        self._run.log(dict(metrics))

    def finish(self) -> None:
        self._run.finish()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-path", type=Path, required=True)
    parser.add_argument("--backend", choices=("mujoco_warp",), default="mujoco_warp")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument(
        "--auto-num-envs",
        action="store_true",
        help="Use the fastest safe world count in a complete-loop benchmark report.",
    )
    parser.add_argument("--auto-num-envs-report", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-id", default="HandManipulateBlockRotateXYZ-v1")
    parser.add_argument("--artifact-root", type=Path, default=Path("."))

    parser.add_argument("--target-position", choices=("random", "ignore"), default="random")
    parser.add_argument("--target-rotation", choices=("xyz",), default="xyz")
    parser.add_argument("--ignore-z-rot", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--action-clip", type=float)
    parser.add_argument("--action-smoothing", type=float, default=0.0)
    parser.add_argument("--reset-settle-steps", type=int, default=0)
    parser.add_argument("--contacts-per-world", type=int, default=8192)
    parser.add_argument("--constraints-per-world", type=int, default=4096)
    parser.add_argument("--no-cuda-graphs", action="store_true")

    parser.add_argument("--n-timesteps", type=float, default=16e6)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-starts", type=int, default=8000)
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help=(
            "Updates per batched policy step. By default this is rounded from "
            "num_envs/6 to preserve the six-env SB3 transition/update ratio."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--ent-coef", default="auto")
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument("--goal-selection-strategy", choices=("future",), default="future")
    parser.add_argument("--arch", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--n-critics", type=int, default=2)
    parser.add_argument("--n-quantiles", type=int, default=25)
    parser.add_argument("--top-quantiles-to-drop-per-critic", type=int, default=2)
    parser.add_argument("--auto-replay-capacity", action="store_true")
    parser.add_argument("--replay-memory-fraction", type=float, default=0.5)
    parser.add_argument("--replay-memory-budget-bytes", type=int)

    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--log-freq", type=int, default=20_000)
    parser.add_argument(
        "--checkpoint-freq",
        "--save-freq",
        dest="checkpoint_freq",
        type=int,
        default=200_000,
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-warmup-steps", type=int)
    parser.add_argument(
        "--save-replay-buffer",
        action="store_true",
        help="Include CUDA replay storage in periodic/final checkpoints.",
    )

    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--task-name")
    parser.add_argument("--object-id")
    parser.add_argument("--candidate-id")
    parser.add_argument("--physics-mode", choices=("rigid", "deformable"), default="rigid")

    parser.add_argument("--wandb-project", default="single_end_to_end_shadowhand")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-id")
    parser.add_argument("--wandb-resume", choices=("allow", "must", "never", "auto"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--disable-eval-video",
        action="store_true",
        help="Accepted for CPU-trainer CLI compatibility; GPU evaluation never records video.",
    )
    args = parser.parse_args()

    args.xml_path = args.xml_path.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
    if args.metrics_json is not None:
        args.metrics_json = args.metrics_json.expanduser().resolve()
    if not args.xml_path.is_file():
        parser.error(f"XML does not exist: {args.xml_path}")
    if args.auto_num_envs:
        if args.auto_num_envs_report is None:
            parser.error("--auto-num-envs requires --auto-num-envs-report from benchmark_training.py")
        try:
            recommendation = load_complete_loop_recommendation(
                args.auto_num_envs_report,
                xml_path=args.xml_path,
            )
            args.num_envs = recommendation.num_envs
            args.contacts_per_world = recommendation.contacts_per_world
            args.constraints_per_world = recommendation.constraints_per_world
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"invalid --auto-num-envs-report: {exc}")
    if args.gradient_steps is None:
        args.gradient_steps = reference_gradient_steps(args.num_envs)
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"checkpoint does not exist: {args.resume}")
    for name in (
        "num_envs",
        "max_episode_steps",
        "buffer_size",
        "batch_size",
        "gradient_steps",
        "eval_episodes",
        "contacts_per_world",
        "constraints_per_world",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("learning_starts", "eval_freq", "log_freq", "checkpoint_freq"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")
    if int(args.n_timesteps) < 1:
        parser.error("--n-timesteps must be positive")
    if args.ent_coef != "auto":
        parser.error("the GPU parity path currently supports the reference --ent-coef auto only")
    if args.physics_mode != "rigid":
        parser.error(
            "the production GPU path supports compiled rigid geoms (including converted "
            "custom mesh geoms); flex/deformable collision models are excluded"
        )
    if args.device == "cuda":
        args.device = "cuda:0"
    if torch.device(args.device).type != "cuda":
        parser.error("direct MuJoCo Warp training requires --device cuda[:index]")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable to PyTorch")
    return args


def _print_json(label: str, payload: Mapping[str, Any]) -> None:
    print(f"{label}: {json.dumps(dict(payload), sort_keys=True)}", flush=True)


def main() -> int:
    args = _parse_args()
    seed_torch(args.seed)
    if args.auto_num_envs:
        print(
            "Auto-selected "
            f"{args.num_envs} worlds, {args.contacts_per_world} contacts/world, and "
            f"{args.constraints_per_world} constraints/world from "
            f"{args.auto_num_envs_report}.",
            flush=True,
        )
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    model_root = args.artifact_root / "models" / args.env_id
    model_root.mkdir(parents=True, exist_ok=True)
    (args.artifact_root / "wandb").mkdir(parents=True, exist_ok=True)

    backend = MujocoWarpBackend(
        args.xml_path,
        worlds=args.num_envs,
        device=args.device,
        contacts_per_world=args.contacts_per_world,
        constraints_per_world=args.constraints_per_world,
        use_cuda_graphs=not args.no_cuda_graphs,
    )
    task = ShadowHandWarpTask(
        backend,
        config=ShadowHandTaskConfig(
            max_episode_steps=args.max_episode_steps,
            additional_reset_settle_policy_steps=args.reset_settle_steps,
            action_scale=args.action_scale,
            action_clip=args.action_clip,
            action_smoothing=args.action_smoothing,
            target_position=args.target_position,
            target_rotation=args.target_rotation,
            ignore_z_rotation=args.ignore_z_rot,
        ),
        seed=args.seed,
    )
    observation_shapes = {
        "achieved_goal": (7,),
        "desired_goal": (7,),
        "observation": (task.observation_width,),
    }
    normalizer = CudaVecNormalize(
        observation_shapes,
        num_envs=args.num_envs,
        gamma=args.gamma,
        device=args.device,
    )
    learner = TQCLearner(
        TQCConfig(
            observation_dim=task.observation_width + 14,
            action_dim=20,
            hidden_dims=tuple(args.arch),
            n_critics=args.n_critics,
            n_quantiles=args.n_quantiles,
            top_quantiles_to_drop_per_critic=args.top_quantiles_to_drop_per_critic,
            gamma=args.gamma,
            tau=args.tau,
            learning_rate=args.learning_rate,
            target_entropy=-20.0,
            device=args.device,
        )
    )
    replay = CudaHERReplayBuffer(
        requested_capacity=args.buffer_size,
        num_envs=args.num_envs,
        observation_shapes=observation_shapes,
        action_dim=20,
        max_episode_steps=args.max_episode_steps,
        n_sampled_goal=args.n_sampled_goal,
        reward_function=task.compute_her_rewards,
        device=args.device,
        memory_budget_bytes=args.replay_memory_budget_bytes,
        memory_fraction=args.replay_memory_fraction,
        auto_capacity=args.auto_replay_capacity,
    )
    trainer = CudaTQCTrainer(
        task=task,
        normalizer=normalizer,
        replay=replay,
        learner=learner,
        config=TrainerConfig(
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            gradient_steps=args.gradient_steps,
        ),
        seed=args.seed,
    )

    resumed_with_replay = None
    if args.resume is not None:
        resumed_with_replay = trainer.load_checkpoint_file(
            args.resume,
            resume_warmup_steps=args.resume_warmup_steps,
        )
        print(
            "Loaded GPU checkpoint "
            f"{args.resume} at {trainer.global_steps:,} transitions "
            f"(replay restored: {resumed_with_replay}).",
            flush=True,
        )
    else:
        trainer.initialize()

    invocation_steps = int(args.n_timesteps)
    target_steps = trainer.global_steps + invocation_steps if args.resume is not None else invocation_steps
    backend_report = backend.report()
    run_config = {
        "env_id": args.env_id,
        "xml_path": str(args.xml_path),
        "backend": args.backend,
        "device": args.device,
        "gpu": backend_report["device_name"],
        "num_envs": args.num_envs,
        "seed": args.seed,
        "candidate_id": args.candidate_id,
        "object_id": args.object_id,
        "physics_mode": args.physics_mode,
        "tactile_dimension": len(backend.sensor_layout.touch_data_indices),
        "observation_dimension": task.observation_width,
        "policy_input_dimension": task.observation_width + 14,
        "target_global_steps": target_steps,
        "task": task.report()["config"],
        "trainer": asdict(trainer.config),
        "transitions_per_gradient_update": args.num_envs / args.gradient_steps,
        "tqc": learner.config.to_dict(),
        "replay": replay.plan.to_dict(),
    }
    _print_json("GPU_RUN_CONFIG", run_config)
    _print_json("REPLAY_MEMORY_PLAN", replay.plan.to_dict())

    sink: _MetricSink
    if args.disable_wandb or args.wandb_mode == "disabled":
        sink = _MetricSink()
    else:
        sink = _WandbSink(args, run_config)

    last_log = trainer.global_steps
    last_eval = trainer.global_steps
    last_checkpoint = trainer.global_steps
    evaluation_steps: list[int] = []
    success_curve: list[float] = []
    latest_update: dict[str, Tensor | int] | None = None
    run_started = time.perf_counter()
    try:
        while trainer.global_steps < target_steps:
            step = trainer.collect_and_update()
            if step.update_metrics is not None:
                latest_update = step.update_metrics

            if args.log_freq and trainer.global_steps - last_log >= args.log_freq:
                metrics = trainer.low_frequency_metrics(reset_episode_window=True)
                if latest_update is not None:
                    for key, value in latest_update.items():
                        metrics[f"train/{key}"] = (
                            float(value.cpu()) if isinstance(value, Tensor) else int(value)
                        )
                _print_json("GPU_TRAIN_METRICS", metrics)
                sink.log(metrics)
                last_log = trainer.global_steps

            if args.eval_freq and trainer.global_steps - last_eval >= args.eval_freq:
                evaluation = trainer.evaluate(args.eval_episodes)
                metrics = evaluation.to_metrics()
                _print_json("GPU_EVAL_METRICS", metrics)
                sink.log(metrics)
                evaluation_steps.append(evaluation.timestep)
                success_curve.append(evaluation.success_rate)
                last_eval = trainer.global_steps

            if (
                args.checkpoint_freq
                and trainer.global_steps - last_checkpoint >= args.checkpoint_freq
            ):
                checkpoint_path = model_root / f"checkpoint_{trainer.global_steps}_steps.pt"
                trainer.save_checkpoint(
                    checkpoint_path,
                    include_replay=args.save_replay_buffer,
                )
                print(f"Saved GPU checkpoint: {checkpoint_path}", flush=True)
                last_checkpoint = trainer.global_steps

        if not evaluation_steps or evaluation_steps[-1] != trainer.global_steps:
            evaluation = trainer.evaluate(args.eval_episodes)
            metrics = evaluation.to_metrics()
            _print_json("GPU_EVAL_METRICS", metrics)
            sink.log(metrics)
            evaluation_steps.append(evaluation.timestep)
            success_curve.append(evaluation.success_rate)

        final_checkpoint = model_root / (
            f"{args.env_id}_{args.num_envs}env_{args.seed}_final.pt"
        )
        trainer.save_checkpoint(
            final_checkpoint,
            include_replay=args.save_replay_buffer,
        )
        print(f"Final GPU checkpoint saved to: {final_checkpoint}", flush=True)

        payload = study_metrics_payload(
            task_name=args.task_name or args.env_id,
            total_timesteps=target_steps,
            checkpoint_steps=evaluation_steps,
            success_curve=success_curve,
            seed=args.seed,
            object_id=args.object_id,
            candidate_id=args.candidate_id,
            physics_mode=args.physics_mode,
            backend=args.backend,
        )
        if args.metrics_json is not None:
            write_json_atomic(args.metrics_json, payload)
            print(f"Study metrics saved to: {args.metrics_json}", flush=True)
        print(f"FINAL_SCORE: {success_curve[-1]:.6f}", flush=True)
        print(
            f"GPU training finished in {time.perf_counter() - run_started:.3f} seconds.",
            flush=True,
        )
    finally:
        sink.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
