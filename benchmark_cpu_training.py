#!/usr/bin/env python3
"""Benchmark the current CPU MuJoCo + SB3 HER + CUDA TQC reference loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch
from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ShadowHand_TQC import make_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--arch", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("num_envs", "batch_size", "horizon", "measured_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.xml = args.xml.expanduser().resolve()
    if not args.xml.is_file():
        parser.error(f"XML does not exist: {args.xml}")

    env_fns = [
        make_env(
            str(args.xml),
            args.seed,
            rank,
            "random",
            "xyz",
            False,
            max_steps=args.horizon,
        )
        for rank in range(args.num_envs)
    ]
    raw_env = DummyVecEnv(env_fns) if args.num_envs == 1 else SubprocVecEnv(
        env_fns,
        start_method="spawn",
    )
    env = VecNormalize(raw_env, gamma=0.95)
    buffer_size = max(2 * args.horizon * args.num_envs, args.batch_size)
    model = TQC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "goal_selection_strategy": "future",
            "n_sampled_goal": 4,
        },
        buffer_size=buffer_size,
        batch_size=args.batch_size,
        gamma=0.95,
        learning_rate=1e-3,
        learning_starts=args.horizon * args.num_envs,
        tau=0.05,
        ent_coef="auto",
        policy_kwargs={"net_arch": args.arch, "n_critics": 2},
        seed=args.seed,
        device=args.device,
        verbose=0,
    )
    try:
        warmup_started = time.perf_counter()
        model.learn(total_timesteps=args.horizon * args.num_envs)
        model.learning_starts = 0
        model.learn(total_timesteps=args.num_envs, reset_num_timesteps=False)
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warmup_started

        updates_before = int(model._n_updates)
        if torch.device(args.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize()
        measured_started = time.perf_counter()
        model.learn(
            total_timesteps=args.measured_steps * args.num_envs,
            reset_num_timesteps=False,
        )
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize()
        measured_seconds = time.perf_counter() - measured_started
        transitions = args.measured_steps * args.num_envs
        updates = int(model._n_updates) - updates_before
        touch_count = int(env.observation_space["observation"].shape[0] - 62)
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "architecture": "cpu_mujoco_vecenv_numpy_her_cuda_sb3_tqc",
            "xml": str(args.xml),
            "num_envs": args.num_envs,
            "tactile_dimension": touch_count,
            "batch_size": args.batch_size,
            "hidden_dims": args.arch,
            "episode_horizon": args.horizon,
            "warmup_seconds": warmup_seconds,
            "measured_steps": args.measured_steps,
            "measured_seconds": measured_seconds,
            "gradient_updates": updates,
            "transitions_per_gradient_update": transitions / updates,
            "transitions_per_second": transitions / measured_seconds,
            "environment_steps_per_second": transitions / measured_seconds,
            "physics_world_steps_per_second": transitions * 20 / measured_seconds,
            "gradient_updates_per_second": updates / measured_seconds,
            "seconds_per_100k_transitions": measured_seconds * 100_000 / transitions,
            "peak_torch_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(args.device)
                if torch.device(args.device).type == "cuda"
                else 0
            ),
        }
    finally:
        env.close()

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
