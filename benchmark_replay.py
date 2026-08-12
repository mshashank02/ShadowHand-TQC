#!/usr/bin/env python3
"""Benchmark CUDA future-HER sampling, normalization, and policy flattening."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import torch

from shadowhand_gpu.rl.normalization import CudaVecNormalize
from shadowhand_gpu.rl.replay import CudaHERReplayBuffer


def _positive_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def benchmark_one(
    observation_width: int,
    *,
    capacity: int,
    num_envs: int,
    batch_size: int,
    warmup_samples: int,
    measured_samples: int,
    device: str,
) -> dict[str, Any]:
    shapes = {
        "achieved_goal": (7,),
        "desired_goal": (7,),
        "observation": (observation_width,),
    }
    buffer = CudaHERReplayBuffer(
        requested_capacity=capacity,
        num_envs=num_envs,
        observation_shapes=shapes,
        action_dim=20,
        max_episode_steps=100,
        device=device,
    )
    normalizer = CudaVecNormalize(shapes, num_envs=num_envs, device=device)
    for values in (*buffer.observations.values(), *buffer.next_observations.values()):
        values.normal_()
    buffer.actions.uniform_(-1.0, 1.0)
    buffer.rewards.fill_(-1.0)
    buffer.dones.zero_()
    buffer.timeouts.zero_()
    rows = torch.arange(buffer.rows, device=device).unsqueeze(1)
    buffer.episode_start.copy_(rows.expand(-1, num_envs))
    buffer.episode_length.fill_(1)

    for _ in range(warmup_samples):
        buffer.sample(batch_size, normalizer=normalizer).to_tqc_batch()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    start = time.perf_counter()
    for _ in range(measured_samples):
        sample = buffer.sample(batch_size, normalizer=normalizer).to_tqc_batch()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "ok": True,
        "observation_width": observation_width,
        "policy_input_width": observation_width + 14,
        "capacity": buffer.plan.effective_capacity,
        "num_envs": num_envs,
        "batch_size": batch_size,
        "warmup_samples": warmup_samples,
        "measured_samples": measured_samples,
        "elapsed_seconds": elapsed,
        "batches_per_second": measured_samples / elapsed,
        "transitions_per_second": measured_samples * batch_size / elapsed,
        "allocated_before_measurement_bytes": allocated_before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "replay_plan": buffer.plan.to_dict(),
        "output_is_cuda": sample.observations.is_cuda,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-widths", type=_positive_ints, default=[562, 1062])
    parser.add_argument("--capacity", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--measured-samples", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.capacity, args.num_envs, args.batch_size, args.measured_samples) < 1:
        parser.error("capacity, environment count, batch size, and measurements must be positive")
    if args.warmup_samples < 0:
        parser.error("warmup count cannot be negative")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "results": [],
    }
    for observation_width in args.observation_widths:
        try:
            result = benchmark_one(
                observation_width,
                capacity=args.capacity,
                num_envs=args.num_envs,
                batch_size=args.batch_size,
                warmup_samples=args.warmup_samples,
                measured_samples=args.measured_samples,
                device=args.device,
            )
        except torch.OutOfMemoryError as exc:
            result = {
                "ok": False,
                "observation_width": observation_width,
                "error": f"{type(exc).__name__}: {exc}",
            }
        report["results"].append(result)
        torch.cuda.empty_cache()

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if all(item["ok"] for item in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
