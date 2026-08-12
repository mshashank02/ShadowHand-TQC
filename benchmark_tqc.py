#!/usr/bin/env python3
"""Benchmark CUDA TQC updates without simulator or replay allocation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import torch

from shadowhand_gpu.rl.tqc import TQCBatch, TQCConfig, TQCLearner


def _positive_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def benchmark_one(
    observation_dim: int,
    *,
    action_dim: int,
    hidden_dims: tuple[int, ...],
    batch_size: int,
    warmup_updates: int,
    measured_updates: int,
    device: str,
) -> dict[str, Any]:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    config = TQCConfig(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        device=device,
    )
    learner = TQCLearner(config)
    batch = TQCBatch(
        observations=torch.randn(batch_size, observation_dim, device=device),
        actions=torch.empty(batch_size, action_dim, device=device).uniform_(-1.0, 1.0),
        next_observations=torch.randn(batch_size, observation_dim, device=device),
        rewards=torch.randn(batch_size, 1, device=device),
        dones=torch.zeros(batch_size, 1, device=device),
    )
    for _ in range(warmup_updates):
        learner.update(batch)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)

    start = time.perf_counter()
    for _ in range(measured_updates):
        learner.update(batch)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "ok": True,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "hidden_dims": hidden_dims,
        "batch_size": batch_size,
        "warmup_updates": warmup_updates,
        "measured_updates": measured_updates,
        "elapsed_seconds": elapsed,
        "updates_per_second": measured_updates / elapsed,
        "samples_per_second": measured_updates * batch_size / elapsed,
        "allocated_before_measurement_bytes": allocated_before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-dims", type=_positive_ints, default=[576, 1076])
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--hidden-dims", type=_positive_ints, default=[512, 512, 512])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measured-updates", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup_updates < 0 or args.measured_updates < 1:
        parser.error("batch size/measurement counts must be positive and warmup nonnegative")
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
    for observation_dim in args.observation_dims:
        try:
            result = benchmark_one(
                observation_dim,
                action_dim=args.action_dim,
                hidden_dims=tuple(args.hidden_dims),
                batch_size=args.batch_size,
                warmup_updates=args.warmup_updates,
                measured_updates=args.measured_updates,
                device=args.device,
            )
        except torch.OutOfMemoryError as exc:
            result = {
                "ok": False,
                "observation_dim": observation_dim,
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
