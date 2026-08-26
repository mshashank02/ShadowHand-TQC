#!/usr/bin/env python3
"""Benchmark native-rigid batched ShadowHand reset and task steps on CUDA."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import torch

from shadowhand_gpu.task import ShadowHandTaskConfig, ShadowHandWarpTask
from shadowhand_gpu.warp_backend import MujocoWarpBackend


def _positive_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def benchmark_one(
    xml_path: Path,
    worlds: int,
    *,
    warmup_steps: int,
    measured_steps: int,
    contacts_per_world: int,
    constraints_per_world: int,
    device: str,
    use_cuda_graphs: bool,
) -> dict[str, Any]:
    free_before_bytes, total_bytes = torch.cuda.mem_get_info(device)
    backend = MujocoWarpBackend(
        xml_path,
        worlds=worlds,
        device=device,
        contacts_per_world=contacts_per_world,
        constraints_per_world=constraints_per_world,
        use_cuda_graphs=use_cuda_graphs,
    )
    task = ShadowHandWarpTask(backend, config=ShadowHandTaskConfig(), seed=0)
    actions = torch.zeros(worlds, 20, dtype=backend.qpos.dtype, device=device)

    # First reset compiles/warms kernels. Report a second all-world reset.
    task.reset()
    reset_start = time.perf_counter()
    task.reset()
    backend.synchronize()
    reset_elapsed = time.perf_counter() - reset_start
    for _ in range(warmup_steps):
        task.step(actions)
    backend.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)
    overflow_seen = torch.zeros((), dtype=torch.int32, device=device)
    active_contacts_high_water = torch.zeros((), dtype=torch.int32, device=device)
    constraints_high_water = torch.zeros((), dtype=torch.int32, device=device)

    start = time.perf_counter()
    for _ in range(measured_steps):
        step = task.step(actions)
        torch.maximum(overflow_seen, backend.overflow_flags.max(), out=overflow_seen)
        torch.maximum(
            active_contacts_high_water,
            backend.active_contact_counts.max(),
            out=active_contacts_high_water,
        )
        torch.maximum(
            constraints_high_water,
            backend.constraint_counts.max(),
            out=constraints_high_water,
        )
    backend.synchronize()
    elapsed = time.perf_counter() - start
    task_report = task.report()
    free_after_bytes = task_report["backend"]["device_free_bytes"]
    return {
        "ok": True,
        "worlds": worlds,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "elapsed_seconds": elapsed,
        "environment_steps_per_second": measured_steps * worlds / elapsed,
        "physics_world_steps_per_second": (
            measured_steps * worlds * task.config.physics_steps_per_action / elapsed
        ),
        "all_world_reset_seconds": reset_elapsed,
        "reset_worlds_per_second": worlds / reset_elapsed,
        "observation_width": task.observation_width,
        "output_is_cuda": step.observations["observation"].is_cuda,
        "allocated_before_measurement_bytes": allocated_before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "device_total_bytes": total_bytes,
        "device_free_before_benchmark_bytes": free_before_bytes,
        "device_free_after_measurement_bytes": free_after_bytes,
        "device_memory_used_delta_bytes": max(0, free_before_bytes - free_after_bytes),
        "overflow_flags_max": int(overflow_seen.cpu()),
        "batch_global_active_contacts_high_water": int(active_contacts_high_water.cpu()),
        "constraints_per_world_high_water": int(constraints_high_water.cpu()),
        "task": task_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--worlds", type=_positive_ints, default=[1, 16, 64])
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument("--contacts-per-world", type=int, default=8192)
    parser.add_argument("--constraints-per-world", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup_steps < 0 or args.measured_steps < 1:
        parser.error("warmup must be nonnegative and measured steps positive")

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "xml": str(args.xml.expanduser().resolve()),
        "device": args.device,
        "results": [],
    }
    for worlds in args.worlds:
        try:
            result = benchmark_one(
                args.xml,
                worlds,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
                contacts_per_world=args.contacts_per_world,
                constraints_per_world=args.constraints_per_world,
                device=args.device,
                use_cuda_graphs=not args.no_cuda_graphs,
            )
        except Exception as exc:
            result = {"ok": False, "worlds": worlds, "error": f"{type(exc).__name__}: {exc}"}
            report["results"].append(result)
            break
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
