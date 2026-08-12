#!/usr/bin/env python3
"""Benchmark direct MJWarp physics on a real generated ShadowHand model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from shadowhand_gpu.warp_backend import MujocoWarpBackend


def _parse_worlds(value: str) -> list[int]:
    worlds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not worlds or any(item < 1 for item in worlds):
        raise argparse.ArgumentTypeError("world counts must be comma-separated positive integers")
    return worlds


def benchmark_one(
    xml_path: Path,
    worlds: int,
    *,
    warmup_policy_steps: int,
    measured_policy_steps: int,
    substeps: int,
    contacts_per_world: int,
    constraints_per_world: int,
    use_cuda_graphs: bool,
) -> dict[str, Any]:
    backend = MujocoWarpBackend(
        xml_path,
        worlds=worlds,
        contacts_per_world=contacts_per_world,
        constraints_per_world=constraints_per_world,
        use_cuda_graphs=use_cuda_graphs,
    )
    for _ in range(warmup_policy_steps):
        backend.step(substeps)
    backend.synchronize()

    start = time.perf_counter()
    for _ in range(measured_policy_steps):
        backend.step(substeps)
    backend.synchronize()
    elapsed = time.perf_counter() - start

    active_contacts = backend.active_contact_counts.cpu().tolist()
    constraints = backend.constraint_counts.cpu().tolist()
    return {
        "ok": True,
        "worlds": worlds,
        "substeps": substeps,
        "warmup_policy_steps": warmup_policy_steps,
        "measured_policy_steps": measured_policy_steps,
        "elapsed_seconds": elapsed,
        "policy_batches_per_second": measured_policy_steps / elapsed,
        "environment_steps_per_second": measured_policy_steps * worlds / elapsed,
        "physics_world_steps_per_second": measured_policy_steps * substeps * worlds / elapsed,
        "active_contacts": active_contacts,
        "constraints": constraints,
        "backend": backend.report(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--worlds", type=_parse_worlds, default=[1])
    parser.add_argument("--warmup-policy-steps", type=int, default=2)
    parser.add_argument("--measured-policy-steps", type=int, default=10)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--contacts-per-world", type=int, default=8192)
    parser.add_argument("--constraints-per-world", type=int, default=4096)
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "xml": str(args.xml.expanduser().resolve()),
        "results": [],
    }
    for worlds in args.worlds:
        try:
            item = benchmark_one(
                args.xml,
                worlds,
                warmup_policy_steps=args.warmup_policy_steps,
                measured_policy_steps=args.measured_policy_steps,
                substeps=args.substeps,
                contacts_per_world=args.contacts_per_world,
                constraints_per_world=args.constraints_per_world,
                use_cuda_graphs=not args.no_cuda_graphs,
            )
        except Exception as exc:
            item = {"ok": False, "worlds": worlds, "error": f"{type(exc).__name__}: {exc}"}
            results["results"].append(item)
            break
        results["results"].append(item)

    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if all(item["ok"] for item in results["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
