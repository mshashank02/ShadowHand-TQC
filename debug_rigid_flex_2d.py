#!/usr/bin/env python3
"""Run the CPU-gated 2D rigid-flex MuJoCo/Warp validation stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shadowhand_gpu.rigid_flex_2d_validation import (
    compare_five_fixtures_cpu_warp,
    write_minimal_surface_reproducer,
)
from shadowhand_gpu.rigid_flex_root_cause import (
    collect_cpu_matrix,
    collect_warp_matrix,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--obj", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for name in ("cpu", "warp"):
        command = commands.add_parser(name)
        command.add_argument("--xml", type=Path, required=True)
        command.add_argument("--states", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    five = commands.add_parser("five-fixtures")
    five.add_argument("--xml", type=Path, required=True)
    five.add_argument("--features", type=Path, required=True)
    five.add_argument("--output", type=Path, required=True)
    five.add_argument("--reference-compat", action="store_true")
    args = parser.parse_args()

    if args.command == "prepare":
        payload = write_minimal_surface_reproducer(args.obj, args.output)
    elif args.command == "cpu":
        payload = collect_cpu_matrix(args.xml, args.states)
        write_json(args.output, payload)
    elif args.command == "warp":
        payload = collect_warp_matrix(args.xml, args.states, apply_tet_guard=False)
        write_json(args.output, payload)
    else:
        payload = compare_five_fixtures_cpu_warp(
            args.xml, args.features, reference_compat=args.reference_compat
        )
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
