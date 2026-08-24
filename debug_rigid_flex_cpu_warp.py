#!/usr/bin/env python3
"""Build and run the focused rigid-flex/sphere CPU-versus-Warp reproducer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shadowhand_gpu.model_loader import load_project_model
from shadowhand_gpu.rigid_flex_root_cause import (
    collect_cpu_matrix,
    collect_warp_matrix,
    write_json,
    write_minimal_reproducer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="write the exact-Gmsh sphere reproducer")
    prepare.add_argument("--source-xml", required=True)
    prepare.add_argument("--source-msh", required=True)
    prepare.add_argument("--output-dir", required=True)

    cpu = subparsers.add_parser("cpu", help="collect a CPU MuJoCo matrix")
    cpu.add_argument("--xml", required=True)
    cpu.add_argument("--states", required=True)
    cpu.add_argument("--output", required=True)

    warp = subparsers.add_parser("warp", help="collect a MuJoCo Warp matrix")
    warp.add_argument("--xml", required=True)
    warp.add_argument("--states", required=True)
    warp.add_argument("--output", required=True)
    warp.add_argument("--experimental-tet-guard", action="store_true")
    warp.add_argument("--contacts-per-world", type=int, default=40000)
    warp.add_argument("--constraints-per-world", type=int, default=40000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        model, _ = load_project_model(args.source_xml, reference_compat=True)
        result = write_minimal_reproducer(model, args.source_msh, args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "cpu":
        result = collect_cpu_matrix(args.xml, args.states)
    else:
        result = collect_warp_matrix(
            args.xml,
            args.states,
            apply_tet_guard=args.experimental_tet_guard,
            contacts_per_world=args.contacts_per_world,
            constraints_per_world=args.constraints_per_world,
        )
    write_json(args.output, result)
    print(Path(args.output).resolve())


if __name__ == "__main__":
    main()
