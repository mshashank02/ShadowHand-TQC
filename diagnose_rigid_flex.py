#!/usr/bin/env python3
"""Run the controlled A-E CPU/MuJoCo-Warp old rigid-flex diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from shadowhand_gpu.rigid_flex_diagnostic import (
    RIGID_FLEX_FIXTURES,
    compare_all_rigid_flex_fixtures,
    compare_rigid_flex_fixture,
)


def main() -> int:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed-xml",
        type=Path,
        help="settle this model, then import its matched state into the comparison model",
    )
    parser.add_argument("--experimental-tet-guard", action="store_true")
    parser.add_argument(
        "--native-mujoco-311-defaults",
        action="store_true",
        help="retain MuJoCo 3.11 collision defaults instead of matching 3.3 defaults",
    )
    parser.add_argument(
        "--fixture",
        choices=[fixture.name for fixture in RIGID_FLEX_FIXTURES],
        help="run only one controlled fixture instead of the complete A-E set",
    )
    args = parser.parse_args()
    if args.fixture:
        fixture = next(item for item in RIGID_FLEX_FIXTURES if item.name == args.fixture)
        fixtures = {
            fixture.name: compare_rigid_flex_fixture(
                args.xml,
                fixture,
                experimental_tet_guard=args.experimental_tet_guard,
                reference_compat=not args.native_mujoco_311_defaults,
                seed_xml_path=args.seed_xml,
            )
        }
    else:
        fixtures = compare_all_rigid_flex_fixtures(
            args.xml,
            experimental_tet_guard=args.experimental_tet_guard,
            reference_compat=not args.native_mujoco_311_defaults,
        )
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mujoco_version": mujoco.__version__,
        "mujoco_warp_version": mjw.__version__,
        "warp_version": wp.__version__,
        "xml": str(args.xml.expanduser().resolve()),
        "warning": (
            "Diagnostic-only single-body rigid-flex workaround; production remains flex-rejecting."
        ),
        "experimental_tet_internal_guard": args.experimental_tet_guard,
        "native_mujoco_311_defaults": args.native_mujoco_311_defaults,
        "seed_xml": None if args.seed_xml is None else str(args.seed_xml.expanduser().resolve()),
        "fixtures": fixtures,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
