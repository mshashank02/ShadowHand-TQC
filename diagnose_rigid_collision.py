#!/usr/bin/env python3
"""Write durable CPU old-flex/new-mesh and CPU/MJWarp new-mesh diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from shadowhand_gpu.parity import compare_cpu_old_new, compare_one_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-xml", type=Path, required=True)
    parser.add_argument("--new-xml", type=Path, required=True)
    parser.add_argument("--settle-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-warp",
        action="store_true",
        help="Write the CPU old/new diagnostic without allocating MuJoCo Warp.",
    )
    args = parser.parse_args()
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "old_xml": str(args.old_xml.expanduser().resolve()),
        "new_xml": str(args.new_xml.expanduser().resolve()),
        "cpu_old_flex_vs_new_mesh": compare_cpu_old_new(
            args.old_xml,
            args.new_xml,
            settle_steps=args.settle_steps,
        ),
    }
    if not args.skip_warp:
        report["cpu_vs_warp_new_mesh"] = {
            mode: compare_one_step(
                args.new_xml,
                mode=mode,
                settle_steps=args.settle_steps,
            )
            for mode in ("no_contact", "settled_contact")
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
