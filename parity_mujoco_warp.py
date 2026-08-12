#!/usr/bin/env python3
"""Report matched-state CPU MuJoCo versus direct MJWarp one-step errors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from shadowhand_gpu.parity import compare_one_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("no_contact", "settled_contact"),
        default=("no_contact", "settled_contact"),
    )
    parser.add_argument("--settle-steps", type=int, default=200)
    parser.add_argument("--contacts-per-world", type=int, default=8192)
    parser.add_argument("--constraints-per-world", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "xml": str(args.xml.expanduser().resolve()),
        "comparisons": [
            compare_one_step(
                args.xml,
                mode=mode,
                settle_steps=args.settle_steps,
                contacts_per_world=args.contacts_per_world,
                constraints_per_world=args.constraints_per_world,
            )
            for mode in args.modes
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
