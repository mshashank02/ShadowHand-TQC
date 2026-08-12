#!/usr/bin/env python3
"""Run the controlled A-E CPU/MuJoCo-Warp old rigid-flex diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from shadowhand_gpu.rigid_flex_diagnostic import compare_all_rigid_flex_fixtures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "xml": str(args.xml.expanduser().resolve()),
        "warning": (
            "Diagnostic-only single-body rigid-flex workaround; production remains flex-rejecting."
        ),
        "fixtures": compare_all_rigid_flex_fixtures(args.xml),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
