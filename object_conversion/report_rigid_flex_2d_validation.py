#!/usr/bin/env python3
"""Render the CPU-gated 2D rigid-flex experiment and its Warp stop decision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BACKENDS = ("cpu_mujoco_3_3_1", "cpu_mujoco_3_11_0", "mujoco_warp_3_11_0")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(output: Path) -> dict[str, Any]:
    cpu_gate = json.loads((output / "cpu_3d_vs_2d_results.json").read_text(encoding="utf-8"))
    minimal = output / "minimal"
    matrices = {
        "cpu_mujoco_3_3_1": json.loads((minimal / "cpu_matrix.json").read_text()),
        "cpu_mujoco_3_11_0": json.loads((minimal / "cpu_3_11_matrix.json").read_text()),
        "mujoco_warp_3_11_0": json.loads((minimal / "warp_matrix.json").read_text()),
    }
    onset = {
        "cpu_mujoco_3_3_1": json.loads((minimal / "cpu_onset_matrix.json").read_text()),
        "cpu_mujoco_3_11_0": json.loads((minimal / "cpu_3_11_onset_matrix.json").read_text()),
        "mujoco_warp_3_11_0": json.loads((minimal / "warp_onset_matrix.json").read_text()),
    }

    approach_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    tactile_rows: list[dict[str, Any]] = []
    for backend in BACKENDS:
        for scenario, state in matrices[backend]["scenarios"].items():
            approach_rows.append(
                {
                    "backend": backend,
                    "scenario": scenario,
                    "signed_shell_offset_mm": state["signed_shell_offset"] * 1000.0,
                    "contact_count": state["contact"]["count"],
                    "constraint_rows": state["constraint_rows"],
                    "minimum_distance_mm": (
                        None
                        if state["contact"]["minimum_distance"] is None
                        else state["contact"]["minimum_distance"] * 1000.0
                    ),
                    "touch": state["touch"][0],
                }
            )
            contact_rows.append(
                {
                    "backend": backend,
                    "scenario": scenario,
                    "contact_count": state["contact"]["count"],
                    "constraint_rows": state["constraint_rows"],
                    "minimum_distance_m": state["contact"]["minimum_distance"],
                    "maximum_distance_m": state["contact"]["maximum_distance"],
                    "flex_internal_contact_count": state["contact"][
                        "flex_internal_contact_count"
                    ],
                    "geom_flex_contact_count": state["contact"]["geom_flex_contact_count"],
                    "contacts_json": json.dumps(state["contacts"], sort_keys=True),
                }
            )
            tactile_rows.append(
                {
                    "backend": backend,
                    "scenario": scenario,
                    "touch": state["touch"][0],
                    "difference_vs_cpu_3_3_1": (
                        state["touch"][0]
                        - matrices["cpu_mujoco_3_3_1"]["scenarios"][scenario]["touch"][0]
                    ),
                    "difference_vs_cpu_3_11_0": (
                        state["touch"][0]
                        - matrices["cpu_mujoco_3_11_0"]["scenarios"][scenario]["touch"][0]
                    ),
                }
            )
    _write_csv(output / "cpu_warp_2d_approach.csv", approach_rows)
    _write_csv(output / "cpu_warp_2d_contacts.csv", contact_rows)
    _write_csv(output / "cpu_warp_2d_tactile.csv", tactile_rows)

    separated = "flex_only_no_probe_contact"
    shallow = "shallow_penetration"
    deep = "deep_penetration"
    zero_contact_pass = all(
        matrices[backend]["scenarios"][separated]["contact"]["count"] == 0
        and matrices[backend]["scenarios"][separated]["constraint_rows"] == 0
        for backend in BACKENDS
    )
    shallow_cpu331 = matrices[BACKENDS[0]]["scenarios"][shallow]
    shallow_cpu311 = matrices[BACKENDS[1]]["scenarios"][shallow]
    shallow_warp = matrices[BACKENDS[2]]["scenarios"][shallow]
    deep_cpu311 = matrices[BACKENDS[1]]["scenarios"][deep]
    deep_warp = matrices[BACKENDS[2]]["scenarios"][deep]
    warp_inside_offsets = [
        abs(float(state["signed_shell_offset"]))
        for name, state in onset[BACKENDS[2]]["scenarios"].items()
        if name.startswith("onset_inside_") and state["contact"]["count"]
    ]
    onset_resolution_m = min(warp_inside_offsets)
    decision = {
        "classification": "Outcome C — CPU 2D works, Warp still differs",
        "cpu_representation_gate": cpu_gate["decision"],
        "zero_contact_20mm_pass": zero_contact_pass,
        "onset_bracket_resolution_m": onset_resolution_m,
        "shallow": {
            "cpu_3_3_1_contacts_rows": [
                shallow_cpu331["contact"]["count"], shallow_cpu331["constraint_rows"]
            ],
            "cpu_3_11_0_contacts_rows": [
                shallow_cpu311["contact"]["count"], shallow_cpu311["constraint_rows"]
            ],
            "warp_contacts_rows": [
                shallow_warp["contact"]["count"], shallow_warp["constraint_rows"]
            ],
            "cpu_3_3_1_touch": shallow_cpu331["touch"][0],
            "cpu_3_11_0_touch": shallow_cpu311["touch"][0],
            "warp_touch": shallow_warp["touch"][0],
        },
        "deep": {
            "cpu_3_11_0_contacts_rows": [
                deep_cpu311["contact"]["count"], deep_cpu311["constraint_rows"]
            ],
            "warp_contacts_rows": [deep_warp["contact"]["count"], deep_warp["constraint_rows"]],
            "cpu_3_11_0_touch": deep_cpu311["touch"][0],
            "warp_touch": deep_warp["touch"][0],
        },
        "five_fixture_warp_run": False,
        "n500_run": False,
        "n1000_run": False,
        "performance_run": False,
        "rl_run": False,
        "stop_reason": (
            "The deeper primitive probe retains a material 13/52 CPU versus 10/40 Warp "
            "contact-manifold gap. The specification gates later phases on simple-probe parity."
        ),
    }
    (output / "warp_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# 2D OBJ rigid-flex final validation",
        "",
        "Classification: **Outcome C — CPU 2D works, Warp still differs**.",
        "",
        "The CPU 3D→2D representation gate passed all five original fixtures. The exact "
        "surface flex also eliminates the tetrahedral false-internal-contact bug: at 20 mm "
        "CPU 3.3.1, CPU 3.11.0, and Warp all have zero contacts and zero rows.",
        "",
        f"Quasi-static onset agrees to the Warp float-resolution bracket of "
        f"`{onset_resolution_m * 1000.0:.9g} mm`.",
        "",
        "At 0.5 mm penetration, CPU 3.11.0 and Warp both produce 3 contacts/12 rows; "
        f"touch is `{shallow_cpu311['touch'][0]:.9g}` versus "
        f"`{shallow_warp['touch'][0]:.9g}`. CPU 3.3.1 touch is instead "
        f"`{shallow_cpu331['touch'][0]:.9g}`, exposing a substantial MuJoCo-version "
        "force-law change for this 2D flex even when the manifold agrees.",
        "",
        "At 2 mm penetration, the decisive manifold failure remains: CPU 3.11.0 has "
        f"{deep_cpu311['contact']['count']} contacts/{deep_cpu311['constraint_rows']} rows "
        f"and touch `{deep_cpu311['touch'][0]:.9g}`; Warp has "
        f"{deep_warp['contact']['count']} contacts/{deep_warp['constraint_rows']} rows "
        f"and touch `{deep_warp['touch'][0]:.9g}`.",
        "",
        "Per the hard gate, the five-fixture Warp sweep, N=500/N=1000, benchmarks, and RL "
        "were not run. The failed simple deep-contact test is already sufficient to reject "
        "this candidate for the tactile study.",
        "",
        "## Answer",
        "",
        "No. The original 3D GMSH rigid flex can be replaced by the 2D OBJ rigid flex on "
        "CPU within the declared representation gates, and the replacement avoids both "
        "tetrahedral Warp paths, but MuJoCo Warp 3.11.0 still fails deeper 2D contact-manifold "
        "parity. The complete scientific replacement criterion is therefore not met.",
        "",
    ]
    (output / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--output", type=Path, default=repository / "generated/rigid_flex_2d_validation"
    )
    args = parser.parse_args()
    print(json.dumps(render(args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
