#!/usr/bin/env python3
"""Render the CPU-gated 2D rigid-flex experiment and its Warp stop decision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BACKENDS = ("cpu_mujoco_3_3_1", "cpu_mujoco_3_11_0", "mujoco_warp_3_11_0")
REVISED_TACTILE_TOLERANCE = 0.065
ONSET_TOLERANCE_MM = 0.10


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
    five_native = json.loads(
        (output / "cpu_warp_five_fixtures.json").read_text(encoding="utf-8")
    )
    five_compat = json.loads(
        (output / "cpu_warp_five_fixtures_compat.json").read_text(encoding="utf-8")
    )
    n500_unseeded = json.loads(
        (output / "n500_cpu_warp_statistics.json").read_text(encoding="utf-8")
    )["fixtures"]["settled_contact"]
    n500_seeded = json.loads(
        (output / "n500_cpu_warp_seeded_3d_state.json").read_text(encoding="utf-8")
    )["fixtures"]["settled_contact"]

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
                        "flex_internal_count"
                    ],
                    "geom_flex_contact_count": state["contact"]["external_count"],
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

    fixture_rows: list[dict[str, Any]] = []
    for mode, payload in (("native_3_11", five_native), ("warp_compatible", five_compat)):
        for fixture_name, fixture in sorted(payload["fixtures"].items()):
            common = fixture["common_pose"]
            cpu_contacts = len(common["cpu"]["contacts"])
            warp_contacts = len(common["warp"]["contacts"])
            cpu_rows = common["cpu"]["constraint_rows"]
            warp_rows = common["warp"]["constraint_rows"]
            tactile = common["comparison"]["tactile"]
            onset_pass = abs(fixture["onset_shift_mm"]) < ONSET_TOLERANCE_MM
            manifold_pass = cpu_contacts == warp_contacts and cpu_rows == warp_rows
            tactile_pass = (
                tactile["total_magnitude_relative_error"] <= REVISED_TACTILE_TOLERANCE
            )
            fixture_rows.append(
                {
                    "mode": mode,
                    "fixture": fixture_name,
                    "onset_shift_mm": fixture["onset_shift_mm"],
                    "onset_pass_0p10mm": onset_pass,
                    "cpu_contacts": cpu_contacts,
                    "warp_contacts": warp_contacts,
                    "cpu_constraint_rows": cpu_rows,
                    "warp_constraint_rows": warp_rows,
                    "manifold_exact": manifold_pass,
                    "cpu_total_touch": tactile["reference_total_magnitude"],
                    "warp_total_touch": tactile["actual_total_magnitude"],
                    "total_touch_relative_error": tactile[
                        "total_magnitude_relative_error"
                    ],
                    "tactile_pass_6p5pct": tactile_pass,
                    "active_jaccard": tactile["active_jaccard"],
                    "fixture_pass": onset_pass and manifold_pass and tactile_pass,
                }
            )
    _write_csv(output / "cpu_warp_2d_five_fixture_summary.csv", fixture_rows)

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
    native_fixture_rows = [row for row in fixture_rows if row["mode"] == "native_3_11"]
    compat_fixture_rows = [
        row for row in fixture_rows if row["mode"] == "warp_compatible"
    ]
    seeded_initial = n500_seeded["snapshots"]["initial"]
    seeded_final = n500_seeded["snapshots"]["final"]
    unseeded_initial = n500_unseeded["snapshots"]["initial"]
    unseeded_final = n500_unseeded["snapshots"]["final"]
    n500_pass = bool(
        seeded_final["touch"]["total_magnitude_relative_error"]
        <= REVISED_TACTILE_TOLERANCE
        and seeded_final["contact_counts"]["cpu"]
        == seeded_final["contact_counts"]["warp"]
    )
    decision = {
        "classification": "Outcome C — CPU 2D works, Warp still differs",
        "revised_reference": "CPU MuJoCo 3.11.0",
        "revised_total_tactile_relative_error_tolerance": REVISED_TACTILE_TOLERANCE,
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
        "five_fixture_warp_run": True,
        "five_fixture_native_all_pass": all(
            row["fixture_pass"] for row in native_fixture_rows
        ),
        "five_fixture_compatible_all_pass": all(
            row["fixture_pass"] for row in compat_fixture_rows
        ),
        "five_fixture_native_tactile_relative_errors": {
            row["fixture"]: row["total_touch_relative_error"]
            for row in native_fixture_rows
        },
        "native_full_n500_supported": False,
        "native_full_n500_blocker": (
            "MuJoCo Warp 3.11 rejects non-zero geom-pair margin while MULTICCD is enabled"
        ),
        "n500_run": True,
        "n500_reference_compat": n500_seeded["reference_compat"],
        "n500_unseeded_control": {
            "initial_contacts_cpu_warp": [
                unseeded_initial["contact_counts"]["cpu"],
                unseeded_initial["contact_counts"]["warp"],
            ],
            "final_contacts_cpu_warp": [
                unseeded_final["contact_counts"]["cpu"],
                unseeded_final["contact_counts"]["warp"],
            ],
            "active_touch_channels": unseeded_final["touch"]["active_sensors_cpu"],
        },
        "n500_seeded_settled_state": {
            "initial_contacts_cpu_warp": [
                seeded_initial["contact_counts"]["cpu"],
                seeded_initial["contact_counts"]["warp"],
            ],
            "final_contacts_cpu_warp": [
                seeded_final["contact_counts"]["cpu"],
                seeded_final["contact_counts"]["warp"],
            ],
            "final_constraint_rows_cpu_warp": [
                seeded_final["constraint_rows"]["cpu"],
                seeded_final["constraint_rows"]["warp"],
            ],
            "qpos_max_abs": seeded_final["qpos"]["max_abs"],
            "qvel_max_abs": seeded_final["qvel"]["max_abs"],
            "physical_object_error": seeded_final["physical_object_error"],
            "tactile": seeded_final["touch"],
            "pass": n500_pass,
        },
        "n1000_run": False,
        "performance_run": False,
        "rl_run": False,
        "stop_reason": (
            "All five exact fixtures fail the revised 6.5% tactile tolerance, and the "
            "active full N=500 settled state has 39.50% total tactile error after one step."
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
        "After the user accepted the 2 mm probe's 6.17% total-touch error and revised the "
        "matching-version tolerance to 6.5%, the exact five fixtures and full N=500 test "
        "were run with CPU MuJoCo 3.11.0 and MuJoCo Warp 3.11.0.",
        "",
        "The five-fixture result is not close to a pass. Common-pose total-touch relative "
        "errors are 100.000% fingertip, 7.667% palm, 24.761% deepest concavity, 14.379% "
        "macro, and 9.434% roughness. Every fixture exceeds 6.5%; the fingertip also shifts "
        "onset by 2.061 mm and loses all 12 CPU contacts in Warp. Disabling MULTICCD and "
        "NATIVECCD for the project-compatible configuration produces the same measured "
        "fixture outcomes.",
        "",
        "The complete N=500 model cannot be transferred to Warp under native MuJoCo 3.11 "
        "defaults because Warp rejects a non-zero geom-pair margin with MULTICCD enabled. "
        "The full comparison therefore disables MULTICCD and NATIVECCD in both CPU 3.11 "
        "and Warp 3.11, without changing geometry or contact parameters.",
        "",
        "The ordinary 2D settled control has 99 matched contacts at transfer but no active "
        "touch channels, so it is not a useful tactile gate. Seeding the exact 2D model from "
        "the previously established active 3D settled state gives 101/101 contacts, nine "
        "active channels, and essentially exact CPU-to-Warp transfer. After one matched "
        f"step, CPU/Warp contacts are {seeded_final['contact_counts']['cpu']}/"
        f"{seeded_final['contact_counts']['warp']} and constraint rows are "
        f"{seeded_final['constraint_rows']['cpu']}/"
        f"{seeded_final['constraint_rows']['warp']}. Total touch changes from "
        f"`{seeded_final['touch']['cpu_total_tactile_magnitude']:.9g}` to "
        f"`{seeded_final['touch']['warp_total_tactile_magnitude']:.9g}` "
        f"({seeded_final['touch']['total_magnitude_relative_error']:.3%} error); RMSE is "
        f"`{seeded_final['touch']['rmse']:.6g}`, correlation "
        f"`{seeded_final['touch']['pearson_correlation']:.6g}`, cosine "
        f"`{seeded_final['touch']['cosine_similarity']:.6g}`, and active Jaccard "
        f"`{seeded_final['touch']['active_sensor_jaccard']:.6g}`.",
        "",
        "N=1000, performance work, and RL were not run because the required N=500 gate "
        "clearly fails.",
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
