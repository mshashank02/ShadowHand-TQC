#!/usr/bin/env python3
"""Render durable artifacts from the rigid-flex CPU/Warp debug matrices."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


BACKEND_FILES = {
    "cpu_3.3.1": "cpu_mujoco_3.3.1.json",
    "cpu_3.11.0": "cpu_mujoco_3.11.0.json",
    "warp_3.11.0_stock": "mujoco_warp_3.11.0_unpatched.json",
    "warp_3.11.0_tet_guard": "mujoco_warp_3.11.0_tet_guard.json",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _scenario_order(matrix: dict[str, Any]) -> list[str]:
    return sorted(
        matrix["scenarios"],
        key=lambda name: matrix["scenarios"][name]["signed_shell_offset"],
        reverse=True,
    )


def _match_contacts(cpu: list[dict[str, Any]], warp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available = set(range(len(warp)))
    rows: list[dict[str, Any]] = []
    for cpu_index, left in enumerate(cpu):
        candidates = [
            index
            for index in available
            if warp[index]["geom"] == left["geom"] and warp[index]["flex"] == left["flex"]
        ]
        if not candidates:
            rows.append({"status": "cpu_unmatched", "cpu_index": cpu_index, "warp_index": ""})
            continue
        lp = np.asarray(left["pos"], dtype=float)
        match = min(candidates, key=lambda index: np.linalg.norm(lp - np.asarray(warp[index]["pos"])))
        available.remove(match)
        right = warp[match]
        rp = np.asarray(right["pos"], dtype=float)
        ln = np.asarray(left["normal"], dtype=float)
        rn = np.asarray(right["normal"], dtype=float)
        normal_dot = float(np.clip(np.dot(ln, rn), -1.0, 1.0))
        rows.append(
            {
                "status": "matched",
                "cpu_index": cpu_index,
                "warp_index": match,
                "cpu_element": left["elem"][1],
                "warp_element": right["elem"][1],
                "position_l2_error_m": float(np.linalg.norm(lp - rp)),
                "position_max_abs_error_m": float(np.max(np.abs(lp - rp))),
                "normal_dot": normal_dot,
                "normal_angle_deg": math.degrees(math.acos(normal_dot)),
                "distance_abs_error_m": abs(float(left["dist"]) - float(right["dist"])),
                "normal_force_abs_error": abs(float(left["force"][0]) - float(right["force"][0])),
            }
        )
    for index in sorted(available):
        rows.append({"status": "warp_unmatched", "cpu_index": "", "warp_index": index})
    return rows


def _svg_counts(path: Path, matrices: dict[str, dict[str, Any]]) -> None:
    names = _scenario_order(matrices["cpu_3.11.0"])
    offsets = [matrices["cpu_3.11.0"]["scenarios"][name]["signed_shell_offset"] * 1000 for name in names]
    series = {
        key: [matrices[key]["scenarios"][name]["contact"]["count"] for name in names]
        for key in ("cpu_3.11.0", "warp_3.11.0_stock", "warp_3.11.0_tet_guard")
    }
    width, height, left, top, plot_w, plot_h = 900, 480, 80, 35, 760, 360
    x_min, x_max = min(offsets), max(offsets)
    max_count = max(max(values) for values in series.values())
    colors = {"cpu_3.11.0": "#1f77b4", "warp_3.11.0_stock": "#d62728", "warp_3.11.0_tet_guard": "#2ca02c"}
    def xy(x: float, y: float) -> tuple[float, float]:
        return (
            left + (x - x_min) / (x_max - x_min) * plot_w,
            top + (1.0 - y / max_count) * plot_h,
        )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>',
        '<text x="450" y="465" text-anchor="middle" font-family="sans-serif">signed sphere-shell offset (mm)</text>',
        '<text x="18" y="220" transform="rotate(-90 18 220)" text-anchor="middle" font-family="sans-serif">contact count</text>',
    ]
    for index, (key, values) in enumerate(series.items()):
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (xy(xv, yv) for xv, yv in zip(offsets, values)))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{colors[key]}" stroke-width="2"/>')
        lines.append(f'<text x="{left+20}" y="{top+20+index*20}" fill="{colors[key]}" font-family="sans-serif">{key}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(base: Path) -> None:
    matrices = {name: _read(base / filename) for name, filename in BACKEND_FILES.items()}
    cpu = matrices["cpu_3.11.0"]
    stock = matrices["warp_3.11.0_stock"]
    guard = matrices["warp_3.11.0_tet_guard"]

    topology_hashes = {name: value["topology"]["hashes"] for name, value in matrices.items()}
    _write_json(
        base / "model_comparison.json",
        {
            "all_topology_hashes_equal": len({json.dumps(value, sort_keys=True) for value in topology_hashes.values()}) == 1,
            "topology_hashes": topology_hashes,
            "compiled_counts": {name: value["topology"]["counts"] for name, value in matrices.items()},
            "flex_flags": {name: value["topology"]["flex"] for name, value in matrices.items()},
            "warp_transfer": stock["transferred"],
            "warp_integer_topology_transfer_hashes": stock["transferred_topology_hashes"],
            "warp_integer_topology_transfer_matches_cpu": {
                name: stock["transferred_topology_hashes"][name] == stock["topology"]["hashes"][name]
                for name in stock["transferred_topology_hashes"]
            },
            "warp_compiled_vertex_transfer_max_abs_error_m": stock[
                "transferred_flex_vert_max_abs_error_m"
            ],
            "world_position_max_abs_error_m": {
                name: value["scenarios"]["flex_only_no_probe_contact"]["flexvert_xpos_max_abs_error_vs_compiled"]
                for name, value in matrices.items()
            },
        },
    )

    _write_json(
        base / "no_contact_comparison.json",
        {
            name: value["scenarios"]["flex_only_no_probe_contact"]
            for name, value in matrices.items()
        },
    )

    approach_rows = []
    constraint_rows = []
    force_rows = []
    tactile_rows = []
    for backend, matrix in matrices.items():
        for scenario_name in _scenario_order(cpu):
            scenario = matrix["scenarios"][scenario_name]
            summary = scenario["contact"]
            approach_rows.append(
                {
                    "backend": backend,
                    "scenario": scenario_name,
                    "signed_shell_offset_mm": scenario["signed_shell_offset"] * 1000,
                    "contact_count": summary["count"],
                    "external_contact_count": summary["external_count"],
                    "false_flex_internal_count": summary["flex_internal_count"],
                    "minimum_distance_m": summary["minimum_distance"],
                }
            )
            constraint_rows.append(
                {
                    "backend": backend,
                    "scenario": scenario_name,
                    "contact_count": summary["count"],
                    "constraint_rows": scenario["constraint_rows"],
                }
            )
            normal_forces = [abs(float(contact["force"][0])) for contact in scenario["contacts"]]
            force_rows.append(
                {
                    "backend": backend,
                    "scenario": scenario_name,
                    "contact_count": summary["count"],
                    "normal_force_sum": sum(normal_forces),
                    "normal_force_max": max(normal_forces, default=0.0),
                }
            )
            tactile_rows.append(
                {
                    "backend": backend,
                    "scenario": scenario_name,
                    "touch": scenario["touch"][0],
                    "abs_error_vs_cpu_3.11.0": abs(
                        float(scenario["touch"][0]) - float(cpu["scenarios"][scenario_name]["touch"][0])
                    ),
                }
            )
    _write_csv(base / "approach_curve.csv", list(approach_rows[0]), approach_rows)
    _write_csv(base / "constraint_comparison.csv", list(constraint_rows[0]), constraint_rows)
    _write_csv(base / "force_comparison.csv", list(force_rows[0]), force_rows)
    _write_csv(base / "tactile_comparison.csv", list(tactile_rows[0]), tactile_rows)

    first_name = next(
        name for name in _scenario_order(cpu)
        if cpu["scenarios"][name]["contact"]["external_count"] > 0
    )
    _write_json(base / "first_contact_cpu.json", {"scenario": first_name, **cpu["scenarios"][first_name]})
    _write_json(
        base / "first_contact_warp.json",
        {
            "stock_first_reported_contact": stock["scenarios"]["flex_only_no_probe_contact"],
            "stock_contact_is_false_internal": True,
            "guarded_first_external_scenario": first_name,
            "guarded": guard["scenarios"][first_name],
        },
    )

    match_rows = []
    for scenario_name in ("penetration_0p1mm", "shallow_penetration", "penetration_1mm", "deep_penetration", "sliding_tangent_negative", "sliding_tangent_positive"):
        for row in _match_contacts(cpu["scenarios"][scenario_name]["contacts"], guard["scenarios"][scenario_name]["contacts"]):
            match_rows.append({"scenario": scenario_name, **row})
    match_fields = [
        "scenario", "status", "cpu_index", "warp_index", "cpu_element", "warp_element",
        "position_l2_error_m", "position_max_abs_error_m", "normal_dot", "normal_angle_deg",
        "distance_abs_error_m", "normal_force_abs_error",
    ]
    _write_csv(base / "contact_matching.csv", match_fields, match_rows)

    radius_cpu = cpu["topology"]["flex"]["radius"][0]
    radius_warp = stock["transferred"]["flex_radius"][0]
    support_rows = [
        {
            "backend": "cpu_mujoco_3.3.1",
            "element_shape": "tetrahedron",
            "radius_m": matrices["cpu_3.3.1"]["topology"]["flex"]["radius"][0],
            "support_or_collision_rule": "full tetrahedron support K plus radius ball; GJK/EPA",
            "difference_from_cpu_3.11_m": 0.0,
        },
        {
            "backend": "cpu_mujoco_3.11.0",
            "element_shape": "tetrahedron",
            "radius_m": radius_cpu,
            "support_or_collision_rule": "full tetrahedron support K plus radius ball; GJK/EPA",
            "difference_from_cpu_3.11_m": 0.0,
        },
        {
            "backend": "mujoco_warp_3.11.0",
            "element_shape": "four triangles per tetrahedron",
            "radius_m": radius_warp,
            "support_or_collision_rule": "analytic geom-triangle tests; 1 mm position deduplication",
            "difference_from_cpu_3.11_m": abs(radius_warp - radius_cpu),
        },
    ]
    _write_csv(base / "support_mapping_comparison.csv", list(support_rows[0]), support_rows)

    version_rows = [
        {"backend": "CPU MuJoCo", "mujoco": "3.3.1", "mujoco_warp": "", "warp": "", "result": "reference contact manifold"},
        {"backend": "CPU MuJoCo", "mujoco": "3.11.0", "mujoco_warp": "", "warp": "", "result": "same counts/distances as CPU 3.3.1"},
        {"backend": "MuJoCo Warp stock", "mujoco": "3.11.0", "mujoco_warp": "3.11.0", "warp": "1.16.0", "result": "2674 false internal contacts"},
        {"backend": "MuJoCo Warp guarded", "mujoco": "3.11.0", "mujoco_warp": "3.11.0", "warp": "1.16.0", "result": "false contacts removed; deeper manifold still differs"},
        {"backend": "MuJoCo Warp main 70c4571", "mujoco": "3.11.0", "mujoco_warp": "unreleased main", "warp": "1.16.0", "result": "diagnosed kernel absent; no-contact fixed; manifold still differs"},
    ]
    _write_csv(base / "version_matrix.csv", list(version_rows[0]), version_rows)

    n500 = _read(base / "n500_settled_tet_guard.json")["fixtures"]["settled_contact"]
    _write_json(
        base / "n500_tactile_statistics.json",
        {
            "stock_3.11_recorded_baseline": {
                "warp_contact_count_after_step": 2757,
                "qpos_max_abs": 0.0016072189813642335,
                "qvel_max_abs": 0.8036151935956877,
                "touch_max_abs": 74.34978351092225,
            },
            "tet_guard": {
                "initial": n500["snapshots"]["initial"],
                "after_one_step": n500["snapshots"]["final"],
            },
            "interpretation": (
                "The guard removes the 2674 impossible rigid-flex internal contacts, reducing "
                "the settled Warp count from 2757 to 83, but CPU has 89 and the one-step tactile "
                "error remains 74.35. The remaining external geom-flex manifold mismatch is material."
            ),
        },
    )

    figures = base / "figures"
    figures.mkdir(exist_ok=True)
    _svg_counts(figures / "approach_contact_counts.svg", matrices)

    shallow_cpu = cpu["scenarios"]["shallow_penetration"]
    shallow_guard = guard["scenarios"]["shallow_penetration"]
    deep_cpu = cpu["scenarios"]["deep_penetration"]
    deep_guard = guard["scenarios"]["deep_penetration"]
    report = f"""# CPU MuJoCo vs MuJoCo Warp rigid-flex root cause

## Conclusion

This is **Outcome E: multiple interacting differences**. The first concrete divergence is
MuJoCo Warp 3.11 collision candidate generation: it runs an intra-tetrahedron
face/opposite-vertex kernel for a rigid flex even though the compiled model has
`internal=false` and `selfcollide=none`. The larger scientific contact/tactile mismatch
persists after guarding that kernel because the external geom-flex narrowphase is not the
CPU algorithm.

## Exact versions and model

- CPU reference: MuJoCo 3.3.1.
- CPU version control: MuJoCo 3.11.0.
- GPU: MuJoCo Warp 3.11.0, MuJoCo 3.11.0, Warp 1.16.0.
- Exact flex: rigid 3-D Gmsh, 2387 vertices, 12602 edges, 8552 tetrahedra, 3328 shell
  triangles, radius 0.00125 m, `internal=false`, `selfcollide=none`.
- All four installed-backend matrices have identical topology hashes. Warp world vertices
  differ from compiled CPU positions by at most {stock['scenarios']['flex_only_no_probe_contact']['flexvert_xpos_max_abs_error_vs_compiled']:.3g} m.

## First divergence

CPU 3.3.1 and CPU 3.11.0 report zero contacts in the 20 mm separated sphere fixture.
Stock Warp reports 2674 contacts, all `geom=(-1,-1)` and `flex=(0,0)`, before any
external collision exists. They create 2674 constraint rows and contact distances from
-0.001559 m to about -2.81e-7 m.

Installed Warp source launches `_flex_tet_internal_collisions_detect` whenever
`m.nflexelem > 0`. The kernel receives neither `flex_internal` nor `flex_rigid`. CPU
MuJoCo's collision driver first excludes `flex_rigid`, then gates internal work on
`flex_internal`. Replacing only that Warp kernel with a no-op removes exactly 2674
contacts in every probe state and restores zero-contact parity.

## External narrowphase/manifold difference

CPU represents an active 3-D flex element as one tetrahedral CCD object. Its support is the
tetrahedron support inflated by the flex radius, and it uses the convex GJK/EPA path.
Warp 3.11 instead loops over all four faces of every tetrahedron, runs analytic
geom-triangle collision with the radius, and deduplicates candidate positions within 1 mm.
It has no tetrahedral geom-flex broadphase in this path.

At 0.5 mm penetration, CPU 3.11 and guarded Warp both produce
{shallow_cpu['contact']['count']} contacts, {shallow_cpu['constraint_rows']} constraint rows,
and touch {shallow_cpu['touch'][0]:.9g} versus {shallow_guard['touch'][0]:.9g}
(absolute error {abs(shallow_cpu['touch'][0]-shallow_guard['touch'][0]):.3g}). At 2 mm
penetration, CPU produces {deep_cpu['contact']['count']} contacts/{deep_cpu['constraint_rows']}
rows while guarded Warp produces {deep_guard['contact']['count']} contacts/
{deep_guard['constraint_rows']} rows; touch differs by
{abs(deep_cpu['touch'][0]-deep_guard['touch'][0]):.6g}. Thus downstream force/touch agrees
when the contact manifold agrees and diverges when the manifold differs.

## Full N=500 result

At import, CPU and Warp have 89 matched contacts, contact position error below 3e-8 m,
touch max error 4.08e-6, and touch correlation 1.0. After one guarded Warp step, CPU has
89 contacts and Warp has 83. The errors are qpos 0.001607, qvel 0.803615, tactile maximum
74.3498, tactile RMSE 6.85656, correlation 0.830962, active Jaccard 0.9, and total tactile
relative error 74.801%. The false-internal guard therefore fixes the first bug but does not
make the original path scientifically usable.

## Upstream status and fixability

The exact defective kernel is present in the v3.11.0 tag. Upstream commit `c822833`
(`flex-flex collisions`, PR #1496, 2026-08-07) removes that kernel during a flex collision
rewrite; current main `70c4571` no longer emits false contacts in the separated reproducer.
Current main still produces different external manifold counts, so this is not a complete
parity fix. The local guard is diagnostic only. Production should continue rejecting flex
until upstream provides and validates CPU-equivalent rigid 3-D geom-flex narrowphase and
the separate `nJfe=0` rigid-edge write is resolved.

## Artifact map

The JSON/CSV files in this directory contain the complete version matrix, topology and
world-position checks, no-contact attribution, approach curve, first contacts, matched
contact geometry, support semantics, constraints, forces, minimal touch output, and full
N=500 tactile statistics. `figures/approach_contact_counts.svg` visualizes the 2674-contact
offset and its removal.
"""
    (base / "root_cause_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("generated/rigid_flex_cpu_warp_debug"))
    args = parser.parse_args()
    render(args.input_dir)


if __name__ == "__main__":
    main()
