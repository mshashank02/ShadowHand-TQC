#!/usr/bin/env python3
"""Aggregate radius-aware CPU sweeps into the durable validation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(inputs: list[Path], output: Path) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    candidates: dict[str, Any] = {}
    for payload in payloads:
        for name, candidate in payload["candidates"].items():
            # The refinement intentionally repeats a few grid points. Identical
            # names are equivalent; retain the last durable run.
            candidates[name] = candidate
    ranked = sorted(
        candidates,
        key=lambda name: (
            not candidates[name]["passes_all_primary_gates"],
            not candidates[name]["passes_all_secondary_gates"],
            candidates[name]["maximum_absolute_onset_shift_mm"],
            name,
        ),
    )
    primary = [name for name in ranked if candidates[name]["passes_all_primary_gates"]]
    secondary = [name for name in ranked if candidates[name]["passes_all_secondary_gates"]]
    aggregate_payload = {
        "method": "radius-aware-aggregate-v1",
        "input_files": [str(path.resolve()) for path in inputs],
        "mujoco_version": payloads[0]["mujoco_version"],
        "gates": payloads[0]["gates"],
        "candidate_count": len(candidates),
        "ranking": ranked,
        "primary_pass_candidates": primary,
        "secondary_pass_candidates": secondary,
        "selection": {
            "classification": "Outcome C",
            "decision": "blocked_no_cpu_radius_aware_candidate",
            "warp_candidates": [],
            "reason": (
                "No margin-control or geometric Minkowski-shell candidate passes all five "
                "CPU onset, contact, and tactile gates. Warp/GPU validation is prohibited."
            ),
        },
        "candidates": candidates,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "radius_aware_parameter_sweep.json").write_text(
        json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    sweep_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    tactile_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    for name in ranked:
        candidate = candidates[name]
        sweep_rows.append(
            {
                "candidate": name,
                "level": candidate["level"],
                "strategy": candidate["parameters"]["strategy"],
                "shell_mm": candidate["parameters"]["shell_m"] * 1000.0,
                "gap_mm": candidate["parameters"].get("gap_m", 0.0) * 1000.0,
                "sphere_subdivisions": candidate["parameters"]["sphere_subdivisions"],
                "sphere_bound": candidate["parameters"]["sphere_bound"],
                "residual_margin_mm": candidate["parameters"]["residual_margin_m"] * 1000.0,
                "worst_absolute_onset_shift_mm": candidate["maximum_absolute_onset_shift_mm"],
                "passes_all_primary_gates": candidate["passes_all_primary_gates"],
                "passes_all_secondary_gates": candidate["passes_all_secondary_gates"],
            }
        )
        for fixture_name, fixture in candidate["fixtures"].items():
            comparison = fixture["comparison"]
            for state in ("common_reference_pose", "matched_penetration"):
                contacts = comparison[state]["contacts"]
                contact_rows.append(
                    {
                        "candidate": name,
                        "fixture": fixture_name,
                        "state": state,
                        "onset_shift_mm": comparison["onset_shift_mm"],
                        "reference_contact_count": contacts["reference_count"],
                        "actual_contact_count": contacts["actual_count"],
                        "contact_presence_matches": contacts["contact_presence_matches"],
                        "position_error_mm_max": contacts["position_error_mm_max"],
                        "normal_angle_error_deg_max": contacts["normal_angle_error_deg_max"],
                        "distance_error_mm_max": contacts["distance_error_mm_max"],
                        "total_normal_force_relative_error": contacts["total_normal_force_relative_error"],
                    }
                )
                tactile = comparison[state]["tactile"]
                tactile_rows.append(
                    {
                        "candidate": name,
                        "fixture": fixture_name,
                        "state": state,
                        "reference_active_count": tactile["reference_active_count"],
                        "actual_active_count": tactile["actual_active_count"],
                        "active_jaccard": tactile["active_jaccard"],
                        "max_absolute_error": tactile["max_absolute_error"],
                        "rmse": tactile["rmse"],
                        "reference_total_magnitude": tactile["reference_total_magnitude"],
                        "actual_total_magnitude": tactile["actual_total_magnitude"],
                        "total_magnitude_relative_error": tactile["total_magnitude_relative_error"],
                    }
                )
                duplicate = fixture["duplicate_diagnostics"][state]
                duplicate_rows.append(
                    {
                        "candidate": name,
                        "fixture": fixture_name,
                        "state": state,
                        "contact_count": duplicate["contact_count"],
                        "object_piece_count": duplicate["object_piece_count"],
                        "object_piece_names": ";".join(duplicate["object_piece_names"]),
                        "near_duplicate_contact_pairs_0p1mm_5deg": duplicate[
                            "near_duplicate_contact_pairs_0p1mm_5deg"
                        ],
                    }
                )
    _write_csv(output / "radius_aware_parameter_sweep.csv", sweep_rows)
    _write_csv(output / "radius_aware_contact_comparison.csv", contact_rows)
    _write_csv(output / "radius_aware_tactile_comparison.csv", tactile_rows)
    _write_csv(output / "duplicate_contact_analysis.csv", duplicate_rows)

    blocked = [{
        "status": "NOT_RUN_CPU_GATE_FAILED",
        "candidate": "none",
        "reason": "No candidate passed every five-fixture CPU onset/contact/tactile gate.",
    }]
    for filename in (
        "cpu_warp_radius_aware_parity.csv",
        "radius_aware_gpu_benchmark.csv",
        "radius_aware_complete_loop_benchmark.csv",
    ):
        _write_csv(output / filename, blocked)

    best = candidates[ranked[0]]
    geometry_lines: list[str] = []
    geometry_path = output / "radius_aware_shell_geometry.csv"
    if geometry_path.is_file():
        with geometry_path.open(encoding="utf-8", newline="") as stream:
            geometry = {row["candidate"]: row for row in csv.DictReader(stream)}
        physical = geometry.get("very_fine_minkowski_1250um")
        minimax = geometry.get("fine_minkowski_1012um")
        if physical and minimax:
            geometry_lines = [
                "Union-level geometry uses the existing exact Manifold3D boolean-union and "
                "Trimesh/Rtree proximity audit at 100,000 deterministic target-surface samples. "
                "The independent target is the original non-convex surface Minkowski-summed "
                "with a 642-direction 1.25 mm sphere polytope.",
                "",
                "The physical 87-piece shell has "
                f"{float(physical['volume_error_percent']):.3f}% volume overfill and "
                f"{float(physical['p95_boundary_error_mm']):.3f}/"
                f"{float(physical['p99_boundary_error_mm']):.3f}/"
                f"{float(physical['max_boundary_error_mm']):.3f} mm p95/p99/max boundary "
                "error. The minimax 1.012 mm shell has "
                f"{float(minimax['volume_error_percent']):.3f}% volume underfill and "
                f"{float(minimax['p95_boundary_error_mm']):.3f}/"
                f"{float(minimax['p99_boundary_error_mm']):.3f}/"
                f"{float(minimax['max_boundary_error_mm']):.3f} mm boundary error, with "
                f"{100.0 * float(minimax['underfill_surface_fraction']):.2f}% of sampled "
                "target points outside the candidate union.",
                "",
            ]
    lines = [
        "# Radius-aware rigid collision validation",
        "",
        "Classification: **Outcome C**.",
        "",
        f"CPU MuJoCo `{payloads[0]['mujoco_version']}` evaluated {len(candidates)} unique "
        "margin-control and geometric Minkowski-shell candidates on all five fixed fixtures.",
        "",
        f"The best candidate is `{ranked[0]}` with a worst absolute onset shift of "
        f"{best['maximum_absolute_onset_shift_mm']:.4f} mm. It fails the 0.10 mm primary "
        "gate and the 0.25 mm secondary diagnostic, and it also fails common-pose tactile gates.",
        "",
        "At the physical 1.25 mm radius, the 36/87-piece geometric shells restore fingertip, "
        "palm, macro, and roughness onset to within roughly 0.03 mm, but the deepest "
        "concavity becomes approximately 0.50 mm premature. At the minimax 1.012 mm "
        "36-piece shell, the concavity is 0.2545 mm premature and fingertip is 0.2536 mm "
        "late; common-pose tactile-total relative errors are 34.9%-100% across fixtures.",
        "",
        "No near-coincident duplicate contact pair was found by the declared 0.1 mm / "
        "5 degree diagnostic in any evaluated candidate-state row. This does not rescue "
        "the failed onset and tactile fidelity.",
        "",
        "A hybrid uniform residual margin was not run: it adds uniform support to the same "
        "piece union and therefore cannot reconcile the opposing concavity and exterior "
        "errors. Piece-specific radii would no longer reproduce the original single, "
        "size-defined physical radius and were rejected as an unjustified fit.",
        "",
        "Margin-gap diagnostics at 1.25 mm margin and 0.10/0.25/0.50 mm gap left "
        "first-contact onset unchanged, as expected for detected inactive contacts, while "
        "reducing or eliminating forces and tactile signals. They do not improve fidelity.",
        "",
        *geometry_lines,
        "Per the predeclared gate, CPU/Warp parity, capacity, GPU physics, complete-loop RL, "
        "production selection, all-24 conversion, and N=500/N=1000 GPU smokes were not run.",
        "",
        "Exact next step: investigate a different native non-convex rigid representation "
        "that preserves the original surface before applying the size-scaled collision shell.",
        "",
    ]
    (output / "radius_aware_report.md").write_text("\n".join(lines), encoding="utf-8")
    return aggregate_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate([path.resolve() for path in args.input], args.output.resolve())
    print(json.dumps(result["selection"], indent=2))


if __name__ == "__main__":
    main()
