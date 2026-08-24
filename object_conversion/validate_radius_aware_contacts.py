#!/usr/bin/env python3
"""Sweep radius-aware rigid collision candidates on the five CPU fixtures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from object_conversion.radius_aware_collision import (
    RADIUS_AWARE_CONVERTER_VERSION,
    ShellParameters,
    build_radius_aware_model,
)
from object_conversion.validate_decomposition_contacts import (
    EVALUATION_PENETRATION_M,
    FEATURE_NAMES,
    FIDELITY_GATES,
    HAND_FIXTURES,
    _comparison,
    _run_feature_model,
    _run_hand_model,
)


VALIDATOR_VERSION = "radius-aware-contact-sweep-v1"
PRIMARY_ONSET_GATE_MM = 0.10
SECONDARY_ONSET_GATE_MM = 0.25


def _duplicate_diagnostics(contacts: list[dict[str, Any]]) -> dict[str, Any]:
    object_names = [
        name
        for contact in contacts
        for name in contact["geom_names"]
        if name and name.startswith("object_collision_")
    ]
    near_duplicate_pairs = 0
    for left in range(len(contacts)):
        for right in range(left + 1, len(contacts)):
            position_delta = np.linalg.norm(
                np.asarray(contacts[left]["position_m"])
                - np.asarray(contacts[right]["position_m"])
            )
            normal_dot = abs(
                float(
                    np.dot(
                        np.asarray(contacts[left]["normal"]),
                        np.asarray(contacts[right]["normal"]),
                    )
                )
            )
            if position_delta <= 0.0001 and normal_dot >= np.cos(np.deg2rad(5.0)):
                near_duplicate_pairs += 1
    return {
        "contact_count": len(contacts),
        "object_piece_names": sorted(set(object_names)),
        "object_piece_count": len(set(object_names)),
        "near_duplicate_contact_pairs_0p1mm_5deg": near_duplicate_pairs,
    }


def _gate_summary(comparison: dict[str, Any], onset_limit_mm: float) -> dict[str, bool]:
    gates = dict(comparison["gate_results"])
    gates["onset"] = comparison["absolute_onset_shift_mm"] <= onset_limit_mm
    return gates


def _candidate_name(level: str, parameters: ShellParameters) -> str:
    shell_um = round(parameters.shell_m * 1e6)
    residual_um = round(parameters.residual_margin_m * 1e6)
    gap_um = round(parameters.gap_m * 1e6)
    suffix = f"_residual_{residual_um:04d}um" if residual_um else ""
    if gap_um:
        suffix += f"_gap_{gap_um:04d}um"
    return f"{level}_{parameters.strategy}_{shell_um:04d}um{suffix}"


def run_sweep(
    *,
    models_path: Path,
    features_path: Path,
    output: Path,
    levels: list[str],
    parameter_sets: list[ShellParameters],
) -> dict[str, Any]:
    import mujoco

    source_models = json.loads(models_path.read_text(encoding="utf-8"))
    features = json.loads(features_path.read_text(encoding="utf-8"))["features"]
    missing = set(levels) - set(source_models)
    if missing:
        raise ValueError(f"models manifest is missing {sorted(missing)}")
    output.mkdir(parents=True, exist_ok=True)
    models_output = output / "models"

    reference_path = Path(source_models["rigid_flex_reference"])
    references: dict[str, dict[str, Any]] = {}
    common_coordinates: dict[str, float] = {}
    for fixture_name, fixture in HAND_FIXTURES.items():
        reference = _run_hand_model(mujoco, reference_path, fixture, None)
        common = reference["onset_coordinate_m"] - EVALUATION_PENETRATION_M
        reference["common_reference_pose"] = reference["matched_penetration"]
        references[fixture_name] = reference
        common_coordinates[fixture_name] = common
    for fixture_name in FEATURE_NAMES:
        reference = _run_feature_model(mujoco, reference_path, features[fixture_name], None)
        common = reference["onset_coordinate_m"] - EVALUATION_PENETRATION_M
        reference["common_reference_pose"] = reference["matched_penetration"]
        references[fixture_name] = reference
        common_coordinates[fixture_name] = common

    candidates: dict[str, Any] = {}
    for parameters in parameter_sets:
        for level in levels:
            name = _candidate_name(level, parameters)
            xml_path = models_output / f"{name}.xml"
            manifest = build_radius_aware_model(source_models[level], xml_path, parameters)
            fixture_results: dict[str, Any] = {}
            for fixture_name, fixture in HAND_FIXTURES.items():
                result = _run_hand_model(
                    mujoco, xml_path, fixture, common_coordinates[fixture_name]
                )
                comparison = _comparison(references[fixture_name], result)
                fixture_results[fixture_name] = {
                    "fixture_type": "n500_hand",
                    "result": result,
                    "comparison": comparison,
                    "duplicate_diagnostics": {
                        state: _duplicate_diagnostics(result[state]["contacts"])
                        for state in ("matched_penetration", "common_reference_pose")
                    },
                    "primary_gate_results": _gate_summary(comparison, PRIMARY_ONSET_GATE_MM),
                    "secondary_gate_results": _gate_summary(comparison, SECONDARY_ONSET_GATE_MM),
                }
            for fixture_name in FEATURE_NAMES:
                result = _run_feature_model(
                    mujoco,
                    xml_path,
                    features[fixture_name],
                    common_coordinates[fixture_name],
                )
                comparison = _comparison(references[fixture_name], result)
                fixture_results[fixture_name] = {
                    "fixture_type": "diagnostic_probe",
                    "result": result,
                    "comparison": comparison,
                    "duplicate_diagnostics": {
                        state: _duplicate_diagnostics(result[state]["contacts"])
                        for state in ("matched_penetration", "common_reference_pose")
                    },
                    "primary_gate_results": _gate_summary(comparison, PRIMARY_ONSET_GATE_MM),
                    "secondary_gate_results": _gate_summary(comparison, SECONDARY_ONSET_GATE_MM),
                }

            primary_pass = all(
                all(fixture["primary_gate_results"].values())
                for fixture in fixture_results.values()
            )
            secondary_pass = all(
                all(fixture["secondary_gate_results"].values())
                for fixture in fixture_results.values()
            )
            candidates[name] = {
                "level": level,
                "parameters": manifest["parameters"],
                "model_xml": str(xml_path),
                "model_manifest": str(xml_path.with_suffix(".manifest.json")),
                "fixtures": fixture_results,
                "maximum_absolute_onset_shift_mm": max(
                    fixture["comparison"]["absolute_onset_shift_mm"]
                    for fixture in fixture_results.values()
                ),
                "passes_all_primary_gates": primary_pass,
                "passes_all_secondary_gates": secondary_pass,
            }

    ranked = sorted(
        candidates,
        key=lambda name: (
            not candidates[name]["passes_all_primary_gates"],
            not candidates[name]["passes_all_secondary_gates"],
            candidates[name]["maximum_absolute_onset_shift_mm"],
            name,
        ),
    )
    payload = {
        "validator_version": VALIDATOR_VERSION,
        "converter_version": RADIUS_AWARE_CONVERTER_VERSION,
        "mujoco_version": mujoco.__version__,
        "models_manifest": str(models_path.resolve()),
        "feature_definitions": str(features_path.resolve()),
        "reference_model": str(reference_path.resolve()),
        "evaluation_penetration_m": EVALUATION_PENETRATION_M,
        "gates": {
            **FIDELITY_GATES,
            "primary_absolute_onset_shift_mm_max": PRIMARY_ONSET_GATE_MM,
            "secondary_absolute_onset_shift_mm_max": SECONDARY_ONSET_GATE_MM,
        },
        "references": references,
        "candidates": candidates,
        "ranking": ranked,
        "primary_pass_candidates": [
            name for name in ranked if candidates[name]["passes_all_primary_gates"]
        ],
        "secondary_pass_candidates": [
            name for name in ranked if candidates[name]["passes_all_secondary_gates"]
        ],
    }
    json_path = output / "radius_aware_cpu_validation.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary(output / "radius_aware_cpu_summary.csv", payload)
    _write_report(output / "radius_aware_cpu_report.md", payload)
    return payload


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for name, candidate in payload["candidates"].items():
        for fixture_name, fixture in candidate["fixtures"].items():
            comparison = fixture["comparison"]
            common = comparison["common_reference_pose"]
            matched = comparison["matched_penetration"]
            rows.append(
                {
                    "candidate": name,
                    "level": candidate["level"],
                    "strategy": candidate["parameters"]["strategy"],
                    "shell_mm": candidate["parameters"]["shell_m"] * 1000.0,
                    "residual_margin_mm": candidate["parameters"]["residual_margin_m"] * 1000.0,
                    "fixture": fixture_name,
                    "onset_shift_mm": comparison["onset_shift_mm"],
                    "matched_position_error_mm_max": matched["contacts"]["position_error_mm_max"],
                    "matched_normal_angle_error_deg_max": matched["contacts"]["normal_angle_error_deg_max"],
                    "common_contact_presence_matches": common["contacts"]["contact_presence_matches"],
                    "common_active_jaccard": common["tactile"]["active_jaccard"],
                    "common_total_tactile_relative_error": common["tactile"]["total_magnitude_relative_error"],
                    "matched_near_duplicate_pairs": fixture["duplicate_diagnostics"]["matched_penetration"]["near_duplicate_contact_pairs_0p1mm_5deg"],
                    "common_near_duplicate_pairs": fixture["duplicate_diagnostics"]["common_reference_pose"]["near_duplicate_contact_pairs_0p1mm_5deg"],
                    "passes_primary_fixture_gates": all(fixture["primary_gate_results"].values()),
                    "passes_secondary_fixture_gates": all(fixture["secondary_gate_results"].values()),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Radius-aware rigid collision CPU validation",
        "",
        f"CPU MuJoCo version: `{payload['mujoco_version']}`.",
        "",
        "Primary onset tolerance is 0.10 mm; 0.25 mm is retained as a secondary diagnostic. "
        "Every pre-existing contact and tactile gate must also pass.",
        "",
        "| Candidate | Worst onset shift mm | Primary all-fixture gate | Secondary all-fixture gate |",
        "|---|---:|---|---|",
    ]
    for name in payload["ranking"]:
        candidate = payload["candidates"][name]
        lines.append(
            f"| {name} | {candidate['maximum_absolute_onset_shift_mm']:.4f} | "
            f"{'PASS' if candidate['passes_all_primary_gates'] else 'FAIL'} | "
            f"{'PASS' if candidate['passes_all_secondary_gates'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Primary pass candidates: " + (", ".join(payload["primary_pass_candidates"]) or "none"),
            "",
            "Secondary pass candidates: " + (", ".join(payload["secondary_pass_candidates"]) or "none"),
            "",
            "Candidate ranking is deterministic: primary pass, secondary pass, worst absolute "
            "onset shift, then candidate name. Duplicate diagnostics count near-coincident "
            "contacts within 0.1 mm and 5 degrees at the common and matched poses.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[1]
    base = repository / "generated/convex_decomposition_validation"
    parser.add_argument("--models", type=Path, default=base / "contact_models/models.json")
    parser.add_argument("--features", type=Path, default=base / "contact_feature_definitions.json")
    parser.add_argument("--output", type=Path, default=base / "radius_aware")
    parser.add_argument("--levels", nargs="+", default=["coarse", "fine", "very_fine"])
    parser.add_argument("--strategy", choices=("margin", "minkowski", "hybrid"), required=True)
    parser.add_argument("--shell-mm", nargs="+", type=float, required=True)
    parser.add_argument("--residual-margin-mm", type=float, default=0.0)
    parser.add_argument("--gap-mm", type=float, default=0.0)
    parser.add_argument("--sphere-subdivisions", type=int, default=2)
    parser.add_argument("--sphere-bound", choices=("inscribed", "circumscribed"), default="circumscribed")
    args = parser.parse_args()
    parameter_sets = [
        ShellParameters(
            strategy=args.strategy,
            shell_m=value / 1000.0,
            gap_m=args.gap_mm / 1000.0,
            residual_margin_m=args.residual_margin_mm / 1000.0,
            sphere_subdivisions=args.sphere_subdivisions,
            sphere_bound=args.sphere_bound,
        )
        for value in args.shell_mm
    ]
    payload = run_sweep(
        models_path=args.models.resolve(),
        features_path=args.features.resolve(),
        output=args.output.resolve(),
        levels=args.levels,
        parameter_sets=parameter_sets,
    )
    print(json.dumps({
        "primary_pass_candidates": payload["primary_pass_candidates"],
        "secondary_pass_candidates": payload["secondary_pass_candidates"],
        "ranking": payload["ranking"],
    }, indent=2))


if __name__ == "__main__":
    main()
