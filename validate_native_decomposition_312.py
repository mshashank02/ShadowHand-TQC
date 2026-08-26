#!/usr/bin/env python3
"""Validate 36/87-piece native-rigid CoACD models on matched MuJoCo 3.12."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch

from benchmark_task import benchmark_one
from shadowhand_gpu.native_decomposition_validation import validate_representation


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "generated/convex_decomposition_validation"
DEFAULT_OUTPUT = ROOT / "generated/native_decomposition_312_validation"
REPRESENTATIONS = {
    "36": {
        "level": "fine",
        "pieces": 36,
        "directory": "collision_fine_500_1.071429_0.714286",
        "xml": "manipulate_collision_fine_touch_sensors_500_1.071429_0.714286.xml",
    },
    "87": {
        "level": "very_fine",
        "pieces": 87,
        "directory": "collision_very_fine_500_1.071429_0.714286",
        "xml": "manipulate_collision_very_fine_touch_sensors_500_1.071429_0.714286.xml",
    },
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _geometry_rows() -> dict[str, dict[str, Any]]:
    with (SOURCE / "geometry_comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["level"]: row for row in rows if row["level"] in {"fine", "very_fine"}}


def _fixture_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        for fixture_name, fixture in result["fixtures"]["fixtures"].items():
            if fixture_name == "separated_contact":
                comparison = fixture["comparison"]
                rows.append(
                    {
                        "representation": name,
                        "pieces": result["piece_count"],
                        "fixture": fixture_name,
                        "cpu_onset_m": "",
                        "warp_onset_m": "",
                        "onset_difference_mm": "",
                        "cpu_contacts": len(fixture["cpu"]["contacts"]),
                        "warp_contacts": len(fixture["warp"]["contacts"]),
                        "cpu_constraint_rows": fixture["cpu"]["constraint_rows"],
                        "warp_constraint_rows": fixture["warp"]["constraint_rows"],
                        "position_error_mm_max": comparison["contacts"]["position_error_mm_max"],
                        "normal_error_deg_max": comparison["contacts"]["normal_angle_error_deg_max"],
                        "force_error_max": comparison["contacts"]["normal_force_error_max"],
                        "cpu_total_touch": fixture["cpu"]["tactile"]["total_magnitude"],
                        "warp_total_touch": fixture["warp"]["tactile"]["total_magnitude"],
                        "total_touch_relative_error": comparison["tactile"]["total_magnitude_relative_error"],
                        "touch_max_error": comparison["tactile"]["max_absolute_error"],
                        "correlation": comparison["tactile"]["correlation"],
                        "cosine_similarity": comparison["tactile"]["cosine_similarity"],
                        "active_sensor_jaccard": comparison["tactile"]["active_jaccard"],
                        "geom_pair_multisets_match": comparison["geom_pair_multisets_match"],
                        "overflow_flags": fixture["warp"]["capacity"]["overflow_flags"],
                        "passed": fixture["passed"],
                    }
                )
                continue
            pose = fixture["cpu_relative_pose"]
            comparison = pose["comparison"]
            rows.append(
                {
                    "representation": name,
                    "pieces": result["piece_count"],
                    "fixture": fixture_name,
                    "cpu_onset_m": fixture["cpu_onset_coordinate_m"],
                    "warp_onset_m": fixture["warp_onset_coordinate_m"],
                    "onset_difference_mm": fixture["onset_difference_mm"],
                    "cpu_contacts": len(pose["cpu"]["contacts"]),
                    "warp_contacts": len(pose["warp"]["contacts"]),
                    "cpu_constraint_rows": pose["cpu"]["constraint_rows"],
                    "warp_constraint_rows": pose["warp"]["constraint_rows"],
                    "position_error_mm_max": comparison["contacts"]["position_error_mm_max"],
                    "normal_error_deg_max": comparison["contacts"]["normal_angle_error_deg_max"],
                    "force_error_max": comparison["contacts"]["normal_force_error_max"],
                    "cpu_total_touch": pose["cpu"]["tactile"]["total_magnitude"],
                    "warp_total_touch": pose["warp"]["tactile"]["total_magnitude"],
                    "total_touch_relative_error": comparison["tactile"]["total_magnitude_relative_error"],
                    "touch_max_error": comparison["tactile"]["max_absolute_error"],
                    "correlation": comparison["tactile"]["correlation"],
                    "cosine_similarity": comparison["tactile"]["cosine_similarity"],
                    "active_sensor_jaccard": comparison["tactile"]["active_jaccard"],
                    "geom_pair_multisets_match": comparison["geom_pair_multisets_match"],
                    "overflow_flags": pose["warp"]["capacity"]["overflow_flags"],
                    "passed": fixture["passed"],
                }
            )
    return rows


def _summary_rows(results: dict[str, Any], geometry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, result in results.items():
        level = REPRESENTATIONS[name]["level"]
        n500 = result["n500_one_step"]
        policy = result["policy_action_20_substeps"]
        rows.append(
            {
                "representation": name,
                "pieces": result["piece_count"],
                "parity_passed": result["parity_passed"],
                "model_contract_passed": result["model_audit"]["passed"],
                "fixtures_passed": result["fixtures"]["passed"],
                "n500_passed": n500["passed"],
                "n500_qpos_max_abs": n500["qpos"]["max_abs"],
                "n500_qvel_max_abs": n500["qvel"]["max_abs"],
                "n500_touch_max_error": n500["tactile_metrics"]["max_absolute_error"],
                "n500_total_touch_relative_error": n500["tactile_metrics"]["total_magnitude_relative_error"],
                "n500_contact_geom_pairs_match": n500["contact_geom_pairs_match"],
                "policy_passed": policy["passed"],
                "policy_qpos_max_abs": policy["qpos"]["max_abs"],
                "policy_qvel_max_abs": policy["qvel"]["max_abs"],
                "policy_touch_max_error": policy["touch"]["max_absolute_error"],
                "policy_reward_exact": policy["gate_results"]["reward_exact"],
                "policy_success_exact": policy["gate_results"]["success_exact"],
                "volume_error_percent": geometry[level]["volume_error_percent"],
                "p95_gap_mm": geometry[level]["p95_gap_mm"],
                "p99_gap_mm": geometry[level]["p99_gap_mm"],
                "max_gap_mm": geometry[level]["max_gap_mm"],
            }
        )
    return rows


def _classification(results: dict[str, Any]) -> str:
    passed = {name for name, result in results.items() if result["parity_passed"]}
    if passed == {"36", "87"}:
        return "PASS_BOTH"
    if passed == {"36"}:
        return "PASS_36"
    if passed == {"87"}:
        return "PASS_87"
    return "FAIL_NEW_RIGID_DEFINITION"


def _report(payload: dict[str, Any]) -> str:
    lines = [
        "# Native-rigid CoACD MuJoCo/Warp 3.12 validation",
        "",
        f"Conclusion: **{payload['conclusion']}**.",
        "",
        "This is a new bare-decomposed-surface collision definition. Historical rigid-flex results were not pooled.",
        "",
        "| pieces | contract | fixtures | N=500 | 20-substep | geometry p95 / p99 / max (mm) | volume error |",
        "|---:|:---:|:---:|:---:|:---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lines.append(
            f"| {row['pieces']} | {'PASS' if row['model_contract_passed'] else 'FAIL'} | "
            f"{'PASS' if row['fixtures_passed'] else 'FAIL'} | {'PASS' if row['n500_passed'] else 'FAIL'} | "
            f"{'PASS' if row['policy_passed'] else 'FAIL'} | {float(row['p95_gap_mm']):.3f} / "
            f"{float(row['p99_gap_mm']):.3f} / {float(row['max_gap_mm']):.3f} | "
            f"{float(row['volume_error_percent']):.3f}% |"
        )
    lines += ["", f"Recommendation: **{payload['recommendation']}**.", ""]
    if payload["benchmarks"]:
        benchmarked = ", ".join(
            sorted({f"{item['pieces']}-piece" for item in payload["benchmarks"]})
        )
        lines += [
            f"GPU throughput and memory were benchmarked for the parity-passing {benchmarked} representation(s) only. See `gpu_benchmarks.csv` for every batch size.",
            "",
        ]
    else:
        lines += ["GPU benchmarks were not run because the parity prerequisite did not pass for both representations.", ""]
    if payload["conclusion"] != "FAIL_NEW_RIGID_DEFINITION":
        lines += [
            "Any selected passing representation starts a new experiment: all 24 objects and every sensor configuration must be regenerated/validated as applicable and retrained. It is not compatible with historical rigid-flex training results.",
            "",
        ]
    lines += [
        "The production default was not changed, and full RL training was not run.",
        "",
        "MuJoCo 3.12 MULTICCD and NATIVECCD were disabled identically on CPU and Warp because Warp 3.12 cannot transfer the unchanged hand model's non-zero-margin self-collision pairs with MULTICCD enabled. Per-geom object contact parameters were unchanged.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark-worlds", default="1,16,64")
    parser.add_argument("--benchmark-warmup", type=int, default=2)
    parser.add_argument("--benchmark-steps", type=int, default=5)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    geometry = _geometry_rows()
    features = SOURCE / "contact_feature_definitions.json"
    results: dict[str, Any] = {}
    for name, config in REPRESENTATIONS.items():
        directory = SOURCE / "contact_models" / config["directory"]
        results[name] = validate_representation(
            name,
            directory / config["xml"],
            directory / "rigid_object_representation.json",
            features,
            config["pieces"],
        )
        _write_json(output / f"representation_{name}.json", results[name])

    conclusion = _classification(results)
    benchmarks: list[dict[str, Any]] = []
    worlds_values = [int(value) for value in args.benchmark_worlds.split(",")]
    for name, result in results.items():
        if result["parity_passed"]:
            for worlds in worlds_values:
                item = benchmark_one(
                    Path(result["compatible_xml"]),
                    worlds,
                    warmup_steps=args.benchmark_warmup,
                    measured_steps=args.benchmark_steps,
                    contacts_per_world=2048,
                    constraints_per_world=4096,
                    device="cuda:0",
                    use_cuda_graphs=True,
                )
                item["representation"] = name
                item["pieces"] = result["piece_count"]
                item["device_used_at_report_bytes"] = (
                    item["device_total_bytes"] - item["device_free_after_measurement_bytes"]
                )
                benchmarks.append(item)
                torch.cuda.empty_cache()

    recommendation = "none"
    if conclusion == "PASS_BOTH":
        recommendation = "87 pieces, because parity is tied and it preserves geometry better"
    elif conclusion == "PASS_36":
        recommendation = "36 pieces (only passing representation)"
    elif conclusion == "PASS_87":
        recommendation = "87 pieces (only passing representation)"
    summary = _summary_rows(results, geometry)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "study": "new_native_rigid_bare_coacd_surface",
        "historical_rigid_flex_results_pooled": False,
        "object": "large/high-macro/high-roughness",
        "runtime_requirement": "MuJoCo 3.12.0 and MuJoCo Warp 3.12.0",
        "production_default_changed": False,
        "full_rl_training_run": False,
        "conclusion": conclusion,
        "recommendation": recommendation,
        "summary": summary,
        "representations": results,
        "benchmarks": benchmarks,
        "passing_representation_requires_retraining_all_24_objects_and_sensor_configurations": conclusion
        != "FAIL_NEW_RIGID_DEFINITION",
    }
    _write_json(output / "validation_results.json", payload)
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "fixtures.csv", _fixture_rows(results))
    benchmark_rows = [
        {
            "representation": item["representation"],
            "pieces": item["pieces"],
            "worlds": item["worlds"],
            "environment_steps_per_second": item["environment_steps_per_second"],
            "physics_world_steps_per_second": item["physics_world_steps_per_second"],
            "all_world_reset_seconds": item["all_world_reset_seconds"],
            "allocated_before_measurement_bytes": item["allocated_before_measurement_bytes"],
            "peak_allocated_bytes": item["peak_allocated_bytes"],
            "device_memory_used_delta_bytes": item["device_memory_used_delta_bytes"],
            "device_used_at_report_bytes": item["device_used_at_report_bytes"],
            "batch_global_active_contacts_high_water": item[
                "batch_global_active_contacts_high_water"
            ],
            "constraints_per_world_high_water": item["constraints_per_world_high_water"],
            "overflow_flags_max": item["overflow_flags_max"],
        }
        for item in benchmarks
    ]
    _write_csv(output / "gpu_benchmarks.csv", benchmark_rows)
    (output / "report.md").write_text(_report(payload), encoding="utf-8")
    print(json.dumps({"conclusion": conclusion, "recommendation": recommendation}, indent=2))
    return 0 if conclusion != "FAIL_NEW_RIGID_DEFINITION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
