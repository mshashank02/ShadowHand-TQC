#!/usr/bin/env python3
"""Checkpointable worst-object CoACD sweep and union-fidelity report."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from object_conversion.audit_convexity import (
    DEFAULT_SEED,
    MANUAL_OBJECT_ID,
    discover_study_objects,
    gap_statistics,
    hull_gaps,
)
from object_conversion.audit_decomposition import audit_piece_union, closest_union_surface
from object_conversion.convex_decomposition import (
    CoACDParameters,
    decompose_surface_cached,
    load_convex_piece_obj,
    load_decomposition_pieces,
)
from object_conversion.gmsh_to_rigid_surface import (
    convert_gmsh_to_rigid_surface,
    extract_exterior_surface,
    parse_gmsh_v2,
)


SWEEP_VERSION = "worst-object-coacd-sweep-v1"
SWEEP_LEVELS: dict[str, CoACDParameters] = {
    # One fixed search configuration isolates the real-metric threshold as the
    # independent variable.  Piece ranges are targets and measured, never assumed.
    "very_coarse": CoACDParameters(0.00400, resolution=1000, mcts_iterations=100),
    "coarse": CoACDParameters(0.00200, resolution=1000, mcts_iterations=100),
    # The uncapped 1 mm pilot produced 164 pieces.  A 24-piece capped candidate and
    # tuned uncapped intermediate thresholds sample the requested practical band.
    "medium": CoACDParameters(
        0.00100, max_convex_hull=24, resolution=1000, mcts_iterations=100
    ),
    "fine": CoACDParameters(0.00140, resolution=1000, mcts_iterations=100),
    "very_fine": CoACDParameters(0.00120, resolution=1000, mcts_iterations=100),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, name: str) -> None:
    lines = [f"o {name}"]
    lines.extend("v " + " ".join(format(float(x), ".17g") for x in v) for v in vertices)
    lines.extend("f " + " ".join(str(int(x) + 1) for x in f) for f in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _worst_object(repository: Path) -> dict[str, Any]:
    rows = discover_study_objects(repository / "study_objects/sphere_study_v1/manifest.csv")
    return next(row for row in rows if row["object_id"] == MANUAL_OBJECT_ID)


def run_level(
    repository: Path,
    output: Path,
    level: str,
    *,
    samples: int,
) -> dict[str, Any]:
    row = _worst_object(repository)
    source = Path(row["source_path"])
    scale = float(row["scale_m_per_source_unit"])
    surface = extract_exterior_surface(parse_gmsh_v2(source))
    vertices_source = np.asarray(surface.vertices, dtype=np.float64)
    faces = np.asarray(surface.faces, dtype=np.int64)
    conversion = convert_gmsh_to_rigid_surface(
        source, repository / "generated/rigid_mesh_cache"
    )
    parameters = SWEEP_LEVELS[level]
    decomposition = decompose_surface_cached(
        source_path=source,
        exterior_path=conversion.mesh_path,
        vertices_source_units=vertices_source,
        faces=faces,
        scale_m_per_source_unit=(scale, scale, scale),
        cache_root=repository / "generated/rigid_mesh_cache/decomposition",
        parameters=parameters,
    )
    pieces = load_decomposition_pieces(decomposition)
    metrics, arrays = audit_piece_union(
        vertices_source * scale, faces, pieces, samples=samples, seed=DEFAULT_SEED
    )
    level_dir = output / "levels" / level
    level_dir.mkdir(parents=True, exist_ok=True)
    _write_obj(
        level_dir / "boolean_union.obj",
        arrays["union_vertices"],
        arrays["union_faces"],
        f"{level}_boolean_union",
    )
    np.savez_compressed(
        level_dir / "geometry_arrays.npz",
        **{key: value for key, value in arrays.items() if key not in {"union_vertices", "union_faces"}},
    )
    payload = {
        "sweep_version": SWEEP_VERSION,
        "object_id": MANUAL_OBJECT_ID,
        "source_path": str(source),
        "level": level,
        "target_piece_range": {
            "very_coarse": [4, 8],
            "coarse": [8, 16],
            "medium": [16, 32],
            "fine": [32, 64],
            "very_fine": [64, 128],
        }[level],
        "parameters": asdict(parameters),
        "decomposition_manifest": str(decomposition.manifest_path),
        "decomposition_cache_key": decomposition.cache_key,
        "decomposition_cache_reused": decomposition.cache_reused,
        "piece_count": len(pieces),
        "geometry_metrics": metrics,
        "union_mesh": str(level_dir / "boolean_union.obj"),
        "geometry_arrays": str(level_dir / "geometry_arrays.npz"),
    }
    _write_json(level_dir / "result.json", payload)
    return payload


def _single_hull_row(repository: Path) -> dict[str, Any]:
    payload = json.loads(
        (repository / f"generated/convexity_audit/{MANUAL_OBJECT_ID}.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = payload.get("metrics")
    if metrics is None:
        objects = payload.get("objects", [])
        if len(objects) != 1:
            raise ValueError("expected one object in per-object convexity audit")
        metrics = objects[0]
    return {
        "representation": "single_hull",
        "level": "single_hull",
        "pieces": 1,
        "threshold_m": "",
        "volume_error_percent": metrics["volume_inflation_percent"],
        "mean_gap_mm": metrics["mean_gap_mm"],
        "median_gap_mm": metrics["median_gap_mm"],
        "rms_gap_mm": metrics["rms_gap_mm"],
        "p90_gap_mm": metrics["p90_gap_mm"],
        "p95_gap_mm": metrics["p95_gap_mm"],
        "p99_gap_mm": metrics["p99_gap_mm"],
        "p999_gap_mm": metrics["p999_gap_mm"],
        "max_gap_mm": metrics["max_gap_mm"],
        "max_gap_normalized_percent": metrics["max_gap_normalized_percent"],
        **{
            f"surface_fraction_gt_{value}": metrics[f"surface_fraction_gt_{value}"]
            for value in ("0p01mm", "0p05mm", "0p10mm", "0p25mm", "0p50mm", "1p00mm")
        },
    }


def _candidate_row(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["geometry_metrics"]
    return {
        "representation": "convex_decomposition",
        "level": payload["level"],
        "pieces": payload["piece_count"],
        "threshold_m": payload["parameters"]["threshold_m"],
        "volume_error_percent": metrics["volume_error_percent"],
        **{
            name: metrics[name]
            for name in (
                "mean_gap_mm", "median_gap_mm", "rms_gap_mm", "p90_gap_mm",
                "p95_gap_mm", "p99_gap_mm", "p999_gap_mm", "max_gap_mm",
                "max_gap_normalized_percent", "surface_fraction_gt_0p01mm",
                "surface_fraction_gt_0p05mm", "surface_fraction_gt_0p10mm",
                "surface_fraction_gt_0p25mm", "surface_fraction_gt_0p50mm",
                "surface_fraction_gt_1p00mm",
            )
        },
        "source_inside_union_fraction": metrics["sampled_source_inside_union_fraction"],
        "underfill_fraction": metrics["sampled_underfill_fraction"],
    }


def _plot_heatmaps(repository: Path, output: Path, payloads: dict[str, dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    row = _worst_object(repository)
    scale = float(row["scale_m_per_source_unit"])
    surface = extract_exterior_surface(parse_gmsh_v2(Path(row["source_path"])))
    vertices = np.asarray(surface.vertices, dtype=np.float64) * scale
    faces = np.asarray(surface.faces, dtype=np.int64)
    single = hull_gaps(vertices, ConvexHull(vertices)) * 1000.0
    selected = [name for name in ("very_coarse", "coarse", "medium", "fine", "very_fine") if name in payloads]
    plots: list[tuple[str, np.ndarray]] = [("single_hull", single)]
    for name in selected:
        arrays = np.load(output / "levels" / name / "geometry_arrays.npz")
        plots.append((name, np.asarray(arrays["vertex_gaps"]) * 1000.0))
    vmax = float(single.max())
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    views = ((25, -55), (20, 65), (80, -90))
    figure = plt.figure(figsize=(12, 3.2 * len(plots)), constrained_layout=True)
    for row_index, (name, gaps) in enumerate(plots):
        for column, (elevation, azimuth) in enumerate(views):
            axis = figure.add_subplot(len(plots), len(views), row_index * len(views) + column + 1, projection="3d")
            surface_plot = axis.plot_trisurf(
                vertices[:, 0] * 1000.0,
                vertices[:, 1] * 1000.0,
                vertices[:, 2] * 1000.0,
                triangles=faces,
                linewidth=0,
                antialiased=False,
                shade=False,
                cmap="magma",
                norm=norm,
            )
            surface_plot.set_array(gaps[faces].mean(axis=1))
            surface_plot.set_clim(0.0, vmax)
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_axis_off()
            axis.set_title(f"{name} — view {column + 1}")
    bar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"), ax=figure.axes, shrink=0.6)
    bar.set_label("original-surface to collision-boundary error (mm)")
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "worst_object_same_scale_gap_heatmaps.png", dpi=180)
    plt.close(figure)


def _plot_decomposition_pieces(output: Path, payloads: dict[str, dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [name for name in ("very_coarse", "coarse", "fine", "very_fine") if name in payloads]
    views = ((25, -55), (20, 65), (80, -90))
    figure = plt.figure(figsize=(12, 3.4 * len(selected)), constrained_layout=True)
    for row_index, name in enumerate(selected):
        manifest_path = Path(payloads[name]["decomposition_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pieces = [
            load_convex_piece_obj(manifest_path.parent / record["file"])
            for record in manifest["pieces"]
        ]
        all_vertices = np.vstack([piece.vertices_m for piece in pieces]) * 1000.0
        center = 0.5 * (all_vertices.min(axis=0) + all_vertices.max(axis=0))
        half_span = 0.55 * float(np.max(np.ptp(all_vertices, axis=0)))
        for column, (elevation, azimuth) in enumerate(views):
            axis = figure.add_subplot(
                len(selected), len(views), row_index * len(views) + column + 1,
                projection="3d",
            )
            for index, piece in enumerate(pieces):
                vertices = piece.vertices_m * 1000.0
                axis.plot_trisurf(
                    vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=piece.faces,
                    color=plt.cm.tab20(index % 20),
                    linewidth=0.04,
                    edgecolor=(0.1, 0.1, 0.1, 0.15),
                    antialiased=True,
                    shade=True,
                )
            for setter, value in zip(
                (axis.set_xlim, axis.set_ylim, axis.set_zlim), center
            ):
                setter(value - half_span, value + half_span)
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_axis_off()
            axis.set_title(f"{name}: {len(pieces)} pieces — view {column + 1}")
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "worst_object_decomposition_pieces.png", dpi=180)
    plt.close(figure)


def _factor_region_analysis(
    repository: Path,
    output: Path,
    payloads: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Measure approximation error specifically where one study factor changes shape."""
    rows = discover_study_objects(repository / "study_objects/sphere_study_v1/manifest.csv")
    by_id = {row["object_id"]: row for row in rows}
    worst = by_id[MANUAL_OBJECT_ID]
    scale = float(worst["scale_m_per_source_unit"])
    source = extract_exterior_surface(parse_gmsh_v2(Path(worst["source_path"])))
    source_vertices = np.asarray(source.vertices, dtype=np.float64) * scale
    source_faces = np.asarray(source.faces, dtype=np.int64)
    first_level = next(iter(payloads))
    first_arrays = np.load(output / "levels" / first_level / "geometry_arrays.npz")
    points = np.asarray(first_arrays["sampled_points"], dtype=np.float64)
    sampled_faces = np.asarray(first_arrays["sampled_faces"], dtype=np.int64)
    representation_gaps: dict[str, np.ndarray] = {
        "single_hull": hull_gaps(points, ConvexHull(source_vertices))
    }
    for level in SWEEP_LEVELS:
        if level not in payloads:
            continue
        arrays = np.load(output / "levels" / level / "geometry_arrays.npz")
        if not np.array_equal(points, arrays["sampled_points"]):
            raise ValueError("sweep levels do not share identical source samples")
        representation_gaps[level] = np.asarray(arrays["sampled_gaps"], dtype=np.float64)

    comparisons = {
        "macro": "obj_size-large_ar-high_macro-low_rough-high",
        "roughness": "obj_size-large_ar-high_macro-high_rough-low",
    }
    output_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "method": (
            "On identical worst-object source samples, first identify points farther "
            "than 0.1 mm from the matched factor-low source, then analyze the strongest "
            "quartile of that factor signal."
        ),
        "minimum_factor_signal_mm": 0.1,
        "active_region_quantile": 0.75,
        "comparisons": {},
    }
    feature_definitions: dict[str, Any] = {
        "object_id": MANUAL_OBJECT_ID,
        "coordinate_frame": "object-local metres, before the object's free-joint pose",
        "surface_normal_convention": "outward unit normal of the sampled source triangle",
        "sampling_seed": DEFAULT_SEED,
        "sampling_count": int(points.shape[0]),
        "features": {},
    }

    def record_feature(name: str, sample_index: int, **details: Any) -> None:
        face_index = int(sampled_faces[sample_index])
        triangle = source_vertices[source_faces[face_index]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        feature_definitions["features"][name] = {
            "sample_index": int(sample_index),
            "source_face_index": face_index,
            "point_m": points[sample_index].tolist(),
            "outward_normal": normal.tolist(),
            **details,
        }

    single_hull_gaps = representation_gaps["single_hull"]
    deepest_index = int(np.argmax(single_hull_gaps))
    record_feature(
        "deepest_concavity",
        deepest_index,
        selection="maximum single-hull gap among the shared 100,000 source samples",
        single_hull_gap_mm=float(single_hull_gaps[deepest_index] * 1000.0),
    )
    for factor, paired_id in comparisons.items():
        paired = by_id[paired_id]
        paired_surface = extract_exterior_surface(parse_gmsh_v2(Path(paired["source_path"])))
        paired_vertices = np.asarray(paired_surface.vertices, dtype=np.float64) * scale
        paired_faces = np.asarray(paired_surface.faces, dtype=np.int64)
        _, signal, _ = closest_union_surface(points, paired_vertices, paired_faces)
        eligible = signal > 0.0001
        if not np.any(eligible):
            raise ValueError(f"no {factor} factor-active samples found")
        region_cutoff = max(0.0001, float(np.quantile(signal[eligible], 0.75)))
        mask = signal >= region_cutoff
        signal_stats = gap_statistics(signal[mask], float(np.linalg.norm(np.ptp(source_vertices, axis=0))))
        active_indices = np.flatnonzero(mask)
        target_signal = float(np.quantile(signal[mask], 0.95))
        feature_index = int(active_indices[np.argmin(np.abs(signal[active_indices] - target_signal))])
        record_feature(
            f"{factor}_feature",
            feature_index,
            selection=(
                "sample nearest the p95 matched-factor surface distance within "
                "the strongest factor-signal quartile"
            ),
            matched_factor_low_object=paired_id,
            factor_signal_mm=float(signal[feature_index] * 1000.0),
            active_region_cutoff_mm=float(region_cutoff * 1000.0),
        )
        metadata["comparisons"][factor] = {
            "factor_high_object": MANUAL_OBJECT_ID,
            "matched_factor_low_object": paired_id,
            "active_sample_count": int(mask.sum()),
            "active_surface_fraction": float(mask.mean()),
            "active_region_cutoff_mm": region_cutoff * 1000.0,
            "factor_signal_mean_mm": signal_stats["mean_gap_mm"],
            "factor_signal_p95_mm": signal_stats["p95_gap_mm"],
            "factor_signal_p99_mm": signal_stats["p99_gap_mm"],
            "factor_signal_max_mm": signal_stats["max_gap_mm"],
        }
        for representation, gaps in representation_gaps.items():
            stats = gap_statistics(gaps[mask], float(np.linalg.norm(np.ptp(source_vertices, axis=0))))
            output_rows.append({
                "factor_region": factor,
                "representation": representation,
                "pieces": 1 if representation == "single_hull" else payloads[representation]["piece_count"],
                "active_sample_count": int(mask.sum()),
                "active_surface_fraction": float(mask.mean()),
                "active_region_cutoff_mm": region_cutoff * 1000.0,
                "factor_signal_p95_mm": signal_stats["p95_gap_mm"],
                "mean_error_mm": stats["mean_gap_mm"],
                "p95_error_mm": stats["p95_gap_mm"],
                "p99_error_mm": stats["p99_gap_mm"],
                "max_error_mm": stats["max_gap_mm"],
                "surface_fraction_error_gt_0p10mm": stats["surface_fraction_gt_0p10mm"],
                "p95_error_to_factor_signal_ratio": (
                    stats["p95_gap_mm"] / signal_stats["p95_gap_mm"]
                    if signal_stats["p95_gap_mm"] else 0.0
                ),
            })
    with (output / "feature_region_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    _write_json(output / "feature_region_comparison.json", {"metadata": metadata, "rows": output_rows})
    _write_json(output / "contact_feature_definitions.json", feature_definitions)
    return output_rows


def assemble(repository: Path, output: Path) -> list[dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for level in SWEEP_LEVELS:
        path = output / "levels" / level / "result.json"
        if path.is_file():
            payloads[level] = json.loads(path.read_text(encoding="utf-8"))
    if not payloads:
        raise ValueError("no completed decomposition levels to assemble")
    rows = [_single_hull_row(repository)] + [_candidate_row(payloads[level]) for level in SWEEP_LEVELS if level in payloads]
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    for filename in ("worst_object_parameter_sweep.csv", "geometry_comparison.csv"):
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "sweep_version": SWEEP_VERSION,
        "object_id": MANUAL_OBJECT_ID,
        "completed_levels": list(payloads),
        "rows": rows,
        "levels": payloads,
        "provisional_geometry_targets": {
            "p95_gap_mm_lt": 0.25,
            "p99_gap_mm_lt": 0.5,
            "max_gap_mm_lt": 1.0,
            "absolute_volume_error_percent_lt": 1.0,
        },
    }
    _write_json(output / "worst_object_parameter_sweep.json", summary)
    _plot_heatmaps(repository, output, payloads)
    _plot_decomposition_pieces(output, payloads)
    summary["factor_region_rows"] = _factor_region_analysis(repository, output, payloads)
    _write_json(output / "worst_object_parameter_sweep.json", summary)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("generated/convex_decomposition_validation"))
    parser.add_argument("--level", choices=tuple(SWEEP_LEVELS))
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = (repository / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.level:
        payload = run_level(repository, output, args.level, samples=args.samples)
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    if args.assemble:
        print(json.dumps(assemble(repository, output), indent=2, sort_keys=True, default=_json_default))
    if not args.level and not args.assemble:
        parser.error("provide --level and/or --assemble")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
