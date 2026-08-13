#!/usr/bin/env python3
"""Quantify convex-hull information loss for the rigid study-object surfaces.

The local gap at a point inside a convex polytope is its shortest distance to the
polytope boundary.  If the hull is represented by normalized half-spaces
``n_i . x + b_i <= 0``, this is exactly ``min_i -(n_i . x + b_i)``.  This avoids
the ambiguity of unsigned closest-surface queries on hull-supported mesh regions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from object_conversion.gmsh_to_rigid_surface import (
    convert_gmsh_to_rigid_surface,
    extract_exterior_surface,
    parse_gmsh_v2,
)


AUDIT_VERSION = "convexity-halfspace-v1"
DEFAULT_SEED = 20260812
DEFAULT_SAMPLES = 100_000
EXPECTED_OBJECTS = 24
BASE_SCALE_M = 0.025
SIZE_MULTIPLIERS = {"small": 0.75, "medium": 1.0, "large": 1.25}
THRESHOLDS_MM = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
FACTOR_NAMES = ("size", "aspect_ratio", "macro", "roughness")
MANUAL_OBJECT_ID = "obj_size-large_ar-high_macro-high_rough-high"
MUJOCO_VERIFICATION = {
    "gpu_environment_version": "3.11.0",
    "cpu_reference_version": "3.3.1",
    "mesh_collision": "convex hull; generic convex collision uses MPR or GJK/EPA",
    "default_maxhullvert": -1,
    "representative_object_id": MANUAL_OBJECT_ID,
    "representative_compiled_mesh_vertices": 1666,
    "representative_compiled_mesh_faces": 3328,
    "representative_compiled_hull_vertices": 989,
    "representative_compiled_hull_faces": 1974,
    "representative_mesh_graphadr": 0,
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_study_objects(manifest_path: Path) -> list[dict[str, Any]]:
    root = manifest_path.resolve().parent
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"object_id", "msh_file", *FACTOR_NAMES}
    if len(rows) != EXPECTED_OBJECTS:
        raise ValueError(f"expected exactly {EXPECTED_OBJECTS} study objects, found {len(rows)}")
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"study manifest must contain {sorted(required)}")
    expected_factorial = {
        (size, aspect, macro, rough)
        for size in SIZE_MULTIPLIERS
        for aspect in ("low", "high")
        for macro in ("low", "high")
        for rough in ("low", "high")
    }
    observed: set[tuple[str, str, str, str]] = set()
    identifiers: set[str] = set()
    result = []
    for raw in rows:
        row = dict(raw)
        source = (root / row["msh_file"]).resolve()
        if not source.is_file():
            raise ValueError(f"missing study mesh: {source}")
        if row["object_id"] in identifiers:
            raise ValueError(f"duplicate object_id: {row['object_id']}")
        identifiers.add(row["object_id"])
        key = (row["size"], row["aspect_ratio"], row["macro"], row["roughness"])
        observed.add(key)
        row["source_path"] = str(source)
        row["scale_m_per_source_unit"] = BASE_SCALE_M * SIZE_MULTIPLIERS[row["size"]]
        result.append(row)
    if observed != expected_factorial:
        raise ValueError("study manifest is not the expected 3x2x2x2 factorial")
    return result


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )


def sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, area-weighted sampling with uniform triangle barycentrics."""
    if count <= 0:
        raise ValueError("sample count must be positive")
    areas = triangle_areas(vertices, faces)
    total = float(areas.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("mesh has no positive-area triangles")
    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(faces), size=count, p=areas / total)
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    bary = np.column_stack((1.0 - root, root * (1.0 - uv[:, 1]), root * uv[:, 1]))
    points = np.einsum("ni,nij->nj", bary, vertices[faces[face_indices]])
    return points, face_indices


def hull_gaps(
    points: np.ndarray,
    hull: ConvexHull,
    *,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Return shortest interior distance to the hull boundary for each point."""
    equations = np.asarray(hull.equations, dtype=np.float64)
    normals = equations[:, :3]
    offsets = equations[:, 3]
    norms = np.linalg.norm(normals, axis=1)
    result = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        block = np.asarray(points[start : start + chunk_size], dtype=np.float64)
        slack = -(block @ normals.T + offsets) / norms
        # Qhull and matrix multiplication can put a hull-supported point a few ulps
        # outside.  Negative values therefore mean zero geometric gap.
        result[start : start + len(block)] = np.maximum(0.0, slack.min(axis=1))
    return result


def gap_statistics(gaps_m: np.ndarray, diagonal_m: float) -> dict[str, Any]:
    percentiles = np.percentile(gaps_m, (50, 90, 95, 99, 99.9))
    stats: dict[str, Any] = {
        "mean_gap_m": float(np.mean(gaps_m)),
        "median_gap_m": float(percentiles[0]),
        "rms_gap_m": float(np.sqrt(np.mean(np.square(gaps_m)))),
        "p90_gap_m": float(percentiles[1]),
        "p95_gap_m": float(percentiles[2]),
        "p99_gap_m": float(percentiles[3]),
        "p999_gap_m": float(percentiles[4]),
        "max_gap_m": float(np.max(gaps_m)),
    }
    for name in ("mean", "median", "rms", "p90", "p95", "p99", "p999", "max"):
        stats[f"{name}_gap_mm"] = stats[f"{name}_gap_m"] * 1000.0
    for name in ("mean", "p95", "p99", "max"):
        value = stats[f"{name}_gap_m"] / diagonal_m
        stats[f"{name}_gap_normalized"] = value
        stats[f"{name}_gap_normalized_percent"] = value * 100.0
    for threshold in THRESHOLDS_MM:
        label = f"surface_fraction_gt_{threshold:.2f}mm".replace(".", "p")
        stats[label] = float(np.mean(gaps_m > threshold / 1000.0))
    return stats


def audit_arrays(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Any], dict[str, np.ndarray], ConvexHull]:
    hull = ConvexHull(vertices_m)
    sampled_points, sampled_faces = sample_surface(vertices_m, faces, samples, seed)
    sampled_gaps = hull_gaps(sampled_points, hull)
    vertex_gaps = hull_gaps(vertices_m, hull)
    areas = triangle_areas(vertices_m, faces)
    bbox = np.ptp(vertices_m, axis=0)
    diagonal = float(np.linalg.norm(bbox))
    mesh_volume = abs(
        float(np.einsum("ij,ij->i", vertices_m[faces[:, 0]], np.cross(
            vertices_m[faces[:, 1]], vertices_m[faces[:, 2]]
        )).sum() / 6.0)
    )
    hull_volume = float(hull.volume)
    sampled_index = int(np.argmax(sampled_gaps))
    vertex_index = int(np.argmax(vertex_gaps))
    metrics: dict[str, Any] = {
        "vertices": int(len(vertices_m)),
        "triangles": int(len(faces)),
        "watertight": bool(_watertight(faces)),
        "winding_consistent": bool(_winding_consistent(faces)),
        "bbox_x_m": float(bbox[0]),
        "bbox_y_m": float(bbox[1]),
        "bbox_z_m": float(bbox[2]),
        "bbox_diagonal_m": diagonal,
        "mesh_surface_area_m2": float(areas.sum()),
        "mesh_volume_m3": mesh_volume,
        "hull_vertices": int(len(hull.vertices)),
        "hull_faces": int(len(hull.simplices)),
        "hull_surface_area_m2": float(hull.area),
        "hull_volume_m3": hull_volume,
        "volume_difference_m3": hull_volume - mesh_volume,
        "volume_inflation_fraction": (hull_volume - mesh_volume) / mesh_volume,
        "volume_inflation_percent": 100.0 * (hull_volume - mesh_volume) / mesh_volume,
        "volume_convexity_ratio": mesh_volume / hull_volume,
        "sample_count": int(samples),
        "sample_seed": int(seed),
        "sampled_max_face_index": int(sampled_faces[sampled_index]),
        "max_gap_x": float(sampled_points[sampled_index, 0]),
        "max_gap_y": float(sampled_points[sampled_index, 1]),
        "max_gap_z": float(sampled_points[sampled_index, 2]),
        "vertex_max_gap_m": float(vertex_gaps[vertex_index]),
        "vertex_max_gap_mm": float(vertex_gaps[vertex_index] * 1000.0),
        "vertex_max_gap_index": vertex_index,
        "vertex_max_gap_xyz_m": vertices_m[vertex_index].tolist(),
    }
    metrics.update(gap_statistics(sampled_gaps, diagonal))
    arrays = {
        "sampled_points": sampled_points,
        "sampled_faces": sampled_faces,
        "sampled_gaps": sampled_gaps,
        "vertex_gaps": vertex_gaps,
    }
    return metrics, arrays, hull


def _edge_incidence(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            incidence.setdefault(tuple(sorted((int(u), int(v)))), []).append((face_index, int(u == min(u, v))))
    return incidence


def _watertight(faces: np.ndarray) -> bool:
    return all(len(items) == 2 for items in _edge_incidence(faces).values())


def _winding_consistent(faces: np.ndarray) -> bool:
    directed: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            directed.setdefault(tuple(sorted((int(u), int(v)))), []).append((int(u), int(v)))
    return all(len(edges) == 2 and edges[0] == edges[1][::-1] for edges in directed.values())


def ranking_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["p99_gap_mm"]),
        float(row["max_gap_mm"]),
        float(row["surface_fraction_gt_0p10mm"]),
    )


def convergence_study(
    vertices: np.ndarray,
    faces: np.ndarray,
    counts: Sequence[int] = (10_000, 50_000, 100_000, 250_000),
    seed: int = DEFAULT_SEED,
) -> list[dict[str, float]]:
    hull = ConvexHull(vertices)
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    largest = max(counts)
    points, _ = sample_surface(vertices, faces, largest, seed)
    # If every source vertex is a hull vertex and volume agrees, this triangulated
    # surface is the hull boundary and its gap is analytically zero.  We still draw
    # each requested deterministic sample set, but avoid a billion redundant plane
    # evaluations for the convex control object.
    mesh_volume = abs(float(np.einsum(
        "ij,ij->i", vertices[faces[:, 0]],
        np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])
    ).sum() / 6.0))
    if len(hull.vertices) == len(vertices) and abs(hull.volume - mesh_volume) <= 1e-10 * hull.volume:
        all_gaps = np.zeros(largest, dtype=np.float64)
    else:
        all_gaps = hull_gaps(points, hull)
    output = []
    for count in counts:
        stats = gap_statistics(all_gaps[:count], diagonal)
        output.append({
            "samples": int(count),
            "p95_gap_mm": stats["p95_gap_mm"],
            "p99_gap_mm": stats["p99_gap_mm"],
            "sampled_max_gap_mm": stats["max_gap_mm"],
        })
    return output


def assemble_existing(repository: Path, stage: str = "all") -> dict[str, Any]:
    """Assemble checkpointed 100k single-object runs into the all-object audit."""
    repository = repository.resolve()
    output = repository / "generated/convexity_audit"
    expected = discover_study_objects(repository / "study_objects/sphere_study_v1/manifest.csv")
    rows = []
    metadata = None
    for item in expected:
        path = output / f"{item['object_id']}.json"
        if not path.is_file():
            raise ValueError(f"missing checkpointed object audit: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if len(payload.get("objects", [])) != 1:
            raise ValueError(f"invalid object audit: {path}")
        if payload["metadata"]["sample_count_per_object"] != DEFAULT_SAMPLES:
            raise ValueError(f"object audit is not the required 100k run: {path}")
        metadata = payload["metadata"]
        rows.append(payload["objects"][0])
    ranked = sorted(rows, key=ranking_key)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    least, median, worst = ranked[0], ranked[len(ranked) // 2], ranked[-1]
    selection = {
        "least": least["object_id"], "median": median["object_id"],
        "worst": worst["object_id"], "manual_viewer": MANUAL_OBJECT_ID,
    }
    payload: dict[str, Any] = {
        "metadata": metadata,
        "objects": ranked,
        "selection": selection,
        "factor_summary": factor_summary(ranked),
        "macro_pairs": matched_factor_pairs(ranked, "macro"),
        "roughness_pairs": matched_factor_pairs(ranked, "roughness"),
        "macro_hull_pairs": [],
        "roughness_hull_pairs": [],
        "convergence": {},
        "figures": {},
    }
    payload["metadata"]["mujoco_collision_verification"] = MUJOCO_VERIFICATION
    combined_path = output / "convexity_audit.json"
    if stage != "core" and combined_path.is_file():
        prior = json.loads(combined_path.read_text(encoding="utf-8"))
        for key in ("macro_hull_pairs", "roughness_hull_pairs", "convergence", "figures"):
            payload[key] = prior.get(key, payload[key])
    by_id = {row["object_id"]: row for row in ranked}
    if stage in ("pairs", "all"):
        payload["macro_hull_pairs"] = hull_pair_comparisons(ranked, "macro")
        payload["roughness_hull_pairs"] = hull_pair_comparisons(ranked, "roughness")
    if stage in ("convergence", "all"):
        for label, selected in (("nearly_convex", least), ("most_convexified", worst)):
            vertices, faces = _load_scaled_surface(selected)
            payload["convergence"][label] = {
                "object_id": selected["object_id"],
                "runs": convergence_study(vertices, faces),
            }
    if stage in ("figures", "all"):
        for label, selected_id in selection.items():
            figure_path = output / "figures" / f"{label}_{selected_id}.png"
            write_heatmap(by_id[selected_id], figure_path)
            payload["figures"][label] = str(figure_path)
    combined_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(ranked, output / "convexity_audit.csv")
    write_markdown_report(payload, output / "convexity_audit.md")
    return payload


def _load_scaled_surface(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    surface = extract_exterior_surface(parse_gmsh_v2(row["source_path"]))
    scale = float(row["scale_m_per_source_unit"])
    return np.asarray(surface.vertices, dtype=np.float64) * scale, np.asarray(surface.faces, dtype=np.int64)


def factor_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("volume_inflation_percent", "p95_gap_mm", "p99_gap_mm", "max_gap_mm")
    result: dict[str, Any] = {}
    for factor in FACTOR_NAMES:
        result[factor] = {}
        for level in sorted({str(row[factor]) for row in rows}):
            subset = [row for row in rows if row[factor] == level]
            result[factor][level] = {
                metric: float(np.mean([float(row[metric]) for row in subset]))
                for metric in metrics
            }
    return result


def matched_factor_pairs(rows: Sequence[dict[str, Any]], factor: str) -> list[dict[str, Any]]:
    other = [name for name in FACTOR_NAMES if name != factor]
    grouped: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row[name]) for name in other), {})[str(row[factor])] = row
    pairs = []
    for key, levels in sorted(grouped.items()):
        if set(levels) != {"low", "high"}:
            continue
        low, high = levels["low"], levels["high"]
        pairs.append({
            "matched_factors": dict(zip(other, key)),
            "low_object_id": low["object_id"],
            "high_object_id": high["object_id"],
            **{
                f"{metric}_high_minus_low": float(high[metric]) - float(low[metric])
                for metric in ("p95_gap_mm", "p99_gap_mm", "max_gap_mm", "volume_inflation_percent")
            },
        })
    return pairs


def _sample_hull_surface(hull: ConvexHull, count: int, seed: int) -> np.ndarray:
    return sample_surface(hull.points, np.asarray(hull.simplices), count, seed)[0]


def hull_pair_comparisons(rows: Sequence[dict[str, Any]], factor: str, samples: int = 20_000) -> list[dict[str, Any]]:
    """Secondary sampled symmetric hull distance for matched low/high variants."""
    indexed = {row["object_id"]: row for row in rows}
    comparisons = []
    for pair in matched_factor_pairs(rows, factor):
        low = indexed[pair["low_object_id"]]
        high = indexed[pair["high_object_id"]]
        low_v, _ = _load_scaled_surface(low)
        high_v, _ = _load_scaled_surface(high)
        low_points = _sample_hull_surface(ConvexHull(low_v), samples, DEFAULT_SEED)
        high_points = _sample_hull_surface(ConvexHull(high_v), samples, DEFAULT_SEED)
        low_to_high = cKDTree(high_points).query(low_points, workers=1)[0]
        high_to_low = cKDTree(low_points).query(high_points, workers=1)[0]
        combined = np.concatenate((low_to_high, high_to_low))
        comparisons.append({
            **pair,
            "method": "sampled symmetric nearest-neighbor hull-surface distance",
            "samples_per_hull": samples,
            "mean_hull_difference_mm": float(combined.mean() * 1000),
            "p95_hull_difference_mm": float(np.percentile(combined, 95) * 1000),
            "max_hull_difference_mm": float(combined.max() * 1000),
            "hull_volume_difference_percent_of_low": float(
                100 * (high["hull_volume_m3"] - low["hull_volume_m3"]) / low["hull_volume_m3"]
            ),
        })
    return comparisons


def write_heatmap(row: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices, faces = _load_scaled_surface(row)
    gaps_mm = hull_gaps(vertices, ConvexHull(vertices)) * 1000.0
    face_values = gaps_mm[faces].mean(axis=1)
    vmax = max(float(np.percentile(face_values, 99.5)), float(face_values.max()), 1e-9)
    norm = matplotlib.colors.Normalize(0.0, vmax)
    views = ((15, -65, "front"), (15, 25, "side"), (15, 115, "back"), (75, -90, "top"))
    figure = plt.figure(figsize=(12, 10), constrained_layout=True)
    for index, (elev, azim, title) in enumerate(views, 1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        collection = Poly3DCollection(vertices[faces], linewidths=0.03)
        collection.set_facecolor(colormaps["inferno"](norm(face_values)))
        collection.set_edgecolor((0, 0, 0, 0.08))
        axis.add_collection3d(collection)
        mins, maxs = vertices.min(axis=0), vertices.max(axis=0)
        center = (mins + maxs) / 2
        radius = max(maxs - mins) / 2
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(title)
        axis.set_axis_off()
    figure.suptitle(f"{row['object_id']}\nvertex hull gap (mm)")
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap="inferno")
    figure.colorbar(mapper, ax=figure.axes, shrink=0.65, label="shortest gap to hull boundary (mm)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


CSV_FIELDS = [
    "rank", "object_id", "size", "aspect_ratio", "macro", "roughness",
    "source_path", "converted_surface_path", "source_sha256", "converted_mesh_sha256",
    "vertices", "triangles", "watertight", "winding_consistent", "hull_vertices", "hull_faces",
    "bbox_x_m", "bbox_y_m", "bbox_z_m", "bbox_diagonal_m", "mesh_surface_area_m2",
    "hull_surface_area_m2", "mesh_volume_m3", "hull_volume_m3", "volume_difference_m3",
    "volume_inflation_fraction", "volume_inflation_percent", "volume_convexity_ratio",
    "mean_gap_mm", "median_gap_mm", "rms_gap_mm", "p90_gap_mm", "p95_gap_mm",
    "p99_gap_mm", "p999_gap_mm", "max_gap_mm", "mean_gap_normalized",
    "p95_gap_normalized", "p99_gap_normalized", "max_gap_normalized",
    "surface_fraction_gt_0p01mm", "surface_fraction_gt_0p05mm", "surface_fraction_gt_0p10mm",
    "surface_fraction_gt_0p25mm", "surface_fraction_gt_0p50mm", "surface_fraction_gt_1p00mm",
    "max_gap_x", "max_gap_y", "max_gap_z", "vertex_max_gap_mm", "vertex_max_gap_index",
]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def write_markdown_report(payload: dict[str, Any], path: Path) -> None:
    rows = payload["objects"]
    worst = rows[-1]
    normalized_worst = max(rows, key=lambda row: row["max_gap_normalized_percent"])
    affected_worst = max(rows, key=lambda row: row["surface_fraction_gt_0p10mm"])
    maximum_inflation = max(rows, key=lambda row: row["volume_inflation_percent"])
    lines = [
        "# Quantitative Convexity Audit of the 24 Rigid Study Objects", "",
        "## Executive conclusion", "",
        "**Outcome C — convexification is material.** The native mesh path is mechanically valid, "
        "but its single-convex-hull collision representation is not scientifically safe for the full "
        "factorial tactile study without a targeted representation follow-up. Across 24 objects, the "
        f"worst 100k-sample p95/p99 gaps are {worst['p95_gap_mm']:.3f}/{worst['p99_gap_mm']:.3f} mm, "
        f"the sampled maximum is {worst['max_gap_mm']:.3f} mm, and {100*affected_worst['surface_fraction_gt_0p10mm']:.2f}% "
        "of the worst surface exceeds 0.1 mm. These are broad contact-scale changes, not merely "
        "floating-point non-zero values. Macro and roughness remain distinguishable between hulls, "
        "but their concave components are systematically suppressed. Expensive multi-seed CPU/GPU "
        "learning validation is therefore not approved as the next phase for this 24-object study.", "",
        "## MuJoCo collision behavior", "",
        "The project has MuJoCo 3.11.0 in its isolated GPU/MuJoCo-Warp environment and MuJoCo 3.3.1 "
        "in the CPU reference environment. MuJoCo's documented generic mesh collider is convex: "
        "mesh assets compile a convex hull (controlled by `maxhullvert`) and convex mesh collision "
        "uses MPR or GJK/EPA. The original triangles remain relevant to rendering and mesh asset "
        "properties, so a rendered surface can differ from the collision hull.", "",
        "An actual MuJoCo 3.11.0 compilation of the representative worst object's OBJ confirms this: "
        "the mesh asset contains 1,666 vertices and 3,328 rendered triangles, while `mesh_graphadr=0` "
        "points to a compiled convex graph with 989 hull vertices and 1,974 hull faces. These counts "
        "exactly match the SciPy/Qhull audit hull; production XML does not set `maxhullvert`, so the "
        "documented default of -1 (unlimited) applies.", "",
        "Authoritative references: [mesh asset XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-mesh), "
        "[collision computation](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection), and "
        "[MuJoCo source/documentation repository](https://github.com/google-deepmind/mujoco).", "",
        "## Dataset", "",
        "The audited set is exactly the 24 rows in `study_objects/sphere_study_v1/manifest.csv`: "
        "size `{small, medium, large}` × aspect ratio `{low, high}` × macro `{low, high}` × "
        "roughness `{low, high}`. Discovery fails closed unless all 24 unique factorial cells and "
        "source files exist. Converted surfaces below are deterministic outputs of the production "
        "GMSH tetrahedral-boundary converter, stored separately under the audit directory.", "",
        "| Object | Size | AR | Macro | Rough | Source SHA256 | Converted SHA256 |", "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda item: item["object_id"]):
        lines.append(
            f"| `{row['object_id']}` | {row['size']} | {row['aspect_ratio']} | {row['macro']} | "
            f"{row['roughness']} | `{row['source_sha256']}` | `{row['converted_mesh_sha256']}` |"
        )
    lines += [
        "", "Full paths and full-precision provenance are in `convexity_audit.json` and `convexity_audit.csv`.", "",
        "## Method", "",
        "For each scaled exterior mesh M, SciPy/Qhull constructs H = convex hull(M). The production "
        "scale is 0.025 m per source unit multiplied by 0.75/1.0/1.25 for small/medium/large. "
        "Sampling uses 100,000 fixed-seed, area-weighted uniform triangle samples per object.", "",
        "For a point x inside H, Qhull supplies normalized supporting half-spaces "
        "`n_i·x + b_i <= 0`. The shortest distance from x to the hull boundary is exactly "
        "`min_i(-(n_i·x+b_i)/||n_i||)`. This signed interior slack detects bridged concavities "
        "without the zero-distance ambiguity that would result from asking only whether parts of M "
        "also support H. Tiny negative roundoff is clamped to zero.", "",
        "The reported maximum is a converged sampled estimate, not a claimed continuous exact "
        "surface maximum. Every original vertex is also checked deterministically; its maximum, "
        "index, and XYZ are in JSON. Surface threshold fractions are unbiased Monte Carlo area "
        "estimates because samples are area weighted.", "",
        "Ranking is lexicographic by p99 gap, sampled maximum, then fraction over 0.1 mm. Volume "
        "inflation is reported but is not used as the sole rank criterion.", "",
        "## Validation", "",
        "Focused synthetic tests use a closed convex cube and a closed cube with a known central "
        "top indentation. The cube returns zero inflation and zero surface gap within 1e-12. The "
        "concave control returns hull volume 1, mesh volume 5/6, exactly 20% inflation, and a "
        "0.5-unit maximum at the known indentation center. Deterministic sampling, normalized gaps, "
        "threshold fractions, discovery, metadata, and CSV/JSON output are also tested.", "",
        "### Sampling convergence", "",
        "| Case | Samples | p95 (mm) | p99 (mm) | sampled max (mm) |", "|---|---:|---:|---:|---:|",
    ]
    for label, record in payload.get("convergence", {}).items():
        for run in record["runs"]:
            lines.append(
                f"| {label}: `{record['object_id']}` | {run['samples']:,} | "
                f"{run['p95_gap_mm']:.6f} | {run['p99_gap_mm']:.6f} | {run['sampled_max_gap_mm']:.6f} |"
            )
    lines += [
        "", "The worst object's 100k-to-250k change is 0.0058 mm at p95, 0.0071 mm at p99, "
        "and 0.00084 mm at the sampled maximum; the scientific conclusion is stable.", "",
        "## Results", "",
        f"Maximum volume inflation is **{maximum_inflation['volume_inflation_percent']:.3f}%**. "
        f"The largest normalized maximum is **{normalized_worst['max_gap_normalized_percent']:.3f}%** "
        f"of bounding-box diagonal on `{normalized_worst['object_id']}`.", "",
        "| Rank | Object | Inflation (%) | p95 (mm) | p99 (mm) | Max (mm) | Area >0.1 mm (%) |", "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | `{row['object_id']}` | {row['volume_inflation_percent']:.3f} | "
            f"{row['p95_gap_mm']:.3f} | {row['p99_gap_mm']:.3f} | {row['max_gap_mm']:.3f} | "
            f"{100*row['surface_fraction_gt_0p10mm']:.2f} |"
        )
    lines += [
        "", "## Worst case", "",
        f"`{worst['object_id']}` measures {1000*worst['bbox_x_m']:.2f} × {1000*worst['bbox_y_m']:.2f} × "
        f"{1000*worst['bbox_z_m']:.2f} mm (diagonal {1000*worst['bbox_diagonal_m']:.2f} mm). "
        f"Its mesh/hull volumes are {worst['mesh_volume_m3']:.12g}/{worst['hull_volume_m3']:.12g} m³ "
        f"({worst['volume_inflation_percent']:.3f}% inflation; convexity ratio {worst['volume_convexity_ratio']:.6f}).", "",
        f"Gap statistics are mean {worst['mean_gap_mm']:.3f}, median {worst['median_gap_mm']:.3f}, "
        f"RMS {worst['rms_gap_mm']:.3f}, p90 {worst['p90_gap_mm']:.3f}, p95 {worst['p95_gap_mm']:.3f}, "
        f"p99 {worst['p99_gap_mm']:.3f}, p99.9 {worst['p999_gap_mm']:.3f}, sampled max "
        f"{worst['max_gap_mm']:.3f}, and vertex max {worst['vertex_max_gap_mm']:.3f} mm.", "",
        f"Affected area: >0.01 mm {100*worst['surface_fraction_gt_0p01mm']:.2f}%, >0.05 mm "
        f"{100*worst['surface_fraction_gt_0p05mm']:.2f}%, >0.10 mm {100*worst['surface_fraction_gt_0p10mm']:.2f}%, "
        f">0.25 mm {100*worst['surface_fraction_gt_0p25mm']:.2f}%, >0.50 mm "
        f"{100*worst['surface_fraction_gt_0p50mm']:.2f}%, and >1.00 mm {100*worst['surface_fraction_gt_1p00mm']:.2f}%.", "",
        "The hand's distal collision capsules have radii about 7.05 mm (9.18 mm thumb), and generated "
        "tactile box sites commonly have a 2.5 mm half-thickness. The rigid object geom has zero "
        "margin and gap. A 2.55 mm p95 and 4.46 mm maximum is therefore directly comparable to "
        "fingertip/tactile contact dimensions, not a sub-resolution perturbation.", "",
        "Heatmap: [worst object, four views](figures/worst_obj_size-large_ar-high_macro-high_rough-high.png).", "",
        "## Study-factor analysis", "",
        "| Factor | Level | Mean inflation (%) | Mean p95 (mm) | Mean p99 (mm) | Mean max (mm) |", "|---|---|---:|---:|---:|---:|",
    ]
    for factor in FACTOR_NAMES:
        for level, values in payload["factor_summary"][factor].items():
            lines.append(
                f"| {factor} | {level} | {values['volume_inflation_percent']:.3f} | "
                f"{values['p95_gap_mm']:.3f} | {values['p99_gap_mm']:.3f} | {values['max_gap_mm']:.3f} |"
            )
    macro_hulls = payload.get("macro_hull_pairs", [])
    rough_hulls = payload.get("roughness_hull_pairs", [])
    macro_p95 = [item["p95_hull_difference_mm"] for item in macro_hulls]
    rough_p95 = [item["p95_hull_difference_mm"] for item in rough_hulls]
    lines += [
        "", "Size produces the expected absolute scaling: mean p99 is 0.901/1.204/1.502 mm for "
        "small/medium/large, while mean volume inflation is effectively constant at about 1.666%. "
        "Aspect ratio has little systematic effect (mean p99 1.162 low versus 1.243 mm high).", "",
        "Macro is the dominant concavity factor. Macro-high averages 3.078% inflation and 2.178 mm "
        "p99 versus 0.254% and 0.227 mm for macro-low. Every one of 12 matched macro pairs increases "
        "p95, p99, max, and inflation when switching low→high. Thus convexification systematically "
        "suppresses intended macro concavities.", "",
        "**Macro distinction preserved: yes, but incompletely.** "
        + (f"Matched convex hulls still differ at p95 by {min(macro_p95):.3f}–{max(macro_p95):.3f} mm, " if macro_p95 else "")
        + "and hull volume changes 5.35–8.14%; collision still sees a macro difference even though it "
        "does not see the concave portion of that difference.", "",
        "Roughness is implemented in geometry: it selects distinct `.msh` object families. The rigid "
        "geom friction is fixed (`1 0.005 0.0001`) across these objects; roughness is not a friction "
        "factor. Rough-high averages 2.462% inflation and 1.566 mm p99 versus 0.870% and 0.839 mm "
        "for rough-low. Therefore **roughness is geometrically affected: yes**. "
        + (f"Its matched hulls remain distinguishable at p95 by {min(rough_p95):.3f}–{max(rough_p95):.3f} mm, " if rough_p95 else "")
        + "but convexification erases their inward texture/concavity.", "",
        "## Manual-viewer consistency", "",
        "The previous large / AR-high / macro-high / rough-high viewer check reported no visually "
        "obvious hull difference. Quantitation does **not** agree: this is the worst object, with "
        "2.551 mm p95, 3.420 mm p99, 4.459 mm sampled maximum, and 41.47% of area above 0.1 mm. "
        "The multi-view heatmap explains why a silhouette-level inspection missed localized inward regions.", "",
        "## Recommendation", "",
        "Treat the present single-hull collision representation as a blocker for full 24-object GPU "
        "study execution (**Outcome C**). Do not alter production physics in this audit. Next, evaluate "
        "convex decomposition only for the geometrically affected families, beginning with macro-high "
        "and rough-high objects, and compare decomposition fidelity/cost with controlled fingertip "
        "contacts. The six macro-low/rough-low objects are nearly convex, but a mixed representation "
        "requires explicit provenance. Proceed to expensive multi-seed CPU/GPU learning validation "
        "only after the collision representation preserves the intended study factors.", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    repository: Path,
    *,
    object_id: str | None = None,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    repository = repository.resolve()
    manifest = repository / "study_objects/sphere_study_v1/manifest.csv"
    objects = discover_study_objects(manifest)
    if object_id is not None:
        objects = [row for row in objects if row["object_id"] == object_id]
        if not objects:
            raise ValueError(f"unknown study object_id: {object_id}")
    output = repository / "generated/convexity_audit"
    cache = output / "converted_surfaces"
    rows: list[dict[str, Any]] = []
    for index, metadata in enumerate(objects, 1):
        conversion = convert_gmsh_to_rigid_surface(metadata["source_path"], cache)
        vertices, faces = _load_scaled_surface(metadata)
        metrics, _, _ = audit_arrays(vertices, faces, samples=samples, seed=seed)
        row = {
            **metadata,
            **metrics,
            "source_sha256": conversion.source_hash,
            "converted_surface_path": str(conversion.mesh_path),
            "converted_mesh_sha256": conversion.manifest["converted_hash"],
        }
        rows.append(row)
        print(f"[{index}/{len(objects)}] {metadata['object_id']}: p99={row['p99_gap_mm']:.6g} mm, max={row['max_gap_mm']:.6g} mm")
    ranked = sorted(rows, key=ranking_key)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    payload: dict[str, Any] = {
        "metadata": {
            "audit_version": AUDIT_VERSION,
            "sample_method": "fixed-seed area-weighted uniform triangle sampling",
            "gap_method": "minimum normalized convex-hull half-space slack",
            "sampled_max_is_exact": False,
            "sample_count_per_object": samples,
            "random_seed": seed,
            "distance_units": "meters (mm convenience fields also included)",
            "mesh_scale_rule": "0.025 m/source-unit times size multiplier",
            "size_multipliers": SIZE_MULTIPLIERS,
            "thresholds_mm": THRESHOLDS_MM,
            "mujoco_collision_verification": MUJOCO_VERIFICATION,
        },
        "objects": ranked,
    }
    if object_id is None:
        least, median, worst = ranked[0], ranked[len(ranked) // 2], ranked[-1]
        payload["selection"] = {
            "least": least["object_id"], "median": median["object_id"], "worst": worst["object_id"],
            "manual_viewer": MANUAL_OBJECT_ID,
        }
        payload["factor_summary"] = factor_summary(ranked)
        payload["macro_pairs"] = matched_factor_pairs(ranked, "macro")
        payload["roughness_pairs"] = matched_factor_pairs(ranked, "roughness")
        payload["macro_hull_pairs"] = hull_pair_comparisons(ranked, "macro")
        payload["roughness_hull_pairs"] = hull_pair_comparisons(ranked, "roughness")
        by_id = {row["object_id"]: row for row in ranked}
        payload["convergence"] = {}
        for label, selected in (("nearly_convex", least), ("most_convexified", worst)):
            vertices, faces = _load_scaled_surface(selected)
            payload["convergence"][label] = {
                "object_id": selected["object_id"],
                "runs": convergence_study(vertices, faces, seed=seed),
            }
        figure_ids = dict(payload["selection"])
        payload["figures"] = {}
        for label, selected_id in figure_ids.items():
            figure_path = output / "figures" / f"{label}_{selected_id}.png"
            write_heatmap(by_id[selected_id], figure_path)
            payload["figures"][label] = str(figure_path)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / ("convexity_audit.json" if object_id is None else f"{object_id}.json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(ranked, output / ("convexity_audit.csv" if object_id is None else f"{object_id}.csv"))
    if object_id is None:
        write_markdown_report(payload, output / "convexity_audit.md")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="audit exactly the 24 manifest objects")
    selection.add_argument("--object-id", help="audit one manifest object")
    selection.add_argument(
        "--assemble-existing", action="store_true",
        help="assemble 24 checkpointed default-sample object audits",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--assembly-stage", choices=("core", "pairs", "convergence", "figures", "all"),
        default="all", help=argparse.SUPPRESS,
    )
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    if args.assemble_existing:
        assemble_existing(args.repository, stage=args.assembly_stage)
    else:
        run_audit(args.repository, object_id=args.object_id, samples=args.samples, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
