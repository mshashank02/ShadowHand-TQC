#!/usr/bin/env python3
"""Audit expanded-piece unions against a radius-dilated source target."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from object_conversion.audit_decomposition import audit_piece_union
from object_conversion.convex_decomposition import load_convex_piece_obj
from object_conversion.radius_aware_collision import (
    _oriented_hull_piece,
    sphere_polytope_vertices,
)


AUDIT_VERSION = "radius-aware-union-target-v1"


def _source_visual_mesh(model_xml: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    root = ET.parse(model_xml).getroot()
    compiler = root.find("compiler")
    mesh_dir = Path(compiler.get("meshdir", "."))
    if not mesh_dir.is_absolute():
        mesh_dir = model_xml.parent / mesh_dir
    asset = root.find("./asset/mesh[@name='custom_object_visual_mesh']")
    if asset is None:
        raise ValueError("model has no custom_object_visual_mesh")
    source_path = Path(asset.get("file", ""))
    if not source_path.is_absolute():
        source_path = mesh_dir / source_path
    source = load_convex_piece_obj(source_path)
    scale = np.fromstring(asset.get("scale", "1 1 1"), sep=" ")
    if scale.size != 3:
        raise ValueError("visual mesh scale must have three values")
    return source.vertices_m * scale, source.faces, source_path.resolve()


def _target_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    radius_m: float,
    sphere_subdivisions: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import manifold3d as m3
    except ImportError as exc:
        raise RuntimeError("Manifold3D is required for radius-aware geometry audit") from exc
    source = m3.Manifold(
        m3.Mesh64(
            np.ascontiguousarray(vertices, dtype=np.float64),
            np.ascontiguousarray(faces, dtype=np.uint64),
        )
    )
    if source.status() != m3.Error.NoError:
        raise ValueError(f"source is not a valid manifold: {source.status()}")
    directions, sphere_metadata = sphere_polytope_vertices(
        sphere_subdivisions, bound="circumscribed"
    )
    sphere = _oriented_hull_piece(directions * radius_m)
    ball = m3.Manifold(
        m3.Mesh64(
            np.ascontiguousarray(sphere.vertices_m, dtype=np.float64),
            np.ascontiguousarray(sphere.faces, dtype=np.uint64),
        )
    )
    target = source.minkowski_sum(ball)
    if target.status() != m3.Error.NoError:
        raise ValueError(f"target Minkowski sum failed: {target.status()}")
    mesh = target.to_mesh64()
    metadata = {
        "target_radius_m": radius_m,
        "target_sphere": sphere_metadata,
        "target_vertex_count": int(target.num_vert()),
        "target_triangle_count": int(target.num_tri()),
        "target_volume_m3": float(target.volume()),
    }
    return (
        np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3].copy(),
        np.asarray(mesh.tri_verts, dtype=np.int64).copy(),
        metadata,
    )


def run_audit(
    *,
    reference_model: Path,
    candidate_manifests: list[Path],
    output: Path,
    target_radius_m: float,
    samples: int,
    seed: int,
    target_sphere_subdivisions: int,
) -> dict[str, Any]:
    source_vertices, source_faces, source_path = _source_visual_mesh(reference_model)
    target_vertices, target_faces, target_metadata = _target_mesh(
        source_vertices, source_faces, target_radius_m, target_sphere_subdivisions
    )
    candidates: dict[str, Any] = {}
    for manifest_path in candidate_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pieces = [load_convex_piece_obj(record["output"]) for record in manifest["piece_records"]]
        metrics, _ = audit_piece_union(
            target_vertices, target_faces, pieces, samples=samples, seed=seed
        )
        name = Path(manifest["output_xml"]).stem
        candidates[name] = {
            "candidate_manifest": str(manifest_path.resolve()),
            "candidate_parameters": manifest["parameters"],
            "metrics": metrics,
        }
    payload = {
        "audit_version": AUDIT_VERSION,
        "reference_model": str(reference_model.resolve()),
        "source_visual_mesh": str(source_path),
        "source_vertex_count": int(len(source_vertices)),
        "source_triangle_count": int(len(source_faces)),
        "sample_count": samples,
        "sample_seed": seed,
        **target_metadata,
        "candidates": candidates,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "radius_aware_shell_geometry.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for name, candidate in candidates.items():
        metrics = candidate["metrics"]
        rows.append(
            {
                "candidate": name,
                "piece_count": metrics["piece_count"],
                "shell_mm": candidate["candidate_parameters"]["shell_m"] * 1000.0,
                "target_volume_m3": metrics["source_volume_m3"],
                "candidate_union_volume_m3": metrics["union_volume_m3"],
                "volume_error_percent": metrics["volume_error_percent"],
                "p95_boundary_error_mm": metrics["p95_gap_mm"],
                "p99_boundary_error_mm": metrics["p99_gap_mm"],
                "max_boundary_error_mm": metrics["max_gap_mm"],
                "overfill_surface_fraction": metrics["sampled_overfill_fraction"],
                "underfill_surface_fraction": metrics["sampled_underfill_fraction"],
                "surface_fraction_gt_0p10mm": metrics["surface_fraction_gt_0p10mm"],
                "surface_fraction_gt_0p25mm": metrics["surface_fraction_gt_0p25mm"],
                "surface_fraction_gt_0p50mm": metrics["surface_fraction_gt_0p50mm"],
            }
        )
    with (output / "radius_aware_shell_geometry.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-radius-mm", type=float, default=1.25)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--target-sphere-subdivisions", type=int, default=3)
    args = parser.parse_args()
    payload = run_audit(
        reference_model=args.reference_model.resolve(),
        candidate_manifests=[path.resolve() for path in args.candidate_manifest],
        output=args.output.resolve(),
        target_radius_m=args.target_radius_mm / 1000.0,
        samples=args.samples,
        seed=args.seed,
        target_sphere_subdivisions=args.target_sphere_subdivisions,
    )
    for name, candidate in payload["candidates"].items():
        metrics = candidate["metrics"]
        print(name, metrics["p95_gap_mm"], metrics["p99_gap_mm"], metrics["max_gap_mm"])


if __name__ == "__main__":
    main()
