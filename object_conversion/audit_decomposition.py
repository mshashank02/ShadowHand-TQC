#!/usr/bin/env python3
"""Measure an exact boolean union of convex pieces against a source surface."""

from __future__ import annotations

from importlib import metadata
import math
from typing import Any, Sequence

import numpy as np
from scipy.spatial import ConvexHull

from .audit_convexity import DEFAULT_SEED, gap_statistics, sample_surface
from .convex_decomposition import ConvexPiece


UNION_AUDIT_VERSION = "manifold-union-trimesh-proximity-v1"


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"{distribution} is required for union geometry auditing") from exc


def union_dependency_versions() -> dict[str, str]:
    return {
        "manifold3d": _version("manifold3d"),
        "trimesh": _version("trimesh"),
        "rtree": _version("rtree"),
    }


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    return abs(float(np.einsum(
        "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
    ).sum() / 6.0))


def boolean_union_mesh(pieces: Sequence[ConvexPiece]) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the exposed boundary of UNION(pieces), with overlap faces removed."""
    if not pieces:
        raise ValueError("at least one convex piece is required")
    try:
        import manifold3d as m3
    except ImportError as exc:
        raise RuntimeError("Manifold3D is required; install manifold3d==3.5.2") from exc
    manifolds = []
    for index, piece in enumerate(pieces):
        mesh = m3.Mesh64(
            np.ascontiguousarray(piece.vertices_m, dtype=np.float64),
            np.ascontiguousarray(piece.faces, dtype=np.uint64),
        )
        manifold = m3.Manifold(mesh)
        if manifold.status() != m3.Error.NoError:
            raise ValueError(f"piece {index} is not a valid manifold: {manifold.status()}")
        manifolds.append(manifold)
    union = m3.Manifold.batch_boolean(manifolds, m3.OpType.Add)
    if union.status() != m3.Error.NoError:
        raise ValueError(f"boolean union failed: {union.status()}")
    output = union.to_mesh64()
    vertices = np.asarray(output.vert_properties, dtype=np.float64)[:, :3].copy()
    faces = np.asarray(output.tri_verts, dtype=np.int64).copy()
    if not len(vertices) or not len(faces):
        raise ValueError("boolean union is empty")
    volume = float(union.volume())
    independent_volume = mesh_volume(vertices, faces)
    if abs(independent_volume - volume) > max(1e-15, volume * 1e-9):
        raise ValueError("Manifold union volume does not match its emitted boundary")
    return vertices, faces, volume


def closest_union_surface(
    points: np.ndarray,
    union_vertices: np.ndarray,
    union_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact triangle closest points/distances using an R-tree query."""
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Trimesh and Rtree are required for proximity queries") from exc
    mesh = trimesh.Trimesh(
        vertices=np.asarray(union_vertices, dtype=np.float64),
        faces=np.asarray(union_faces, dtype=np.int64),
        process=False,
        validate=False,
    )
    closest, distance, triangle_id = trimesh.proximity.closest_point(
        mesh, np.asarray(points, dtype=np.float64)
    )
    return (
        np.asarray(closest, dtype=np.float64),
        np.asarray(distance, dtype=np.float64),
        np.asarray(triangle_id, dtype=np.int64),
    )


def points_inside_piece_union(
    points: np.ndarray,
    pieces: Sequence[ConvexPiece],
    *,
    tolerance_m: float = 2e-10,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Classify union membership exactly from each convex piece's half-spaces."""
    points = np.asarray(points, dtype=np.float64)
    result = np.zeros(len(points), dtype=bool)
    hulls = [ConvexHull(np.asarray(piece.vertices_m, dtype=np.float64)) for piece in pieces]
    for hull in hulls:
        equations = np.asarray(hull.equations, dtype=np.float64)
        undecided = np.flatnonzero(~result)
        for start in range(0, len(undecided), chunk_size):
            indices = undecided[start : start + chunk_size]
            values = points[indices] @ equations[:, :3].T + equations[:, 3]
            result[indices] |= np.all(values <= tolerance_m, axis=1)
    return result


def audit_piece_union(
    source_vertices_m: np.ndarray,
    source_faces: np.ndarray,
    pieces: Sequence[ConvexPiece],
    *,
    samples: int = 100_000,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Measure unsigned boundary error and explicitly separate over/under-fill."""
    source_vertices_m = np.asarray(source_vertices_m, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    union_vertices, union_faces, union_volume = boolean_union_mesh(pieces)
    points, sampled_faces = sample_surface(source_vertices_m, source_faces, samples, seed)
    closest, gaps, closest_triangles = closest_union_surface(points, union_vertices, union_faces)
    vertex_closest, vertex_gaps, vertex_closest_triangles = closest_union_surface(
        source_vertices_m, union_vertices, union_faces
    )
    inside = points_inside_piece_union(points, pieces)
    vertex_inside = points_inside_piece_union(source_vertices_m, pieces)
    diagonal = float(np.linalg.norm(np.ptp(source_vertices_m, axis=0)))
    source_volume = mesh_volume(source_vertices_m, source_faces)
    volume_difference = union_volume - source_volume
    metrics: dict[str, Any] = {
        "audit_version": UNION_AUDIT_VERSION,
        "dependencies": union_dependency_versions(),
        "piece_count": len(pieces),
        "sample_count": int(samples),
        "sample_seed": int(seed),
        "source_vertex_count": int(len(source_vertices_m)),
        "source_triangle_count": int(len(source_faces)),
        "union_vertex_count": int(len(union_vertices)),
        "union_triangle_count": int(len(union_faces)),
        "bbox_diagonal_m": diagonal,
        "source_volume_m3": source_volume,
        "union_volume_m3": union_volume,
        "volume_difference_m3": volume_difference,
        "volume_error_fraction": volume_difference / source_volume,
        "volume_error_percent": 100.0 * volume_difference / source_volume,
        "volume_absolute_error_fraction": abs(volume_difference) / source_volume,
        "volume_absolute_error_percent": 100.0 * abs(volume_difference) / source_volume,
        "sampled_source_inside_union_fraction": float(np.mean(inside)),
        "sampled_overfill_fraction": float(np.mean(inside & (gaps > 1e-10))),
        "sampled_underfill_fraction": float(np.mean(~inside)),
        "sampled_overfill_mean_gap_mm": float(np.mean(gaps[inside]) * 1000.0) if np.any(inside) else 0.0,
        "sampled_underfill_mean_gap_mm": float(np.mean(gaps[~inside]) * 1000.0) if np.any(~inside) else 0.0,
        "vertex_max_gap_m": float(np.max(vertex_gaps, initial=0.0)),
        "vertex_max_gap_mm": float(np.max(vertex_gaps, initial=0.0) * 1000.0),
        "vertex_inside_union_fraction": float(np.mean(vertex_inside)),
    }
    metrics.update(gap_statistics(gaps, diagonal))
    maximum = int(np.argmax(gaps))
    metrics.update({
        "max_gap_x": float(points[maximum, 0]),
        "max_gap_y": float(points[maximum, 1]),
        "max_gap_z": float(points[maximum, 2]),
        "max_gap_is_overfill": bool(inside[maximum]),
    })
    arrays = {
        "sampled_points": points,
        "sampled_faces": sampled_faces,
        "sampled_gaps": gaps,
        "sampled_inside_union": inside,
        "sampled_closest_points": closest,
        "sampled_closest_triangles": closest_triangles,
        "vertex_gaps": vertex_gaps,
        "vertex_inside_union": vertex_inside,
        "vertex_closest_points": vertex_closest,
        "vertex_closest_triangles": vertex_closest_triangles,
        "union_vertices": union_vertices,
        "union_faces": union_faces,
    }
    return metrics, arrays

