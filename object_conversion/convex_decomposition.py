#!/usr/bin/env python3
"""Deterministic, validated, cached CoACD decomposition of rigid surfaces.

Collision pieces are stored in metres.  This is intentional: CoACD's real-metric
threshold then has an unambiguous physical meaning and generated MuJoCo assets use
unit scale.  The original exterior OBJ remains in source units for visualization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
from scipy.spatial import ConvexHull


DECOMPOSITION_CONVERTER_VERSION = "coacd-real-metric-cache-v1"
DEFAULT_DECOMPOSITION_SEED = 20260812


@dataclass(frozen=True)
class CoACDParameters:
    """All CoACD arguments that can affect output geometry."""

    threshold_m: float
    max_convex_hull: int = -1
    preprocess_mode: str = "off"
    preprocess_resolution: int = 50
    resolution: int = 2000
    mcts_nodes: int = 20
    mcts_iterations: int = 150
    mcts_max_depth: int = 3
    pca: bool = False
    merge: bool = True
    decimate: bool = False
    max_ch_vertex: int = 256
    extrude: bool = False
    extrude_margin: float = 0.01
    apx_mode: str = "ch"
    seed: int = DEFAULT_DECOMPOSITION_SEED
    real_metric: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold_m) or self.threshold_m <= 0.0:
            raise ValueError("threshold_m must be positive and finite")
        if self.max_convex_hull == 0 or self.max_convex_hull < -1:
            raise ValueError("max_convex_hull must be -1 or positive")
        if self.preprocess_mode not in {"auto", "on", "off"}:
            raise ValueError("unsupported preprocess_mode")
        if self.apx_mode not in {"ch", "box"}:
            raise ValueError("unsupported apx_mode")

    def coacd_kwargs(self) -> dict[str, Any]:
        values = asdict(self)
        values["threshold"] = values.pop("threshold_m")
        return values


@dataclass(frozen=True)
class ConvexPiece:
    vertices_m: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class DecompositionResult:
    source_path: Path
    source_hash: str
    exterior_hash: str
    cache_key: str
    cache_dir: Path
    piece_paths: tuple[Path, ...]
    manifest_path: Path
    cache_reused: bool
    manifest: dict[str, Any]


def _dependency_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"{distribution} is required for convex-decomposition validation"
        ) from exc


def decomposition_dependency_versions() -> dict[str, str]:
    return {"coacd": _dependency_version("coacd")}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decomposition_cache_key(
    *,
    source_hash: str,
    exterior_hash: str,
    scale: Sequence[float],
    parameters: CoACDParameters,
    coacd_version: str,
) -> str:
    values = tuple(float(value) for value in scale)
    if len(values) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("scale must contain three positive finite values")
    payload = {
        "source_sha256": source_hash,
        "exterior_sha256": exterior_hash,
        "scale_m_per_source_unit": list(values),
        "algorithm": "CoACD",
        "algorithm_version": coacd_version,
        "converter_version": DECOMPOSITION_CONVERTER_VERSION,
        "parameters": asdict(parameters),
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(rendered)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _mesh_signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    return float(
        np.einsum(
            "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
        ).sum()
        / 6.0
    )


def validate_convex_piece(piece: ConvexPiece) -> dict[str, Any]:
    vertices = np.asarray(piece.vertices_m, dtype=np.float64)
    faces = np.asarray(piece.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise ValueError("piece vertices must have shape (N>=4, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
        raise ValueError("piece faces must have shape (M>=4, 3)")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("piece contains non-finite vertices")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("piece face index is out of bounds")
    edge_counts: dict[tuple[int, int], int] = {}
    edge_directions: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for left, right in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((int(left), int(right))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            edge_directions[edge] = edge_directions.get(edge, 0) + (
                1 if (int(left), int(right)) == edge else -1
            )
    watertight = bool(edge_counts) and all(count == 2 for count in edge_counts.values())
    winding_consistent = watertight and all(
        edge_directions[edge] == 0 for edge in edge_counts
    )
    hull = ConvexHull(vertices)
    signed_volume = _mesh_signed_volume(vertices, faces)
    mesh_volume = abs(signed_volume)
    volume_tolerance = max(1e-15, float(hull.volume) * 1e-8)
    volume_error = abs(mesh_volume - float(hull.volume))
    equations = np.asarray(hull.equations, dtype=np.float64)
    max_halfspace_violation = float(
        np.max(vertices @ equations[:, :3].T + equations[:, 3], initial=0.0)
    )
    convex = (
        watertight
        and winding_consistent
        and volume_error <= volume_tolerance
        and max_halfspace_violation <= 1e-10
    )
    if not convex:
        raise ValueError(
            "CoACD emitted a non-convex or invalid piece: "
            f"volume_error={volume_error}, halfspace_violation={max_halfspace_violation}"
        )
    return {
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "mesh_volume_m3": mesh_volume,
        "convex_hull_volume_m3": float(hull.volume),
        "mesh_to_hull_volume_abs_error_m3": volume_error,
        "max_halfspace_violation_m": max_halfspace_violation,
        "watertight": watertight,
        "winding_consistent": winding_consistent,
        "convex": True,
        "bbox_min_m": vertices.min(axis=0).tolist(),
        "bbox_max_m": vertices.max(axis=0).tolist(),
        "centroid_m": vertices.mean(axis=0).tolist(),
    }


def _canonical_piece_sort_key(piece: ConvexPiece) -> tuple[float, ...]:
    vertices = np.asarray(piece.vertices_m, dtype=np.float64)
    hull = ConvexHull(vertices)
    centroid = vertices.mean(axis=0)
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    return tuple(float(value) for value in (*centroid, -float(hull.volume), *low, *high))


def _render_obj(piece: ConvexPiece, *, piece_index: int, cache_key: str) -> bytes:
    vertices = np.asarray(piece.vertices_m, dtype=np.float64)
    faces = np.asarray(piece.faces, dtype=np.int64)
    lines = [
        f"# {DECOMPOSITION_CONVERTER_VERSION}",
        f"# cache_key {cache_key}",
        f"o convex_piece_{piece_index:03d}",
    ]
    lines.extend("v " + " ".join(format(float(value), ".17g") for value in vertex) for vertex in vertices)
    lines.extend("f " + " ".join(str(int(index) + 1) for index in face) for face in faces)
    return ("\n".join(lines) + "\n").encode("utf-8")


def load_convex_piece_obj(path: str | Path) -> ConvexPiece:
    path = Path(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(value.split("/")[0]) - 1 for value in line.split()[1:4]])
    return ConvexPiece(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))


def load_decomposition_pieces(result: DecompositionResult) -> tuple[ConvexPiece, ...]:
    return tuple(load_convex_piece_obj(path) for path in result.piece_paths)


def _valid_cached(
    manifest_path: Path,
    *,
    cache_key: str,
    source_hash: str,
    exterior_hash: str,
    coacd_version: str,
    parameters: CoACDParameters,
) -> tuple[dict[str, Any], tuple[Path, ...]] | None:
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("cache_key") != cache_key
        or manifest.get("source_hash") != source_hash
        or manifest.get("exterior_hash") != exterior_hash
        or manifest.get("converter_version") != DECOMPOSITION_CONVERTER_VERSION
        or manifest.get("algorithm") != "CoACD"
        or manifest.get("algorithm_version") != coacd_version
        or manifest.get("parameters") != asdict(parameters)
    ):
        return None
    piece_records = manifest.get("pieces")
    if not isinstance(piece_records, list) or len(piece_records) != manifest.get("piece_count"):
        return None
    paths: list[Path] = []
    for record in piece_records:
        path = manifest_path.parent / str(record.get("file", ""))
        if not path.is_file() or sha256_path(path) != record.get("sha256"):
            return None
        try:
            validate_convex_piece(load_convex_piece_obj(path))
        except ValueError:
            return None
        paths.append(path)
    return manifest, tuple(paths)


def decompose_surface_cached(
    *,
    source_path: str | Path,
    exterior_path: str | Path,
    vertices_source_units: np.ndarray,
    faces: np.ndarray,
    scale_m_per_source_unit: Sequence[float],
    cache_root: str | Path,
    parameters: CoACDParameters,
) -> DecompositionResult:
    """Run CoACD once for this exact geometry/configuration and cache its pieces."""
    try:
        import coacd
    except ImportError as exc:
        raise RuntimeError("CoACD is required; install coacd==1.0.11") from exc

    source = Path(source_path).expanduser().resolve()
    exterior = Path(exterior_path).expanduser().resolve()
    source_hash = sha256_path(source)
    exterior_hash = sha256_path(exterior)
    versions = decomposition_dependency_versions()
    scale = np.asarray(tuple(scale_m_per_source_unit), dtype=np.float64)
    if scale.shape != (3,) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("scale must contain three positive finite values")
    vertices_m = np.ascontiguousarray(
        np.asarray(vertices_source_units, dtype=np.float64) * scale[None, :]
    )
    faces_i32 = np.ascontiguousarray(np.asarray(faces, dtype=np.int32))
    cache_key = decomposition_cache_key(
        source_hash=source_hash,
        exterior_hash=exterior_hash,
        scale=scale,
        parameters=parameters,
        coacd_version=versions["coacd"],
    )
    cache_dir = Path(cache_root).expanduser().resolve() / source_hash[:16] / cache_key
    manifest_path = cache_dir / "manifest.json"
    cached = _valid_cached(
        manifest_path,
        cache_key=cache_key,
        source_hash=source_hash,
        exterior_hash=exterior_hash,
        coacd_version=versions["coacd"],
        parameters=parameters,
    )
    if cached is not None:
        manifest, piece_paths = cached
        return DecompositionResult(
            source, source_hash, exterior_hash, cache_key, cache_dir,
            piece_paths, manifest_path, True, manifest,
        )

    raw = coacd.run_coacd(coacd.Mesh(vertices_m, faces_i32), **parameters.coacd_kwargs())
    pieces = [
        ConvexPiece(
            np.ascontiguousarray(vertices, dtype=np.float64),
            np.ascontiguousarray(triangles, dtype=np.int64),
        )
        for vertices, triangles in raw
    ]
    if not pieces:
        raise RuntimeError("CoACD returned no convex pieces")
    pieces.sort(key=_canonical_piece_sort_key)
    records: list[dict[str, Any]] = []
    piece_paths: list[Path] = []
    for index, piece in enumerate(pieces):
        validation = validate_convex_piece(piece)
        filename = f"piece_{index:03d}.obj"
        path = cache_dir / filename
        data = _render_obj(piece, piece_index=index, cache_key=cache_key)
        _atomic_write(path, data)
        piece_paths.append(path)
        records.append({
            "index": index,
            "file": filename,
            "sha256": sha256_bytes(data),
            **validation,
        })
    manifest: dict[str, Any] = {
        "representation": "convex_decomposition",
        "source_mesh": str(source),
        "source_hash": source_hash,
        "exterior_mesh": str(exterior),
        "exterior_hash": exterior_hash,
        "scale_m_per_source_unit": scale.tolist(),
        "collision_piece_units": "metres",
        "algorithm": "CoACD",
        "algorithm_version": versions["coacd"],
        "converter_version": DECOMPOSITION_CONVERTER_VERSION,
        "parameters": asdict(parameters),
        "cache_key": cache_key,
        "piece_count": len(records),
        "pieces": records,
        "all_pieces_convex": all(record["convex"] for record in records),
    }
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return DecompositionResult(
        source, source_hash, exterior_hash, cache_key, cache_dir,
        tuple(piece_paths), manifest_path, False, manifest,
    )
