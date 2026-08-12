#!/usr/bin/env python3
"""Extract a deterministic, validated rigid surface from a tetrahedral GMSH mesh.

The study objects currently use GMSH 2.2 volume meshes.  This module deliberately
supports both the ASCII and binary variants of that format without depending on a
candidate or tactile layout.  Internal tetrahedral faces are removed by incidence
counting and each retained face is wound away from its tetrahedron interior.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Iterable, Sequence


CONVERTER_VERSION = "gmsh-tet-boundary-v1"
CONVERSION_PARAMETERS = {
    "compact_boundary_vertices": True,
    "output_format": "wavefront_obj",
    "preserve_coordinates": True,
}

# GMSH 2.x element-node counts.  Tetrahedron variants use their first four corner
# nodes for the collision surface; higher-order nodes do not change that topology.
_ELEMENT_NODE_COUNTS = {
    1: 2,
    2: 3,
    3: 4,
    4: 4,
    5: 8,
    6: 6,
    7: 5,
    8: 3,
    9: 6,
    10: 9,
    11: 10,
    12: 27,
    13: 18,
    14: 14,
    15: 1,
    16: 8,
    17: 20,
    18: 15,
    19: 13,
    20: 9,
    21: 10,
    22: 12,
    23: 15,
    24: 15,
    25: 21,
    26: 4,
    27: 5,
    28: 6,
    29: 20,
    30: 35,
    31: 56,
}
_TETRAHEDRON_TYPES = {4, 11, 29, 30, 31}


class GmshFormatError(ValueError):
    """Raised when a source is not a supported tetrahedral GMSH 2.x mesh."""


Vec3 = tuple[float, float, float]
Tet = tuple[int, int, int, int]
Face = tuple[int, int, int]


@dataclass(frozen=True)
class GmshVolumeMesh:
    path: str
    format_version: str
    binary: bool
    endian: str
    data_size: int
    node_order: tuple[int, ...]
    nodes: dict[int, Vec3]
    tetrahedra: tuple[Tet, ...]
    element_count: int
    element_type_counts: dict[int, int]


@dataclass(frozen=True)
class SurfaceMesh:
    vertices: tuple[Vec3, ...]
    source_node_ids: tuple[int, ...]
    faces: tuple[Face, ...]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ConversionResult:
    source_path: Path
    source_hash: str
    cache_key: str
    mesh_path: Path
    conversion_manifest_path: Path
    cache_reused: bool
    manifest: dict[str, Any]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _section_count_and_offset(raw: bytes, name: str) -> tuple[int, int]:
    marker = f"${name}".encode("ascii")
    start = raw.find(marker)
    if start < 0:
        raise GmshFormatError(f"missing ${name} section")
    line_end = raw.find(b"\n", start + len(marker))
    if line_end < 0:
        raise GmshFormatError(f"unterminated ${name} marker")
    count_end = raw.find(b"\n", line_end + 1)
    if count_end < 0:
        raise GmshFormatError(f"missing ${name} count")
    try:
        count = int(raw[line_end + 1 : count_end].strip())
    except ValueError as exc:
        raise GmshFormatError(f"invalid ${name} count") from exc
    return count, count_end + 1


def _parse_format(raw: bytes) -> tuple[str, bool, int, str]:
    marker = b"$MeshFormat"
    start = raw.find(marker)
    if start < 0:
        raise GmshFormatError("missing $MeshFormat section")
    line_start = raw.find(b"\n", start + len(marker)) + 1
    line_end = raw.find(b"\n", line_start)
    if line_start <= 0 or line_end < 0:
        raise GmshFormatError("malformed $MeshFormat section")
    parts = raw[line_start:line_end].decode("ascii", errors="strict").split()
    if len(parts) != 3:
        raise GmshFormatError(f"invalid MeshFormat line: {parts!r}")
    version, file_type_text, data_size_text = parts
    if not version.startswith("2."):
        raise GmshFormatError(f"only GMSH 2.x is supported, found {version}")
    binary = file_type_text == "1"
    if file_type_text not in {"0", "1"}:
        raise GmshFormatError(f"invalid GMSH file type {file_type_text!r}")
    data_size = int(data_size_text)
    if data_size not in {4, 8}:
        raise GmshFormatError(f"unsupported floating-point size {data_size}")
    endian = "<"
    if binary:
        probe = raw[line_end + 1 : line_end + 5]
        if len(probe) != 4:
            raise GmshFormatError("missing binary endian probe")
        if struct.unpack("<i", probe)[0] == 1:
            endian = "<"
        elif struct.unpack(">i", probe)[0] == 1:
            endian = ">"
        else:
            raise GmshFormatError("invalid binary endian probe")
    return version, binary, data_size, endian


def _parse_ascii_nodes(raw: bytes, count: int, offset: int) -> tuple[tuple[int, ...], dict[int, Vec3]]:
    end = raw.find(b"$EndNodes", offset)
    if end < 0:
        raise GmshFormatError("missing $EndNodes marker")
    lines = raw[offset:end].decode("utf-8", errors="strict").splitlines()
    if len(lines) < count:
        raise GmshFormatError(f"expected {count} nodes, found {len(lines)}")
    order: list[int] = []
    nodes: dict[int, Vec3] = {}
    for line in lines[:count]:
        parts = line.split()
        if len(parts) != 4:
            raise GmshFormatError(f"invalid node record {line!r}")
        node_id = int(parts[0])
        if node_id in nodes:
            raise GmshFormatError(f"duplicate node id {node_id}")
        order.append(node_id)
        nodes[node_id] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return tuple(order), nodes


def _parse_binary_nodes(
    raw: bytes, count: int, offset: int, endian: str, data_size: int
) -> tuple[tuple[int, ...], dict[int, Vec3]]:
    scalar = "d" if data_size == 8 else "f"
    record = struct.Struct(endian + "i" + scalar * 3)
    required = offset + count * record.size
    if required > len(raw):
        raise GmshFormatError("binary node section is truncated")
    order: list[int] = []
    nodes: dict[int, Vec3] = {}
    for index in range(count):
        node_id, x, y, z = record.unpack_from(raw, offset + index * record.size)
        if node_id in nodes:
            raise GmshFormatError(f"duplicate node id {node_id}")
        order.append(int(node_id))
        nodes[int(node_id)] = (float(x), float(y), float(z))
    return tuple(order), nodes


def _append_element(
    element_type: int,
    node_ids: Sequence[int],
    tetrahedra: list[Tet],
) -> None:
    if element_type not in _TETRAHEDRON_TYPES:
        return
    if len(node_ids) < 4:
        raise GmshFormatError(f"tetrahedron type {element_type} has fewer than four nodes")
    corners = tuple(int(value) for value in node_ids[:4])
    if len(set(corners)) != 4:
        raise GmshFormatError(f"degenerate tetrahedron references nodes {corners}")
    tetrahedra.append(corners)  # type: ignore[arg-type]


def _parse_ascii_elements(
    raw: bytes, count: int, offset: int
) -> tuple[tuple[Tet, ...], dict[int, int]]:
    end = raw.find(b"$EndElements", offset)
    if end < 0:
        raise GmshFormatError("missing $EndElements marker")
    lines = raw[offset:end].decode("utf-8", errors="strict").splitlines()
    if len(lines) < count:
        raise GmshFormatError(f"expected {count} elements, found {len(lines)}")
    tetrahedra: list[Tet] = []
    type_counts: Counter[int] = Counter()
    for line in lines[:count]:
        parts = line.split()
        if len(parts) < 3:
            raise GmshFormatError(f"invalid element record {line!r}")
        element_type = int(parts[1])
        tag_count = int(parts[2])
        node_count = _ELEMENT_NODE_COUNTS.get(element_type)
        if node_count is None:
            raise GmshFormatError(f"unsupported GMSH element type {element_type}")
        start = 3 + tag_count
        node_ids = [int(value) for value in parts[start : start + node_count]]
        if len(node_ids) != node_count:
            raise GmshFormatError(f"truncated element type {element_type} record")
        type_counts[element_type] += 1
        _append_element(element_type, node_ids, tetrahedra)
    return tuple(tetrahedra), dict(sorted(type_counts.items()))


def _parse_binary_elements(
    raw: bytes, count: int, offset: int, endian: str
) -> tuple[tuple[Tet, ...], dict[int, int]]:
    tetrahedra: list[Tet] = []
    type_counts: Counter[int] = Counter()
    header = struct.Struct(endian + "iii")
    parsed = 0
    cursor = offset
    while parsed < count:
        if cursor + header.size > len(raw):
            raise GmshFormatError("binary element block header is truncated")
        element_type, block_count, tag_count = header.unpack_from(raw, cursor)
        cursor += header.size
        node_count = _ELEMENT_NODE_COUNTS.get(int(element_type))
        if node_count is None:
            raise GmshFormatError(f"unsupported GMSH element type {element_type}")
        if block_count < 0 or parsed + block_count > count:
            raise GmshFormatError("binary element block count exceeds section count")
        record = struct.Struct(endian + "i" * (1 + tag_count + node_count))
        for _ in range(block_count):
            if cursor + record.size > len(raw):
                raise GmshFormatError("binary element record is truncated")
            values = record.unpack_from(raw, cursor)
            cursor += record.size
            node_ids = values[1 + tag_count :]
            _append_element(int(element_type), node_ids, tetrahedra)
        type_counts[int(element_type)] += int(block_count)
        parsed += int(block_count)
    return tuple(tetrahedra), dict(sorted(type_counts.items()))


def parse_gmsh_v2(path: str | Path) -> GmshVolumeMesh:
    """Parse nodes and tetrahedra from an ASCII or binary GMSH 2.x source."""
    source = Path(path).expanduser().resolve()
    raw = source.read_bytes()
    version, binary, data_size, endian = _parse_format(raw)
    node_count, node_offset = _section_count_and_offset(raw, "Nodes")
    element_count, element_offset = _section_count_and_offset(raw, "Elements")
    if binary:
        node_order, nodes = _parse_binary_nodes(raw, node_count, node_offset, endian, data_size)
        tetrahedra, type_counts = _parse_binary_elements(raw, element_count, element_offset, endian)
    else:
        node_order, nodes = _parse_ascii_nodes(raw, node_count, node_offset)
        tetrahedra, type_counts = _parse_ascii_elements(raw, element_count, element_offset)
    if not tetrahedra:
        raise GmshFormatError("source contains no supported tetrahedral volume elements")
    missing = sorted({node for tet in tetrahedra for node in tet if node not in nodes})
    if missing:
        raise GmshFormatError(f"tetrahedra reference missing node ids: {missing[:8]}")
    return GmshVolumeMesh(
        path=str(source),
        format_version=version,
        binary=binary,
        endian="little" if endian == "<" else "big",
        data_size=data_size,
        node_order=node_order,
        nodes=nodes,
        tetrahedra=tetrahedra,
        element_count=element_count,
        element_type_counts=type_counts,
    )


def _bbox(points: Iterable[Vec3]) -> tuple[list[float], list[float], list[float]]:
    values = tuple(points)
    if not values:
        return [0.0] * 3, [0.0] * 3, [0.0] * 3
    minimum = [min(point[axis] for point in values) for axis in range(3)]
    maximum = [max(point[axis] for point in values) for axis in range(3)]
    dimensions = [maximum[axis] - minimum[axis] for axis in range(3)]
    return minimum, maximum, dimensions


def _tet_volume_and_centroid(mesh: GmshVolumeMesh) -> tuple[float, list[float]]:
    total = 0.0
    weighted = [0.0, 0.0, 0.0]
    for tet in mesh.tetrahedra:
        a, b, c, d = (mesh.nodes[node_id] for node_id in tet)
        volume = abs(_dot(_sub(b, a), _cross(_sub(c, a), _sub(d, a)))) / 6.0
        if volume <= 1e-18:
            raise GmshFormatError(f"zero-volume tetrahedron {tet}")
        centroid = [(a[i] + b[i] + c[i] + d[i]) / 4.0 for i in range(3)]
        total += volume
        for axis in range(3):
            weighted[axis] += volume * centroid[axis]
    return total, [value / total for value in weighted]


def _connected_face_components(faces: Sequence[Face]) -> int:
    if not faces:
        return 0
    parent = list(range(len(faces)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces[tuple(sorted((a, b)))].append(face_id)
    for owners in edge_faces.values():
        for other in owners[1:]:
            union(owners[0], other)
    return len({find(index) for index in range(len(faces))})


def _surface_metrics(
    vertices: Sequence[Vec3],
    faces: Sequence[Face],
    source: GmshVolumeMesh,
    internal_face_count: int,
) -> dict[str, Any]:
    bbox_min, bbox_max, dimensions = _bbox(vertices)
    max_dimension = max(dimensions, default=0.0)
    aspect_ratios = [value / max_dimension if max_dimension else 0.0 for value in dimensions]
    area = 0.0
    signed_volume = 0.0
    volume_centroid_numerator = [0.0, 0.0, 0.0]
    edge_counts: Counter[tuple[int, int]] = Counter()
    edge_orientation: Counter[tuple[int, int]] = Counter()
    for face in faces:
        a, b, c = (vertices[index] for index in face)
        cross = _cross(_sub(b, a), _sub(c, a))
        area += 0.5 * math.sqrt(_dot(cross, cross))
        volume = _dot(a, _cross(b, c)) / 6.0
        signed_volume += volume
        for axis in range(3):
            volume_centroid_numerator[axis] += volume * (a[axis] + b[axis] + c[axis]) / 4.0
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            canonical = tuple(sorted((left, right)))
            edge_counts[canonical] += 1
            edge_orientation[canonical] += 1 if (left, right) == canonical else -1
    enclosed_volume = abs(signed_volume)
    volume_centroid = (
        [value / signed_volume for value in volume_centroid_numerator]
        if abs(signed_volume) > 1e-18
        else [0.0, 0.0, 0.0]
    )
    source_volume, source_centroid = _tet_volume_and_centroid(source)
    source_bbox_min, source_bbox_max, source_dimensions = _bbox(source.nodes.values())
    boundary_edges = sum(count == 1 for count in edge_counts.values())
    nonmanifold_edges = sum(count > 2 for count in edge_counts.values())
    winding_mismatches = sum(
        count == 2 and edge_orientation[edge] != 0 for edge, count in edge_counts.items()
    )
    bbox_error = max(
        [abs(a - b) for a, b in zip(source_bbox_min, bbox_min)]
        + [abs(a - b) for a, b in zip(source_bbox_max, bbox_max)],
        default=0.0,
    )
    volume_error = abs(enclosed_volume - source_volume)
    centroid_error = math.sqrt(sum((a - b) ** 2 for a, b in zip(volume_centroid, source_centroid)))
    tolerance = max(1e-12, source_volume * 1e-9)
    valid = (
        bbox_error <= 1e-12
        and volume_error <= tolerance
        and centroid_error <= 1e-10
        and boundary_edges == 0
        and nonmanifold_edges == 0
        and winding_mismatches == 0
        and signed_volume > 0.0
    )
    return {
        "source_node_count": len(source.nodes),
        "source_tetrahedron_count": len(source.tetrahedra),
        "source_element_count": source.element_count,
        "source_element_type_counts": {str(k): v for k, v in source.element_type_counts.items()},
        "converted_vertex_count": len(vertices),
        "triangle_count": len(faces),
        "removed_internal_face_pairs": internal_face_count,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_dimensions": dimensions,
        "bbox_center": [(low + high) / 2.0 for low, high in zip(bbox_min, bbox_max)],
        "aspect_ratios_to_max_dimension": aspect_ratios,
        "surface_area": area,
        "signed_volume": signed_volume,
        "enclosed_volume": enclosed_volume,
        "volume_centroid": volume_centroid,
        "connected_components": _connected_face_components(faces),
        "unique_edge_count": len(edge_counts),
        "boundary_edge_count": boundary_edges,
        "nonmanifold_edge_count": nonmanifold_edges,
        "winding_mismatch_edge_count": winding_mismatches,
        "watertight": boundary_edges == 0 and nonmanifold_edges == 0,
        "source_bbox_min": source_bbox_min,
        "source_bbox_max": source_bbox_max,
        "source_bbox_dimensions": source_dimensions,
        "source_tetrahedral_volume": source_volume,
        "source_volume_centroid": source_centroid,
        "bbox_max_abs_error": bbox_error,
        "volume_abs_error": volume_error,
        "volume_relative_error": volume_error / source_volume if source_volume else 0.0,
        "volume_centroid_l2_error": centroid_error,
        "geometry_validation_passed": valid,
    }


def extract_exterior_surface(mesh: GmshVolumeMesh) -> SurfaceMesh:
    """Return the once-occurring, outward-wound faces of all tetrahedra."""
    face_incidence: dict[tuple[int, int, int], list[Face]] = defaultdict(list)
    for tet in mesh.tetrahedra:
        for opposite_index in range(4):
            opposite = tet[opposite_index]
            face_nodes = [tet[index] for index in range(4) if index != opposite_index]
            p0, p1, p2 = (mesh.nodes[node_id] for node_id in face_nodes)
            inward = _sub(mesh.nodes[opposite], p0)
            if _dot(_cross(_sub(p1, p0), _sub(p2, p0)), inward) > 0.0:
                face_nodes[1], face_nodes[2] = face_nodes[2], face_nodes[1]
            oriented = tuple(face_nodes)
            face_incidence[tuple(sorted(face_nodes))].append(oriented)  # type: ignore[arg-type]
    excessive = [key for key, owners in face_incidence.items() if len(owners) > 2]
    if excessive:
        raise GmshFormatError(
            f"non-manifold volume mesh: {len(excessive)} tetrahedron faces have incidence > 2"
        )
    exterior_by_key = {
        key: owners[0] for key, owners in face_incidence.items() if len(owners) == 1
    }
    if not exterior_by_key:
        raise GmshFormatError("tetrahedral mesh has no exterior faces")
    boundary_nodes = {node_id for face in exterior_by_key.values() for node_id in face}
    source_node_ids = tuple(node_id for node_id in mesh.node_order if node_id in boundary_nodes)
    compact_index = {node_id: index for index, node_id in enumerate(source_node_ids)}
    vertices = tuple(mesh.nodes[node_id] for node_id in source_node_ids)
    faces = tuple(
        tuple(compact_index[node_id] for node_id in exterior_by_key[key])
        for key in sorted(exterior_by_key)
    )
    internal_face_pairs = sum(len(owners) == 2 for owners in face_incidence.values())
    metrics = _surface_metrics(vertices, faces, mesh, internal_face_pairs)
    if not metrics["geometry_validation_passed"]:
        raise GmshFormatError(
            "extracted surface failed geometry validation: "
            f"bbox_error={metrics['bbox_max_abs_error']}, "
            f"volume_error={metrics['volume_abs_error']}, "
            f"boundary_edges={metrics['boundary_edge_count']}, "
            f"nonmanifold_edges={metrics['nonmanifold_edge_count']}, "
            f"winding_mismatches={metrics['winding_mismatch_edge_count']}"
        )
    return SurfaceMesh(vertices, source_node_ids, faces, metrics)


def _render_obj(surface: SurfaceMesh, *, source_hash: str, source_name: str) -> str:
    lines = [
        f"# {CONVERTER_VERSION}",
        f"# source {source_name}",
        f"# source_sha256 {source_hash}",
        "o rigid_exterior_surface",
    ]
    lines.extend("v " + " ".join(format(value, ".17g") for value in vertex) for vertex in surface.vertices)
    lines.extend("f " + " ".join(str(index + 1) for index in face) for face in surface.faces)
    return "\n".join(lines) + "\n"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cache_key(source_hash: str) -> str:
    payload = json.dumps(
        {
            "source_sha256": source_hash,
            "converter_version": CONVERTER_VERSION,
            "conversion_parameters": CONVERSION_PARAMETERS,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _valid_cached_result(
    mesh_path: Path,
    manifest_path: Path,
    source_hash: str,
    cache_key: str,
) -> dict[str, Any] | None:
    if not mesh_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        manifest.get("source_hash") != source_hash
        or manifest.get("cache_key") != cache_key
        or manifest.get("converter_version") != CONVERTER_VERSION
        or manifest.get("conversion_parameters") != CONVERSION_PARAMETERS
        or not manifest.get("geometry", {}).get("geometry_validation_passed")
    ):
        return None
    if _sha256_bytes(mesh_path.read_bytes()) != manifest.get("converted_hash"):
        return None
    return manifest


def convert_gmsh_to_rigid_surface(
    source_path: str | Path,
    cache_dir: str | Path,
) -> ConversionResult:
    """Convert once per source hash and return a validated cached OBJ surface."""
    source = Path(source_path).expanduser().resolve()
    source_bytes = source.read_bytes()
    source_hash = _sha256_bytes(source_bytes)
    cache_key = _cache_key(source_hash)
    cache = Path(cache_dir).expanduser().resolve()
    stem = f"{source.stem}-{cache_key[:16]}"
    mesh_path = cache / f"{stem}.obj"
    manifest_path = cache / f"{stem}.conversion.json"
    cached = _valid_cached_result(mesh_path, manifest_path, source_hash, cache_key)
    if cached is not None:
        return ConversionResult(
            source, source_hash, cache_key, mesh_path, manifest_path, True, cached
        )

    parsed = parse_gmsh_v2(source)
    surface = extract_exterior_surface(parsed)
    obj_bytes = _render_obj(surface, source_hash=source_hash, source_name=source.name).encode("utf-8")
    manifest: dict[str, Any] = {
        "source_mesh": str(source),
        "source_hash": source_hash,
        "source_format": {
            "version": parsed.format_version,
            "binary": parsed.binary,
            "endian": parsed.endian,
            "data_size": parsed.data_size,
        },
        "cache_key": cache_key,
        "converter_version": CONVERTER_VERSION,
        "conversion_parameters": dict(CONVERSION_PARAMETERS),
        "converted_mesh": str(mesh_path),
        "converted_hash": _sha256_bytes(obj_bytes),
        "geometry": surface.metrics,
    }
    _atomic_write(mesh_path, obj_bytes)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return ConversionResult(
        source, source_hash, cache_key, mesh_path, manifest_path, False, manifest
    )


def scaled_geometry_metrics(geometry: dict[str, Any], scale: Sequence[float]) -> dict[str, Any]:
    if len(scale) != 3 or any(value <= 0.0 for value in scale):
        raise ValueError("scale must contain three positive values")
    result = dict(geometry)
    for key in ("bbox_min", "bbox_max", "bbox_dimensions", "bbox_center", "volume_centroid"):
        result[key] = [float(value) * float(scale[axis]) for axis, value in enumerate(geometry[key])]
    result["surface_area"] = None if len(set(scale)) != 1 else geometry["surface_area"] * scale[0] ** 2
    result["enclosed_volume"] = geometry["enclosed_volume"] * math.prod(scale)
    result["signed_volume"] = geometry["signed_volume"] * math.prod(scale)
    result["mesh_scale"] = [float(value) for value in scale]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--scale", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = convert_gmsh_to_rigid_surface(args.source, args.cache_dir)
    report = dict(result.manifest)
    report["cache_reused"] = result.cache_reused
    report["scaled_geometry"] = scaled_geometry_metrics(report["geometry"], args.scale)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.report is not None:
        _atomic_write(args.report.expanduser().resolve(), rendered.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
