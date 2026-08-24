#!/usr/bin/env python3
"""Build radius-aware convex-piece collision models for offline validation.

The rigid-flex reference collides each active tetrahedron as ``K + B(radius)``.
This module provides two deliberately distinct rigid-geom candidates:

* ``margin`` is a MuJoCo contact-margin diagnostic.  It can reproduce an onset
  offset, but it is not a geometric Minkowski sum and is not the production
  recommendation.
* ``minkowski`` replaces every convex piece with the convex hull of its exact
  Minkowski sum with a deterministic polyhedral approximation of a sphere.

All source meshes and generated vertices are in metres.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

from object_conversion.convex_decomposition import (
    ConvexPiece,
    load_convex_piece_obj,
    validate_convex_piece,
)


RADIUS_AWARE_CONVERTER_VERSION = "radius-aware-convex-shell-v1"
BASE_RIGID_FLEX_RADIUS_M = 0.001
SIZE_SCALE_MULTIPLIERS = {"small": 0.75, "medium": 1.0, "large": 1.25}


@dataclass(frozen=True)
class ShellParameters:
    strategy: str
    shell_m: float
    gap_m: float = 0.0
    sphere_subdivisions: int = 2
    sphere_bound: str = "circumscribed"
    residual_margin_m: float = 0.0

    def __post_init__(self) -> None:
        if self.strategy not in {"margin", "minkowski", "hybrid"}:
            raise ValueError(f"unsupported strategy {self.strategy!r}")
        if not math.isfinite(self.shell_m) or self.shell_m < 0.0:
            raise ValueError("shell_m must be finite and non-negative")
        if not math.isfinite(self.gap_m) or self.gap_m < 0.0:
            raise ValueError("gap_m must be finite and non-negative")
        if self.gap_m > 0.0 and self.strategy != "margin":
            raise ValueError("nonzero gap_m is only valid for margin diagnostics")
        if self.sphere_subdivisions < 0 or self.sphere_subdivisions > 4:
            raise ValueError("sphere_subdivisions must be between 0 and 4")
        if self.sphere_bound not in {"inscribed", "circumscribed"}:
            raise ValueError("sphere_bound must be inscribed or circumscribed")
        if not math.isfinite(self.residual_margin_m) or self.residual_margin_m < 0.0:
            raise ValueError("residual_margin_m must be finite and non-negative")
        if self.strategy != "hybrid" and self.residual_margin_m != 0.0:
            raise ValueError("residual_margin_m is only valid for hybrid candidates")


def rigid_flex_radius_m(size_label: str) -> float:
    """Return the production rigid-flex radius after object-size scaling."""
    try:
        multiplier = SIZE_SCALE_MULTIPLIERS[size_label]
    except KeyError as exc:
        raise ValueError(f"unsupported object size {size_label!r}") from exc
    return BASE_RIGID_FLEX_RADIUS_M * multiplier


def extract_rigid_flex_radius_m(xml_path: str | Path) -> float:
    """Extract the one explicit rigid ``flexcomp`` radius from a generated XML."""
    root = ET.parse(Path(xml_path)).getroot()
    rigid_flexcomps = [
        flexcomp
        for flexcomp in root.findall(".//flexcomp")
        if flexcomp.get("rigid", "false").lower() == "true"
    ]
    if len(rigid_flexcomps) != 1:
        raise ValueError(
            f"expected exactly one rigid flexcomp, found {len(rigid_flexcomps)}"
        )
    radius = rigid_flexcomps[0].get("radius")
    if radius is None:
        raise ValueError("rigid flexcomp has no explicit radius")
    value = float(radius)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("rigid flexcomp radius must be finite and non-negative")
    return value


def _normalized(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def sphere_polytope_vertices(
    subdivisions: int = 2, *, bound: str = "circumscribed"
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Return deterministic unit-sphere polytope vertices and error metadata.

    The inscribed form places vertices on the unit sphere.  The circumscribed
    form uniformly scales that polytope by its inradius, guaranteeing that it
    contains the unit sphere.  The latter avoids silently under-filling the
    reference flex radius.
    """
    if subdivisions < 0 or subdivisions > 4:
        raise ValueError("subdivisions must be between 0 and 4")
    if bound not in {"inscribed", "circumscribed"}:
        raise ValueError("bound must be inscribed or circumscribed")

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    vertices = _normalized(vertices)
    faces = [list(map(int, face)) for face in ConvexHull(vertices).simplices]

    for _ in range(subdivisions):
        vertex_list = [vertex.copy() for vertex in vertices]
        midpoint_cache: dict[tuple[int, int], int] = {}

        def midpoint(left: int, right: int) -> int:
            edge = tuple(sorted((left, right)))
            if edge not in midpoint_cache:
                value = vertex_list[left] + vertex_list[right]
                value /= np.linalg.norm(value)
                midpoint_cache[edge] = len(vertex_list)
                vertex_list.append(value)
            return midpoint_cache[edge]

        refined: list[list[int]] = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]))
        vertices = np.asarray(vertex_list, dtype=np.float64)
        faces = refined

    hull = ConvexHull(vertices)
    normals = hull.equations[:, :3]
    offsets = hull.equations[:, 3]
    inradius = float(np.min(-offsets / np.linalg.norm(normals, axis=1)))
    scale = 1.0 if bound == "inscribed" else 1.0 / inradius
    result = vertices * scale
    metadata: dict[str, float | int | str] = {
        "bound": bound,
        "subdivisions": subdivisions,
        "direction_count": int(len(result)),
        "inscribed_polytope_inradius": inradius,
        "radial_scale": scale,
        "maximum_radial_excess_fraction": max(0.0, scale - 1.0),
    }
    return result, metadata


def _oriented_hull_piece(points: np.ndarray) -> ConvexPiece:
    hull = ConvexHull(np.asarray(points, dtype=np.float64))
    used = np.asarray(hull.vertices, dtype=np.int64)
    remap = {int(old): new for new, old in enumerate(used)}
    vertices = np.asarray(points, dtype=np.float64)[used]
    faces: list[list[int]] = []
    for simplex, equation in zip(hull.simplices, hull.equations):
        a, b, c = [int(value) for value in simplex]
        cross = np.cross(points[b] - points[a], points[c] - points[a])
        if float(np.dot(cross, equation[:3])) < 0.0:
            b, c = c, b
        faces.append([remap[a], remap[b], remap[c]])
    piece = ConvexPiece(vertices_m=vertices, faces=np.asarray(faces, dtype=np.int64))
    validate_convex_piece(piece)
    return piece


def expand_convex_piece(
    piece: ConvexPiece,
    shell_m: float,
    *,
    sphere_subdivisions: int = 2,
    sphere_bound: str = "circumscribed",
) -> tuple[ConvexPiece, dict[str, Any]]:
    """Approximate ``piece + B(shell_m)`` as a validated convex mesh."""
    if not math.isfinite(shell_m) or shell_m < 0.0:
        raise ValueError("shell_m must be finite and non-negative")
    validate_convex_piece(piece)
    directions, sphere_metadata = sphere_polytope_vertices(
        sphere_subdivisions, bound=sphere_bound
    )
    source_vertices = np.asarray(piece.vertices_m, dtype=np.float64)
    if shell_m == 0.0:
        expanded = _oriented_hull_piece(source_vertices)
    else:
        sums = source_vertices[:, None, :] + shell_m * directions[None, :, :]
        expanded = _oriented_hull_piece(sums.reshape((-1, 3)))
    validation = validate_convex_piece(expanded)
    metadata = {
        "shell_m": shell_m,
        "sphere": sphere_metadata,
        "source_vertex_count": int(len(source_vertices)),
        "expanded_vertex_count": int(len(expanded.vertices_m)),
        "expanded_triangle_count": int(len(expanded.faces)),
        "validation": validation,
    }
    return expanded, metadata


def _render_obj(piece: ConvexPiece, metadata: dict[str, Any]) -> str:
    lines = [
        f"# {RADIUS_AWARE_CONVERTER_VERSION}",
        "# " + json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        "o radius_aware_convex_piece",
    ]
    lines.extend(
        "v " + " ".join(format(float(value), ".17g") for value in vertex)
        for vertex in piece.vertices_m
    )
    lines.extend(
        "f " + " ".join(str(int(index) + 1) for index in face)
        for face in piece.faces
    )
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_compiler_directory(source_xml: Path, value: str | None) -> Path:
    path = Path(value or ".")
    if not path.is_absolute():
        path = source_xml.parent / path
    return path.resolve()


def _object_collision_geoms(root: ET.Element) -> list[ET.Element]:
    return [
        geom
        for geom in root.findall(".//geom")
        if (geom.get("name") or "").startswith("object_collision_")
    ]


def build_radius_aware_model(
    source_xml: str | Path,
    output_xml: str | Path,
    parameters: ShellParameters,
) -> dict[str, Any]:
    """Write one validation model and return its reproducibility manifest."""
    source_xml = Path(source_xml).resolve()
    output_xml = Path(output_xml).resolve()
    tree = ET.parse(source_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    mesh_dir = _resolve_compiler_directory(source_xml, compiler.get("meshdir"))
    texture_dir = _resolve_compiler_directory(source_xml, compiler.get("texturedir"))
    compiler.set("meshdir", str(mesh_dir))
    compiler.set("texturedir", str(texture_dir))
    for include in root.findall(".//include"):
        include_path = Path(include.get("file", ""))
        if not include_path.is_absolute():
            include.set("file", str((source_xml.parent / include_path).resolve()))

    collision_geoms = _object_collision_geoms(root)
    if not collision_geoms:
        raise ValueError(f"no object_collision_* geoms in {source_xml}")
    margin_m = parameters.shell_m if parameters.strategy == "margin" else parameters.residual_margin_m
    for geom in collision_geoms:
        geom.set("margin", format(margin_m, ".17g"))
        geom.set("gap", format(parameters.gap_m, ".17g"))

    piece_records: list[dict[str, Any]] = []
    if parameters.strategy in {"minkowski", "hybrid"}:
        mesh_assets = {
            mesh.get("name"): mesh for mesh in root.findall("./asset/mesh") if mesh.get("name")
        }
        asset_dir = output_xml.parent / "assets" / output_xml.stem
        for geom in collision_geoms:
            mesh_name = geom.get("mesh")
            if not mesh_name or mesh_name not in mesh_assets:
                raise ValueError(f"collision geom {geom.get('name')!r} has no local mesh asset")
            mesh = mesh_assets[mesh_name]
            source_mesh = Path(mesh.get("file", ""))
            if not source_mesh.is_absolute():
                source_mesh = mesh_dir / source_mesh
            source_mesh = source_mesh.resolve()
            expanded, expansion = expand_convex_piece(
                load_convex_piece_obj(source_mesh),
                parameters.shell_m,
                sphere_subdivisions=parameters.sphere_subdivisions,
                sphere_bound=parameters.sphere_bound,
            )
            asset_dir.mkdir(parents=True, exist_ok=True)
            output_mesh = asset_dir / f"{mesh_name}.obj"
            record = {
                "geom": geom.get("name"),
                "mesh": mesh_name,
                "source": str(source_mesh),
                "source_sha256": _sha256(source_mesh),
                **expansion,
            }
            output_mesh.write_text(_render_obj(expanded, record), encoding="utf-8")
            record["output"] = str(output_mesh)
            record["output_sha256"] = _sha256(output_mesh)
            piece_records.append(record)
            mesh.set("file", str(output_mesh))
            mesh.set("scale", "1 1 1")
            mesh.set("maxhullvert", "-1")

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    manifest = {
        "converter_version": RADIUS_AWARE_CONVERTER_VERSION,
        "source_xml": str(source_xml),
        "source_xml_sha256": _sha256(source_xml),
        "output_xml": str(output_xml),
        "output_xml_sha256": _sha256(output_xml),
        "parameters": asdict(parameters),
        "collision_geom_count": len(collision_geoms),
        "piece_records": piece_records,
        "invariants": {
            "visual_mesh_unchanged": True,
            "body_joint_inertial_elements_unchanged": True,
            "contact_parameters_unchanged_except_margin_gap": True,
            "collision_piece_count_unchanged": True,
        },
    }
    manifest_path = output_xml.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_candidate_manifest(
    models_manifest: str | Path,
    output_dir: str | Path,
    levels: Iterable[str],
    parameters: ShellParameters,
) -> dict[str, str]:
    models_path = Path(models_manifest).resolve()
    models = json.loads(models_path.read_text(encoding="utf-8"))
    output_dir = Path(output_dir).resolve()
    result = {"rigid_flex_reference": str(Path(models["rigid_flex_reference"]).resolve())}
    shell_um = round(parameters.shell_m * 1e6)
    for level in levels:
        if level not in models:
            raise ValueError(f"model manifest has no level {level!r}")
        output_xml = output_dir / f"{level}_{parameters.strategy}_{shell_um:04d}um.xml"
        build_radius_aware_model(models[level], output_xml, parameters)
        result[level] = str(output_xml)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"models_{parameters.strategy}_{shell_um:04d}um.json"
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", default=["coarse", "fine", "very_fine"])
    parser.add_argument("--strategy", choices=("margin", "minkowski", "hybrid"), required=True)
    parser.add_argument("--shell-mm", type=float, required=True)
    parser.add_argument("--residual-margin-mm", type=float, default=0.0)
    parser.add_argument("--gap-mm", type=float, default=0.0)
    parser.add_argument("--sphere-subdivisions", type=int, default=2)
    parser.add_argument("--sphere-bound", choices=("inscribed", "circumscribed"), default="circumscribed")
    args = parser.parse_args()
    parameters = ShellParameters(
        strategy=args.strategy,
        shell_m=args.shell_mm / 1000.0,
        gap_m=args.gap_mm / 1000.0,
        sphere_subdivisions=args.sphere_subdivisions,
        sphere_bound=args.sphere_bound,
        residual_margin_m=args.residual_margin_mm / 1000.0,
    )
    result = build_candidate_manifest(
        args.models_manifest, args.output_dir, args.levels, parameters
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
