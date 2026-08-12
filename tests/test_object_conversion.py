import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from object_conversion.gmsh_to_rigid_surface import (
    GmshFormatError,
    convert_gmsh_to_rigid_surface,
    extract_exterior_surface,
    parse_gmsh_v2,
)


def write_ascii_gmsh(path: Path, nodes, tetrahedra) -> None:
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(nodes))]
    lines.extend(f"{node_id} {x} {y} {z}" for node_id, x, y, z in nodes)
    lines.extend(["$EndNodes", "$Elements", str(len(tetrahedra))])
    lines.extend(
        f"{element_id} 4 0 " + " ".join(str(node) for node in tet)
        for element_id, tet in enumerate(tetrahedra, start=1)
    )
    lines.extend(["$EndElements", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


class ObjectConversionTests(unittest.TestCase):
    def test_two_tetrahedra_remove_shared_internal_face_and_preserve_geometry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "two_tets.msh"
            nodes = [
                (1, 0, 0, 0),
                (2, 1, 0, 0),
                (3, 0, 1, 0),
                (4, 0, 0, 1),
                (5, 0, 0, -1),
            ]
            write_ascii_gmsh(source, nodes, [(1, 2, 3, 4), (1, 3, 2, 5)])
            parsed = parse_gmsh_v2(source)
            surface = extract_exterior_surface(parsed)
            self.assertEqual(len(parsed.tetrahedra), 2)
            self.assertEqual(len(surface.faces), 6)
            self.assertEqual(surface.metrics["removed_internal_face_pairs"], 1)
            self.assertTrue(surface.metrics["watertight"])
            self.assertEqual(surface.metrics["winding_mismatch_edge_count"], 0)
            self.assertAlmostEqual(surface.metrics["enclosed_volume"], 1.0 / 3.0)
            self.assertEqual(surface.metrics["bbox_dimensions"], [1.0, 1.0, 2.0])
            shared = {0, 1, 2}
            self.assertFalse(any(set(face) == shared for face in surface.faces))

    def test_conversion_is_deterministic_and_cacheable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "tet.msh"
            write_ascii_gmsh(
                source,
                [(10, 0, 0, 0), (20, 1, 0, 0), (30, 0, 1, 0), (40, 0, 0, 1)],
                [(10, 20, 30, 40)],
            )
            first = convert_gmsh_to_rigid_surface(source, root / "cache")
            first_bytes = first.mesh_path.read_bytes()
            second = convert_gmsh_to_rigid_surface(source, root / "cache")
            self.assertFalse(first.cache_reused)
            self.assertTrue(second.cache_reused)
            self.assertEqual(first.mesh_path, second.mesh_path)
            self.assertEqual(first_bytes, second.mesh_path.read_bytes())
            self.assertEqual(
                hashlib.sha256(first_bytes).hexdigest(),
                second.manifest["converted_hash"],
            )
            self.assertNotIn("candidate", json.dumps(second.manifest).lower())

    def test_nonmanifold_tetrahedral_face_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "bad.msh"
            write_ascii_gmsh(
                source,
                [
                    (1, 0, 0, 0),
                    (2, 1, 0, 0),
                    (3, 0, 1, 0),
                    (4, 0, 0, 1),
                    (5, 0, 0, -1),
                    (6, 0, 0, 2),
                ],
                [(1, 2, 3, 4), (1, 3, 2, 5), (1, 2, 3, 6)],
            )
            with self.assertRaisesRegex(GmshFormatError, "incidence > 2"):
                extract_exterior_surface(parse_gmsh_v2(source))

    def test_real_study_family_format_topology_and_variations_are_preserved(self):
        repository = Path(__file__).resolve().parents[1]
        sources = sorted((repository / "study_objects/sphere_study_v1").glob("*.msh"))
        if len(sources) != 24:
            self.skipTest("the complete 24-object sphere study is unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            summaries = {}
            hashes = set()
            for source in sources:
                parsed = parse_gmsh_v2(source)
                self.assertEqual(parsed.format_version, "2.2")
                self.assertTrue(parsed.binary)
                self.assertEqual(parsed.endian, "little")
                self.assertEqual(parsed.data_size, 8)
                self.assertEqual(parsed.element_type_counts, {4: parsed.element_count})
                self.assertEqual(len(parsed.tetrahedra), parsed.element_count)
                result = convert_gmsh_to_rigid_surface(source, Path(tmpdir) / "cache")
                geometry = result.manifest["geometry"]
                self.assertTrue(geometry["watertight"])
                self.assertEqual(geometry["connected_components"], 1)
                self.assertEqual(geometry["boundary_edge_count"], 0)
                self.assertEqual(geometry["nonmanifold_edge_count"], 0)
                self.assertEqual(geometry["winding_mismatch_edge_count"], 0)
                self.assertTrue(geometry["geometry_validation_passed"])
                summaries[source.stem] = (
                    tuple(geometry["bbox_dimensions"]),
                    geometry["surface_area"],
                    geometry["enclosed_volume"],
                    geometry["triangle_count"],
                )
                hashes.add(result.source_hash)

            self.assertEqual(len(hashes), 24)
            base = "obj_size-medium_ar-low_macro-low_rough-low"
            for variant in (
                "obj_size-small_ar-low_macro-low_rough-low",
                "obj_size-large_ar-low_macro-low_rough-low",
                "obj_size-medium_ar-high_macro-low_rough-low",
                "obj_size-medium_ar-low_macro-high_rough-low",
                "obj_size-medium_ar-low_macro-low_rough-high",
            ):
                self.assertNotEqual(summaries[base], summaries[variant])


if __name__ == "__main__":
    unittest.main()
