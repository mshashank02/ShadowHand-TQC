import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from object_conversion.convex_decomposition import ConvexPiece, load_convex_piece_obj
from object_conversion.radius_aware_collision import (
    ShellParameters,
    build_radius_aware_model,
    expand_convex_piece,
    extract_rigid_flex_radius_m,
    rigid_flex_radius_m,
    sphere_polytope_vertices,
)


def tetrahedron() -> ConvexPiece:
    return ConvexPiece(
        vertices_m=np.asarray(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64),
    )


def write_piece(path: Path, piece: ConvexPiece) -> None:
    lines = [*("v " + " ".join(map(str, vertex)) for vertex in piece.vertices_m)]
    lines.extend("f " + " ".join(str(int(index) + 1) for index in face) for face in piece.faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model(path: Path) -> None:
    path.write_text(
        """<mujoco>
  <compiler meshdir="meshes" texturedir="textures"/>
  <asset>
    <mesh name="visual" file="visual.obj" scale="2 2 2"/>
    <mesh name="piece" file="piece.obj" scale="1 1 1"/>
  </asset>
  <worldbody>
    <body name="object" pos="1 2 3">
      <joint name="object:joint" type="free" damping="0.05"/>
      <inertial pos="0 0 0" mass="0.7" diaginertia=".1 .2 .3"/>
      <geom name="object_visual" type="mesh" mesh="visual" contype="0" conaffinity="0"/>
      <geom name="object_collision_000" type="mesh" mesh="piece" margin="0" gap="0"
            friction="1 .005 .0001" solref=".02 1" solimp=".9 .95 .001 .5 2"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )


class RadiusAwareCollisionTests(unittest.TestCase):
    def test_extracts_original_worst_object_flex_radius(self):
        repository = Path(__file__).resolve().parents[1]
        models = json.loads(
            (repository / "generated/convex_decomposition_validation/contact_models/models.json").read_text()
        )
        self.assertEqual(extract_rigid_flex_radius_m(models["rigid_flex_reference"]), 0.00125)

    def test_size_scaled_rigid_radius_rule(self):
        self.assertEqual(rigid_flex_radius_m("small"), 0.00075)
        self.assertEqual(rigid_flex_radius_m("medium"), 0.001)
        self.assertEqual(rigid_flex_radius_m("large"), 0.00125)
        with self.assertRaisesRegex(ValueError, "unsupported object size"):
            rigid_flex_radius_m("giant")

    def test_circumscribed_sphere_polytope_contains_unit_sphere(self):
        vertices, metadata = sphere_polytope_vertices(2, bound="circumscribed")
        directions = np.random.default_rng(17).normal(size=(10_000, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        support = np.max(directions @ vertices.T, axis=1)
        self.assertGreaterEqual(float(support.min()), 1.0 - 1e-12)
        self.assertEqual(metadata["direction_count"], 162)
        self.assertLess(float(metadata["maximum_radial_excess_fraction"]), 0.02)

    def test_minkowski_expansion_is_convex_and_offsets_support(self):
        source = tetrahedron()
        shell = 0.00125
        expanded, metadata = expand_convex_piece(source, shell)
        directions = np.random.default_rng(31).normal(size=(2_000, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        source_support = np.max(directions @ source.vertices_m.T, axis=1)
        expanded_support = np.max(directions @ expanded.vertices_m.T, axis=1)
        offsets = expanded_support - source_support
        self.assertGreaterEqual(float(offsets.min()), shell - 1e-12)
        self.assertLessEqual(float(offsets.max()), shell * 1.019)
        self.assertTrue(metadata["validation"]["convex"])
        repeated, repeated_metadata = expand_convex_piece(source, shell)
        np.testing.assert_array_equal(expanded.vertices_m, repeated.vertices_m)
        np.testing.assert_array_equal(expanded.faces, repeated.faces)
        self.assertEqual(metadata, repeated_metadata)

    def test_model_builder_changes_only_collision_shell_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meshes = root / "meshes"
            meshes.mkdir()
            (root / "textures").mkdir()
            write_piece(meshes / "piece.obj", tetrahedron())
            write_piece(meshes / "visual.obj", tetrahedron())
            source = root / "source.xml"
            write_model(source)

            margin_xml = root / "output" / "margin.xml"
            build_radius_aware_model(source, margin_xml, ShellParameters("margin", 0.00125))
            margin_root = ET.parse(margin_xml).getroot()
            margin_geom = margin_root.find(".//geom[@name='object_collision_000']")
            self.assertEqual(float(margin_geom.get("margin")), 0.00125)
            self.assertEqual(margin_geom.get("gap"), "0")
            self.assertEqual(margin_root.find(".//inertial").attrib["mass"], "0.7")
            object_body = margin_root.find(".//body[@name='object']")
            self.assertEqual(len(object_body.findall("joint[@type='free']")), 1)
            self.assertEqual(object_body.find("inertial").attrib["diaginertia"], ".1 .2 .3")
            visual = margin_root.find(".//geom[@name='object_visual']")
            self.assertEqual(visual.get("mesh"), "visual")
            self.assertEqual(visual.get("contype"), "0")
            self.assertEqual(visual.get("conaffinity"), "0")
            self.assertEqual(margin_geom.get("friction"), "1 .005 .0001")
            self.assertEqual(margin_geom.get("solref"), ".02 1")
            self.assertEqual(margin_geom.get("solimp"), ".9 .95 .001 .5 2")

            gap_xml = root / "output" / "margin_gap.xml"
            build_radius_aware_model(
                source, gap_xml, ShellParameters("margin", 0.00125, gap_m=0.00025)
            )
            gap_geom = ET.parse(gap_xml).getroot().find(
                ".//geom[@name='object_collision_000']"
            )
            self.assertEqual(float(gap_geom.get("margin")), 0.00125)
            self.assertEqual(float(gap_geom.get("gap")), 0.00025)

            shell_xml = root / "output" / "shell.xml"
            manifest = build_radius_aware_model(
                source, shell_xml, ShellParameters("minkowski", 0.00125)
            )
            shell_root = ET.parse(shell_xml).getroot()
            shell_geom = shell_root.find(".//geom[@name='object_collision_000']")
            shell_mesh = shell_root.find("./asset/mesh[@name='piece']")
            self.assertEqual(shell_geom.get("margin"), "0")
            self.assertEqual(shell_geom.get("gap"), "0")
            self.assertEqual(shell_mesh.get("maxhullvert"), "-1")
            expanded = load_convex_piece_obj(shell_mesh.get("file"))
            self.assertGreater(len(expanded.vertices_m), 4)
            self.assertEqual(manifest["collision_geom_count"], 1)
            persisted = json.loads(shell_xml.with_suffix(".manifest.json").read_text())
            self.assertEqual(persisted["output_xml_sha256"], manifest["output_xml_sha256"])
            try:
                import mujoco
            except ImportError:
                return
            model = mujoco.MjModel.from_xml_path(str(shell_xml))
            self.assertEqual(model.nflex, 0)
            self.assertEqual(model.nq, 7)


if __name__ == "__main__":
    unittest.main()
