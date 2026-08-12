import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from object_conversion import convert_gmsh_to_rigid_surface
from pipeline_generate import (
    RIGID_GEOM_CONTACT_ATTRIBUTES,
    RIGID_MESH_ASSET_NAME,
    build_candidate_standalone,
    patch_env_object_to_custom_msh,
    write_rigid_representation_manifest,
)


def write_tetrahedron(path: Path) -> None:
    path.write_text(
        """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
$EndNodes
$Elements
1
1 4 0 1 2 3 4
$EndElements
""",
        encoding="utf-8",
    )


def write_template(path: Path) -> None:
    path.write_text(
        """<mujoco>
  <asset/>
  <worldbody>
    <body name="object" pos="0 0 0">
      <joint name="old" type="free"/>
      <geom name="old" type="box" size=".1 .1 .1"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )


class RigidMeshGenerationTests(unittest.TestCase):
    def test_rigid_branch_uses_mesh_geom_and_preserves_dynamics_and_contacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.xml"
            write_template(path)
            patch_env_object_to_custom_msh(
                str(path),
                "source.msh",
                rigid_surface_file_for_xml="surface.obj",
                object_mass=0.75,
                object_inertia="0.1 0.2 0.3",
                object_pos="1 2 3",
                flex_scale="0.01 0.02 0.03",
            )
            root = ET.parse(path).getroot()
            body = root.find("./worldbody/body[@name='object']")
            self.assertIsNotNone(body)
            self.assertIsNone(body.find("flexcomp"))
            joint = body.find("joint")
            self.assertEqual(joint.attrib["name"], "object:joint")
            self.assertEqual(joint.attrib["type"], "free")
            inertial = body.find("inertial")
            self.assertEqual(inertial.attrib["mass"], "0.75")
            self.assertEqual(inertial.attrib["diaginertia"], "0.1 0.2 0.3")
            geom = body.find("./geom[@name='object']")
            self.assertEqual(geom.attrib["type"], "mesh")
            self.assertEqual(geom.attrib["mesh"], RIGID_MESH_ASSET_NAME)
            for name, value in RIGID_GEOM_CONTACT_ATTRIBUTES.items():
                self.assertEqual(geom.attrib[name], value)
            mesh = root.find(f"./asset/mesh[@name='{RIGID_MESH_ASSET_NAME}']")
            self.assertEqual(mesh.attrib["file"], "surface.obj")
            self.assertEqual(mesh.attrib["scale"], "0.01 0.02 0.03")

    def test_deformable_branch_still_uses_flexcomp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.xml"
            write_template(path)
            patch_env_object_to_custom_msh(
                str(path),
                "source.msh",
                deformable=True,
                deformable_preset="soft_rubber_stable",
            )
            root = ET.parse(path).getroot()
            body = root.find("./worldbody/body[@name='object']")
            flex = body.find("flexcomp")
            self.assertIsNotNone(flex)
            self.assertEqual(flex.attrib["type"], "gmsh")
            self.assertEqual(flex.attrib["rigid"], "false")
            self.assertIsNotNone(flex.find("elasticity"))
            self.assertIsNone(root.find(f"./asset/mesh[@name='{RIGID_MESH_ASSET_NAME}']"))

    def test_representation_manifest_records_geometry_mass_and_inertia(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.msh"
            write_tetrahedron(source)
            conversion = convert_gmsh_to_rigid_surface(source, root / "cache")
            manifest = root / "representation.json"
            write_rigid_representation_manifest(
                str(manifest),
                conversion_result=conversion,
                generated_surface_path=str(conversion.mesh_path),
                mesh_scale="0.1 0.1 0.1",
                object_mass=0.5,
                object_inertia="0.01 0.02 0.03",
                object_pos="1 2 3",
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["representation"], "rigid_mesh_geom")
            self.assertEqual(payload["mass"], 0.5)
            self.assertEqual(payload["diaginertia"], [0.01, 0.02, 0.03])
            self.assertEqual(payload["source_hash"], conversion.source_hash)
            self.assertTrue(payload["scaled_geometry"]["geometry_validation_passed"])
            self.assertEqual(payload["scaled_geometry"]["bbox_dimensions"], [0.1, 0.1, 0.1])

    def test_real_custom_mesh_n500_n1000_contract_and_candidate_independent_cache(self):
        repository = Path(__file__).resolve().parents[1]
        source = (
            repository
            / "study_objects/sphere_study_v1/obj_size-large_ar-high_macro-high_rough-high.msh"
        )
        if not source.is_file():
            self.skipTest("representative study object is unavailable")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = Path(tmpdir) / "cache"
            results = []
            for sensors in (500, 1000):
                paths = build_candidate_standalone(
                    task="custom_contract",
                    Ntotal=sensors,
                    Rppx=1.071429,
                    Rpt=0.714286,
                    Ap=6557,
                    Apx=26885,
                    At=7193,
                    Ap1=5557,
                    Ap2=1000,
                    base_xml=str(repository / "assets/hand_base.xml"),
                    template_xml=str(repository / "assets/manipulate_block_touch_sensors.xml"),
                    out_root=tmpdir,
                    force=True,
                    custom_msh=str(source),
                    deformable_object=False,
                    rigid_mesh_cache_dir=str(cache),
                )
                shared_root = ET.parse(paths["shared"]).getroot()
                self.assertEqual(len(shared_root.findall(".//touch")), sensors)
                env_root = ET.parse(paths["env"]).getroot()
                body = env_root.find("./worldbody/body[@name='object']")
                self.assertIsNone(body.find("flexcomp"))
                self.assertEqual(body.find("./geom[@name='object']").attrib["type"], "mesh")
                manifest = json.loads(
                    Path(paths["rigid_representation_manifest"]).read_text(encoding="utf-8")
                )
                results.append((paths, manifest))
                # The task contract is 61 physical values plus N touch values and
                # one TimeFeatureWrapper value.
                self.assertEqual(sensors + 62, 562 if sensors == 500 else 1062)
            self.assertEqual(results[0][1]["cache_key"], results[1][1]["cache_key"])
            self.assertEqual(
                Path(results[0][0]["rigid_surface"]).name,
                Path(results[1][0]["rigid_surface"]).name,
            )
            self.assertTrue(results[1][1]["cache_reused"])


if __name__ == "__main__":
    unittest.main()
