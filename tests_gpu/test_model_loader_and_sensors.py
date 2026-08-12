from pathlib import Path
import tempfile
import unittest

import mujoco

from shadowhand_gpu.model_loader import is_single_body_rigid_flex, load_project_model
from shadowhand_gpu.sensors import build_sensor_layout


N500_XML = Path(
    "generated/smoke_tests/rigid_n0500_a0p3_b0p6_large_high_high_high/"
    "custom_obj_size_large_ar_high_macro_high_rough_high_500_1.071429_0.714286/"
    "manipulate_custom_obj_size_large_ar_high_macro_high_rough_high_touch_sensors_"
    "500_1.071429_0.714286.xml"
)


class ModelLoaderAndSensorTests(unittest.TestCase):
    def test_sensor_layout_uses_data_addresses_not_sensor_ids(self):
        xml = """
<mujoco>
  <worldbody>
    <body><joint name="j" type="hinge"/><geom type="sphere" size=".1"/>
      <site name="s0"/><site name="s1"/>
    </body>
  </worldbody>
  <sensor>
    <framequat name="orientation" objtype="site" objname="s0"/>
    <touch name="robot0:TS_first" site="s0"/>
    <touch name="robot0:TS_second" site="s1"/>
  </sensor>
</mujoco>
"""
        model = mujoco.MjModel.from_xml_string(xml)
        layout = build_sensor_layout(model)
        self.assertEqual(layout.touch_sensor_ids, (1, 2))
        self.assertEqual(layout.touch_data_indices, (4, 5))
        self.assertEqual(layout.contiguous_touch_span, (4, 6))

    def test_loader_removes_apirate_only_in_memory(self):
        xml = '<mujoco><option timestep="0.002" apirate="200"/><worldbody/></mujoco>'
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.xml"
            path.write_text(xml, encoding="utf-8")
            model, report = load_project_model(path)
            self.assertEqual(model.opt.timestep, 0.002)
            if tuple(int(part) for part in mujoco.__version__.split(".")[:2]) >= (3, 11):
                self.assertIn("removed obsolete option.apirate in memory", report.compatibility_changes)
            else:
                self.assertNotIn("removed obsolete option.apirate in memory", report.compatibility_changes)
            self.assertIn('apirate="200"', path.read_text(encoding="utf-8"))

    @unittest.skipUnless(N500_XML.is_file(), "generated N=500 smoke model is unavailable")
    def test_actual_n500_model_layout_and_rigid_flex(self):
        model, report = load_project_model(N500_XML)
        layout = build_sensor_layout(model)
        self.assertEqual(report.nflex, 1)
        self.assertTrue(report.rigid_flex)
        self.assertTrue(is_single_body_rigid_flex(model))
        self.assertEqual(report.nsensordata, 529)
        self.assertEqual(layout.touch_count, 500)
        self.assertEqual(layout.contiguous_touch_span, (29, 529))


if __name__ == "__main__":
    unittest.main()
