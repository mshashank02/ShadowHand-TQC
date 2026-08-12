import importlib.util
import os
from pathlib import Path
import unittest


N500_XML = Path(
    "generated/smoke_tests/rigid_n0500_a0p3_b0p6_large_high_high_high/"
    "custom_obj_size_large_ar_high_macro_high_rough_high_500_1.071429_0.714286/"
    "manipulate_custom_obj_size_large_ar_high_macro_high_rough_high_touch_sensors_"
    "500_1.071429_0.714286.xml"
)
NATIVE_RIGID_XML = Path(__file__).parent / "fixtures" / "native_rigid_touch.xml"
RUN_MJW = (
    os.environ.get("SHADOWHAND_RUN_MJW_TESTS") == "1"
    and importlib.util.find_spec("mujoco_warp") is not None
)


@unittest.skipUnless(RUN_MJW, "set SHADOWHAND_RUN_MJW_TESTS=1 in the isolated GPU environment")
class WarpBackendTests(unittest.TestCase):
    def test_native_rigid_model_steps_and_exposes_touch_cuda_view(self):
        from shadowhand_gpu.warp_backend import MujocoWarpBackend

        backend = MujocoWarpBackend(NATIVE_RIGID_XML, worlds=1)
        backend.step()
        backend.synchronize()
        self.assertEqual(tuple(backend.qpos.shape), (1, 7))
        self.assertEqual(tuple(backend.sensordata.shape), (1, 1))
        self.assertEqual(tuple(backend.touch.shape), (1, 1))
        self.assertTrue(backend.touch.is_cuda)
        self.assertLessEqual(int(backend.active_contact_counts.sum()), 8192)
        self.assertEqual(backend.report()["model_support"], "gpu_rigid_supported")

    def test_state_transfer_broadcasts_without_host_round_trip(self):
        import numpy as np

        from shadowhand_gpu.warp_backend import MujocoWarpBackend

        backend = MujocoWarpBackend(NATIVE_RIGID_XML, worlds=2)
        qpos = np.asarray(backend.model.qpos0).copy()
        qvel = np.zeros(backend.model.nv)
        ctrl = np.zeros(backend.model.nu)
        backend.set_state(qpos=qpos, qvel=qvel, ctrl=ctrl, time=0.25)
        backend.synchronize()
        self.assertTrue(backend.qpos.is_cuda)
        self.assertTrue(backend.qvel.is_cuda)
        self.assertEqual(tuple(backend.qpos.shape), (2, 7))
        self.assertAlmostEqual(float(backend.time[1].cpu()), 0.25, places=6)

    @unittest.skipUnless(N500_XML.is_file(), "real generated N=500 fixture is absent")
    def test_flex_model_fails_closed(self):
        from shadowhand_gpu.warp_backend import MujocoWarpBackend

        with self.assertRaisesRegex(NotImplementedError, "flex collision.*nflex=1"):
            MujocoWarpBackend(N500_XML, worlds=1)


if __name__ == "__main__":
    unittest.main()
