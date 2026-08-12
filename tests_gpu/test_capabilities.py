import unittest

from shadowhand_gpu.capabilities import collect_capabilities


class CapabilityTests(unittest.TestCase):
    def test_report_is_safe_without_optional_mujoco_warp(self):
        report = collect_capabilities()
        self.assertIn("versions", report)
        self.assertIn("direct_backend_importable", report)
        self.assertIn("torch_cuda_available", report)


if __name__ == "__main__":
    unittest.main()
