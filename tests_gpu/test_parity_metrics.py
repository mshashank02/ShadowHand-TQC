import unittest

import numpy as np

from shadowhand_gpu.parity import error_metrics, tactile_metrics


class ParityMetricTests(unittest.TestCase):
    def test_error_metrics(self):
        result = error_metrics(np.array([1.0, -2.0]), np.array([1.5, -1.5]))
        self.assertEqual(result.max_abs, 0.5)
        self.assertEqual(result.mean_abs, 0.5)
        self.assertEqual(result.root_mean_square, 0.5)
        self.assertEqual(result.reference_abs_max, 2.0)

    def test_tactile_metrics_include_activation_correlation_and_sensor_mapping(self):
        result = tactile_metrics(
            np.array([0.0, 2.0, 1.0]),
            np.array([0.0, 2.2, 0.0]),
            names=("s0", "s1", "s2"),
            sites=("robot0:palm_0", "robot0:fftip_0", "robot0:thtip_0"),
            top_k=2,
        )
        self.assertAlmostEqual(result["max_absolute_error"], 1.0)
        self.assertEqual(result["active_sensors_cpu"], 2)
        self.assertEqual(result["active_sensors_warp"], 1)
        self.assertEqual(result["active_sensor_intersection"], 1)
        self.assertEqual(result["active_sensor_union"], 2)
        self.assertAlmostEqual(result["active_sensor_jaccard"], 0.5)
        self.assertIsNotNone(result["pearson_correlation"])
        self.assertEqual(result["top_error_sensors"][0]["sensor_name"], "s2")
        self.assertEqual(result["top_error_sensors"][0]["site_name"], "robot0:thtip_0")


if __name__ == "__main__":
    unittest.main()
