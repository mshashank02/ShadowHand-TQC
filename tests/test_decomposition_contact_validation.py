import unittest

import numpy as np

from object_conversion.validate_decomposition_contacts import (
    _find_onset,
    compare_contacts,
    compare_tactile,
)


class DecompositionContactValidationTests(unittest.TestCase):
    def test_onset_bisection_finds_descending_contact_boundary(self):
        def evaluate(coordinate):
            return {"contacts": [{}] if coordinate <= 0.125 else []}

        onset, result = _find_onset(evaluate, 0.25, 0.0)
        self.assertAlmostEqual(onset, 0.125, places=8)
        self.assertTrue(result["contacts"])

    def test_contact_matching_ignores_normal_sign_and_order(self):
        reference = [
            {
                "position_m": [0.0, 0.0, 0.0],
                "normal": [1.0, 0.0, 0.0],
                "distance_m": -0.001,
                "normal_force": 2.0,
            },
            {
                "position_m": [0.01, 0.0, 0.0],
                "normal": [0.0, 1.0, 0.0],
                "distance_m": -0.002,
                "normal_force": 3.0,
            },
        ]
        actual = [
            {
                "position_m": [0.01, 0.0, 0.0],
                "normal": [0.0, -1.0, 0.0],
                "distance_m": -0.002,
                "normal_force": 3.0,
            },
            {
                "position_m": [0.0, 0.0, 0.0],
                "normal": [-1.0, 0.0, 0.0],
                "distance_m": -0.001,
                "normal_force": 2.0,
            },
        ]
        metrics = compare_contacts(reference, actual)
        self.assertEqual(metrics["paired_count"], 2)
        self.assertEqual(metrics["position_error_mm_max"], 0.0)
        self.assertEqual(metrics["normal_angle_error_deg_max"], 0.0)
        self.assertEqual(metrics["total_normal_force_relative_error"], 0.0)

    def test_tactile_metrics_preserve_sensor_mapping(self):
        records = [
            {"sensor_name": "a", "site_name": "site_a", "region": "palm", "value": 1.0},
            {"sensor_name": "b", "site_name": "site_b", "region": "fftip", "value": 0.0},
        ]
        metrics = compare_tactile(
            np.asarray([1.0, 0.0]),
            np.asarray([0.5, 2.0]),
            records,
            records,
        )
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.125))
        self.assertEqual(metrics["active_intersection"], 1)
        self.assertEqual(metrics["active_union"], 2)
        self.assertEqual(metrics["top_error_sensors"][0]["region"], "fftip")


if __name__ == "__main__":
    unittest.main()
