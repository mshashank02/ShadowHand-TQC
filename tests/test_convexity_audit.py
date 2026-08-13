import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from object_conversion.audit_convexity import (
    audit_arrays,
    discover_study_objects,
    gap_statistics,
    hull_gaps,
    sample_surface,
    write_csv,
)


def cube_mesh():
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=int)
    return vertices, faces


def indented_cube_mesh():
    vertices, _ = cube_mesh()
    vertices = np.vstack((vertices, [0.5, 0.5, 0.5]))
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        [4, 5, 8], [5, 6, 8], [6, 7, 8], [7, 4, 8],
    ], dtype=int)
    return vertices, faces


class ConvexityAuditTests(unittest.TestCase):
    def test_exact_24_object_discovery_and_metadata(self):
        root = Path(__file__).resolve().parents[1]
        rows = discover_study_objects(root / "study_objects/sphere_study_v1/manifest.csv")
        self.assertEqual(len(rows), 24)
        self.assertEqual({row["size"] for row in rows}, {"small", "medium", "large"})
        self.assertEqual({row["roughness"] for row in rows}, {"low", "high"})

    def test_discovery_fails_closed_for_non_24_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.csv"
            path.write_text("object_id,msh_file,size,aspect_ratio,macro,roughness\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 24"):
                discover_study_objects(path)

    def test_convex_cube_control_has_zero_gap_and_inflation(self):
        vertices, faces = cube_mesh()
        metrics, arrays, _ = audit_arrays(vertices, faces, samples=20_000)
        self.assertLess(metrics["volume_inflation_fraction"], 1e-12)
        self.assertLess(metrics["max_gap_m"], 1e-12)
        self.assertLess(float(arrays["vertex_gaps"].max()), 1e-12)
        self.assertTrue(metrics["watertight"])
        self.assertTrue(metrics["winding_consistent"])

    def test_concave_control_finds_indentation_and_volume_inflation(self):
        vertices, faces = indented_cube_mesh()
        metrics, arrays, hull = audit_arrays(vertices, faces, samples=80_000)
        self.assertAlmostEqual(metrics["mesh_volume_m3"], 5.0 / 6.0, places=12)
        self.assertAlmostEqual(metrics["hull_volume_m3"], 1.0, places=12)
        self.assertAlmostEqual(metrics["volume_inflation_fraction"], 0.2, places=12)
        center_gap = hull_gaps(vertices[[8]], hull)[0]
        self.assertAlmostEqual(center_gap, 0.5, places=12)
        self.assertEqual(metrics["vertex_max_gap_index"], 8)
        self.assertTrue(np.allclose(metrics["vertex_max_gap_xyz_m"], [0.5, 0.5, 0.5]))
        sampled_max = arrays["sampled_points"][np.argmax(arrays["sampled_gaps"])]
        self.assertLess(np.linalg.norm(sampled_max - [0.5, 0.5, 0.5]), 0.03)

    def test_sampling_is_deterministic_and_area_weighted(self):
        vertices, faces = cube_mesh()
        first = sample_surface(vertices, faces, 5000, seed=17)
        second = sample_surface(vertices, faces, 5000, seed=17)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))

    def test_convex_hull_and_metrics_are_deterministic(self):
        vertices, faces = indented_cube_mesh()
        first, first_arrays, first_hull = audit_arrays(vertices, faces, samples=5000, seed=23)
        second, second_arrays, second_hull = audit_arrays(vertices, faces, samples=5000, seed=23)
        self.assertEqual(first, second)
        self.assertTrue(np.array_equal(first_hull.simplices, second_hull.simplices))
        self.assertTrue(np.array_equal(first_arrays["sampled_gaps"], second_arrays["sampled_gaps"]))

    def test_normalized_gaps_and_threshold_fractions(self):
        gaps = np.array([0, 0.00002, 0.0002, 0.002])
        stats = gap_statistics(gaps, diagonal_m=0.02)
        self.assertAlmostEqual(stats["mean_gap_normalized"], gaps.mean() / 0.02)
        self.assertEqual(stats["surface_fraction_gt_0p01mm"], 0.75)
        self.assertEqual(stats["surface_fraction_gt_0p10mm"], 0.5)
        self.assertEqual(stats["surface_fraction_gt_1p00mm"], 0.25)

    def test_csv_and_json_serializable_output(self):
        vertices, faces = cube_mesh()
        metrics, _, _ = audit_arrays(vertices, faces, samples=1000)
        row = {**metrics, "object_id": "cube", "rank": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "audit.csv"
            write_csv([row], path)
            with path.open(encoding="utf-8") as stream:
                loaded = list(csv.DictReader(stream))
            self.assertEqual(loaded[0]["object_id"], "cube")
            json.dumps({"objects": [row]})


if __name__ == "__main__":
    unittest.main()
