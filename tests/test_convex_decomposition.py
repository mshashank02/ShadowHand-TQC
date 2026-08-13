import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial import ConvexHull

from object_conversion.audit_convexity import hull_gaps, sample_surface
from object_conversion.audit_decomposition import audit_piece_union, boolean_union_mesh
from object_conversion.convex_decomposition import (
    CoACDParameters,
    ConvexPiece,
    decomposition_cache_key,
    decompose_surface_cached,
    load_decomposition_pieces,
    validate_convex_piece,
)


def box_piece(low=(0.0, 0.0, 0.0), high=(1.0, 1.0, 1.0)) -> ConvexPiece:
    x0, y0, z0 = low
    x1, y1, z1 = high
    vertices = np.asarray([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    faces = np.asarray([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return ConvexPiece(vertices, faces)


def dependencies_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("manifold3d", "trimesh", "rtree"))


class ConvexDecompositionTests(unittest.TestCase):
    def test_cache_key_is_deterministic_and_parameter_sensitive(self):
        parameters = CoACDParameters(0.001)
        arguments = dict(
            source_hash="a" * 64,
            exterior_hash="b" * 64,
            scale=(0.03125, 0.03125, 0.03125),
            parameters=parameters,
            coacd_version="1.0.11",
        )
        self.assertEqual(decomposition_cache_key(**arguments), decomposition_cache_key(**arguments))
        changed = dict(arguments)
        changed["parameters"] = CoACDParameters(0.0005)
        self.assertNotEqual(decomposition_cache_key(**arguments), decomposition_cache_key(**changed))
        # Sensor count/allocation are intentionally absent from the cache-key API.
        self.assertNotIn("sensor", parameters.coacd_kwargs())

    def test_convex_validation_accepts_box_and_rejects_concave_shell(self):
        metrics = validate_convex_piece(box_piece())
        self.assertTrue(metrics["convex"])
        self.assertTrue(metrics["watertight"])
        box = box_piece()
        vertices = np.vstack((box.vertices_m, [0.5, 0.5, 0.5]))
        faces = np.asarray([
            [0, 2, 1], [0, 3, 2], [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7], [4, 5, 8], [5, 6, 8],
            [6, 7, 8], [7, 4, 8],
        ], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "non-convex"):
            validate_convex_piece(ConvexPiece(vertices, faces))

    @unittest.skipUnless(dependencies_available(), "decomposition audit extras are unavailable")
    def test_boolean_union_removes_overlap_instead_of_summing_piece_volumes(self):
        pieces = (box_piece(), box_piece((0.5, 0.0, 0.0), (1.5, 1.0, 1.0)))
        vertices, faces, volume = boolean_union_mesh(pieces)
        self.assertAlmostEqual(volume, 1.5, places=12)
        self.assertLess(volume, sum(validate_convex_piece(piece)["mesh_volume_m3"] for piece in pieces))
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(faces), 0)

    @unittest.skipUnless(dependencies_available(), "decomposition audit extras are unavailable")
    def test_exact_concave_union_has_zero_error_and_beats_single_hull(self):
        pieces = (
            box_piece((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            box_piece((0.0, 1.0, 0.0), (1.0, 2.0, 1.0)),
        )
        source_vertices, source_faces, _ = boolean_union_mesh(pieces)
        metrics, _ = audit_piece_union(source_vertices, source_faces, pieces, samples=20_000, seed=17)
        points, _ = sample_surface(source_vertices, source_faces, 20_000, seed=17)
        single_hull_error = hull_gaps(points, ConvexHull(source_vertices))
        self.assertLess(metrics["max_gap_m"], 1e-12)
        self.assertLess(abs(metrics["volume_error_percent"]), 1e-10)
        self.assertGreater(float(single_hull_error.max()), 0.4)

    @unittest.skipUnless(importlib.util.find_spec("coacd") is not None, "CoACD is unavailable")
    def test_real_coacd_cache_reuse_and_manifest_integrity(self):
        piece = box_piece()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.msh"
            exterior = root / "surface.obj"
            source.write_bytes(b"synthetic source identity")
            exterior.write_bytes(b"synthetic exterior identity")
            kwargs = dict(
                source_path=source,
                exterior_path=exterior,
                vertices_source_units=piece.vertices_m,
                faces=piece.faces,
                scale_m_per_source_unit=(1.0, 1.0, 1.0),
                cache_root=root / "cache",
                parameters=CoACDParameters(
                    0.01, resolution=100, mcts_nodes=4, mcts_iterations=10,
                    mcts_max_depth=2, max_ch_vertex=64,
                ),
            )
            first = decompose_surface_cached(**kwargs)
            second = decompose_surface_cached(**kwargs)
            self.assertFalse(first.cache_reused)
            self.assertTrue(second.cache_reused)
            self.assertEqual(first.cache_key, second.cache_key)
            self.assertEqual(first.manifest, second.manifest)
            self.assertTrue(first.manifest["all_pieces_convex"])
            self.assertTrue(all(path.is_file() for path in first.piece_paths))
            self.assertEqual(len(load_decomposition_pieces(second)), first.manifest["piece_count"])


if __name__ == "__main__":
    unittest.main()
