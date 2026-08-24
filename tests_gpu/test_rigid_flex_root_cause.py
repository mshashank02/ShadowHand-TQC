from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from shadowhand_gpu.rigid_flex_root_cause import (
    build_probe_scenarios,
    topology_snapshot,
    validate_guard_model,
)


DEBUG_DIR = Path("generated/rigid_flex_cpu_warp_debug")


class _RigidFlex:
    nflex = 1
    nflexvert = 4
    nflexedge = 6
    nflexelem = 1
    nq = nv = nbody = ngeom = 1
    flex_dim = np.array([3])
    flex_rigid = np.array([True])
    flex_internal = np.array([False])
    flex_selfcollide = np.array([0])
    flex_radius = np.array([0.00125])
    flex_contype = np.array([1])
    flex_conaffinity = np.array([1])
    flex_vert = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    flex_edge = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=int)
    flex_elem = np.array([0, 1, 2, 3], dtype=int)
    flex_shell = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=int)
    flex_vertbodyid = np.array([1, 1, 1, 1], dtype=int)
    flex_vertadr = np.array([0])
    flex_edgeadr = np.array([0])
    flex_edgenum = np.array([6])


def test_guard_accepts_only_compiled_cpu_rigid_exclusion_case() -> None:
    validate_guard_model(_RigidFlex())
    model = _RigidFlex()
    model.flex_internal = np.array([True])
    with pytest.raises(ValueError, match="internal=true"):
        validate_guard_model(model)


def test_probe_scenarios_span_no_contact_onset_penetration_and_sliding() -> None:
    scenarios = build_probe_scenarios(_RigidFlex(), 0.006)
    names = {scenario.name for scenario in scenarios}
    assert {"flex_only_no_probe_contact", "shell_onset", "deep_penetration"} <= names
    assert sum(name.startswith("sliding_") for name in names) == 2
    assert max(s.signed_shell_offset for s in scenarios) > 0
    assert min(s.signed_shell_offset for s in scenarios) < 0


def test_topology_snapshot_is_stable_and_includes_collision_flags() -> None:
    first = topology_snapshot(_RigidFlex())
    second = topology_snapshot(_RigidFlex())
    assert first == second
    assert first["flex"]["rigid"] == [True]
    assert first["flex"]["internal"] == [False]
    assert first["flex"]["selfcollide"] == [0]


@pytest.mark.skipif(not (DEBUG_DIR / "model_comparison.json").is_file(), reason="debug matrix not generated")
class TestRecordedRigidFlexRootCause:
    def test_topology_radius_transfer_and_world_positions(self) -> None:
        report = json.loads((DEBUG_DIR / "model_comparison.json").read_text())
        assert report["all_topology_hashes_equal"]
        assert report["warp_transfer"]["flex_radius"] == pytest.approx([0.00125], abs=1e-10)
        assert report["warp_transfer"]["flex_internal"] == [0]
        assert report["warp_transfer"]["flex_selfcollide"] == [0]
        assert all(report["warp_integer_topology_transfer_matches_cpu"].values())
        assert report["warp_compiled_vertex_transfer_max_abs_error_m"] < 2e-9
        assert report["world_position_max_abs_error_m"]["warp_3.11.0_stock"] < 2e-9

    def test_no_contact_and_first_contact_localize_candidate_generation(self) -> None:
        report = json.loads((DEBUG_DIR / "no_contact_comparison.json").read_text())
        assert report["cpu_3.3.1"]["contact"]["count"] == 0
        assert report["cpu_3.11.0"]["contact"]["count"] == 0
        assert report["warp_3.11.0_stock"]["contact"]["flex_internal_count"] == 2674
        assert report["warp_3.11.0_tet_guard"]["contact"]["count"] == 0
        first = json.loads((DEBUG_DIR / "first_contact_cpu.json").read_text())
        assert first["scenario"] == "penetration_0p1mm"
        assert first["contact"]["count"] == 1

    def test_matching_support_constraint_and_tactile_artifacts(self) -> None:
        with (DEBUG_DIR / "support_mapping_comparison.csv").open(newline="") as stream:
            support = list(csv.DictReader(stream))
        assert support[1]["element_shape"] == "tetrahedron"
        assert support[2]["element_shape"] == "four triangles per tetrahedron"
        with (DEBUG_DIR / "contact_matching.csv").open(newline="") as stream:
            matches = list(csv.DictReader(stream))
        onset = [row for row in matches if row["scenario"] == "penetration_0p1mm"]
        assert len(onset) == 1 and onset[0]["status"] == "matched"
        assert float(onset[0]["distance_abs_error_m"]) < 1e-9
        with (DEBUG_DIR / "tactile_comparison.csv").open(newline="") as stream:
            tactile = list(csv.DictReader(stream))
        shallow = next(
            row
            for row in tactile
            if row["backend"] == "warp_3.11.0_tet_guard" and row["scenario"] == "shallow_penetration"
        )
        assert float(shallow["abs_error_vs_cpu_3.11.0"]) < 3e-6

    def test_version_metadata_and_minimal_reproducer_are_durable(self) -> None:
        with (DEBUG_DIR / "version_matrix.csv").open(newline="") as stream:
            versions = list(csv.DictReader(stream))
        assert {(row["backend"], row["mujoco"]) for row in versions} >= {
            ("CPU MuJoCo", "3.3.1"),
            ("CPU MuJoCo", "3.11.0"),
            ("MuJoCo Warp stock", "3.11.0"),
        }
        reproducer = DEBUG_DIR / "minimal_reproducer"
        assert (reproducer / "rigid_flex_sphere_probe.xml").is_file()
        states = json.loads((reproducer / "probe_states.json").read_text())
        assert states["source_msh_sha256"] == "eaa78c4a15423bf7346f120e1802f0201ee18711834cd703e6a4ef5f98e2b3ef"
