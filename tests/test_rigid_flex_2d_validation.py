from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from shadowhand_gpu.parity import tactile_metrics
from shadowhand_gpu.rigid_flex_2d_validation import _surface_probe_scenarios


VALIDATION_DIR = Path("generated/rigid_flex_2d_validation")
EXPECTED_FIXTURES = {
    "isolated_fingertip",
    "isolated_palm",
    "deepest_concavity",
    "macro_feature",
    "roughness_feature",
}


class _RigidSurfaceFlex:
    flex_vert = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    flex_elem = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int64)
    flex_radius = np.asarray([0.00125], dtype=np.float64)


def test_surface_probe_scenarios_cover_separation_onset_and_deep_contact() -> None:
    scenarios = _surface_probe_scenarios(_RigidSurfaceFlex(), 0.006)
    by_name = {scenario.name: scenario for scenario in scenarios}
    assert by_name["flex_only_no_probe_contact"].signed_shell_offset == pytest.approx(0.020)
    assert by_name["shell_onset"].signed_shell_offset == 0.0
    assert by_name["shallow_penetration"].signed_shell_offset == pytest.approx(-0.0005)
    assert by_name["deep_penetration"].signed_shell_offset == pytest.approx(-0.002)


def test_tactile_metrics_reports_cosine_similarity() -> None:
    metrics = tactile_metrics(np.asarray([1.0, 2.0]), np.asarray([2.0, 4.0]))
    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert tactile_metrics(np.zeros(2), np.zeros(2))["cosine_similarity"] == 1.0
    assert tactile_metrics(np.zeros(2), np.ones(2))["cosine_similarity"] == 0.0


@pytest.mark.skipif(
    not (VALIDATION_DIR / "cpu_3d_vs_2d_results.json").is_file(),
    reason="durable 2D rigid-flex validation artifacts are absent",
)
class TestRecordedRigidFlex2DValidation:
    def test_exact_exterior_obj_is_deterministic(self) -> None:
        payload = json.loads(
            (VALIDATION_DIR / "cpu_3d_vs_2d_results.json").read_text(encoding="utf-8")
        )
        obj = Path(payload["geometry"]["exterior_obj"])
        assert hashlib.sha256(obj.read_bytes()).hexdigest() == (
            "e8fd23e33daf4bd61f71ee31d9e3ae8cdf5bd574e411fc608f5e9094def0d70c"
        )
        assert payload["geometry"]["vertex_count"] == 1666
        assert payload["geometry"]["triangle_count"] == 3328
        assert payload["geometry"]["watertight"]

    def test_compiled_surface_preserves_rigid_body_and_contact_contract(self) -> None:
        payload = json.loads(
            (VALIDATION_DIR / "cpu_3d_vs_2d_model.json").read_text(encoding="utf-8")
        )
        audits = payload["model_audits"]
        reference = audits["original_3d_gmsh_flex"]
        surface = audits["surface_2d_obj_flex"]
        assert surface["flex_dim"] == [2]
        assert surface["flex_rigid"] == [1]
        assert surface["flex_radius"] == pytest.approx([0.00125])
        assert surface["nflexvert"] == 1666
        assert surface["nflexelem"] == 3328
        assert surface["object_free_joint_count"] == 1
        assert surface["object_global_dof_count"] == 6
        assert surface["all_flex_vertices_on_object_body"]
        for field in (
            "nq",
            "nv",
            "object_mass",
            "object_inertial_pos",
            "object_inertia",
            "object_dof_damping",
            "flex_contact",
        ):
            assert surface[field] == reference[field]
        assert all(payload["model_gate_results"].values())

    def test_no_contact_cpu_motion_and_five_fixture_cpu_gate(self) -> None:
        payload = json.loads(
            (VALIDATION_DIR / "cpu_3d_vs_2d_results.json").read_text(encoding="utf-8")
        )
        assert payload["no_contact"]["passed"]
        assert {row["steps"] for row in payload["no_contact"]["checkpoints"]} == {1, 10, 100}
        assert set(payload["fixtures"]) == EXPECTED_FIXTURES
        assert payload["decision"]["classification"].startswith("CPU Outcome A")
        assert payload["decision"]["all_primary_fixture_gates_pass"]
        for fixture in payload["fixtures"].values():
            assert all(fixture["primary_gate_results"].values())

    @pytest.mark.skipif(
        not (VALIDATION_DIR / "warp_decision.json").is_file(),
        reason="revised CPU/Warp decision artifact is absent",
    )
    def test_warp_avoids_tetrahedral_contacts_but_fails_revised_gates(self) -> None:
        decision = json.loads(
            (VALIDATION_DIR / "warp_decision.json").read_text(encoding="utf-8")
        )
        assert decision["zero_contact_20mm_pass"]
        assert decision["classification"] == "Outcome C — CPU 2D works, Warp still differs"
        assert decision["shallow"]["cpu_3_11_0_contacts_rows"] == [3, 12]
        assert decision["shallow"]["warp_contacts_rows"] == [3, 12]
        assert abs(
            decision["shallow"]["cpu_3_11_0_touch"]
            - decision["shallow"]["warp_touch"]
        ) < 3e-6
        assert decision["deep"]["cpu_3_11_0_contacts_rows"] == [13, 52]
        assert decision["deep"]["warp_contacts_rows"] == [10, 40]
        assert decision["revised_reference"] == "CPU MuJoCo 3.11.0"
        assert decision["revised_total_tactile_relative_error_tolerance"] == 0.065
        assert decision["five_fixture_warp_run"]
        assert not decision["five_fixture_native_all_pass"]
        assert not decision["five_fixture_compatible_all_pass"]
        assert decision["n500_run"]
        assert not decision["native_full_n500_supported"]
        n500 = decision["n500_seeded_settled_state"]
        assert n500["initial_contacts_cpu_warp"] == [101, 101]
        assert n500["final_contacts_cpu_warp"] == [101, 55]
        assert n500["final_constraint_rows_cpu_warp"] == [431, 247]
        assert n500["tactile"]["total_magnitude_relative_error"] == pytest.approx(
            0.3949663, rel=1e-5
        )
        assert n500["tactile"]["active_sensor_jaccard"] == pytest.approx(0.9)
        assert not n500["pass"]
        assert not decision["performance_run"]
        assert not decision["rl_run"]

    @pytest.mark.skipif(
        not all(
            (VALIDATION_DIR / filename).is_file()
            for filename in (
                "cpu_warp_five_fixtures.json",
                "cpu_warp_five_fixtures_compat.json",
            )
        ),
        reason="exact CPU/Warp five-fixture artifacts are absent",
    )
    def test_exact_five_fixture_cpu_warp_artifacts_fail_6p5_percent(self) -> None:
        for filename, reference_compat in (
            ("cpu_warp_five_fixtures.json", False),
            ("cpu_warp_five_fixtures_compat.json", True),
        ):
            payload = json.loads((VALIDATION_DIR / filename).read_text(encoding="utf-8"))
            assert payload["mujoco_version"] == "3.11.0"
            assert payload["mujoco_warp_version"] == "3.11.0"
            assert payload["reference_compat"] is reference_compat
            assert set(payload["fixtures"]) == EXPECTED_FIXTURES
            for fixture in payload["fixtures"].values():
                assert (
                    fixture["common_pose"]["comparison"]["tactile"][
                        "total_magnitude_relative_error"
                    ]
                    > 0.065
                )
