from __future__ import annotations

import json
from pathlib import Path

from validate_native_decomposition_312 import _classification


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "generated/native_decomposition_312_validation/validation_results.json"


def test_native_rigid_classification_is_fail_closed() -> None:
    assert _classification({"36": {"parity_passed": True}, "87": {"parity_passed": True}}) == "PASS_BOTH"
    assert _classification({"36": {"parity_passed": True}, "87": {"parity_passed": False}}) == "PASS_36"
    assert _classification({"36": {"parity_passed": False}, "87": {"parity_passed": True}}) == "PASS_87"
    assert (
        _classification({"36": {"parity_passed": False}, "87": {"parity_passed": False}})
        == "FAIL_NEW_RIGID_DEFINITION"
    )


def test_recorded_native_rigid_decision_and_provenance() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert payload["conclusion"] == "PASS_36"
    assert payload["historical_rigid_flex_results_pooled"] is False
    assert payload["production_default_changed"] is False
    assert payload["full_rl_training_run"] is False
    assert payload[
        "passing_representation_requires_retraining_all_24_objects_and_sensor_configurations"
    ] is True
    assert payload["representations"]["36"]["parity_passed"] is True
    assert payload["representations"]["87"]["parity_passed"] is False


def test_recorded_model_and_primary_gates() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    for name, pieces in (("36", 36), ("87", 87)):
        result = payload["representations"][name]
        audit = result["model_audit"]
        assert audit["piece_count"] == pieces
        assert audit["nflex"] == 0
        assert audit["passed"] is True
        assert all(audit["checks"].values())
        assert result["fixtures"]["passed"] is True
        assert result["fixtures"]["fixtures"]["separated_contact"]["passed"] is True
        assert result["n500_one_step"]["passed"] is True
        assert result["n500_one_step"]["contact_geom_pairs_match"] is True
        assert max(result["n500_one_step"]["gpu_overflow_flags"]) == 0
    assert payload["representations"]["36"]["policy_action_20_substeps"]["passed"] is True
    failed = payload["representations"]["87"]["policy_action_20_substeps"]
    assert failed["passed"] is False
    assert failed["gate_results"]["qpos"] is False
    assert failed["gate_results"]["qvel"] is False


def test_benchmarks_only_follow_a_passing_parity_gate() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert payload["benchmarks"]
    for benchmark in payload["benchmarks"]:
        representation = benchmark["representation"]
        assert payload["representations"][representation]["parity_passed"] is True
        assert benchmark["overflow_flags_max"] == 0
