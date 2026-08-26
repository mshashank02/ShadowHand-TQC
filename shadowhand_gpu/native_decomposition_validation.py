"""MuJoCo 3.12 CPU/Warp validation for native-rigid CoACD models.

This module intentionally treats each bare convex decomposition as a new
collision definition.  It never compares against, or imports state from, a flex
representation.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
import xml.etree.ElementTree as ET

import numpy as np

from object_conversion.validate_decomposition_contacts import (
    EVALUATION_PENETRATION_M,
    FEATURE_NAMES,
    HAND_FIXTURES,
    PROBE_GEOM_NAME,
    _configure_collision_geoms,
    _contact_records,
    _find_onset,
    _temporary_probe_model,
    _touch_vector,
    _zero_action_control,
    compare_contacts,
    compare_tactile,
)
from .parity import (
    _contact_snapshot,
    _object_qpos_address,
    _touch_site_names,
    compare_one_step,
    contact_geom_pair_counts,
    error_metrics,
    tactile_metrics,
)
FIXTURE_GATES = {
    "onset_difference_mm_max": 0.10,
    "total_touch_relative_error_max": 0.065,
    "active_sensor_jaccard_min": 0.9,
}
N500_GATES = {
    "qpos_max_abs": 1e-5,
    "qvel_max_abs": 1e-3,
    "touch_max_abs": 0.05,
    "total_touch_relative_error_max": 0.02,
    "contact_geom_pairs_match": True,
    "overflow_flags_max": 0,
}
POLICY_ACTION_GATES = {
    "qpos_max_abs": 3e-4,
    "qvel_max_abs": 0.02,
    "touch_max_abs": 0.2,
    "reward_exact": True,
    "success_exact": True,
    "overflow_flags_max": 0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mujoco_312_xml(source: str | Path) -> Path:
    """Create a same-directory 3.12 schema copy, removing only ``apirate``."""
    source = Path(source).resolve()
    tree = ET.parse(source)
    option = tree.getroot().find("option")
    if option is None or "apirate" not in option.attrib:
        return source
    option.attrib.pop("apirate")
    output = source.with_name(f"{source.stem}_mujoco312{source.suffix}")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def _apply_shared_collision_options(mujoco: Any, model: Any) -> list[str]:
    """Disable only CCD modes unsupported by Warp for the unchanged hand model."""
    changes: list[str] = []
    for name in ("mjDSBL_MULTICCD", "mjDSBL_NATIVECCD"):
        bit = getattr(mujoco.mjtDisableBit, name, None)
        if bit is not None and not (int(model.opt.disableflags) & int(bit)):
            model.opt.disableflags |= int(bit)
            changes.append(f"disabled {name.removeprefix('mjDSBL_')}")
    return changes


def _object_geom_ids(mujoco: Any, model: Any) -> set[int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    return {
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) == body_id
        and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
            "object_collision_"
        )
    }


def audit_native_model(
    mujoco: Any,
    model: Any,
    xml_path: Path,
    manifest_path: Path,
    expected_pieces: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
    visual_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_visual")
    collision_ids = sorted(_object_geom_ids(mujoco, model))
    qpos_address = int(model.jnt_qposadr[object_joint])
    qvel_address = int(model.jnt_dofadr[object_joint])
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)

    contact_records = []
    for geom_id in collision_ids:
        contact_records.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
                "body_id": int(model.geom_bodyid[geom_id]),
                "type": int(model.geom_type[geom_id]),
                "friction": np.asarray(model.geom_friction[geom_id]).tolist(),
                "condim": int(model.geom_condim[geom_id]),
                "solref": np.asarray(model.geom_solref[geom_id]).tolist(),
                "solimp": np.asarray(model.geom_solimp[geom_id]).tolist(),
                "margin": float(model.geom_margin[geom_id]),
                "gap": float(model.geom_gap[geom_id]),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "priority": int(model.geom_priority[geom_id]),
            }
        )

    expected = manifest["contact_parameters"]
    expected_friction = [float(value) for value in expected["friction"].split()]
    expected_solref = [float(value) for value in expected["solref"].split()]
    expected_solimp = [float(value) for value in expected["solimp"].split()]
    xml_tags = {element.tag for element in ET.parse(xml_path).getroot().iter()}
    piece_paths = [Path(path) for path in manifest["decomposition"]["generated_piece_paths"]]
    piece_hashes = manifest["decomposition"]["piece_hashes"]
    cached_assets_match = len(piece_paths) == len(piece_hashes) and all(
        path.is_file() and _sha256(path) == expected_hash
        for path, expected_hash in zip(piece_paths, piece_hashes)
    )
    checks = {
        "no_flex_or_flexcomp": int(model.nflex) == 0
        and "flex" not in xml_tags
        and "flexcomp" not in xml_tags,
        "piece_count": len(collision_ids) == expected_pieces,
        "all_piece_geoms_are_meshes": all(
            int(model.geom_type[geom_id]) == mesh_type for geom_id in collision_ids
        ),
        "all_pieces_on_object_body": all(
            int(model.geom_bodyid[geom_id]) == object_body for geom_id in collision_ids
        ),
        "one_free_joint_six_dofs": int(model.jnt_type[object_joint])
        == int(mujoco.mjtJoint.mjJNT_FREE),
        "visual_collision_disabled": int(model.geom_contype[visual_geom]) == 0
        and int(model.geom_conaffinity[visual_geom]) == 0,
        "original_obj_is_visual_only": all(
            int(model.geom_dataid[geom_id]) != int(model.geom_dataid[visual_geom])
            for geom_id in collision_ids
        ),
        "bare_surface_zero_margin_gap": all(
            float(model.geom_margin[geom_id]) == 0.0
            and float(model.geom_gap[geom_id]) == 0.0
            for geom_id in collision_ids
        ),
        "contact_contract": all(
            np.allclose(record["friction"], expected_friction)
            and record["condim"] == int(expected["condim"])
            and np.allclose(record["solref"], expected_solref)
            and np.allclose(record["solimp"], expected_solimp)
            and record["contype"] == int(expected["contype"])
            and record["conaffinity"] == int(expected["conaffinity"])
            and record["priority"] == int(expected["priority"])
            for record in contact_records
        ),
        "mass_preserved": bool(
            np.isclose(model.body_mass[object_body], float(manifest["mass"]))
        ),
        "inertia_preserved": np.allclose(
            model.body_inertia[object_body], np.asarray(manifest["diaginertia"], dtype=float)
        ),
        "inertial_position_preserved": np.allclose(
            model.body_ipos[object_body], np.asarray(manifest["inertial_position"], dtype=float)
        ),
        "body_pose_preserved": np.allclose(
            model.body_pos[object_body], np.asarray(manifest["body_position"], dtype=float)
        ),
        "free_joint_name_preserved": manifest["free_joint"] == "object:joint",
        "manifest_same_body": manifest["decomposition"]["all_pieces_same_body"]
        == "object"
        and manifest["decomposition"]["independent_dynamic_bodies"] == 0,
        "cached_piece_count": manifest["decomposition"]["piece_count"]
        == expected_pieces,
        "cache_reused": bool(manifest["decomposition"]["cache_reused"]),
        "cached_assets_match_manifest_hashes": cached_assets_match,
    }
    return {
        "xml": str(xml_path),
        "xml_sha256": _sha256(xml_path),
        "manifest": str(manifest_path),
        "source_hash": manifest["source_hash"],
        "converted_visual_hash": manifest["converted_hash"],
        "decomposition_cache_key": manifest["decomposition"]["cache_key"],
        "piece_hashes": piece_hashes,
        "piece_count": len(collision_ids),
        "nflex": int(model.nflex),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "object_body_id": int(object_body),
        "free_joint": "object:joint",
        "initial_pose": np.asarray(model.qpos0[qpos_address : qpos_address + 7]).tolist(),
        "free_joint_damping": np.asarray(
            model.dof_damping[qvel_address : qvel_address + 6]
        ).tolist(),
        "mass": float(model.body_mass[object_body]),
        "inertial_position": np.asarray(model.body_ipos[object_body]).tolist(),
        "diagonal_inertia": np.asarray(model.body_inertia[object_body]).tolist(),
        "visual_geom": {
            "name": "object_visual",
            "contype": int(model.geom_contype[visual_geom]),
            "conaffinity": int(model.geom_conaffinity[visual_geom]),
            "mesh": manifest["original_visual"]["mesh"],
        },
        "collision_geoms": contact_records,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _warp_bindings() -> tuple[Any, Any]:
    import mujoco_warp as mjw
    import warp as wp

    return mjw, wp


def _warp_forward_host(
    mujoco: Any,
    mjw: Any,
    wp: Any,
    model: Any,
    warp_model: Any,
    seed: Any,
    *,
    contacts_per_world: int = 2048,
    constraints_per_world: int = 4096,
) -> tuple[Any, dict[str, Any]]:
    warp_data = mjw.put_data(
        model,
        seed,
        nworld=1,
        nconmax=contacts_per_world,
        njmax=constraints_per_world,
    )
    mjw.forward(warp_model, warp_data)
    wp.synchronize()
    host = mujoco.MjData(model)
    mjw.get_data_into(host, model, warp_data, world_id=0)
    capacity = {
        "active_contacts": int(np.asarray(warp_data.nacon.numpy()).reshape(-1)[0]),
        "constraints": int(np.asarray(warp_data.nefc.numpy()).reshape(-1)[0]),
        "overflow_flags": int(np.asarray(warp_data.overflow.numpy()).reshape(-1)[0]),
        "contact_capacity": contacts_per_world,
        "constraint_capacity": constraints_per_world,
    }
    return host, capacity


def _set_static_object(
    mujoco: Any, model: Any, data: Any, position: tuple[float, float, float] | np.ndarray
) -> None:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
    qadr = int(model.jnt_qposadr[joint])
    data.qpos[qadr : qadr + 3] = position
    data.qpos[qadr + 3 : qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[:] = 0.0
    data.qacc_warmstart[:] = 0.0


def _hand_seed(
    mujoco: Any, model: Any, fixture: dict[str, Any], coordinate: float
) -> Any:
    data = mujoco.MjData(model)
    data.ctrl[:] = _zero_action_control(model)
    _set_static_object(mujoco, model, data, (*fixture["object_xy_m"], coordinate))
    mujoco.mj_forward(model, data)
    return data


def _feature_seed(
    mujoco: Any,
    model: Any,
    feature: dict[str, Any],
    probe_id: int,
    coordinate: float,
) -> Any:
    data = mujoco.MjData(model)
    data.ctrl[:] = _zero_action_control(model)
    object_position = np.asarray((0.0, 0.0, 0.5), dtype=np.float64)
    _set_static_object(mujoco, model, data, object_position)
    point = np.asarray(feature["point_m"], dtype=np.float64)
    normal = np.asarray(feature["outward_normal"], dtype=np.float64)
    body_id = int(model.geom_bodyid[probe_id])
    mocap_id = int(model.body_mocapid[body_id])
    data.mocap_pos[mocap_id] = object_position + point + normal * (0.003 + coordinate)
    data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    return data


def _state_from_data(
    mujoco: Any,
    model: Any,
    data: Any,
    relevant: Callable[[Any], bool],
    *,
    diagnostic_probe: bool,
    coordinate: float,
    capacity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contacts = _contact_records(mujoco, model, data, relevant)
    values, records = _touch_vector(
        mujoco, model, data, diagnostic_probe=diagnostic_probe
    )
    return {
        "coordinate_m": float(coordinate),
        "contacts": contacts,
        "constraint_rows": int(sum(item["constraint_rows"] for item in contacts)),
        "total_model_contacts": int(data.ncon),
        "total_model_constraint_rows": int(data.nefc),
        "tactile_values": values.tolist(),
        "tactile_records": records,
        "tactile": {
            "sensor_count": int(values.size),
            "active_count": int(np.count_nonzero(np.abs(values) > 1e-8)),
            "maximum": float(np.max(np.abs(values), initial=0.0)),
            "total_magnitude": float(np.sum(np.abs(values))),
        },
        "capacity": capacity,
    }


def _geom_pair_counts(contacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        tuple(sorted(str(name) for name in item["geom_names"] if name is not None))
        for item in contacts
    )
    return [
        {"geom_names": list(pair), "count": int(count)}
        for pair, count in sorted(counts.items())
    ]


def _compare_states(cpu: dict[str, Any], warp: dict[str, Any]) -> dict[str, Any]:
    contact = compare_contacts(cpu["contacts"], warp["contacts"])
    tactile = compare_tactile(
        np.asarray(cpu["tactile_values"], dtype=np.float64),
        np.asarray(warp["tactile_values"], dtype=np.float64),
        cpu["tactile_records"],
        warp["tactile_records"],
    )
    cpu_pairs = _geom_pair_counts(cpu["contacts"])
    warp_pairs = _geom_pair_counts(warp["contacts"])
    return {
        "contacts": contact,
        "cpu_geom_pairs": cpu_pairs,
        "warp_geom_pairs": warp_pairs,
        "geom_pair_multisets_match": cpu_pairs == warp_pairs,
        "tactile": tactile,
    }


def _evaluate_separated(
    mujoco: Any, mjw: Any, wp: Any, model: Any, warp_model: Any
) -> dict[str, Any]:
    object_ids = _object_geom_ids(mujoco, model)

    def relevant(contact: Any) -> bool:
        return any(int(value) in object_ids for value in np.asarray(contact.geom))

    seed = mujoco.MjData(model)
    seed.ctrl[:] = _zero_action_control(model)
    _set_static_object(mujoco, model, seed, (1.0, 0.87, 2.0))
    mujoco.mj_forward(model, seed)
    cpu = _state_from_data(
        mujoco, model, seed, relevant, diagnostic_probe=False, coordinate=2.0
    )
    warp_host, capacity = _warp_forward_host(mujoco, mjw, wp, model, warp_model, seed)
    warp = _state_from_data(
        mujoco,
        model,
        warp_host,
        relevant,
        diagnostic_probe=False,
        coordinate=2.0,
        capacity=capacity,
    )
    comparison = _compare_states(cpu, warp)
    gates = {
        "cpu_zero_contacts": len(cpu["contacts"]) == 0,
        "warp_zero_contacts": len(warp["contacts"]) == 0,
        "cpu_zero_constraint_rows": cpu["constraint_rows"] == 0,
        "warp_zero_constraint_rows": warp["constraint_rows"] == 0,
        "no_overflow": capacity["overflow_flags"] == 0,
    }
    return {
        "fixture_type": "separated_object_control",
        "cpu": cpu,
        "warp": warp,
        "comparison": comparison,
        "gate_results": gates,
        "passed": all(gates.values()),
    }


def _fixture_result(
    cpu_evaluate: Callable[[float], dict[str, Any]],
    warp_evaluate: Callable[[float], dict[str, Any]],
    outside: float,
    inside: float,
    *,
    fixture_type: str,
) -> dict[str, Any]:
    cpu_onset, cpu_first = _find_onset(cpu_evaluate, outside, inside, iterations=28)
    warp_onset, warp_first = _find_onset(warp_evaluate, outside, inside, iterations=28)
    cpu_relative_coordinate = cpu_onset - EVALUATION_PENETRATION_M
    cpu_pose = cpu_evaluate(cpu_relative_coordinate)
    warp_cpu_pose = warp_evaluate(cpu_relative_coordinate)
    warp_own_pose = warp_evaluate(warp_onset - EVALUATION_PENETRATION_M)
    comparison = _compare_states(cpu_pose, warp_cpu_pose)
    onset_error = abs((warp_onset - cpu_onset) * 1000.0)
    total_error = comparison["tactile"]["total_magnitude_relative_error"]
    overflow = int((warp_cpu_pose["capacity"] or {}).get("overflow_flags", 0))
    gates = {
        "onset_difference": onset_error <= FIXTURE_GATES["onset_difference_mm_max"],
        "total_touch": total_error is not None
        and total_error <= FIXTURE_GATES["total_touch_relative_error_max"],
        "active_sensor_jaccard": comparison["tactile"]["active_jaccard"]
        >= FIXTURE_GATES["active_sensor_jaccard_min"],
        "no_overflow": overflow == 0,
    }
    return {
        "fixture_type": fixture_type,
        "evaluation_rule": "0.25 mm inside this representation's CPU 3.12 onset",
        "cpu_onset_coordinate_m": cpu_onset,
        "warp_onset_coordinate_m": warp_onset,
        "onset_difference_mm": (warp_onset - cpu_onset) * 1000.0,
        "cpu_first_contact": cpu_first,
        "warp_first_contact": warp_first,
        "cpu_relative_pose": {
            "coordinate_m": cpu_relative_coordinate,
            "cpu": cpu_pose,
            "warp": warp_cpu_pose,
            "comparison": comparison,
        },
        "warp_own_relative_pose": warp_own_pose,
        "gate_results": gates,
        "passed": all(gates.values()),
    }


def compare_static_fixtures(
    xml_path: str | Path, features_path: str | Path
) -> dict[str, Any]:
    import mujoco

    mjw, wp = _warp_bindings()
    xml_path = _mujoco_312_xml(xml_path)
    features = json.loads(Path(features_path).read_text(encoding="utf-8"))["features"]
    fixtures: dict[str, Any] = {}
    compatibility_changes: list[str] = []
    with wp.ScopedDevice("cuda:0"):
        separated_model = mujoco.MjModel.from_xml_path(str(xml_path))
        compatibility_changes = _apply_shared_collision_options(mujoco, separated_model)
        separated_warp_model = mjw.put_model(separated_model)
        fixtures["separated_contact"] = _evaluate_separated(
            mujoco, mjw, wp, separated_model, separated_warp_model
        )

        for fixture_name, fixture in HAND_FIXTURES.items():
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            _apply_shared_collision_options(mujoco, model)
            target_id, _ = _configure_collision_geoms(
                mujoco, model, target_geom_name=fixture["target_geom"]
            )
            assert target_id is not None
            object_ids = _object_geom_ids(mujoco, model)

            def relevant(contact: Any, target: int = target_id) -> bool:
                ids = {int(value) for value in np.asarray(contact.geom)}
                return target in ids and bool(ids & object_ids)

            warp_model = mjw.put_model(model)

            def cpu_evaluate(coordinate: float) -> dict[str, Any]:
                seed = _hand_seed(mujoco, model, fixture, coordinate)
                return _state_from_data(
                    mujoco,
                    model,
                    seed,
                    relevant,
                    diagnostic_probe=False,
                    coordinate=coordinate,
                )

            def warp_evaluate(coordinate: float) -> dict[str, Any]:
                seed = _hand_seed(mujoco, model, fixture, coordinate)
                host, capacity = _warp_forward_host(
                    mujoco, mjw, wp, model, warp_model, seed
                )
                return _state_from_data(
                    mujoco,
                    model,
                    host,
                    relevant,
                    diagnostic_probe=False,
                    coordinate=coordinate,
                    capacity=capacity,
                )

            fixtures[fixture_name] = _fixture_result(
                cpu_evaluate,
                warp_evaluate,
                fixture["outside_coordinate_m"],
                fixture["inside_coordinate_m"],
                fixture_type="n500_hand",
            )

        for fixture_name in FEATURE_NAMES:
            model, temporary_path = _temporary_probe_model(mujoco, xml_path)
            try:
                _apply_shared_collision_options(mujoco, model)
                _, probe_id = _configure_collision_geoms(
                    mujoco, model, probe_geom_name=PROBE_GEOM_NAME
                )
                assert probe_id is not None
                warp_model = mjw.put_model(model)

                def relevant(contact: Any, probe: int = probe_id) -> bool:
                    return probe in [int(value) for value in np.asarray(contact.geom)]

                feature = features[fixture_name]

                def cpu_evaluate(coordinate: float) -> dict[str, Any]:
                    seed = _feature_seed(mujoco, model, feature, probe_id, coordinate)
                    return _state_from_data(
                        mujoco,
                        model,
                        seed,
                        relevant,
                        diagnostic_probe=True,
                        coordinate=coordinate,
                    )

                def warp_evaluate(coordinate: float) -> dict[str, Any]:
                    seed = _feature_seed(mujoco, model, feature, probe_id, coordinate)
                    host, capacity = _warp_forward_host(
                        mujoco, mjw, wp, model, warp_model, seed
                    )
                    return _state_from_data(
                        mujoco,
                        model,
                        host,
                        relevant,
                        diagnostic_probe=True,
                        coordinate=coordinate,
                        capacity=capacity,
                    )

                fixtures[fixture_name] = _fixture_result(
                    cpu_evaluate,
                    warp_evaluate,
                    0.006,
                    -0.003,
                    fixture_type="diagnostic_probe",
                )
            finally:
                temporary_path.unlink(missing_ok=True)

    return {
        "mujoco_version": mujoco.__version__,
        "mujoco_warp_version": mjw.__version__,
        "warp_version": wp.__version__,
        "xml": str(xml_path),
        "compatibility_changes": compatibility_changes,
        "gates": FIXTURE_GATES,
        "fixtures": fixtures,
        "passed": all(fixture["passed"] for fixture in fixtures.values()),
    }


def compare_n500_one_step(xml_path: str | Path) -> dict[str, Any]:
    result = compare_one_step(
        xml_path,
        mode="settled_contact",
        settle_steps=200,
        contacts_per_world=2048,
        constraints_per_world=4096,
    )
    tactile = result["tactile_metrics"]
    overflow = max(result["gpu_overflow_flags"], default=0)
    gates = {
        "qpos": result["qpos"]["max_abs"] <= N500_GATES["qpos_max_abs"],
        "qvel": result["qvel"]["max_abs"] <= N500_GATES["qvel_max_abs"],
        "touch_max": tactile["max_absolute_error"] <= N500_GATES["touch_max_abs"],
        "total_touch": tactile["total_magnitude_relative_error"]
        <= N500_GATES["total_touch_relative_error_max"],
        "contact_geom_pairs": result["contact_geom_pairs_match"],
        "no_overflow": overflow <= N500_GATES["overflow_flags_max"],
    }
    result["gates"] = N500_GATES
    result["gate_results"] = gates
    result["passed"] = all(gates.values())
    return result


def compare_policy_action(xml_path: str | Path) -> dict[str, Any]:
    import mujoco
    import torch

    from .task import ShadowHandTaskConfig, ShadowHandWarpTask
    from .warp_backend import MujocoWarpBackend

    backend = MujocoWarpBackend(
        xml_path,
        worlds=1,
        contacts_per_world=2048,
        constraints_per_world=4096,
    )
    task = ShadowHandWarpTask(
        backend, config=ShadowHandTaskConfig(max_episode_steps=2), seed=123
    )
    task.reset()
    backend.synchronize()
    actions = torch.linspace(-0.5, 0.5, 20, device="cuda").reshape(1, 20)
    task.apply_actions(actions)
    backend.synchronize()
    cpu_data = mujoco.MjData(backend.model)
    cpu_data.qpos[:] = backend.qpos[0].cpu().numpy()
    cpu_data.qvel[:] = backend.qvel[0].cpu().numpy()
    cpu_data.ctrl[:] = backend.ctrl[0].cpu().numpy()
    cpu_data.qacc_warmstart[:] = backend.qacc_warmstart[0].cpu().numpy()
    cpu_data.time = float(backend.time[0].cpu())
    mujoco.mj_step(backend.model, cpu_data, nstep=20)

    step = task.step(actions)
    backend.synchronize()
    warp_host = mujoco.MjData(backend.model)
    backend.mjw.get_data_into(warp_host, backend.model, backend.data, world_id=0)
    touch_indices = np.asarray(backend.sensor_layout.touch_data_indices, dtype=np.int64)
    cpu_touch = np.asarray(cpu_data.sensordata[touch_indices], dtype=np.float64)
    warp_touch = np.asarray(backend.touch[0].cpu().numpy(), dtype=np.float64)
    sites = _touch_site_names(mujoco, backend.model, backend.sensor_layout)
    touch = tactile_metrics(
        cpu_touch,
        warp_touch,
        names=backend.sensor_layout.touch_names,
        sites=sites,
    )
    cpu_achieved = torch.as_tensor(
        cpu_data.qpos[task.layout.object_qpos_start : task.layout.object_qpos_start + 7],
        dtype=backend.qpos.dtype,
        device="cuda",
    ).unsqueeze(0)
    expected_reward = task.compute_rewards(cpu_achieved, task.goals)
    expected_success = expected_reward + 1.0
    cpu_contacts = _contact_snapshot(mujoco, backend.model, cpu_data)
    warp_contacts = _contact_snapshot(mujoco, backend.model, warp_host)
    overflow = int(backend.overflow_flags.max().cpu())
    qpos = error_metrics(cpu_data.qpos, backend.qpos[0].cpu().numpy()).to_dict()
    qvel = error_metrics(cpu_data.qvel, backend.qvel[0].cpu().numpy()).to_dict()
    gates = {
        "qpos": qpos["max_abs"] <= POLICY_ACTION_GATES["qpos_max_abs"],
        "qvel": qvel["max_abs"] <= POLICY_ACTION_GATES["qvel_max_abs"],
        "touch_max": touch["max_absolute_error"]
        <= POLICY_ACTION_GATES["touch_max_abs"],
        "reward_exact": bool(torch.equal(step.rewards, expected_reward)),
        "success_exact": bool(torch.equal(step.success, expected_success)),
        "no_overflow": overflow <= POLICY_ACTION_GATES["overflow_flags_max"],
    }
    return {
        "physics_substeps": 20,
        "action": actions[0].cpu().tolist(),
        "qpos": qpos,
        "qvel": qvel,
        "touch": touch,
        "cpu_contacts": len(cpu_contacts),
        "warp_contacts": len(warp_contacts),
        "cpu_constraint_rows": int(cpu_data.nefc),
        "warp_constraint_rows": int(warp_host.nefc),
        "cpu_contact_geom_pairs": contact_geom_pair_counts(cpu_contacts),
        "warp_contact_geom_pairs": contact_geom_pair_counts(warp_contacts),
        "reward_cpu": expected_reward.cpu().tolist(),
        "reward_warp": step.rewards.cpu().tolist(),
        "success_cpu": expected_success.cpu().tolist(),
        "success_warp": step.success.cpu().tolist(),
        "active_contacts": int(backend.active_contact_counts.sum().cpu()),
        "constraints": backend.constraint_counts.cpu().tolist(),
        "overflow_flags": backend.overflow_flags.cpu().tolist(),
        "gates": POLICY_ACTION_GATES,
        "gate_results": gates,
        "passed": all(gates.values()),
    }


def validate_representation(
    name: str,
    xml_path: str | Path,
    manifest_path: str | Path,
    features_path: str | Path,
    expected_pieces: int,
) -> dict[str, Any]:
    import mujoco

    compatible_xml = _mujoco_312_xml(xml_path)
    audit_model = mujoco.MjModel.from_xml_path(str(compatible_xml))
    compatibility_changes = _apply_shared_collision_options(mujoco, audit_model)
    audit = audit_native_model(
        mujoco,
        audit_model,
        compatible_xml,
        Path(manifest_path).resolve(),
        expected_pieces,
    )
    fixtures = compare_static_fixtures(compatible_xml, features_path)
    n500 = compare_n500_one_step(compatible_xml)
    policy = compare_policy_action(compatible_xml)
    gates = {
        "model_contract": audit["passed"],
        "fixtures": fixtures["passed"],
        "n500_one_step": n500["passed"],
        "policy_action_20_substeps": policy["passed"],
    }
    return {
        "name": name,
        "piece_count": expected_pieces,
        "new_experiment_definition": True,
        "historical_flex_results_pooled": False,
        "xml": str(Path(xml_path).resolve()),
        "compatible_xml": str(compatible_xml),
        "compatibility_changes": compatibility_changes,
        "model_audit": audit,
        "fixtures": fixtures,
        "n500_one_step": n500,
        "policy_action_20_substeps": policy,
        "gate_results": gates,
        "parity_passed": all(gates.values()),
    }
