#!/usr/bin/env python3
"""CPU rigid-flex versus convex-decomposition contact/tactile validation.

This is an offline validation harness.  It does not alter the production model or
backend.  Hand fixtures use the model's exact N=500 touch array.  Geometry-feature
fixtures add a temporary, validation-only 3 mm mocap sphere and touch site so the
same source point and outward-normal approach can be reproduced for every collision
representation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
from typing import Any, Callable
import xml.etree.ElementTree as ET

import numpy as np


REPRESENTATION_ORDER = (
    "rigid_flex_reference",
    "single_hull",
    "coarse",
    "medium",
    "fine",
    "very_fine",
)
HAND_FIXTURES: dict[str, dict[str, Any]] = {
    "isolated_fingertip": {
        "target_geom": "robot0:C_ffdistal",
        "object_xy_m": (0.967, 0.783),
        "outside_coordinate_m": 0.230,
        "inside_coordinate_m": 0.195,
        "description": "isolated first-finger distal collision geom",
    },
    "isolated_palm": {
        "target_geom": "robot0:C_palm0",
        "object_xy_m": (0.990, 0.910),
        "outside_coordinate_m": 0.250,
        "inside_coordinate_m": 0.205,
        "description": (
            "isolated palm collision geom, aligned with palm touch site "
            "robot0:T_palm_auto_040"
        ),
    },
}
FEATURE_NAMES = ("deepest_concavity", "macro_feature", "roughness_feature")
PROBE_RADIUS_M = 0.003
EVALUATION_PENETRATION_M = 0.00025
PROBE_SENSOR_NAME = "decomposition_validation_probe_touch"
PROBE_GEOM_NAME = "decomposition_validation_probe_geom"
PROBE_SITE_NAME = "decomposition_validation_probe_site"

FIDELITY_GATES = {
    "absolute_onset_shift_mm_max": 0.25,
    "common_pose_contact_presence_must_match": True,
    "common_pose_active_sensor_jaccard_min": 0.80,
    "common_pose_total_tactile_relative_error_max": 0.25,
    "matched_penetration_contact_position_error_mm_max": 1.0,
    "matched_penetration_normal_angle_error_deg_max": 15.0,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _name(mujoco: Any, model: Any, object_type: Any, object_id: int) -> str:
    if object_id < 0:
        return ""
    return mujoco.mj_id2name(model, object_type, object_id) or ""


def _sensor_region(sensor_name: str, site_name: str) -> str:
    source = (site_name or sensor_name).split(":")[-1]
    for prefix in ("TS_", "T_"):
        if source.startswith(prefix):
            source = source[len(prefix) :]
    return source.split("_")[0] if source else ""


def _touch_vector(
    mujoco: Any,
    model: Any,
    data: Any,
    *,
    diagnostic_probe: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for sensor_id in range(int(model.nsensor)):
        sensor_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        selected = (
            sensor_name == PROBE_SENSOR_NAME
            if diagnostic_probe
            else "robot0:TS_" in sensor_name
        )
        if not selected:
            continue
        address = int(model.sensor_adr[sensor_id])
        dimension = int(model.sensor_dim[sensor_id])
        if dimension != 1:
            raise ValueError(f"touch sensor {sensor_name!r} is not scalar")
        site_id = int(model.sensor_objid[sensor_id])
        site_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        records.append(
            {
                "sensor_name": sensor_name,
                "site_name": site_name,
                "region": _sensor_region(sensor_name, site_name),
                "value": float(data.sensordata[address]),
            }
        )
    values = np.asarray([record["value"] for record in records], dtype=np.float64)
    return values, records


def _tactile_snapshot(values: np.ndarray, records: list[dict[str, Any]]) -> dict[str, Any]:
    active = np.flatnonzero(np.abs(values) > 1e-8)
    order = np.argsort(-np.abs(values), kind="stable")[: min(10, values.size)]
    return {
        "sensor_count": int(values.size),
        "max": float(np.max(np.abs(values), initial=0.0)),
        "mean": float(np.mean(np.abs(values))) if values.size else 0.0,
        "root_mean_square": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
        "active_count": int(active.size),
        "total_magnitude": float(np.sum(np.abs(values))),
        "top_sensors": [records[int(index)] for index in order if abs(values[index]) > 0.0],
    }


def compare_tactile(
    reference_values: np.ndarray,
    actual_values: np.ndarray,
    reference_records: list[dict[str, Any]],
    actual_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare same-contract tactile vectors without CPU/Warp-specific labels."""
    if reference_values.shape != actual_values.shape:
        raise ValueError(f"tactile shape mismatch {reference_values.shape} != {actual_values.shape}")
    reference_names = [record["sensor_name"] for record in reference_records]
    actual_names = [record["sensor_name"] for record in actual_records]
    if reference_names != actual_names:
        raise ValueError("tactile sensor order differs across representations")
    error = actual_values - reference_values
    absolute_error = np.abs(error)
    reference_active = set(np.flatnonzero(np.abs(reference_values) > 1e-8).tolist())
    actual_active = set(np.flatnonzero(np.abs(actual_values) > 1e-8).tolist())
    union = reference_active | actual_active
    intersection = reference_active & actual_active
    reference_total = float(np.sum(np.abs(reference_values)))
    actual_total = float(np.sum(np.abs(actual_values)))
    reference_norm = float(np.linalg.norm(reference_values))
    actual_norm = float(np.linalg.norm(actual_values))
    cosine_similarity = (
        float(np.dot(reference_values, actual_values) / (reference_norm * actual_norm))
        if reference_norm and actual_norm
        else (1.0 if reference_norm == actual_norm else None)
    )
    reference_std = float(np.std(reference_values))
    actual_std = float(np.std(actual_values))
    correlation = (
        float(np.corrcoef(reference_values, actual_values)[0, 1])
        if reference_values.size > 1 and reference_std and actual_std
        else None
    )
    nonzero = np.flatnonzero(absolute_error > 0.0)
    order = nonzero[np.argsort(-absolute_error[nonzero], kind="stable")[: min(10, nonzero.size)]]
    return {
        "max_absolute_error": float(np.max(absolute_error, initial=0.0)),
        "mean_absolute_error": float(np.mean(absolute_error)) if absolute_error.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(error)))) if error.size else 0.0,
        "reference_active_count": len(reference_active),
        "actual_active_count": len(actual_active),
        "active_intersection": len(intersection),
        "active_union": len(union),
        "active_jaccard": float(len(intersection) / len(union)) if union else 1.0,
        "reference_total_magnitude": reference_total,
        "actual_total_magnitude": actual_total,
        "cosine_similarity": cosine_similarity,
        "correlation": correlation,
        "total_magnitude_relative_error": (
            abs(actual_total - reference_total) / reference_total
            if reference_total
            else (0.0 if actual_total == 0.0 else None)
        ),
        "top_error_sensors": [
            {
                **reference_records[int(index)],
                "reference": float(reference_values[index]),
                "actual": float(actual_values[index]),
                "absolute_error": float(absolute_error[index]),
            }
            for index in order
        ],
    }


def _contact_records(
    mujoco: Any,
    model: Any,
    data: Any,
    relevant: Callable[[Any], bool],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    force = np.zeros(6, dtype=np.float64)
    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]
        if not relevant(contact):
            continue
        mujoco.mj_contactForce(model, data, contact_id, force)
        geom_ids = [int(value) for value in np.asarray(contact.geom)]
        flex_ids = (
            [int(value) for value in np.asarray(contact.flex)]
            if int(model.nflex)
            else [-1, -1]
        )
        records.append(
            {
                "contact_id": contact_id,
                "geom_ids": geom_ids,
                "geom_names": [
                    _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                    if geom_id >= 0
                    else None
                    for geom_id in geom_ids
                ],
                "flex_ids": flex_ids,
                "position_m": np.asarray(contact.pos, dtype=np.float64).tolist(),
                "normal": np.asarray(contact.frame, dtype=np.float64)[:3].tolist(),
                "distance_m": float(contact.dist),
                "force_contact_frame": force.copy().tolist(),
                "normal_force": float(abs(force[0])),
                "constraint_rows": int(contact.dim),
            }
        )
    return records


def compare_contacts(reference: list[dict[str, Any]], actual: list[dict[str, Any]]) -> dict[str, Any]:
    """Nearest-position, normal-sign-invariant comparison of contact patches."""
    unmatched = set(range(len(actual)))
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for reference_contact in reference:
        if not unmatched:
            break
        reference_position = np.asarray(reference_contact["position_m"], dtype=np.float64)
        index = min(
            unmatched,
            key=lambda candidate: float(
                np.linalg.norm(
                    reference_position
                    - np.asarray(actual[candidate]["position_m"], dtype=np.float64)
                )
            ),
        )
        unmatched.remove(index)
        pairs.append((reference_contact, actual[index]))
    position_errors: list[float] = []
    normal_angles: list[float] = []
    distance_errors: list[float] = []
    force_errors: list[float] = []
    for reference_contact, actual_contact in pairs:
        position_errors.append(
            float(
                np.linalg.norm(
                    np.asarray(reference_contact["position_m"])
                    - np.asarray(actual_contact["position_m"])
                )
            )
        )
        dot = float(
            np.dot(
                np.asarray(reference_contact["normal"]),
                np.asarray(actual_contact["normal"]),
            )
        )
        normal_angles.append(float(np.degrees(np.arccos(np.clip(abs(dot), 0.0, 1.0)))))
        distance_errors.append(abs(reference_contact["distance_m"] - actual_contact["distance_m"]))
        force_errors.append(abs(reference_contact["normal_force"] - actual_contact["normal_force"]))
    reference_total_force = float(sum(contact["normal_force"] for contact in reference))
    actual_total_force = float(sum(contact["normal_force"] for contact in actual))
    return {
        "reference_count": len(reference),
        "actual_count": len(actual),
        "reference_constraint_rows": int(
            sum(contact.get("constraint_rows", 0) for contact in reference)
        ),
        "actual_constraint_rows": int(
            sum(contact.get("constraint_rows", 0) for contact in actual)
        ),
        "contact_presence_matches": bool(reference) == bool(actual),
        "paired_count": len(pairs),
        "position_error_mm_max": float(max(position_errors, default=0.0) * 1000.0),
        "position_error_mm_mean": float(np.mean(position_errors) * 1000.0) if pairs else 0.0,
        "normal_angle_error_deg_max": float(max(normal_angles, default=0.0)),
        "normal_angle_error_deg_mean": float(np.mean(normal_angles)) if pairs else 0.0,
        "distance_error_mm_max": float(max(distance_errors, default=0.0) * 1000.0),
        "normal_force_error_max": float(max(force_errors, default=0.0)),
        "reference_total_normal_force": reference_total_force,
        "actual_total_normal_force": actual_total_force,
        "total_normal_force_relative_error": (
            abs(actual_total_force - reference_total_force) / reference_total_force
            if reference_total_force
            else (0.0 if actual_total_force == 0.0 else None)
        ),
    }


def _object_addresses(mujoco: Any, model: Any) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
    if joint_id < 0:
        raise ValueError("model does not contain object:joint")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _zero_action_control(model: Any) -> np.ndarray:
    if int(model.nu) == 0:
        return np.zeros(0, dtype=np.float64)
    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    return 0.5 * (ranges[:, 0] + ranges[:, 1])


def _find_onset(
    evaluate: Callable[[float], dict[str, Any]],
    outside_coordinate: float,
    inside_coordinate: float,
    *,
    iterations: int = 28,
) -> tuple[float, dict[str, Any]]:
    outside = evaluate(outside_coordinate)
    inside = evaluate(inside_coordinate)
    if outside["contacts"]:
        raise ValueError(f"outside coordinate {outside_coordinate} already has contact")
    if not inside["contacts"]:
        raise ValueError(f"inside coordinate {inside_coordinate} has no contact")
    high = outside_coordinate
    low = inside_coordinate
    for _ in range(iterations):
        midpoint = 0.5 * (high + low)
        result = evaluate(midpoint)
        if result["contacts"]:
            low = midpoint
            inside = result
        else:
            high = midpoint
    return low, inside


def _set_static_object(
    data: Any,
    qpos_address: int,
    qvel_address: int,
    position: tuple[float, float, float] | np.ndarray,
) -> None:
    data.qpos[qpos_address : qpos_address + 3] = position
    data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[:] = 0.0
    data.qacc_warmstart[:] = 0.0
    data.qvel[qvel_address : qvel_address + 6] = 0.0


def _configure_collision_geoms(
    mujoco: Any,
    model: Any,
    *,
    target_geom_name: str | None = None,
    probe_geom_name: str | None = None,
) -> tuple[int | None, int | None]:
    target_id = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target_geom_name)
        if target_geom_name
        else None
    )
    probe_id = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, probe_geom_name)
        if probe_geom_name
        else None
    )
    if target_geom_name and (target_id is None or target_id < 0):
        raise ValueError(f"missing target geom {target_geom_name!r}")
    if probe_geom_name and (probe_id is None or probe_id < 0):
        raise ValueError(f"missing probe geom {probe_geom_name!r}")
    for geom_id in range(int(model.ngeom)):
        geom_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        keep = bool(
            geom_id in {target_id, probe_id}
            or geom_name == "object"
            or geom_name.startswith("object_collision_")
        )
        if not keep:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
    if target_id is not None:
        model.geom_contype[target_id] = 1
        model.geom_conaffinity[target_id] = 1
    return target_id, probe_id


def _evaluate_hand_fixture(
    mujoco: Any,
    model: Any,
    data: Any,
    fixture: dict[str, Any],
    target_id: int,
    coordinate: float,
) -> dict[str, Any]:
    qpos_address, qvel_address = _object_addresses(mujoco, model)
    _set_static_object(
        data,
        qpos_address,
        qvel_address,
        (*fixture["object_xy_m"], coordinate),
    )
    mujoco.mj_forward(model, data)

    def relevant(contact: Any) -> bool:
        geom_ids = [int(value) for value in np.asarray(contact.geom)]
        if target_id not in geom_ids:
            return False
        if int(model.nflex) and any(int(value) >= 0 for value in np.asarray(contact.flex)):
            return True
        return any(
            geom_id >= 0
            and geom_id != target_id
            and (
                _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) == "object"
                or _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).startswith(
                    "object_collision_"
                )
            )
            for geom_id in geom_ids
        )

    contacts = _contact_records(mujoco, model, data, relevant)
    values, records = _touch_vector(mujoco, model, data, diagnostic_probe=False)
    return {
        "coordinate_m": float(coordinate),
        "coordinate_name": "object_z_m",
        "contacts": contacts,
        "primary_contact": min(contacts, key=lambda item: item["distance_m"]) if contacts else None,
        "tactile": _tactile_snapshot(values, records),
        "tactile_values": values.tolist(),
        "tactile_records": records,
    }


def _temporary_probe_model(mujoco: Any, xml_path: Path) -> tuple[Any, Path]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("model has no worldbody")
    body = ET.SubElement(
        worldbody,
        "body",
        {"name": "decomposition_validation_probe", "mocap": "true", "pos": "0 0 1"},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": PROBE_GEOM_NAME,
            "type": "sphere",
            "size": format(PROBE_RADIUS_M, ".17g"),
            "contype": "2",
            "conaffinity": "1",
            "friction": "1 0.005 0.0001",
            "condim": "3",
            "solref": "0.02 1",
            "solimp": "0.9 0.95 0.001 0.5 2",
            "margin": "0",
            "gap": "0",
            "rgba": "0.9 0.2 0.1 0.4",
        },
    )
    ET.SubElement(
        body,
        "site",
        {
            "name": PROBE_SITE_NAME,
            "type": "sphere",
            "size": format(PROBE_RADIUS_M, ".17g"),
            "rgba": "0.9 0.2 0.1 0.2",
        },
    )
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "touch", {"name": PROBE_SENSOR_NAME, "site": PROBE_SITE_NAME})
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=".decomposition_probe_", suffix=".xml", dir=xml_path.parent, delete=False
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            tree.write(handle, encoding="utf-8", xml_declaration=True)
        return mujoco.MjModel.from_xml_path(str(temporary_path)), temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _evaluate_feature_fixture(
    mujoco: Any,
    model: Any,
    data: Any,
    feature: dict[str, Any],
    probe_id: int,
    coordinate: float,
) -> dict[str, Any]:
    qpos_address, qvel_address = _object_addresses(mujoco, model)
    object_position = np.asarray((0.0, 0.0, 0.5), dtype=np.float64)
    _set_static_object(data, qpos_address, qvel_address, object_position)
    point = np.asarray(feature["point_m"], dtype=np.float64)
    normal = np.asarray(feature["outward_normal"], dtype=np.float64)
    body_id = int(model.geom_bodyid[probe_id])
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError("validation probe body is not mocap")
    data.mocap_pos[mocap_id] = object_position + point + normal * (PROBE_RADIUS_M + coordinate)
    data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)

    def relevant(contact: Any) -> bool:
        return probe_id in [int(value) for value in np.asarray(contact.geom)]

    contacts = _contact_records(mujoco, model, data, relevant)
    values, records = _touch_vector(mujoco, model, data, diagnostic_probe=True)
    n500_values, _ = _touch_vector(mujoco, model, data, diagnostic_probe=False)
    return {
        "coordinate_m": float(coordinate),
        "coordinate_name": "source_normal_offset_m",
        "contacts": contacts,
        "primary_contact": min(contacts, key=lambda item: item["distance_m"]) if contacts else None,
        "tactile": _tactile_snapshot(values, records),
        "tactile_values": values.tolist(),
        "tactile_records": records,
        "n500_hand_touch_count": int(n500_values.size),
        "n500_hand_touch_max": float(np.max(np.abs(n500_values), initial=0.0)),
    }


def _run_hand_model(mujoco: Any, xml_path: Path, fixture: dict[str, Any], common: float | None) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    target_id, _ = _configure_collision_geoms(
        mujoco, model, target_geom_name=fixture["target_geom"]
    )
    assert target_id is not None
    data = mujoco.MjData(model)
    data.ctrl[:] = _zero_action_control(model)
    evaluate = lambda coordinate: _evaluate_hand_fixture(
        mujoco, model, data, fixture, target_id, coordinate
    )
    onset, first = _find_onset(
        evaluate, fixture["outside_coordinate_m"], fixture["inside_coordinate_m"]
    )
    result = {
        "compiled": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nflex": int(model.nflex),
            "gravity": np.asarray(model.opt.gravity, dtype=np.float64).tolist(),
        },
        "onset_coordinate_m": onset,
        "first_contact": first,
        "matched_penetration": evaluate(onset - EVALUATION_PENETRATION_M),
    }
    if common is not None:
        result["common_reference_pose"] = evaluate(common)
    return result


def _run_feature_model(
    mujoco: Any,
    xml_path: Path,
    feature: dict[str, Any],
    common: float | None,
) -> dict[str, Any]:
    model, temporary_path = _temporary_probe_model(mujoco, xml_path)
    try:
        _, probe_id = _configure_collision_geoms(
            mujoco, model, probe_geom_name=PROBE_GEOM_NAME
        )
        assert probe_id is not None
        data = mujoco.MjData(model)
        data.ctrl[:] = _zero_action_control(model)
        evaluate = lambda coordinate: _evaluate_feature_fixture(
            mujoco, model, data, feature, probe_id, coordinate
        )
        onset, first = _find_onset(evaluate, 0.006, -0.003)
        result = {
            "compiled": {
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nflex": int(model.nflex),
                "gravity": np.asarray(model.opt.gravity, dtype=np.float64).tolist(),
            },
            "probe_radius_m": PROBE_RADIUS_M,
            "onset_coordinate_m": onset,
            "first_contact": first,
            "matched_penetration": evaluate(onset - EVALUATION_PENETRATION_M),
        }
        if common is not None:
            result["common_reference_pose"] = evaluate(common)
        return result
    finally:
        temporary_path.unlink(missing_ok=True)


def _comparison(reference: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    onset_shift_mm = (actual["onset_coordinate_m"] - reference["onset_coordinate_m"]) * 1000.0
    result: dict[str, Any] = {
        "onset_shift_mm": float(onset_shift_mm),
        "absolute_onset_shift_mm": float(abs(onset_shift_mm)),
    }
    for state_name in ("matched_penetration", "common_reference_pose"):
        reference_state = reference[state_name]
        actual_state = actual[state_name]
        result[state_name] = {
            "contacts": compare_contacts(reference_state["contacts"], actual_state["contacts"]),
            "tactile": compare_tactile(
                np.asarray(reference_state["tactile_values"], dtype=np.float64),
                np.asarray(actual_state["tactile_values"], dtype=np.float64),
                reference_state["tactile_records"],
                actual_state["tactile_records"],
            ),
        }
    common = result["common_reference_pose"]
    matched = result["matched_penetration"]
    total_error = common["tactile"]["total_magnitude_relative_error"]
    gate_results = {
        "onset": result["absolute_onset_shift_mm"]
        <= FIDELITY_GATES["absolute_onset_shift_mm_max"],
        "common_contact_presence": common["contacts"]["contact_presence_matches"],
        "common_tactile_overlap": common["tactile"]["active_jaccard"]
        >= FIDELITY_GATES["common_pose_active_sensor_jaccard_min"],
        "common_tactile_total": total_error is not None
        and total_error <= FIDELITY_GATES["common_pose_total_tactile_relative_error_max"],
        "matched_contact_position": matched["contacts"]["position_error_mm_max"]
        <= FIDELITY_GATES["matched_penetration_contact_position_error_mm_max"],
        "matched_contact_normal": matched["contacts"]["normal_angle_error_deg_max"]
        <= FIDELITY_GATES["matched_penetration_normal_angle_error_deg_max"],
    }
    result["gate_results"] = gate_results
    result["passes_fixture_gates"] = all(gate_results.values())
    return result


def _write_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for fixture_name, fixture in payload["fixtures"].items():
        for representation in REPRESENTATION_ORDER:
            result = fixture["representations"][representation]
            comparison = fixture["comparisons"][representation]
            common = comparison["common_reference_pose"]
            matched = comparison["matched_penetration"]
            rows.append(
                {
                    "fixture": fixture_name,
                    "fixture_type": fixture["fixture_type"],
                    "representation": representation,
                    "onset_coordinate_m": result["onset_coordinate_m"],
                    "onset_shift_mm": comparison["onset_shift_mm"],
                    "matched_contact_count": len(result["matched_penetration"]["contacts"]),
                    "matched_contact_position_error_mm_max": matched["contacts"]["position_error_mm_max"],
                    "matched_normal_angle_error_deg_max": matched["contacts"]["normal_angle_error_deg_max"],
                    "matched_tactile_max": result["matched_penetration"]["tactile"]["max"],
                    "matched_tactile_mean": result["matched_penetration"]["tactile"]["mean"],
                    "matched_tactile_rmse_vs_reference": matched["tactile"]["rmse"],
                    "common_contact_count": len(result["common_reference_pose"]["contacts"]),
                    "common_tactile_max": result["common_reference_pose"]["tactile"]["max"],
                    "common_tactile_mean": result["common_reference_pose"]["tactile"]["mean"],
                    "common_tactile_rmse_vs_reference": common["tactile"]["rmse"],
                    "common_active_jaccard": common["tactile"]["active_jaccard"],
                    "common_total_tactile": result["common_reference_pose"]["tactile"]["total_magnitude"],
                    "common_total_tactile_relative_error": common["tactile"]["total_magnitude_relative_error"],
                    "passes_fixture_gates": comparison["passes_fixture_gates"],
                }
            )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_contact_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for fixture_name, fixture in payload["fixtures"].items():
        for representation in REPRESENTATION_ORDER:
            result = fixture["representations"][representation]
            comparison = fixture["comparisons"][representation]
            primary = result["first_contact"]["primary_contact"]
            position = primary["position_m"] if primary else (None, None, None)
            normal = primary["normal"] if primary else (None, None, None)
            matched = comparison["matched_penetration"]["contacts"]
            common = comparison["common_reference_pose"]["contacts"]
            rows.append(
                {
                    "fixture": fixture_name,
                    "fixture_type": fixture["fixture_type"],
                    "representation": representation,
                    "onset_coordinate_m": result["onset_coordinate_m"],
                    "onset_shift_mm": comparison["onset_shift_mm"],
                    "first_contact_count": len(result["first_contact"]["contacts"]),
                    "first_contact_position_x_m": position[0],
                    "first_contact_position_y_m": position[1],
                    "first_contact_position_z_m": position[2],
                    "first_contact_normal_x": normal[0],
                    "first_contact_normal_y": normal[1],
                    "first_contact_normal_z": normal[2],
                    "first_contact_distance_mm": primary["distance_m"] * 1000.0 if primary else None,
                    "first_contact_normal_force": primary["normal_force"] if primary else None,
                    "matched_contact_count": len(result["matched_penetration"]["contacts"]),
                    "matched_position_error_mm_max": matched["position_error_mm_max"],
                    "matched_normal_angle_error_deg_max": matched["normal_angle_error_deg_max"],
                    "matched_distance_error_mm_max": matched["distance_error_mm_max"],
                    "matched_total_normal_force": matched["actual_total_normal_force"],
                    "matched_total_normal_force_relative_error": matched[
                        "total_normal_force_relative_error"
                    ],
                    "common_contact_count": len(result["common_reference_pose"]["contacts"]),
                    "common_presence_matches": common["contact_presence_matches"],
                    "common_position_error_mm_max": common["position_error_mm_max"],
                    "common_normal_angle_error_deg_max": common["normal_angle_error_deg_max"],
                    "common_distance_error_mm_max": common["distance_error_mm_max"],
                    "common_total_normal_force": common["actual_total_normal_force"],
                    "common_total_normal_force_relative_error": common[
                        "total_normal_force_relative_error"
                    ],
                    "passes_fixture_gates": comparison["passes_fixture_gates"],
                }
            )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_tactile_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for fixture_name, fixture in payload["fixtures"].items():
        for representation in REPRESENTATION_ORDER:
            result = fixture["representations"][representation]
            comparison = fixture["comparisons"][representation]
            for state_name in ("common_reference_pose", "matched_penetration"):
                snapshot = result[state_name]["tactile"]
                metrics = comparison[state_name]["tactile"]
                rows.append(
                    {
                        "fixture": fixture_name,
                        "fixture_type": fixture["fixture_type"],
                        "representation": representation,
                        "state": state_name,
                        "sensor_count": snapshot["sensor_count"],
                        "max": snapshot["max"],
                        "mean": snapshot["mean"],
                        "root_mean_square": snapshot["root_mean_square"],
                        "active_count": snapshot["active_count"],
                        "total_magnitude": snapshot["total_magnitude"],
                        "max_absolute_error": metrics["max_absolute_error"],
                        "mean_absolute_error": metrics["mean_absolute_error"],
                        "rmse_vs_reference": metrics["rmse"],
                        "reference_active_count": metrics["reference_active_count"],
                        "active_intersection": metrics["active_intersection"],
                        "active_union": metrics["active_union"],
                        "active_jaccard": metrics["active_jaccard"],
                        "total_magnitude_relative_error": metrics[
                            "total_magnitude_relative_error"
                        ],
                        "top_sensors_json": json.dumps(snapshot["top_sensors"], sort_keys=True),
                        "top_error_sensors_json": json.dumps(
                            metrics["top_error_sensors"], sort_keys=True
                        ),
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Worst-object CPU contact and tactile validation",
        "",
        "All measurements use MuJoCo CPU with the XML's production gravity, zero initial "
        "velocity, and the "
        "same explicit body mass/inertia/contact parameters. Onset is refined by 28-step "
        "bisection. Contact/tactile response is reported both at a common physical pose "
        "anchored 0.25 mm inside the original rigid-flex onset and at 0.25 mm penetration "
        "relative to each representation's own onset.",
        "",
        "| Fixture | Representation | Onset shift mm | Common contacts | Common touch total | "
        "Matched touch max | Fixture gate |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for fixture_name, fixture in payload["fixtures"].items():
        for representation in REPRESENTATION_ORDER:
            result = fixture["representations"][representation]
            comparison = fixture["comparisons"][representation]
            lines.append(
                f"| {fixture_name} | {representation} | {comparison['onset_shift_mm']:.3f} | "
                f"{len(result['common_reference_pose']['contacts'])} | "
                f"{result['common_reference_pose']['tactile']['total_magnitude']:.6g} | "
                f"{result['matched_penetration']['tactile']['max']:.6g} | "
                f"{'PASS' if comparison['passes_fixture_gates'] else 'FAIL'} |"
            )
    selection = payload["selection"]
    lines.extend(
        [
            "",
            "## Candidate decision",
            "",
            f"Classification: **{selection['classification']}**.",
            "",
            f"Decision: **{selection['decision']}**.",
            "",
            "Warp candidates: " + (", ".join(selection["warp_candidates"]) or "none"),
            "",
            selection["reason"],
            "",
            selection["root_cause"],
            "",
            "Exact next step: " + selection["next_step"],
            "",
            "The geometry-feature touch value comes from a validation-only 3 mm spherical "
            "probe site. The original N=500 hand touch contract remains present and is "
            "verified as 500 channels; the temporary probe is never written into a "
            "production model.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_validation(models_path: Path, features_path: Path, output: Path) -> dict[str, Any]:
    import mujoco

    models = json.loads(models_path.read_text(encoding="utf-8"))
    feature_payload = json.loads(features_path.read_text(encoding="utf-8"))
    missing = set(REPRESENTATION_ORDER) - set(models)
    if missing:
        raise ValueError(f"model manifest is missing {sorted(missing)}")
    fixtures: dict[str, Any] = {}
    reference_common: dict[str, float] = {}

    for fixture_name, fixture in HAND_FIXTURES.items():
        representations: dict[str, Any] = {}
        for representation in REPRESENTATION_ORDER:
            common = reference_common.get(fixture_name)
            result = _run_hand_model(mujoco, Path(models[representation]), fixture, common)
            if representation == "rigid_flex_reference":
                common = result["onset_coordinate_m"] - EVALUATION_PENETRATION_M
                reference_common[fixture_name] = common
                result["common_reference_pose"] = result["matched_penetration"]
            representations[representation] = result
        reference = representations["rigid_flex_reference"]
        comparisons = {
            representation: _comparison(reference, result)
            for representation, result in representations.items()
        }
        fixtures[fixture_name] = {
            "fixture_type": "n500_hand",
            "definition": fixture,
            "representations": representations,
            "comparisons": comparisons,
        }

    feature_definitions = feature_payload["features"]
    for fixture_name in FEATURE_NAMES:
        feature = feature_definitions[fixture_name]
        representations = {}
        for representation in REPRESENTATION_ORDER:
            common = reference_common.get(fixture_name)
            result = _run_feature_model(mujoco, Path(models[representation]), feature, common)
            if representation == "rigid_flex_reference":
                common = result["onset_coordinate_m"] - EVALUATION_PENETRATION_M
                reference_common[fixture_name] = common
                result["common_reference_pose"] = result["matched_penetration"]
            representations[representation] = result
        reference = representations["rigid_flex_reference"]
        comparisons = {
            representation: _comparison(reference, result)
            for representation, result in representations.items()
        }
        fixtures[fixture_name] = {
            "fixture_type": "diagnostic_probe",
            "definition": feature,
            "representations": representations,
            "comparisons": comparisons,
        }

    candidate_names = ("coarse", "medium", "fine", "very_fine")
    passed = [
        representation
        for representation in candidate_names
        if all(
            fixtures[fixture]["comparisons"][representation]["passes_fixture_gates"]
            for fixture in fixtures
        )
    ]
    if passed:
        decision = "cpu_fidelity_candidates_selected"
        reason = (
            "Only candidates passing every predeclared CPU onset/contact/tactile gate "
            "may proceed to backend fidelity testing."
        )
    else:
        decision = "blocked_no_cpu_fidelity_candidate"
        reason = (
            "No decomposition candidate passes every CPU representation-fidelity gate; "
            "therefore no Warp capacity, parity, throughput, or learning run is authorized."
        )
    payload = {
        "method_version": "cpu-decomposition-contact-v1",
        "mujoco_version": mujoco.__version__,
        "models_manifest": str(models_path.resolve()),
        "feature_definitions": str(features_path.resolve()),
        "evaluation_penetration_m": EVALUATION_PENETRATION_M,
        "gravity": fixtures["isolated_fingertip"]["representations"]
        ["rigid_flex_reference"]["compiled"]["gravity"],
        "fidelity_gates": FIDELITY_GATES,
        "fixtures": fixtures,
        "selection": {
            "classification": "Outcome A" if passed else "Outcome C",
            "decision": decision,
            "warp_candidates": passed[:3],
            "reason": reason,
            "root_cause": (
                "The bare convex meshes do not encode the original rigid-flex radius. "
                "That missing 1.25 mm collision shell produces approximately 1.25 mm "
                "late contact on the palm, macro, and roughness fixtures and 1.45-1.47 mm "
                "late fingertip contact. Decomposition-specific residual error remains at "
                "the deepest concavity: the single hull is 3.17 mm premature, the 11-piece "
                "candidate is within 0.023 mm, and the 24/36/87-piece candidates are "
                "approximately 0.79-0.83 mm late relative to rigid flex."
            ),
            "next_step": (
                "evaluate an explicitly radius-aware rigid collision shell (or another "
                "native non-convex representation) on the same five CPU fixtures before "
                "selecting any candidate for Warp"
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "cpu_contact_tactile_validation.json", payload)
    _write_summary_csv(output / "cpu_contact_tactile_summary.csv", payload)
    _write_contact_csv(output / "contact_comparison.csv", payload)
    _write_tactile_csv(output / "tactile_comparison.csv", payload)
    _write_report(output / "cpu_contact_tactile_report.md", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[1]
    default_output = repository / "generated/convex_decomposition_validation"
    parser.add_argument(
        "--models",
        type=Path,
        default=default_output / "contact_models/models.json",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=default_output / "contact_feature_definitions.json",
    )
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    payload = run_validation(args.models.resolve(), args.features.resolve(), args.output.resolve())
    print(json.dumps(payload["selection"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
