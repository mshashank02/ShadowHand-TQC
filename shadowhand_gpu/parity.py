"""Matched-state CPU MuJoCo versus direct MuJoCo Warp parity measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .warp_backend import MujocoWarpBackend


@dataclass(frozen=True)
class ErrorMetrics:
    max_abs: float
    mean_abs: float
    root_mean_square: float
    reference_abs_max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def error_metrics(reference: np.ndarray, actual: np.ndarray) -> ErrorMetrics:
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {actual.shape}")
    error = actual - reference
    return ErrorMetrics(
        max_abs=float(np.max(np.abs(error), initial=0.0)),
        mean_abs=float(np.mean(np.abs(error))) if error.size else 0.0,
        root_mean_square=float(np.sqrt(np.mean(np.square(error)))) if error.size else 0.0,
        reference_abs_max=float(np.max(np.abs(reference), initial=0.0)),
    )


def tactile_metrics(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    names: tuple[str, ...] = (),
    sites: tuple[str, ...] = (),
    active_threshold: float = 1e-8,
    top_k: int = 10,
) -> dict[str, Any]:
    """Return tactile-specific magnitude, activation, correlation, and error data."""
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {actual.shape}")
    absolute_error = np.abs(actual - reference)
    ref_active = set(np.flatnonzero(np.abs(reference) > active_threshold).tolist())
    actual_active = set(np.flatnonzero(np.abs(actual) > active_threshold).tolist())
    intersection = ref_active & actual_active
    union = ref_active | actual_active
    correlation: float | None = None
    if reference.size >= 2 and np.std(reference) > 0.0 and np.std(actual) > 0.0:
        correlation = float(np.corrcoef(reference, actual)[0, 1])
    nonzero_errors = np.flatnonzero(absolute_error > 0.0)
    order = nonzero_errors[
        np.argsort(-absolute_error[nonzero_errors], kind="stable")[: min(top_k, nonzero_errors.size)]
    ]
    top_errors = []
    for index in order:
        sensor_name = names[index] if index < len(names) else ""
        site_name = sites[index] if index < len(sites) else ""
        region_source = (site_name or sensor_name).split(":")[-1]
        for prefix in ("TS_", "T_"):
            if region_source.startswith(prefix):
                region_source = region_source[len(prefix) :]
        region = region_source.split("_")[0] if region_source else ""
        top_errors.append(
            {
                "touch_index": int(index),
                "sensor_name": sensor_name,
                "site_name": site_name,
                "region": region,
                "cpu": float(reference[index]),
                "warp": float(actual[index]),
                "absolute_error": float(absolute_error[index]),
            }
        )
    cpu_total = float(np.sum(np.abs(reference)))
    return {
        "max_absolute_error": float(np.max(absolute_error, initial=0.0)),
        "mean_absolute_error": float(np.mean(absolute_error)) if absolute_error.size else 0.0,
        "median_absolute_error": float(np.median(absolute_error)) if absolute_error.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(actual - reference)))) if reference.size else 0.0,
        "pearson_correlation": correlation,
        "active_threshold": float(active_threshold),
        "active_sensors_cpu": len(ref_active),
        "active_sensors_warp": len(actual_active),
        "active_sensor_intersection": len(intersection),
        "active_sensor_union": len(union),
        "active_sensor_jaccard": float(len(intersection) / len(union)) if union else 1.0,
        "cpu_total_tactile_magnitude": cpu_total,
        "warp_total_tactile_magnitude": float(np.sum(np.abs(actual))),
        "total_magnitude_relative_error": (
            float(abs(np.sum(np.abs(actual)) - cpu_total) / cpu_total) if cpu_total else 0.0
        ),
        "top_error_sensors": top_errors,
    }


def _touch_site_names(mujoco: Any, model: Any, layout: Any) -> tuple[str, ...]:
    names: list[str] = []
    sensor_by_id = {sensor.sensor_id: sensor for sensor in layout.sensors}
    for sensor_id in layout.touch_sensor_ids:
        sensor = sensor_by_id[sensor_id]
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sensor.object_id)
        names.append(site_name or "")
    return tuple(names)


def _contact_snapshot(mujoco: Any, model: Any, data: Any, *, limit: int = 128) -> list[dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    force = np.zeros(6, dtype=np.float64)
    for contact_id in range(min(int(data.ncon), limit)):
        contact = data.contact[contact_id]
        mujoco.mj_contactForce(model, data, contact_id, force)
        geom_ids = [int(value) for value in np.asarray(contact.geom)]
        if int(model.nflex):
            flex_ids = [int(value) for value in np.asarray(contact.flex)]
            elem_ids = [int(value) for value in np.asarray(contact.elem)]
            vert_ids = [int(value) for value in np.asarray(contact.vert)]
        else:
            flex_ids = elem_ids = vert_ids = [-1, -1]
        contacts.append(
            {
                "contact_id": contact_id,
                "geom_ids": geom_ids,
                "geom_names": [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) if geom_id >= 0 else None
                    for geom_id in geom_ids
                ],
                "flex_ids": flex_ids,
                "element_ids": elem_ids,
                "vertex_ids": vert_ids,
                "position": np.asarray(contact.pos, dtype=np.float64).tolist(),
                "normal": np.asarray(contact.frame, dtype=np.float64)[:3].tolist(),
                "distance": float(contact.dist),
                "dimension": int(contact.dim),
                "force": force.copy().tolist(),
            }
        )
    return contacts


def _contact_comparison(cpu: list[dict[str, Any]], warp: list[dict[str, Any]]) -> dict[str, Any]:
    paired = min(len(cpu), len(warp))
    if not paired:
        return {
            "cpu_count": len(cpu),
            "warp_count": len(warp),
            "paired_contacts": 0,
            "same_geom_pair_count": 0,
            "position_max_abs_error": 0.0,
            "normal_max_abs_error": 0.0,
            "distance_max_abs_error": 0.0,
            "force_max_abs_error": 0.0,
        }
    position_errors = []
    normal_errors = []
    distance_errors = []
    force_errors = []
    same_pairs = 0
    unmatched = set(range(len(warp)))
    matched_contacts: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for cpu_contact in cpu:
        if not unmatched:
            break
        same_geom = [
            index
            for index in unmatched
            if warp[index]["geom_ids"] == cpu_contact["geom_ids"]
        ]
        candidates = same_geom or list(unmatched)
        cpu_position = np.asarray(cpu_contact["position"])
        selected = min(
            candidates,
            key=lambda index: float(
                np.linalg.norm(cpu_position - np.asarray(warp[index]["position"]))
            ),
        )
        unmatched.remove(selected)
        warp_contact = warp[selected]
        same_pairs += bool(same_geom)
        matched_contacts.append((cpu_contact, warp_contact))
    for cpu_contact, warp_contact in matched_contacts:
        position_errors.extend(
            np.abs(np.asarray(cpu_contact["position"]) - np.asarray(warp_contact["position"])).tolist()
        )
        normal_errors.extend(
            np.abs(np.asarray(cpu_contact["normal"]) - np.asarray(warp_contact["normal"])).tolist()
        )
        distance_errors.append(abs(cpu_contact["distance"] - warp_contact["distance"]))
        force_errors.extend(
            np.abs(np.asarray(cpu_contact["force"]) - np.asarray(warp_contact["force"])).tolist()
        )
    return {
        "cpu_count": len(cpu),
        "warp_count": len(warp),
        "paired_contacts": paired,
        "same_geom_pair_count": same_pairs,
        "position_max_abs_error": float(max(position_errors, default=0.0)),
        "normal_max_abs_error": float(max(normal_errors, default=0.0)),
        "distance_max_abs_error": float(max(distance_errors, default=0.0)),
        "force_max_abs_error": float(max(force_errors, default=0.0)),
    }


def zero_action_control(model: Any) -> np.ndarray:
    """Return the absolute actuator control produced by the task's zero action."""
    if int(model.nu) == 0:
        return np.zeros(0, dtype=np.float64)
    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    return 0.5 * (ranges[:, 0] + ranges[:, 1])


def _object_qpos_address(mujoco: Any, model: Any) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
    if joint_id < 0:
        raise ValueError("model does not contain object:joint")
    return int(model.jnt_qposadr[joint_id])


def prepare_cpu_state(
    mujoco: Any,
    model: Any,
    *,
    mode: str,
    settle_steps: int,
) -> Any:
    data = mujoco.MjData(model)
    data.ctrl[:] = zero_action_control(model)
    if mode == "no_contact":
        qpos_address = _object_qpos_address(mujoco, model)
        data.qpos[qpos_address : qpos_address + 3] = (1.0, 0.87, 2.0)
        data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
    elif mode == "settled_contact":
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
    else:
        raise ValueError(f"unknown parity state mode {mode!r}")
    return data


def compare_one_step(
    xml_path: str | Path,
    *,
    mode: str,
    settle_steps: int = 200,
    contacts_per_world: int = 8192,
    constraints_per_world: int = 4096,
) -> dict[str, Any]:
    """Compare one CPU and GPU physics step from exactly the same integration state."""
    import mujoco

    backend = MujocoWarpBackend(
        xml_path,
        worlds=1,
        contacts_per_world=contacts_per_world,
        constraints_per_world=constraints_per_world,
    )
    cpu_data = prepare_cpu_state(
        mujoco,
        backend.model,
        mode=mode,
        settle_steps=settle_steps,
    )
    pre_cpu_contacts = int(cpu_data.ncon)
    pre_state = {
        "qpos": cpu_data.qpos.copy(),
        "qvel": cpu_data.qvel.copy(),
        "ctrl": cpu_data.ctrl.copy(),
        "qacc_warmstart": cpu_data.qacc_warmstart.copy(),
        "time": float(cpu_data.time),
    }

    backend.set_state(**pre_state)
    mujoco.mj_step(backend.model, cpu_data)
    backend.step()
    backend.synchronize()

    gpu_data = mujoco.MjData(backend.model)
    backend.mjw.get_data_into(gpu_data, backend.model, backend.data, world_id=0)

    gpu_qpos = backend.qpos[0].detach().cpu().numpy()
    gpu_qvel = backend.qvel[0].detach().cpu().numpy()
    gpu_sensordata = backend.sensordata[0].detach().cpu().numpy()
    gpu_touch = backend.touch[0].detach().cpu().numpy()
    touch_indices = np.asarray(backend.sensor_layout.touch_data_indices, dtype=np.int64)
    cpu_touch = cpu_data.sensordata[touch_indices]
    touch_sites = _touch_site_names(mujoco, backend.model, backend.sensor_layout)
    cpu_contacts = _contact_snapshot(mujoco, backend.model, cpu_data)
    gpu_contacts = _contact_snapshot(mujoco, backend.model, gpu_data)
    qpos_address = _object_qpos_address(mujoco, backend.model)
    object_joint_id = mujoco.mj_name2id(
        backend.model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint"
    )
    qvel_address = int(backend.model.jnt_dofadr[object_joint_id])

    return {
        "mode": mode,
        "settle_steps": settle_steps if mode == "settled_contact" else 0,
        "cpu_contacts_before_step": pre_cpu_contacts,
        "cpu_contacts_after_step": int(cpu_data.ncon),
        "gpu_active_contact_candidates": int(backend.active_contact_counts.detach().cpu().sum()),
        "gpu_constraints": backend.constraint_counts.detach().cpu().tolist(),
        "qpos": error_metrics(cpu_data.qpos, gpu_qpos).to_dict(),
        "qvel": error_metrics(cpu_data.qvel, gpu_qvel).to_dict(),
        "object_pose": {
            "cpu": cpu_data.qpos[qpos_address : qpos_address + 7].tolist(),
            "warp": gpu_qpos[qpos_address : qpos_address + 7].tolist(),
            "error": error_metrics(
                cpu_data.qpos[qpos_address : qpos_address + 7],
                gpu_qpos[qpos_address : qpos_address + 7],
            ).to_dict(),
        },
        "object_velocity": {
            "cpu": cpu_data.qvel[qvel_address : qvel_address + 6].tolist(),
            "warp": gpu_qvel[qvel_address : qvel_address + 6].tolist(),
            "error": error_metrics(
                cpu_data.qvel[qvel_address : qvel_address + 6],
                gpu_qvel[qvel_address : qvel_address + 6],
            ).to_dict(),
        },
        "sensordata": error_metrics(cpu_data.sensordata, gpu_sensordata).to_dict(),
        "touch": error_metrics(cpu_touch, gpu_touch).to_dict(),
        "tactile_metrics": tactile_metrics(
            cpu_touch,
            gpu_touch,
            names=backend.sensor_layout.touch_names,
            sites=touch_sites,
        ),
        "cpu_touch_nonzero": int(np.count_nonzero(cpu_touch)),
        "gpu_touch_nonzero": int(np.count_nonzero(gpu_touch)),
        "cpu_touch_max": float(np.max(cpu_touch, initial=0.0)),
        "gpu_touch_max": float(np.max(gpu_touch, initial=0.0)),
        "contact_comparison": _contact_comparison(cpu_contacts, gpu_contacts),
        "cpu_contacts": cpu_contacts,
        "warp_contacts": gpu_contacts,
        "backend": backend.report(),
    }


def _copy_integration_state(source: Any, target: Any) -> None:
    target.qpos[:] = source.qpos
    target.qvel[:] = source.qvel
    target.ctrl[:] = source.ctrl
    target.qacc_warmstart[:] = source.qacc_warmstart
    target.time = source.time


def _object_inertial_summary(mujoco: Any, model: Any) -> dict[str, Any]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if body_id < 0:
        raise ValueError("model does not contain object body")
    return {
        "body_id": int(body_id),
        "mass": float(model.body_mass[body_id]),
        "center_of_mass": np.asarray(model.body_ipos[body_id], dtype=np.float64).tolist(),
        "diagonal_inertia": np.asarray(model.body_inertia[body_id], dtype=np.float64).tolist(),
    }


def compare_cpu_old_new(
    old_xml_path: str | Path,
    new_xml_path: str | Path,
    *,
    settle_steps: int = 200,
) -> dict[str, Any]:
    """Quantify the CPU-only scientific change from rigid flex to rigid mesh geom."""
    import mujoco

    from .model_loader import load_project_model
    from .sensors import build_sensor_layout

    old_model, old_report = load_project_model(old_xml_path, reference_compat=True)
    new_model, new_report = load_project_model(new_xml_path, reference_compat=True)
    if (old_model.nq, old_model.nv, old_model.nu, old_model.nsensordata) != (
        new_model.nq,
        new_model.nv,
        new_model.nu,
        new_model.nsensordata,
    ):
        raise ValueError("old and new models do not share the same state/sensor contract")
    old_layout = build_sensor_layout(old_model)
    new_layout = build_sensor_layout(new_model)
    if old_layout.touch_names != new_layout.touch_names:
        raise ValueError("old and new models do not share touch sensor ordering")
    old_touch_indices = np.asarray(old_layout.touch_data_indices, dtype=np.int64)
    new_touch_indices = np.asarray(new_layout.touch_data_indices, dtype=np.int64)
    sites = _touch_site_names(mujoco, old_model, old_layout)

    old_settled = prepare_cpu_state(
        mujoco, old_model, mode="settled_contact", settle_steps=settle_steps
    )
    new_settled = prepare_cpu_state(
        mujoco, new_model, mode="settled_contact", settle_steps=settle_steps
    )
    independent = {
        "qpos": error_metrics(old_settled.qpos, new_settled.qpos).to_dict(),
        "qvel": error_metrics(old_settled.qvel, new_settled.qvel).to_dict(),
        "touch": tactile_metrics(
            old_settled.sensordata[old_touch_indices],
            new_settled.sensordata[new_touch_indices],
            names=old_layout.touch_names,
            sites=sites,
        ),
        "old_contact_count": int(old_settled.ncon),
        "new_contact_count": int(new_settled.ncon),
    }

    new_matched = mujoco.MjData(new_model)
    _copy_integration_state(old_settled, new_matched)
    mujoco.mj_forward(old_model, old_settled)
    mujoco.mj_forward(new_model, new_matched)
    old_contacts_before = _contact_snapshot(mujoco, old_model, old_settled)
    new_contacts_before = _contact_snapshot(mujoco, new_model, new_matched)
    old_touch_before = old_settled.sensordata[old_touch_indices].copy()
    new_touch_before = new_matched.sensordata[new_touch_indices].copy()

    mujoco.mj_step(old_model, old_settled)
    mujoco.mj_step(new_model, new_matched)
    old_contacts_after = _contact_snapshot(mujoco, old_model, old_settled)
    new_contacts_after = _contact_snapshot(mujoco, new_model, new_matched)
    old_touch_after = old_settled.sensordata[old_touch_indices]
    new_touch_after = new_matched.sensordata[new_touch_indices]

    return {
        "settle_steps": settle_steps,
        "old_model": old_report.to_dict(),
        "new_model": new_report.to_dict(),
        "old_inertial": _object_inertial_summary(mujoco, old_model),
        "new_inertial": _object_inertial_summary(mujoco, new_model),
        "independent_settled_trajectories": independent,
        "matched_old_settled_state": {
            "contacts_before_step": {
                "old_count": len(old_contacts_before),
                "new_count": len(new_contacts_before),
                "old": old_contacts_before,
                "new": new_contacts_before,
            },
            "touch_before_step": tactile_metrics(
                old_touch_before,
                new_touch_before,
                names=old_layout.touch_names,
                sites=sites,
            ),
            "qpos_after_step": error_metrics(old_settled.qpos, new_matched.qpos).to_dict(),
            "qvel_after_step": error_metrics(old_settled.qvel, new_matched.qvel).to_dict(),
            "touch_after_step": tactile_metrics(
                old_touch_after,
                new_touch_after,
                names=old_layout.touch_names,
                sites=sites,
            ),
            "contacts_after_step": {
                "old_count": len(old_contacts_after),
                "new_count": len(new_contacts_after),
                "old": old_contacts_after,
                "new": new_contacts_after,
            },
        },
    }
