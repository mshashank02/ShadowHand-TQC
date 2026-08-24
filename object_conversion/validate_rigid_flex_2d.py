#!/usr/bin/env python3
"""CPU-gated 3D GMSH rigid-flex versus 2D OBJ rigid-flex validation.

This offline experiment changes only the rigid flex representation.  It deliberately
does not import MuJoCo Warp: a Warp run is authorized only after the CPU gate reports
strong representation fidelity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from object_conversion.validate_decomposition_contacts import (
    EVALUATION_PENETRATION_M,
    FEATURE_NAMES,
    FIDELITY_GATES,
    HAND_FIXTURES,
    _comparison,
    _run_feature_model,
    _run_hand_model,
)


VALIDATOR_VERSION = "rigid-flex-2d-cpu-validation-v1"
PRIMARY_ONSET_GATE_MM = 0.10
SECONDARY_ONSET_GATE_MM = 0.25
NO_CONTACT_TOLERANCE = 1e-10
REPRESENTATIONS = ("original_3d_gmsh_flex", "surface_2d_obj_flex")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _absolute_from(base: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def _rigid_flexcomp(root: ET.Element) -> ET.Element:
    matches = [
        element
        for element in root.findall(".//flexcomp")
        if element.get("rigid", "false").lower() == "true"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one rigid flexcomp, found {len(matches)}")
    return matches[0]


def _contact_xml(flexcomp: ET.Element) -> dict[str, str]:
    contact = flexcomp.find("contact")
    return dict(contact.attrib) if contact is not None else {}


def build_surface_flex_model(
    source_xml: str | Path,
    exterior_obj: str | Path,
    output_xml: str | Path,
) -> dict[str, Any]:
    """Derive a portable full model with only the rigid flex representation changed."""
    source_xml = Path(source_xml).resolve()
    exterior_obj = Path(exterior_obj).resolve()
    output_xml = Path(output_xml).resolve()
    if not source_xml.is_file() or not exterior_obj.is_file():
        raise FileNotFoundError(source_xml if not source_xml.is_file() else exterior_obj)

    tree = ET.parse(source_xml)
    root = tree.getroot()
    source_flex = _rigid_flexcomp(root)
    source_attributes = dict(source_flex.attrib)
    source_contact = _contact_xml(source_flex)
    if source_attributes.get("type") != "gmsh" or source_attributes.get("dim") != "3":
        raise ValueError("source rigid flex must be the 3D GMSH reference")
    if source_attributes.get("rigid", "false").lower() != "true":
        raise ValueError("source flex is not rigid")

    # The output lives outside the source model directory.  Resolve packaging paths
    # without changing model semantics so includes and assets remain deterministic.
    for include in root.findall(".//include"):
        file_value = include.get("file")
        if file_value:
            include.set("file", _absolute_from(source_xml.parent, file_value))
    compiler = root.find("compiler")
    if compiler is not None:
        for attribute in ("meshdir", "texturedir", "assetdir"):
            value = compiler.get(attribute)
            if value:
                compiler.set(attribute, _absolute_from(source_xml.parent, value))

    source_flex.set("type", "mesh")
    source_flex.set("file", str(exterior_obj))
    source_flex.set("dim", "2")
    # trilinear interpolation is specific to a volumetric dim=3 flex and is
    # irrelevant for a rigid surface whose vertices all belong to one body.
    source_flex.attrib.pop("dof", None)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    manifest = {
        "validator_version": VALIDATOR_VERSION,
        "source_xml": str(source_xml),
        "source_xml_sha256": _sha256(source_xml),
        "exterior_obj": str(exterior_obj),
        "exterior_obj_sha256": _sha256(exterior_obj),
        "output_xml": str(output_xml),
        "output_xml_sha256": _sha256(output_xml),
        "source_flex_attributes": source_attributes,
        "surface_flex_attributes": dict(source_flex.attrib),
        "source_contact_attributes": source_contact,
        "surface_contact_attributes": _contact_xml(source_flex),
        "semantic_changes": {
            "type": [source_attributes.get("type"), "mesh"],
            "dim": [source_attributes.get("dim"), "2"],
            "file": [source_attributes.get("file"), str(exterior_obj)],
            "removed_dim3_only_attribute": "dof=trilinear",
        },
    }
    _write_json(output_xml.with_suffix(".manifest.json"), manifest)
    return manifest


def _name(mujoco: Any, model: Any, object_type: Any, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or ""


def _compiled_model_audit(mujoco: Any, xml_path: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
    if object_body < 0 or object_joint < 0:
        raise ValueError("compiled model is missing object body or free joint")
    qpos_address = int(model.jnt_qposadr[object_joint])
    dof_address = int(model.jnt_dofadr[object_joint])
    flex_body_ids = np.asarray(model.flex_vertbodyid, dtype=np.int64)
    return {
        "xml": str(xml_path.resolve()),
        "xml_sha256": _sha256(xml_path),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "object_body_id": int(object_body),
        "object_joint_id": int(object_joint),
        "object_joint_type": int(model.jnt_type[object_joint]),
        "object_qpos_address": qpos_address,
        "object_dof_address": dof_address,
        "object_global_dof_count": 6,
        "object_free_joint_count": int(
            sum(
                int(model.jnt_bodyid[joint]) == object_body
                and int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE)
                for joint in range(int(model.njnt))
            )
        ),
        "object_initial_qpos": np.asarray(
            model.qpos0[qpos_address : qpos_address + 7], dtype=np.float64
        ).tolist(),
        "object_mass": float(model.body_mass[object_body]),
        "object_inertial_pos": np.asarray(model.body_ipos[object_body], dtype=np.float64).tolist(),
        "object_inertia": np.asarray(model.body_inertia[object_body], dtype=np.float64).tolist(),
        "object_dof_damping": np.asarray(
            model.dof_damping[dof_address : dof_address + 6], dtype=np.float64
        ).tolist(),
        "nflex": int(model.nflex),
        "nflexvert": int(model.nflexvert),
        "nflexedge": int(model.nflexedge),
        "nflexelem": int(model.nflexelem),
        "flex_dim": np.asarray(model.flex_dim, dtype=np.int64).tolist(),
        "flex_rigid": np.asarray(model.flex_rigid, dtype=np.int64).tolist(),
        "flex_radius": np.asarray(model.flex_radius, dtype=np.float64).tolist(),
        "flex_vertex_body_ids": sorted(set(flex_body_ids.tolist())),
        "all_flex_vertices_on_object_body": bool(
            flex_body_ids.size and np.all(flex_body_ids == object_body)
        ),
        "flex_contact": {
            "selfcollide": np.asarray(model.flex_selfcollide, dtype=np.int64).tolist(),
            "internal": np.asarray(model.flex_internal, dtype=np.int64).tolist(),
            "condim": np.asarray(model.flex_condim, dtype=np.int64).tolist(),
            "friction": np.asarray(model.flex_friction, dtype=np.float64).tolist(),
            "solref": np.asarray(model.flex_solref, dtype=np.float64).tolist(),
            "solimp": np.asarray(model.flex_solimp, dtype=np.float64).tolist(),
            "margin": np.asarray(model.flex_margin, dtype=np.float64).tolist(),
            "gap": np.asarray(model.flex_gap, dtype=np.float64).tolist(),
            "contype": np.asarray(model.flex_contype, dtype=np.int64).tolist(),
            "conaffinity": np.asarray(model.flex_conaffinity, dtype=np.int64).tolist(),
        },
    }


def _quaternion_error_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if not left_norm or not right_norm:
        raise ValueError("orientation quaternion has zero norm")
    dot = abs(float(np.dot(left / left_norm, right / right_norm)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))))


def _midpoint_control(model: Any) -> np.ndarray:
    if int(model.nu) == 0:
        return np.zeros(0, dtype=np.float64)
    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    return 0.5 * (ranges[:, 0] + ranges[:, 1])


def _object_flex_contact_count(model: Any, data: Any) -> int:
    if not int(model.nflex):
        return 0
    return sum(
        any(int(value) >= 0 for value in np.asarray(data.contact[index].flex))
        for index in range(int(data.ncon))
    )


def run_no_contact_control(
    mujoco: Any, reference_xml: Path, surface_xml: Path
) -> dict[str, Any]:
    models = {
        REPRESENTATIONS[0]: mujoco.MjModel.from_xml_path(str(reference_xml)),
        REPRESENTATIONS[1]: mujoco.MjModel.from_xml_path(str(surface_xml)),
    }
    data: dict[str, Any] = {}
    addresses: dict[str, tuple[int, int]] = {}
    for name, model in models.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
        qadr, dadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        addresses[name] = (qadr, dadr)
        current = mujoco.MjData(model)
        current.qpos[:] = model.qpos0
        current.qpos[qadr : qadr + 7] = (1.0, 0.87, 2.0, 1.0, 0.0, 0.0, 0.0)
        current.qvel[:] = 0.0
        current.qvel[dadr : dadr + 6] = (0.1, -0.05, 0.02, 0.2, -0.1, 0.15)
        current.ctrl[:] = _midpoint_control(model)
        mujoco.mj_forward(model, current)
        data[name] = current

    rows: list[dict[str, Any]] = []
    checkpoints = {1, 10, 100}
    for step in range(1, 101):
        for name in REPRESENTATIONS:
            mujoco.mj_step(models[name], data[name])
        if step not in checkpoints:
            continue
        states: dict[str, dict[str, Any]] = {}
        for name in REPRESENTATIONS:
            qadr, dadr = addresses[name]
            current = data[name]
            states[name] = {
                "position": np.asarray(current.qpos[qadr : qadr + 3]).copy(),
                "quaternion": np.asarray(current.qpos[qadr + 3 : qadr + 7]).copy(),
                "linear_velocity": np.asarray(current.qvel[dadr : dadr + 3]).copy(),
                "angular_velocity": np.asarray(current.qvel[dadr + 3 : dadr + 6]).copy(),
                "object_flex_contacts": _object_flex_contact_count(models[name], current),
            }
        reference, surface = (states[name] for name in REPRESENTATIONS)
        rows.append(
            {
                "steps": step,
                "position_error_m": float(
                    np.linalg.norm(surface["position"] - reference["position"])
                ),
                "orientation_error_deg": _quaternion_error_degrees(
                    reference["quaternion"], surface["quaternion"]
                ),
                "linear_velocity_error": float(
                    np.linalg.norm(surface["linear_velocity"] - reference["linear_velocity"])
                ),
                "angular_velocity_error": float(
                    np.linalg.norm(surface["angular_velocity"] - reference["angular_velocity"])
                ),
                "reference_object_flex_contacts": reference["object_flex_contacts"],
                "surface_object_flex_contacts": surface["object_flex_contacts"],
            }
        )
    passed = all(
        row[metric] <= NO_CONTACT_TOLERANCE
        for row in rows
        for metric in (
            "position_error_m",
            "orientation_error_deg",
            "linear_velocity_error",
            "angular_velocity_error",
        )
    ) and all(
        row["reference_object_flex_contacts"] == row["surface_object_flex_contacts"] == 0
        for row in rows
    )
    return {"tolerance": NO_CONTACT_TOLERANCE, "checkpoints": rows, "passed": passed}


def _fixture_gate(comparison: dict[str, Any], onset_limit_mm: float) -> dict[str, bool]:
    result = dict(comparison["gate_results"])
    result["onset"] = comparison["absolute_onset_shift_mm"] < onset_limit_mm
    return result


def _run_fixture_validation(
    mujoco: Any,
    reference_xml: Path,
    surface_xml: Path,
    features_path: Path,
) -> dict[str, Any]:
    features = json.loads(features_path.read_text(encoding="utf-8"))["features"]
    fixtures: dict[str, Any] = {}
    for fixture_name, definition in HAND_FIXTURES.items():
        reference = _run_hand_model(mujoco, reference_xml, definition, None)
        common = reference["onset_coordinate_m"] - EVALUATION_PENETRATION_M
        reference["common_reference_pose"] = reference["matched_penetration"]
        surface = _run_hand_model(mujoco, surface_xml, definition, common)
        comparison = _comparison(reference, surface)
        fixtures[fixture_name] = {
            "fixture_type": "n500_hand",
            "definition": definition,
            "reference": reference,
            "surface": surface,
            "comparison": comparison,
            "primary_gate_results": _fixture_gate(comparison, PRIMARY_ONSET_GATE_MM),
            "secondary_gate_results": _fixture_gate(comparison, SECONDARY_ONSET_GATE_MM),
        }
    for fixture_name in FEATURE_NAMES:
        definition = features[fixture_name]
        reference = _run_feature_model(mujoco, reference_xml, definition, None)
        common = reference["onset_coordinate_m"] - EVALUATION_PENETRATION_M
        reference["common_reference_pose"] = reference["matched_penetration"]
        surface = _run_feature_model(mujoco, surface_xml, definition, common)
        comparison = _comparison(reference, surface)
        fixtures[fixture_name] = {
            "fixture_type": "diagnostic_probe",
            "definition": definition,
            "reference": reference,
            "surface": surface,
            "comparison": comparison,
            "primary_gate_results": _fixture_gate(comparison, PRIMARY_ONSET_GATE_MM),
            "secondary_gate_results": _fixture_gate(comparison, SECONDARY_ONSET_GATE_MM),
        }
    return fixtures


def _write_onset_csv(path: Path, fixtures: dict[str, Any]) -> None:
    rows = [
        {
            "fixture": name,
            "fixture_type": fixture["fixture_type"],
            "reference_onset_coordinate_m": fixture["reference"]["onset_coordinate_m"],
            "surface_onset_coordinate_m": fixture["surface"]["onset_coordinate_m"],
            "onset_shift_mm": fixture["comparison"]["onset_shift_mm"],
            "absolute_onset_shift_mm": fixture["comparison"]["absolute_onset_shift_mm"],
            "passes_primary_0p10mm": fixture["primary_gate_results"]["onset"],
            "passes_secondary_0p25mm": fixture["secondary_gate_results"]["onset"],
        }
        for name, fixture in fixtures.items()
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_contact_csv(path: Path, fixtures: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for name, fixture in fixtures.items():
        for state in ("common_reference_pose", "matched_penetration"):
            reference = fixture["reference"][state]
            surface = fixture["surface"][state]
            metrics = fixture["comparison"][state]["contacts"]
            rows.append(
                {
                    "fixture": name,
                    "fixture_type": fixture["fixture_type"],
                    "state": state,
                    "reference_contact_count": len(reference["contacts"]),
                    "surface_contact_count": len(surface["contacts"]),
                    "reference_constraint_rows": metrics["reference_constraint_rows"],
                    "surface_constraint_rows": metrics["actual_constraint_rows"],
                    "position_error_mm_max": metrics["position_error_mm_max"],
                    "position_error_mm_mean": metrics["position_error_mm_mean"],
                    "normal_angle_error_deg_max": metrics["normal_angle_error_deg_max"],
                    "distance_error_mm_max": metrics["distance_error_mm_max"],
                    "reference_total_normal_force": metrics["reference_total_normal_force"],
                    "surface_total_normal_force": metrics["actual_total_normal_force"],
                    "total_normal_force_relative_error": metrics[
                        "total_normal_force_relative_error"
                    ],
                    "reference_contacts_json": json.dumps(reference["contacts"], sort_keys=True),
                    "surface_contacts_json": json.dumps(surface["contacts"], sort_keys=True),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_tactile_csv(path: Path, fixtures: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for name, fixture in fixtures.items():
        for state in ("common_reference_pose", "matched_penetration"):
            reference = fixture["reference"][state]["tactile"]
            surface = fixture["surface"][state]["tactile"]
            metrics = fixture["comparison"][state]["tactile"]
            rows.append(
                {
                    "fixture": name,
                    "fixture_type": fixture["fixture_type"],
                    "state": state,
                    "sensor_count": reference["sensor_count"],
                    "reference_max": reference["max"],
                    "surface_max": surface["max"],
                    "max_absolute_error": metrics["max_absolute_error"],
                    "mean_absolute_error": metrics["mean_absolute_error"],
                    "rmse": metrics["rmse"],
                    "reference_total_magnitude": metrics["reference_total_magnitude"],
                    "surface_total_magnitude": metrics["actual_total_magnitude"],
                    "total_magnitude_relative_error": metrics[
                        "total_magnitude_relative_error"
                    ],
                    "reference_active_count": metrics["reference_active_count"],
                    "surface_active_count": metrics["actual_active_count"],
                    "active_jaccard": metrics["active_jaccard"],
                    "cosine_similarity": metrics["cosine_similarity"],
                    "correlation": metrics["correlation"],
                    "top_error_sensors_json": json.dumps(
                        metrics["top_error_sensors"], sort_keys=True
                    ),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _semantics_markdown(payload: dict[str, Any]) -> str:
    geometry = payload["geometry"]
    return f"""# MuJoCo 2D rigid-flex semantics

CPU validation runtime: MuJoCo `{payload['mujoco_version']}`.  Warp source inspected:
MuJoCo Warp `3.11.0` with Warp `1.16.0`.

`flexcomp type=\"mesh\"` accepts OBJ and `mjCFlexcomp::MakeMesh` copies mesh
vertices into flex points and OBJ faces into `dim=2` triangular elements.  With
`rigid=\"true\"`, every flex vertex belongs to the parent object body.  The compiled
model therefore retains one free joint and six global rigid-body DOFs without
deformation DOFs.  A 2D element is a triangle inflated by `flex_radius`; individual
triangles retain the closed surface's non-convex structure instead of forming the
convex hull used by an ordinary mesh geom.

CPU MuJoCo dispatches geometry-versus-triangle elements in `mj_collideGeomElem`.
MuJoCo Warp dispatches `dim=2` through `_flex_narrowphase_unified` and
`_collide_geom_triangle_detect`, passing `flex_radius` as `tri_radius`.
`_flex_tet_internal_collisions_detect` explicitly returns unless `flex_dim == 3`,
and the tetrahedron-to-four-face `_flex_narrowphase_tet_detect` path is separate.
Both known tetrahedral paths are therefore avoided by this representation.

Exact surface:

- source GMSH SHA-256: `{geometry['source_gmsh_sha256']}`
- exterior OBJ SHA-256: `{geometry['exterior_obj_sha256']}`
- vertices: `{geometry['vertex_count']}`
- triangles: `{geometry['triangle_count']}`
- source-coordinate bbox min: `{geometry['bbox_min']}`
- source-coordinate bbox max: `{geometry['bbox_max']}`
- compiled scale: `0.03125`; physical radius: `0.00125 m`

The contact attributes are not tuned: self-collision, internal collision, friction,
condim, solver parameters, margin/gap, contype, and conaffinity are inherited exactly
from the original flex and verified from compiled arrays.
"""


def _report_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CPU 3D GMSH flex versus 2D OBJ flex",
        "",
        f"CPU MuJoCo: `{payload['mujoco_version']}`. Classification: "
        f"**{payload['decision']['classification']}**.",
        "",
        payload["decision"]["reason"],
        "",
        "| Fixture | Onset shift mm | Common contacts 3D/2D | Common touch total error | "
        "Common active Jaccard | Primary gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, fixture in payload["fixtures"].items():
        comparison = fixture["comparison"]
        contacts = comparison["common_reference_pose"]["contacts"]
        tactile = comparison["common_reference_pose"]["tactile"]
        total_error = tactile["total_magnitude_relative_error"]
        lines.append(
            f"| {name} | {comparison['onset_shift_mm']:.6f} | "
            f"{contacts['reference_count']}/{contacts['actual_count']} | "
            f"{total_error if total_error is not None else 'n/a'} | "
            f"{tactile['active_jaccard']:.6f} | "
            f"{'PASS' if all(fixture['primary_gate_results'].values()) else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"No-contact 1/10/100-step control: "
            f"**{'PASS' if payload['no_contact']['passed'] else 'FAIL'}**.",
            "",
            "The common pose is 0.25 mm inside the original 3D onset. The matched pose "
            "is 0.25 mm inside each representation's own onset. Element IDs are not "
            "compared because tetrahedra and triangles have different internal topology.",
            "",
            payload["decision"]["next_action"],
            "",
        ]
    )
    return "\n".join(lines)


def run_validation(
    *,
    reference_xml: Path,
    exterior_obj: Path,
    geometry_manifest: Path,
    features_path: Path,
    native_control_xml: Path,
    output: Path,
) -> dict[str, Any]:
    import mujoco

    output.mkdir(parents=True, exist_ok=True)
    models_dir = output / "models"
    surface_xml = models_dir / "surface_2d_obj_flex.xml"
    model_manifest = build_surface_flex_model(reference_xml, exterior_obj, surface_xml)
    conversion = json.loads(geometry_manifest.read_text(encoding="utf-8"))
    geometry_data = conversion["geometry"]
    models = {
        REPRESENTATIONS[0]: str(reference_xml.resolve()),
        REPRESENTATIONS[1]: str(surface_xml.resolve()),
        "native_mesh_control": str(native_control_xml.resolve()),
    }
    _write_json(models_dir / "models.json", models)

    audits = {
        name: _compiled_model_audit(mujoco, Path(path))
        for name, path in models.items()
    }
    reference_audit, surface_audit = (audits[name] for name in REPRESENTATIONS)
    preserved_fields = (
        "nq",
        "nv",
        "object_free_joint_count",
        "object_global_dof_count",
        "object_mass",
        "object_inertial_pos",
        "object_inertia",
        "object_dof_damping",
        "nflex",
        "flex_radius",
        "flex_rigid",
        "flex_contact",
    )
    model_gate = {
        field: reference_audit[field] == surface_audit[field] for field in preserved_fields
    }
    model_gate.update(
        {
            "surface_dim_is_2": surface_audit["flex_dim"] == [2],
            "surface_topology_matches_obj": (
                surface_audit["nflexvert"] == geometry_data["converted_vertex_count"]
                and surface_audit["nflexelem"] == geometry_data["triangle_count"]
            ),
            "surface_all_vertices_on_object": surface_audit[
                "all_flex_vertices_on_object_body"
            ],
        }
    )
    no_contact = run_no_contact_control(mujoco, reference_xml, surface_xml)
    fixtures = _run_fixture_validation(mujoco, reference_xml, surface_xml, features_path)
    primary_pass = all(
        all(fixture["primary_gate_results"].values()) for fixture in fixtures.values()
    )
    secondary_pass = all(
        all(fixture["secondary_gate_results"].values()) for fixture in fixtures.values()
    )
    if all(model_gate.values()) and no_contact["passed"] and primary_pass:
        classification = "CPU Outcome A — strong representation fidelity"
        proceed_to_warp = True
        reason = "All model, no-contact, and five-fixture primary CPU gates pass."
        next_action = "Proceed to the gated MuJoCo Warp 2D-flex validation phases."
    elif all(model_gate.values()) and no_contact["passed"] and secondary_pass:
        classification = "CPU Outcome B — mostly equivalent"
        proceed_to_warp = False
        reason = (
            "The candidate clears secondary diagnostics but not every primary CPU gate; "
            "Warp is not run automatically."
        )
        next_action = "Stop before Warp and assess whether the localized differences are defensible."
    else:
        classification = "CPU Outcome C — representation differs materially"
        proceed_to_warp = False
        reason = (
            "The 2D surface flex fails at least one required model, no-contact, or "
            "five-fixture CPU representation-fidelity gate."
        )
        next_action = "Stop. Do not run Warp, N=1000, performance benchmarks, or RL."

    payload = {
        "validator_version": VALIDATOR_VERSION,
        "mujoco_version": mujoco.__version__,
        "models": models,
        "model_builder_manifest": model_manifest,
        "model_audits": audits,
        "model_gate_results": model_gate,
        "geometry": {
            "source_gmsh": conversion["source_mesh"],
            "source_gmsh_sha256": conversion["source_hash"],
            "exterior_obj": str(exterior_obj.resolve()),
            "exterior_obj_sha256": conversion["converted_hash"],
            "vertex_count": geometry_data["converted_vertex_count"],
            "triangle_count": geometry_data["triangle_count"],
            "bbox_min": geometry_data["bbox_min"],
            "bbox_max": geometry_data["bbox_max"],
            "watertight": geometry_data["watertight"],
            "connected_components": geometry_data["connected_components"],
        },
        "no_contact": no_contact,
        "evaluation_penetration_m": EVALUATION_PENETRATION_M,
        "gates": {
            **FIDELITY_GATES,
            "primary_absolute_onset_shift_mm_max_exclusive": PRIMARY_ONSET_GATE_MM,
            "secondary_absolute_onset_shift_mm_max_exclusive": SECONDARY_ONSET_GATE_MM,
        },
        "fixtures": fixtures,
        "decision": {
            "classification": classification,
            "proceed_to_warp": proceed_to_warp,
            "all_primary_fixture_gates_pass": primary_pass,
            "all_secondary_fixture_gates_pass": secondary_pass,
            "reason": reason,
            "next_action": next_action,
        },
    }
    _write_json(output / "cpu_3d_vs_2d_model.json", {
        key: payload[key]
        for key in ("validator_version", "mujoco_version", "models", "model_builder_manifest", "model_audits", "model_gate_results", "geometry", "no_contact")
    })
    _write_json(output / "cpu_3d_vs_2d_results.json", payload)
    _write_onset_csv(output / "cpu_3d_vs_2d_onset.csv", fixtures)
    _write_contact_csv(output / "cpu_3d_vs_2d_contact.csv", fixtures)
    _write_tactile_csv(output / "cpu_3d_vs_2d_tactile.csv", fixtures)
    (output / "semantics.md").write_text(_semantics_markdown(payload), encoding="utf-8")
    (output / "cpu_representation_report.md").write_text(
        _report_markdown(payload), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[1]
    base = repository / "generated/convex_decomposition_validation"
    models = json.loads((base / "contact_models/models.json").read_text(encoding="utf-8"))
    parser.add_argument("--reference-xml", type=Path, default=Path(models["rigid_flex_reference"]))
    parser.add_argument(
        "--exterior-obj",
        type=Path,
        default=repository / "generated/rigid_mesh_cache/obj_size-large_ar-high_macro-high_rough-high-a22bb0a0de0df202.obj",
    )
    parser.add_argument(
        "--geometry-manifest",
        type=Path,
        default=repository / "generated/rigid_mesh_cache/obj_size-large_ar-high_macro-high_rough-high-a22bb0a0de0df202.conversion.json",
    )
    parser.add_argument("--features", type=Path, default=base / "contact_feature_definitions.json")
    parser.add_argument("--native-control-xml", type=Path, default=Path(models["single_hull"]))
    parser.add_argument(
        "--output", type=Path, default=repository / "generated/rigid_flex_2d_validation"
    )
    args = parser.parse_args()
    payload = run_validation(
        reference_xml=args.reference_xml.resolve(),
        exterior_obj=args.exterior_obj.resolve(),
        geometry_manifest=args.geometry_manifest.resolve(),
        features_path=args.features.resolve(),
        native_control_xml=args.native_control_xml.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
