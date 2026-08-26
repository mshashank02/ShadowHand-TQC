"""Diagnostic-only MuJoCo Warp validation for the CPU-approved 2D rigid flex."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
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
    _evaluate_feature_fixture,
    _evaluate_hand_fixture,
    _find_onset,
    _set_static_object,
    _tactile_snapshot,
    _temporary_probe_model,
    _touch_vector,
    _zero_action_control,
    compare_contacts,
    compare_tactile,
)
from .rigid_flex_diagnostic import rigid_flex_edge_workaround
from .rigid_flex_root_cause import ProbeScenario, topology_snapshot
from .model_loader import _apply_reference_collision_defaults


def _mujoco_311_compatible_xml(source: str | Path) -> Path:
    """Remove only MuJoCo 3.11's obsolete, non-physical ``apirate`` attribute."""
    source = Path(source).resolve()
    tree = ET.parse(source)
    option = tree.getroot().find("option")
    if option is None or "apirate" not in option.attrib:
        return source
    option.attrib.pop("apirate")
    output = source.with_name(f"{source.stem}_mujoco311{source.suffix}")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def _surface_probe_scenarios(model: Any, probe_radius: float) -> list[ProbeScenario]:
    """Choose the same deterministic well-conditioned face logic for a 2D flex."""
    vertices = np.asarray(model.flex_vert, dtype=np.float64)
    triangles = vertices[np.asarray(model.flex_elem, dtype=np.int64).reshape(-1, 3)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    center = vertices.mean(axis=0)
    centroids = triangles.mean(axis=1)
    index = int(np.argmax(area2 * (1.0 + np.linalg.norm(centroids - center, axis=1))))
    if area2[index] <= 0.0:
        raise ValueError("compiled 2D flex contains no nondegenerate triangle")
    normal = cross[index] / area2[index]
    if np.dot(normal, centroids[index] - center) < 0.0:
        normal = -normal
    tangent = triangles[index, 1] - triangles[index, 0]
    tangent /= np.linalg.norm(tangent)
    radius = float(model.flex_radius[0])
    offsets = (
        ("flex_only_no_probe_contact", 0.020, 0.0),
        ("approach_2mm", 0.002, 0.0),
        ("approach_0p5mm", 0.0005, 0.0),
        ("approach_0p1mm", 0.0001, 0.0),
        ("shell_onset", 0.0, 0.0),
        ("penetration_0p1mm", -0.0001, 0.0),
        ("penetration_0p25mm", -0.00025, 0.0),
        ("shallow_penetration", -0.0005, 0.0),
        ("penetration_1mm", -0.001, 0.0),
        ("deep_penetration", -0.002, 0.0),
        ("sliding_tangent_positive", -0.0005, 0.003),
        ("sliding_tangent_negative", -0.0005, -0.003),
    )
    return [
        ProbeScenario(
            name=name,
            probe_position=tuple(
                float(value)
                for value in centroids[index]
                + normal * (probe_radius + radius + offset)
                + tangent * tangent_offset
            ),
            signed_shell_offset=float(offset),
        )
        for name, offset, tangent_offset in offsets
    ]


def write_minimal_surface_reproducer(
    exterior_obj: str | Path,
    output_dir: str | Path,
    *,
    probe_radius: float = 0.006,
) -> dict[str, Any]:
    """Create a small exact-OBJ 2D rigid-flex/sphere model and approach states."""
    import mujoco

    exterior_obj = Path(exterior_obj).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    asset = output / exterior_obj.name
    if asset.resolve() != exterior_obj:
        shutil.copyfile(exterior_obj, asset)
    xml = output / "rigid_flex_2d_sphere_probe.xml"
    xml.write_text(
        f'''<mujoco model="rigid_flex_2d_sphere_probe">
  <option timestep="0.002" gravity="0 0 0" iterations="20">
    <flag warmstart="enable"/>
  </option>
  <default>
    <geom contype="1" conaffinity="1" condim="3" friction="1 0.005 0.0001"
          solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2" margin="0" gap="0"/>
  </default>
  <worldbody>
    <body name="object">
      <joint name="object:joint" type="free" damping="0.05"/>
      <inertial pos="0 0 0" mass="0.976562"
                diaginertia="0.00305176 0.00305176 0.00305176"/>
      <flexcomp name="soft" type="mesh" file="{asset.name}" dim="2"
                scale="0.03125 0.03125 0.03125" radius="0.00125" rigid="true">
        <contact contype="1" conaffinity="1" condim="3" friction="1 0.005 0.0001"
                 solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2"
                 margin="0" gap="0" selfcollide="none" internal="false"/>
      </flexcomp>
    </body>
    <body name="probe" mocap="true">
      <geom name="probe_sphere" type="sphere" size="{probe_radius:.9g}"/>
      <site name="probe_touch_site" type="sphere" size="{probe_radius:.9g}"/>
    </body>
  </worldbody>
  <sensor><touch name="probe_touch" site="probe_touch_site"/></sensor>
</mujoco>
''',
        encoding="utf-8",
    )
    model = mujoco.MjModel.from_xml_path(str(xml))
    if int(model.nv) != 6 or np.asarray(model.flex_dim).tolist() != [2]:
        raise ValueError("minimal surface model did not compile as one rigid 2D flex")
    scenarios = _surface_probe_scenarios(model, probe_radius)
    states = output / "probe_states.json"
    states.write_text(
        json.dumps(
            {
                "probe_radius": probe_radius,
                "scenarios": [
                    {
                        "name": scenario.name,
                        "probe_position": scenario.probe_position,
                        "signed_shell_offset": scenario.signed_shell_offset,
                    }
                    for scenario in scenarios
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    scenario_by_name = {scenario.name: scenario for scenario in scenarios}
    outside = np.asarray(
        scenario_by_name["approach_0p1mm"].probe_position, dtype=np.float64
    )
    inside = np.asarray(
        scenario_by_name["penetration_0p1mm"].probe_position, dtype=np.float64
    )
    onset_position = 0.5 * (outside + inside)
    normal = (outside - inside) / 0.0002
    onset_scenarios = []
    for level in range(21):
        magnitude = 0.0001 / (2**level)
        for side, signed_offset in (("outside", magnitude), ("inside", -magnitude)):
            onset_scenarios.append(
                {
                    "name": f"onset_{side}_{level:02d}",
                    "probe_position": (onset_position + normal * signed_offset).tolist(),
                    "signed_shell_offset": signed_offset,
                }
            )
    onset_scenarios.append(
        {
            "name": "onset_exact_plane",
            "probe_position": onset_position.tolist(),
            "signed_shell_offset": 0.0,
        }
    )
    onset_states = output / "probe_onset_states.json"
    onset_states.write_text(
        json.dumps(
            {"probe_radius": probe_radius, "scenarios": onset_scenarios},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "xml": str(xml),
        "states": str(states),
        "onset_states": str(onset_states),
        "asset": str(asset),
        "topology": topology_snapshot(model),
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
    contacts_per_world: int = 8192,
    constraints_per_world: int = 8192,
) -> Any:
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
    return host


def _hand_seed(
    mujoco: Any,
    model: Any,
    fixture: dict[str, Any],
    target_id: int,
    coordinate: float,
) -> Any:
    data = mujoco.MjData(model)
    data.ctrl[:] = _zero_action_control(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
    _set_static_object(
        data,
        int(model.jnt_qposadr[joint]),
        int(model.jnt_dofadr[joint]),
        (*fixture["object_xy_m"], coordinate),
    )
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
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
    qadr, dadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    object_position = np.asarray((0.0, 0.0, 0.5), dtype=np.float64)
    _set_static_object(data, qadr, dadr, object_position)
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
) -> dict[str, Any]:
    contacts = _contact_records(mujoco, model, data, relevant)
    values, records = _touch_vector(
        mujoco, model, data, diagnostic_probe=diagnostic_probe
    )
    return {
        "coordinate_m": float(coordinate),
        "contacts": contacts,
        "constraint_rows": int(sum(contact["constraint_rows"] for contact in contacts)),
        "tactile": _tactile_snapshot(values, records),
        "tactile_values": values.tolist(),
        "tactile_records": records,
    }


def _compare_states(reference: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    return {
        "contacts": compare_contacts(reference["contacts"], actual["contacts"]),
        "tactile": compare_tactile(
            np.asarray(reference["tactile_values"], dtype=np.float64),
            np.asarray(actual["tactile_values"], dtype=np.float64),
            reference["tactile_records"],
            actual["tactile_records"],
        ),
    }


def _warp_onset(
    evaluate: Callable[[float], dict[str, Any]], outside: float, inside: float
) -> tuple[float, dict[str, Any]]:
    return _find_onset(evaluate, outside, inside, iterations=28)


def compare_five_fixtures_cpu_warp(
    surface_xml: str | Path,
    features_path: str | Path,
    *,
    reference_compat: bool = False,
) -> dict[str, Any]:
    """Compare static CPU and Warp contact/tactile states on the five CPU fixtures."""
    import mujoco

    mjw, wp = _warp_bindings()
    surface_xml = _mujoco_311_compatible_xml(surface_xml)
    features = json.loads(Path(features_path).read_text(encoding="utf-8"))["features"]
    fixtures: dict[str, Any] = {}
    with wp.ScopedDevice("cuda:0"):
        for fixture_name, fixture in HAND_FIXTURES.items():
            model = mujoco.MjModel.from_xml_path(str(surface_xml))
            compatibility_changes = (
                _apply_reference_collision_defaults(mujoco, model)
                if reference_compat
                else []
            )
            target_id, _ = _configure_collision_geoms(
                mujoco, model, target_geom_name=fixture["target_geom"]
            )
            assert target_id is not None

            def relevant(contact: Any, target: int = target_id) -> bool:
                return target in [int(value) for value in np.asarray(contact.geom)] and any(
                    int(value) >= 0 for value in np.asarray(contact.flex)
                )

            with rigid_flex_edge_workaround(model):
                warp_model = mjw.put_model(model)

                def cpu_evaluate(coordinate: float) -> dict[str, Any]:
                    seed = _hand_seed(mujoco, model, fixture, target_id, coordinate)
                    return _state_from_data(
                        mujoco, model, seed, relevant,
                        diagnostic_probe=False, coordinate=coordinate
                    )

                def warp_evaluate(coordinate: float) -> dict[str, Any]:
                    seed = _hand_seed(mujoco, model, fixture, target_id, coordinate)
                    host = _warp_forward_host(mujoco, mjw, wp, model, warp_model, seed)
                    return _state_from_data(
                        mujoco, model, host, relevant,
                        diagnostic_probe=False, coordinate=coordinate
                    )

                cpu_onset, _ = _warp_onset(
                    cpu_evaluate,
                    fixture["outside_coordinate_m"], fixture["inside_coordinate_m"]
                )
                warp_onset, _ = _warp_onset(
                    warp_evaluate,
                    fixture["outside_coordinate_m"], fixture["inside_coordinate_m"]
                )
                common_coordinate = cpu_onset - EVALUATION_PENETRATION_M
                cpu_common = cpu_evaluate(common_coordinate)
                warp_common = warp_evaluate(common_coordinate)
                cpu_matched = cpu_evaluate(cpu_onset - EVALUATION_PENETRATION_M)
                warp_matched = warp_evaluate(warp_onset - EVALUATION_PENETRATION_M)
            fixtures[fixture_name] = {
                "fixture_type": "n500_hand",
                "cpu_onset_coordinate_m": cpu_onset,
                "warp_onset_coordinate_m": warp_onset,
                "onset_shift_mm": (warp_onset - cpu_onset) * 1000.0,
                "common_pose": {
                    "cpu": cpu_common,
                    "warp": warp_common,
                    "comparison": _compare_states(cpu_common, warp_common),
                },
                "matched_penetration": {
                    "cpu": cpu_matched,
                    "warp": warp_matched,
                    "comparison": _compare_states(cpu_matched, warp_matched),
                },
            }

        for fixture_name in FEATURE_NAMES:
            model, temporary_path = _temporary_probe_model(mujoco, Path(surface_xml))
            try:
                if reference_compat:
                    _apply_reference_collision_defaults(mujoco, model)
                _, probe_id = _configure_collision_geoms(
                    mujoco, model, probe_geom_name=PROBE_GEOM_NAME
                )
                assert probe_id is not None

                def relevant(contact: Any, probe: int = probe_id) -> bool:
                    return probe in [int(value) for value in np.asarray(contact.geom)]

                feature = features[fixture_name]
                with rigid_flex_edge_workaround(model):
                    warp_model = mjw.put_model(model)

                    def cpu_evaluate(coordinate: float) -> dict[str, Any]:
                        seed = _feature_seed(mujoco, model, feature, probe_id, coordinate)
                        return _state_from_data(
                            mujoco, model, seed, relevant,
                            diagnostic_probe=True, coordinate=coordinate
                        )

                    def warp_evaluate(coordinate: float) -> dict[str, Any]:
                        seed = _feature_seed(mujoco, model, feature, probe_id, coordinate)
                        host = _warp_forward_host(mujoco, mjw, wp, model, warp_model, seed)
                        return _state_from_data(
                            mujoco, model, host, relevant,
                            diagnostic_probe=True, coordinate=coordinate
                        )

                    cpu_onset, _ = _warp_onset(cpu_evaluate, 0.006, -0.003)
                    warp_onset, _ = _warp_onset(warp_evaluate, 0.006, -0.003)
                    common_coordinate = cpu_onset - EVALUATION_PENETRATION_M
                    cpu_common = cpu_evaluate(common_coordinate)
                    warp_common = warp_evaluate(common_coordinate)
                    cpu_matched = cpu_evaluate(cpu_onset - EVALUATION_PENETRATION_M)
                    warp_matched = warp_evaluate(warp_onset - EVALUATION_PENETRATION_M)
                fixtures[fixture_name] = {
                    "fixture_type": "diagnostic_probe",
                    "cpu_onset_coordinate_m": cpu_onset,
                    "warp_onset_coordinate_m": warp_onset,
                    "onset_shift_mm": (warp_onset - cpu_onset) * 1000.0,
                    "common_pose": {
                        "cpu": cpu_common,
                        "warp": warp_common,
                        "comparison": _compare_states(cpu_common, warp_common),
                    },
                    "matched_penetration": {
                        "cpu": cpu_matched,
                        "warp": warp_matched,
                        "comparison": _compare_states(cpu_matched, warp_matched),
                    },
                }
            finally:
                temporary_path.unlink(missing_ok=True)
    return {
        "mujoco_version": mujoco.__version__,
        "mujoco_warp_version": mjw.__version__,
        "warp_version": wp.__version__,
        "surface_xml": str(Path(surface_xml).resolve()),
        "reference_compat": reference_compat,
        "compatibility_changes": compatibility_changes,
        "fixtures": fixtures,
    }
