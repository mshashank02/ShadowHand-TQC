"""Focused CPU MuJoCo versus MuJoCo Warp rigid-flex collision diagnostics.

The experimental guard in this module is intentionally narrower than the normal
GPU backend: it only applies to single-body rigid 3-D flexes whose compiled model
disables both internal and self collision.  It exists to test one localized
MuJoCo Warp 3.11 collision-kernel hypothesis and is not a production fallback.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterator, Sequence

import numpy as np

from .model_loader import is_single_body_rigid_flex
from .rigid_flex_diagnostic import rigid_flex_edge_workaround


@dataclass(frozen=True)
class ProbeScenario:
    name: str
    probe_position: tuple[float, float, float]
    signed_shell_offset: float


try:
    import warp as _wp

    @_wp.kernel(module="unique", enable_backward=False)
    def _skip_tet_internal_contacts(
        nflex: int,
        flex_dim: _wp.array[int],
        flex_vertadr: _wp.array[int],
        flex_elemadr: _wp.array[int],
        flex_elemnum: _wp.array[int],
        flex_elemdataadr: _wp.array[int],
        flex_elem: _wp.array[int],
        flex_radius: _wp.array[float],
        flex_elemflexid: _wp.array[int],
        flexvert_xpos_in: _wp.array2d[_wp.vec3],
        max_candidates: int,
        overflow_out: _wp.array[int],
        cand_dist_out: _wp.array[float],
        cand_pos_out: _wp.array[_wp.vec3],
        cand_nrm_out: _wp.array[_wp.vec3],
        cand_geom_out: _wp.array[_wp.vec2i],
        cand_flex_out: _wp.array[_wp.vec2i],
        cand_elem_out: _wp.array[_wp.vec2i],
        cand_vert_out: _wp.array[_wp.vec2i],
        cand_worldid_out: _wp.array[int],
        cand_type_out: _wp.array[int],
        cand_geomcollisionid_out: _wp.array[int],
        ncand_out: _wp.array[int],
    ):
        # CPU MuJoCo skips this whole path for rigid flexes.  The no-op kernel is
        # installed only by rigid_tet_internal_guard after validating that case.
        return

except ImportError:  # CPU-only reference environments can still run the collector.
    _wp = None
    _skip_tet_internal_contacts = None


def _sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def topology_snapshot(model: Any) -> dict[str, Any]:
    """Return stable compiled-structure evidence for one flex model."""
    fields = ("flex_vert", "flex_edge", "flex_elem", "flex_shell", "flex_vertbodyid")
    return {
        "counts": {
            name: int(getattr(model, name))
            for name in ("nq", "nv", "nbody", "ngeom", "nflex", "nflexvert", "nflexedge", "nflexelem")
        },
        "hashes": {name: _sha256_array(getattr(model, name)) for name in fields},
        "flex": {
            "dim": np.asarray(model.flex_dim, dtype=int).tolist(),
            "rigid": np.asarray(model.flex_rigid, dtype=bool).tolist(),
            "internal": np.asarray(model.flex_internal, dtype=bool).tolist(),
            "selfcollide": np.asarray(model.flex_selfcollide, dtype=int).tolist(),
            "radius": np.asarray(model.flex_radius, dtype=float).tolist(),
            "contype": np.asarray(model.flex_contype, dtype=int).tolist(),
            "conaffinity": np.asarray(model.flex_conaffinity, dtype=int).tolist(),
        },
    }


def validate_guard_model(model: Any) -> None:
    """Refuse to broaden the experimental guard beyond the proven CPU branch."""
    if not is_single_body_rigid_flex(model):
        raise ValueError("tet-internal guard requires a single-body rigid flex")
    if not np.all(np.asarray(model.flex_rigid, dtype=bool)):
        raise ValueError("tet-internal guard refuses deformable flexes")
    if not np.all(np.asarray(model.flex_dim, dtype=int) == 3):
        raise ValueError("tet-internal guard requires 3-D tetrahedral flexes")
    if np.any(np.asarray(model.flex_internal, dtype=bool)):
        raise ValueError("tet-internal guard refuses internal=true")
    if np.any(np.asarray(model.flex_selfcollide, dtype=int) != 0):
        raise ValueError("tet-internal guard refuses enabled self collision")


@contextmanager
def rigid_tet_internal_guard(model: Any) -> Iterator[None]:
    """Skip only Warp's unconditional tetrahedron-internal candidate kernel."""
    validate_guard_model(model)
    if _skip_tet_internal_contacts is None:
        raise RuntimeError("Warp is unavailable")
    from mujoco_warp._src import collision_flex

    if not hasattr(collision_flex, "_flex_tet_internal_collisions_detect"):
        raise RuntimeError("this MuJoCo Warp build no longer contains the diagnosed kernel")
    original = collision_flex._flex_tet_internal_collisions_detect
    collision_flex._flex_tet_internal_collisions_detect = _skip_tet_internal_contacts
    try:
        yield
    finally:
        collision_flex._flex_tet_internal_collisions_detect = original


def _shell_probe_frame(model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(model.flex_vert, dtype=float)
    shell = np.asarray(model.flex_shell, dtype=np.int64).reshape(-1, 3)
    triangles = vertices[shell]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    if not np.any(area2 > 0):
        raise ValueError("compiled flex shell contains no nondegenerate triangle")
    center = vertices.mean(axis=0)
    centroids = triangles.mean(axis=1)
    score = area2 * (1.0 + np.linalg.norm(centroids - center, axis=1))
    index = int(np.argmax(score))
    normal = cross[index] / area2[index]
    if np.dot(normal, centroids[index] - center) < 0:
        normal = -normal
    tangent = triangles[index, 1] - triangles[index, 0]
    tangent /= np.linalg.norm(tangent)
    return centroids[index], normal, tangent


def build_probe_scenarios(model: Any, probe_radius: float) -> list[ProbeScenario]:
    """Place a sphere along one well-conditioned exterior-shell face normal."""
    surface, normal, tangent = _shell_probe_frame(model)
    flex_radius = float(np.asarray(model.flex_radius)[0])
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
    scenarios: list[ProbeScenario] = []
    for name, signed_offset, tangent_offset in offsets:
        position = surface + normal * (probe_radius + flex_radius + signed_offset) + tangent * tangent_offset
        scenarios.append(
            ProbeScenario(
                name=name,
                probe_position=tuple(float(value) for value in position),
                signed_shell_offset=float(signed_offset),
            )
        )
    return scenarios


def write_minimal_reproducer(
    source_model: Any,
    source_msh: str | Path,
    output_dir: str | Path,
    *,
    probe_radius: float = 0.006,
) -> dict[str, Any]:
    """Write a tiny rigid-flex/sphere MJCF using the exact original Gmsh asset."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(source_msh).resolve()
    asset = output / source.name
    if source != asset.resolve():
        shutil.copyfile(source, asset)

    xml = output / "rigid_flex_sphere_probe.xml"
    xml.write_text(
        f'''<mujoco model="rigid_flex_sphere_probe">
  <compiler angle="radian" coordinate="local"/>
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
      <flexcomp name="soft" type="gmsh" file="{asset.name}" dim="3" dof="trilinear"
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
    scenarios = build_probe_scenarios(source_model, probe_radius)
    state_path = output / "probe_states.json"
    payload = {
        "source_msh": str(source),
        "source_msh_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "probe_radius": probe_radius,
        "topology": topology_snapshot(source_model),
        "scenarios": [scenario.__dict__ for scenario in scenarios],
    }
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"xml": str(xml), "states": str(state_path), **payload}


def _contact_records(mujoco: Any, model: Any, data: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    force = np.zeros(6, dtype=float)
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        mujoco.mj_contactForce(model, data, index, force)
        records.append(
            {
                "index": index,
                "dist": float(contact.dist),
                "pos": np.asarray(contact.pos, dtype=float).tolist(),
                "normal": np.asarray(contact.frame, dtype=float)[:3].tolist(),
                "geom": np.asarray(contact.geom, dtype=int).tolist(),
                "flex": np.asarray(getattr(contact, "flex", (-1, -1)), dtype=int).tolist(),
                "elem": np.asarray(getattr(contact, "elem", (-1, -1)), dtype=int).tolist(),
                "vert": np.asarray(getattr(contact, "vert", (-1, -1)), dtype=int).tolist(),
                "dim": int(contact.dim),
                "force": force.copy().tolist(),
            }
        )
    return records


def _contact_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    internal = [row for row in records if row["geom"] == [-1, -1] and row["flex"][0] >= 0]
    external = [row for row in records if row not in internal]
    distances = np.asarray([row["dist"] for row in records], dtype=float)
    return {
        "count": len(records),
        "flex_internal_count": len(internal),
        "external_count": len(external),
        "minimum_distance": float(distances.min()) if distances.size else None,
        "maximum_distance": float(distances.max()) if distances.size else None,
    }


def collect_cpu_matrix(xml_path: str | Path, states_path: str | Path) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    states = json.loads(Path(states_path).read_text(encoding="utf-8"))
    scenarios: dict[str, Any] = {}
    for scenario in states["scenarios"]:
        data = mujoco.MjData(model)
        data.mocap_pos[0] = scenario["probe_position"]
        mujoco.mj_forward(model, data)
        contacts = _contact_records(mujoco, model, data)
        scenarios[scenario["name"]] = {
            "probe_position": scenario["probe_position"],
            "signed_shell_offset": scenario["signed_shell_offset"],
            "contact": _contact_summary(contacts),
            "contacts": contacts,
            "constraint_rows": int(data.nefc),
            "touch": np.asarray(data.sensordata, dtype=float).tolist(),
            "flexvert_xpos_sha256": _sha256_array(data.flexvert_xpos),
            "flexvert_xpos_max_abs_error_vs_compiled": float(
                np.max(np.abs(np.asarray(data.flexvert_xpos) - np.asarray(model.flex_vert)))
            ),
            "flexvert_xpos_bbox": {
                "min": np.min(data.flexvert_xpos, axis=0).tolist(),
                "max": np.max(data.flexvert_xpos, axis=0).tolist(),
            },
        }
    return {
        "backend": "cpu_mujoco",
        "mujoco_version": mujoco.__version__,
        "xml": str(Path(xml_path).resolve()),
        "topology": topology_snapshot(model),
        "scenarios": scenarios,
    }


def collect_warp_matrix(
    xml_path: str | Path,
    states_path: str | Path,
    *,
    apply_tet_guard: bool,
    contacts_per_world: int = 40000,
    constraints_per_world: int = 40000,
) -> dict[str, Any]:
    import mujoco
    import mujoco_warp as mjw
    from mujoco_warp._src import collision_flex

    if _wp is None:
        raise RuntimeError("Warp is unavailable")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    states = json.loads(Path(states_path).read_text(encoding="utf-8"))
    guard = rigid_tet_internal_guard(model) if apply_tet_guard else nullcontext()
    scenarios: dict[str, Any] = {}
    original_kernel = getattr(collision_flex, "_flex_tet_internal_collisions_detect", None)
    with _wp.ScopedDevice("cuda:0"), rigid_flex_edge_workaround(model), guard:
        warp_model = mjw.put_model(model)
        transferred = {}
        for name in (
            "flex_dim",
            "flex_internal",
            "flex_selfcollide",
            "flex_radius",
            "flex_contype",
            "flex_conaffinity",
        ):
            value = getattr(warp_model, name, None)
            transferred[name] = None if value is None else np.asarray(value.numpy()).tolist()
        transferred["has_flex_selfcollide"] = bool(warp_model.has_flex_selfcollide)
        transferred_topology_hashes = {
            name: _sha256_array(getattr(warp_model, name).numpy())
            for name in ("flex_edge", "flex_elem", "flex_shell", "flex_vertbodyid")
        }
        transferred_flex_vert = np.asarray(warp_model.flex_vert.numpy())
        transferred_flex_vert_max_abs_error_m = float(
            np.max(np.abs(transferred_flex_vert - np.asarray(model.flex_vert)))
        )
        for scenario in states["scenarios"]:
            seed_data = mujoco.MjData(model)
            seed_data.mocap_pos[0] = scenario["probe_position"]
            warp_data = mjw.put_data(
                model,
                seed_data,
                nworld=1,
                nconmax=contacts_per_world,
                njmax=constraints_per_world,
            )
            mjw.forward(warp_model, warp_data)
            _wp.synchronize()
            host = mujoco.MjData(model)
            mjw.get_data_into(host, model, warp_data, world_id=0)
            contacts = _contact_records(mujoco, model, host)
            flexvert = np.asarray(warp_data.flexvert_xpos.numpy())[0]
            scenarios[scenario["name"]] = {
                "probe_position": scenario["probe_position"],
                "signed_shell_offset": scenario["signed_shell_offset"],
                "contact": _contact_summary(contacts),
                "contacts": contacts,
                "constraint_rows": int(host.nefc),
                "touch": np.asarray(host.sensordata, dtype=float).tolist(),
                "flexvert_xpos_sha256": _sha256_array(flexvert),
                "flexvert_xpos_max_abs_error_vs_compiled": float(
                    np.max(np.abs(flexvert - np.asarray(model.flex_vert)))
                ),
                "flexvert_xpos_bbox": {
                    "min": np.min(flexvert, axis=0).tolist(),
                    "max": np.max(flexvert, axis=0).tolist(),
                },
                "warp_flex_aabb": {
                    "min": np.asarray(warp_data.flex_aabb_min.numpy())[0, 0].tolist(),
                    "max": np.asarray(warp_data.flex_aabb_max.numpy())[0, 0].tolist(),
                },
            }
    return {
        "backend": "mujoco_warp",
        "mujoco_version": mujoco.__version__,
        "mujoco_warp_version": mjw.__version__,
        "warp_version": _wp.__version__,
        "xml": str(Path(xml_path).resolve()),
        "experimental_tet_guard": apply_tet_guard,
        "guard_replaced_exact_kernel": (
            apply_tet_guard and original_kernel is not None and original_kernel is not _skip_tet_internal_contacts
        ),
        "diagnosed_kernel_present": original_kernel is not None,
        "transferred": transferred,
        "transferred_topology_hashes": transferred_topology_hashes,
        "transferred_flex_vert_max_abs_error_m": transferred_flex_vert_max_abs_error_m,
        "topology": topology_snapshot(model),
        "scenarios": scenarios,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
