"""Diagnostic-only CPU/MuJoCo-Warp comparisons for single-body rigid flexes.

MuJoCo Warp 3.11 writes a flex-edge Jacobian even when a rigid flex has no
edge-Jacobian storage (``nJfe == 0``).  That is an illegal CUDA access.  The
workaround in this module is deliberately narrow: for a validated flex whose
vertices all belong to one rigid body, edge lengths are invariant and relative
edge velocities are zero.  Production simulation continues to reject all flex
models; this module exists only to preserve and localize the old collision-path
baseline.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .model_loader import is_single_body_rigid_flex, load_project_model
from .parity import (
    _contact_comparison,
    _contact_snapshot,
    _object_qpos_address,
    _touch_site_names,
    error_metrics,
    tactile_metrics,
    zero_action_control,
)
from .sensors import build_sensor_layout


@dataclass(frozen=True)
class RigidFlexFixture:
    name: str
    object_position: tuple[float, float, float] | None
    object_linear_velocity: tuple[float, float, float]
    enabled_geom: str | None
    settle_steps: int
    rollout_steps: int
    purpose: str


RIGID_FLEX_FIXTURES = (
    RigidFlexFixture(
        "free_no_contact",
        (1.0, 0.87, 2.0),
        (0.01, -0.02, 0.03),
        None,
        0,
        20,
        "A: free rigid-body motion without contact",
    ),
    RigidFlexFixture(
        "single_geom_approach",
        (0.967, 0.783, 0.205),
        (0.0, 0.0, -0.05),
        "robot0:C_ffdistal",
        0,
        10,
        "B: one fingertip collision geom approaches from a contact-free pose",
    ),
    RigidFlexFixture(
        "isolated_fingertip",
        (0.967, 0.783, 0.2035),
        (0.0, 0.0, 0.0),
        "robot0:C_ffdistal",
        0,
        10,
        "C: isolated shallow first-finger distal contact",
    ),
    RigidFlexFixture(
        "isolated_palm",
        (0.94, 0.86, 0.184),
        (0.0, 0.0, 0.0),
        "robot0:C_palm0",
        0,
        10,
        "D: isolated shallow palm contact",
    ),
    RigidFlexFixture(
        "settled_contact",
        None,
        (0.0, 0.0, 0.0),
        None,
        200,
        1,
        "E: default object settled for 200 CPU steps, then matched-state comparison",
    ),
)


def _warp_bindings() -> tuple[Any, Any, Any]:
    import mujoco_warp as mjw
    import mujoco_warp._src.smooth as smooth
    import warp as wp

    return mjw, smooth, wp


# Warp kernels must be module-level so Warp can retrieve their source reliably.
try:
    import warp as _wp

    @_wp.kernel
    def _single_body_rigid_flex_edges(
        flex_edge: _wp.array(dtype=_wp.vec2i),
        flexvert_xpos: _wp.array2d(dtype=_wp.vec3),
        flexedge_length: _wp.array2d(dtype=float),
        flexedge_velocity: _wp.array2d(dtype=float),
    ):
        world_id, edge_id = _wp.tid()
        vertices = flex_edge[edge_id]
        delta = flexvert_xpos[world_id, vertices[1]] - flexvert_xpos[world_id, vertices[0]]
        flexedge_length[world_id, edge_id] = _wp.length(delta)
        flexedge_velocity[world_id, edge_id] = 0.0

except ImportError:  # CPU-only environments can still import the package/tests.
    _wp = None
    _single_body_rigid_flex_edges = None


@contextmanager
def rigid_flex_edge_workaround(model: Any) -> Iterator[None]:
    """Install the MJWarp 3.11 rigid-flex edge workaround for one diagnostic."""
    if not is_single_body_rigid_flex(model):
        raise ValueError("diagnostic workaround requires a single-body rigid flex")
    vertex_bodies = np.asarray(model.flex_vertbodyid, dtype=np.int64)
    if vertex_bodies.size == 0 or np.any(vertex_bodies < 0) or np.unique(vertex_bodies).size != 1:
        raise ValueError("diagnostic workaround requires every flex vertex on one rigid body")
    if not np.all(np.asarray(model.flex_rigid, dtype=bool)):
        raise ValueError("diagnostic workaround refuses non-rigid flexes")
    if int(model.nJfe) != 0:
        raise ValueError(f"expected the diagnosed zero-sized edge Jacobian, found nJfe={model.nJfe}")

    _, smooth, wp = _warp_bindings()
    if _single_body_rigid_flex_edges is None:
        raise RuntimeError("Warp is unavailable")
    original = smooth.flex

    def safe_flex(warp_model: Any, data: Any) -> None:
        wp.launch(
            smooth._flex_nodes,
            dim=(data.nworld, warp_model.nflexnode),
            inputs=[
                warp_model.nflex,
                warp_model.flex_nodeadr,
                warp_model.flex_nodenum,
                warp_model.flex_nodebodyid,
                warp_model.flex_node,
                warp_model.flex_centered,
                data.xpos,
                data.xmat,
            ],
            outputs=[data.flexnode_xpos],
        )
        wp.launch(
            smooth._flex_vertices,
            dim=(data.nworld, warp_model.nflexvert),
            inputs=[
                warp_model.nflex,
                warp_model.flex_interp,
                warp_model.flex_cellnum,
                warp_model.flex_nodeadr,
                warp_model.flex_vertadr,
                warp_model.flex_vertnum,
                warp_model.flex_vertbodyid,
                warp_model.flex_vert,
                warp_model.flex_vert0,
                warp_model.flex_centered,
                data.xpos,
                data.xmat,
                data.flexnode_xpos,
            ],
            outputs=[data.flexvert_xpos],
        )
        wp.launch(
            _single_body_rigid_flex_edges,
            dim=(data.nworld, warp_model.nflexedge),
            inputs=[warp_model.flex_edge, data.flexvert_xpos],
            outputs=[data.flexedge_length, data.flexedge_velocity],
        )
        wp.launch(
            smooth._flex_face_kinematics,
            dim=(data.nworld, warp_model.nflexface),
            inputs=[
                warp_model.flex_interp,
                warp_model.flex_cellnum,
                warp_model.flex_face_map,
                warp_model.flex_face,
                data.flexnode_xpos,
            ],
            outputs=[data.face_xpos, data.face_quat],
        )

    smooth.flex = safe_flex
    try:
        yield
    finally:
        smooth.flex = original


def _configure_fixture(mujoco: Any, model: Any, fixture: RigidFlexFixture) -> Any:
    if fixture.enabled_geom is not None:
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, fixture.enabled_geom)
        if geom_id < 0:
            raise ValueError(f"fixture geom {fixture.enabled_geom!r} is absent")
        model.geom_contype[geom_id] = 1

    data = mujoco.MjData(model)
    data.ctrl[:] = zero_action_control(model)
    object_qpos = _object_qpos_address(mujoco, model)
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
    object_qvel = int(model.jnt_dofadr[object_joint])
    if fixture.object_position is not None:
        data.qpos[object_qpos : object_qpos + 3] = fixture.object_position
        data.qpos[object_qpos + 3 : object_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[object_qvel : object_qvel + 3] = fixture.object_linear_velocity
    if fixture.settle_steps:
        for _ in range(fixture.settle_steps):
            mujoco.mj_step(model, data)
    else:
        mujoco.mj_forward(model, data)
    return data


def _state_snapshot(
    mujoco: Any,
    model: Any,
    cpu_data: Any,
    warp_data: Any,
    touch_indices: np.ndarray,
    touch_names: tuple[str, ...],
    touch_sites: tuple[str, ...],
) -> dict[str, Any]:
    cpu_contacts = _contact_snapshot(mujoco, model, cpu_data)
    warp_contacts = _contact_snapshot(mujoco, model, warp_data)
    cpu_touch = cpu_data.sensordata[touch_indices]
    warp_touch = warp_data.sensordata[touch_indices]
    object_qpos = _object_qpos_address(mujoco, model)
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object:joint")
    object_qvel = int(model.jnt_dofadr[object_joint])
    return {
        "qpos": error_metrics(cpu_data.qpos, warp_data.qpos).to_dict(),
        "qvel": error_metrics(cpu_data.qvel, warp_data.qvel).to_dict(),
        "object_pose": error_metrics(
            cpu_data.qpos[object_qpos : object_qpos + 7],
            warp_data.qpos[object_qpos : object_qpos + 7],
        ).to_dict(),
        "object_velocity": error_metrics(
            cpu_data.qvel[object_qvel : object_qvel + 6],
            warp_data.qvel[object_qvel : object_qvel + 6],
        ).to_dict(),
        "touch": tactile_metrics(
            cpu_touch,
            warp_touch,
            names=touch_names,
            sites=touch_sites,
        ),
        "contact_counts": {
            "cpu": int(cpu_data.ncon),
            "warp": int(warp_data.ncon),
            "stored_contact_details_per_backend": 128,
        },
        "contact_comparison": _contact_comparison(cpu_contacts, warp_contacts),
        "cpu_contacts": cpu_contacts,
        "warp_contacts": warp_contacts,
    }


def _meaningfully_diverged(snapshot: dict[str, Any]) -> bool:
    return bool(
        snapshot["qpos"]["max_abs"] > 1e-6
        or snapshot["qvel"]["max_abs"] > 1e-4
        or snapshot["touch"]["max_absolute_error"] > 1e-3
        or snapshot["contact_comparison"]["cpu_count"]
        != snapshot["contact_comparison"]["warp_count"]
    )


def compare_rigid_flex_fixture(
    xml_path: str | Path,
    fixture: RigidFlexFixture,
    *,
    contacts_per_world: int = 8192,
    constraints_per_world: int = 4096,
    experimental_tet_guard: bool = False,
) -> dict[str, Any]:
    """Compare one controlled old rigid-flex rollout and retain divergence evidence."""
    import mujoco

    mjw, _, wp = _warp_bindings()
    model, model_report = load_project_model(xml_path, reference_compat=True)
    if model_report.object_collision_representation != "rigid_flex":
        raise ValueError("controlled old-path diagnostic requires a compiled rigid flex")
    cpu_data = _configure_fixture(mujoco, model, fixture)
    layout = build_sensor_layout(model)
    touch_indices = np.asarray(layout.touch_data_indices, dtype=np.int64)
    touch_sites = _touch_site_names(mujoco, model, layout)

    initial_cpu_contact_count = int(cpu_data.ncon)
    first_cpu_contact_step = 0 if cpu_data.ncon else None
    first_warp_contact_step: int | None = None
    first_divergence_step: int | None = None
    retained: dict[str, Any] = {}

    if experimental_tet_guard:
        from .rigid_flex_root_cause import rigid_tet_internal_guard

        collision_guard = rigid_tet_internal_guard(model)
    else:
        collision_guard = nullcontext()

    with wp.ScopedDevice("cuda:0"), rigid_flex_edge_workaround(model), collision_guard:
        warp_model = mjw.put_model(model)
        warp_data = mjw.put_data(
            model,
            cpu_data,
            nworld=1,
            nconmax=contacts_per_world,
            njmax=constraints_per_world,
        )
        wp.synchronize()
        warp_host = mujoco.MjData(model)
        mjw.get_data_into(warp_host, model, warp_data, world_id=0)
        first_warp_contact_step = 0 if warp_host.ncon else None
        retained["initial"] = _state_snapshot(
            mujoco,
            model,
            cpu_data,
            warp_host,
            touch_indices,
            layout.touch_names,
            touch_sites,
        )

        for step in range(1, fixture.rollout_steps + 1):
            mujoco.mj_step(model, cpu_data)
            mjw.step(warp_model, warp_data)
            wp.synchronize()
            mjw.get_data_into(warp_host, model, warp_data, world_id=0)
            snapshot = _state_snapshot(
                mujoco,
                model,
                cpu_data,
                warp_host,
                touch_indices,
                layout.touch_names,
                touch_sites,
            )
            if first_cpu_contact_step is None and cpu_data.ncon:
                first_cpu_contact_step = step
                retained["first_cpu_contact"] = snapshot
            if first_warp_contact_step is None and warp_host.ncon:
                first_warp_contact_step = step
                retained["first_warp_contact"] = snapshot
            if first_divergence_step is None and _meaningfully_diverged(snapshot):
                first_divergence_step = step
                retained["first_meaningful_divergence"] = snapshot
            if step == fixture.rollout_steps:
                retained["final"] = snapshot

    return {
        "fixture": asdict(fixture),
        "model": model_report.to_dict(),
        "diagnostic_only_workaround": {
            "applied": True,
            "reason": "MJWarp 3.11 nJfe=0 illegal write for a single-body rigid flex",
            "production_backend_changed": False,
            "invariant": "rigid-body edge length; zero relative edge velocity",
        },
        "experimental_tet_internal_guard": experimental_tet_guard,
        "initial_cpu_contact_count": initial_cpu_contact_count,
        "first_cpu_contact_step": first_cpu_contact_step,
        "first_warp_contact_step": first_warp_contact_step,
        "first_meaningful_divergence_step": first_divergence_step,
        "thresholds": {
            "qpos_max_abs": 1e-6,
            "qvel_max_abs": 1e-4,
            "touch_max_abs": 1e-3,
            "contact_count_exact": True,
        },
        "snapshots": retained,
    }


def compare_all_rigid_flex_fixtures(
    xml_path: str | Path,
    *,
    experimental_tet_guard: bool = False,
) -> dict[str, Any]:
    return {
        fixture.name: compare_rigid_flex_fixture(
            xml_path,
            fixture,
            experimental_tet_guard=experimental_tet_guard,
        )
        for fixture in RIGID_FLEX_FIXTURES
    }
