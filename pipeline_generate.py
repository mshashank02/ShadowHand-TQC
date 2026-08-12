#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Tuple
import os, sys, math, argparse, re, shutil, json, struct
import xml.etree.ElementTree as ET
from collections import defaultdict, namedtuple
from pathlib import Path

from object_conversion import convert_gmsh_to_rigid_surface, scaled_geometry_metrics

# =========================
# util: formatting + paths
# =========================

def _fmt_num(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"

def make_output_names(out_dir: str, Ntotal: int, Rppx: float, Rpt: float) -> Tuple[str, str]:
    """Return (shared_path, robot_path) obeying the required naming scheme."""
    r1 = _fmt_num(Rppx); r2 = _fmt_num(Rpt)
    shared = f"shared_touch_sensors_{Ntotal}_{r1}_{r2}.xml"
    robot  = f"Sensors_withPos_{Ntotal}_{r1}_{r2}.xml"
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, shared), os.path.join(out_dir, robot)

def make_candidate_paths(out_root: str, task: str, Ntotal: int, Rppx: float, Rpt: float) -> Dict[str, str]:
    # file tag (unchanged in filenames)
    tag = f"{Ntotal}_{_fmt_num(Rppx)}_{_fmt_num(Rpt)}"
    # directory tag (now includes the task)
    dir_tag = f"{task}_{tag}"

    cand_dir = os.path.join(out_root, dir_tag)
    os.makedirs(cand_dir, exist_ok=True)

    env_base = f"manipulate_{task}_touch_sensors_{tag}.xml"

    return {
        "dir": cand_dir,                                   # e.g., generated/block_90_1_1
        "shared": os.path.join(cand_dir, f"shared_touch_sensors_{tag}.xml"),
        "robot":  os.path.join(cand_dir, f"Sensors_withPos_{tag}.xml"),
        "env":    os.path.join(cand_dir, env_base),
        "env_basename": env_base,
        "tag":    tag,                                     # keep old tag for filenames/metadata
        "dir_tag": dir_tag,                                # new: task-prefixed directory tag
    }


def _sanitize_task_label(s: str) -> str:
    s = os.path.splitext(os.path.basename(s))[0]
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or "custom"


SIZE_SCALE_MULTIPLIERS = {
    "small": 0.75,
    "medium": 1.0,
    "large": 1.25,
}

BASE_RIGID_MASS = 0.5
BASE_DEFORMABLE_MASS = 0.5
BASE_RIGID_DIAGINERTIA = (1e-3, 1e-3, 1e-3)
BASE_DEFORMABLE_DIAGINERTIA = (1e-3, 1e-3, 1e-3)

SIZE_SPAWN_HEIGHTS = {
    "rigid": {
        "small": 0.36,
        "medium": 0.40,
        "large": 0.46,
    },
    "deformable": {
        "small": 0.15,
        "medium": 0.17,
        "large": 0.20,
    },
}

RIGID_MESH_ASSET_NAME = "custom_object_mesh"
RIGID_GEOM_CONTACT_ATTRIBUTES = {
    "friction": "1 0.005 0.0001",
    "condim": "3",
    "solref": "0.02 1",
    "solimp": "0.9 0.95 0.001 0.5 2",
    "margin": "0",
    "gap": "0",
    "contype": "1",
    "conaffinity": "1",
    "priority": "0",
}


DEFORMABLE_PRESETS: Dict[str, Dict[str, object]] = {
    "debug_soft": {
        "young": 5.0e4,
        "poisson": 0.35,
        "damping": 0.02,
        "mass": 0.35,
        "flex_radius": 0.0012,
        "joint_damping": 0.8,
        "option": {
            "timestep": "0.00004",
            "integrator": "implicitfast",
            "solver": "CG",
            "tolerance": "1e-9",
            "iterations": "150",
            "impratio": "1",
            "apirate": "200",
        },
        "contact": {
            "selfcollide": "none",
            "internal": "false",
            "condim": "1",
            "friction": "0.5 0.002 0.0001",
            "contype": "0",
            "conaffinity": "1",
            "solref": "0.09 1",
            "solimp": "0.55 0.8 0.005",
            "margin": "0.001",
        },
        "safe_spawn_heights": {"small": 0.18, "medium": 0.22, "large": 0.27},
        "action_scale": 0.55,
        "action_clip": 0.75,
        "action_smoothing": 0.65,
        "reset_settle_steps": 12,
        "nconmax": 10000,
        "nstack": 8000000,
    },
    "soft_rubber_stable": {
        "young": 5.0e5,
        "poisson": 0.38,
        "damping": 0.02,
        "mass": 0.50,
        "flex_radius": 0.0010,
        "joint_damping": 1.0,
        "option": {
            "timestep": "0.00002",
            "integrator": "implicitfast",
            "solver": "CG",
            "tolerance": "1e-10",
            "iterations": "200",
            "impratio": "1",
            "apirate": "200",
        },
        "contact": {
            "selfcollide": "none",
            "internal": "false",
            "condim": "1",
            "friction": "0.6 0.002 0.0001",
            "contype": "0",
            "conaffinity": "1",
            "solref": "0.08 1",
            "solimp": "0.6 0.82 0.005",
            "margin": "0.001",
        },
        "safe_spawn_heights": {"small": 0.18, "medium": 0.22, "large": 0.27},
        "action_scale": 0.60,
        "action_clip": 0.80,
        "action_smoothing": 0.55,
        "reset_settle_steps": 10,
        "nconmax": 10000,
        "nstack": 8000000,
    },
    "medium_rubber": {
        "young": 1.0e6,
        "poisson": 0.40,
        "damping": 0.018,
        "mass": 0.55,
        "flex_radius": 0.0010,
        "joint_damping": 1.0,
        "option": {
            "timestep": "0.000015",
            "integrator": "implicitfast",
            "solver": "CG",
            "tolerance": "1e-10",
            "iterations": "240",
            "impratio": "1",
            "apirate": "200",
        },
        "contact": {
            "selfcollide": "none",
            "internal": "false",
            "condim": "1",
            "friction": "0.7 0.002 0.0001",
            "contype": "0",
            "conaffinity": "1",
            "solref": "0.07 1",
            "solimp": "0.62 0.84 0.005",
            "margin": "0.001",
        },
        "safe_spawn_heights": {"small": 0.19, "medium": 0.23, "large": 0.28},
        "action_scale": 0.55,
        "action_clip": 0.75,
        "action_smoothing": 0.60,
        "reset_settle_steps": 12,
        "nconmax": 12000,
        "nstack": 10000000,
    },
    "stiff_rubber": {
        "young": 5.0e6,
        "poisson": 0.35,
        "damping": 0.015,
        "mass": 0.70,
        "flex_radius": 0.0009,
        "joint_damping": 1.0,
        "option": {
            "timestep": "0.000005",
            "integrator": "implicitfast",
            "solver": "CG",
            "tolerance": "1e-10",
            "iterations": "320",
            "impratio": "1",
            "apirate": "200",
        },
        "contact": {
            "selfcollide": "none",
            "internal": "false",
            "condim": "1",
            "friction": "0.7 0.0015 0.0001",
            "contype": "0",
            "conaffinity": "1",
            "solref": "0.06 1",
            "solimp": "0.65 0.85 0.005",
            "margin": "0.001",
        },
        "safe_spawn_heights": {"small": 0.20, "medium": 0.24, "large": 0.30},
        "action_scale": 0.45,
        "action_clip": 0.65,
        "action_smoothing": 0.70,
        "reset_settle_steps": 16,
        "nconmax": 14000,
        "nstack": 12000000,
    },
}

DEFAULT_DEFORMABLE_PRESET = "soft_rubber_stable"
EGG_DEFORMABLE_FAST_OPTION_OVERRIDES = {
    "timestep": "0.00004",
    "iterations": "150",
}


def deformable_preset_names() -> list[str]:
    return sorted(DEFORMABLE_PRESETS)


def get_deformable_preset(name: str | None) -> Dict[str, object]:
    preset_name = name or DEFAULT_DEFORMABLE_PRESET
    try:
        return DEFORMABLE_PRESETS[preset_name]
    except KeyError as exc:
        valid = ", ".join(deformable_preset_names())
        raise ValueError(f"Unknown deformable preset {preset_name!r}. Valid presets: {valid}") from exc


def deformable_preset_spawn_position(name: str | None, size_label: str) -> str:
    preset = get_deformable_preset(name)
    heights = preset.get("safe_spawn_heights", {})
    if not isinstance(heights, dict) or size_label not in heights:
        return infer_custom_object_spawn_position(size_label, deformable=True)
    return f"1 0.87 {float(heights[size_label]):.6f}"


def _scale_triplet(base_value: float, multiplier: float) -> str:
    scaled = base_value * multiplier
    return f"{scaled:.6f} {scaled:.6f} {scaled:.6f}"


def _scale_scalar(base_value: float, multiplier: float) -> str:
    return f"{base_value * multiplier:.6f}"


def _scale_mass(base_mass: float, multiplier: float) -> str:
    return f"{base_mass * (multiplier ** 3):.6f}"


def _scale_diaginertia(base_inertia: tuple[float, float, float], multiplier: float) -> str:
    scaled = [value * (multiplier ** 5) for value in base_inertia]
    return " ".join(f"{value:.8f}" for value in scaled)


def infer_custom_object_size_label(custom_msh: str | None) -> str | None:
    if not custom_msh:
        return None
    mesh_name = os.path.basename(custom_msh).lower()
    match = re.search(r"(?:^|[_-])size[-_](small|medium|large)(?:[_-]|$)", mesh_name)
    if match:
        return match.group(1)
    return None


def infer_custom_object_spawn_position(size_label: str, deformable: bool) -> str:
    physics_mode = "deformable" if deformable else "rigid"
    z = SIZE_SPAWN_HEIGHTS[physics_mode][size_label]
    return f"1 0.87 {z:.6f}"


def parse_task_arg(task_arg: str) -> Dict[str, str | None]:
    """Parse --task which may be a built-in task name or a custom .msh path."""
    builtins = {"block", "egg", "pen"}
    if task_arg in builtins:
        return {
            "template_task": task_arg,
            "task_label": task_arg,
            "custom_msh": None,
        }

    if task_arg.lower().endswith(".msh"):
        msh_path = os.path.abspath(task_arg)
        if not os.path.isfile(msh_path):
            raise SystemExit(f"ERROR: custom mesh file not found: {msh_path}")
        return {
            "template_task": "block",  # use block task XML as the base task wrapper
            "task_label": f"custom_{_sanitize_task_label(msh_path)}",
            "custom_msh": msh_path,
        }

    raise SystemExit(
        "ERROR: --task must be one of {block, egg, pen} or a path to a .msh file."
    )


def write_builtin_egg_msh(msh_path: str, ntheta: int = 16, nlayers: int = 9) -> None:
    """Write a small tetrahedral egg volume mesh in Gmsh v2 ASCII format.

    The mesh is normalized for flexcomp scale="0.025 0.025 0.025":
    x/y radius ~= 1.2 -> 0.03m, z radius ~= 1.6 -> 0.04m.
    """
    if ntheta < 8:
        raise ValueError("ntheta must be at least 8")
    if nlayers < 5:
        raise ValueError("nlayers must be at least 5")

    nodes: list[tuple[float, float, float]] = []
    centers: list[int] = []
    rings: list[list[int] | None] = []

    for layer_idx in range(nlayers):
        u = -1.0 + 2.0 * layer_idx / (nlayers - 1)
        z = 1.6 * u
        # Make the lower half slightly fuller and the top slightly tapered, like the rigid egg task.
        radius = max(0.0, math.sqrt(max(0.0, 1.0 - u * u)) * (1.0 - 0.16 * u))
        center_id = len(nodes) + 1
        nodes.append((0.0, 0.0, z))
        centers.append(center_id)

        if radius < 1e-8:
            rings.append(None)
            continue

        ring: list[int] = []
        for theta_idx in range(ntheta):
            theta = 2.0 * math.pi * theta_idx / ntheta
            x = 1.2 * radius * math.cos(theta)
            y = 1.2 * radius * math.sin(theta)
            ring.append(len(nodes) + 1)
            nodes.append((x, y, z))
        rings.append(ring)

    elements: list[tuple[int, int, int, int]] = []

    def add_tet(a: int, b: int, c: int, d: int) -> None:
        if len({a, b, c, d}) == 4:
            elements.append((a, b, c, d))

    for layer_idx in range(nlayers - 1):
        c0 = centers[layer_idx]
        c1 = centers[layer_idx + 1]
        r0 = rings[layer_idx]
        r1 = rings[layer_idx + 1]

        if r0 is None and r1 is None:
            continue
        if r0 is None and r1 is not None:
            for i in range(ntheta):
                j = (i + 1) % ntheta
                add_tet(c0, c1, r1[i], r1[j])
            continue
        if r0 is not None and r1 is None:
            for i in range(ntheta):
                j = (i + 1) % ntheta
                add_tet(c1, c0, r0[j], r0[i])
            continue

        assert r0 is not None and r1 is not None
        for i in range(ntheta):
            j = (i + 1) % ntheta
            # Triangular prism between matching fan triangles, split into three tetrahedra.
            add_tet(c0, r0[i], r0[j], r1[j])
            add_tet(c0, r0[i], r1[j], r1[i])
            add_tet(c0, r1[i], r1[j], c1)

    os.makedirs(os.path.dirname(msh_path), exist_ok=True)
    with open(msh_path, "w", encoding="utf-8") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{len(nodes)}\n")
        for idx, (x, y, z) in enumerate(nodes, start=1):
            f.write(f"{idx} {x:.9g} {y:.9g} {z:.9g}\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(elements)}\n")
        for idx, tet in enumerate(elements, start=1):
            f.write(f"{idx} 4 0 {tet[0]} {tet[1]} {tet[2]} {tet[3]}\n")
        f.write("$EndElements\n")


def save_text_with_header(path: str, xml: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if not xml.lstrip().startswith("<?xml"):
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(xml)
        if not xml.endswith("\n"):
            f.write("\n")


def analyze_gmsh_msh(msh_path: str) -> Dict[str, object]:
    """Best-effort Gmsh v2 diagnostics for scale and element sanity."""
    raw = Path(msh_path).read_bytes()
    out: Dict[str, object] = {
        "path": os.path.abspath(msh_path),
        "format": "unknown",
        "binary": None,
        "node_count": None,
        "bbox_min": None,
        "bbox_max": None,
        "bbox_size": None,
        "element_count": None,
        "element_types": {},
        "warnings": [],
    }

    fmt_match = re.search(rb"\$MeshFormat\s+([^\n\r]+)", raw)
    if not fmt_match:
        out["warnings"].append("missing $MeshFormat section")
        return out

    fmt_parts = fmt_match.group(1).decode("ascii", errors="replace").split()
    if len(fmt_parts) >= 3:
        out["format"] = fmt_parts[0]
        out["binary"] = fmt_parts[1] == "1"
        data_size = int(fmt_parts[2])
    else:
        out["warnings"].append(f"unrecognized MeshFormat line: {fmt_parts}")
        return out

    endian = "<"
    endian_probe = raw[fmt_match.end():fmt_match.end() + 16]
    if out["binary"] and len(endian_probe) >= 5:
        probe = endian_probe.lstrip(b"\r\n")
        if len(probe) >= 4 and struct.unpack(">i", probe[:4])[0] == 1:
            endian = ">"

    def section_after_count(section: bytes):
        marker = b"$" + section
        start = raw.find(marker)
        if start < 0:
            return None, None
        offset = start + len(marker)
        while offset < len(raw) and raw[offset] in b"\r\n\t ":
            offset += 1
        end = raw.find(b"\n", offset)
        if end < 0:
            return None, None
        count = int(raw[offset:end].strip())
        return count, end + 1

    node_count, node_offset = section_after_count(b"Nodes")
    if node_count is not None:
        out["node_count"] = node_count
        points = []
        try:
            if out["binary"]:
                rec = struct.Struct(endian + "i" + ("d" if data_size == 8 else "f") * 3)
                for i in range(node_count):
                    _, x, y, z = rec.unpack_from(raw, node_offset + i * rec.size)
                    points.append((x, y, z))
            else:
                text = raw[node_offset:raw.find(b"$EndNodes", node_offset)].decode("utf-8", errors="replace")
                for line in text.splitlines()[:node_count]:
                    parts = line.split()
                    if len(parts) >= 4:
                        points.append(tuple(float(x) for x in parts[1:4]))
        except Exception as exc:
            out["warnings"].append(f"could not parse nodes: {type(exc).__name__}: {exc}")
        if points:
            cols = list(zip(*points))
            bbox_min = [min(col) for col in cols]
            bbox_max = [max(col) for col in cols]
            bbox_size = [hi - lo for lo, hi in zip(bbox_min, bbox_max)]
            out["bbox_min"] = [float(x) for x in bbox_min]
            out["bbox_max"] = [float(x) for x in bbox_max]
            out["bbox_size"] = [float(x) for x in bbox_size]
            if max(bbox_size) < 1e-4:
                out["warnings"].append("mesh bounding box is extremely small before flexcomp scaling")
            if max(bbox_size) > 10.0:
                out["warnings"].append("mesh bounding box is very large before flexcomp scaling")

    elem_count, elem_offset = section_after_count(b"Elements")
    if elem_count is not None:
        out["element_count"] = elem_count
        node_counts = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 8: 3, 9: 6, 10: 9, 11: 10, 15: 1}
        element_types: Dict[int, int] = defaultdict(int)
        try:
            if out["binary"]:
                offset = elem_offset
                parsed = 0
                block_header = struct.Struct(endian + "iii")
                while parsed < elem_count:
                    etype, block_n, tag_n = block_header.unpack_from(raw, offset)
                    offset += block_header.size
                    n_nodes = node_counts.get(etype)
                    if n_nodes is None:
                        out["warnings"].append(f"unknown Gmsh element type {etype}")
                        break
                    rec_ints = 1 + tag_n + n_nodes
                    rec = struct.Struct(endian + "i" * rec_ints)
                    for _ in range(block_n):
                        rec.unpack_from(raw, offset)
                        offset += rec.size
                    element_types[etype] += block_n
                    parsed += block_n
            else:
                text = raw[elem_offset:raw.find(b"$EndElements", elem_offset)].decode("utf-8", errors="replace")
                for line in text.splitlines()[:elem_count]:
                    parts = line.split()
                    if len(parts) >= 2:
                        element_types[int(parts[1])] += 1
        except Exception as exc:
            out["warnings"].append(f"could not parse elements: {type(exc).__name__}: {exc}")
        out["element_types"] = {str(k): int(v) for k, v in sorted(element_types.items())}
        if element_types and not any(k in element_types for k in (4, 11)):
            out["warnings"].append("no tetrahedral element types found; MuJoCo gmsh flexcomp expects a volume mesh")

    return out


def write_msh_diagnostics(msh_path: str, out_path: str) -> None:
    report = analyze_gmsh_msh(msh_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    warnings = report.get("warnings") or []
    if warnings:
        print(f"[WARN] MSH diagnostics for {msh_path}: {'; '.join(str(w) for w in warnings)}")
    print(f"[OK] Wrote MSH diagnostics: {out_path}")

# =========================
# 1) SHARED SENSOR BUILDER
# =========================

def _largest_remainder(values, labels, total_int, priority=None) -> Dict[str, int]:
    floors = [math.floor(v) for v in values]
    result = {lab: f for lab, f in zip(labels, floors)}
    leftover = total_int - sum(floors)
    if leftover <= 0:
        return result
    rema = [v - f for v, f in zip(values, floors)]
    pr_rank = {lab: i for i, lab in enumerate(priority)} if priority else {}
    order = list(range(len(labels)))
    order.sort(key=lambda i: (-rema[i], pr_rank.get(labels[i], 10**9), labels[i]))
    for i in order[:leftover]:
        result[labels[i]] += 1
    return result

def _allocate_groups_int(Ap, Apx, At, Ntotal, Rppx, Rpt) -> Dict[str, int]:
    if min(Ap, Apx, At) <= 0 or min(Rppx, Rpt) <= 0 or Ntotal <= 0:
        raise ValueError("Areas, ratios, and Ntotal must be positive.")
    Dp = Ntotal / (Ap + (Apx / Rppx) + (At / Rpt))
    vals = [Dp * Ap, (Dp / Rppx) * Apx, (Dp / Rpt) * At]
    out = _largest_remainder(vals, ["Np", "Npx", "Nt"], Ntotal, priority=["Nt", "Npx", "Np"])
    assert sum(out.values()) == Ntotal
    return out

def _split_palm_area_weighted(Np: int, Ap1: float, Ap2: float) -> Dict[str, int]:
    tot = Ap1 + Ap2
    if tot <= 0: raise ValueError("Ap1 + Ap2 must be > 0")
    return _largest_remainder(
        [Np*(Ap1/tot), Np*(Ap2/tot)],
        ["TS_palm","TS_lfmetacarpal"], Np,
        priority=["TS_palm","TS_lfmetacarpal"]
    )

SEEDS: Dict[str, List[str]] = {
    "TS_palm": [
        "robot0:T_palm_b0","robot0:T_palm_bl","robot0:T_palm_bm","robot0:T_palm_br",
        "robot0:T_palm_fl","robot0:T_palm_fm","robot0:T_palm_fr","robot0:T_palm_b1",
    ],
    "TS_lfmetacarpal": ["robot0:T_lfmetacarpal_front"],
    "TS_ffproximal": [
        "robot0:T_ffproximal_front_left_bottom","robot0:T_ffproximal_front_right_bottom",
        "robot0:T_ffproximal_front_left_top","robot0:T_ffproximal_front_right_top",
        "robot0:T_ffproximal_back_left","robot0:T_ffproximal_back_right",
        "robot0:T_ffproximal_tip",
    ],
    "TS_ffmiddle": [
        "robot0:T_ffmiddle_front_left","robot0:T_ffmiddle_front_right",
        "robot0:T_ffmiddle_back_left","robot0:T_ffmiddle_back_right",
        "robot0:T_ffmiddle_tip",
    ],
    "TS_fftip": [
        "robot0:T_fftip_front_left","robot0:T_fftip_front_right",
        "robot0:T_fftip_back_left","robot0:T_fftip_back_right","robot0:T_fftip_tip",
    ],
    "TS_mfproximal": [
        "robot0:T_mfproximal_front_left_bottom","robot0:T_mfproximal_front_right_bottom",
        "robot0:T_mfproximal_front_left_top","robot0:T_mfproximal_front_right_top",
        "robot0:T_mfproximal_back_left","robot0:T_mfproximal_back_right",
        "robot0:T_mfproximal_tip",
    ],
    "TS_mfmiddle": [
        "robot0:T_mfmiddle_front_left","robot0:T_mfmiddle_front_right",
        "robot0:T_mfmiddle_back_left","robot0:T_mfmiddle_back_right",
        "robot0:T_mfmiddle_tip",
    ],
    "TS_mftip": [
        "robot0:T_mftip_front_left","robot0:T_mftip_front_right",
        "robot0:T_mftip_back_left","robot0:T_mftip_back_right","robot0:T_mftip_tip",
    ],
    "TS_rfproximal": [
        "robot0:T_rfproximal_front_left_bottom","robot0:T_rfproximal_front_right_bottom",
        "robot0:T_rfproximal_front_left_top","robot0:T_rfproximal_front_right_top",
        "robot0:T_rfproximal_back_left","robot0:T_rfproximal_back_right",
        "robot0:T_rfproximal_tip",
    ],
    "TS_rfmiddle": [
        "robot0:T_rfmiddle_front_left","robot0:T_rfmiddle_front_right",
        "robot0:T_rfmiddle_back_left","robot0:T_rfmiddle_back_right",
        "robot0:T_rfmiddle_tip",
    ],
    "TS_rftip": [
        "robot0:T_rftip_front_left","robot0:T_rftip_front_right",
        "robot0:T_rftip_back_left","robot0:T_rftip_back_right","robot0:T_rftip_tip",
    ],
    "TS_lfproximal": [
        "robot0:T_lfproximal_front_left_bottom","robot0:T_lfproximal_front_right_bottom",
        "robot0:T_lfproximal_front_left_top","robot0:T_lfproximal_front_right_top",
        "robot0:T_lfproximal_back_left","robot0:T_lfproximal_back_right",
        "robot0:T_lfproximal_tip",
    ],
    "TS_lfmiddle": [
        "robot0:T_lfmiddle_front_left","robot0:T_lfmiddle_front_right",
        "robot0:T_lfmiddle_back_left","robot0:T_lfmiddle_back_right",
        "robot0:T_lfmiddle_tip",
    ],
    "TS_lftip": [
        "robot0:T_lftip_front_left","robot0:T_lftip_front_right",
        "robot0:T_lftip_back_left","robot0:T_lftip_back_right","robot0:T_lftip_tip",
    ],
    "TS_thproximal": [
        "robot0:T_thproximal_front_left","robot0:T_thproximal_front_right",
        "robot0:T_thproximal_back_left","robot0:T_thproximal_back_right",
        "robot0:T_thproximal_tip",
    ],
    "TS_thmiddle": [
        "robot0:T_thmiddle_front_left","robot0:T_thmiddle_front_right",
        "robot0:T_thmiddle_back_left","robot0:T_thmiddle_back_right",
        "robot0:T_thmiddle_tip",
    ],
    "TS_thtip": [
        "robot0:T_thtip_front_left","robot0:T_thtip_front_right",
        "robot0:T_thtip_back_left","robot0:T_thtip_back_right","robot0:T_thtip_tip",
    ],
}

PHALANX_KEYS = [
    "TS_ffproximal","TS_ffmiddle",
    "TS_mfproximal","TS_mfmiddle",
    "TS_rfproximal","TS_rfmiddle",
    "TS_lfproximal","TS_lfmiddle",
    "TS_thproximal","TS_thmiddle",
]
TIP_KEYS = ["TS_fftip","TS_mftip","TS_rftip","TS_lftip","TS_thtip"]

PREFIX = {k: k.replace("TS_", "robot0:T_") + "_auto" for k in (
    ["TS_palm","TS_lfmetacarpal"] + PHALANX_KEYS + TIP_KEYS
)}

def _names_for_region(region: str, count: int) -> List[Tuple[str, str]]:
    pairs = []
    seeds = SEEDS.get(region, [])
    take = min(count, len(seeds))
    for s in seeds[:take]:
        pairs.append((s.replace("robot0:T_", "robot0:TS_"), s))
    remain = count - take
    if remain > 0:
        base = PREFIX[region]
        for i in range(1, remain + 1):
            site = f"{base}_{i:03d}"
            pairs.append((site.replace("robot0:T_", "robot0:TS_"), site))
    return pairs

def build_sensor_xml_scaled(Ap, Apx, At, Ntotal, Rppx, Rpt, Ap1, Ap2):
    groups = _allocate_groups_int(Ap, Apx, At, Ntotal, Rppx, Rpt)
    Np, Npx, Nt = groups["Np"], groups["Npx"], groups["Nt"]
    palm = _split_palm_area_weighted(Np, Ap1, Ap2)
    phal = _largest_remainder([Npx/len(PHALANX_KEYS)]*len(PHALANX_KEYS), PHALANX_KEYS, Npx)
    tips = _largest_remainder([Nt/len(TIP_KEYS)]*len(TIP_KEYS), TIP_KEYS, Nt)

    desired = {}
    desired.update(palm); desired.update(phal); desired.update(tips)

    sections = [
        ("PALM", ["TS_palm", "TS_lfmetacarpal"]),
        ("FOREFINGER", ["TS_ffproximal","TS_ffmiddle","TS_fftip"]),
        ("MIDDLE FINGER", ["TS_mfproximal","TS_mfmiddle","TS_mftip"]),
        ("RING FINGER", ["TS_rfproximal","TS_rfmiddle","TS_rftip"]),
        ("LITTLE FINGER", ["TS_lfproximal","TS_lfmiddle","TS_lftip"]),
        ("THUMB", ["TS_thproximal","TS_thmiddle","TS_thtip"]),
    ]

    lines = ['<mujoco>', '    <sensor>']
    for title, keys in sections:
        lines.append(f'\n        <!--{title}-->')
        for k in keys:
            n = desired.get(k, 0)
            for touch_name, site_name in _names_for_region(k, n):
                lines.append(f'        <touch name="{touch_name}" site="{site_name}"></touch>')
    lines += ['\n    </sensor>', '</mujoco>']
    xml = "\n".join(lines)

    stats = {"Np": Np, "Npx": Npx, "Nt": Nt, "Ntotal": Ntotal}
    for k in ["TS_palm", "TS_lfmetacarpal"] + PHALANX_KEYS + TIP_KEYS:
        stats[k] = desired.get(k, 0)
    stats["check_sum"] = sum(stats[k] for k in ["TS_palm", "TS_lfmetacarpal"] + PHALANX_KEYS + TIP_KEYS)
    return xml, stats

# ================================
# 2) MERGE & LAYOUT SITES ON BODIES
# ================================

ALPHA = 0.95; BETA = 0.90; T = 0.0025
GAP_U = 0.0015; GAP_Z = 0.0015; MARGIN = 0.0005
FRONT, BACK, LEFT, RIGHT = "front","back","left","right"
FaceLayout = namedtuple("FaceLayout", "axis tang_half ax_half normal_center face")

def parse_sensor_sites(sensor_xml_path):
    root = ET.parse(sensor_xml_path).getroot()
    return sorted({t.get("site") for t in root.findall(".//touch") if t.get("site")})

def site_to_body(site_name):
    if ":" not in site_name: raise ValueError(f"Bad site name: {site_name}")
    _, tail = site_name.split(":", 1)
    if not tail.startswith("T_"): raise ValueError(f"Site must start with T_: {site_name}")
    tag = tail[2:]
    if tag.startswith("palm"): return "robot0:palm"
    if tag.startswith("lfmetacarpal"): return "robot0:lfmetacarpal"

    m = re.match(r"(ff|mf|rf|lf)(proximal|middle|tip)", tag)
    if m: finger, seg = m.groups(); seg_body = "distal" if seg == "tip" else seg; return f"robot0:{finger}{seg_body}"
    m = re.match(r"(th)(proximal|middle|tip)", tag)
    if m: thumb, seg = m.groups(); seg_body = "distal" if seg == "tip" else seg; return f"robot0:{thumb}{seg_body}"

    if tag.startswith("palm_"): return "robot0:palm"
    if tag.startswith("lfmetacarpal_"): return "robot0:lfmetacarpal"
    m = re.match(r"(ff|mf|rf|lf)(proximal|middle|tip)_", tag)
    if m: finger, seg = m.groups(); seg_body = "distal" if seg == "tip" else seg; return f"robot0:{finger}{seg_body}"
    m = re.match(r"(th)(proximal|middle|tip)_", tag)
    if m: thumb, seg = m.groups(); seg_body = "distal" if seg == "tip" else seg; return f"robot0:{thumb}{seg_body}"

    raise ValueError(f"Cannot infer body for site '{site_name}'")

def find_body_elem(root, body_name_full):
    for b in root.findall(".//body"):
        if b.get("name") == body_name_full:
            return b
    return None

def find_primary_geom_on_body(body_elem):
    for g in body_elem.findall("geom"):
        t = g.get("type", "mesh")
        if t in ("capsule", "box"):
            return g
    return None

def capsule_dims(geom):
    parts = [float(x) for x in geom.get("size","").split()]
    if len(parts) < 2: raise ValueError("Capsule geom missing size")
    return parts[0], parts[1]

def box_dims(geom):
    parts = [float(x) for x in geom.get("size","").split()]
    if len(parts) < 3: raise ValueError("Box geom missing size")
    return parts[0], parts[1], parts[2]

# ---------- NEW: pose / quaternion helpers (for geom→body transform) ----------
def _parse_vec(s: str | None, n: int):
    vals = [float(x) for x in s.split()] if s else []
    vals += [0.0] * (n - len(vals))
    return tuple(vals[:n])

def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    )

def _quat_from_axisangle(axisangle):
    x, y, z, ang = axisangle
    half = 0.5 * ang
    s = math.sin(half)
    c = math.cos(half)
    return (c, x*s, y*s, z*s)

def _quat_from_euler_xyz(angles):
    rx, ry, rz = angles
    cx, sx = math.cos(rx/2), math.sin(rx/2)
    cy, sy = math.cos(ry/2), math.sin(ry/2)
    cz, sz = math.cos(rz/2), math.sin(rz/2)
    qx = (cx, sx, 0.0, 0.0)
    qy = (cy, 0.0, sy, 0.0)
    qz = (cz, 0.0, 0.0, sz)
    return _quat_mul(_quat_mul(qx, qy), qz)

def _quat_rotate(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y*vz - z*vy)
    ty = 2.0 * (z*vx - x*vz)
    tz = 2.0 * (x*vy - y*vx)
    vpx = vx + w*tx + (y*tz - z*ty)
    vpy = vy + w*ty + (z*tx - x*tz)
    vpz = vz + w*tz + (x*ty - y*tx)
    return (vpx, vpy, vpz)

def _geom_pose(geom):
    gpos = _parse_vec(geom.get("pos"), 3)
    if geom.get("quat"):
        gquat = _parse_vec(geom.get("quat"), 4)
    elif geom.get("euler"):
        gquat = _quat_from_euler_xyz(_parse_vec(geom.get("euler"), 3))
    elif geom.get("axisangle"):
        gquat = _quat_from_axisangle(_parse_vec(geom.get("axisangle"), 4))
    else:
        gquat = (1.0, 0.0, 0.0, 0.0)
    return gpos, gquat
# ------------------------------------------------------------------------------

def face_layout_for_capsule(face, r, L):
    wx = ALPHA * r; wy = ALPHA * r; z_half = L * ALPHA
    if face == FRONT:   return FaceLayout("capsule_front", wx, z_half, (0.0, -BETA*r, 0.0), FRONT)
    if face == BACK:    return FaceLayout("capsule_back",  wx, z_half, (0.0, +BETA*r, 0.0), BACK)
    if face == LEFT:    return FaceLayout("capsule_left",  wy, z_half, (-BETA*r, 0.0, 0.0), LEFT)
    if face == RIGHT:   return FaceLayout("capsule_right", wy, z_half, (+BETA*r, 0.0, 0.0), RIGHT)
    raise ValueError(face)

def face_layout_for_box(face, sx, sy, sz):
    if face in (FRONT, BACK):
        y = -BETA*sy if face == FRONT else +BETA*sy
        return FaceLayout("box_fb", ALPHA*sx, ALPHA*sz, (0.0, y, 0.0), face)
    if face in (LEFT, RIGHT):
        x = -BETA*sx if face == LEFT else +BETA*sx
        return FaceLayout("box_lr", ALPHA*sy, ALPHA*sz, (x, 0.0, 0.0), face)
    raise ValueError(face)

def choose_base_grid(N, aspect_t_over_z):
    if N <= 2: return (N, 1)
    best = None; root = int(math.ceil(math.sqrt(N)))
    for nz in range(1, root+3):
        nx = int(math.ceil(N / nz))
        for nx_try in (nx, nx+1):
            ar = nx_try / nz
            cost = abs(ar - aspect_t_over_z) + 0.1*(nx_try*nz - N)
            cand = (cost, nx_try, nz)
            if best is None or cand < best: best = cand
    _, nx_base, nz_base = best
    return nx_base, nz_base

def row_distribution(N, nz):
    q, r = divmod(N, nz)
    m = [q]*nz
    for i in range(r):
        m[nz - 1 - i] += 1
    return m

def layout_cover_full(face_layout, N, gap_u=GAP_U, gap_z=GAP_Z, margin=MARGIN):
    if N <= 0: return []
    tang_half = max(0.0, face_layout.tang_half - margin)
    ax_half   = max(0.0, face_layout.ax_half   - margin)
    W = 2.0 * tang_half; H = 2.0 * ax_half
    aspect = (W / H) if H > 1e-9 else 1.0
    nx_base, nz_base = choose_base_grid(N, aspect)
    nz = min(nz_base, N)
    m_per_row = row_distribution(N, nz)

    if nz == 1:
        row_h = H; row_center_z = [0.0]
    else:
        total_gap_z = gap_z * (nz - 1)
        row_h = (H - total_gap_z) / nz
        z0 = -ax_half + row_h/2.0
        row_center_z = [z0 + i*(row_h + gap_z) for i in range(nz)]

    out = []
    for row_idx, m in enumerate(m_per_row):
        zc = row_center_z[row_idx]
        if m <= 0: continue
        if m == 1:
            cell_w = W; xs = [0.0]
        else:
            total_gap_u = gap_u * (m - 1)
            cell_w = (W - total_gap_u) / m
            x0 = -tang_half + cell_w/2.0
            xs = [x0 + j*(cell_w + gap_u) for j in range(m)]
        half_u = cell_w/2.0; half_z = row_h/2.0
        for x in xs:
            nx, ny, nz_pos = face_layout.normal_center
            if face_layout.face in (FRONT, BACK):
                pos = (x, ny, zc); size = (half_u, T, half_z)
            else:
                pos = (nx, x, zc); size = (T, half_u, half_z)
            out.append({"pos": pos, "size": size})
    return out

def split_counts_7_1_1(N):
    total = 9
    nf = (7*N)//total; nb = (1*N)//total; ns = (1*N)//total
    assigned = nf + nb + ns; rem = N - assigned
    order = ['front', 'back', 'sides']; i = 0
    while rem > 0:
        tgt = order[i % len(order)]
        if tgt == 'front': nf += 1
        elif tgt == 'back': nb += 1
        else: ns += 1
        rem -= 1; i += 1
    return {'front': nf, 'back': nb, 'sides': ns}

def split_sides_left_right(nsides):
    left = nsides // 2
    right = nsides - left
    return left, right

def assign_faces_by_ratio(site_names_for_body, body_name):
    N = len(site_names_for_body)
    if body_name in ("robot0:palm", "robot0:lfmetacarpal"):
        return [(s, FRONT) for s in site_names_for_body]
    split = split_counts_7_1_1(N)
    n_front, n_back, n_sides = split['front'], split['back'], split['sides']
    n_left, n_right = split_sides_left_right(n_sides)

    out = []; idx = 0
    for _ in range(min(n_front, N-idx)): out.append((site_names_for_body[idx], FRONT)); idx += 1
    for _ in range(min(n_back,  N-idx)): out.append((site_names_for_body[idx], BACK));  idx += 1
    for _ in range(min(n_left,  N-idx)): out.append((site_names_for_body[idx], LEFT));  idx += 1
    for _ in range(min(n_right, N-idx)): out.append((site_names_for_body[idx], RIGHT)); idx += 1
    while idx < N: out.append((site_names_for_body[idx], FRONT)); idx += 1
    return out

def ensure_site(body_elem, site_name):
    site = body_elem.find(f"./site[@name='{site_name}']")
    if site is None:
        site = ET.Element("site", {"name": site_name, "type": "box"})
        children = list(body_elem)
        insert_idx = None
        for i, ch in enumerate(children):
            if ch.tag == "body":
                insert_idx = i; break
        if insert_idx is None: body_elem.append(site)
        else: body_elem.insert(insert_idx, site)
    else:
        site.set("type", "box")
    return site

def set_site_pose(site_elem, pos, size):
    site_elem.set("pos", f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    site_elem.set("size", f"{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}")

def merge_sites_with_layout(base_xml_path, sensor_xml_path, out_xml_path):
    sensor_sites = parse_sensor_sites(sensor_xml_path)
    base_tree = ET.parse(base_xml_path); base_root = base_tree.getroot()

    by_body = defaultdict(list); unresolved = []
    for s in sensor_sites:
        try: b = site_to_body(s)
        except Exception as e: unresolved.append((s, f"body: {e}")); continue
        by_body[b].append(s)

    debug_counts = defaultdict(int); missing_body = []; updated = 0

    for body_name, sites in by_body.items():
        body = find_body_elem(base_root, body_name)
        if body is None: missing_body.append((body_name, f"Body not found")); continue
        geom = find_primary_geom_on_body(body)
        if geom is None:
            for s in sites: unresolved.append((s, f"no geom on {body_name}")); continue

        # NEW: capture geom pose (rotation+translation) to transform local→body
        gpos, gquat = _geom_pose(geom)

        gtype = geom.get("type")
        if gtype == "capsule":
            r, L = capsule_dims(geom)
            face_layout = {
                FRONT: face_layout_for_capsule(FRONT, r, L),
                BACK:  face_layout_for_capsule(BACK,  r, L),
                LEFT:  face_layout_for_capsule(LEFT,  r, L),
                RIGHT: face_layout_for_capsule(RIGHT, r, L),
            }
        elif gtype == "box":
            sx, sy, sz = box_dims(geom)
            face_layout = {
                FRONT: face_layout_for_box(FRONT, sx, sy, sz),
                BACK:  face_layout_for_box(BACK,  sx, sy, sz),
                LEFT:  face_layout_for_box(LEFT,  sx, sy, sz),
                RIGHT: face_layout_for_box(RIGHT, sx, sy, sz),
            }
        else:
            for s in sites: unresolved.append((s, f"unsupported geom {gtype}")); continue

        face_assignments = assign_faces_by_ratio(sites, body_name)
        grouped = defaultdict(list)
        for site_name, face in face_assignments:
            grouped[face].append(site_name)

        for face, site_list in grouped.items():
            if body_name in ("robot0:palm", "robot0:lfmetacarpal") and face in (LEFT, RIGHT, BACK):
                continue
            N = len(site_list)
            rects = layout_cover_full(face_layout[face], N, GAP_U, GAP_Z, MARGIN)
            for site_name, spec in zip(site_list, rects):
                # transform local (geom frame) → body frame
                p_local = spec["pos"]
                p_rot   = _quat_rotate(gquat, p_local)
                p_body  = (p_rot[0] + gpos[0], p_rot[1] + gpos[1], p_rot[2] + gpos[2])

                site_elem = ensure_site(body, site_name)
                set_site_pose(site_elem, p_body, spec["size"]); updated += 1
            for site_name in site_list[len(rects):]:
                site_elem = ensure_site(body, site_name)
                set_site_pose(site_elem, (0.0,0.0,0.0), (0.0,0.0,0.0)); updated += 1
            debug_counts[(body_name, face)] += N

    ET.indent(base_tree, space="    ")
    os.makedirs(os.path.dirname(out_xml_path), exist_ok=True)
    base_tree.write(out_xml_path, encoding="utf-8", xml_declaration=True)

    print("# ---- Merge Touch Sites (7:1:1 split + coverage) ----")
    print(f"Input base:    {base_xml_path}")
    print(f"Input sensors: {sensor_xml_path}")
    print(f"Output file:   {out_xml_path}")
    if debug_counts:
        print("\nPer (body, face) site counts:")
        for (b, f), c in sorted(debug_counts.items()):
            print(f"  {b:24s} {f:5s}: {c}")
    if missing_body or unresolved:
        print("\nWarnings:")
        for item in missing_body: print(" ", item)
        for s, msg in unresolved: print(" ", s, ":", msg)

# =======================
# 3) INCLUDE FILE UPDATER
# =======================

def _find_parent_tag(root: ET.Element, child: ET.Element) -> str | None:
    for elem in root.iter():
        for ch in list(elem):
            if ch is child:
                return elem.tag
    return None

def update_includes_by_prefix(tree: ET.ElementTree, new_shared_basename: str, new_robot_basename: str) -> dict:
    """
    Replace current includes for:
      - shared: filename that starts with 'shared_touch_sensors'
      - robot : filename that starts with 'Sensors_withPos'
    Fallbacks:
      - shared: first non-worldbody include
      - robot : first worldbody include
    Returns {'shared_updated', 'robot_updated'}.
    """
    root = tree.getroot()
    includes = [e for e in root.iter() if e.tag == "include"]
    files = [e.attrib.get("file", "") for e in includes]

    shared_idx = next((i for i, f in enumerate(files) if os.path.basename(f).startswith("shared_touch_sensors")), None)
    robot_idx  = next((i for i, f in enumerate(files) if os.path.basename(f).startswith("Sensors_withPos")), None)

    # If either wasn't found, use structure-based fallbacks
    if shared_idx is None or robot_idx is None:
        parents = []
        for elem in root.iter():
            if elem.tag == "include":
                parents.append(_find_parent_tag(root, elem))
        if robot_idx is None:
            for i, tag in enumerate(parents):
                if tag == "worldbody":
                    robot_idx = i; break
        if shared_idx is None:
            for i, tag in enumerate(parents):
                if tag != "worldbody":
                    shared_idx = i; break

    counts = {'shared_updated': 0, 'robot_updated': 0}
    if shared_idx is not None:
        includes[shared_idx].set("file", new_shared_basename)
        counts['shared_updated'] += 1
    if robot_idx is not None:
        includes[robot_idx].set("file", new_robot_basename)
        counts['robot_updated'] += 1
    return counts

def write_standalone_env(template_xml: str, out_env: str, shared_basename: str, robot_basename: str):
    """Copy template and point its includes to the given basenames (placed in the same folder as out_env)."""
    tree = ET.parse(template_xml)
    update_includes_by_prefix(tree, shared_basename, robot_basename)
    os.makedirs(os.path.dirname(out_env), exist_ok=True)
    tree.write(out_env, encoding="utf-8", xml_declaration=True)


def patch_env_object_to_custom_msh(
    env_xml: str,
    msh_file_for_xml: str,
    *,
    deformable: bool = False,
    rigid_surface_file_for_xml: str | None = None,
    object_mass: float = 0.07,
    object_inertia: str = "1e-3 1e-3 1e-3",
    object_pos: str = "1 0.87 0.4",
    flex_scale: str = "0.025 0.025 0.025",
    flex_radius: str = "0.001",
    deformable_preset: str = DEFAULT_DEFORMABLE_PRESET,
    deformable_option_overrides: Dict[str, str] | None = None,
):
    """
    Replace the default object with a custom deformable flex or native rigid mesh.

    The deformable branch continues to consume the volume ``.msh`` through
    ``flexcomp``.  The rigid branch requires a previously validated exterior surface
    and emits a conventional mesh geom.  Both retain the task's free-joint and
    object:center names.
    """
    tree = ET.parse(env_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise SystemExit(f"ERROR: no <worldbody> in {env_xml}")

    preset = get_deformable_preset(deformable_preset) if deformable else {}

    if deformable:
        # Flex simulations must survive contact-rich RL rollouts, not just passive drops.
        option = root.find("option")
        if option is not None:
            option_attrs = preset.get("option", {})
            if isinstance(option_attrs, dict):
                for key, value in option_attrs.items():
                    option.set(str(key), str(value))
            if deformable_option_overrides:
                for key, value in deformable_option_overrides.items():
                    option.set(str(key), str(value))
            flag = option.find("flag")
            if flag is None:
                ET.SubElement(option, "flag", {"warmstart": "enable"})
            else:
                flag.set("warmstart", "enable")

        floor0_geom = worldbody.find("./geom[@name='floor0']")
        if floor0_geom is not None:
            floor0_geom.set("condim", "1")
            floor0_geom.set("contype", "2")
            floor0_geom.set("conaffinity", "2")

    object_body = None
    for body in worldbody.findall("body"):
        if body.get("name") == "object":
            object_body = body
            break
    if object_body is None:
        raise SystemExit(f"ERROR: no <body name='object'> found in {env_xml}")

    if not deformable:
        if not rigid_surface_file_for_xml:
            raise ValueError("rigid custom objects require an exterior surface mesh")
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        for old_mesh in list(asset.findall(f"./mesh[@name='{RIGID_MESH_ASSET_NAME}']")):
            asset.remove(old_mesh)
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": RIGID_MESH_ASSET_NAME,
                "file": rigid_surface_file_for_xml,
                "scale": flex_scale,
            },
        )

    # Preserve ordering expected by other tooling: joint + inertial + collision + site.
    object_body.set("pos", object_pos)
    for ch in list(object_body):
        object_body.remove(ch)

    joint_damping = str(preset.get("joint_damping", "1.0")) if deformable else "0.05"
    ET.SubElement(object_body, "joint", {"name": "object:joint", "type": "free", "damping": joint_damping})

    if deformable:
        object_body.set("quat", "1 0 0 0")
        proxy_radius = flex_scale.split()[0]
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": "object",
                "type": "sphere",
                "size": proxy_radius,
                "rgba": "0 0 0 0",
                "condim": "1",
                "contype": "0",
                "conaffinity": "3",
            },
        )
        ET.SubElement(
            object_body,
            "inertial",
            {"pos": "0 0 0", "mass": str(object_mass), "diaginertia": object_inertia},
        )
        flex = ET.SubElement(
            object_body,
            "flexcomp",
            {
                "name": "soft",
                "type": "gmsh",
                "file": msh_file_for_xml,
                "dim": "3",
                "dof": "trilinear",
                "pos": "0 0 0",
                "scale": flex_scale,
                "radius": flex_radius,
                "rigid": "false",
                "rgba": "0.7 0.8 1.0 0.5",
            },
        )
        ET.SubElement(
            flex,
            "elasticity",
            {
                "young": f"{float(preset['young']):.12g}",
                "poisson": f"{float(preset['poisson']):.12g}",
                "damping": f"{float(preset['damping']):.12g}",
            },
        )
        contact_attrs = preset.get("contact", {})
        if not isinstance(contact_attrs, dict):
            contact_attrs = {}
        ET.SubElement(
            flex,
            "contact",
            {str(key): str(value) for key, value in contact_attrs.items()},
        )
        ET.SubElement(
            object_body,
            "site",
            {"name": "object:center", "pos": "0 0 0", "rgba": "1 0 0 1", "size": "0.001 0.001 0.001"},
        )
        ET.SubElement(object_body, "site", {"name": "obj:x", "pos": "0.03 0 0", "size": "0.004", "rgba": "1 0 0 1"})
        ET.SubElement(object_body, "site", {"name": "obj:y", "pos": "0 0.03 0", "size": "0.004", "rgba": "0 1 0 1"})
        ET.SubElement(object_body, "site", {"name": "obj:z", "pos": "0 0 0.03", "size": "0.004", "rgba": "0 0 1 1"})
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": "obj:x_axis",
                "type": "capsule",
                "fromto": "0 0 0 0.03 0 0",
                "size": "0.0015",
                "rgba": "1 0 0 1",
                "contype": "0",
                "conaffinity": "0",
                "group": "3",
            },
        )
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": "obj:y_axis",
                "type": "capsule",
                "fromto": "0 0 0 0 0.03 0",
                "size": "0.0015",
                "rgba": "0 1 0 1",
                "contype": "0",
                "conaffinity": "0",
                "group": "3",
            },
        )
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": "obj:z_axis",
                "type": "capsule",
                "fromto": "0 0 0 0 0 0.03",
                "size": "0.0015",
                "rgba": "0 0 1 1",
                "contype": "0",
                "conaffinity": "0",
                "group": "3",
            },
        )
    else:
        ET.SubElement(
            object_body,
            "inertial",
            {"pos": "0 0 0", "mass": str(object_mass), "diaginertia": object_inertia},
        )
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": "object",
                "type": "mesh",
                "mesh": RIGID_MESH_ASSET_NAME,
                **RIGID_GEOM_CONTACT_ATTRIBUTES,
            },
        )
        ET.SubElement(
            object_body,
            "site",
            {"name": "object:center", "pos": "0 0 0", "rgba": "1 0 0 0", "size": "0.01 0.01 0.01"},
        )

    tree.write(env_xml, encoding="utf-8", xml_declaration=True)


def patch_shared_for_deformable(shared_xml: str, deformable_preset: str = DEFAULT_DEFORMABLE_PRESET) -> None:
    """Raise contact/stack limits for generated deformable candidates."""
    preset = get_deformable_preset(deformable_preset)
    tree = ET.parse(shared_xml)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
    size.set("nconmax", str(int(preset.get("nconmax", 10000))))
    size.set("nstack", str(int(preset.get("nstack", 8000000))))
    tree.write(shared_xml, encoding="utf-8", xml_declaration=True)


def _parse_scale_triplet(scale: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in scale.split())
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError(f"mesh scale must contain three positive values, found {scale!r}")
    return values  # type: ignore[return-value]


def _rigid_env_matches_surface(env_xml: str, surface_basename: str) -> bool:
    if not os.path.isfile(env_xml):
        return False
    try:
        root = ET.parse(env_xml).getroot()
    except ET.ParseError:
        return False
    object_body = root.find("./worldbody/body[@name='object']")
    if object_body is None or object_body.find("flexcomp") is not None:
        return False
    geom = object_body.find("./geom[@name='object']")
    mesh = root.find(f"./asset/mesh[@name='{RIGID_MESH_ASSET_NAME}']")
    return bool(
        geom is not None
        and geom.get("type") == "mesh"
        and geom.get("mesh") == RIGID_MESH_ASSET_NAME
        and mesh is not None
        and mesh.get("file") == surface_basename
    )


def write_rigid_representation_manifest(
    path: str,
    *,
    conversion_result,
    generated_surface_path: str,
    mesh_scale: str,
    object_mass: float,
    object_inertia: str,
    object_pos: str,
) -> None:
    scale = _parse_scale_triplet(mesh_scale)
    geometry = conversion_result.manifest["geometry"]
    payload = {
        "representation": "rigid_mesh_geom",
        "source_mesh": str(conversion_result.source_path),
        "source_hash": conversion_result.source_hash,
        "converted_mesh": os.path.abspath(generated_surface_path),
        "converted_hash": conversion_result.manifest["converted_hash"],
        "conversion_manifest": str(conversion_result.conversion_manifest_path),
        "converter_version": conversion_result.manifest["converter_version"],
        "conversion_parameters": conversion_result.manifest["conversion_parameters"],
        "cache_key": conversion_result.cache_key,
        "cache_reused": bool(conversion_result.cache_reused),
        "source_format": conversion_result.manifest["source_format"],
        "source_bbox": {
            "min": geometry["source_bbox_min"],
            "max": geometry["source_bbox_max"],
            "dimensions": geometry["source_bbox_dimensions"],
        },
        "converted_bbox": {
            "min": geometry["bbox_min"],
            "max": geometry["bbox_max"],
            "dimensions": geometry["bbox_dimensions"],
        },
        "scaled_geometry": scaled_geometry_metrics(geometry, scale),
        "mass": float(object_mass),
        "inertial_position": [0.0, 0.0, 0.0],
        "diaginertia": [float(value) for value in object_inertia.split()],
        "body_position": [float(value) for value in object_pos.split()],
        "free_joint": "object:joint",
        "mesh_asset": RIGID_MESH_ASSET_NAME,
        "geom_name": "object",
        "contact_parameters": dict(RIGID_GEOM_CONTACT_ATTRIBUTES),
        "unmapped_rigid_flex_parameters": {
            "radius": "no exact native mesh-geom equivalent",
            "selfcollide": "not applicable to a single rigid geom",
            "internal": "internal tetrahedral faces were removed during conversion",
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

# =======================
# 4) HIGH-LEVEL MODES
# =======================

def build_shared_and_robot(Ap, Apx, At, Ntotal, Rppx, Rpt, Ap1, Ap2, base_xml, out_shared, out_robot, force=False):
    """Generate shared sensors XML and robot-with-sites XML."""
    if (not force) and os.path.exists(out_shared) and os.path.exists(out_robot):
        print(f"[SKIP] Using cached files:\n  {out_shared}\n  {out_robot}")
        return
    xml, stats = build_sensor_xml_scaled(Ap, Apx, At, Ntotal, Rppx, Rpt, Ap1, Ap2)
    save_text_with_header(out_shared, xml)
    print(f"[OK] Wrote shared sensors: {out_shared}")
    print(f"     Totals: Np={stats['Np']} Npx={stats['Npx']} Nt={stats['Nt']} (sum {stats['check_sum']})")
    merge_sites_with_layout(base_xml, out_shared, out_robot)
    print(f"[OK] Wrote robot hand with sites: {out_robot}")

def resolve_task_template(task: str, explicit_template: str | None, main_fallback: str | None) -> str:
    """Choose the template XML for the task."""
    if explicit_template:
        return explicit_template
    if main_fallback:
        return main_fallback
    # Default guesses (edit if your repo uses different paths/names)
    defaults = {
        "block": "assets/manipulate_block_touch_sensors.xml",
        "egg":   "assets/manipulate_egg_touch_sensors.xml",
        "pen":   "assets/manipulate_pen_touch_sensors.xml",
    }
    if task not in defaults:
        raise SystemExit(f"ERROR: unknown task {task!r} and no --template given.")
    return defaults[task]

def build_candidate_standalone(
    task: str,
    Ntotal, Rppx, Rpt, Ap, Apx, At, Ap1, Ap2,
    base_xml: str, template_xml: str, out_root: str, force=False,
    custom_msh: str | None = None,
    custom_msh_name: str | None = None,
    deformable_object: bool = False,
    flex_scale: str = "0.025 0.025 0.025",
    flex_radius: str = "0.001",
    object_pos: str = "1 0.87 0.4",
    object_mass: str | float = "0.07",
    object_inertia: str = "1e-3 1e-3 1e-3",
    deformable_preset: str = DEFAULT_DEFORMABLE_PRESET,
    rigid_mesh_cache_dir: str | None = None,
) -> Dict[str, str]:
    """
    No side effects. Returns dict with paths:
      {dir, shared, robot, env, env_basename, tag}
    """
    paths = make_candidate_paths(out_root, task, Ntotal, Rppx, Rpt)
    builtin_msh: str | None = None
    deformable_option_overrides = None
    if deformable_object and custom_msh is None:
        if task != "egg_deformable":
            raise SystemExit("ERROR: built-in deformable generation currently supports --task egg only.")
        builtin_mesh_dir = os.path.join(out_root, "builtin_deformable")
        builtin_msh = os.path.join(builtin_mesh_dir, "egg_deformable.msh")
        if force or (not os.path.exists(builtin_msh)):
            write_builtin_egg_msh(builtin_msh)
            print(f"[OK] Wrote built-in deformable egg .msh: {builtin_msh}")
        custom_msh = builtin_msh
        custom_msh_name = "egg_deformable.msh"
        deformable_option_overrides = EGG_DEFORMABLE_FAST_OPTION_OVERRIDES

    rigid_conversion = None
    rigid_surface_dst: str | None = None
    msh_dst: str | None = None
    msh_basename: str | None = None
    if custom_msh:
        # Preserve the source volume mesh beside other compiler assets for audit and
        # for the unchanged deformable branch.
        custom_mesh_dst_dir = os.path.join(out_root, "stls", "hand")
        os.makedirs(custom_mesh_dst_dir, exist_ok=True)
        msh_basename = custom_msh_name or os.path.basename(custom_msh)
        msh_dst = os.path.join(custom_mesh_dst_dir, msh_basename)
        if force or (not os.path.exists(msh_dst)):
            shutil.copy2(custom_msh, msh_dst)
            print(f"[OK] Copied custom .msh: {msh_dst}")
        diag_path = os.path.join(paths["dir"], f"{os.path.splitext(msh_basename)[0]}_msh_diagnostics.json")
        if force or (not os.path.exists(diag_path)):
            write_msh_diagnostics(msh_dst, diag_path)

        if not deformable_object:
            cache_dir = rigid_mesh_cache_dir or os.path.join(out_root, "rigid_mesh_cache")
            rigid_conversion = convert_gmsh_to_rigid_surface(custom_msh, cache_dir)
            rigid_surface_dst = os.path.join(custom_mesh_dst_dir, rigid_conversion.mesh_path.name)
            if force or not os.path.exists(rigid_surface_dst):
                if rigid_conversion.mesh_path.resolve() != Path(rigid_surface_dst).resolve():
                    shutil.copy2(rigid_conversion.mesh_path, rigid_surface_dst)
                print(f"[OK] Installed cached rigid surface: {rigid_surface_dst}")
            else:
                print(f"[SKIP] Using cached rigid surface: {rigid_surface_dst}")
            paths["rigid_surface"] = rigid_surface_dst
            paths["rigid_conversion_manifest"] = str(rigid_conversion.conversion_manifest_path)
            paths["rigid_cache_key"] = rigid_conversion.cache_key

    # 1) shared + robot
    build_shared_and_robot(Ap, Apx, At, Ntotal, Rppx, Rpt, Ap1, Ap2, base_xml, paths["shared"], paths["robot"], force=force)
    # 2) standalone env that includes the basenames
    needs_env = force or (not os.path.exists(paths["env"]))
    if rigid_surface_dst is not None and not _rigid_env_matches_surface(
        paths["env"], os.path.basename(rigid_surface_dst)
    ):
        needs_env = True
    if needs_env:
        write_standalone_env(
            template_xml=template_xml,
            out_env=paths["env"],
            shared_basename=os.path.basename(paths["shared"]),
            robot_basename=os.path.basename(paths["robot"]),
        )
        if custom_msh:
            assert msh_dst is not None
            patch_env_object_to_custom_msh(
                paths["env"],
                os.path.basename(msh_dst),
                deformable=deformable_object,
                rigid_surface_file_for_xml=(
                    os.path.basename(rigid_surface_dst) if rigid_surface_dst else None
                ),
                object_mass=float(object_mass),
                object_inertia=object_inertia,
                object_pos=object_pos,
                flex_scale=flex_scale,
                flex_radius=flex_radius,
                deformable_preset=deformable_preset,
                deformable_option_overrides=deformable_option_overrides,
            )
        print(f"[OK] Wrote standalone env: {paths['env']}")
    else:
        print(f"[SKIP] Using cached env: {paths['env']}")

    if rigid_conversion is not None and rigid_surface_dst is not None:
        representation_manifest = os.path.join(paths["dir"], "rigid_object_representation.json")
        write_rigid_representation_manifest(
            representation_manifest,
            conversion_result=rigid_conversion,
            generated_surface_path=rigid_surface_dst,
            mesh_scale=flex_scale,
            object_mass=float(object_mass),
            object_inertia=object_inertia,
            object_pos=object_pos,
        )
        paths["rigid_representation_manifest"] = representation_manifest
        print(f"[OK] Wrote rigid representation manifest: {representation_manifest}")
    
    # Copy shared assets needed by the standalone environment
    assets_dir = os.path.dirname(base_xml)
    for fname in ("shared.xml", "shared_asset.xml"):
        src = os.path.join(assets_dir, fname)
        dst = os.path.join(paths["dir"], fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Required asset file not found: {src}")
        if force or (not os.path.exists(dst)):
            shutil.copy2(src, dst)
            print(f"[OK] Copied asset: {dst}")
        else:
            print(f"[SKIP] Using cached asset: {dst}")

    if deformable_object:
        patch_shared_for_deformable(os.path.join(paths["dir"], "shared.xml"), deformable_preset=deformable_preset)
        print(f"[OK] Patched deformable shared limits: {os.path.join(paths['dir'], 'shared.xml')}")

    # Copy mesh assets referenced by shared_asset.xml
    shared_asset_path = os.path.join(paths["dir"], "shared_asset.xml")
    asset_tree = ET.parse(shared_asset_path)
    mesh_files = {m.get("file") for m in asset_tree.findall(".//mesh") if m.get("file")}

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(base_xml)))
    mesh_src_dir = os.path.join(repo_root, "stls", "hand")
    mesh_dst_dir = os.path.join(out_root, "stls", "hand")
    os.makedirs(mesh_dst_dir, exist_ok=True)

    for mesh in mesh_files:
        src = os.path.join(mesh_src_dir, mesh)
        dst = os.path.join(mesh_dst_dir, os.path.basename(mesh))
        if not os.path.exists(src):
            raise FileNotFoundError(f"Mesh file referenced in shared_asset.xml not found: {src}")
        if force or (not os.path.exists(dst)):
            shutil.copy2(src, dst)
            print(f"[OK] Copied mesh: {dst}")
        else:
            print(f"[SKIP] Using cached mesh: {dst}")

    # Copy texture assets referenced by the standalone env
    env_tree = ET.parse(paths["env"])
    texture_files = {t.get("file") for t in env_tree.findall(".//texture") if t.get("file")}

    texture_src_dir = os.path.join(repo_root, "textures")
    texture_dst_dir = os.path.join(out_root, "textures")
    os.makedirs(texture_dst_dir, exist_ok=True)

    for tex in texture_files:
        src = os.path.join(texture_src_dir, tex)
        dst = os.path.join(texture_dst_dir, os.path.basename(tex))
        if not os.path.exists(src):
            raise FileNotFoundError(f"Texture file referenced in {paths['env']} not found: {src}")
        if force or (not os.path.exists(dst)):
            shutil.copy2(src, dst)
            print(f"[OK] Copied texture: {dst}")
        else:
            print(f"[SKIP] Using cached texture: {dst}")
    return paths

# ==========
# MAIN CLI
# ==========

def main():
    p = argparse.ArgumentParser(
        description="End-to-end: build sensors → layout sites → update main XML (legacy) OR emit standalone per-candidate env (no side effects)."
    )
    p.add_argument("--base",  required=True, help="Path to base hand XML (bodies + geoms), e.g., assets/hand_base.xml")

    # Task selection
    p.add_argument("--task", default="block",
                   help="Built-in task name (block/egg/pen) OR path to a custom .msh file (uses block template and patches object body).")
    p.add_argument(
        "--deformable",
        action="store_true",
        help="Patch object body as deformable. Supports custom .msh tasks and built-in egg.",
    )
    p.add_argument(
        "--deformable-preset",
        choices=deformable_preset_names(),
        default=DEFAULT_DEFORMABLE_PRESET,
        help="Named rubber-like deformable material/contact/solver preset.",
    )

    # Legacy/in-place mode
    p.add_argument("--main",  help="Path to main task XML to update includes, e.g., assets/manipulate_block_touch_sensors.xml")
    p.add_argument("--out-dir", default=None, help="Directory to write shared/robot in legacy mode (and update --main includes)")

    # Standalone mode
    p.add_argument("--standalone", action="store_true", help="Generate a per-candidate folder with shared, robot, and a standalone env (no side effects).")
    p.add_argument("--template", help="Template task XML used to create the standalone env. If omitted, a default matching --task is used, or --main if provided.")
    p.add_argument("--out-root", default="generated", help="Root folder for standalone candidates (each under <N>_<r1>_<r2>/).")
    p.add_argument("--rigid-mesh-cache", default=None,
                   help="Shared source-hash cache for converted rigid OBJ surfaces.")
    p.add_argument("--force", action="store_true", help="Overwrite/cached outputs for this candidate.")

    # Allocation / areas / ratios
    p.add_argument("--Ntotal", type=int, required=True)
    p.add_argument("--Rppx", type=float, required=True, help="Palm : Phalanx ratio scale")
    p.add_argument("--Rpt",  type=float, required=True, help="Palm : Tip ratio scale")
    p.add_argument("--Ap",   type=float, default= 6557, help="Area weight: Palm")
    p.add_argument("--Apx",  type=float, default=26885, help="Area weight: Phalanx")
    p.add_argument("--At",   type=float, default=7193, help="Area weight: Tips")
    p.add_argument("--Ap1",  type=float, default=5557, help="Palm sub-area 1 (palm)")
    p.add_argument("--Ap2",  type=float, default=1000, help="Palm sub-area 2 (lfmetacarpal)")

    # legacy safety
    p.add_argument("--backup", action="store_true", help="(Legacy mode) Backup main XML to .bak before editing")

    args = p.parse_args()
    task_cfg = parse_task_arg(args.task)
    if args.deformable and task_cfg["custom_msh"] is None:
        if task_cfg["template_task"] != "egg":
            sys.exit("ERROR: built-in --deformable currently supports --task egg only.")
        task_cfg["task_label"] = "egg_deformable"

    size_label = infer_custom_object_size_label(task_cfg["custom_msh"]) or "medium"
    size_multiplier = SIZE_SCALE_MULTIPLIERS[size_label]
    flex_scale = _scale_triplet(0.025, size_multiplier)
    preset = get_deformable_preset(args.deformable_preset) if args.deformable else None
    flex_radius_base = float(preset["flex_radius"]) if preset else 0.001
    mass_base = float(preset["mass"]) if preset else BASE_RIGID_MASS
    flex_radius = _scale_scalar(flex_radius_base, size_multiplier)
    object_pos = (
        deformable_preset_spawn_position(args.deformable_preset, size_label)
        if args.deformable
        else infer_custom_object_spawn_position(size_label, False)
    )
    object_mass = _scale_mass(mass_base, size_multiplier)
    object_inertia = _scale_diaginertia(
        BASE_DEFORMABLE_DIAGINERTIA if args.deformable else BASE_RIGID_DIAGINERTIA,
        size_multiplier,
    )

    # Decide mode
    if args.standalone:
        template_xml = resolve_task_template(task_cfg["template_task"], args.template, args.main)
        paths = build_candidate_standalone(
            task=task_cfg["task_label"],
            Ntotal=args.Ntotal, Rppx=args.Rppx, Rpt=args.Rpt,
            Ap=args.Ap, Apx=args.Apx, At=args.At, Ap1=args.Ap1, Ap2=args.Ap2,
            base_xml=args.base, template_xml=template_xml,
            out_root=args.out_root, force=args.force,
            custom_msh=task_cfg["custom_msh"],
            custom_msh_name=None,
            deformable_object=args.deformable,
            flex_scale=flex_scale,
            flex_radius=flex_radius,
            object_pos=object_pos,
            object_mass=object_mass,
            object_inertia=object_inertia,
            deformable_preset=args.deformable_preset,
            rigid_mesh_cache_dir=args.rigid_mesh_cache,
        )
        # Emit a small machine-friendly summary for BO loops
        print(json.dumps(paths, indent=2))
        return

    # Legacy/in-place mode
    if not args.main:
        sys.exit("ERROR: legacy mode requires --main (the task XML to update). Use --standalone for no-side-effects generation.")
    if not args.out_dir:
        sys.exit("ERROR: legacy mode requires --out-dir (where to put shared/robot). Use --standalone to avoid editing files.")

    shared_path, robot_path = make_output_names(args.out_dir, args.Ntotal, args.Rppx, args.Rpt)
    build_shared_and_robot(args.Ap, args.Apx, args.At, args.Ntotal, args.Rppx, args.Rpt, args.Ap1, args.Ap2, args.base, shared_path, robot_path, force=args.force)

    if args.backup:
        with open(args.main, "rb") as src, open(args.main + ".bak", "wb") as dst:
            dst.write(src.read())
        print(f"[OK] Backup saved: {args.main}.bak")

    main_tree = ET.parse(args.main)
    counts = update_includes_by_prefix(
        main_tree,
        new_shared_basename=os.path.basename(shared_path),
        new_robot_basename=os.path.basename(robot_path)
    )
    main_tree.write(args.main, encoding="utf-8", xml_declaration=True)
    print(f"[OK] Updated main includes in: {args.main}")
    print(f"     Replaced: shared={counts['shared_updated']} robot={counts['robot_updated']}")

if __name__ == "__main__":
    main()
