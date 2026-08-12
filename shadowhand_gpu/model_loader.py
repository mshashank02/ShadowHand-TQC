"""Load project XMLs under both the reference and current MuJoCo schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable
import xml.etree.ElementTree as ET


_APIRATE_RE = re.compile(r"\s+apirate=(?:\"[^\"]*\"|'[^']*')")
_MESH_ASSET_TAGS = {"mesh", "skin", "flexcomp"}
_TEXTURE_ASSET_TAGS = {"texture", "hfield"}


@dataclass(frozen=True)
class ModelLoadReport:
    xml_path: str
    mujoco_version: str
    compatibility_changes: tuple[str, ...]
    nq: int
    nv: int
    nu: int
    nbody: int
    ngeom: int
    nsite: int
    nsensor: int
    nsensordata: int
    nflex: int
    nflexvert: int
    nflexedge: int
    rigid_flex: bool
    object_collision_representation: str
    gpu_collision_support: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_legacy_attributes(text: str) -> tuple[str, bool]:
    sanitized, count = _APIRATE_RE.subn("", text)
    return sanitized, bool(count)


def _version_at_least(version: str, minimum: tuple[int, int]) -> bool:
    numeric: list[int] = []
    for part in version.split(".")[:2]:
        match = re.match(r"\d+", part)
        numeric.append(int(match.group(0)) if match else 0)
    while len(numeric) < 2:
        numeric.append(0)
    return tuple(numeric) >= minimum


def _add_asset(assets: dict[str, bytes], path: Path) -> None:
    """Add an asset using MuJoCo VFS basename semantics."""
    data = path.read_bytes()
    old = assets.get(path.name)
    if old is not None and old != data:
        raise ValueError(
            f"MuJoCo virtual assets contain conflicting basename {path.name!r}: "
            f"at least one source is {path}"
        )
    assets[path.name] = data


def _resolve_existing(candidates: Iterable[Path], description: str) -> Path:
    tried: list[str] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        tried.append(str(resolved))
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Could not resolve {description}; tried: {', '.join(tried)}")


def _compiler_directories(main_root: ET.Element, xml_dir: Path) -> tuple[Path | None, Path | None]:
    compiler = main_root.find("compiler")
    if compiler is None:
        return None, None
    meshdir = compiler.get("meshdir")
    texturedir = compiler.get("texturedir")
    return (
        (xml_dir / meshdir).resolve() if meshdir else None,
        (xml_dir / texturedir).resolve() if texturedir else None,
    )


def _collect_virtual_assets(xml_path: Path, main_text: str) -> dict[str, bytes]:
    """Collect includes and referenced binary assets without rewriting the XML tree."""
    assets: dict[str, bytes] = {}
    main_root = ET.fromstring(main_text)
    meshdir, texturedir = _compiler_directories(main_root, xml_path.parent)
    pending: list[tuple[Path, str]] = [(xml_path, main_text)]
    seen_xml: set[Path] = set()

    while pending:
        source_path, source_text = pending.pop()
        source_path = source_path.resolve()
        if source_path in seen_xml:
            continue
        seen_xml.add(source_path)
        root = ET.fromstring(source_text)

        for element in root.iter():
            file_value = element.get("file")
            if not file_value:
                continue

            if element.tag == "include":
                include_path = _resolve_existing(
                    [source_path.parent / file_value, xml_path.parent / file_value],
                    f"include {file_value!r} from {source_path}",
                )
                include_text, _ = _strip_legacy_attributes(include_path.read_text(encoding="utf-8"))
                old = assets.get(include_path.name)
                include_bytes = include_text.encode("utf-8")
                if old is not None and old != include_bytes:
                    raise ValueError(f"Conflicting XML include basename {include_path.name!r}")
                assets[include_path.name] = include_bytes
                pending.append((include_path, include_text))
                continue

            candidates = [source_path.parent / file_value, xml_path.parent / file_value]
            if element.tag in _MESH_ASSET_TAGS and meshdir is not None:
                candidates.insert(0, meshdir / file_value)
            if element.tag in _TEXTURE_ASSET_TAGS and texturedir is not None:
                candidates.insert(0, texturedir / file_value)
            asset_path = _resolve_existing(
                candidates,
                f"asset {file_value!r} referenced by <{element.tag}> in {source_path}",
            )
            _add_asset(assets, asset_path)

    return assets


def is_single_body_rigid_flex(model: Any) -> bool:
    """Return whether every flex edge has both vertices attached to one rigid body."""
    if int(getattr(model, "nflex", 0)) == 0:
        return False

    import numpy as np

    flex_edge = np.asarray(model.flex_edge)
    vert_body = np.asarray(model.flex_vertbodyid)
    for flex_id in range(int(model.nflex)):
        vert_adr = int(model.flex_vertadr[flex_id])
        edge_adr = int(model.flex_edgeadr[flex_id])
        edge_num = int(model.flex_edgenum[flex_id])
        edges = flex_edge[edge_adr : edge_adr + edge_num]
        if edges.size == 0:
            continue
        body0 = vert_body[vert_adr + edges[:, 0]]
        body1 = vert_body[vert_adr + edges[:, 1]]
        if np.any(body0 < 0) or np.any(body0 != body1):
            return False
    return True


def classify_collision_representation(mujoco: Any, model: Any) -> tuple[str, str]:
    """Classify support from compiled model structure, never filenames or CLI labels."""
    if int(getattr(model, "nflex", 0)):
        rigid_flags = [bool(value) for value in model.flex_rigid]
        if rigid_flags and all(rigid_flags):
            return "rigid_flex", "rigid_flex_not_production_validated"
        return "deformable_flex", "deformable_flex_experimental"

    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if object_body_id >= 0:
        geom_ids = [
            geom_id
            for geom_id in range(int(model.ngeom))
            if int(model.geom_bodyid[geom_id]) == int(object_body_id)
            and int(model.geom_contype[geom_id]) != 0
        ]
        mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
        if any(int(model.geom_type[geom_id]) == mesh_type for geom_id in geom_ids):
            return "rigid_mesh_geom", "gpu_rigid_supported"
        if geom_ids:
            return "native_rigid_geom", "gpu_rigid_supported"
    return "rigid_geom_model", "gpu_rigid_supported"


def _apply_reference_collision_defaults(mujoco: Any, model: Any) -> list[str]:
    """Disable CCD modes that were not enabled in the MuJoCo 3.3 reference model."""
    changes: list[str] = []
    disable_enum = getattr(mujoco, "mjtDisableBit", None)
    if disable_enum is None:
        return changes
    for enum_name, label in (
        ("mjDSBL_MULTICCD", "disabled MULTICCD to match MuJoCo 3.3 reference defaults"),
        ("mjDSBL_NATIVECCD", "disabled NATIVECCD to match MuJoCo 3.3 reference defaults"),
    ):
        bit = getattr(disable_enum, enum_name, None)
        if bit is not None and not (int(model.opt.disableflags) & int(bit)):
            model.opt.disableflags |= int(bit)
            changes.append(label)
    return changes


def load_project_model(xml_path: str | Path, *, reference_compat: bool = True) -> tuple[Any, ModelLoadReport]:
    """Load a model while preserving the effective MuJoCo 3.3 project settings.

    MuJoCo 3.11 removed the non-dynamical ``option.apirate`` attribute. When it is
    present, the XML and its assets are loaded through MuJoCo's in-memory VFS with
    only that attribute removed. Generated XML files on disk are never modified.
    """
    import mujoco

    path = Path(xml_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    original_text = path.read_text(encoding="utf-8")
    sanitized_text, removed_apirate = _strip_legacy_attributes(original_text)
    changes: list[str] = []
    if removed_apirate and _version_at_least(str(mujoco.__version__), (3, 11)):
        assets = _collect_virtual_assets(path, sanitized_text)
        model = mujoco.MjModel.from_xml_string(sanitized_text, assets=assets)
        changes.append("removed obsolete option.apirate in memory")
    else:
        model = mujoco.MjModel.from_xml_path(str(path))

    if reference_compat:
        changes.extend(_apply_reference_collision_defaults(mujoco, model))

    representation, gpu_collision_support = classify_collision_representation(mujoco, model)
    report = ModelLoadReport(
        xml_path=str(path),
        mujoco_version=str(mujoco.__version__),
        compatibility_changes=tuple(changes),
        nq=int(model.nq),
        nv=int(model.nv),
        nu=int(model.nu),
        nbody=int(model.nbody),
        ngeom=int(model.ngeom),
        nsite=int(model.nsite),
        nsensor=int(model.nsensor),
        nsensordata=int(model.nsensordata),
        nflex=int(model.nflex),
        nflexvert=int(model.nflexvert),
        nflexedge=int(model.nflexedge),
        rigid_flex=is_single_body_rigid_flex(model),
        object_collision_representation=representation,
        gpu_collision_support=gpu_collision_support,
    )
    return model, report
