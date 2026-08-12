"""Machine-readable capability report for the optional direct GPU backend."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
from typing import Any


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def collect_capabilities(xml_path: str | Path | None = None) -> dict[str, Any]:
    versions = {
        "mujoco": _package_version("mujoco"),
        "mujoco_warp": _package_version("mujoco-warp"),
        "warp": _package_version("warp-lang"),
        "torch": _package_version("torch"),
    }
    result: dict[str, Any] = {
        "versions": versions,
        "direct_backend_importable": bool(versions["mujoco_warp"] and versions["warp"]),
    }

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        import torch

        result["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            result["torch_cuda"] = {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
                "torch_cuda_version": torch.version.cuda,
            }
    else:
        result["torch_cuda_available"] = False

    if xml_path is not None and versions["mujoco"] is not None:
        from .model_loader import load_project_model
        from .sensors import build_sensor_layout

        model, report = load_project_model(xml_path)
        result["model"] = report.to_dict()
        result["sensor_layout"] = build_sensor_layout(model).to_dict(
            include_sensors=False,
            include_touch_entries=False,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, help="Optional generated model to inspect")
    parser.add_argument("--probe-step", action="store_true", help="Allocate and step one CUDA world")
    parser.add_argument("--contacts-per-world", type=int, default=8192)
    parser.add_argument("--constraints-per-world", type=int, default=4096)
    args = parser.parse_args()

    report = collect_capabilities(args.xml)
    exit_code = 0
    if args.probe_step:
        if args.xml is None:
            parser.error("--probe-step requires --xml")
        try:
            from .warp_backend import MujocoWarpBackend

            backend = MujocoWarpBackend(
                args.xml,
                worlds=1,
                contacts_per_world=args.contacts_per_world,
                constraints_per_world=args.constraints_per_world,
            )
            backend.step()
            backend.synchronize()
            report["step_probe"] = {
                "ok": True,
                "backend": backend.report(),
                "active_contacts": backend.active_contact_counts.cpu().tolist(),
                "constraints": backend.constraint_counts.cpu().tolist(),
            }
        except Exception as exc:  # CLI must preserve diagnostics as JSON.
            report["step_probe"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            exit_code = 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
