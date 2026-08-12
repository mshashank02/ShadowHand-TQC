import csv
import json
import math
import os
import re
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import yaml
except ImportError:  # pragma: no cover - exercised in runtime, not tests
    yaml = None


ALLOWED_N_VALUES = (10, 50, 100, 200, 500, 1000)
ALLOWED_RATIO_VALUES = tuple(round(x / 10.0, 1) for x in range(1, 10))
PHYSICS_MODES = ("deformable", "rigid")
ASPECT_RATIO_VALUES = ("low", "high")
SIZE_VALUES = ("small", "medium", "large")
NATIVE_RIGID_TASKS = ("block", "egg", "pen")


@dataclass(frozen=True)
class StudyObject:
    object_id: str
    msh_file: str
    base_object: str
    aspect_ratio: str
    size: str
    abs_msh_path: str
    native_task: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    N: int
    alpha: float
    beta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "N": self.N,
            "alpha": self.alpha,
            "beta": self.beta,
        }


@dataclass(frozen=True)
class HostConfig:
    host: str
    ssh_target: str
    gpu_count: int
    cpu_cores: Optional[int]
    num_envs_per_job: Optional[int]
    work_root: str
    python_root: str
    priority: int
    repo_path: str
    python_bin: str
    role: str = "worker"
    run_worker: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_worker(self) -> "HostConfig":
        return HostConfig(
            host=self.host,
            ssh_target=self.ssh_target,
            gpu_count=self.gpu_count,
            cpu_cores=self.cpu_cores,
            num_envs_per_job=self.num_envs_per_job,
            work_root=self.work_root,
            python_root=self.python_root,
            priority=self.priority,
            repo_path=self.repo_path,
            python_bin=self.python_bin,
            role="worker",
            run_worker=True,
        )

    def resolved_num_envs_per_job(self) -> Optional[int]:
        if self.num_envs_per_job is not None:
            return max(1, int(self.num_envs_per_job))
        if self.cpu_cores is None or self.gpu_count <= 0:
            return None
        return max(1, int((int(self.cpu_cores) - 2) // int(self.gpu_count)))

    def resolved_cpu_threads_per_job(self) -> Optional[int]:
        if self.cpu_cores is None or self.gpu_count <= 0:
            return None
        return max(1, int((int(self.cpu_cores) - 2) // int(self.gpu_count)))


@dataclass(frozen=True)
class ClusterConfig:
    coordinator: HostConfig
    hosts: List[HostConfig]
    repo_dirname: str

    def worker_hosts(self) -> List[HostConfig]:
        workers = [host for host in self.hosts if host.role != "coordinator"]
        if self.coordinator.run_worker and self.coordinator.gpu_count > 0:
            workers = [self.coordinator.as_worker(), *workers]
        return self._dedupe_hosts(workers)

    def all_hosts(self) -> List[HostConfig]:
        return self._dedupe_hosts([self.coordinator, *self.hosts])

    @staticmethod
    def _dedupe_hosts(hosts: Sequence[HostConfig]) -> List[HostConfig]:
        deduped: List[HostConfig] = []
        seen: set[str] = set()
        for host in hosts:
            if host.host in seen:
                continue
            seen.add(host.host)
            deduped.append(host)
        return deduped


def sanitize_identifier(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return text or "item"


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_repo_dirname() -> str:
    return resolve_repo_root().name


def resolve_study_root(study_name: str, repo_root: Optional[Path] = None) -> Path:
    root = repo_root or resolve_repo_root()
    return root / "generated" / "studies" / sanitize_identifier(study_name)


def build_candidate_id(N: int, alpha: float, beta: float) -> str:
    alpha_tag = str(alpha).replace(".", "p")
    beta_tag = str(beta).replace(".", "p")
    return f"n{int(N):04d}_a{alpha_tag}_b{beta_tag}"


def build_run_label(candidate_id: str, object_id: str, physics_mode: str) -> str:
    return sanitize_identifier(f"{candidate_id}_{object_id}_{physics_mode}")


def build_run_artifact_relpath(
    study_name: str,
    candidate_id: str,
    object_id: str,
    physics_mode: str,
    study_root: Optional[str | Path] = None,
) -> str:
    root_parts = (
        [str(study_root)]
        if study_root is not None
        else ["generated", "studies", sanitize_identifier(study_name)]
    )
    return os.path.join(
        *root_parts,
        "candidates",
        candidate_id,
        sanitize_identifier(object_id),
        sanitize_identifier(physics_mode),
    )


def build_candidate_grid() -> List[Candidate]:
    grid: List[Candidate] = []
    for N in ALLOWED_N_VALUES:
        for alpha in ALLOWED_RATIO_VALUES:
            for beta in ALLOWED_RATIO_VALUES:
                grid.append(Candidate(build_candidate_id(N, alpha, beta), N, alpha, beta))
    return grid


def quantize_candidate(N: float, alpha: float, beta: float) -> Candidate:
    n_value = min(ALLOWED_N_VALUES, key=lambda v: abs(v - float(N)))
    alpha_value = min(ALLOWED_RATIO_VALUES, key=lambda v: abs(v - float(alpha)))
    beta_value = min(ALLOWED_RATIO_VALUES, key=lambda v: abs(v - float(beta)))
    return Candidate(build_candidate_id(n_value, alpha_value, beta_value), n_value, alpha_value, beta_value)


def sobol_initial_candidates(k: int, excluded_ids: Optional[Iterable[str]] = None) -> List[Candidate]:
    if k <= 0:
        return []
    excluded = set(excluded_ids or [])
    selected: List[Candidate] = []
    seen = set(excluded)
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
    batch = max(16, k * 4)
    while len(selected) < k:
        pts = engine.draw(batch).double()
        for point in pts:
            candidate = quantize_candidate(
                N=float(point[0].item()) * (max(ALLOWED_N_VALUES) - min(ALLOWED_N_VALUES)) + min(ALLOWED_N_VALUES),
                alpha=float(point[1].item()) * 0.8 + 0.1,
                beta=float(point[2].item()) * 0.8 + 0.1,
            )
            if candidate.candidate_id in seen:
                continue
            seen.add(candidate.candidate_id)
            selected.append(candidate)
            if len(selected) >= k:
                break
    return selected


def map_alpha_beta_to_ratios(N: int, alpha: float, beta: float) -> Tuple[int, int, int, int, float, float]:
    N = int(N)
    Np = int(round(alpha * N))
    N_non = max(N - Np, 0)
    Nt = int(round(beta * N_non))
    Nx = max(N_non - Nt, 0)
    eps = 1e-6
    Rppx = float(Np) / float(max(Nx, eps))
    Rpt = float(Np) / float(max(Nt, eps))
    return N, Np, Nt, Nx, Rppx, Rpt


def _validate_choice(value: str, allowed: Sequence[str], field_name: str, row_idx: int) -> str:
    normalized = sanitize_identifier(value)
    if normalized not in allowed:
        raise ValueError(f"Row {row_idx}: invalid {field_name}={value!r}; expected one of {allowed}.")
    return normalized


def load_study_manifest(objects_root: str, expected_base_objects: Optional[int] = 6) -> List[StudyObject]:
    root = Path(objects_root).expanduser().resolve()
    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Study manifest not found: {manifest_path}")

    rows: List[StudyObject] = []
    seen_ids = set()
    combos_by_base: Dict[str, set[Tuple[str, str]]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"object_id", "base_object", "aspect_ratio", "size"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest.csv missing required columns: {sorted(missing)}")
        if "msh_file" not in fieldnames and "native_task" not in fieldnames:
            raise ValueError("manifest.csv must include 'msh_file' and/or 'native_task'.")

        for idx, row in enumerate(reader, start=2):
            object_id = sanitize_identifier(row["object_id"])
            if object_id in seen_ids:
                raise ValueError(f"Duplicate object_id {object_id!r} in manifest.csv")
            seen_ids.add(object_id)

            base_object = sanitize_identifier(row["base_object"])
            aspect_ratio = _validate_choice(row["aspect_ratio"], ASPECT_RATIO_VALUES, "aspect_ratio", idx)
            size = _validate_choice(row["size"], SIZE_VALUES, "size", idx)
            msh_file = (row.get("msh_file") or "").strip()
            native_task_raw = (row.get("native_task") or "").strip()
            native_task = sanitize_identifier(native_task_raw) if native_task_raw else None
            if bool(msh_file) == bool(native_task):
                raise ValueError(
                    f"Row {idx}: provide exactly one of msh_file or native_task."
                )
            if native_task is not None:
                if native_task not in NATIVE_RIGID_TASKS:
                    raise ValueError(
                        f"Row {idx}: invalid native_task={native_task!r}; "
                        f"expected one of {NATIVE_RIGID_TASKS}."
                    )
                abs_msh_path = None
            else:
                abs_msh_path = (root / msh_file).resolve()
                if not abs_msh_path.is_file():
                    raise FileNotFoundError(f"Row {idx}: mesh file not found: {abs_msh_path}")

            if native_task is None:
                combos_by_base.setdefault(base_object, set()).add((aspect_ratio, size))
            rows.append(
                StudyObject(
                    object_id=object_id,
                    msh_file=msh_file,
                    base_object=base_object,
                    aspect_ratio=aspect_ratio,
                    size=size,
                    abs_msh_path=str(abs_msh_path) if abs_msh_path is not None else "",
                    native_task=native_task,
                )
            )

    if not rows:
        raise ValueError(f"No rows found in {manifest_path}")

    base_objects = {row.base_object for row in rows}
    if expected_base_objects is not None and len(base_objects) != expected_base_objects:
        raise ValueError(
            f"Expected {expected_base_objects} base objects, found {len(base_objects)} in {manifest_path}"
        )

    sizes_present = {row.size for row in rows if row.native_task is None}
    expected_sizes = tuple(size for size in SIZE_VALUES if size in sizes_present)
    expected_combos = {(aspect_ratio, size) for aspect_ratio in ASPECT_RATIO_VALUES for size in expected_sizes}
    for base_object, combos in combos_by_base.items():
        if combos != expected_combos:
            raise ValueError(
                f"Base object {base_object!r} has combos {sorted(combos)}; expected {sorted(expected_combos)}"
            )

    return sorted(rows, key=lambda row: (row.base_object, row.aspect_ratio, row.size, row.object_id))


def expand_jobs_for_candidate(
    study_objects: Sequence[StudyObject],
    physics_modes: Sequence[str] = PHYSICS_MODES,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for obj in study_objects:
        for physics_mode in physics_modes:
            jobs.append(
                {
                    "object_id": obj.object_id,
                    "msh_file": obj.msh_file,
                    "native_task": obj.native_task,
                    "base_object": obj.base_object,
                    "aspect_ratio": obj.aspect_ratio,
                    "size": obj.size,
                    "physics_mode": physics_mode,
                    "seed": int(seed),
                }
            )
    return jobs


_SCORE_PATTERNS = [
    r"FINAL_SCORE[:=]\s*([0-9]*\.?[0-9]+)",
    r"mean[_\s-]?success[^0-9]*([0-9]*\.?[0-9]+)",
    r"success[_\s-]*rate[^0-9]*([0-9]*\.?[0-9]+)",
    r"eval[/\s]*success[^0-9]*([0-9]*\.?[0-9]+)",
]


def parse_score_from_text(text: str) -> Optional[float]:
    for pattern in _SCORE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            continue
    return None


def load_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def compute_scalar_from_metrics(
    metrics: dict,
    theta: float = 0.8,
    w_final: float = 0.5,
    w_aulc: float = 0.3,
    w_speed: float = 0.2,
    w_disp: float = 0.1,
) -> float:
    if "score" in metrics and not metrics.get("tasks"):
        return float(metrics["score"])

    tasks = metrics.get("tasks", [])
    if not tasks:
        raise ValueError("metrics.json missing 'tasks' or 'score'.")

    checkpoints = metrics.get("checkpoints")
    finals: List[float] = []
    aulcs: List[float] = []
    speeds: List[float] = []
    dispersions: List[float] = []

    for task in tasks:
        final_success = float(metrics["final_success"][task])
        finals.append(final_success)

        if checkpoints is not None and "success" in metrics and task in metrics["success"]:
            curve = [float(value) for value in metrics["success"][task]]
            x_axis = [float(value) for value in checkpoints]
            if len(curve) >= 2 and x_axis[-1] > x_axis[0]:
                trapezoid = getattr(np, "trapezoid", None)
                if trapezoid is None:  # NumPy < 2.0
                    trapezoid = np.trapz
                aulc = float(trapezoid(curve, x=x_axis) / (x_axis[-1] - x_axis[0]))
            else:
                aulc = float(np.mean(curve)) if curve else final_success
            try:
                idx = next(i for i, value in enumerate(curve) if value >= theta)
                speed = 1.0 - x_axis[idx]
            except StopIteration:
                speed = 0.0
        else:
            aulc = final_success
            speed = 0.0

        if "final_success_seeds" in metrics and task in metrics["final_success_seeds"]:
            values = [float(value) for value in metrics["final_success_seeds"][task]]
            dispersion = float(np.std(values)) if len(values) >= 2 else 0.0
        else:
            dispersion = 0.0

        aulcs.append(aulc)
        speeds.append(speed)
        dispersions.append(dispersion)

    perf = np.mean(w_final * np.array(finals) + w_aulc * np.array(aulcs) + w_speed * np.array(speeds))
    robustness_penalty = w_disp * np.mean(dispersions)
    return float(perf - robustness_penalty)


def load_score_from_artifacts(metrics_path: str, stdout_path: Optional[str] = None, stderr_path: Optional[str] = None) -> Optional[float]:
    metrics = load_json(metrics_path)
    if metrics is not None:
        return compute_scalar_from_metrics(metrics)

    for path in (stdout_path, stderr_path):
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    parsed = parse_score_from_text(handle.read())
                if parsed is not None:
                    return parsed
            except Exception:
                continue
    return None


def summarize_candidate_scores(job_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not job_rows:
        raise ValueError("Cannot summarize an empty candidate.")

    overall_scores = [float(row["score"]) for row in job_rows]
    by_physics: Dict[str, List[float]] = {}
    by_base_object: Dict[str, List[float]] = {}
    by_aspect_ratio: Dict[str, List[float]] = {}
    by_size: Dict[str, List[float]] = {}

    for row in job_rows:
        by_physics.setdefault(row["physics_mode"], []).append(float(row["score"]))
        by_base_object.setdefault(row["base_object"], []).append(float(row["score"]))
        by_aspect_ratio.setdefault(row["aspect_ratio"], []).append(float(row["score"]))
        by_size.setdefault(row["size"], []).append(float(row["score"]))

    def _mean_map(values: Dict[str, List[float]]) -> Dict[str, float]:
        return {key: float(np.mean(series)) for key, series in sorted(values.items())}

    return {
        "score": float(np.mean(overall_scores)),
        "physics_means": _mean_map(by_physics),
        "base_object_means": _mean_map(by_base_object),
        "aspect_ratio_means": _mean_map(by_aspect_ratio),
        "size_means": _mean_map(by_size),
    }


def _normalize_host_entry(entry: Dict[str, Any], repo_dirname: str, role: str = "worker") -> HostConfig:
    required = {"host", "ssh_target", "gpu_count", "work_root", "python_root", "priority"}
    missing = required.difference(entry.keys())
    if missing:
        raise ValueError(f"Host config missing keys {sorted(missing)}: {entry}")

    work_root = os.path.expanduser(str(entry["work_root"]))
    python_root = os.path.expanduser(str(entry["python_root"]))
    python_bin = python_root
    if not python_bin.endswith("python"):
        python_bin = os.path.join(python_root, "bin", "python")
    repo_path = entry.get("repo_path") or os.path.join(work_root, repo_dirname)

    return HostConfig(
        host=str(entry["host"]),
        ssh_target=str(entry["ssh_target"]),
        gpu_count=int(entry["gpu_count"]),
        cpu_cores=int(entry["cpu_cores"]) if entry.get("cpu_cores") is not None else None,
        num_envs_per_job=int(entry["num_envs_per_job"]) if entry.get("num_envs_per_job") is not None else None,
        work_root=work_root,
        python_root=python_root,
        priority=int(entry["priority"]),
        repo_path=os.path.expanduser(str(repo_path)),
        python_bin=os.path.expanduser(str(entry.get("python_bin") or python_bin)),
        role=str(entry.get("role", role)),
        run_worker=bool(entry.get("run_worker", False)),
    )


def load_cluster_config(path: str, repo_dirname: Optional[str] = None) -> ClusterConfig:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load cluster_hosts.yaml. Install pyyaml first.")

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if "coordinator" not in raw or "hosts" not in raw:
        raise ValueError("cluster_hosts.yaml must contain top-level 'coordinator' and 'hosts' keys.")

    repo_dir = repo_dirname or resolve_repo_dirname()
    coordinator = _normalize_host_entry(raw["coordinator"], repo_dir, role="coordinator")
    hosts = [_normalize_host_entry(entry, repo_dir) for entry in raw["hosts"]]
    return ClusterConfig(coordinator=coordinator, hosts=hosts, repo_dirname=repo_dir)


def detect_local_host_aliases() -> set[str]:
    aliases = {
        socket.gethostname(),
        socket.getfqdn(),
        sanitize_identifier(socket.gethostname()),
        sanitize_identifier(socket.getfqdn()),
    }
    return {alias for alias in aliases if alias}
