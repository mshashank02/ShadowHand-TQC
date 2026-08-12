import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import IO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from study_common import (
    HostConfig,
    build_run_label,
    load_cluster_config,
    load_score_from_artifacts,
    resolve_repo_root,
    sanitize_identifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-lived distributed study worker.")
    parser.add_argument("--cluster-config", default="cluster_hosts.yaml")
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--host-name", default=None, help="Logical host name from cluster_hosts.yaml.")
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--lease-seconds", type=int, default=172800)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--free-memory-mib", type=int, default=512)
    parser.add_argument("--free-utilization", type=int, default=10)
    parser.add_argument("--consecutive-free-polls", type=int, default=2)
    parser.add_argument("--allow-foreign-processes", action="store_true",
                        help="Treat GPUs with foreign compute processes as eligible if memory/utilization are low.")
    return parser.parse_args()


def _run_command(command: Sequence[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)


def has_opt(opt: str, argv: Sequence[str]) -> bool:
    if opt in argv:
        return True
    return any(str(arg).startswith(opt + "=") for arg in argv)


def _expand_trainer_args(
    trainer_args: Sequence[object],
    *,
    job: Dict[str, object],
    artifact_abs: Path,
) -> List[str]:
    replacements = {
        "{artifact_root}": str(artifact_abs),
        "{candidate_id}": str(job["candidate_id"]),
        "{object_id}": str(job["object_id"]),
        "{physics_mode}": str(job["physics_mode"]),
    }
    expanded: List[str] = []
    for argument in trainer_args:
        value = str(argument)
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        expanded.append(value)
    return expanded


def validate_job_trainer(job: Dict[str, object], trainer_args: Sequence[str]) -> str:
    trainer = str(job.get("trainer", "cpu"))
    if has_opt("--trainer", trainer_args):
        raise ValueError(
            "Queued trainer_args cannot override the job's validated trainer field."
        )
    if trainer not in {"cpu", "gpu"}:
        raise ValueError(f"Unknown trainer {trainer!r} in queued job.")
    if trainer == "cpu":
        return trainer
    if str(job["physics_mode"]) != "rigid":
        raise ValueError("GPU queued jobs must use rigid physics mode.")
    if str(job.get("task_kind", "mesh")) not in {"native", "mesh"}:
        raise ValueError("GPU queued jobs require a native task or convertible custom mesh.")
    if has_opt("--auto-num-envs", trainer_args):
        if not has_opt("--auto-num-envs-report", trainer_args):
            raise ValueError("GPU --auto-num-envs requires --auto-num-envs-report.")
    elif not all(
        has_opt(option, trainer_args)
        for option in ("--num-envs", "--contacts-per-world", "--constraints-per-world")
    ):
        raise ValueError(
            "GPU queued jobs need a complete explicit allocation or --auto-num-envs."
        )
    if not (
        has_opt("--auto-replay-capacity", trainer_args)
        or has_opt("--buffer-size", trainer_args)
    ):
        raise ValueError("GPU queued jobs need an explicit replay-memory policy.")
    return trainer


def parse_gpu_query_output(raw_output: str) -> List[Dict[str, int | str]]:
    rows = []
    for line in raw_output.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "memory_used_mib": int(parts[2]),
                "utilization_gpu": int(parts[3]),
            }
        )
    return rows


def parse_compute_query_output(raw_output: str) -> Set[str]:
    uuids: Set[str] = set()
    for line in raw_output.strip().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if not parts:
            continue
        uuids.add(parts[0])
    return uuids


def collect_gpu_snapshot() -> List[Dict[str, int | str | bool]]:
    gpu_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if gpu_query.returncode != 0:
        raise RuntimeError(f"nvidia-smi gpu query failed: {gpu_query.stderr.strip()}")

    compute_query = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    compute_uuids = parse_compute_query_output(compute_query.stdout if compute_query.returncode == 0 else "")
    snapshot = []
    for row in parse_gpu_query_output(gpu_query.stdout):
        row = dict(row)
        row["has_compute_process"] = row["uuid"] in compute_uuids
        snapshot.append(row)
    return snapshot


def free_gpu_ids(
    snapshot: Sequence[Dict[str, int | str | bool]],
    memory_threshold_mib: int,
    utilization_threshold: int,
    allow_foreign_processes: bool,
) -> Set[int]:
    free_ids: Set[int] = set()
    for row in snapshot:
        if int(row["memory_used_mib"]) >= int(memory_threshold_mib):
            continue
        if int(row["utilization_gpu"]) >= int(utilization_threshold):
            continue
        if bool(row["has_compute_process"]) and not allow_foreign_processes:
            continue
        free_ids.add(int(row["index"]))
    return free_ids


class FreeGpuTracker:
    def __init__(self, required_free_polls: int):
        self.required_free_polls = max(1, int(required_free_polls))
        self.free_counts: Dict[int, int] = {}

    def update(self, currently_free: Set[int]) -> Set[int]:
        eligible: Set[int] = set()
        tracked_ids = set(self.free_counts.keys()) | set(currently_free)
        for gpu_id in tracked_ids:
            if gpu_id in currently_free:
                self.free_counts[gpu_id] = self.free_counts.get(gpu_id, 0) + 1
            else:
                self.free_counts[gpu_id] = 0
            if self.free_counts[gpu_id] >= self.required_free_polls:
                eligible.add(gpu_id)
        return eligible


@dataclass
class RunningJob:
    job: Dict[str, object]
    process: subprocess.Popen
    artifact_abs: Path
    stdout_abs: Path
    stderr_abs: Path
    stdout_handle: IO[str]
    stderr_handle: IO[str]


class CoordinatorClient:
    def __init__(self, coordinator_cfg, study_name: str):
        self.coordinator_cfg = coordinator_cfg
        self.study_name = study_name
        self.db_path = os.path.join(
            coordinator_cfg.repo_path,
            "generated",
            "studies",
            sanitize_identifier(study_name),
            "study.db",
        )
        self.script_path = os.path.join(coordinator_cfg.repo_path, "study_queue.py")

    def _remote_command(self, args: Sequence[str]) -> List[str]:
        remote_command = "cd {repo} && {python} {script} --db {db} {args}".format(
            repo=shlex.quote(self.coordinator_cfg.repo_path),
            python=shlex.quote(self.coordinator_cfg.python_bin),
            script=shlex.quote(self.script_path),
            db=shlex.quote(self.db_path),
            args=" ".join(shlex.quote(str(arg)) for arg in args),
        )
        return ["ssh", self.coordinator_cfg.ssh_target, remote_command]

    def run_queue_command(self, args: Sequence[str]) -> Dict[str, object]:
        proc = _run_command(self._remote_command(args))
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
        stdout = proc.stdout.strip()
        return json.loads(stdout) if stdout else {}

    def heartbeat(self, host_name: str, worker_id: str, pid: int, details: Dict[str, object]) -> None:
        self.run_queue_command(
            [
                "heartbeat",
                "--host",
                host_name,
                "--worker-id",
                worker_id,
                "--pid",
                str(pid),
                "--details-json",
                json.dumps(details, sort_keys=True),
            ]
        )

    def claim_job(self, host_name: str, worker_id: str, gpu_id: int, lease_seconds: int, max_attempts: int) -> Optional[Dict[str, object]]:
        payload = self.run_queue_command(
            [
                "claim",
                "--host",
                host_name,
                "--worker-id",
                worker_id,
                "--gpu-id",
                str(gpu_id),
                "--lease-seconds",
                str(lease_seconds),
                "--max-attempts",
                str(max_attempts),
            ]
        )
        return payload or None

    def finish_job(self, job_id: int, score: float, job: Dict[str, object]) -> None:
        self.run_queue_command(
            [
                "finish",
                "--job-id",
                str(job_id),
                "--score",
                str(score),
                "--metrics-relpath",
                str(job["metrics_relpath"]),
                "--stdout-relpath",
                str(job["stdout_relpath"]),
                "--stderr-relpath",
                str(job["stderr_relpath"]),
                "--artifact-relpath",
                str(job["artifact_relpath"]),
            ]
        )

    def fail_job(self, job_id: int, reason: str, max_attempts: int) -> None:
        self.run_queue_command(
            [
                "fail",
                "--job-id",
                str(job_id),
                "--reason",
                reason,
                "--max-attempts",
                str(max_attempts),
            ]
        )


def detect_host_name(cluster_cfg, explicit_host_name: Optional[str]) -> str:
    if explicit_host_name:
        return explicit_host_name
    local_aliases = {
        socket.gethostname(),
        socket.getfqdn(),
        sanitize_identifier(socket.gethostname()),
        sanitize_identifier(socket.getfqdn()),
    }
    for host_cfg in cluster_cfg.worker_hosts():
        if host_cfg.host in local_aliases or sanitize_identifier(host_cfg.host) in local_aliases:
            return host_cfg.host
    raise ValueError("Could not infer host name from cluster_hosts.yaml; pass --host-name explicitly.")


def ensure_remote_dir(ssh_target: str, remote_dir: str) -> None:
    proc = _run_command(["ssh", ssh_target, f"mkdir -p {shlex.quote(remote_dir)}"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())


def sync_artifacts_to_coordinator(job: Dict[str, object], artifact_abs: Path, coordinator_cfg) -> None:
    remote_dir = os.path.join(coordinator_cfg.repo_path, str(job["artifact_relpath"]))
    ensure_remote_dir(coordinator_cfg.ssh_target, remote_dir)
    proc = _run_command(
        [
            "rsync",
            "-az",
            f"{artifact_abs}/",
            f"{coordinator_cfg.ssh_target}:{remote_dir}/",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())


def build_job_command(job: Dict[str, object], repo_root: Path, host_cfg: HostConfig) -> List[str]:
    artifact_abs = repo_root / str(job["artifact_relpath"])
    task_kind = str(job.get("task_kind", "mesh"))
    if task_kind == "native":
        task_arg = str(job["task_arg"])
    else:
        task_relpath = str(job.get("task_arg", job.get("object_msh_relpath", "")))
        if not task_relpath:
            raise ValueError("Queued mesh job is missing task_arg/object_msh_relpath.")
        task_arg = str(repo_root / task_relpath)
    base_xml_abs = repo_root / str(job["base_xml_relpath"])
    trainer_args = _expand_trainer_args(
        job.get("trainer_args", []),
        job=job,
        artifact_abs=artifact_abs,
    )
    trainer = validate_job_trainer(job, trainer_args)
    run_label = build_run_label(str(job["candidate_id"]), str(job["object_id"]), str(job["physics_mode"]))
    cmd = [
        sys.executable,
        str(repo_root / "generate_and_train.py"),
        "--base",
        str(base_xml_abs),
        "--task",
        task_arg,
        "--trainer",
        trainer,
        "--Ntotal",
        str(job["Ntotal"]),
        "--Rppx",
        str(job["Rppx"]),
        "--Rpt",
        str(job["Rpt"]),
        "--out-root",
        str(artifact_abs),
        "--artifact-root",
        str(artifact_abs),
        "--object-id",
        str(job["object_id"]),
        "--candidate-id",
        str(job["candidate_id"]),
        "--object-size",
        str(job["size"]),
        "--physics-mode",
        str(job["physics_mode"]),
        "--run-label",
        run_label,
    ]
    if bool(job.get("force")):
        cmd.append("--force")
    if str(job["physics_mode"]) == "deformable":
        cmd.append("--deformable")

    cmd.extend(
        [
            "--",
            "--seed",
            str(job["seed"]),
            "--eval-episodes",
            str(job["eval_episodes"]),
            "--metrics-json",
            str(artifact_abs / "metrics.json"),
            "--task-name",
            str(job["object_id"]),
            "--object-id",
            str(job["object_id"]),
            "--candidate-id",
            str(job["candidate_id"]),
            "--physics-mode",
            str(job["physics_mode"]),
        ]
    )
    resolved_num_envs = host_cfg.resolved_num_envs_per_job()
    if (
        trainer == "cpu"
        and resolved_num_envs is not None
        and not has_opt("--num-envs", trainer_args)
    ):
        trainer_args.extend(["--num-envs", str(resolved_num_envs)])
    cmd.extend(trainer_args)
    return cmd


def launch_job(job: Dict[str, object], gpu_id: int, repo_root: Path, host_cfg: HostConfig) -> RunningJob:
    command = build_job_command(job, repo_root, host_cfg)
    artifact_abs = repo_root / str(job["artifact_relpath"])
    artifact_abs.mkdir(parents=True, exist_ok=True)
    stdout_abs = repo_root / str(job["stdout_relpath"])
    stderr_abs = repo_root / str(job["stderr_relpath"])
    stdout_abs.parent.mkdir(parents=True, exist_ok=True)
    stderr_abs.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("WANDB_CONSOLE", "off")
    # Prevent NumPy/BLAS/OpenMP from oversubscribing CPU cores on multi-job hosts.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    resolved_num_envs = host_cfg.resolved_num_envs_per_job()
    resolved_cpu_threads = host_cfg.resolved_cpu_threads_per_job()
    print(
        f"[worker] launching job_id={job['job_id']} host={host_cfg.host} gpu={gpu_id} "
        f"trainer={job.get('trainer', 'cpu')} "
        f"cpu_num_envs_hint={resolved_num_envs if resolved_num_envs is not None else 'unset'} "
        f"cpu_threads_per_job={resolved_cpu_threads if resolved_cpu_threads is not None else 'unknown'}"
    )

    stdout_handle = open(stdout_abs, "w", encoding="utf-8")
    stderr_handle = open(stderr_abs, "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(repo_root),
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    return RunningJob(
        job=job,
        process=process,
        artifact_abs=artifact_abs,
        stdout_abs=stdout_abs,
        stderr_abs=stderr_abs,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
    )


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    cluster_cfg = load_cluster_config(args.cluster_config, repo_dirname=repo_root.name)
    host_name = detect_host_name(cluster_cfg, args.host_name)
    host_cfg = next((host for host in cluster_cfg.worker_hosts() if host.host == host_name), None)
    if host_cfg is None:
        raise ValueError(f"Host {host_name!r} not found in {args.cluster_config}")

    coordinator = CoordinatorClient(cluster_cfg.coordinator, args.study_name)
    worker_id = f"{host_name}-{os.getpid()}"
    tracker = FreeGpuTracker(args.consecutive_free_polls)
    running_jobs: Dict[int, RunningJob] = {}

    while True:
        snapshot = collect_gpu_snapshot()
        details = {
            "snapshot": snapshot,
            "running_job_ids": {str(gpu_id): int(job.job["job_id"]) for gpu_id, job in running_jobs.items()},
        }
        coordinator.heartbeat(host_name, worker_id, os.getpid(), details)

        finished_gpu_ids = []
        for gpu_id, running in list(running_jobs.items()):
            returncode = running.process.poll()
            if returncode is None:
                continue

            try:
                running.stdout_handle.close()
                running.stderr_handle.close()
                sync_artifacts_to_coordinator(running.job, running.artifact_abs, cluster_cfg.coordinator)
                score = load_score_from_artifacts(
                    str(running.artifact_abs / "metrics.json"),
                    stdout_path=str(running.stdout_abs),
                    stderr_path=str(running.stderr_abs),
                )
                if returncode == 0 and score is not None:
                    coordinator.finish_job(int(running.job["job_id"]), float(score), running.job)
                else:
                    reason = f"Command exited with rc={returncode}; score={score!r}"
                    coordinator.fail_job(int(running.job["job_id"]), reason, args.max_attempts)
            except Exception as exc:
                coordinator.fail_job(int(running.job["job_id"]), f"Worker post-processing failed: {exc}", args.max_attempts)
            finished_gpu_ids.append(gpu_id)

        for gpu_id in finished_gpu_ids:
            running_jobs.pop(gpu_id, None)

        currently_free = free_gpu_ids(
            snapshot,
            memory_threshold_mib=args.free_memory_mib,
            utilization_threshold=args.free_utilization,
            allow_foreign_processes=args.allow_foreign_processes,
        )
        eligible_ids = tracker.update(currently_free)
        for busy_gpu in running_jobs:
            eligible_ids.discard(busy_gpu)

        for gpu_id in sorted(eligible_ids):
            if gpu_id in running_jobs:
                continue
            job = coordinator.claim_job(host_name, worker_id, gpu_id, args.lease_seconds, args.max_attempts)
            if not job:
                break
            running_jobs[gpu_id] = launch_job(job, gpu_id, repo_root, host_cfg)

        time.sleep(max(5, int(args.poll_interval_seconds)))


if __name__ == "__main__":
    main()
