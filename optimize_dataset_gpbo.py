import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from botorch.acquisition import qLogNoisyExpectedImprovement as qLogNEI
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from study_common import (
    Candidate,
    build_candidate_grid,
    build_run_artifact_relpath,
    load_cluster_config,
    load_study_manifest,
    map_alpha_beta_to_ratios,
    resolve_repo_root,
    resolve_study_root,
    sanitize_identifier,
    sobol_initial_candidates,
)
from study_queue import StudyQueue


VALID_PHYSICS_MODES = ("deformable", "rigid")
VALID_TRAINERS = ("cpu", "gpu")


def parse_physics_modes(raw_value: str, flag_name: str) -> List[str]:
    modes: List[str] = []
    seen = set()
    for piece in str(raw_value).split(","):
        mode = piece.strip().lower()
        if not mode:
            continue
        if mode not in VALID_PHYSICS_MODES:
            raise ValueError(f"{flag_name} must contain only {VALID_PHYSICS_MODES}; got {mode!r}.")
        if mode in seen:
            continue
        seen.add(mode)
        modes.append(mode)
    if not modes:
        raise ValueError(f"{flag_name} must include at least one physics mode.")
    return modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed discrete GPBO coordinator for dataset-scale sensor search.")
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--objects-root", required=True, help="Path to the study object folder containing manifest.csv.")
    parser.add_argument("--cluster-config", default="cluster_hosts.yaml")
    parser.add_argument("--db", default=None, help="Optional explicit path to the sqlite study DB.")
    parser.add_argument(
        "--study-root",
        default=None,
        help="Optional root for the study DB, reports, generated XMLs, metrics, stdout/stderr, models, videos, runs, and wandb files.",
    )
    parser.add_argument("--base", default="assets/hand_base.xml", help="Base hand XML used by generate_and_train.py.")
    parser.add_argument("--init-candidates", type=int, default=4)
    parser.add_argument("--bo-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument(
        "--trainer",
        choices=VALID_TRAINERS,
        default="cpu",
        help=(
            "Training implementation stored with queued jobs. GPU supports rigid-only "
            "native tasks and custom meshes converted to native rigid mesh geoms."
        ),
    )
    parser.add_argument("--heartbeat-timeout-seconds", type=int, default=900)
    parser.add_argument("--loop-sleep-seconds", type=int, default=60)
    parser.add_argument("--expected-base-objects", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="Forward --force into generate_and_train.py jobs.")
    parser.add_argument(
        "--sobol-physics-modes",
        default="deformable,rigid",
        help="Comma-separated physics modes to enqueue for the initial Sobol candidates.",
    )
    parser.add_argument(
        "--bo-physics-modes",
        default="deformable,rigid",
        help="Comma-separated physics modes to enqueue for BO-selected candidates.",
    )
    parser.add_argument(
        "--backfill-missing-physics-modes",
        default="",
        help="Optional comma-separated physics modes to add to already-selected completed/failed candidates.",
    )
    parser.add_argument("--once", action="store_true", help="Run one coordinator iteration and exit.")
    parser.add_argument("--report-only", action="store_true", help="Export reports from current DB state and exit.")
    parser.add_argument(
        "--preview-initial-only",
        action="store_true",
        help="Initialize the study, print the initial Sobol candidates, export reports, and exit without enqueuing jobs.",
    )
    parser.add_argument("--trainer-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def row_to_candidate(row) -> Candidate:
    return Candidate(candidate_id=row["candidate_id"], N=int(row["N"]), alpha=float(row["alpha"]), beta=float(row["beta"]))


def _has_trainer_opt(option: str, trainer_args: Sequence[str]) -> bool:
    return option in trainer_args or any(
        str(argument).startswith(option + "=") for argument in trainer_args
    )


def validate_study_trainer(
    trainer: str,
    *,
    study_objects,
    sobol_physics_modes: Sequence[str],
    bo_physics_modes: Sequence[str],
    trainer_args: Sequence[str],
) -> None:
    """Fail before enqueueing a scientifically unsupported GPU study."""
    if _has_trainer_opt("--trainer", trainer_args):
        raise ValueError(
            "Select the study trainer with top-level --trainer, not inside "
            "--trainer-args."
        )
    if trainer not in VALID_TRAINERS:
        raise ValueError(f"trainer must be one of {VALID_TRAINERS}; got {trainer!r}.")
    if trainer == "cpu":
        return

    all_modes = set(sobol_physics_modes) | set(bo_physics_modes)
    if all_modes != {"rigid"}:
        raise ValueError(
            "GPU studies are rigid-geom-only; set both --sobol-physics-modes and "
            "--bo-physics-modes to rigid."
        )

    auto_worlds = _has_trainer_opt("--auto-num-envs", trainer_args)
    if auto_worlds:
        if not _has_trainer_opt("--auto-num-envs-report", trainer_args):
            raise ValueError("GPU --auto-num-envs requires --auto-num-envs-report.")
    elif not all(
        _has_trainer_opt(option, trainer_args)
        for option in ("--num-envs", "--contacts-per-world", "--constraints-per-world")
    ):
        raise ValueError(
            "GPU studies require either --auto-num-envs with its report or explicit "
            "--num-envs, --contacts-per-world, and --constraints-per-world. CPU host "
            "num_envs_per_job values are not GPU capacity recommendations."
        )
    if not (
        _has_trainer_opt("--auto-replay-capacity", trainer_args)
        or _has_trainer_opt("--buffer-size", trainer_args)
    ):
        raise ValueError(
            "GPU studies require --auto-replay-capacity or an explicit --buffer-size."
        )


def initialize_study(
    queue: StudyQueue,
    args: argparse.Namespace,
    cluster_cfg,
    study_objects,
    study_root: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    queue.register_hosts([host.to_dict() for host in cluster_cfg.all_hosts()])
    queue.register_candidates(build_candidate_grid())

    objects_root = Path(args.objects_root).expanduser().resolve()
    base_xml = Path(args.base).expanduser().resolve()
    if not base_xml.is_file():
        raise FileNotFoundError(base_xml)

    trainer_args = args.trainer_args[1:] if (args.trainer_args and args.trainer_args[0] == "--") else list(args.trainer_args)
    spec = queue.get_metadata("study_spec")
    if spec is None:
        sobol_physics_modes = parse_physics_modes(args.sobol_physics_modes, "--sobol-physics-modes")
        bo_physics_modes = parse_physics_modes(args.bo_physics_modes, "--bo-physics-modes")
        initial_candidates = sobol_initial_candidates(args.init_candidates)
        spec = {
            "study_name": args.study_name,
            "objects_root_relpath": os.path.relpath(objects_root, repo_root),
            "base_xml_relpath": os.path.relpath(base_xml, repo_root),
            "artifact_root_relpath": str(study_root),
            "repo_root": str(repo_root),
            "seed": int(args.seed),
            "eval_episodes": int(args.eval_episodes),
            "trainer": args.trainer,
            "force": bool(args.force),
            "trainer_args": trainer_args,
            "init_candidates": int(args.init_candidates),
            "bo_candidates": int(args.bo_candidates),
            "budget_total": int(args.init_candidates + args.bo_candidates),
            "expected_base_objects": int(args.expected_base_objects),
            "sobol_physics_modes": sobol_physics_modes,
            "bo_physics_modes": bo_physics_modes,
            "initial_candidate_ids": [candidate.candidate_id for candidate in initial_candidates],
        }
        queue.set_metadata("study_spec", spec)
    else:
        updated = False
        if "trainer" not in spec:
            spec["trainer"] = "cpu"
            updated = True
        if "sobol_physics_modes" not in spec:
            spec["sobol_physics_modes"] = parse_physics_modes(args.sobol_physics_modes, "--sobol-physics-modes")
            updated = True
        if "bo_physics_modes" not in spec:
            spec["bo_physics_modes"] = parse_physics_modes(args.bo_physics_modes, "--bo-physics-modes")
            updated = True
        if updated:
            queue.set_metadata("study_spec", spec)
    validate_study_trainer(
        str(spec["trainer"]),
        study_objects=study_objects,
        sobol_physics_modes=physics_modes_for_source(spec, "sobol"),
        bo_physics_modes=physics_modes_for_source(spec, "bo"),
        trainer_args=spec["trainer_args"],
    )
    return spec


def build_jobs_for_candidate(
    candidate: Candidate,
    study_name: str,
    study_objects,
    objects_root_relpath: str,
    base_xml_relpath: str,
    trainer_args: Sequence[str],
    trainer: str,
    seed: int,
    eval_episodes: int,
    force: bool,
    physics_modes: Sequence[str],
    study_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    _, _, _, _, Rppx, Rpt = map_alpha_beta_to_ratios(candidate.N, candidate.alpha, candidate.beta)
    jobs: List[Dict[str, Any]] = []
    for obj in study_objects:
        for physics_mode in physics_modes:
            artifact_relpath = build_run_artifact_relpath(
                study_name,
                candidate.candidate_id,
                obj.object_id,
                physics_mode,
                study_root=study_root,
            )
            payload = {
                "study_name": study_name,
                "candidate_id": candidate.candidate_id,
                "candidate": candidate.to_dict(),
                "object_id": obj.object_id,
                "base_object": obj.base_object,
                "aspect_ratio": obj.aspect_ratio,
                "size": obj.size,
                "physics_mode": physics_mode,
                "trainer": trainer,
                "seed": int(seed),
                "base_xml_relpath": base_xml_relpath,
                "task_arg": (
                    obj.native_task
                    if obj.native_task is not None
                    else os.path.join(objects_root_relpath, obj.msh_file)
                ),
                "task_kind": "native" if obj.native_task is not None else "mesh",
                "artifact_relpath": artifact_relpath,
                "metrics_relpath": os.path.join(artifact_relpath, "metrics.json"),
                "stdout_relpath": os.path.join(artifact_relpath, "stdout.txt"),
                "stderr_relpath": os.path.join(artifact_relpath, "stderr.txt"),
                "Ntotal": int(candidate.N),
                "Rppx": float(Rppx),
                "Rpt": float(Rpt),
                "force": bool(force),
                "trainer_args": list(trainer_args),
                "eval_episodes": int(eval_episodes),
            }
            jobs.append(
                {
                    "object_id": obj.object_id,
                    "physics_mode": physics_mode,
                    "base_object": obj.base_object,
                    "aspect_ratio": obj.aspect_ratio,
                    "size": obj.size,
                    "seed": int(seed),
                    "priority": 0 if physics_mode == "deformable" else 1,
                    "artifact_relpath": artifact_relpath,
                    "metrics_relpath": payload["metrics_relpath"],
                    "stdout_relpath": payload["stdout_relpath"],
                    "stderr_relpath": payload["stderr_relpath"],
                    "payload": payload,
                }
            )
    return jobs


def choose_bo_candidate(completed_rows, available_rows) -> Candidate:
    if len(completed_rows) < 2:
        return row_to_candidate(available_rows[0])

    train_X = torch.tensor(
        [[float(row["N"]), float(row["alpha"]), float(row["beta"])] for row in completed_rows],
        dtype=torch.double,
    )
    train_Y = torch.tensor([[float(row["score"])] for row in completed_rows], dtype=torch.double)
    candidate_X = torch.tensor(
        [[float(row["N"]), float(row["alpha"]), float(row["beta"])] for row in available_rows],
        dtype=torch.double,
    )

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=3),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acq = qLogNEI(model=model, X_baseline=train_X, prune_baseline=True)
    acquisition_values = acq(candidate_X.unsqueeze(1))
    best_idx = int(torch.argmax(acquisition_values).item())
    return row_to_candidate(available_rows[best_idx])


def candidate_has_completed_physics_modes(queue: StudyQueue, candidate_id: str, required_modes: Sequence[str]) -> bool:
    required = {str(mode).strip().lower() for mode in required_modes}
    if not required:
        return True
    row = queue.conn.execute(
        "SELECT status FROM candidates WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None or row["status"] != "completed":
        return False
    mode_rows = queue.conn.execute(
        """
        SELECT DISTINCT physics_mode
        FROM jobs
        WHERE candidate_id = ? AND status = 'succeeded'
        """,
        (candidate_id,),
    ).fetchall()
    completed_modes = {str(mode_row["physics_mode"]).strip().lower() for mode_row in mode_rows}
    return required.issubset(completed_modes)


def bo_ready_completed_rows(queue: StudyQueue, spec: Dict[str, Any]) -> List[Any]:
    required_modes = physics_modes_for_source(spec, "bo")
    rows = queue.completed_candidates()
    return [
        row for row in rows
        if candidate_has_completed_physics_modes(queue, str(row["candidate_id"]), required_modes)
    ]


def choose_next_candidate(queue: StudyQueue, spec: Dict[str, Any]) -> Tuple[Optional[Candidate], Optional[str]]:
    selected_rows = queue.list_candidates(statuses=["queued", "running", "completed", "failed"])
    selected_ids = {row["candidate_id"] for row in selected_rows}

    for candidate_id in spec["initial_candidate_ids"]:
        if candidate_id in selected_ids:
            continue
        row = queue.conn.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
        if row is not None:
            return row_to_candidate(row), "sobol"

    required_modes = physics_modes_for_source(spec, "bo")
    for candidate_id in spec["initial_candidate_ids"]:
        if not candidate_has_completed_physics_modes(queue, candidate_id, required_modes):
            return None, None

    completed_rows = bo_ready_completed_rows(queue, spec)
    available_rows = queue.available_candidates()
    if not available_rows:
        return None, None
    return choose_bo_candidate(completed_rows, available_rows), "bo"


def physics_modes_for_source(spec: Dict[str, Any], source: str) -> List[str]:
    key = "sobol_physics_modes" if source == "sobol" else "bo_physics_modes"
    raw_modes = spec.get(key, list(VALID_PHYSICS_MODES))
    if isinstance(raw_modes, str):
        return parse_physics_modes(raw_modes, key)
    modes: List[str] = []
    seen = set()
    for mode in raw_modes:
        normalized = str(mode).strip().lower()
        if normalized not in VALID_PHYSICS_MODES or normalized in seen:
            continue
        seen.add(normalized)
        modes.append(normalized)
    return modes or list(VALID_PHYSICS_MODES)


def backfill_missing_candidate_jobs(
    queue: StudyQueue,
    spec: Dict[str, Any],
    study_objects,
    physics_modes: Sequence[str],
) -> Dict[str, Any]:
    requested_modes = parse_physics_modes(",".join(physics_modes), "--backfill-missing-physics-modes")
    rows = queue.conn.execute(
        """
        SELECT * FROM candidates
        WHERE selection_order IS NOT NULL
        ORDER BY selection_order, candidate_id
        """
    ).fetchall()

    updated_candidates: List[Dict[str, Any]] = []
    skipped_active: List[str] = []
    for row in rows:
        if row["status"] in {"queued", "running"}:
            skipped_active.append(str(row["candidate_id"]))
            continue
        candidate = row_to_candidate(row)
        jobs = build_jobs_for_candidate(
            candidate=candidate,
            study_name=spec["study_name"],
            study_objects=study_objects,
            objects_root_relpath=spec["objects_root_relpath"],
            base_xml_relpath=spec["base_xml_relpath"],
            trainer_args=spec["trainer_args"],
            trainer=str(spec.get("trainer", "cpu")),
            seed=int(spec["seed"]),
            eval_episodes=int(spec["eval_episodes"]),
            force=bool(spec["force"]),
            physics_modes=requested_modes,
            study_root=Path(spec["artifact_root_relpath"]),
        )
        inserted = queue.enqueue_candidate_jobs(candidate, jobs, source=str(row["source"] or "backfill"))
        if inserted:
            updated_candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "jobs_added": inserted,
                }
            )

    return {
        "requested_physics_modes": requested_modes,
        "updated_candidates": updated_candidates,
        "skipped_active_candidates": skipped_active,
    }


def export_reports(queue: StudyQueue, study_root: Path) -> None:
    study_root.mkdir(parents=True, exist_ok=True)
    spec = queue.get_metadata("study_spec", {})
    candidate_rows = queue.list_candidates()
    candidate_history_path = study_root / "candidate_history.json"
    leaderboard_path = study_root / "leaderboard.csv"
    per_condition_path = study_root / "per_condition_scores.csv"
    study_spec_path = study_root / "study_spec.json"
    initial_candidates_path = study_root / "initial_candidates.json"

    history = []
    for row in candidate_rows:
        history.append(
            {
                "candidate_id": row["candidate_id"],
                "N": int(row["N"]),
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "status": row["status"],
                "source": row["source"],
                "selection_order": row["selection_order"],
                "score": row["score"],
                "rigid_mean": row["rigid_mean"],
                "deformable_mean": row["deformable_mean"],
            }
        )
    with candidate_history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    with study_spec_path.open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)

    initial_candidate_rows = []
    for candidate_id in spec.get("initial_candidate_ids", []):
        row = queue.conn.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
        if row is None:
            continue
        initial_candidate_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "N": int(row["N"]),
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "status": row["status"],
                "source": row["source"],
                "selection_order": row["selection_order"],
            }
        )
    with initial_candidates_path.open("w", encoding="utf-8") as handle:
        json.dump(initial_candidate_rows, handle, indent=2)

    completed = [row for row in candidate_rows if row["status"] == "completed"]
    completed.sort(key=lambda row: float(row["score"]), reverse=True)
    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_id", "N", "alpha", "beta", "score", "rigid_mean", "deformable_mean", "source"],
        )
        writer.writeheader()
        for row in completed:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "N": int(row["N"]),
                    "alpha": float(row["alpha"]),
                    "beta": float(row["beta"]),
                    "score": row["score"],
                    "rigid_mean": row["rigid_mean"],
                    "deformable_mean": row["deformable_mean"],
                    "source": row["source"],
                }
            )

    with per_condition_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "object_id",
                "trainer",
                "physics_mode",
                "base_object",
                "aspect_ratio",
                "size",
                "status",
                "score",
                "artifact_relpath",
                "metrics_relpath",
            ],
        )
        writer.writeheader()
        for row in queue.conn.execute(
            """
            SELECT candidate_id, object_id, physics_mode, base_object, aspect_ratio, size,
                   status, score, artifact_relpath, metrics_relpath, payload_json
            FROM jobs
            ORDER BY candidate_id, object_id, physics_mode
            """
        ).fetchall():
            rendered = dict(row)
            payload = json.loads(rendered.pop("payload_json"))
            rendered["trainer"] = payload.get("trainer", "cpu")
            writer.writerow(rendered)


def coordinator_iteration(
    queue: StudyQueue,
    spec: Dict[str, Any],
    study_objects,
    study_root: Path,
) -> Dict[str, Any]:
    requeued = queue.requeue_stale_jobs()
    active = queue.active_candidates()
    completed = queue.completed_candidates()
    bo_ready_completed = bo_ready_completed_rows(queue, spec)
    blocking_initial_candidates = [
        candidate_id
        for candidate_id in spec["initial_candidate_ids"]
        if not candidate_has_completed_physics_modes(queue, candidate_id, physics_modes_for_source(spec, "bo"))
    ]
    status = {
        "requeued_jobs": requeued,
        "active_candidates": [row["candidate_id"] for row in active],
        "completed_candidates": len(completed),
        "bo_ready_completed_candidates": len(bo_ready_completed),
        "blocking_initial_candidates": blocking_initial_candidates,
        "budget_total": int(spec["budget_total"]),
    }

    if not active and len(completed) < int(spec["budget_total"]):
        next_candidate, source = choose_next_candidate(queue, spec)
        if next_candidate is not None and source is not None:
            physics_modes = physics_modes_for_source(spec, source)
            jobs = build_jobs_for_candidate(
                candidate=next_candidate,
                study_name=spec["study_name"],
                study_objects=study_objects,
                objects_root_relpath=spec["objects_root_relpath"],
                base_xml_relpath=spec["base_xml_relpath"],
                trainer_args=spec["trainer_args"],
                trainer=str(spec.get("trainer", "cpu")),
                seed=int(spec["seed"]),
                eval_episodes=int(spec["eval_episodes"]),
                force=bool(spec["force"]),
                physics_modes=physics_modes,
                study_root=study_root,
            )
            inserted = queue.enqueue_candidate_jobs(next_candidate, jobs, source=source)
            status["enqueued_candidate"] = next_candidate.candidate_id
            status["enqueued_source"] = source
            status["enqueued_physics_modes"] = physics_modes
            status["enqueued_job_count"] = inserted
        else:
            status["enqueued_candidate"] = None

    export_reports(queue, study_root)
    status["summary"] = queue.summary()
    return status


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    study_root = Path(args.study_root).expanduser().resolve() if args.study_root else resolve_study_root(args.study_name, repo_root=repo_root)
    db_path = os.path.abspath(args.db or str(study_root / "study.db"))

    cluster_cfg = load_cluster_config(args.cluster_config, repo_dirname=repo_root.name)
    study_objects = load_study_manifest(args.objects_root, expected_base_objects=args.expected_base_objects)

    queue = StudyQueue(db_path)
    try:
        spec = initialize_study(queue, args, cluster_cfg, study_objects, study_root, repo_root)
        if args.backfill_missing_physics_modes.strip():
            backfill = backfill_missing_candidate_jobs(
                queue,
                spec,
                study_objects,
                parse_physics_modes(args.backfill_missing_physics_modes, "--backfill-missing-physics-modes"),
            )
            export_reports(queue, study_root)
            if args.once or args.report_only:
                print(json.dumps({"backfill": backfill, "summary": queue.summary()}, indent=2))
                return
        if args.preview_initial_only:
            export_reports(queue, study_root)
            preview = []
            for candidate_id in spec.get("initial_candidate_ids", []):
                row = queue.conn.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
                if row is None:
                    continue
                preview.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "N": int(row["N"]),
                        "alpha": float(row["alpha"]),
                        "beta": float(row["beta"]),
                        "status": row["status"],
                    }
                )
            print(
                json.dumps(
                    {
                        "study_name": spec["study_name"],
                        "study_root": str(study_root),
                        "db_path": db_path,
                        "initial_candidates": preview,
                    },
                    indent=2,
                )
            )
            return
        if args.report_only:
            export_reports(queue, study_root)
            print(json.dumps(queue.summary(), indent=2))
            return

        while True:
            status = coordinator_iteration(queue, spec, study_objects, study_root)
            print(json.dumps(status, indent=2))

            completed_count = len(queue.completed_candidates())
            if completed_count >= int(spec["budget_total"]):
                break
            if args.once:
                break
            time.sleep(max(1, int(args.loop_sleep_seconds)))
    finally:
        queue.close()


if __name__ == "__main__":
    main()
