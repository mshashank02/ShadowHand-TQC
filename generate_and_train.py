import argparse, os, re, subprocess, sys
from pipeline_generate import (
    CoACDParameters,
    DEFAULT_DEFORMABLE_PRESET,
    RIGID_COLLISION_REPRESENTATIONS,
    build_candidate_standalone,
    deformable_preset_names,
    deformable_preset_spawn_position,
    get_deformable_preset,
    resolve_task_template,
    parse_task_arg,
)
from registration import stable_env_id


def sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_") or "run"


def has_flag(flag: str, argv: list[str]) -> bool:
    return flag in argv


def has_opt(opt: str, argv: list[str]) -> bool:
    if opt in argv:
        return True
    return any(arg.startswith(opt + "=") for arg in argv)


def pop_opt_value(opt: str, argv: list[str]) -> str | None:
    """Remove one forwarded option and return its value."""
    for index, value in enumerate(argv):
        if value == opt:
            if index + 1 >= len(argv):
                raise SystemExit(f"ERROR: {opt} requires a value")
            result = argv[index + 1]
            del argv[index : index + 2]
            return result
        if value.startswith(opt + "="):
            del argv[index]
            return value.split("=", 1)[1]
    return None


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


def scale_triplet(base_value: float, multiplier: float) -> str:
    scaled = base_value * multiplier
    return f"{scaled:.6f} {scaled:.6f} {scaled:.6f}"


def scale_scalar(base_value: float, multiplier: float) -> str:
    return f"{base_value * multiplier:.6f}"


def scale_mass(base_mass: float, multiplier: float) -> str:
    return f"{base_mass * (multiplier ** 3):.6f}"


def scale_diaginertia(base_inertia: tuple[float, float, float], multiplier: float) -> str:
    scaled = [value * (multiplier ** 5) for value in base_inertia]
    return " ".join(f"{value:.8f}" for value in scaled)


def infer_object_size_label(custom_msh: str | None) -> str | None:
    if not custom_msh:
        return None
    mesh_name = os.path.basename(custom_msh).lower()
    match = re.search(r"(?:^|[_-])size[-_](small|medium|large)(?:[_-]|$)", mesh_name)
    if match:
        return match.group(1)
    return None


def infer_object_spawn_position(size_label: str, deformable: bool) -> str:
    physics_mode = "deformable" if deformable else "rigid"
    z = SIZE_SPAWN_HEIGHTS[physics_mode][size_label]
    return f"1 0.87 {z:.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument(
        '--task',
        default='block',
        help="Built-in task name (block/egg/pen) OR absolute path to a custom .msh file.",
    )
    p.add_argument("--Ntotal", type=int, required=True)
    p.add_argument('--Rppx', type=float, required=True)
    p.add_argument('--Rpt',  type=float, required=True)
    p.add_argument("--Ap",   type=float, default= 6557, help="Area weight: Palm")
    p.add_argument("--Apx",  type=float, default=26885, help="Area weight: Phalanx")
    p.add_argument("--At",   type=float, default=7193, help="Area weight: Tips")
    p.add_argument("--Ap1",  type=float, default=5557, help="Palm sub-area 1 (palm)")
    p.add_argument("--Ap2",  type=float, default=1000, help="Palm sub-area 2 (lfmetacarpal)")
    p.add_argument('--out-root', default='generated')
    p.add_argument('--artifact-root', default=None,
                   help="Optional run-specific root for generated XMLs and training artifacts.")
    p.add_argument('--object-id', default=None,
                   help="Stable object identifier used for artifact naming and metrics metadata.")
    p.add_argument('--run-label', default=None,
                   help="Stable run label used for env_id and logging.")
    p.add_argument('--candidate-id', default=None,
                   help="Optional candidate identifier forwarded into metrics metadata.")
    p.add_argument('--physics-mode', choices=["rigid", "deformable"], default=None,
                   help="Optional physics mode metadata. Defaults to deformable when --deformable is set, else rigid.")
    p.add_argument('--object-size', choices=sorted(SIZE_SCALE_MULTIPLIERS), default=None,
                   help="Optional object size label used to scale custom .msh objects in generated XMLs.")
    p.add_argument('--deformable', action='store_true',
                   help="Generate a deformable custom object when --task points to a .msh file.")
    p.add_argument('--deformable-preset', choices=deformable_preset_names(), default=DEFAULT_DEFORMABLE_PRESET,
                   help="Named rubber-like deformable material/contact/solver preset.")
    p.add_argument('--rigid-mesh-cache', default=None,
                   help="Shared source-hash cache for converted rigid OBJ surfaces.")
    p.add_argument(
        '--rigid-collision-representation',
        choices=RIGID_COLLISION_REPRESENTATIONS,
        default='single_mesh',
        help="Explicit validation selector; production remains single_mesh until accepted.",
    )
    p.add_argument('--coacd-threshold-mm', type=float, default=None,
                   help="Required physical CoACD threshold for convex_decomposition.")
    p.add_argument('--coacd-max-convex-hull', type=int, default=-1,
                   help="Optional CoACD piece cap; -1 is fidelity-driven.")
    p.add_argument('--skip-deformable-validation', action='store_true',
                   help="Skip XML/Gym/action-rollout validation before training deformable objects.")
    p.add_argument('--preflight-passive-steps', type=int, default=2000,
                   help="MuJoCo passive physics steps for deformable preflight validation.")
    p.add_argument('--preflight-env-steps', type=int, default=80,
                   help="Gym environment steps for each zero/random preflight validation rollout.")
    p.add_argument('--preflight-train-steps', type=int, default=64,
                   help="Short TQC+HER smoke-training steps to run before the full deformable training job.")
    p.add_argument('--force', action='store_true')
    p.add_argument(
        '--trainer',
        choices=["cpu", "gpu"],
        default="cpu",
        help="Select the unchanged SB3 CPU reference or direct MuJoCo Warp GPU trainer.",
    )
    # everything after “--” is passed directly to ShadowHand_TQC.py
    args, train_args = p.parse_known_args()
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    forwarded_trainer = pop_opt_value("--trainer", train_args)
    if forwarded_trainer is not None:
        if forwarded_trainer not in ("cpu", "gpu"):
            raise SystemExit("ERROR: --trainer must be cpu or gpu")
        if args.trainer != "cpu" and forwarded_trainer != args.trainer:
            raise SystemExit("ERROR: conflicting --trainer selections")
        args.trainer = forwarded_trainer

    task_cfg = parse_task_arg(args.task)
    decomposition_parameters = None
    if args.rigid_collision_representation == "convex_decomposition":
        if args.coacd_threshold_mm is None or args.coacd_threshold_mm <= 0.0:
            p.error("convex_decomposition requires a positive --coacd-threshold-mm")
        decomposition_parameters = CoACDParameters(
            threshold_m=args.coacd_threshold_mm / 1000.0,
            max_convex_hull=args.coacd_max_convex_hull,
            resolution=1000,
            mcts_iterations=100,
        )
    elif args.coacd_threshold_mm is not None or args.coacd_max_convex_hull != -1:
        p.error("CoACD parameters require --rigid-collision-representation convex_decomposition")
    physics_mode = args.physics_mode or ("deformable" if args.deformable else "rigid")
    if args.deformable and physics_mode != "deformable":
        raise SystemExit("ERROR: --deformable conflicts with --physics-mode rigid")
    deformable_object = physics_mode == "deformable"
    if deformable_object and task_cfg["custom_msh"] is None:
        if task_cfg["template_task"] != "egg":
            raise SystemExit("ERROR: built-in --deformable currently supports --task egg only.")
        task_cfg["task_label"] = "egg_deformable"
    if args.trainer == "gpu" and physics_mode != "rigid":
        raise SystemExit(
            "ERROR: the direct MuJoCo Warp trainer supports rigid mesh/geom models only; "
            "flex/deformable support was intentionally excluded."
        )
    out_root = os.path.abspath(args.artifact_root or args.out_root)
    rigid_mesh_cache = os.path.abspath(
        args.rigid_mesh_cache
        or os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated", "rigid_mesh_cache")
    )
    inferred_size_label = infer_object_size_label(task_cfg["custom_msh"])
    size_label = args.object_size or inferred_size_label or "medium"
    size_multiplier = SIZE_SCALE_MULTIPLIERS[size_label]
    deformable_preset = get_deformable_preset(args.deformable_preset) if deformable_object else None
    flex_scale = scale_triplet(0.025, size_multiplier)
    flex_radius_base = float(deformable_preset["flex_radius"]) if deformable_preset else 0.001
    mass_base = float(deformable_preset["mass"]) if deformable_preset else BASE_RIGID_MASS
    flex_radius = scale_scalar(flex_radius_base, size_multiplier)
    object_pos = (
        deformable_preset_spawn_position(args.deformable_preset, size_label)
        if deformable_object
        else infer_object_spawn_position(size_label, False)
    )
    object_mass = scale_mass(mass_base, size_multiplier)
    object_inertia = scale_diaginertia(
        BASE_DEFORMABLE_DIAGINERTIA if deformable_object else BASE_RIGID_DIAGINERTIA,
        size_multiplier,
    )

    if args.object_id and task_cfg["custom_msh"] is not None:
        task_cfg["task_label"] = f"custom_{sanitize_label(args.object_id)}"

    # -----------------------------
    # Inject per-task env defaults
    # -----------------------------
    # Desired defaults by task
    if task_cfg["template_task"] == "pen":
        desired_target_position = "ignore"   # no position goal
        desired_ignore_z = True              # XY-only rotation
    else:  # block, egg
        desired_target_position = "random"   # keep position goal
        desired_ignore_z = False             # full xyz

    # Only append if not already set by the caller
    if not has_opt("--target-position", train_args):
        train_args += ["--target-position", desired_target_position]

    # --ignore-z-rot is a boolean flag; only add it when desired and not present
    if desired_ignore_z and not has_flag("--ignore-z-rot", train_args):
        train_args += ["--ignore-z-rot"]
        
    tmpl = resolve_task_template(task_cfg["template_task"], None, None)
    custom_msh_name = None
    if task_cfg["custom_msh"] is not None and args.object_id:
        custom_msh_name = f"{sanitize_label(args.object_id)}_{os.path.basename(task_cfg['custom_msh'])}"
    paths = build_candidate_standalone(
        task=task_cfg["task_label"],
        Ntotal=args.Ntotal, Rppx=args.Rppx, Rpt=args.Rpt,
        Ap=args.Ap, Apx=args.Apx, At=args.At, Ap1=args.Ap1, Ap2=args.Ap2,
        base_xml=args.base, template_xml=tmpl,
        out_root=out_root, force=args.force,
        custom_msh=task_cfg["custom_msh"],
        custom_msh_name=custom_msh_name,
        deformable_object=deformable_object,
        flex_scale=flex_scale,
        flex_radius=flex_radius,
        object_pos=object_pos,
        object_mass=object_mass,
        object_inertia=object_inertia,
        deformable_preset=args.deformable_preset,
        rigid_mesh_cache_dir=rigid_mesh_cache,
        rigid_collision_representation=args.rigid_collision_representation,
        rigid_decomposition_parameters=decomposition_parameters,
    )

    xml_abs = os.path.abspath(paths["env"])  # <-- make it absolute
    env_id = sanitize_label(args.run_label) if args.run_label else stable_env_id(xml_abs)

    if args.artifact_root and not has_opt("--artifact-root", train_args):
        train_args += ["--artifact-root", os.path.abspath(args.artifact_root)]
    if args.object_id and not has_opt("--object-id", train_args):
        train_args += ["--object-id", sanitize_label(args.object_id)]
    if args.candidate_id and not has_opt("--candidate-id", train_args):
        train_args += ["--candidate-id", sanitize_label(args.candidate_id)]
    if physics_mode and not has_opt("--physics-mode", train_args):
        train_args += ["--physics-mode", physics_mode]
    if args.run_label and not has_opt("--wandb-name", train_args):
        train_args += ["--wandb-name", sanitize_label(args.run_label)]
    if deformable_object and deformable_preset:
        preset_train_defaults = {
            "--action-scale": deformable_preset.get("action_scale"),
            "--action-clip": deformable_preset.get("action_clip"),
            "--action-smoothing": deformable_preset.get("action_smoothing"),
            "--reset-settle-steps": deformable_preset.get("reset_settle_steps"),
        }
        for opt, value in preset_train_defaults.items():
            if value is not None and not has_opt(opt, train_args):
                train_args += [opt, str(value)]

    if deformable_object and not args.skip_deformable_validation:
        validation_cmd = [
            sys.executable,
            "validate_deformable_rollout.py",
            "--xml-path",
            xml_abs,
            "--target-position",
            desired_target_position,
            "--target-rotation",
            "xyz",
            "--passive-steps",
            str(args.preflight_passive_steps),
            "--env-steps",
            str(args.preflight_env_steps),
            "--training-steps",
            str(args.preflight_train_steps),
            "--action-scale",
            str(deformable_preset.get("action_scale", 1.0)),
            "--action-clip",
            str(deformable_preset.get("action_clip", 1.0)),
            "--action-smoothing",
            str(deformable_preset.get("action_smoothing", 0.0)),
            "--reset-settle-steps",
            str(deformable_preset.get("reset_settle_steps", 0)),
        ]
        if desired_ignore_z:
            validation_cmd.append("--ignore-z-rot")
        subprocess.run(validation_cmd, check=True)

    #env_id = stable_env_id(paths["env"])           # optional: for naming only
    trainer_script = "train_gpu.py" if args.trainer == "gpu" else "ShadowHand_TQC.py"
    cmd = [
        sys.executable, trainer_script,
        "--env-id", env_id,                        # just for logs/dirs
        "--xml-path", xml_abs,                # REQUIRED for direct construction
        *train_args,
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
