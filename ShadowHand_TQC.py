import argparse
import glob
import json
import re
import shutil
import subprocess
from copy import deepcopy
from functools import partial
import gymnasium as gym
import gymnasium_robotics
import os
import numpy as np
import torch
from sb3_contrib import TQC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecNormalize, DummyVecEnv
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from gymnasium.wrappers import TimeLimit
import wandb
from wandb.integration.sb3 import WandbCallback
import warnings
from custom_envs.hand_block_forward_face_env import MujocoHandBlockForwardFaceTouchEnv
from custom_envs.hand_block_yaw import MujocoHandBlockYawTouchEnv
from custom_envs.dynamic_touch_env import DynamicXMLTouchEnv
from custom_wrappers.remove_object_state import RemoveObjectStateWrapper



# ignore warning. it does not affect the training
warnings.filterwarnings("ignore", message=".*method is not within the observation space*")

ENV_HYPERPARAMS = {
    "n_timesteps": 16e6,
    "policy": "MultiInputPolicy",
    "buffer_size": 1000000,
    "ent_coef": "auto",
    "batch_size": 2048,
    "gamma": 0.95,
    "learning_rate": 1e-3,
    "learning_starts": 8000,
    "tau": 0.05,
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
    "arch": [512, 512, 512],
    "n_critics": 2,
}

def parse_args():
    parser = argparse.ArgumentParser()
    # Experiment config
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--verbose", type=int, default=2,
            help="the verbosity of the logs")
    parser.add_argument("--num-envs", type=int, default=6,
        help="number of parallel environments")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Torch device for TQC. 'auto' uses CUDA when available, otherwise CPU.")
    parser.add_argument("--eval-freq", type=int, default=20000,
        help="frequency of evaluation (in timesteps)")
    parser.add_argument("--eval-episodes", type=int, default=50,
        help="number of episodes for evaluation")
    parser.add_argument("--eval-video-steps", type=int, default=200,
        help="Maximum env steps to record for each evaluation video.")
    parser.add_argument("--save-freq", type=int, default=200000,
        help="frequency of saving model and stats (in timesteps)")
    parser.add_argument("--gradient-save-freq", type=int, default=100000,
        help="frequency of saving gradients (in timesteps)")
    parser.add_argument("--model-save-freq", type=int, default=100000,
        help="frequency of saving model (in timesteps)")
    
    # Environment
    parser.add_argument("--env-id", type=str, default="HandManipulateBlockRotateXYZ-v1",
        help="env id")
    parser.add_argument("--xml-path", type=str, required=True,
                        help="Path to generated MuJoCo XML")
    parser.add_argument("--artifact-root", type=str, default=".",
        help="Root directory for models, videos, tensorboard logs, and wandb artifacts.")
    # Env behavior knobs (forwarded to MujocoManipulateTouchSensorsEnv)
    parser.add_argument("--target-position", type=str, default="random",
        choices=["random", "ignore"], help="goal position usage")
    parser.add_argument("--ignore-z-rot", action="store_true",
        help="if set, ignore Z axis in rotation error (XY-only rotate)")
    parser.add_argument("--target-rotation", type=str, default="xyz",
        choices=["xyz"], help="rotation axes spec (kept for completeness)")
    parser.add_argument("--render-human", action="store_true",
        help="Render training live in a MuJoCo viewer window (recommended with --num-envs 1).")
    parser.add_argument("--max-episode-steps", type=int, default=100,
        help="Maximum policy steps per episode before TimeLimit reset.")
    parser.add_argument("--debug-goals-live", action="store_true",
        help="Print achieved_goal and desired_goal live from the training env.")
    parser.add_argument("--debug-goals-every", type=int, default=1,
        help="Print goal debug every N env steps when --debug-goals-live is set.")
    parser.add_argument("--action-scale", type=float, default=1.0,
        help="Optional debug scale applied to policy actions before actuator mapping.")
    parser.add_argument("--action-clip", type=float, default=None,
        help="Optional symmetric clip applied to policy actions after scaling.")
    parser.add_argument("--action-smoothing", type=float, default=0.0,
        help="Exponential action smoothing factor in [0, 0.98]; 0 disables smoothing.")
    parser.add_argument("--reset-settle-steps", type=int, default=0,
        help="Zero-action env steps to settle the object immediately after reset.")
    
    # model hyperparameters
    parser.add_argument("--n-timesteps", type=float, default=ENV_HYPERPARAMS["n_timesteps"],
        help="total number of timesteps")
    parser.add_argument("--buffer-size", type=int, default=ENV_HYPERPARAMS["buffer_size"],
        help="replay buffer size")
    parser.add_argument("--batch-size", type=int, default=ENV_HYPERPARAMS["batch_size"],
        help="batch size")
    parser.add_argument("--gamma", type=float, default=ENV_HYPERPARAMS["gamma"],
        help="discount factor")
    parser.add_argument("--learning-rate", type=float, default=ENV_HYPERPARAMS["learning_rate"],
        help="learning rate")
    parser.add_argument("--learning-starts", type=int, default=ENV_HYPERPARAMS["learning_starts"],
        help="steps before learning starts")
    parser.add_argument("--tau", type=float, default=ENV_HYPERPARAMS["tau"],
        help="tau")
    parser.add_argument("--ent-coef", type=str, default=ENV_HYPERPARAMS["ent_coef"],
        help="entropy coefficient")
    
    # HER hyperparameters
    parser.add_argument("--n-sampled-goal", type=int, default=ENV_HYPERPARAMS["n_sampled_goal"],
        help="number of sampled goals for HER")
    parser.add_argument("--goal-selection-strategy", type=str, default=ENV_HYPERPARAMS["goal_selection_strategy"],
        help="goal selection strategy for HER")
    parser.add_argument("--arch", type=int, nargs="+", default=ENV_HYPERPARAMS["arch"],
        help="network architecture (list of layer sizes)")
    parser.add_argument("--n-critics", type=int, default=ENV_HYPERPARAMS["n_critics"],
        help="number of critic networks")
    

    # WandB config
    parser.add_argument("--wandb-project", type=str, default="in-hand manipulation",
        help="wandb project name")
    parser.add_argument("--wandb-name", type=str, default=None,
        help="wandb run name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="Optional wandb entity/team name.")
    parser.add_argument("--wandb-id", type=str, default=None,
        help="Optional wandb run id to resume.")
    parser.add_argument("--wandb-resume", type=str, default=None,
        choices=["allow", "must", "never", "auto"],
        help="W&B resume behavior used with --wandb-id.")
    parser.add_argument("--wandb-resume-max-step-gap", type=int, default=250000,
        help="Maximum allowed difference between the remote W&B training step and a resume checkpoint.")
    parser.add_argument("--skip-wandb-resume-check", action="store_true",
        help="Skip the strict remote W&B/checkpoint continuity check (not recommended).")
    parser.add_argument("--skip-wandb-local-sync", action="store_true",
        help="Do not sync prior local sessions for the W&B run ID before resume validation.")
    parser.add_argument("--no-wandb-checkpoint-fallback", action="store_true",
        help="Fail instead of selecting an older checkpoint when W&B history is behind the requested model.")
    parser.add_argument("--disable-eval-video", action="store_true",
        help="Disable expensive evaluation video generation during training.")
    
    #GP-BO configs
    parser.add_argument("--metrics-json", type=str, default=None)
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--object-id", type=str, default=None)
    parser.add_argument("--candidate-id", type=str, default=None)
    parser.add_argument("--physics-mode", type=str, default=None, choices=["rigid", "deformable"])
    parser.add_argument("--resume-model", type=str, default=None,
        help="Path to a saved SB3 model .zip to continue training from.")
    parser.add_argument("--resume-vecnorm", type=str, default=None,
        help="Path to saved VecNormalize stats .pkl matching --resume-model.")
    parser.add_argument("--resume-replay-buffer", type=str, default=None,
        help="Path to a saved replay buffer .pkl matching --resume-model. If omitted, a matching sibling file is detected automatically.")
    replay_buffer_saving = parser.add_mutually_exclusive_group()
    replay_buffer_saving.add_argument("--save-replay-buffer", action="store_true",
        help="Save full replay buffers with checkpoints. Disabled by default because HER buffers can be very large.")
    replay_buffer_saving.add_argument("--no-save-replay-buffer", action="store_true",
        help="Deprecated compatibility flag; replay-buffer saving is now disabled by default.")
    parser.add_argument("--keep-replay-buffers-after-training", action="store_true",
        help="Keep local replay-buffer files after successful training when --save-replay-buffer is enabled.")
    parser.add_argument("--resume-warmup-steps", type=int, default=None,
        help="Fresh transitions to collect before updating after a bufferless resume (default: --learning-starts).")
    parser.add_argument("--resume-reset-num-timesteps", action="store_true",
        help="Reset SB3 timestep counter when resuming. By default resumed runs continue the counter.")
    args = parser.parse_args()

    if args.resume_warmup_steps is not None and args.resume_warmup_steps < 0:
        parser.error("--resume-warmup-steps must be non-negative")

    # Normalize to absolute path and sanity-check
    args.xml_path = os.path.abspath(args.xml_path)
    if not os.path.isfile(args.xml_path):
        raise FileNotFoundError(f"XML not found at {args.xml_path}")
    args.artifact_root = os.path.abspath(args.artifact_root)
    if args.resume_model is not None:
        args.resume_model = os.path.abspath(args.resume_model)
        if not os.path.isfile(args.resume_model):
            raise FileNotFoundError(f"Resume model not found at {args.resume_model}")
    if args.resume_vecnorm is not None:
        args.resume_vecnorm = os.path.abspath(args.resume_vecnorm)
        if not os.path.isfile(args.resume_vecnorm):
            raise FileNotFoundError(f"Resume VecNormalize stats not found at {args.resume_vecnorm}")
    elif args.resume_model is not None:
        checkpoint_match = re.search(r"model_(\d+)_steps\.zip$", args.resume_model)
        if checkpoint_match:
            inferred_vecnorm = os.path.join(
                os.path.dirname(args.resume_model),
                f"vecnorm_{checkpoint_match.group(1)}.pkl",
            )
            if os.path.isfile(inferred_vecnorm):
                args.resume_vecnorm = inferred_vecnorm
                print(f"Auto-detected VecNormalize stats: {args.resume_vecnorm}")
    if args.resume_replay_buffer is not None:
        args.resume_replay_buffer = os.path.abspath(args.resume_replay_buffer)
        if not os.path.isfile(args.resume_replay_buffer):
            raise FileNotFoundError(f"Resume replay buffer not found at {args.resume_replay_buffer}")
    elif args.resume_model is not None:
        checkpoint_match = re.search(r"model_(\d+)_steps\.zip$", args.resume_model)
        if checkpoint_match:
            inferred_replay_buffer = os.path.join(
                os.path.dirname(args.resume_model),
                f"replay_buffer_{checkpoint_match.group(1)}.pkl",
            )
            if os.path.isfile(inferred_replay_buffer):
                args.resume_replay_buffer = inferred_replay_buffer
                print(f"Auto-detected replay buffer: {args.resume_replay_buffer}")
    
    # Auto-generate wandb name if not provided
    if args.wandb_name is None:
        args.wandb_name = f"{args.env_id}_{args.num_envs}env_{args.seed}"
    
    return args


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("Warning: --device cuda requested but CUDA is unavailable; falling back to CPU.")
        return "cpu"
    return requested_device


def _summary_training_step(remote_run):
    """Return the latest server-side training step recorded in W&B."""
    summary = remote_run.summary
    for key in ("training_step", "global_step", "time/total_timesteps"):
        value = summary.get(key)
        if value is not None:
            try:
                return int(value), key
            except (TypeError, ValueError):
                pass
    return None, None


def _checkpoint_step_from_path(model_path):
    if not model_path:
        return None
    match = re.search(r"model_(\d+)_steps\.zip$", model_path)
    return int(match.group(1)) if match else None


def _format_bytes(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _replay_buffer_array_bytes(replay_buffer):
    """Estimate serialized payload size from unique NumPy arrays."""
    seen_objects = set()
    seen_arrays = set()

    def visit(value):
        object_id = id(value)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)
        if isinstance(value, np.ndarray):
            if object_id in seen_arrays:
                return 0
            seen_arrays.add(object_id)
            return int(value.nbytes)
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(item) for item in value)
        return 0

    return visit(vars(replay_buffer))


def _select_checkpoint_for_remote_step(args, remote_step, max_gap):
    """Select the complete local checkpoint closest to remote W&B history."""
    model_dir = os.path.dirname(args.resume_model)
    candidates = []
    for model_path in glob.glob(os.path.join(model_dir, "model_*_steps.zip")):
        step = _checkpoint_step_from_path(model_path)
        if step is None or abs(step - remote_step) > max_gap:
            continue
        vecnorm_path = os.path.join(model_dir, f"vecnorm_{step}.pkl")
        if not os.path.isfile(vecnorm_path):
            continue
        replay_buffer_path = os.path.join(model_dir, f"replay_buffer_{step}.pkl")
        candidates.append(
            (
                step,
                model_path,
                vecnorm_path,
                replay_buffer_path if os.path.isfile(replay_buffer_path) else None,
            )
        )

    if not candidates:
        raise RuntimeError(
            f"W&B history ends at training step {remote_step:,}, but no complete "
            "model_<step>_steps.zip and vecnorm_<step>.pkl pair exists within "
            f"{max_gap:,} timesteps in {model_dir}."
        )
    # Prefer the closest checkpoint. On equal distance, prefer one just ahead
    # of W&B so newly logged steps remain monotonic, then prefer the newer one.
    return min(
        candidates,
        key=lambda item: (
            abs(item[0] - remote_step),
            0 if item[0] >= remote_step else 1,
            -item[0],
        ),
    )


def _find_large_history_gap(remote_run, step_key, max_gap):
    """Find the first large jump or rewind in an exact W&B metric history."""
    previous = None
    for row in remote_run.scan_history(keys=[step_key], page_size=1000):
        try:
            current = int(row[step_key])
        except (KeyError, TypeError, ValueError):
            continue
        if previous is not None and abs(current - previous) > max_gap:
            return previous, current
        previous = current
    return None


def sync_prior_wandb_sessions(args, wandb_root):
    """Upload incomplete local sessions for this run ID before live resume."""
    if args.skip_wandb_local_sync:
        return

    run_patterns = (
        os.path.join(wandb_root, f"run-*-{args.wandb_id}"),
        os.path.join(wandb_root, f"offline-run-*-{args.wandb_id}"),
        os.path.join(wandb_root, "wandb", f"run-*-{args.wandb_id}"),
        os.path.join(wandb_root, "wandb", f"offline-run-*-{args.wandb_id}"),
    )
    run_dirs = sorted({
        path
        for pattern in run_patterns
        for path in glob.glob(pattern)
        if os.path.isdir(path)
    })
    if not run_dirs:
        print("No prior local W&B session directories found for pre-resume sync.")
        return

    wandb_cli = shutil.which("wandb")
    if not wandb_cli:
        raise RuntimeError(
            "Prior local W&B sessions exist, but the wandb CLI was not found. "
            "Activate the training environment or use --skip-wandb-local-sync."
        )

    command = [
        wandb_cli,
        "sync",
        "--include-online",
        "--include-offline",
        "--append",
        "--id",
        args.wandb_id,
        "--project",
        args.wandb_project,
    ]
    if args.wandb_entity:
        command.extend(["--entity", args.wandb_entity])
    command.extend(run_dirs)

    print(f"Syncing {len(run_dirs)} prior local W&B session(s) before resume...")
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        raise RuntimeError(
            "Local W&B sync failed, so live resume was stopped before wandb.init. "
            f"wandb sync exited with status {result.returncode}."
        )


def validate_wandb_resume_history(args, checkpoint_step, wandb_root):
    """Sync W&B and reconcile the requested checkpoint before wandb.init."""
    if (
        not args.resume_model
        or not args.wandb_id
        or args.skip_wandb_resume_check
    ):
        return checkpoint_step

    if args.wandb_resume not in ("allow", "must", "auto"):
        raise RuntimeError(
            "A resume model and --wandb-id were supplied, but --wandb-resume "
            "does not permit resuming the existing W&B run."
        )

    sync_prior_wandb_sessions(args, wandb_root)

    try:
        api = wandb.Api(timeout=60)
        entity = args.wandb_entity or getattr(api, "default_entity", None)
        if not entity:
            raise RuntimeError(
                "W&B entity could not be inferred; pass --wandb-entity explicitly."
            )
        run_path = f"{entity}/{args.wandb_project}/{args.wandb_id}"
        remote_run = api.run(run_path)
        remote_run.reload()
        remote_step, step_key = _summary_training_step(remote_run)
    except Exception as exc:
        raise RuntimeError(
            "W&B resume preflight failed before wandb.init, so no new data was "
            "written. Ensure prior local W&B data is synced, verify network/login "
            f"access and the run path, then retry. Details: {exc}"
        ) from exc

    if remote_step is None:
        raise RuntimeError(
            "W&B resume preflight found no training_step, global_step, or "
            "time/total_timesteps in the remote run summary. Refusing to append "
            "because checkpoint continuity cannot be verified. Sync the previous "
            "session first, or explicitly use --skip-wandb-resume-check."
        )

    max_gap = max(0, int(args.wandb_resume_max_step_gap))
    try:
        history_gap = _find_large_history_gap(remote_run, step_key, max_gap)
    except Exception as exc:
        raise RuntimeError(
            "W&B resume preflight could not scan the remote metric history, so "
            "continuity cannot be guaranteed and no new data was written. "
            f"Details: {exc}"
        ) from exc
    if history_gap is not None:
        before, after = history_gap
        direction = "jump" if after > before else "rewind"
        raise RuntimeError(
            "Unsafe W&B resume refused before wandb.init: the existing remote "
            f"{step_key} history contains a {direction} from {before:,} to "
            f"{after:,} ({abs(after - before):,} timesteps), exceeding the "
            f"allowed gap of {max_gap:,}. The run already has a large missing "
            "or non-monotonic section; recover/sync its source session or use a "
            "clean W&B run."
        )

    gap = int(checkpoint_step) - remote_step
    if abs(gap) > max_gap:
        if gap > 0:
            if args.no_wandb_checkpoint_fallback:
                raise RuntimeError(
                    "Unsafe W&B resume refused before wandb.init: "
                    f"remote {step_key}={remote_step:,}, checkpoint={checkpoint_step:,}, "
                    f"allowed gap={max_gap:,}; the checkpoint is {gap:,} timesteps "
                    "ahead and --no-wandb-checkpoint-fallback was supplied."
                )

            original_target_step = int(checkpoint_step) + int(args.n_timesteps)
            (
                fallback_step,
                fallback_model,
                fallback_vecnorm,
                fallback_replay_buffer,
            ) = _select_checkpoint_for_remote_step(args, remote_step, max_gap)
            args.resume_model = fallback_model
            args.resume_vecnorm = fallback_vecnorm
            args.resume_replay_buffer = fallback_replay_buffer
            args.n_timesteps = max(0, original_target_step - fallback_step)
            print(
                "W&B history is behind the requested checkpoint by "
                f"{gap:,} timesteps. Automatically rolling back to the closest "
                "checkpoint aligned with the remote logs:"
            )
            print(f"  model: {fallback_model}")
            print(f"  VecNormalize: {fallback_vecnorm}")
            if fallback_replay_buffer:
                print(f"  replay buffer: {fallback_replay_buffer}")
            else:
                print("  replay buffer: unavailable; resume warm-up will be used")
            print(
                f"  adjusted additional training: {int(args.n_timesteps):,} "
                f"timesteps (absolute target remains {original_target_step:,})"
            )
            return fallback_step
        else:
            reason = (
                f"the W&B server is {-gap:,} timesteps ahead of the checkpoint; "
                "the selected checkpoint is stale relative to this run"
            )
        raise RuntimeError(
            "Unsafe W&B resume refused before wandb.init: "
            f"remote {step_key}={remote_step:,}, checkpoint={checkpoint_step:,}, "
            f"allowed gap={max_gap:,}; {reason}. Sync/recover the earlier W&B "
            "session or select the matching checkpoint/run ID."
        )

    print(
        "W&B resume preflight passed: "
        f"remote {step_key}={remote_step:,}, checkpoint={checkpoint_step:,}, "
        f"gap={gap:,}."
    )
    return checkpoint_step

def make_env(
    xml_path,
    seed,
    rank,
    target_position,
    target_rotation,
    ignore_z_rot,
    max_steps=100,
    render_mode=None,
    debug_goals_live=False,
    debug_goals_every=1,
    action_scale=1.0,
    action_clip=None,
    action_smoothing=0.0,
    reset_settle_steps=0,
):
    def _init():
        env = DynamicXMLTouchEnv(
            xml_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            ignore_z_target_rotation=ignore_z_rot,
            render_mode=render_mode,
            debug_goal_print=debug_goals_live,
            debug_goal_print_every=debug_goals_every,
            action_scale=action_scale,
            action_clip=action_clip,
            action_smoothing=action_smoothing,
            reset_settle_steps=reset_settle_steps,
        )
        env = TimeLimit(env, max_episode_steps=max_steps)
        env.reset(seed=seed + rank)
        env = Monitor(env)
        env = TimeFeatureWrapper(env)
        return env
    return _init

def make_eval_env(
    xml_path,
    seed,
    target_position,
    target_rotation,
    ignore_z_rot,
    max_steps=100,
    action_scale=1.0,
    action_clip=None,
    action_smoothing=0.0,
    reset_settle_steps=0,
):
    def _init():
        env = DynamicXMLTouchEnv(
            xml_path=xml_path,
            target_position=target_position,
            target_rotation=target_rotation,
            ignore_z_target_rotation=ignore_z_rot,
            render_mode="rgb_array",
            action_scale=action_scale,
            action_clip=action_clip,
            action_smoothing=action_smoothing,
            reset_settle_steps=reset_settle_steps,
        )
        env = TimeLimit(env, max_episode_steps=max_steps)
        env.reset(seed=seed)
        env = Monitor(env)
        env = TimeFeatureWrapper(env)
        return env

    env = DummyVecEnv([_init])
    env.seed(seed + 1000)
    return env

def evaluate_policy(model, env, n_eval_episodes=10):
    episode_rewards = []
    episode_successes = []
    
    for _ in range(n_eval_episodes):
        reset_return = env.reset()
        
        if isinstance(reset_return, tuple):
            if len(reset_return) >= 1:
                obs = reset_return[0]
            else:
                print("Warning: reset() returned an empty tuple")
                continue
        elif isinstance(reset_return, dict):
            obs = reset_return
        else:
            obs = reset_return
        
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_return = env.step(action)
            
            if isinstance(step_return, tuple):
                if len(step_return) == 5:  # obs, reward, terminated, truncated, info
                    obs, reward, terminated, truncated, info = step_return
                    
                    # Handle different formats of terminated/truncated
                    if isinstance(terminated, bool):
                        done = terminated or truncated
                    elif hasattr(terminated, '__getitem__'):
                        done = terminated[0] or truncated[0]
                    else:
                        done = bool(terminated) or bool(truncated)
                        
                    if hasattr(reward, '__getitem__'):
                        episode_reward += reward[0]
                    else:
                        episode_reward += reward
                        
                    # Check for success
                    if done and isinstance(info, dict) and 'is_success' in info:
                        episode_successes.append(float(info['is_success']))
                    elif done and hasattr(info, '__getitem__') and len(info) > 0:
                        if isinstance(info[0], dict) and 'is_success' in info[0]:
                            episode_successes.append(float(info[0]['is_success']))
                
                elif len(step_return) == 4:  # obs, reward, done, info 
                    obs, reward, done_var, info = step_return
                    
                    if isinstance(done_var, bool):
                        done = done_var
                    elif hasattr(done_var, '__getitem__'):
                        done = done_var[0]
                    else:
                        done = bool(done_var)
                    
                    if hasattr(reward, '__getitem__'):
                        episode_reward += reward[0]
                    else:
                        episode_reward += reward
                        
                    # Check for success
                    if done and isinstance(info, dict) and 'is_success' in info:
                        episode_successes.append(float(info['is_success']))
                    elif done and hasattr(info, '__getitem__') and len(info) > 0:
                        if isinstance(info[0], dict) and 'is_success' in info[0]:
                            episode_successes.append(float(info[0]['is_success']))
            else:
                print(f"Warning: Unexpected return format from step(): {type(step_return)}")
                done = True
        
        episode_rewards.append(episode_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    # Calculate success rate
    success_rate = np.mean(episode_successes) if episode_successes else None
    
    eval_metrics = {
        'eval/mean_reward': mean_reward,
        'eval/std_reward': std_reward,
    }
    
    if success_rate is not None:
        eval_metrics['eval/success_rate'] = success_rate
    
    return eval_metrics

if __name__ == "__main__":
    args = parse_args()
    print(f"Starting training with environment: {args.env_id}")
    print(f"Using {args.num_envs} parallel environments")
    print(f"Eval frequency: {args.eval_freq} timesteps")
    print(f"Total timesteps: {args.n_timesteps}")
    device = resolve_device(args.device)
    print(f"Using torch device: {device}")
    if args.debug_goals_live:
        print(f"Goal debug printing enabled every {max(1, args.debug_goals_every)} step(s)")
    if args.render_human:
        print("Live MuJoCo rendering enabled for training env")
    
    # Ensure directories exist
    video_root = os.path.join(args.artifact_root, "videos", f"{args.env_id}_{args.seed}")
    model_root = os.path.join(args.artifact_root, "models", args.env_id)
    run_root = os.path.join(args.artifact_root, "runs", f"{args.env_id}_{args.num_envs}env_{args.seed}")
    wandb_root = os.path.join(args.artifact_root, "wandb")
    wandb_artifact_root = os.path.join(args.artifact_root, "artifacts", "wandb", f"{args.env_id}_{args.num_envs}env_{args.seed}")
    os.makedirs(video_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(run_root, exist_ok=True)
    os.makedirs(wandb_root, exist_ok=True)
    os.makedirs(wandb_artifact_root, exist_ok=True)

    requested_checkpoint_step = _checkpoint_step_from_path(args.resume_model)
    if args.resume_model and args.wandb_id and requested_checkpoint_step is None:
        raise RuntimeError(
            "Strict W&B resume requires a standard model_<step>_steps.zip "
            "checkpoint filename so continuity can be checked before loading."
        )
    if args.resume_reset_num_timesteps and args.wandb_id:
        raise RuntimeError(
            "--resume-reset-num-timesteps cannot safely append to an existing "
            "W&B run because it would make training steps non-monotonic."
        )
    prepared_checkpoint_step = validate_wandb_resume_history(
        args,
        requested_checkpoint_step,
        wandb_root,
    )
    
    # Build hyperparameters dict
    hyperparams = {
        "policy": "MultiInputPolicy",
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "learning_starts": args.learning_starts,
        "tau": args.tau,
        "ent_coef": args.ent_coef,
        "replay_buffer_kwargs": {
            "goal_selection_strategy": args.goal_selection_strategy,
            "n_sampled_goal": args.n_sampled_goal,
        },
        "policy_kwargs": {
            "net_arch": args.arch,
            "n_critics": args.n_critics,
        }
    }
    
    if hyperparams["ent_coef"] != "auto":
        hyperparams["ent_coef"] = float(hyperparams["ent_coef"])
    

    env_config = {
        'env_id': args.env_id,
        'num_envs': args.num_envs,
        'eval_freq': args.eval_freq,
        'eval_episodes': args.eval_episodes,
        'save_freq': args.save_freq,
        'action_scale': args.action_scale,
        'action_clip': args.action_clip,
        'action_smoothing': args.action_smoothing,
        'reset_settle_steps': args.reset_settle_steps,
        **hyperparams
    }
    
    # Create parallel environments
    xml_path = args.xml_path  # passed from generate_and_train
    train_render_mode = "human" if args.render_human else None
    env_fns = [
        make_env(
            args.xml_path,
            args.seed,
            i,
            args.target_position,
            args.target_rotation,
            args.ignore_z_rot,
            max_steps=args.max_episode_steps,
            render_mode=train_render_mode,
            debug_goals_live=args.debug_goals_live,
            debug_goals_every=args.debug_goals_every,
            action_scale=args.action_scale,
            action_clip=args.action_clip,
            action_smoothing=args.action_smoothing,
            reset_settle_steps=args.reset_settle_steps,
        )
        for i in range(args.num_envs)
    ]
    if args.num_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="spawn")

    normalize_kwargs = {"gamma": hyperparams["gamma"]}
    if args.resume_vecnorm:
        env = VecNormalize.load(args.resume_vecnorm, env)
        env.training = True
        env.norm_reward = True
        print(f"Loaded VecNormalize stats from: {args.resume_vecnorm}")
    else:
        env = VecNormalize(env, **normalize_kwargs)

    # env = VecVideoRecorder(
    #     env,
    #     f"videos/{args.env_id}_{args.seed}",
    #     record_video_trigger=lambda x: x % args.model_save_freq == 0,
    #     video_length=200
    # )

    # Eval env (also direct)
    eval_env = make_eval_env(args.xml_path, args.seed,
                             args.target_position, args.target_rotation, args.ignore_z_rot,
                             max_steps=args.max_episode_steps,
                             action_scale=args.action_scale,
                             action_clip=args.action_clip,
                             action_smoothing=args.action_smoothing,
                             reset_settle_steps=args.reset_settle_steps,
                             )
    eval_env = VecNormalize(eval_env, **normalize_kwargs)
    eval_env.training = False
    eval_env.norm_reward = False
    
    # Adjust buffer size for multiple environments
    if args.num_envs > 1:
        hyperparams["buffer_size"] = max(hyperparams["buffer_size"], hyperparams["buffer_size"] * args.num_envs // 4)
        print(f"Adjusted buffer size: {hyperparams['buffer_size']}")
    
    n_timesteps = int(args.n_timesteps)
    replay_buffer_cleanup_paths = set()
    
    if args.resume_model:
        # SB3 checkpoints persist the original buffer_size.  When resuming
        # without a saved replay buffer, let the current --num-envs setting
        # determine the capacity of the newly allocated buffer instead of
        # recreating the (possibly much larger) capacity from the checkpoint.
        # A loaded replay buffer owns its own storage, so preserve its size.
        resume_load_kwargs = {
            "env": env,
            "device": device,
            "tensorboard_log": run_root,
        }
        if not args.resume_replay_buffer:
            resume_load_kwargs["buffer_size"] = hyperparams["buffer_size"]
            print(
                "Bufferless resume will rebuild the replay buffer with "
                f"capacity {hyperparams['buffer_size']:,} for "
                f"{args.num_envs} environment(s)."
            )
        model = TQC.load(args.resume_model, **resume_load_kwargs)
        model.verbose = args.verbose
        print(f"Loaded model checkpoint from: {args.resume_model}")
        if args.resume_replay_buffer:
            model.load_replay_buffer(
                args.resume_replay_buffer,
                truncate_last_traj=True,
            )
            print(f"Loaded replay buffer from: {args.resume_replay_buffer}")
            if args.resume_reset_num_timesteps:
                # A loaded buffer is immediately sampleable. Do not retain an
                # absolute learning_starts value from the prior timestep axis.
                model.learning_starts = 0
        else:
            # The checkpoint contains the networks and optimizer state, but a
            # fresh HER buffer needs complete episodes before it can be sampled.
            requested_warmup_steps = (
                args.learning_starts
                if args.resume_warmup_steps is None
                else args.resume_warmup_steps
            )
            # SB3 counts timesteps across every vectorized environment. Waiting
            # this long guarantees that at least one TimeLimit episode can
            # finish before HER attempts its first sample.
            minimum_warmup_steps = args.max_episode_steps * args.num_envs
            resume_warmup_steps = max(
                requested_warmup_steps,
                minimum_warmup_steps,
            )
            if resume_warmup_steps != requested_warmup_steps:
                print(
                    "Increasing bufferless-resume warm-up from "
                    f"{requested_warmup_steps:,} to {resume_warmup_steps:,} "
                    "timesteps so HER receives a complete episode."
                )
            warmup_start_step = (
                0 if args.resume_reset_num_timesteps else model.num_timesteps
            )
            model.learning_starts = warmup_start_step + resume_warmup_steps
            print(
                "No replay buffer was loaded. Bufferless resume is collecting "
                f"{resume_warmup_steps:,} fresh transitions before updates "
                f"(until timestep {model.learning_starts:,})."
            )
    else:
        model = TQC(
            env=env, 
            replay_buffer_class=HerReplayBuffer, 
            verbose=args.verbose,
            seed=args.seed, 
            device=device,
            tensorboard_log=run_root,
            **hyperparams
        )

    replay_buffer_bytes = _replay_buffer_array_bytes(model.replay_buffer)
    print(
        "Replay-buffer NumPy payload is approximately "
        f"{_format_bytes(replay_buffer_bytes)} in memory; local pickle size may vary."
    )
    if args.save_replay_buffer:
        print(
            "Replay-buffer checkpoints are local-only and are never passed to "
            "wandb.save or W&B artifacts."
        )
    else:
        print(
            "Compact checkpoints enabled: replay buffers will not be saved. "
            "Resuming will rebuild the buffer from fresh experience."
        )

    checkpoint_step = int(model.num_timesteps)
    if (
        prepared_checkpoint_step is not None
        and checkpoint_step != int(prepared_checkpoint_step)
    ):
        env.close()
        eval_env.close()
        raise RuntimeError(
            "Loaded model timestep does not match its checkpoint filename: "
            f"model contains {checkpoint_step:,}, filename says "
            f"{int(prepared_checkpoint_step):,}. Refusing W&B resume."
        )

    # Do not initialize (and therefore do not write to) W&B until the remote
    # history has been refreshed and checked against the loaded checkpoint.
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=env_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=args.wandb_name,
        id=args.wandb_id,
        resume=args.wandb_resume,
        dir=wandb_root,
    )
    run.define_metric("training_step")
    run.define_metric("eval/*", step_metric="training_step")

    class EvalAndSaveCallback(WandbCallback):
        def __init__(self, vec_env, eval_env, model, save_freq, eval_freq, eval_episodes, 
                 save_path, env_id, seed, normalize_kwargs, xml_path,
                 target_position, target_rotation, ignore_z_rot,
                 metrics_path=None, total_timesteps=None, task_label=None,
                 object_id=None, candidate_id=None, physics_mode=None,
                 video_root=None, wandb_artifact_path=None, disable_eval_video=False, action_scale=1.0,
                 action_clip=None, action_smoothing=0.0, reset_settle_steps=0,
                 max_episode_steps=100, eval_video_steps=200,
                 save_replay_buffer=True, replay_buffer_cleanup_paths=None,
                 initial_timestep=None,
                 **kwargs):
            super().__init__(**kwargs)
            self.vec_env = vec_env
            self.eval_env = eval_env
            self.model = model
            self.save_freq = save_freq
            self.eval_freq = eval_freq
            self.eval_episodes = eval_episodes
            self.save_path = save_path
            self.env_id = env_id
            self.seed = seed
            self.normalize_kwargs = normalize_kwargs
            self.xml_path = xml_path
            self.target_position = target_position
            self.target_rotation = target_rotation
            self.ignore_z_rot = ignore_z_rot
            self.best_success_rate = 0.0

            #New BO metrics 
            self.metrics_path = metrics_path
            self.total_timesteps = int(total_timesteps) if total_timesteps is not None else None
            self.task_label = (task_label or env_id)
            self.object_id = object_id
            self.candidate_id = candidate_id
            self.physics_mode = physics_mode
            self.video_root = video_root
            self.wandb_artifact_path = wandb_artifact_path
            self.disable_eval_video = disable_eval_video
            self.action_scale = action_scale
            self.action_clip = action_clip
            self.action_smoothing = action_smoothing
            self.reset_settle_steps = reset_settle_steps
            self.max_episode_steps = max_episode_steps
            self.eval_video_steps = eval_video_steps
            self.save_replay_buffer = save_replay_buffer
            self.replay_buffer_cleanup_paths = replay_buffer_cleanup_paths
            self.checkpoint_steps = []
            self.success_curve = []
            # On resume, schedule the next save/eval relative to the loaded
            # checkpoint. A reset resume starts its schedule from zero.
            schedule_start = (
                int(model.num_timesteps)
                if initial_timestep is None
                else int(initial_timestep)
            )
            self._last_eval_ts = schedule_start
            self._last_save_ts = schedule_start

        def _safe_artifact_name(self, value):
            return "".join(c if c.isalnum() or c in "-_." else "-" for c in value)
        
        def _save_and_upload_best_model(self, step_ts, success_rate):
            if self.wandb_artifact_path is None:
                self.wandb_artifact_path = self.save_path
            os.makedirs(self.wandb_artifact_path, exist_ok=True)

            best_model_path = os.path.join(self.wandb_artifact_path, "best_model.zip")
            best_stats_path = os.path.join(self.wandb_artifact_path, "best_vecnorm.pkl")
            self.model.save(best_model_path)
            self.vec_env.save(best_stats_path)

            artifact_name = self._safe_artifact_name(f"{self.env_id}-{self.seed}-best-model")
            try:
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="model",
                    metadata={
                        "env_id": self.env_id,
                        "seed": int(self.seed),
                        "step": int(step_ts),
                        "success_rate": float(success_rate),
                    },
                )
                artifact.add_file(best_model_path, name="best_model.zip")
                artifact.add_file(best_stats_path, name="best_vecnorm.pkl")
                logged_artifact = wandb.run.log_artifact(
                    artifact,
                    aliases=["best", "latest"],
                )
                logged_artifact.wait()
                print(
                    f"Uploaded best model artifact '{artifact_name}' "
                    f"at {step_ts} timesteps"
                )
            except Exception as exc:
                # W&B is observability, not part of the training state. The
                # model and VecNormalize statistics were saved locally above,
                # so a transient upload failure must not terminate training.
                print(
                    "WARNING: W&B best-model artifact upload failed; "
                    "continuing training with the locally saved files. "
                    f"Artifact: '{artifact_name}'. Error: {exc}"
                )
            return best_model_path
            
        def _on_step(self):
            super()._on_step()
            step_ts = int(self.model.num_timesteps)
            
            if step_ts - self._last_save_ts >= self.save_freq:
                stats_path = os.path.join(self.save_path, f"vecnorm_{step_ts}.pkl")
                self.vec_env.save(stats_path)
                wandb.save(stats_path)
                print(f"Saved VecNormalize stats to {stats_path}")

                if self.save_replay_buffer:
                    replay_buffer_path = os.path.join(
                        self.save_path, f"replay_buffer_{step_ts}.pkl"
                    )
                    self.model.save_replay_buffer(replay_buffer_path)
                    if self.replay_buffer_cleanup_paths is not None:
                        self.replay_buffer_cleanup_paths.add(replay_buffer_path)
                    print(f"Saved replay buffer to {replay_buffer_path}")

                # Save the model last. Its presence indicates that the paired
                # normalization (and optional replay-buffer) write completed.
                model_path = os.path.join(self.save_path, f"model_{step_ts}_steps.zip")
                self.model.save(model_path)
                print(f"Saved model to {model_path}")

                self._last_save_ts = step_ts
            
            # Run evaluation periodically
            if step_ts - self._last_eval_ts >= self.eval_freq:
                print(f"\nRunning evaluation at {step_ts} timesteps...")
                
                try:
                    self.eval_env.obs_rms = deepcopy(self.vec_env.obs_rms)
                    self.eval_env.ret_rms = deepcopy(self.vec_env.ret_rms)
                    print("Successfully copied normalization stats")
                except Exception as e:
                    print(f"Warning: Could not copy normalization stats: {e}")
                
                # Run evaluation and log metrics
                eval_metrics = evaluate_policy(
                    model=self.model, 
                    env=self.eval_env, 
                    n_eval_episodes=self.eval_episodes
                )
                
                # Log evaluation metrics to wandb
                wandb.log({"training_step": step_ts, **eval_metrics})
                
                print(f"Evaluation results: {eval_metrics}")

                #Record success curve point 
                succ = float(eval_metrics.get('eval/success_rate', 0.0))
                self.checkpoint_steps.append(step_ts)
                self.success_curve.append(succ)
                
                # Save best model based on success rate if available
                if 'eval/success_rate' in eval_metrics and eval_metrics['eval/success_rate'] > self.best_success_rate:
                    self.best_success_rate = eval_metrics['eval/success_rate']
                    best_model_path = self._save_and_upload_best_model(step_ts, self.best_success_rate)
                    print(f"New best model with success rate {self.best_success_rate:.2f} saved to {best_model_path}")
                
                if not self.disable_eval_video:
                    try:
                        print(f"Creating evaluation video at step {step_ts}...")
                        
                        video_path = os.path.join(self.video_root, f"eval_{step_ts}")
                        os.makedirs(video_path, exist_ok=True)

                        # fresh env dedicated to video (has render_mode='rgb_array')
                        video_eval_env = make_eval_env(self.xml_path, self.seed,
                                      self.target_position, self.target_rotation, self.ignore_z_rot,
                                      max_steps=self.max_episode_steps,
                                      action_scale=self.action_scale,
                                      action_clip=self.action_clip,
                                      action_smoothing=self.action_smoothing,
                                      reset_settle_steps=self.reset_settle_steps)
                        video_eval_env = VecNormalize(video_eval_env, **self.normalize_kwargs)
                        # copy normalization stats so obs are comparable
                        video_eval_env.obs_rms = deepcopy(self.eval_env.obs_rms)
                        video_eval_env.ret_rms = deepcopy(self.eval_env.ret_rms)
                        video_eval_env.training = False
                        video_eval_env.norm_reward = False

                        video_env = VecVideoRecorder(
                            video_eval_env,
                            video_path,
                            record_video_trigger=lambda x: x == 0,  # record only first episode
                            video_length=self.eval_video_steps,
                            name_prefix=f"eval-{step_ts}"
                        )
                        
                        reset_return = video_env.reset()
                        if isinstance(reset_return, tuple):
                            if len(reset_return) >= 1:
                                obs = reset_return[0]
                            else:
                                raise ValueError("Reset returned an empty tuple")
                        elif isinstance(reset_return, dict):
                            obs = reset_return
                        else:
                            obs = reset_return
                        
                        try:
                            video_env.render("rgb_array")
                        except Exception as e:
                            print(f"Initial render failed, but continuing: {e}")
                            
                        done = False
                        step_count = 0
                        max_steps = self.eval_video_steps  # Maximum video length
                        
                        print("Recording evaluation episode...")
                        while not done and step_count < max_steps:
                            action, _ = self.model.predict(obs, deterministic=True)
                            
                            try:
                                video_env.render("rgb_array")
                            except Exception as e:
                                pass 
                            
                            step_return = video_env.step(action)
                            
                            if isinstance(step_return, tuple):
                                if len(step_return) == 5:  # obs, reward, terminated, truncated, info
                                    obs, _, terminated, truncated, _ = step_return
                                    
                                    if isinstance(terminated, bool):
                                        done = terminated or truncated
                                    elif hasattr(terminated, '__getitem__'):
                                        done = terminated[0] or truncated[0]
                                    else:
                                        done = bool(terminated) or bool(truncated)
                                
                                elif len(step_return) == 4:  # obs, reward, done, info (old Gym API)
                                    obs, _, done_var, _ = step_return
                                    
                                    if isinstance(done_var, bool):
                                        done = done_var
                                    elif hasattr(done_var, '__getitem__'):
                                        done = done_var[0]
                                    else:
                                        done = bool(done_var)
                            else:
                                print(f"Warning: Unexpected return format from step(): {type(step_return)}")
                                done = True
                            
                            try:
                                video_env.render("rgb_array")
                            except Exception as e:
                                pass  # VecVideoRecorder might handle this internally
                                
                            step_count += 1
                        
                        video_env.close()
                        print(f"Finished recording evaluation video after {step_count} steps")
                    
                        video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]
                        if video_files:
                            eval_video_path = os.path.join(video_path, video_files[0])
                            print(f"Found video file: {eval_video_path}")
                            
                            if os.path.exists(eval_video_path) and os.path.getsize(eval_video_path) > 1000:
                                print(f"Logging video to wandb: {eval_video_path}")
                                wandb.log({
                                    "training_step": step_ts,
                                    "eval/video": wandb.Video(eval_video_path, fps=30, format="mp4"),
                                    "eval/video_step": step_ts
                                })
                                print("Successfully logged video to wandb")
                            else:
                                print(f"Warning: Video file is empty or too small: {eval_video_path}")
                        else:
                            print(f"Warning: No video files found in {video_path}")
                        
                    except Exception as e:
                        print(f"Error creating evaluation video: {e}")
                        import traceback
                        traceback.print_exc()
                self._last_eval_ts = step_ts

            return True
        
        def _on_training_end(self) -> None:
            #ensure at least one eval point 
            step_ts = int(self.model.num_timesteps)
            if not self.checkpoint_steps or self.checkpoint_steps[-1] < step_ts:
                eval_metrics = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.eval_episodes)
                succ = float(eval_metrics.get('eval/success_rate', 0.0))
                self.checkpoint_steps.append(step_ts)
                self.success_curve.append(succ)
                wandb.log({"training_step": step_ts, **eval_metrics})
            
            # write BO metrics JSON
            if self.metrics_path:
                denom = self.total_timesteps if self.total_timesteps else max(1, step_ts)
                fracs = [min(1.0, s / denom) for s in self.checkpoint_steps]  # 0..1 axis
                task = self.task_label
                data = {
                    "tasks": [task],
                    "checkpoints": fracs,
                    "success": {task: [float(x) for x in self.success_curve]},
                    "final_success": {task: float(self.success_curve[-1])}
                }
                if self.object_id is not None:
                    data["object_id"] = self.object_id
                if self.candidate_id is not None:
                    data["candidate_id"] = self.candidate_id
                if self.physics_mode is not None:
                    data["physics_mode"] = self.physics_mode
                data["seed"] = int(self.seed)
                os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
                with open(self.metrics_path, "w") as f:
                    json.dump(data, f, indent=2)

            # print a simple scalar for fallback parsers
            print(f"FINAL_SCORE: {self.success_curve[-1]:.6f}")
    
    # Custom callback
    model.learn(
        total_timesteps=n_timesteps,
        reset_num_timesteps=(not args.resume_model) or args.resume_reset_num_timesteps,
        callback=EvalAndSaveCallback(
            vec_env=env,
            xml_path=args.xml_path,
            target_position=args.target_position,
            target_rotation=args.target_rotation,
            ignore_z_rot=args.ignore_z_rot,
            eval_env=eval_env,
            model=model,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_path=model_root,
            env_id=args.env_id,
            seed=args.seed,
            normalize_kwargs=normalize_kwargs,
            #for GP-BO 
            metrics_path=args.metrics_json,
            total_timesteps=n_timesteps,
            task_label=(args.task_name or args.env_id),
            object_id=args.object_id,
            candidate_id=args.candidate_id,
            physics_mode=args.physics_mode,
            video_root=video_root,
            wandb_artifact_path=wandb_artifact_root,
            disable_eval_video=args.disable_eval_video,
            action_scale=args.action_scale,
            action_clip=args.action_clip,
            action_smoothing=args.action_smoothing,
            reset_settle_steps=args.reset_settle_steps,
            max_episode_steps=args.max_episode_steps,
            eval_video_steps=args.eval_video_steps,
            save_replay_buffer=args.save_replay_buffer,
            replay_buffer_cleanup_paths=replay_buffer_cleanup_paths,
            initial_timestep=(
                0 if args.resume_reset_num_timesteps else model.num_timesteps
            ),


            gradient_save_freq=args.gradient_save_freq,
            model_save_freq=args.model_save_freq,
            model_save_path=model_root,
            verbose=args.verbose,
        ),
    )
    
    # Save final model and normalization stats
    final_model_path = os.path.join(model_root, f"{args.env_id}_{args.num_envs}env_{args.seed}_final")
    final_stats_path = os.path.join(model_root, f"{args.env_id}_{args.num_envs}env_{args.seed}_vecnorm_final.pkl")
    final_replay_buffer_path = os.path.join(
        model_root,
        f"{args.env_id}_{args.num_envs}env_{args.seed}_replay_buffer_final.pkl",
    )
    
    env.save(final_stats_path)
    # A final replay buffer is useful only when the user explicitly keeps it;
    # otherwise avoid a large write that would be deleted moments later.
    if args.save_replay_buffer and args.keep_replay_buffers_after_training:
        model.save_replay_buffer(final_replay_buffer_path)
        replay_buffer_cleanup_paths.add(final_replay_buffer_path)
        print(f"Final replay buffer saved to: {final_replay_buffer_path}")
    model.save(final_model_path)
    
    # Run a final evaluation
    final_eval_metrics = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
    wandb.log({"training_step": int(model.num_timesteps), **final_eval_metrics})
    
    print(f"Training finished!")
    print(f"Final evaluation results: {final_eval_metrics}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Final normalization stats saved to: {final_stats_path}")
    
    run.finish()

    if not args.keep_replay_buffers_after_training:
        removed_count = 0
        removed_bytes = 0
        for replay_buffer_path in sorted(replay_buffer_cleanup_paths):
            if not os.path.isfile(replay_buffer_path):
                continue
            try:
                removed_bytes += os.path.getsize(replay_buffer_path)
                os.remove(replay_buffer_path)
                removed_count += 1
            except OSError as exc:
                print(f"Warning: could not delete replay buffer {replay_buffer_path}: {exc}")
        print(
            f"Deleted {removed_count} local replay-buffer file(s) "
            f"({_format_bytes(removed_bytes)}) after successful training."
        )
