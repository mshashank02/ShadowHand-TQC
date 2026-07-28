#!/usr/bin/env bash
#
# Submit both experiments as a two-task array:
#   sbatch train_n0500_a0p5_b0p2_two_configs.sbatch
#
# Each array task requests one GPU. Slurm chooses the device and exposes it
# through CUDA_VISIBLE_DEVICES; never replace that value inside the job.
#
# IMPORTANT: Slurm processes #SBATCH lines before Bash runs, so resource values
# cannot reference the shell variables below. Edit the directives and the
# matching documentation variables together when changing resource requests.
#
#SBATCH --job-name=n0500-a0p5-b0p2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=0
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# ---------------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------------
REPO_ROOT="/home/mshashank02/ShadowHand-TQC"
CONDA_ENV="base"
LOG_DIR="${REPO_ROOT}/logs"

# Keep these values synchronized with the #SBATCH resource directives above.
CPUS_PER_TASK=12
GPUS_PER_TASK=1
MEMORY_PER_TASK="48G"
TIME_LIMIT="unlimited (subject to partition MaxTime)"

# Shared experiment settings.
NUM_ENVS=36
SEED=0
N_TIMESTEPS=16000000
WANDB_PROJECT="single_end_to_end_shadowhand"

# candidate_id|Ntotal|Rppx|Rpt|object_size|object_id|mesh_path
CONFIGS=(
  "n0500_a0p5_b0p2|500|1.25|5|medium|obj_size_medium_ar_low_macro_high_rough_low|study_objects/sphere_study_v1/obj_size-medium_ar-low_macro-high_rough-low.msh"
  "n0500_a0p5_b0p2|500|1.25|5|small|obj_size_small_ar_low_macro_low_rough_low|study_objects/sphere_study_v1/obj_size-small_ar-low_macro-low_rough-low.msh"
)

# ---------------------------------------------------------------------------
# Job setup
# ---------------------------------------------------------------------------
START_EPOCH=$SECONDS
ARRAY_INDEX=${SLURM_ARRAY_TASK_ID:?Submit this file with sbatch as an array job}

if (( ARRAY_INDEX < 0 || ARRAY_INDEX >= ${#CONFIGS[@]} )); then
  printf 'Invalid array index %s; expected 0-%s\n' \
    "$ARRAY_INDEX" "$(( ${#CONFIGS[@]} - 1 ))" >&2
  exit 2
fi

IFS='|' read -r \
  CANDIDATE_ID NTOTAL RPPX RPT OBJECT_SIZE OBJECT_ID TASK_PATH \
  <<< "${CONFIGS[$ARRAY_INDEX]}"

RUN_LABEL="${CANDIDATE_ID}_${OBJECT_ID}_rigid"
RUN_ROOT="generated/studies/sphere_v1_demo/candidates/${CANDIDATE_ID}/${OBJECT_ID}/rigid"
METRICS_JSON="${RUN_ROOT}/metrics.json"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

if [[ ! -f "$TASK_PATH" ]]; then
  printf 'Object mesh not found: %s\n' "$TASK_PATH" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"

# Make `conda activate` available in non-interactive batch shells.
CONDA_BASE=$(conda info --base)
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Headless rendering and deterministic CPU-thread ownership. Do not set or
# modify CUDA_VISIBLE_DEVICES: it is supplied by Slurm for this allocation.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SDL_VIDEODRIVER=dummy
export WANDB_CONSOLE=off
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Store the command as an array so every argument remains correctly quoted.
# This array is invoked exactly once below.
TRAIN_COMMAND=(
  python generate_and_train.py
  --base assets/hand_base.xml
  --task "$TASK_PATH"
  --object-size "$OBJECT_SIZE"
  --object-id "$OBJECT_ID"
  --run-label "$RUN_LABEL"
  --candidate-id "$CANDIDATE_ID"
  --physics-mode rigid
  --Ntotal "$NTOTAL"
  --Rppx "$RPPX"
  --Rpt "$RPT"
  --out-root "$RUN_ROOT"
  --artifact-root "$RUN_ROOT"
  --force
  --
  --device cuda
  --num-envs "$NUM_ENVS"
  --seed "$SEED"
  --n-timesteps "$N_TIMESTEPS"
  --learning-starts 10000
  --eval-freq 20000
  --eval-episodes 10
  --disable-eval-video
  --wandb-project "$WANDB_PROJECT"
  --target-position random
  --metrics-json "$METRICS_JSON"
  --task-name "$OBJECT_ID"
  --object-id "$OBJECT_ID"
  --candidate-id "$CANDIDATE_ID"
  --physics-mode rigid
)

# ---------------------------------------------------------------------------
# Reproducibility and allocation diagnostics
# ---------------------------------------------------------------------------
printf '%s\n' '========== Job information =========='
printf 'Run label:            %s\n' "$RUN_LABEL"
printf 'Slurm job ID:         %s\n' "${SLURM_JOB_ID:-unknown}"
printf 'Slurm array task:     %s\n' "$ARRAY_INDEX"
printf 'Hostname:             %s\n' "$(hostname)"
printf 'Start date:           %s\n' "$(date --iso-8601=seconds)"
printf 'Working directory:    %s\n' "$PWD"
printf 'Conda environment:    %s\n' "${CONDA_DEFAULT_ENV:-unknown}"
printf 'Allocated GPU IDs:    %s\n' "${SLURM_JOB_GPUS:-unknown}"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
printf 'Allocated CPUs:       %s\n' "${SLURM_CPUS_PER_TASK:-$CPUS_PER_TASK}"
printf 'CPU binding:          %s\n' "${SLURM_CPU_BIND:-not reported}"
printf 'Allocated memory:     %s MB/node\n' "${SLURM_MEM_PER_NODE:-unknown}"
printf 'Requested resources:  %s CPU, %s GPU, %s RAM, %s\n' \
  "$CPUS_PER_TASK" "$GPUS_PER_TASK" "$MEMORY_PER_TASK" "$TIME_LIMIT"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  printf 'Git commit:           %s\n' "$(git rev-parse HEAD)"
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    printf '%s\n' 'Git working tree:     modified'
  else
    printf '%s\n' 'Git working tree:     clean'
  fi
else
  printf '%s\n' 'Git commit:           not in a Git repository'
fi
printf 'Training command:     '
printf '%q ' "${TRAIN_COMMAND[@]}"
printf '\n%s\n' '========== NVIDIA status =========='
nvidia-smi
printf '%s\n' '==================================='

# Run one training process in the allocation and preserve its return status.
set +e
srun --cpu-bind=cores "${TRAIN_COMMAND[@]}"
RETURN_CODE=$?
set -e

ELAPSED=$(( SECONDS - START_EPOCH ))
printf 'Finished: %s\n' "$(date --iso-8601=seconds)"
printf 'Total runtime: %02d:%02d:%02d\n' \
  "$(( ELAPSED / 3600 ))" "$(( (ELAPSED % 3600) / 60 ))" "$(( ELAPSED % 60 ))"
printf 'Training return code: %d\n' "$RETURN_CODE"

exit "$RETURN_CODE"
