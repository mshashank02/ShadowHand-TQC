#!/bin/bash
# Resume the three crashed/failed rigid-object runs shown in W&B.
# Each array task discovers the newest complete model/VecNormalize checkpoint
# and trains the remaining portion of the original 16,000,000-step budget.
# The replay buffer was not checkpointed, so it is rebuilt during warm-up.
#SBATCH --account=bewv-dtai-gh
#SBATCH --job-name=crashed-resume-env16
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --array=0-2%3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate base

REPO_ROOT=/work/hdd/bewv/smahesh1/ShadowHand-TQC
cd "$REPO_ROOT"
mkdir -p logs

export PYTHONPATH=/u/smahesh1/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SDL_VIDEODRIVER=dummy
export WANDB_CONSOLE=off
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# candidate_id|object_size|object_id
CONFIGS=(
  "n1000_a0p1_b0p9|small|obj_size_small_ar_low_macro_low_rough_low"
  "n0500_a0p5_b0p2|large|obj_size_large_ar_high_macro_high_rough_high"
  "n0500_a0p5_b0p2|small|obj_size_small_ar_low_macro_low_rough_low"
)

array_index=${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array job}
if (( array_index < 0 || array_index >= ${#CONFIGS[@]} )); then
  echo "Invalid array index ${array_index}; expected 0-$(( ${#CONFIGS[@]} - 1 ))" >&2
  exit 2
fi

IFS='|' read -r candidate_id object_size object_id <<< "${CONFIGS[$array_index]}"

run_label="${candidate_id}_${object_id}_rigid"
run_root="generated/studies/sphere_v1_demo/candidates/${candidate_id}/${object_id}/rigid"
model_root="${run_root}/models/${run_label}"
metrics_json="${run_root}/metrics.json"
target_steps=16000000

checkpoint_name=$(find "$model_root" -maxdepth 1 -type f -name 'model_*_steps.zip' -printf '%f\n' | sort -V | tail -n 1)
if [[ -z "$checkpoint_name" ]]; then
  echo "No model checkpoint found in ${model_root}" >&2
  exit 2
fi
if [[ ! "$checkpoint_name" =~ ^model_([0-9]+)_steps\.zip$ ]]; then
  echo "Unexpected checkpoint filename: ${checkpoint_name}" >&2
  exit 2
fi

checkpoint_step=${BASH_REMATCH[1]}
resume_model="${model_root}/${checkpoint_name}"
resume_vecnorm="${model_root}/vecnorm_${checkpoint_step}.pkl"
if [[ ! -f "$resume_vecnorm" ]]; then
  echo "Matching VecNormalize checkpoint not found: ${resume_vecnorm}" >&2
  exit 2
fi

xml_path=$(find "$run_root" -type f -name 'manipulate_*_touch_sensors_*.xml' -print -quit)
if [[ -z "$xml_path" ]]; then
  echo "No generated touch-sensor XML found below ${run_root}" >&2
  exit 2
fi

remaining_steps=$((target_steps - checkpoint_step))
if (( remaining_steps <= 0 )); then
  echo "${run_label} is already at or above ${target_steps} steps; nothing to resume."
  exit 0
fi

echo "[$(hostname)] Resuming ${run_label}"
echo "Slurm array job: ${SLURM_ARRAY_JOB_ID:-unknown}_${array_index}"
echo "Allocated GPU(s): ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Checkpoint: ${resume_model} (${checkpoint_step} steps)"
echo "Remaining training: ${remaining_steps} steps; num-envs: 16"
echo "XML: ${xml_path}"

python ShadowHand_TQC.py \
  --env-id "$run_label" \
  --xml-path "$xml_path" \
  --artifact-root "$run_root" \
  --device cuda \
  --num-envs 16 \
  --seed 0 \
  --n-timesteps "$remaining_steps" \
  --learning-starts 10000 \
  --resume-warmup-steps 10000 \
  --eval-freq 20000 \
  --eval-episodes 10 \
  --disable-eval-video \
  --wandb-project single_end_to_end_shadowhand \
  --wandb-name "${run_label}_resume16" \
  --target-position random \
  --metrics-json "$metrics_json" \
  --task-name "$object_id" \
  --object-id "$object_id" \
  --candidate-id "$candidate_id" \
  --physics-mode rigid \
  --resume-model "$resume_model" \
  --resume-vecnorm "$resume_vecnorm" \
  >"logs/${run_label}_resume16.out" 2>"logs/${run_label}_resume16.err"

echo "Finished ${run_label}"
