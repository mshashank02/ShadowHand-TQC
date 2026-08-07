#!/bin/bash
# Resume n0500_a0p5_b0p2_obj_size_large_ar_high_macro_high_rough_high_rigid.
# This handles legacy checkpoints whose filename step differs from the global
# vectorized timestep stored inside the SB3 model.
#SBATCH --account=bewv-dtai-gh
#SBATCH --job-name=n0500-large-high-resume16
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

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

candidate_id=n0500_a0p5_b0p2
object_id=obj_size_large_ar_high_macro_high_rough_high
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

filename_step=${BASH_REMATCH[1]}
resume_model="${model_root}/${checkpoint_name}"
resume_vecnorm="${model_root}/vecnorm_${filename_step}.pkl"
if [[ ! -f "$resume_vecnorm" ]]; then
  echo "Matching VecNormalize checkpoint not found: ${resume_vecnorm}" >&2
  exit 2
fi

# Use the authoritative timestep stored in the model. If this legacy pair is
# mislabeled, create canonical symlink names so the training script's strict
# filename-versus-model validation continues to protect W&B history.
checkpoint_step=$(python -c \
  'import sys; from stable_baselines3.common.save_util import load_from_zip_file; data, _, _ = load_from_zip_file(sys.argv[1], device="cpu"); print(int(data["num_timesteps"]))' \
  "$resume_model" | tail -n 1)
if [[ ! "$checkpoint_step" =~ ^[0-9]+$ ]]; then
  echo "Could not read a valid internal timestep from ${resume_model}: ${checkpoint_step}" >&2
  exit 2
fi

if (( checkpoint_step != filename_step )); then
  canonical_model="${model_root}/model_${checkpoint_step}_steps.zip"
  canonical_vecnorm="${model_root}/vecnorm_${checkpoint_step}.pkl"
  echo "Checkpoint filename says ${filename_step} steps, but model contains ${checkpoint_step}."

  if [[ ! -e "$canonical_model" && ! -e "$canonical_vecnorm" ]]; then
    ln -s "$checkpoint_name" "$canonical_model"
    ln -s "$(basename "$resume_vecnorm")" "$canonical_vecnorm"
    echo "Created canonical checkpoint symlinks for step ${checkpoint_step}."
  elif [[ ! -f "$canonical_model" || ! -f "$canonical_vecnorm" ]]; then
    echo "Incomplete canonical checkpoint pair at step ${checkpoint_step}; refusing to overwrite it." >&2
    exit 2
  fi

  resume_model="$canonical_model"
  resume_vecnorm="$canonical_vecnorm"
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
echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
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
