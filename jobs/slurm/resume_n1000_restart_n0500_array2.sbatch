#!/bin/bash
# Submit both requested jobs with:
#   sbatch resume_n1000_restart_n0500_array2.sbatch
# Array task 0 resumes the n1000 run from its newest matched checkpoint.
# Array task 1 restarts the crashed n0500 run from scratch.
#SBATCH --account=bewv-dtai-gh
#SBATCH --job-name=resume-n1000-restart-n0500
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --array=0-1%2
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate base

REPO_ROOT=/work/hdd/bewv/smahesh1/ShadowHand-TQC
cd "$REPO_ROOT"
mkdir -p logs generated

export PYTHONPATH=/u/smahesh1/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SDL_VIDEODRIVER=dummy
export WANDB_CONSOLE=off
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

array_index=${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch as an array job}
target_steps=16000000

case "$array_index" in
  0)
    # Screenshot run: n1000_a0p1_b0p9_obj_size_medium_ar_low_macro_low_rough_low_rigid_resume8
    candidate_id=n1000_a0p1_b0p9
    object_id=obj_size_medium_ar_low_macro_low_rough_low
    run_label="${candidate_id}_${object_id}_rigid"
    run_root="generated/studies/sphere_v1_demo/candidates/${candidate_id}/${object_id}/rigid"
    model_root="${run_root}/models/${run_label}"
    metrics_json="${run_root}/metrics.json"

    checkpoint_name=$(find "$model_root" -maxdepth 1 -type f -name 'model_*_steps.zip' -printf '%f\n' | sort -V | tail -n 1)
    if [[ -z "$checkpoint_name" || ! "$checkpoint_name" =~ ^model_([0-9]+)_steps\.zip$ ]]; then
      echo "No valid model checkpoint found in ${model_root}" >&2
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

    echo "[$(hostname)] Resuming ${run_label} from ${checkpoint_step} steps"
    echo "Remaining training: ${remaining_steps} steps"
    python ShadowHand_TQC.py \
      --env-id "$run_label" \
      --xml-path "$xml_path" \
      --artifact-root "$run_root" \
      --device cuda \
      --num-envs 12 \
      --seed 0 \
      --n-timesteps "$remaining_steps" \
      --learning-starts 10000 \
      --resume-warmup-steps 10000 \
      --eval-freq 20000 \
      --eval-episodes 10 \
      --disable-eval-video \
      --wandb-project single_end_to_end_shadowhand \
      --wandb-name "${run_label}_resume9" \
      --target-position random \
      --metrics-json "$metrics_json" \
      --task-name "$object_id" \
      --object-id "$object_id" \
      --candidate-id "$candidate_id" \
      --physics-mode rigid \
      --resume-model "$resume_model" \
      --resume-vecnorm "$resume_vecnorm" \
      >"logs/${run_label}_resume9.out" 2>"logs/${run_label}_resume9.err"
    ;;

  1)
    # Crashed run supplied by name; restart it from scratch.
    candidate_id=n0500_a0p5_b0p2
    object_size=medium
    object_id=obj_size_medium_ar_low_macro_high_rough_low
    task_path=study_objects/sphere_study_v1/obj_size-medium_ar-low_macro-high_rough-low.msh
    run_label="${candidate_id}_${object_id}_rigid"
    run_root="generated/studies/sphere_v1_demo/candidates/${candidate_id}/${object_id}/rigid"
    metrics_json="${run_root}/metrics.json"

    if [[ ! -f "$task_path" ]]; then
      echo "Object mesh not found: ${task_path}" >&2
      exit 2
    fi

    mkdir -p "$run_root"
    echo "[$(hostname)] Restarting ${run_label} from scratch"
    python generate_and_train.py \
      --base assets/hand_base.xml \
      --task "$task_path" \
      --object-size "$object_size" \
      --object-id "$object_id" \
      --run-label "$run_label" \
      --candidate-id "$candidate_id" \
      --physics-mode rigid \
      --Ntotal 500 \
      --Rppx 1.25 \
      --Rpt 5 \
      --out-root "$run_root" \
      --artifact-root "$run_root" \
      --force \
      -- \
      --device cuda \
      --num-envs 32 \
      --seed 0 \
      --n-timesteps "$target_steps" \
      --learning-starts 10000 \
      --eval-freq 20000 \
      --eval-episodes 10 \
      --disable-eval-video \
      --wandb-project single_end_to_end_shadowhand \
      --target-position random \
      --metrics-json "$metrics_json" \
      --task-name "$object_id" \
      --object-id "$object_id" \
      --candidate-id "$candidate_id" \
      --physics-mode rigid \
      >"logs/${run_label}_restart.out" 2>"logs/${run_label}_restart.err"
    ;;

  *)
    echo "Invalid array index ${array_index}; expected 0 or 1" >&2
    exit 2
    ;;
esac

echo "Finished array task ${array_index}"
