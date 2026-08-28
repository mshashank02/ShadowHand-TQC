#!/bin/bash
#SBATCH --account=bewv-dtai-gh
#SBATCH --job-name=shadow16-r2
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=30:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err

set -uo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate base

REPO_ROOT=/work/hdd/bewv/smahesh1/ShadowHand-TQC
cd "$REPO_ROOT"
mkdir -p logs generated

export PYTHONPATH=/u/smahesh1/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy
export WANDB_CONSOLE=off PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CONFIGS=(
  "small|obj_size_small_ar_high_macro_high_rough_low|study_objects/sphere_study_v1/obj_size-small_ar-high_macro-high_rough-low.msh"
  "large|obj_size_large_ar_low_macro_high_rough_low|study_objects/sphere_study_v1/obj_size-large_ar-low_macro-high_rough-low.msh"
  "medium|obj_size_medium_ar_low_macro_high_rough_low|study_objects/sphere_study_v1/obj_size-medium_ar-low_macro-high_rough-low.msh"
  "small|obj_size_small_ar_low_macro_high_rough_low|study_objects/sphere_study_v1/obj_size-small_ar-low_macro-high_rough-low.msh"
)

run_one() {
  local gpu="$1" object_size="$2" object_id="$3" task_path="$4"
  local candidate_id="n0500_a0p5_b0p2"
  local run_label="${candidate_id}_${object_id}_rigid"
  local run_root="generated/studies/sphere_v1_demo/candidates/${candidate_id}/${object_id}/rigid"
  mkdir -p "$run_root"
  echo "[$(hostname)] Starting ${run_label} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" python generate_and_train.py \
    --base assets/hand_base.xml --task "$task_path" \
    --object-size "$object_size" --object-id "$object_id" \
    --run-label "$run_label" --candidate-id "$candidate_id" \
    --physics-mode rigid --Ntotal 500 --Rppx 1.25 --Rpt 5 \
    --out-root "$run_root" --artifact-root "$run_root" --force -- \
    --device cuda --num-envs 36 --seed 0 --n-timesteps 16000000 \
    --learning-starts 10000 --eval-freq 20000 --eval-episodes 10 \
    --disable-eval-video --wandb-project single_end_to_end_shadowhand \
    --target-position random --metrics-json "${run_root}/metrics.json" \
    --task-name "$object_id" --object-id "$object_id" \
    --candidate-id "$candidate_id" --physics-mode rigid \
    >"logs/${run_label}.out" 2>"logs/${run_label}.err"
}

pids=()
labels=()
for gpu in 0 1 2 3; do
  IFS='|' read -r object_size object_id task_path <<< "${CONFIGS[$gpu]}"
  run_one "$gpu" "$object_size" "$object_id" "$task_path" &
  pids+=("$!")
  labels+=("$object_id")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "Finished ${labels[$i]}"
  else
    echo "FAILED ${labels[$i]}" >&2
    status=1
  fi
done
exit "$status"
