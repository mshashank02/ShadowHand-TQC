#!/bin/bash
#SBATCH --account=bewv-dtai-gh
#SBATCH --job-name=shadow4
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

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

echo "Node: $(hostname)"
echo "Slurm job id: ${SLURM_JOB_ID:-unknown}"
echo "CUDA_VISIBLE_DEVICES at batch start: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

run_one () {
  local gpu="$1"
  local object_size="$2"
  local object_id="$3"
  local task_path="$4"

  local candidate_id="n0500_a0p5_b0p2"
  local run_label="${candidate_id}_${object_id}_rigid"
  local run_root="generated/studies/sphere_v1_demo/candidates/${candidate_id}/${object_id}/rigid"
  local metrics_json="${run_root}/metrics.json"

  mkdir -p "$run_root" logs

  echo "Starting ${run_label} on GPU ${gpu}"

  CUDA_VISIBLE_DEVICES="$gpu" \
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
    --force -- \
    --device cuda \
    --num-envs 48 \
    --seed 0 \
    --n-timesteps 16000000 \
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
    > "logs/${run_label}.out" 2> "logs/${run_label}.err"
}

run_one 0 large  obj_size_large_ar_high_macro_high_rough_high   study_objects/sphere_study_v1/obj_size-large_ar-high_macro-high_rough-high.msh &
pid0=$!

run_one 1 medium obj_size_medium_ar_high_macro_high_rough_high  study_objects/sphere_study_v1/obj_size-medium_ar-high_macro-high_rough-high.msh &
pid1=$!

run_one 2 small  obj_size_small_ar_high_macro_high_rough_high   study_objects/sphere_study_v1/obj_size-small_ar-high_macro-high_rough-high.msh &
pid2=$!

run_one 3 large  obj_size_large_ar_low_macro_high_rough_high    study_objects/sphere_study_v1/obj_size-large_ar-low_macro-high_rough-high.msh &
pid3=$!

wait "$pid0" "$pid1" "$pid2" "$pid3"

echo "All four jobs finished."
