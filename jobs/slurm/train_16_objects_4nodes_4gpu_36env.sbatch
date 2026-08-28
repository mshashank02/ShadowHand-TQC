#!/bin/bash
#SBATCH --account=bewv-dtai-gh
#SBATCH --job-name=shadow16
#SBATCH --partition=ghx4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=30:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"

module load python/miniforge3_pytorch/2.10.0
conda activate base

REPO_ROOT=/work/hdd/bewv/smahesh1/ShadowHand-TQC
cd "$REPO_ROOT"

mkdir -p logs generated slurm_logs

export PYTHONPATH=/u/smahesh1/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SDL_VIDEODRIVER=dummy
export WANDB_CONSOLE=off
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONFIGS=(
  "medium|obj_size_medium_ar_low_macro_high_rough_high|study_objects/sphere_study_v1/obj_size-medium_ar-low_macro-high_rough-high.msh"
  "small|obj_size_small_ar_low_macro_high_rough_high|study_objects/sphere_study_v1/obj_size-small_ar-low_macro-high_rough-high.msh"
  "large|obj_size_large_ar_high_macro_high_rough_low|study_objects/sphere_study_v1/obj_size-large_ar-high_macro-high_rough-low.msh"
  "medium|obj_size_medium_ar_high_macro_high_rough_low|study_objects/sphere_study_v1/obj_size-medium_ar-high_macro-high_rough-low.msh"
  "small|obj_size_small_ar_high_macro_high_rough_low|study_objects/sphere_study_v1/obj_size-small_ar-high_macro-high_rough-low.msh"
  "large|obj_size_large_ar_low_macro_high_rough_low|study_objects/sphere_study_v1/obj_size-large_ar-low_macro-high_rough-low.msh"
  "medium|obj_size_medium_ar_low_macro_high_rough_low|study_objects/sphere_study_v1/obj_size-medium_ar-low_macro-high_rough-low.msh"
  "small|obj_size_small_ar_low_macro_high_rough_low|study_objects/sphere_study_v1/obj_size-small_ar-low_macro-high_rough-low.msh"
  "large|obj_size_large_ar_high_macro_low_rough_high|study_objects/sphere_study_v1/obj_size-large_ar-high_macro-low_rough-high.msh"
  "medium|obj_size_medium_ar_high_macro_low_rough_high|study_objects/sphere_study_v1/obj_size-medium_ar-high_macro-low_rough-high.msh"
  "small|obj_size_small_ar_high_macro_low_rough_high|study_objects/sphere_study_v1/obj_size-small_ar-high_macro-low_rough-high.msh"
  "large|obj_size_large_ar_low_macro_low_rough_high|study_objects/sphere_study_v1/obj_size-large_ar-low_macro-low_rough-high.msh"
  "medium|obj_size_medium_ar_low_macro_low_rough_high|study_objects/sphere_study_v1/obj_size-medium_ar-low_macro-low_rough-high.msh"
  "small|obj_size_small_ar_low_macro_low_rough_high|study_objects/sphere_study_v1/obj_size-small_ar-low_macro-low_rough-high.msh"
  "large|obj_size_large_ar_high_macro_low_rough_low|study_objects/sphere_study_v1/obj_size-large_ar-high_macro-low_rough-low.msh"
  "medium|obj_size_medium_ar_high_macro_low_rough_low|study_objects/sphere_study_v1/obj_size-medium_ar-high_macro-low_rough-low.msh"
)

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

  echo "[$(hostname)] Starting ${run_label} on GPU ${gpu}"

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
    --num-envs 36 \
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

worker () {
  local node_index="$1"
  local start=$((node_index * 4))
  local pids=()

  echo "Node index: ${node_index}"
  echo "Node host: $(hostname)"
  echo "Slurm job id: ${SLURM_JOB_ID:-unknown}"
  echo "CUDA_VISIBLE_DEVICES at worker start: ${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi || true

  for local_index in 0 1 2 3; do
    local config_index=$((start + local_index))
    local entry="${CONFIGS[$config_index]}"
    IFS='|' read -r object_size object_id task_path <<< "$entry"

    run_one "$local_index" "$object_size" "$object_id" "$task_path" &
    pids+=("$!")
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "Training process $pid failed on node index ${node_index}"
      status=1
    fi
  done

  echo "Node index ${node_index} finished all four jobs with status ${status}."
  exit "$status"
}

if [[ "${1:-}" == "--worker" ]]; then
  worker "$2"
  exit 0
fi

echo "Allocated nodes: ${SLURM_JOB_NODELIST:-unknown}"
echo "Slurm job id: ${SLURM_JOB_ID:-unknown}"

for node_index in 0 1 2 3; do
  srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=256 \
    --gpus-per-node=4 \
    --exclusive \
    "$SCRIPT_PATH" --worker "$node_index" &
done

wait

echo "All 16 jobs finished."
