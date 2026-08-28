+#!/bin/bash
#SBATCH --account=bewv-dtai-gh
#SBATCH --job-name=retry-failed6
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --array=0-5%4
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

# candidate_id|Ntotal|Rppx|Rpt|object_size|object_id|mesh_path
CONFIGS=(
  "n0500_a0p3_b0p6|500|1.071428571|0.7142857143|large|obj_size_large_ar_low_macro_low_rough_low|study_objects/sphere_study_v1/obj_size-large_ar-low_macro-low_rough-low.msh"
  "n0500_a0p3_b0p6|500|1.071428571|0.7142857143|medium|obj_size_medium_ar_low_macro_low_rough_low|study_objects/sphere_study_v1/obj_size-medium_ar-low_macro-low_rough-low.msh"
  "n0500_a0p3_b0p6|500|1.071428571|0.7142857143|small|obj_size_small_ar_low_macro_low_rough_low|study_objects/sphere_study_v1/obj_size-small_ar-low_macro-low_rough-low.msh"
  "n0500_a0p5_b0p2|500|1.25|5|small|obj_size_small_ar_low_macro_high_rough_high|study_objects/sphere_study_v1/obj_size-small_ar-low_macro-high_rough-high.msh"
  "n0500_a0p5_b0p2|500|1.25|5|medium|obj_size_medium_ar_low_macro_high_rough_low|study_objects/sphere_study_v1/obj_size-medium_ar-low_macro-high_rough-low.msh"
  "n0500_a0p5_b0p2|500|1.25|5|small|obj_size_small_ar_low_macro_low_rough_low|study_objects/sphere_study_v1/obj_size-small_ar-low_macro-low_rough-low.msh"
)

array_index=${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array job}
if (( array_index < 0 || array_index >= ${#CONFIGS[@]} )); then
  echo "Invalid array index ${array_index}; expected 0-$(( ${#CONFIGS[@]} - 1 ))" >&2
  exit 2
fi

IFS='|' read -r \
  candidate_id ntotal rppx rpt object_size object_id task_path \
  <<< "${CONFIGS[$array_index]}"

run_label="${candidate_id}_${object_id}_rigid"
run_root="generated/studies/sphere_v1_demo/candidates/${candidate_id}/${object_id}/rigid"
metrics_json="${run_root}/metrics.json"

if [[ ! -f "$task_path" ]]; then
  echo "Object mesh not found: ${task_path}" >&2
  exit 2
fi

mkdir -p "$run_root"

echo "[$(hostname)] Starting ${run_label} from scratch"
echo "Slurm array job: ${SLURM_ARRAY_JOB_ID:-unknown}_${array_index}"
echo "Allocated GPU(s): ${CUDA_VISIBLE_DEVICES:-unset}"

python generate_and_train.py \
  --base assets/hand_base.xml \
  --task "$task_path" \
  --object-size "$object_size" \
  --object-id "$object_id" \
  --run-label "$run_label" \
  --candidate-id "$candidate_id" \
  --physics-mode rigid \
  --Ntotal "$ntotal" \
  --Rppx "$rppx" \
  --Rpt "$rpt" \
  --out-root "$run_root" \
  --artifact-root "$run_root" --force -- \
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
  >"logs/${run_label}.out" 2>"logs/${run_label}.err"

echo "Finished ${run_label}"
