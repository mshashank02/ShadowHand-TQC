#!/bin/bash
#SBATCH --account=bewv-dtai-gh
#SBATCH --job-name=n0500-small-rigid
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=30:00:00
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err

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

candidate_id=n0500_a0p5_b0p2
ntotal=500
rppx=1.25
rpt=5
object_size=small
object_id=obj_size_small_ar_low_macro_high_rough_high
task_path=study_objects/sphere_study_v1/obj_size-small_ar-low_macro-high_rough-high.msh

run_label="${candidate_id}_${object_id}_rigid"
run_root="generated/studies/sphere_v1_demo/candidates/${candidate_id}/${object_id}/rigid"
metrics_json="${run_root}/metrics.json"

if [[ ! -f "$task_path" ]]; then
  echo "Object mesh not found: ${task_path}" >&2
  exit 2
fi

mkdir -p "$run_root"

echo "[$(hostname)] Starting ${run_label} from scratch"
echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
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
