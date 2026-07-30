#!/bin/bash
#SBATCH --account=bewv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=shadowhand-rigid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
conda activate base

cd /work/hdd/bewv/smahesh1/ShadowHand-TQC

export PYTHONPATH=/u/smahesh1/.local/lib/python3.12/site-packages:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export SDL_VIDEODRIVER=dummy
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

python generate_and_train.py \
  --base assets/hand_base.xml \
  --task study_objects/sphere_study_v1/obj_size-small_ar-high_macro-low_rough-high.msh \
  --object-size small \
  --object-id obj_size_small_ar_high_macro_low_rough_high \
  --run-label n0500_a0p3_b0p6_obj_size_small_ar_high_macro_low_rough_high_rigid \
  --candidate-id n0500_a0p3_b0p6 \
  --physics-mode rigid \
  --Ntotal 500 \
  --Rppx 1.071428571 \
  --Rpt 0.7142857143 \
  --out-root generated/studies/sphere_v1_demo/candidates/n0500_a0p3_b0p6/obj_size_small_ar_high_macro_low_rough_high/rigid \
  --artifact-root generated/studies/sphere_v1_demo/candidates/n0500_a0p3_b0p6/obj_size_small_ar_high_macro_low_rough_high/rigid \
  --force -- \
  --device cuda \
  --num-envs 16 \
  --seed 0 \
  --n-timesteps 16000000 \
  --eval-freq 20000 \
  --eval-episodes 50 \
  --max-episode-steps 200 \
  --metrics-json generated/studies/sphere_v1_demo/candidates/n0500_a0p3_b0p6/obj_size_small_ar_high_macro_low_rough_high/rigid/metrics.json \
  --task-name obj_size_small_ar_high_macro_low_rough_high \
  --object-id obj_size_small_ar_high_macro_low_rough_high \
  --candidate-id n0500_a0p3_b0p6 \
  --physics-mode rigid
