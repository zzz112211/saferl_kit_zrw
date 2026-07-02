#!/bin/bash
#SBATCH -J saferl_td3_50k
#SBATCH -p Physics
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 24:00:00
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

PROJECT=/datapool/home/1001900174/home/DYJ/saferl_kit
cd "$PROJECT"
mkdir -p slurm_logs logs models

source "$PROJECT/.venv-saferlkit/bin/activate"

export SDL_VIDEODRIVER=dummy
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python train_metadrive.py \
  --use_td3 \
  --env SafeMetaDriveEnv \
  --save_model \
  --exp_name td3_baseline_50k_seed0 \
  --seed 0 \
  --max_timesteps 50000 \
  --start_timesteps 5000 \
  --eval_freq 5000 \
  --eval_episodes 5 \
  --batch_size 256
