#!/bin/bash
#SBATCH -J saferl_all_smoke
#SBATCH -p Physics
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --array=0-5%3
#SBATCH -o slurm_logs/%x-%A_%a.out
#SBATCH -e slurm_logs/%x-%A_%a.err

set -euo pipefail

PROJECT=/datapool/home/1001900174/home/DYJ/saferl_kit
cd "$PROJECT"
mkdir -p slurm_logs logs models

source "$PROJECT/.venv-saferlkit/bin/activate"

export SDL_VIDEODRIVER=dummy
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ALGOS=(td3 epo lag qpsl recovery fac)
FLAGS=(--use_td3 --use_epo --use_lag --use_qpsl --use_recovery --use_fac)

IDX="${SLURM_ARRAY_TASK_ID}"
ALGO="${ALGOS[$IDX]}"
FLAG="${FLAGS[$IDX]}"
SEED=0
EXP_NAME="smoke_all_${ALGO}_seed${SEED}"

echo "Running smoke: algo=${ALGO} seed=${SEED} task=${SLURM_ARRAY_TASK_ID}"

python train_metadrive.py \
  "$FLAG" \
  --env SafeMetaDriveEnv \
  --save_model \
  --exp_name "$EXP_NAME" \
  --seed "$SEED" \
  --max_timesteps 300 \
  --start_timesteps 50 \
  --eval_freq 300 \
  --eval_episodes 1 \
  --batch_size 64
