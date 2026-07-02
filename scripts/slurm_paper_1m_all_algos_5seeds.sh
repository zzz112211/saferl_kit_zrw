#!/bin/bash
#SBATCH -J saferl_paper_1m
#SBATCH -p Physics
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 48:00:00
#SBATCH --array=0-29%6
#SBATCH -o slurm_logs/%x-%A_%a.out
#SBATCH -e slurm_logs/%x-%A_%a.err

set -euo pipefail

PROJECT=/datapool/home/1001900174/home/DYJ/saferl_kit
cd "$PROJECT"
mkdir -p slurm_logs logs models runs/tables

source "$PROJECT/.venv-saferlkit/bin/activate"

export SDL_VIDEODRIVER=dummy
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_TAG="${RUN_TAG:-paper_1m_20260629}"
ALGOS=(td3 epo lag qpsl recovery fac)
FLAGS=(--use_td3 --use_epo --use_lag --use_qpsl --use_recovery --use_fac)

TASK_ID="${SLURM_ARRAY_TASK_ID}"
ALGO_IDX=$((TASK_ID / 5))
SEED=$((TASK_ID % 5))
ALGO="${ALGOS[$ALGO_IDX]}"
FLAG="${FLAGS[$ALGO_IDX]}"
EXP_NAME="${RUN_TAG}_${ALGO}_seed${SEED}"

echo "Running full paper-style training: tag=${RUN_TAG} algo=${ALGO} seed=${SEED} task=${TASK_ID}"

python train_metadrive.py \
  "$FLAG" \
  --env SafeMetaDriveEnv \
  --save_model \
  --exp_name "$EXP_NAME" \
  --seed "$SEED" \
  --max_timesteps 1000000 \
  --start_timesteps 10000 \
  --eval_freq 10000 \
  --eval_episodes 20 \
  --batch_size 256
