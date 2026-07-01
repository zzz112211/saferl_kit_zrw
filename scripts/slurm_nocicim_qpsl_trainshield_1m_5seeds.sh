#!/bin/bash
#SBATCH -J saferl_nocicim_qpsl
#SBATCH -p Physics
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 48:00:00
#SBATCH --array=0-4%5
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

RUN_TAG="${RUN_TAG:-nocicim_qpsl_trainshield_1m_20260701}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-1000000}"
START_TIMESTEPS="${START_TIMESTEPS:-10000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PAIRED_EVAL_EPISODES="${PAIRED_EVAL_EPISODES:-20}"
PAIRED_EVAL_START_SEED="${PAIRED_EVAL_START_SEED:-100}"
NOCICIM_PROFILE="${NOCICIM_PROFILE:-default}"

SEED="${SLURM_ARRAY_TASK_ID}"
EXP_NAME="${RUN_TAG}_qpsl_seed${SEED}"

echo "Running NociCIM train-shield QPSL: tag=${RUN_TAG} seed=${SEED} steps=${MAX_TIMESTEPS} profile=${NOCICIM_PROFILE}"

python train_metadrive.py \
  --use_qpsl \
  --env SafeMetaDriveEnv \
  --save_model \
  --exp_name "$EXP_NAME" \
  --seed "$SEED" \
  --max_timesteps "$MAX_TIMESTEPS" \
  --start_timesteps "$START_TIMESTEPS" \
  --eval_freq "$EVAL_FREQ" \
  --eval_episodes "$EVAL_EPISODES" \
  --batch_size "$BATCH_SIZE" \
  --train_frontend nocicim_risk_field \
  --eval_frontend nocicim_risk_field \
  --nocicim_profile "$NOCICIM_PROFILE"

python eval_memristive_frontend.py \
  --algo qpsl \
  --checkpoint "models/${EXP_NAME}" \
  --episodes "$PAIRED_EVAL_EPISODES" \
  --start-seed "$PAIRED_EVAL_START_SEED" \
  --frontends none,nocicim_risk_field \
  --outdir "logs/eval_${EXP_NAME}_${PAIRED_EVAL_EPISODES}ep"
