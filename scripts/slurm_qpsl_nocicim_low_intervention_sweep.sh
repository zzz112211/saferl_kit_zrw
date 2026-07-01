#!/bin/bash
#SBATCH -J qpsl_nocicim_sweep
#SBATCH -p Physics
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH --array=0-4%2
#SBATCH -o slurm_logs/%x-%A_%a.out
#SBATCH -e slurm_logs/%x-%A_%a.err

set -euo pipefail

PROJECT=/datapool/home/1001900174/home/DYJ/saferl_kit
cd "$PROJECT"
mkdir -p slurm_logs logs runs/tables

source "$PROJECT/.venv-saferlkit/bin/activate"

export SDL_VIDEODRIVER=dummy
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

RUN_TAG="${RUN_TAG:-sweep_qpsl_nocicim_low_intervention_20260701}"
EPISODES="${EPISODES:-20}"
START_SEED="${START_SEED:-100}"
CANDIDATES="${CANDIDATES:-nocicim_panic_front_only,nocicim_soft_front_trim,nocicim_route_preserve,nocicim_micro_guard,nocicim_risk_gated_micro_guard,nocicim_memory_confirmed_stop}"

SEED="${SLURM_ARRAY_TASK_ID}"

echo "Running QPSL NociCIM eval sweep: tag=${RUN_TAG} train_seed=${SEED} episodes=${EPISODES}"

python sweep_memristive_frontend.py \
  --algo qpsl \
  --checkpoint "models/paper_1m_20260629_qpsl_seed${SEED}" \
  --episodes "$EPISODES" \
  --start-seed "$START_SEED" \
  --candidates "$CANDIDATES" \
  --outdir "logs/${RUN_TAG}_seed${SEED}"
