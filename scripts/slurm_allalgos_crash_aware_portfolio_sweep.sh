#!/bin/bash
#SBATCH -J allalgos_crash_portfolio
#SBATCH -p Physics
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH --array=0-29%6
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

RUN_TAG="${RUN_TAG:-eval_allalgos_crash_aware_portfolio_20260702}"
EPISODES="${EPISODES:-20}"
START_SEED="${START_SEED:-100}"
CANDIDATES="${CANDIDATES:-nocicim_soft_crash_guard,nocicim_crash_aware_route,nocicim_conservative_clean,nocicim_route_preserve,nocicim_micro_guard,nocicim_cost_guard,nocicim_soft_front_trim,nocicim_close_range,nocicim_memory_confirmed_stop,nocicim_risk_gated_micro_guard}"

ALGOS=(td3 epo rec qpsl lag fac)
ALGO_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5))
ALGO="${ALGOS[$ALGO_INDEX]}"
CHECKPOINT_ALGO="$ALGO"
if [ "$ALGO" = "rec" ]; then
  CHECKPOINT_ALGO="recovery"
fi

echo "Running crash-aware NociCIM sweep: tag=${RUN_TAG} algo=${ALGO} train_seed=${SEED} episodes=${EPISODES}"

python sweep_memristive_frontend.py \
  --algo "$ALGO" \
  --checkpoint "models/paper_1m_20260629_${CHECKPOINT_ALGO}_seed${SEED}" \
  --episodes "$EPISODES" \
  --start-seed "$START_SEED" \
  --candidates "$CANDIDATES" \
  --outdir "logs/${RUN_TAG}_${ALGO}_seed${SEED}"
