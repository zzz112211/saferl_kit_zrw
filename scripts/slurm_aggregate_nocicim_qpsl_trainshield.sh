#!/bin/bash
#SBATCH -J nocicim_qpsl_agg
#SBATCH -p Physics
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

PROJECT=/datapool/home/1001900174/home/DYJ/saferl_kit
cd "$PROJECT"
mkdir -p slurm_logs runs/tables

source "$PROJECT/.venv-saferlkit/bin/activate"

RUN_TAG="${RUN_TAG:-nocicim_qpsl_trainshield_1m_20260701}"
SEEDS="${SEEDS:-5}"
PAIRED_EVAL_EPISODES="${PAIRED_EVAL_EPISODES:-20}"

python scripts/aggregate_nocicim_trainshield_results.py \
  --tag "$RUN_TAG" \
  --seeds "$SEEDS" \
  --paired-eval-episodes "$PAIRED_EVAL_EPISODES" \
  --require-complete
