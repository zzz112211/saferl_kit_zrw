#!/usr/bin/env bash
#SBATCH --job-name=qpsl_reflex_screen
#SBATCH --partition=Physics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-364%20
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR=/datapool/home/1001900174/home/DYJ/saferl_kit
VENV_DIR=/datapool/home/1001900174/home/DYJ/saferl_kit/.venv-saferlkit

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

export SDL_VIDEODRIVER=dummy
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

SCREEN_TAG="${SCREEN_TAG:-risk_reflex_screen_20260701}"
OUT_DIR="runs/${SCREEN_TAG}"
mkdir -p "$OUT_DIR" slurm_logs runs/tables

python scripts/sweep_qpsl_risk_reflex.py \
  --task_index "$SLURM_ARRAY_TASK_ID" \
  --task_count "$SLURM_ARRAY_TASK_COUNT" \
  --seeds "${SEEDS:-0,1,2,3,4}" \
  --eval_episodes "${EVAL_EPISODES:-20}" \
  --out_dir "$OUT_DIR" \
  --thetas "${THETAS:-0.40,0.45,0.50}" \
  --lambda_brakes "${LAMBDA_BRAKES:-0.20,0.30,0.40}" \
  --lambda_throttles "${LAMBDA_THROTTLES:-0.35}" \
  --alphas "${ALPHAS:-0.88,0.92}" \
  --betas "${BETAS:-0.35,0.45}" \
  --gammas "${GAMMAS:-0.06,0.08}" \
  --ks "${KS:-12.0}" \
  --overwrite

# After all array tasks complete, aggregate with:
# python scripts/aggregate_risk_reflex_results.py --input_dir "$OUT_DIR" --out "runs/tables/${SCREEN_TAG}_summary.csv" --expected_seeds 5