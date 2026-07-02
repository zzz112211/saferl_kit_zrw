#!/usr/bin/env bash
set -u

cd /datapool/home/1001900174/home/DYJ/saferl_kit || exit 1
mkdir -p runs/risk_reflex_eval_smoke backups
LOG="runs/risk_reflex_eval_smoke/task3_validation.log"
: > "$LOG"

log_line() {
  printf '%s\n' "$*" | tee -a "$LOG"
}

run_cmd() {
  log_line ""
  log_line "===== $* ====="
  "$@" >> "$LOG" 2>&1
  status=$?
  log_line "EXIT_STATUS: $status"
  if [ "$status" -ne 0 ]; then
    log_line "FAILED_COMMAND: $*"
    exit "$status"
  fi
}

log_line "Task 3 validation started: $(date)"
log_line "PWD: $(pwd)"
log_line "Python: $(.venv-saferlkit/bin/python --version 2>&1)"
source .venv-saferlkit/bin/activate

run_cmd python -m py_compile eval_qpsl_risk_reflex.py
run_cmd python -m pytest tests/test_risk_reflex.py -q
run_cmd python eval_qpsl_risk_reflex.py --model_tag paper_1m_20260629_qpsl_seed0 --variant baseline --seed 0 --eval_episodes 2 --out_dir runs/risk_reflex_eval_smoke
run_cmd python eval_qpsl_risk_reflex.py --model_tag paper_1m_20260629_qpsl_seed0 --variant longitudinal --seed 0 --eval_episodes 2 --theta 0.45 --lambda_brake 0.25 --out_dir runs/risk_reflex_eval_smoke
run_cmd python eval_qpsl_risk_reflex.py --model_tag paper_1m_20260629_qpsl_seed0 --variant steering_limit --seed 0 --eval_episodes 2 --theta 0.45 --lambda_brake 0.25 --out_dir runs/risk_reflex_eval_smoke

log_line ""
log_line "===== SUMMARY CSV FILES ====="
ls -l runs/risk_reflex_eval_smoke/*_summary.csv | tee -a "$LOG"
for f in runs/risk_reflex_eval_smoke/*_summary.csv; do
  log_line "--- $f ---"
  cat "$f" | tee -a "$LOG"
done

log_line ""
log_line "===== RECREATE BACKUP ====="
tar -czf backups/task3-risk-reflex-evaluator.tgz eval_qpsl_risk_reflex.py runs/risk_reflex_eval_smoke >> "$LOG" 2>&1
status=$?
log_line "EXIT_STATUS: $status"
if [ "$status" -ne 0 ]; then
  exit "$status"
fi
ls -l backups/task3-risk-reflex-evaluator.tgz | tee -a "$LOG"

log_line ""
log_line "===== TAR LISTING ====="
tar -tzf backups/task3-risk-reflex-evaluator.tgz | tee -a "$LOG"

log_line ""
log_line "===== TAR EXCLUSION CHECK ====="
if tar -tzf backups/task3-risk-reflex-evaluator.tgz | grep -E '(^|/)risk_reflex/|(^|/)tests/' >> "$LOG" 2>&1; then
  log_line "TAR_EXCLUSION_CHECK: FAIL"
  exit 1
else
  log_line "TAR_EXCLUSION_CHECK: PASS"
fi

log_line "Task 3 validation finished: $(date)"
