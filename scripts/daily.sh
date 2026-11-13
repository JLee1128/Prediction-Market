#!/usr/bin/env bash
# Daily refresh: pull new settled events (already-downloaded ones are skipped), rebuild features
# (including any recorded order books), retrain, backtest, and write a fresh report under reports/.
#   ./scripts/daily.sh                 # series from config.yaml
#   ./scripts/daily.sh KXBTCD KXETHD   # explicit series
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
SERIES=("$@")
ARGS=()
if [ ${#SERIES[@]} -gt 0 ]; then ARGS=(--series "${SERIES[@]}"); fi
{
  echo "=== $(date)"
  .venv/bin/python -m predmkt download "${ARGS[@]+"${ARGS[@]}"}" --days 3
  .venv/bin/python -m predmkt run "${ARGS[@]+"${ARGS[@]}"}" --rebuild --ablation
} >> logs/daily.log 2>&1
tail -n 3 logs/daily.log
