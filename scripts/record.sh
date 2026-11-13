#!/usr/bin/env bash
# Run the live order-book recorder forever, restarting it if it crashes.
# `caffeinate -i` stops the Mac from idle-sleeping while this runs (closing the lid still sleeps it).
#   ./scripts/record.sh            # poll mode (no API key)
#   ./scripts/record.sh ws         # websocket mode (needs KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_PATH)
set -u
cd "$(dirname "$0")/.."
MODE="${1:-poll}"
mkdir -p logs
while true; do
  echo "$(date) starting recorder ($MODE)" >> logs/record.log
  caffeinate -i .venv/bin/python -m predmkt record --mode "$MODE" >> logs/record.log 2>&1
  echo "$(date) recorder exited with $?; restarting in 10s" >> logs/record.log
  sleep 10
done
