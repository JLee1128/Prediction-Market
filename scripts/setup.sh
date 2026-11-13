#!/usr/bin/env bash
# One-time setup on macOS (Apple silicon): virtualenv, Python packages, and the OpenMP runtime XGBoost needs.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! [ -e "$(brew --prefix)/opt/libomp/lib/libomp.dylib" ]; then
  brew install libomp
fi
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import xgboost; print('xgboost', xgboost.__version__, 'OK')"
.venv/bin/python -m pytest -q
