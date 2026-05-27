#!/bin/bash
# Bootstrap the trading_enhancer environment.
# Idempotent: safe to run multiple times.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PY="${PYTHON:-python3}"

if [ ! -d "venv" ]; then
  echo "[setup] Creating virtual environment in ./venv"
  "${PY}" -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip wheel
pip install -r requirements.txt

# Sanity check: confirm the source tree imports cleanly.
PYTHONPATH="${SCRIPT_DIR}" python -c "
import importlib
for mod in [
    'src.data_loader',
    'src.model.transformer_backbone',
    'src.model.ensemble',
    'src.risk.risk_manager',
    'src.signals.signal_generator',
    'src.backtest.backtest_runner',
    'src.utils.logging_setup',
]:
    importlib.import_module(mod)
print('Environment ready')
"
