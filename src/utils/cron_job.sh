#!/bin/bash
# Daily cron entry for the trading_enhancer live pipeline.
# Adjust PROJECT_DIR to your install location.
#
# Install (run from this directory):
#   chmod +x cron_job.sh
#   (crontab -l 2>/dev/null; echo "0 2 * * * /path/to/trading_enhancer/src/utils/cron_job.sh") | crontab -

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/trading_enhancer}"
cd "${PROJECT_DIR}"

# Activate the virtual environment created by setup.sh
# shellcheck disable=SC1091
source venv/bin/activate

# Optional: source secrets from .env if you store them there.
if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
fi

python run.py --mode live >> "${PROJECT_DIR}/cron.log" 2>&1
