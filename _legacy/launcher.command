#!/bin/bash
# Drop this file on the macOS Dock for a one-click launch.
# Opens a tiny Terminal window, activates the venv, and starts the
# desktop app. Close the Terminal window to quit the app.

set -e
cd "$(dirname "$0")"

if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
fi

if [ ! -d venv ]; then
  echo "venv missing — run ./setup.sh first."
  read -n 1 -s
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate
exec python desktop_app.py
