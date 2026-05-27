#!/bin/bash
# BRZRKR Dock launcher.
#
# Drag this file onto the macOS Dock. Clicking it opens a small
# Terminal window, activates the venv, generates the icon if missing,
# and launches the BRZRKR desktop app. Closing the Terminal window
# closes the app.
#
# This launcher always runs the LIVE source — any update to BRZRKR.py
# or brzrkr_app/* is reflected next time you click. No rebuild needed.

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

# Call the venv's Python directly. This bypasses 'source venv/bin/activate',
# which breaks when the project directory is renamed (venv stores absolute
# paths). The python symlink inside venv/bin still resolves correctly.
exec ./venv/bin/python BRZRKR.py
