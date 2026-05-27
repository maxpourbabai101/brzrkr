#!/bin/bash
# Build a proper macOS .app bundle for trading_enhancer.
# Produces dist/trading_enhancer.app which is dock-droppable.
#
# Requirements:
#   - venv already created via ./setup.sh
#   - pyinstaller installed (in requirements.txt)
#
# Notes:
#   - First build takes 2-5 min; subsequent builds are much faster.
#   - The bundle is ~300-500 MB because it ships its own Python + torch.
#     This is normal for PyInstaller and unavoidable for self-contained apps.
#   - The bundle reads .env / config/ / models/ from wherever it's launched
#     from, NOT from inside the bundle. Keep this repo around.

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  echo "venv missing — run ./setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "Installing pyinstaller..."
  pip install -q pyinstaller
fi

echo "Cleaning previous build artifacts..."
rm -rf build dist trading_enhancer.spec

echo "Building .app bundle (this can take a few minutes)..."
pyinstaller \
  --name trading_enhancer \
  --windowed \
  --noconfirm \
  --osx-bundle-identifier com.trading_enhancer.desktop \
  --hidden-import customtkinter \
  --hidden-import PIL \
  --hidden-import alpaca \
  --collect-all customtkinter \
  desktop_app.py

if [ -d "dist/trading_enhancer.app" ]; then
  echo ""
  echo "✓  Built: dist/trading_enhancer.app"
  echo ""
  echo "Try it:"
  echo "    open dist/trading_enhancer.app"
  echo ""
  echo "To install: drag dist/trading_enhancer.app into /Applications,"
  echo "then drag from /Applications onto your Dock."
else
  echo "✗  Build failed — check the pyinstaller output above."
  exit 1
fi
