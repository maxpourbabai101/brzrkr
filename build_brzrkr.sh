#!/bin/bash
# Build a macOS .app bundle for BRZRKR.
#
# Output: dist/BRZRKR.app
#   - Drop in /Applications, drag to Dock.
#   - Bundles its own Python + dependencies (~300-500 MB; normal for PyInstaller).
#   - Uses assets/BRZRKR_icon.icns if it exists.
#
# The bundle is a thin launcher: it imports BRZRKR.py and brzrkr_app/
# from the original project directory at runtime, so updating the
# Python source instantly reflects in the next launch. The .app does
# NOT need to be rebuilt for code updates — only rebuild if you change
# the bundled Python version or add new top-level dependencies.

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

# Make sure the icon exists.
python -c "from brzrkr_app.icon import generate, to_icns; generate(); to_icns()"

ICON_ARG=""
if [ -f "assets/BRZRKR_icon.icns" ]; then
  ICON_ARG="--icon=assets/BRZRKR_icon.icns"
fi

echo "Cleaning previous build artifacts..."
rm -rf build dist BRZRKR.spec

echo "Building BRZRKR.app (this can take 2-5 minutes)..."
# shellcheck disable=SC2086
pyinstaller \
  --name BRZRKR \
  --windowed \
  --noconfirm \
  --osx-bundle-identifier com.brzrkr.desktop \
  $ICON_ARG \
  --hidden-import customtkinter \
  --hidden-import PIL \
  --hidden-import alpaca \
  --collect-all customtkinter \
  --collect-all brzrkr_app \
  BRZRKR.py

if [ -d "dist/BRZRKR.app" ]; then
  # Copy the icon into the bundle resources so macOS picks it up.
  if [ -f "assets/BRZRKR_icon.icns" ]; then
    cp assets/BRZRKR_icon.icns "dist/BRZRKR.app/Contents/Resources/icon-windowed.icns" 2>/dev/null || true
  fi
  echo ""
  echo "✓  Built: dist/BRZRKR.app"
  echo ""
  echo "Try it:    open dist/BRZRKR.app"
  echo "Install:   mv dist/BRZRKR.app /Applications/"
  echo "Dock:      drag BRZRKR from /Applications onto the Dock."
else
  echo "✗  Build failed — check the pyinstaller output above."
  exit 1
fi
