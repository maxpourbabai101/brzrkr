#!/bin/bash
#
# BRZRKR_LAUNCH.command — bulletproof launcher.
#
# This script will:
#   1. Find a working Python (tries 3.13, 3.12, 3.14, brew, system)
#   2. Build or repair the venv if it's missing/broken
#   3. Install any missing dependencies
#   4. Launch the BRZRKR desktop app
#
# Drop on the Dock: macOS treats .command files as launchable.
# Double-click to run. Closing the Terminal window quits the app.
#
# Rename for a cleaner Dock label:
#   mv BRZRKR_LAUNCH.command BRZRKR.command
# (the .command extension is required for double-click to work)
#

set -u   # error on undefined vars (but NOT -e — we want to handle errors)

# Always run from the script's directory so relative paths resolve.
cd "$(dirname "$0")" || { echo "FATAL: cannot cd to script dir"; exit 1; }

# ANSI colors for nicer output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
DIM='\033[0;90m'
NC='\033[0m'

banner() {
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                  ✠   B R Z R K R   ✠                    ║${NC}"
    echo -e "${RED}║                   the forge of trade                     ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

step() {
    echo -e "${BLUE}[${1}]${NC} ${2}"
}

ok() {
    echo -e "  ${GREEN}✓${NC} ${1}"
}

warn() {
    echo -e "  ${YELLOW}⚠${NC} ${1}"
}

fail() {
    echo -e "  ${RED}✗${NC} ${1}"
}

pause_on_fail() {
    echo ""
    echo -e "${YELLOW}Press any key to close this window.${NC}"
    read -n 1 -s
    exit 1
}

banner

# Source .env so API keys are available to the app.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    ok ".env loaded"
fi

# ─────────────────────────────────────────────────────────────────────
# Step 1 — find a working Python
# ─────────────────────────────────────────────────────────────────────
step "1/4" "Finding a working Python interpreter..."

# Order matters: prefer stable versions over bleeding-edge.
PYTHON_CANDIDATES=(
    "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13"
    "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
    "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11"
    "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14"
    "/opt/homebrew/bin/python3.13"
    "/opt/homebrew/bin/python3.12"
    "/opt/homebrew/bin/python3.11"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3"
    "/usr/bin/python3"
)

WORKING_PYTHON=""
for py in "${PYTHON_CANDIDATES[@]}"; do
    if [ -x "$py" ] && "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
        WORKING_PYTHON="$py"
        VERSION=$("$py" --version 2>&1)
        ok "$py — $VERSION"
        break
    fi
done

if [ -z "$WORKING_PYTHON" ]; then
    fail "No working Python ≥ 3.10 found on this system."
    echo ""
    echo "  Install one from https://www.python.org/downloads/macos/"
    echo "  or with Homebrew:  brew install python@3.13"
    pause_on_fail
fi

# ─────────────────────────────────────────────────────────────────────
# Step 2 — verify or build the venv
# ─────────────────────────────────────────────────────────────────────
step "2/4" "Verifying virtualenv..."

VENV_PYTHON="./venv/bin/python"
VENV_OK=0
if [ -x "$VENV_PYTHON" ]; then
    if "$VENV_PYTHON" -c "import sys" 2>/dev/null; then
        VENV_VERSION=$("$VENV_PYTHON" --version 2>&1)
        ok "venv works — $VENV_VERSION"
        VENV_OK=1
    else
        warn "venv/bin/python exists but cannot launch (corrupted)."
    fi
else
    warn "venv missing."
fi

if [ "$VENV_OK" -eq 0 ]; then
    echo ""
    warn "Rebuilding venv with $WORKING_PYTHON ..."
    rm -rf venv
    if "$WORKING_PYTHON" -m venv venv; then
        ok "Fresh venv created."
    else
        fail "venv creation failed."
        pause_on_fail
    fi
fi

# ─────────────────────────────────────────────────────────────────────
# Step 3 — verify dependencies
# ─────────────────────────────────────────────────────────────────────
step "3/4" "Checking dependencies..."

# Probe a representative set of imports.
REQUIRED_IMPORTS=("customtkinter" "PIL" "pandas" "numpy"
                  "yfinance" "alpaca" "requests" "psutil")

MISSING_IMPORTS=""
for mod in "${REQUIRED_IMPORTS[@]}"; do
    if ! "$VENV_PYTHON" -c "import $mod" 2>/dev/null; then
        MISSING_IMPORTS="$MISSING_IMPORTS $mod"
    fi
done

if [ -n "$MISSING_IMPORTS" ]; then
    warn "Missing modules:$MISSING_IMPORTS"
    echo ""
    echo "  Installing dependencies (takes 3-5 minutes the first time)..."
    echo ""
    "$VENV_PYTHON" -m pip install --upgrade pip 2>&1 | tail -3
    if "$VENV_PYTHON" -m pip install -r requirements.txt; then
        ok "Dependencies installed."
    else
        fail "pip install failed."
        pause_on_fail
    fi
else
    ok "All required modules present."
fi

# Final clear pycache (defensive — handles stale .pyc after edits).
find . -path ./venv -prune -o -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
ok "Caches cleared."

# ─────────────────────────────────────────────────────────────────────
# Step 4 — launch
# ─────────────────────────────────────────────────────────────────────
step "4/4" "Launching BRZRKR..."
echo ""
echo -e "${DIM}  (close this window to quit the app)${NC}"
echo ""

# exec replaces this shell with Python — saves a fork.
exec "$VENV_PYTHON" BRZRKR.py
