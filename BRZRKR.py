"""BRZRKR — main launcher.

Single entry point for the gothic dark trading dashboard. Imports
live from ``brzrkr_app/`` and ``src/`` so any update to the codebase
is reflected on the next launch — no rebuild required.

Run directly:
    python BRZRKR.py

Or use the Dock launcher:
    open BRZRKR.command

Or build a proper .app bundle:
    ./build_brzrkr.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    # Load .env FIRST — must happen before any broker/API code runs.
    # This ensures ALPACA_API_KEY / ALPACA_SECRET_KEY are available
    # whether the app is launched from a terminal, Dock, or .command file.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass  # dotenv not installed; keys must already be in the environment

    # Generate icon if missing (one-time, ~0.5s).
    try:
        from brzrkr_app.icon import generate
        generate()
    except Exception:
        pass

    # Start background market data collector (runs every 20 min, daemon).
    try:
        from src.data.market_collector import BackgroundCollector
        BackgroundCollector().start()
    except Exception:
        pass

    from brzrkr_app.main_window import MainWindow
    app = MainWindow()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
