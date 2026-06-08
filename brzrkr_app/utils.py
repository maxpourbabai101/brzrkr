"""Shared utilities for BRZRKR app pages."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

AGENT_PID  = ROOT / "agent.pid"
AGENT_STOP = ROOT / "AGENT_STOP"
AUTO_PID   = ROOT / ".auto_trader.pid"
AUTO_STOP  = ROOT / "AUTO_TRADER_STOP"


def agent_running() -> bool:
    """Return True if the trading agent process is alive."""
    if not AGENT_PID.exists():
        return False
    try:
        pid = int(AGENT_PID.read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def auto_trader_running() -> tuple[bool, int]:
    """Return (alive, pid) for the auto-trader process."""
    if not AUTO_PID.exists():
        return (False, 0)
    try:
        pid = int(AUTO_PID.read_text().strip())
        os.kill(pid, 0)
        return (True, pid)
    except Exception:
        return (False, 0)
