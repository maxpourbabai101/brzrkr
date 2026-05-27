"""Auto-trader — waits for the market to open, runs the agent through
the session, sleeps until the next open.

Long-running daemon. Sleeps until ``clock.next_open`` (queried from
Alpaca), then launches ``agent.py`` as a subprocess for the duration
of regular trading hours. At close, sends the agent a stop signal
gracefully and goes back to sleep.

Run detached:
    nohup python auto_trader.py > auto_trader.out 2>&1 &
    echo $! > .auto_trader.pid

Stop:
    touch AUTO_TRADER_STOP        # graceful
    kill -INT $(cat .auto_trader.pid)   # equivalent

Quick test (paper mode, dry-run):
    python auto_trader.py --dry-run

Real paper trading:
    python auto_trader.py --execute

Real money (requires both --live-money AND ALPACA_LIVE=true AND a
passing promotion-gate track record):
    export ALPACA_LIVE=true
    python auto_trader.py --execute --live-money
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.brokers import get_broker
from src.utils.logging_setup import configure_logging

logger = logging.getLogger("trading_enhancer.auto_trader")

STOP_FILE = ROOT / "AUTO_TRADER_STOP"
AGENT_STOP = ROOT / "AGENT_STOP"
AGENT_PID = ROOT / "agent.pid"


def _interruptible_sleep(seconds: float, *, chunk: float = 5.0) -> bool:
    """Sleep for up to `seconds`, but break early if the stop file
    appears. Returns True if interrupted."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if STOP_FILE.exists():
            return True
        remaining = end - time.monotonic()
        time.sleep(min(chunk, max(0.1, remaining)))
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Market-open auto-trader")
    parser.add_argument("--source", choices=["api", "scraper", "both"],
                        default="api")
    parser.add_argument("--tick-seconds", type=int, default=300)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.03)
    parser.add_argument("--broker", choices=["alpaca", "ibkr", "paper_only"],
                        default="alpaca")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--execute", action="store_true")
    grp.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live-money", action="store_true")
    parser.add_argument("--max-sleep-seconds", type=int, default=600,
                        help="Cap on a single sleep (default 10 min) so the "
                             "daemon stays responsive to STOP signals.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(level=args.log_level, log_path="trading_enhancer.log")

    if args.live_money and not args.execute:
        parser.error("--live-money requires --execute")

    # Track-record reachable? Catch this up-front, not at first market open.
    try:
        executor = get_broker(args.broker, live_money=args.live_money)
        equity = executor.get_account_equity()
        endpoint = "paper" if getattr(executor, "_paper", True) else "LIVE"
        logger.info("Auto-trader broker connected — %s endpoint, equity $%.2f",
                     endpoint, equity)
    except Exception as exc:
        logger.error("Cannot reach broker; auto-trader halted: %s", exc)
        return 2

    # SIGINT/SIGTERM → write the stop file so the loop exits cleanly.
    def _stop_handler(_sig, _frame):
        logger.info("Received signal — flagging shutdown.")
        STOP_FILE.touch()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try: signal.signal(sig, _stop_handler)
        except Exception: pass

    print(f"\n  ✠  Auto-trader online.  ({args.broker} / "
          f"{'execute' if args.execute else 'dry-run'})")
    print(f"  Stop with:   touch {STOP_FILE.name}")
    print(f"  Or:          kill -INT {os.getpid()}\n")

    sessions = 0
    while not STOP_FILE.exists():
        try:
            clock = executor._client.get_clock()
        except Exception as exc:
            logger.warning("Clock fetch failed (%s); retrying in 30s.", exc)
            if _interruptible_sleep(30):
                break
            continue

        if not getattr(clock, "is_open", False):
            next_open = getattr(clock, "next_open", None)
            now = getattr(clock, "timestamp", None) or datetime.now(timezone.utc)
            try:
                wait = (next_open - now).total_seconds() if next_open else 600
            except Exception:
                wait = 600
            wait = max(60.0, min(wait, args.max_sleep_seconds))
            logger.info("Market closed. Next open %s. Sleeping %d s.",
                         next_open, wait)
            if _interruptible_sleep(wait):
                break
            continue

        # Market is open → spawn the agent for this session.
        sessions += 1
        next_close = getattr(clock, "next_close", None)
        logger.info("Market is OPEN. Launching agent for session %d "
                     "(closes at %s).", sessions, next_close)
        agent_started = _spawn_agent(args)
        if agent_started is None:
            logger.error("Failed to spawn agent. Sleeping 60s.")
            if _interruptible_sleep(60):
                break
            continue

        # Watch the clock; stop the agent at close.
        while not STOP_FILE.exists():
            time.sleep(15)
            try:
                clock = executor._client.get_clock()
                if not clock.is_open:
                    logger.info("Market closed. Sending stop signal to agent.")
                    AGENT_STOP.touch()
                    break
            except Exception as exc:
                logger.debug("Clock check during session failed: %s", exc)
            # Also check if the agent died unexpectedly.
            if agent_started.poll() is not None:
                logger.warning("Agent process exited unexpectedly "
                                "(code %s). Will respawn at next open.",
                                agent_started.returncode)
                break

        # Wait for the agent to actually exit (give it ~30s).
        try:
            agent_started.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("Agent didn't exit in 30s — terminating.")
            agent_started.terminate()
            try: agent_started.wait(timeout=10)
            except Exception: pass

        # Clean up the stop sentinel.
        if AGENT_STOP.exists():
            try: AGENT_STOP.unlink()
            except Exception: pass
        if AGENT_PID.exists():
            try: AGENT_PID.unlink()
            except Exception: pass

        logger.info("Session %d ended.", sessions)

    if STOP_FILE.exists():
        try: STOP_FILE.unlink()
        except Exception: pass
    logger.info("Auto-trader shutting down. Sessions completed: %d.", sessions)
    return 0


def _spawn_agent(args) -> subprocess.Popen | None:
    cmd = [sys.executable, str(ROOT / "agent.py"),
           "--execute" if args.execute else "--dry-run",
           "--source", args.source,
           "--tick-seconds", str(args.tick_seconds),
           "--max-positions", str(args.max_positions),
           "--max-daily-loss-pct", str(args.max_daily_loss_pct),
           "--broker", args.broker]
    if args.live_money:
        cmd.append("--live-money")

    log = open(ROOT / "agent.out", "ab")
    try:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=str(ROOT),
        )
        AGENT_PID.write_text(str(proc.pid))
        logger.info("Spawned agent PID %d", proc.pid)
        return proc
    except Exception as exc:
        logger.error("Spawn failed: %s", exc)
        return None


if __name__ == "__main__":
    sys.exit(main())
