"""Manual trade CLI — declare and place orders directly.

Use this when you want to send a specific trade without waiting for
the autonomous pipeline to clear its confidence threshold. Goes
through the same :class:`AlpacaExecutor` the live pipeline uses, so
paper-vs-live safety guards still apply.

Examples
--------
List current paper account state:
    python trade.py status

Place a long bracket order on SPY (1% stop, 2% take-profit):
    python trade.py buy --symbol SPY --notional 1000

Place a short bracket on AAPL with custom risk parameters:
    python trade.py sell --symbol AAPL --notional 2500 --stop-pct 0.02 --tp-pct 0.05

Cancel every open order:
    python trade.py cancel-all

Flatten a single position:
    python trade.py close --symbol SPY

Real money (requires both --live-money AND ALPACA_LIVE=true):
    export ALPACA_LIVE=true
    python trade.py buy --symbol SPY --notional 500 --live-money
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.broker import AlpacaExecutor
from src.utils.logging_setup import configure_logging

logger = logging.getLogger("trading_enhancer.trade")


# ---------------------------------------------------------------------------
# Quote lookup — used to compute entry / stop / take-profit
# ---------------------------------------------------------------------------
def _get_latest_quote(symbol: str) -> float:
    """Latest trade price from Alpaca's market data endpoint."""
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise EnvironmentError("ALPACA_API_KEY / ALPACA_SECRET_KEY must be set.")

    # Paper and live share the same market data endpoint.
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
    resp = requests.get(
        url,
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=10,
    )
    resp.raise_for_status()
    trade = (resp.json() or {}).get("trade") or {}
    price = trade.get("p")
    if price is None:
        raise RuntimeError(f"No latest price returned for {symbol}")
    return float(price)


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def _build_signal(
    *,
    symbol: str,
    direction: str,
    notional: float,
    stop_pct: float,
    tp_pct: float,
    entry: float,
) -> dict:
    if direction == "long":
        stop = entry * (1.0 - stop_pct)
        tp = entry * (1.0 + tp_pct)
    elif direction == "short":
        stop = entry * (1.0 + stop_pct)
        tp = entry * (1.0 - tp_pct)
    else:
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    return {
        "asset": symbol.upper(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "direction": direction,
        "entry_price": round(entry, 2),
        "stop_loss": round(stop, 2),
        "take_profit": round(tp, 2),
        "position_size_usd": round(notional, 2),
        "expected_return_pct": tp_pct,
        "iv_change_pct": 0.0,
        "confidence": 1.0,        # manual = full conviction by definition
        "risk_flags": {"manual": True},
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_status(args: argparse.Namespace) -> int:
    ex = AlpacaExecutor(live_money=args.live_money)
    equity = ex.get_account_equity()
    print(f"\nAccount equity: ${equity:,.2f}  (paper={ex._paper})")

    positions = ex.get_open_positions()
    if positions:
        print("\nOpen positions:")
        for p in positions:
            print(f"  {p['symbol']:6s}  qty={p['qty']:>8}  "
                  f"mv=${p['market_value']:>10,.2f}  pnl=${p['unrealized_pl']:>+10,.2f}")
    else:
        print("\nNo open positions.")

    # Open orders (separate API call, do it inline so we don't bloat the executor).
    orders = ex._client.get_orders()
    if orders:
        print("\nOpen / recent orders:")
        for o in orders:
            print(f"  {o.symbol:6s}  {o.side:<4}  qty={o.qty:<6}  "
                  f"{o.order_type:<8}  {o.status:<15}  submitted={o.submitted_at}")
    else:
        print("\nNo open orders.")
    return 0


def cmd_trade(args: argparse.Namespace, direction: str) -> int:
    if args.notional <= 0:
        print("Notional must be positive.", file=sys.stderr)
        return 2

    try:
        entry = _get_latest_quote(args.symbol)
    except Exception as exc:
        print(f"Could not fetch quote for {args.symbol}: {exc}", file=sys.stderr)
        return 3

    signal = _build_signal(
        symbol=args.symbol,
        direction=direction,
        notional=args.notional,
        stop_pct=args.stop_pct,
        tp_pct=args.tp_pct,
        entry=entry,
    )

    # Confirm before sending — easy to misclick a sell as a buy.
    print(f"\nAbout to submit:")
    print(json.dumps(signal, indent=2))
    if not args.yes:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return 0

    ex = AlpacaExecutor(live_money=args.live_money)
    result = ex.submit_signal(signal)
    if result.submitted:
        print(f"\n✓ Order submitted. Alpaca order id: {result.order_id}")
        return 0
    print(f"\n✗ Order NOT submitted: {result.reason}", file=sys.stderr)
    return 4


def cmd_close(args: argparse.Namespace) -> int:
    ex = AlpacaExecutor(live_money=args.live_money)
    symbol = args.symbol.upper()
    try:
        ex._client.close_position(symbol)
    except Exception as exc:
        print(f"Close failed for {symbol}: {exc}", file=sys.stderr)
        return 5
    print(f"✓ Close requested for {symbol}")
    return 0


def cmd_cancel_all(args: argparse.Namespace) -> int:
    ex = AlpacaExecutor(live_money=args.live_money)
    cancelled = ex._client.cancel_orders()
    print(f"✓ Cancel requested for {len(cancelled)} order(s).")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Manual trade entry for trading_enhancer.")
    parser.add_argument(
        "--live-money",
        action="store_true",
        help=("DANGER: route to the REAL-MONEY Alpaca endpoint. Requires "
              "ALPACA_LIVE=true in the environment as well."),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show paper account state.")
    p_status.set_defaults(func=cmd_status)

    for verb, direction in (("buy", "long"), ("sell", "short")):
        p = sub.add_parser(verb, help=f"Place a {direction} bracket order.")
        p.add_argument("--symbol", required=True)
        p.add_argument("--notional", type=float, required=True,
                       help="Dollar notional to allocate.")
        p.add_argument("--stop-pct", type=float, default=0.01,
                       help="Stop-loss distance as a fraction (default 0.01 = 1%%).")
        p.add_argument("--tp-pct", type=float, default=0.02,
                       help="Take-profit distance as a fraction (default 0.02 = 2%%).")
        p.add_argument("--yes", "-y", action="store_true",
                       help="Skip the y/N confirmation prompt.")
        p.set_defaults(func=lambda a, d=direction: cmd_trade(a, d))

    p_close = sub.add_parser("close", help="Flatten a single position.")
    p_close.add_argument("--symbol", required=True)
    p_close.set_defaults(func=cmd_close)

    p_cancel = sub.add_parser("cancel-all", help="Cancel all open orders.")
    p_cancel.set_defaults(func=cmd_cancel_all)

    args = parser.parse_args()
    configure_logging(level="INFO", log_path="trading_enhancer.log")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
