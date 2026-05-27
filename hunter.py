#!/usr/bin/env python3
"""✠ BRZRKR HUNTER ✠ — High-Yield Trade Finder

Scans ~66 symbols (3× the active agent universe) for the highest
expected-return setups, ranks them by yield score, and prints a
colour-coded ranked trade queue to the terminal.

Usage
─────
  # Quick scan — top 10 signals, min yield score 40
  python hunter.py

  # Top 20, lower threshold, save JSON results
  python hunter.py --top 20 --min-score 30 --save

  # Only STOCK and LEVERAGED_ETF asset classes
  python hunter.py --types STOCK LEVERAGED_ETF

  # Long signals only
  python hunter.py --direction long

  # Add extra tickers on the fly
  python hunter.py --add MARA RIOT BITF

  # Quiet mode (no banner, table only — pipe-friendly)
  python hunter.py --quiet

Output columns
──────────────
  Rank  Symbol  Type  Dir  Entry  Stop   Target  R:R  Yield%  Score  Note
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ── optional colour support ──────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False


def _c(text: str, color: str) -> str:
    """Wrap text in ANSI colour if colorama is available."""
    if not _HAS_COLOR:
        return text
    palette = {
        "red":    Fore.RED,
        "green":  Fore.GREEN,
        "yellow": Fore.YELLOW,
        "cyan":   Fore.CYAN,
        "white":  Fore.WHITE,
        "bold":   Style.BRIGHT,
        "dim":    Style.DIM,
    }
    return f"{palette.get(color, '')}{text}{Style.RESET_ALL}"


# ── logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,   # suppress noisy sub-module logs by default
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("hunter")


# ── banner ───────────────────────────────────────────────────────────────
_BANNER = r"""
  ██████  ██████  ███████ ███████ ██████  ██   ██ ██████
  ██   ██ ██   ██    ███     ███  ██   ██ ██  ██  ██   ██
  ██████  ██████    ███     ███   ██████  █████   ██████
  ██   ██ ██   ██  ███     ███    ██   ██ ██  ██  ██   ██
  ██████  ██   ██ ███████ ███████ ██   ██ ██   ██ ██   ██

         ✠  H U N T E R  ✠   High-Yield Trade Finder
"""


def _print_banner() -> None:
    print(_c(_BANNER, "red"))


# ── table printer ────────────────────────────────────────────────────────

def _col(text: str, width: int, align: str = "<") -> str:
    return f"{str(text):{align}{width}}"


def _print_table(signals: list, quiet: bool = False) -> None:
    from src.scanners.high_yield_scanner import HunterSignal

    if not signals:
        print(_c("  No signals above threshold.", "yellow"))
        return

    # Header
    hdr = (
        _col("#",      3)
        + _col("Symbol",  7)
        + _col("Type",    14)
        + _col("Dir",     6)
        + _col("Entry",   9)
        + _col("Stop",    9)
        + _col("Target",  9)
        + _col("R:R",     5)
        + _col("Yield%",  8)
        + _col("Score",   7)
        + "Note"
    )
    sep = "─" * len(hdr)

    if not quiet:
        print()
    print(_c(hdr, "bold"))
    print(_c(sep, "dim"))

    for i, sig in enumerate(signals, 1):
        dir_col   = _c("▲ LONG ", "green") if sig.direction == "long" \
                    else _c("▼ SHORT", "red")
        score_col = (
            _c(f"{sig.yield_score:5.1f}", "green")  if sig.yield_score >= 70
            else _c(f"{sig.yield_score:5.1f}", "yellow") if sig.yield_score >= 50
            else f"{sig.yield_score:5.1f}"
        )
        yield_col = (
            _c(f"{sig.expected_yield_pct:+.2f}%", "green")  if sig.expected_yield_pct >= 5
            else _c(f"{sig.expected_yield_pct:+.2f}%", "yellow") if sig.expected_yield_pct >= 2
            else f"{sig.expected_yield_pct:+.2f}%"
        )

        row = (
            _col(f"{i:2}.", 3)
            + _col(sig.symbol, 7)
            + _col(sig.asset_type[:13], 14)
            + dir_col + "  "
            + _col(f"{sig.entry_price:.2f}", 9)
            + _col(f"{sig.stop_price:.2f}", 9)
            + _col(f"{sig.target_price:.2f}", 9)
            + _col(f"{sig.rr_ratio:.1f}×", 5)
            + _col(yield_col, 8)
            + _col(score_col, 7)
            + _c(sig.note[:80], "dim")
        )
        print(row)

    print(_c(sep, "dim"))
    print(
        _c(f"  {len(signals)} signal(s) | ", "dim")
        + _c(f"scanned {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", "dim")
    )
    print()


# ── save results ─────────────────────────────────────────────────────────

def _save_signals(signals: list, out_dir: Path) -> Path:
    from dataclasses import asdict
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"hunter_{ts}.json"
    data = [asdict(s) for s in signals]
    out_path.write_text(json.dumps(data, indent=2))
    return out_path


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hunter",
        description="BRZRKR Hunter — scan ~66 symbols for the highest-yield setups",
    )
    parser.add_argument(
        "--top", "-n", type=int, default=10,
        help="Number of top signals to display (default: 10)",
    )
    parser.add_argument(
        "--min-score", type=float, default=40.0,
        help="Minimum yield score 0–100 (default: 40)",
    )
    parser.add_argument(
        "--direction", choices=["long", "short", "both"], default="both",
        help="Filter by trade direction (default: both)",
    )
    parser.add_argument(
        "--types", nargs="+", metavar="TYPE",
        help=(
            "Filter by asset type(s): STOCK LEVERAGED_ETF SECTOR_ETF "
            "INDEX_ETF COMMODITY_ETF BOND_ETF VOLATILITY CRYPTO_ETF"
        ),
    )
    parser.add_argument(
        "--add", nargs="+", metavar="SYM",
        help="Extra tickers to add to this scan run",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results to data/hunter/ as JSON",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Skip banner and extra formatting (pipe-friendly)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show DEBUG logs from sub-modules",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.quiet:
        _print_banner()

    # ── run scanner ──────────────────────────────────────────────────
    from src.scanners.high_yield_scanner import HighYieldScanner

    if not args.quiet:
        print(_c(f"  Scanning {_count_universe()} symbols across 8 asset classes…", "cyan"))
        print()

    scanner = HighYieldScanner(
        min_yield_score=args.min_score,
        top_n=None,   # we'll slice after direction/type filter
    )

    try:
        signals = scanner.scan(extra_symbols=args.add)
    except Exception as exc:
        logger.error("Scanner failed: %s", exc, exc_info=True)
        print(_c(f"\n  ERROR: scanner failed — {exc}", "red"))
        sys.exit(1)

    # ── apply filters ────────────────────────────────────────────────
    if args.direction != "both":
        signals = [s for s in signals if s.direction == args.direction]

    if args.types:
        allowed = {t.upper() for t in args.types}
        signals = [s for s in signals if s.asset_type.upper() in allowed]

    # Top-N after filters
    signals = signals[: args.top]

    # ── display ──────────────────────────────────────────────────────
    if not args.quiet:
        longs  = sum(1 for s in signals if s.direction == "long")
        shorts = sum(1 for s in signals if s.direction == "short")
        print(
            _c(f"  Top {len(signals)} signal(s)  │  ", "bold")
            + _c(f"{longs} long", "green")
            + _c("  /  ", "dim")
            + _c(f"{shorts} short", "red")
        )

    _print_table(signals, quiet=args.quiet)

    # ── save ─────────────────────────────────────────────────────────
    if args.save and signals:
        out_path = _save_signals(signals, Path("data/hunter"))
        print(_c(f"  Saved → {out_path}", "cyan"))
        print()


def _count_universe() -> int:
    try:
        from src.scanners.high_yield_scanner import HIGH_YIELD_UNIVERSE
        return len(HIGH_YIELD_UNIVERSE)
    except Exception:
        return 66


if __name__ == "__main__":
    main()
