"""Sweep scanner CLI — run on demand or as a cron job.

Quick scan (default watchlist, write alerts to data/sweeps.jsonl):
    python scan_sweeps.py

Penny-only (price <= $5):
    python scan_sweeps.py --max-price 5

Custom watchlist from a file (one symbol per line):
    python scan_sweeps.py --watchlist my_smallcaps.txt

Loop every 5 minutes:
    python scan_sweeps.py --interval 300

The scanner uses free Yahoo OHLCV via the WebDataScraper. No
secondary API keys required.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scanners.sweep_detector import (
    DEFAULT_SMALL_CAP_WATCHLIST, SweepDetector,
)
from src.utils.logging_setup import configure_logging

logger = logging.getLogger("trading_enhancer.scan_sweeps")


def main() -> int:
    parser = argparse.ArgumentParser(description="Small-cap / penny stock sweep scanner")
    parser.add_argument("--watchlist",
                        help="Path to a text file with one symbol per line. "
                             "Uses bundled default list otherwise.")
    parser.add_argument("--volume-multiplier", type=float, default=3.0,
                        help="Today's volume must exceed N × 20-day avg "
                             "(default 3.0).")
    parser.add_argument("--sigma-multiplier", type=float, default=2.0,
                        help="|return| must exceed M × 20-day realized vol "
                             "(default 2.0).")
    parser.add_argument("--price-threshold", type=float, default=0.05,
                        help="Absolute price-move floor (default 0.05 = 5%%).")
    parser.add_argument("--min-dollar-volume", type=float, default=100_000,
                        help="Min trailing avg dollar volume (default 100k).")
    parser.add_argument("--max-price", type=float, default=None,
                        help="Optional price ceiling (e.g. 5 for true penny scope).")
    parser.add_argument("--interval", type=int, default=0,
                        help="Loop every N seconds (default 0 = one-shot).")
    parser.add_argument("--output", default="data/sweeps.jsonl",
                        help="Where to append alerts.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    configure_logging(level=args.log_level, log_path="trading_enhancer.log")

    if args.watchlist:
        symbols = [s.strip() for s in Path(args.watchlist).read_text().splitlines()
                    if s.strip() and not s.startswith("#")]
    else:
        symbols = list(DEFAULT_SMALL_CAP_WATCHLIST)

    detector = SweepDetector(
        watchlist=symbols,
        output_path=Path(args.output),
        volume_multiplier=args.volume_multiplier,
        sigma_multiplier=args.sigma_multiplier,
        price_threshold=args.price_threshold,
        min_dollar_volume_avg=args.min_dollar_volume,
        max_price=args.max_price,
    )

    while True:
        print(f"\n=== Scanning {len(symbols)} symbols ===")
        alerts = detector.scan()
        if alerts:
            print(f"Found {len(alerts)} sweep(s):")
            for a in alerts[:20]:
                arrow = "↑" if a.direction == "up" else "↓"
                print(f"  {arrow}  {a.symbol:<6}  ${a.close:>8.2f}  "
                      f"{a.price_change_pct:+6.2f}%   "
                      f"vol {a.volume_ratio:>5.2f}×   "
                      f"σ {a.realized_vol_sigmas:>4.2f}   "
                      f"score {a.score:>6.2f}")
            detector.write(alerts)
            print(f"\nAppended to: {detector.output_path}")
        else:
            print("No sweeps detected.")

        if args.interval <= 0:
            break
        print(f"\nSleeping {args.interval}s... (Ctrl-C to stop)")
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
