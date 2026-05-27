"""Continuous scenario battery — runs forever, archives between batches.

After each full battery completes, the per-scenario detail is
aggregated into ``data/all_trades.csv`` + tar-gzipped into
``data/archive/batch_<ts>.tar.gz``, the live scenario directories are
cleaned out, and a new battery starts.

The Backtests tab's live equity cards stay populated as the new batch
takes over.

Stop with Ctrl-C or by creating a sentinel file ``CONTINUOUS_STOP`` in
the project root.

Examples
--------
Default — full battery, 4 parallel, 30s rest between iterations:
    python continuous_practice.py

Quick mode — one symbol per scenario, no rest between:
    python continuous_practice.py --first-symbol-only --rest 0

Specific category only:
    python continuous_practice.py --category crash --rest 60

Stop after N batches (for testing):
    python continuous_practice.py --max-batches 3
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.scenario_runner import ScenarioRunner
from src.backtest.scenarios import all_scenarios, by_category, by_name
from src.utils.archiver import BatchArchiver
from src.utils.logging_setup import configure_logging

from run import _build_ensemble

logger = logging.getLogger("trading_enhancer.continuous_practice")

STOP_FILE = ROOT / "CONTINUOUS_STOP"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Continuously rerun the scenario battery, archiving each round."
    )
    parser.add_argument("--category", help="Limit to one category.")
    parser.add_argument("--scenario", help="Single scenario by name.")
    parser.add_argument("--first-symbol-only", action="store_true",
                        help="One symbol per scenario per batch.")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--equity", type=float, default=100_000)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--rest", type=int, default=30,
                        help="Seconds to rest between batches (default 30).")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Stop after this many batches (0 = forever).")
    parser.add_argument("--keep-archives", type=int, default=30,
                        help="How many old batch tarballs to keep (default 30).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(level=args.log_level, log_path="trading_enhancer.log")

    if args.scenario:
        s = by_name(args.scenario)
        if s is None:
            print(f"Unknown scenario {args.scenario}", file=sys.stderr)
            return 2
        scenarios = [s]
    elif args.category:
        scenarios = by_category(args.category)
    else:
        scenarios = all_scenarios()

    archiver = BatchArchiver(keep_archives=args.keep_archives)
    runner = ScenarioRunner(
        ensemble_factory=_build_ensemble,
        initial_equity=args.equity,
        confidence_threshold=args.confidence,
        seq_len=args.seq_len,
        parallel_workers=max(1, args.parallel),
    )

    # Catch Ctrl-C cleanly.
    stopped = {"flag": False}

    def _stop(_sig=None, _frame=None):
        if not stopped["flag"]:
            print("\nStopping after current batch...")
            stopped["flag"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try: signal.signal(sig, _stop)
        except Exception: pass

    print(f"\nContinuous practice begins. Stop with Ctrl-C or "
          f"`touch {STOP_FILE.name}`.\n")

    batch_n = 0
    total_archived = 0
    total_trades = 0
    while not stopped["flag"] and not STOP_FILE.exists():
        batch_n += 1
        print(f"{'═' * 64}")
        print(f"  ✠  BATCH {batch_n}  —  {len(scenarios)} scenario(s), "
              f"{args.parallel} parallel  ✠")
        print(f"{'═' * 64}\n")

        t0 = time.time()
        results = runner.run_all(
            scenarios,
            symbol_limit_per_scenario=1 if args.first_symbol_only else None,
        )
        runner.write_report(results)
        elapsed = time.time() - t0

        # Archive — aggregates trades + tars per-scenario dirs
        stats = archiver.archive_current_batch()
        total_archived += stats.scenarios_archived
        total_trades += stats.trades_appended

        # Disk-usage snapshot
        ds = archiver.stats()
        print(f"\n  Battery finished in {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
        print(f"  Archived: {stats.scenarios_archived} scenarios, "
              f"{stats.trades_appended} trades, {stats.archive_size_kb:.1f} KB")
        print(f"  Master trades total : {ds['master_trades_kb']} KB")
        print(f"  Archives total      : {ds['archives_count']} files, "
              f"{ds['archives_total_kb']} KB")

        if args.max_batches and batch_n >= args.max_batches:
            print(f"\nReached --max-batches={args.max_batches}. Done.")
            break

        if stopped["flag"] or STOP_FILE.exists():
            break

        # Rest a bit so the dashboard can refresh and so we don't hammer Yahoo.
        if args.rest > 0:
            print(f"\n  Resting {args.rest}s before next batch...")
            for _ in range(args.rest):
                if stopped["flag"] or STOP_FILE.exists():
                    break
                time.sleep(1)

    # Cleanup sentinel
    if STOP_FILE.exists():
        try: STOP_FILE.unlink()
        except Exception: pass

    print()
    print("=" * 64)
    print(f"Total batches: {batch_n}")
    print(f"Total scenarios archived: {total_archived}")
    print(f"Total trades aggregated: {total_trades}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
