"""Weekend practice — replay the system through the historical scenario battery.

Designed to run unattended for hours. Replays ~20 documented market
episodes (COVID crash, Volmageddon, GameStop squeeze, 2022 rate shock,
boring grinds, etc.) using the SAME ensemble + feature engineer +
risk manager that the live agent uses. Writes a categorized
markdown + CSV report when done.

Quick start (everything, all symbols):
    python weekend_practice.py

A faster preview (one symbol per scenario):
    python weekend_practice.py --first-symbol-only

A single category:
    python weekend_practice.py --category crash

Just preview what would run:
    python weekend_practice.py --list

Outputs:
    data/scenario_runs/scenario_report_<ts>.md   — readable report
    data/scenario_runs/scenario_report_<ts>.csv  — raw numbers
    data/scenario_runs/<scenario>/<symbol>/      — per-run artifacts
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

from src.backtest.scenario_runner import ScenarioRunner
from src.backtest.scenarios import (
    all_scenarios, benchmark_summary, by_category, categories,
)
from src.learning.preflight import run_preflight
from src.utils.logging_setup import configure_logging

# Reuse the ensemble factory from run.py so we don't drift.
from run import _build_ensemble


logger = logging.getLogger("trading_enhancer.weekend_practice")


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekend scenario battery")
    parser.add_argument("--list", action="store_true",
                        help="Print the scenario library and exit.")
    parser.add_argument("--category",
                        help=("Limit to one category, e.g. crash, rally, "
                              "vol_spike, regime_change, grinding, crisis, "
                              "squeeze, post_event."))
    parser.add_argument("--scenario",
                        help="Run a single scenario by name.")
    parser.add_argument("--first-symbol-only", action="store_true",
                        help=("Run just one symbol per scenario instead of all. "
                              "Use this for a quick (~5 min) sanity pass."))
    parser.add_argument("--equity", type=float, default=100_000,
                        help="Starting equity for each backtest (default 100k).")
    parser.add_argument("--confidence", type=float, default=0.75,
                        help=("Confidence threshold for signal acceptance. "
                              "Lower (e.g. 0.55) to see what the model would do "
                              "without the 75%% gate."))
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Feature window length (default 256).")
    parser.add_argument("--parallel", type=int, default=4,
                        help=("Number of scenarios to run concurrently "
                              "(default 4; set 1 for sequential)."))
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    configure_logging(level=args.log_level, log_path="trading_enhancer.log")

    if args.list:
        print(benchmark_summary())
        return 0

    # Pick scenarios.
    if args.scenario:
        from src.backtest.scenarios import by_name
        s = by_name(args.scenario)
        if s is None:
            print(f"No scenario {args.scenario!r}. Use --list to see options.",
                  file=sys.stderr)
            return 2
        scenarios = [s]
    elif args.category:
        if args.category not in categories():
            print(f"Unknown category. Available: {', '.join(categories())}",
                  file=sys.stderr)
            return 2
        scenarios = by_category(args.category)
    else:
        scenarios = all_scenarios()

    # Preflight from the learning system. Surfaces relevant lessons.
    pre = run_preflight("train")
    print(pre.to_text())

    print(f"\nWill replay {len(scenarios)} scenario(s)")
    total_runs = sum(
        len(s.symbols) if not args.first_symbol_only else min(1, len(s.symbols))
        for s in scenarios
    )
    print(f"Total runs:  {total_runs}")
    print(f"Estimated:   {total_runs * 5} – {total_runs * 30} seconds total")
    print()

    runner = ScenarioRunner(
        ensemble_factory=_build_ensemble,
        initial_equity=args.equity,
        confidence_threshold=args.confidence,
        seq_len=args.seq_len,
        parallel_workers=max(1, args.parallel),
    )

    t0 = time.time()
    results = runner.run_all(
        scenarios,
        symbol_limit_per_scenario=1 if args.first_symbol_only else None,
    )
    elapsed = time.time() - t0

    report_path = runner.write_report(results)

    print()
    print("=" * 60)
    print(f"Battery complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Runs:     {len(results)}")
    print(f"Failed:   {sum(1 for r in results if r.failed)}")
    print(f"Report:   {report_path}")
    print("=" * 60)
    print()
    print("View the report:")
    print(f"  cat {report_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
