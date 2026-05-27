"""BRZRKR Strategy Doctor — standalone diagnostic CLI.

Usage
─────
    python doctor.py               # run diagnosis, print report
    python doctor.py --verbose     # also dump per-trade breakdown
    python doctor.py --reset       # clear adjustments (back to defaults)

The doctor analyses every closed trade in data/trade_journal.jsonl,
cross-references with Alpaca order history, and:

  1. Identifies systematic weaknesses (short in uptrend, confidence
     not predictive, stops too tight, etc.)
  2. Writes calibrated parameter adjustments to data/strategy_state.json
  3. Writes a human-readable report to data/doctor_report.md
  4. Prints a colour-coded summary to the terminal

The live trading agent reads strategy_state.json on every regime refresh
(every ~1 hour) so changes take effect without restarting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from src.learning.strategy_doctor import (
    StrategyDoctor, STATE_PATH, REPORT_PATH, _load_journal,
    _win_rate, _expectancy, _mean_r, _stop_hit_rate, _scratch_rate,
)


# ── ANSI colour helpers ────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _colour(val: float, good: float, bad: float, *, invert: bool = False) -> str:
    """Return ANSI colour for a metric (green=good, red=bad)."""
    if invert:
        good, bad = bad, good
    if val >= good:
        return GREEN
    if val <= bad:
        return RED
    return YELLOW


def _pct(val: float) -> str:
    return f"{val:.1%}"


def _r(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}R"


def _print_summary(trades, prescription):
    n   = len(trades)
    wr  = _win_rate(trades)
    er  = _expectancy(trades)
    sr  = _stop_hit_rate(trades)
    scr = _scratch_rate(trades)

    print(f"\n{BOLD}{CYAN}╔══ BRZRKR STRATEGY DOCTOR ══════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║  Analysed: {n:>4d} closed trades{RESET}")
    print(f"{CYAN}╚════════════════════════════════════════════════════════╝{RESET}\n")

    # Overall metrics
    wrc = _colour(wr, 0.55, 0.40)
    erc = _colour(er, 0.20, -0.10)
    src = _colour(sr, 0.20, 0.55, invert=True)
    scrc= _colour(scr, 0.20, 0.50, invert=True)

    print(f"  {BOLD}Overall metrics{RESET}")
    print(f"    Win rate     {wrc}{_pct(wr)}{RESET}")
    print(f"    Expectancy   {erc}{_r(er)}{RESET}")
    print(f"    Stop-hit     {src}{_pct(sr)}{RESET}")
    print(f"    Scratch rate {scrc}{_pct(scr)}{RESET}")
    print()

    # By regime
    from collections import defaultdict
    by_regime = defaultdict(list)
    for t in trades:
        r = t.get("regime_label") or t.get("features", {}).get("regime", "unknown")
        by_regime[r].append(t)

    if by_regime:
        print(f"  {BOLD}By regime{RESET}")
        for regime, group in sorted(by_regime.items()):
            rwr = _win_rate(group)
            rer = _mean_r(group)
            rwc = _colour(rwr, 0.55, 0.38)
            rec = _colour(rer, 0.10, -0.10)
            print(
                f"    {regime:<18s}  n={len(group):>3d}  "
                f"WR={rwc}{_pct(rwr)}{RESET}  R={rec}{_r(rer)}{RESET}"
            )
        print()

    # By side
    by_side = defaultdict(list)
    for t in trades:
        by_side[t.get("side", "?")].append(t)

    if len(by_side) > 1:
        print(f"  {BOLD}By side{RESET}")
        for side, group in sorted(by_side.items()):
            rwr = _win_rate(group)
            rer = _mean_r(group)
            rwc = _colour(rwr, 0.55, 0.38)
            rec = _colour(rer, 0.10, -0.10)
            print(
                f"    {side:<10s}  n={len(group):>3d}  "
                f"WR={rwc}{_pct(rwr)}{RESET}  R={rec}{_r(rer)}{RESET}"
            )
        print()

    # Prescription
    adj = prescription.get("adjustments", {})
    reg_bias = prescription.get("regime_overrides", {}).get("side_bias", {})
    blacklist = prescription.get("symbol_blacklist", [])
    rationale = prescription.get("rationale", [])

    print(f"  {BOLD}Diagnosis{RESET}")
    for line in rationale:
        colour = GREEN if "→" in line and any(
            w in line.lower() for w in ("relax", "restor", "remov")
        ) else YELLOW if "→" in line else ""
        print(f"  {colour}{line}{RESET}")
    print()

    if adj or reg_bias or blacklist:
        print(f"  {BOLD}Active adjustments  (data/strategy_state.json){RESET}")
        for k, v in adj.items():
            print(f"    {k} = {v}")
        for regime, bias in reg_bias.items():
            print(f"    side_bias[{regime}] = {bias}")
        if blacklist:
            print(f"    blacklist = {blacklist}")
    else:
        print(f"  {GREEN}No adjustments needed — strategy is performing well.{RESET}")

    print(f"\n  Report saved → {REPORT_PATH}")
    print(f"  State saved  → {STATE_PATH}\n")


def _print_trade_breakdown(trades):
    if not trades:
        print("  (no closed trades yet)\n")
        return

    print(f"\n  {BOLD}Per-trade breakdown (last 30){RESET}")
    print(f"  {'Symbol':<7} {'Side':<6} {'Entry':>8} {'Exit':>8} "
          f"{'P&L%':>7} {'R':>5} {'Outcome':<8} {'Regime'}")
    print("  " + "-" * 72)

    for t in trades[-30:]:
        outcome  = t.get("outcome", "?")
        col      = GREEN if outcome == "win" else RED if outcome == "loss" else YELLOW
        pnl_pct  = t.get("pnl_pct")
        r_mult   = t.get("r_multiple")
        pnl_str  = f"{pnl_pct:+.2f}%" if pnl_pct is not None else "  ?"
        r_str    = f"{r_mult:+.2f}" if r_mult is not None else " ?"
        entry    = t.get("entry_price", 0)
        exit_p   = t.get("exit_price")
        exit_str = f"${exit_p:,.2f}" if exit_p else "  open"
        regime   = t.get("regime_label", t.get("features", {}).get("regime", ""))

        print(
            f"  {t.get('symbol','?'):<7} "
            f"{t.get('side','?'):<6} "
            f"${entry:>7,.2f} "
            f"{exit_str:>8} "
            f"{col}{pnl_str:>7}{RESET} "
            f"{col}{r_str:>5}{RESET} "
            f"{col}{outcome:<8}{RESET} "
            f"{regime}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="BRZRKR Strategy Doctor")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-trade breakdown")
    parser.add_argument("--reset", action="store_true",
                        help="Clear all adjustments (back to config defaults)")
    args = parser.parse_args()

    if args.reset:
        STATE_PATH.unlink(missing_ok=True)
        print(f"{GREEN}Strategy state cleared. Agent will use config.yaml defaults.{RESET}")
        return

    doctor = StrategyDoctor()
    prescription = doctor.run(force=True)

    trades = _load_journal()
    _print_summary(trades, prescription)

    if args.verbose:
        _print_trade_breakdown(trades)


if __name__ == "__main__":
    main()
