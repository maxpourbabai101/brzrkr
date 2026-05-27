#!/usr/bin/env python3
"""✠ BRZRKR BREAKOUT ✠ — Full NYSE/NASDAQ Breakout Scanner

Runs the institutional-grade 6-stage pipeline across the full NYSE and
NASDAQ to surface the highest-probability life-changing setups:

  Stage 1  Finviz pre-screen  ~800 liquid candidates from ~7000 stocks
  Stage 2  Trend template     Minervini 8-condition SMA stack
  Stage 3  VCP detection      ATR contraction · volume dry-up · tight closes
  Stage 4  RS Rating          IBD formula, percentile-ranked vs. universe
  Stage 5  Fundamentals       EPS/revenue acceleration · short squeeze fuel
  Stage 6  Breakout trigger   Price vs pivot · volume surge confirmation

Usage
─────
  ./breakout.py                    # scan NYSE+NASDAQ, top 20
  ./breakout.py --top 50           # top 50 results
  ./breakout.py --min-score 65     # raise threshold (tighter filter)
  ./breakout.py --exchange NYSE    # NYSE only
  ./breakout.py --add RXRX IONQ    # add specific tickers
  ./breakout.py --filter breakout  # only confirmed breakouts
  ./breakout.py --save             # save full JSON to data/breakouts/
  ./breakout.py --full             # print detailed scorecard per stock
"""

from __future__ import annotations

# ── self-activating venv bootstrap ──────────────────────────────────────────
import sys as _sys, os as _os
_VENV_PY = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "venv", "bin", "python")
if _os.path.exists(_VENV_PY) and _os.path.abspath(_sys.executable) != _os.path.abspath(_VENV_PY):
    _os.execv(_VENV_PY, [_VENV_PY] + _sys.argv)
del _sys, _os, _VENV_PY
# ────────────────────────────────────────────────────────────────────────────

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# ── colour support ───────────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as _cinit
    _cinit(autoreset=True)
    _C = True
except ImportError:
    _C = False

def _c(text, col):
    if not _C:
        return str(text)
    palette = {
        "red":    Fore.RED,    "green": Fore.GREEN, "yellow": Fore.YELLOW,
        "cyan":   Fore.CYAN,   "white": Fore.WHITE, "magenta": Fore.MAGENTA,
        "bold":   Style.BRIGHT,"dim":   Style.DIM,
    }
    return f"{palette.get(col,'')}{text}{Style.RESET_ALL}"

# ── banner ───────────────────────────────────────────────────────────────────
_BANNER = r"""
  ██████  ██████  ███████ █████  ██   ██  ██████  ██    ██ ████████
  ██   ██ ██   ██ ██      ██  ██ ██  ██  ██    ██ ██    ██    ██
  ██████  ██████  █████   ███████ █████   ██    ██ ██    ██    ██
  ██   ██ ██   ██ ██      ██  ██ ██  ██  ██    ██ ██    ██    ██
  ██████  ██   ██ ███████ ██  ██ ██   ██  ██████   ██████     ██

         ✠  NYSE / NASDAQ  ·  Life-Changing Breakout Hunter  ✠
"""


def _print_banner():
    print(_c(_BANNER, "red"))


# ── progress bar ─────────────────────────────────────────────────────────────
_PROG_STEPS = {
    "universe": "Fetching universe…",
    "ohlcv":    "Downloading OHLCV… ",
    "rs_rating":"Computing RS Ratings",
    "analysis": "Deep analysis…     ",
    "done":     "Done               ",
}

def _prog_cb(step: str, pct: float):
    label = _PROG_STEPS.get(step, step)
    bar_w = 30
    filled = int(bar_w * pct)
    bar = "█" * filled + "░" * (bar_w - filled)
    print(f"\r  {_c(label, 'cyan')}  [{_c(bar, 'red')}]  {pct*100:4.0f}%",
          end="", flush=True)
    if pct >= 1.0:
        print()


# ── result table ─────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 80:  return "green"
    if score >= 65:  return "yellow"
    return "white"

def _col(text, width, align="<"):
    return f"{str(text):{align}{width}}"

def _print_table(results: list, full: bool = False) -> None:
    from src.scanners.breakout_hunter import BreakoutResult

    if not results:
        print(_c("\n  No signals above threshold.\n", "yellow"))
        return

    # ── Summary table ────────────────────────────────────────────────────────
    hdr = (
        _col("#",     3)
        + _col("Symbol",  7)
        + _col("Score",   7)
        + _col("Setup",   26)
        + _col("RS",      5)
        + _col("Trend",   7)
        + _col("VCP",     6)
        + _col("Entry",   9)
        + _col("Stop",    9)
        + _col("Target",  9)
        + _col("R:R",     5)
        + _col("EPS%",    7)
        + _col("Short%",  8)
        + "Note"
    )
    sep = "─" * min(len(hdr), 160)
    print()
    print(_c(hdr, "bold"))
    print(_c(sep, "dim"))

    for i, r in enumerate(results, 1):
        score_str  = _c(f"{r.composite_score:5.1f}", _score_color(r.composite_score))
        trend_str  = (
            _c(f"✓{r.trend_score}/8", "green") if r.trend_score >= 7
            else _c(f"~{r.trend_score}/8", "yellow") if r.trend_score >= 5
            else _c(f"✗{r.trend_score}/8", "dim")
        )
        bk_flag    = _c("⚡LIVE ", "green") if r.breakout_confirmed else "      "
        eps_str    = _c(f"{r.eps_growth_pct:+.0f}%", "green") if r.eps_growth_pct >= 25 else f"{r.eps_growth_pct:+.0f}%"
        short_str  = _c(f"{r.short_float_pct:.1f}%", "magenta") if r.short_float_pct >= 15 else f"{r.short_float_pct:.1f}%"
        rr_str     = _c(f"{r.rr_ratio:.1f}×", "green") if r.rr_ratio >= 2.5 else f"{r.rr_ratio:.1f}×"

        row = (
            _col(f"{i:2}.", 3)
            + _col(r.symbol,        7)
            + _col(score_str,       7)
            + _col(r.setup_type[:25], 26)
            + _col(f"{r.rs_rating:.0f}", 5)
            + _col(trend_str,       7)
            + _col(f"{r.vcp_score:.0f}", 6)
            + _col(f"{r.entry_price:.2f}", 9)
            + _col(f"{r.stop_price:.2f}",  9)
            + _col(f"{r.target_price:.2f}", 9)
            + _col(rr_str,          5)
            + _col(eps_str,         7)
            + _col(short_str,       8)
            + _c(r.note[:60], "dim")
        )
        print(row)

        if full:
            _print_detail(r)

    print(_c(sep, "dim"))
    scanned_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(
        _c(f"  {len(results)} setup(s)  ·  ", "dim")
        + _c(scanned_at, "dim")
        + (_c("  ·  ⚡ = confirmed breakout today", "dim")
           if any(r.breakout_confirmed for r in results) else "")
    )
    print()


def _print_detail(r) -> None:
    """Expanded scorecard printed below each row in --full mode."""
    indent = "        "
    flags  = []
    if r.passes_trend_template: flags.append(_c("Trend Template ✓", "green"))
    if r.vcp_detected:          flags.append(_c("VCP ✓", "green"))
    if r.breakout_confirmed:    flags.append(_c("BREAKOUT ✓", "green"))
    if r.insider_buying:        flags.append(_c("Insider Buy ✓", "cyan"))

    print(_c(indent + f"Sector: {r.sector}  ·  Industry: {r.industry}  ·  Mkt Cap: ${r.market_cap_b:.1f}B", "dim"))
    print(_c(indent + f"Momentum: 5d {r.momentum_5d:+.1f}%  20d {r.momentum_20d:+.1f}%  13wk {r.momentum_63d:+.1f}%", "dim"))
    print(_c(indent + f"ATR contraction: {r.atr_contraction:.0%}  ·  Vol ratio: {r.volume_ratio:.1f}×  ·  Short float: {r.short_float_pct:.1f}%  ·  DTC: {r.days_to_cover:.1f}d", "dim"))
    print(_c(indent + f"Scores → RS:{r.rs_rating:.0f}  Trend:{r.trend_score}/8  VCP:{r.vcp_score:.0f}  Breakout:{r.breakout_score:.0f}  Fund:{r.fundamental_score:.0f}", "dim"))
    if flags:
        print(indent + "  ".join(flags))
    print()


# ── save ─────────────────────────────────────────────────────────────────────

def _save(results: list, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"breakouts_{ts}.json"
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    return path


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="breakout",
        description="BRZRKR Breakout — full NYSE/NASDAQ institutional setup scanner",
    )
    p.add_argument("--top",         type=int,   default=20,
                   help="Number of results to display (default: 20)")
    p.add_argument("--min-score",   type=float, default=50.0,
                   help="Minimum composite score 0–100 (default: 50)")
    p.add_argument("--exchange",    choices=["NYSE","NASDAQ","BOTH"], default="BOTH",
                   help="Exchange filter (default: BOTH)")
    p.add_argument("--add",         nargs="+", metavar="SYM",
                   help="Extra symbols to add to this scan")
    p.add_argument("--filter",      choices=["all","breakout","vcp","stage2"],
                   default="all",
                   help="Show only confirmed breakouts / VCP / Stage2 setups")
    p.add_argument("--max-universe",type=int, default=800,
                   help="Cap finviz universe at N tickers (default: 800)")
    p.add_argument("--save",        action="store_true",
                   help="Save results to data/breakouts/ as JSON")
    p.add_argument("--full",        action="store_true",
                   help="Print expanded scorecard per result")
    p.add_argument("--quiet","-q",  action="store_true",
                   help="Skip banner (pipe-friendly)")
    p.add_argument("--verbose","-v",action="store_true",
                   help="Show debug logs")
    args = p.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(levelname)s | %(name)s | %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING,
                            format="%(levelname)s | %(message)s")

    if not args.quiet:
        _print_banner()

    exchanges = (
        ["NYSE", "NASDAQ"] if args.exchange == "BOTH"
        else [args.exchange]
    )

    from src.scanners.breakout_hunter import BreakoutHunter

    hunter = BreakoutHunter(
        min_composite  = args.min_score,
        top_n          = None,        # we filter after
        exchanges      = exchanges,
        max_universe   = args.max_universe,
        verbose        = args.verbose,
    )

    if not args.quiet:
        universe_label = " + ".join(exchanges)
        print(_c(f"  Scanning {universe_label} — this takes ~60–90 seconds…", "cyan"))
        print()

    t0 = time.time()
    try:
        results = hunter.scan(
            extra_symbols=args.add,
            progress_cb=None if args.quiet else _prog_cb,
        )
    except KeyboardInterrupt:
        print(_c("\n  Interrupted.", "yellow"))
        sys.exit(0)
    except Exception as exc:
        print(_c(f"\n  ERROR: {exc}", "red"))
        if args.verbose:
            import traceback; traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0

    # ── Post-scan filters ────────────────────────────────────────────────────
    if args.filter == "breakout":
        results = [r for r in results if r.breakout_confirmed]
    elif args.filter == "vcp":
        results = [r for r in results if r.vcp_detected]
    elif args.filter == "stage2":
        results = [r for r in results if r.passes_trend_template]

    results = results[: args.top]

    # ── Print summary ────────────────────────────────────────────────────────
    if not args.quiet:
        confirmed  = sum(1 for r in results if r.breakout_confirmed)
        vcp_cnt    = sum(1 for r in results if r.vcp_detected and not r.breakout_confirmed)
        elite_rs   = sum(1 for r in results if r.rs_rating >= 90)
        print(
            _c(f"  {len(results)} setup(s) found  │  ", "bold")
            + _c(f"{confirmed} ⚡confirmed breakouts", "green") + "  "
            + _c(f"{vcp_cnt} VCP coiling", "yellow") + "  "
            + _c(f"{elite_rs} RS≥90 leaders", "cyan")
            + _c(f"  │  {elapsed:.0f}s", "dim")
        )

    _print_table(results, full=args.full)

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.save and results:
        out = _save(results, Path("data") / "breakouts")
        print(_c(f"  Saved → {out}", "cyan"))
        print()


if __name__ == "__main__":
    main()
