"""StrategyDoctor — self-diagnosing performance analyst.

Reads every closed trade from the journal + Alpaca activity feed,
identifies *why* trades won or lost, and writes calibrated parameter
adjustments to ``data/strategy_state.json``.  The TradingAgent overlays
that file on every regime refresh so changes take effect within minutes.

What it analyses
────────────────
  ┌─────────────────────────────────────────────────────────────────┐
  │  Dimension          │  Metric              │  Adjustment         │
  ├─────────────────────┼──────────────────────┼─────────────────────┤
  │  regime × side      │  win rate / R-mult   │  side_bias          │
  │  confidence bucket  │  win rate            │  confidence_threshold│
  │  stop distance      │  stop-hit rate       │  stop_pct_multiplier│
  │  holding duration   │  win vs scratch rate │  tp_pct_multiplier  │
  │  per-symbol         │  cumulative R        │  symbol_blacklist   │
  └─────────────────────┴──────────────────────┴─────────────────────┘

Adjustment bounds
─────────────────
  confidence_threshold   [0.26 – 0.55]
  stop_pct_multiplier    [0.60 – 2.50]
  tp_pct_multiplier      [0.50 – 2.50]
  max_positions_factor   [0.40 – 1.00]
  side_bias              "long_only" | "short_only" | "both"
  symbol_blacklist       []  (cleared after 5 consecutive wins)

Run triggers
────────────
  • Automatically after every 5 closed trades (agent calls maybe_run())
  • EOD summary (agent calls end_of_day())
  • Manual: python doctor.py [--verbose]
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
STATE_PATH    = ROOT / "data" / "strategy_state.json"
REPORT_PATH   = ROOT / "data" / "doctor_report.md"
JOURNAL_PATH  = ROOT / "data" / "trade_journal.jsonl"

# ── Tuneable bounds ────────────────────────────────────────────────────────────
CONF_MIN, CONF_MAX         = 0.26, 0.55
STOP_MULT_MIN, STOP_MULT_MAX = 0.60, 2.50
TP_MULT_MIN, TP_MULT_MAX   = 0.50, 2.50
POS_FACTOR_MIN             = 0.40

# ── Minimum sample sizes before a metric earns a rule change ──────────────────
MIN_TRADES_FOR_SIDE_BIAS   = 8    # per regime × side bucket
MIN_TRADES_FOR_CONF_TUNE   = 12   # total
MIN_TRADES_FOR_STOP_TUNE   = 10
MIN_TRADES_FOR_BAN         = 3    # symbol-level: ban after 3 losses, 0 wins

# ── What counts as a "bad" win rate ───────────────────────────────────────────
SIDE_BAD_WIN_RATE   = 0.38   # below this → flip side_bias
CONF_BAD_WIN_RATE   = 0.42   # below this → raise threshold
STOP_HIT_RATE_HIGH  = 0.55   # stops fire more than 55% → widen them
TP_SCRATCH_RATE_HIGH= 0.45   # scratches > 45% → tighten TP


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_journal() -> List[Dict[str, Any]]:
    """Return all closed trade records from the journal file."""
    if not JOURNAL_PATH.exists():
        return []
    records = []
    with open(JOURNAL_PATH) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") == "closed":
                    records.append(rec)
            except json.JSONDecodeError:
                pass
    return records


def _enrich_with_alpaca(
    records: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Cross-reference journal records with Alpaca filled orders for accuracy.

    Fills in ``exit_price`` and ``pnl_usd`` from the broker feed when the
    journal values are missing (e.g. manual closes, stop hits).
    """
    try:
        import os
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        key    = api_key    or os.getenv("ALPACA_API_KEY")
        secret = api_secret or os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            return records

        client = TradingClient(key, secret, paper=True)
        req    = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=100)
        orders = {
            o.symbol: o for o in client.get_orders(filter=req)
            if o.filled_avg_price is not None
        }

        for rec in records:
            sym = rec.get("symbol", "")
            if rec.get("exit_price") is None and sym in orders:
                o  = orders[sym]
                ep = float(o.filled_avg_price)
                rec["exit_price"]  = ep
                # Recalculate pnl_pct
                entry = float(rec.get("entry_price", 0) or 0)
                side  = rec.get("side", "long")
                if entry > 0:
                    pnl_pct = ((ep / entry) - 1) * (1 if side == "long" else -1) * 100
                    rec["pnl_pct"] = round(pnl_pct, 4)
                    rec["outcome"] = (
                        "win" if pnl_pct > 0.1
                        else "loss" if pnl_pct < -0.1
                        else "scratch"
                    )
    except Exception as exc:
        logger.debug("Alpaca enrichment skipped: %s", exc)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def _win_rate(trades: List[Dict]) -> float:
    if not trades:
        return 0.5
    wins = sum(1 for t in trades if t.get("outcome") == "win")
    return wins / len(trades)


def _mean_r(trades: List[Dict]) -> float:
    """Mean R-multiple; default 0 when R is missing."""
    rs = [float(t["r_multiple"]) for t in trades if t.get("r_multiple") is not None]
    return float(np.mean(rs)) if rs else 0.0


def _expectancy(trades: List[Dict]) -> float:
    """win_rate * avg_win_R + loss_rate * avg_loss_R  (in R units)."""
    wins   = [float(t["r_multiple"]) for t in trades
              if t.get("outcome") == "win" and t.get("r_multiple") is not None]
    losses = [float(t["r_multiple"]) for t in trades
              if t.get("outcome") == "loss" and t.get("r_multiple") is not None]
    if not wins and not losses:
        return 0.0
    wr   = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0.5
    awR  = float(np.mean(wins))   if wins   else 0.0
    alR  = float(np.mean(losses)) if losses else 0.0
    return wr * awR + (1 - wr) * alR


def _stop_hit_rate(trades: List[Dict]) -> float:
    """Fraction of losses that were stopped out (R < -0.85 = likely stop hit)."""
    losses = [t for t in trades if t.get("outcome") == "loss"]
    if not losses:
        return 0.0
    stopped = sum(
        1 for t in losses
        if t.get("r_multiple") is not None and float(t["r_multiple"]) < -0.85
    )
    return stopped / len(losses)


def _scratch_rate(trades: List[Dict]) -> float:
    if not trades:
        return 0.0
    scratches = sum(1 for t in trades if t.get("outcome") == "scratch")
    return scratches / len(trades)


# ─────────────────────────────────────────────────────────────────────────────
# Core Doctor
# ─────────────────────────────────────────────────────────────────────────────

class StrategyDoctor:
    """Diagnoses trading performance and prescribes parameter adjustments.

    Parameters
    ----------
    lookback_trades : int
        Only consider the most recent N closed trades.
    min_sample : int
        Override the global minimum sample sizes.
    """

    def __init__(
        self,
        lookback_trades: int = 200,
        *,
        state_path: Path = STATE_PATH,
        report_path: Path = REPORT_PATH,
    ) -> None:
        self.lookback    = lookback_trades
        self.state_path  = state_path
        self.report_path = report_path
        state_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Debounce: don't run more than once per 10 minutes unless forced.
        self._last_run: Optional[datetime] = None
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, *, force: bool = False) -> Dict[str, Any]:
        """Full diagnosis cycle.  Returns the prescription dict."""
        with self._lock:
            if not force and self._last_run:
                age = (datetime.now(timezone.utc) - self._last_run).total_seconds()
                if age < 600:
                    logger.debug("StrategyDoctor: skipping (ran %.0fs ago)", age)
                    return self._load_state()

            trades = _load_journal()[-self.lookback:]
            trades = _enrich_with_alpaca(trades)
            logger.info(
                "StrategyDoctor: analysing %d closed trades", len(trades),
            )

            prescription = self._prescribe(trades)
            self._save_state(prescription)
            report       = self._build_report(trades, prescription)
            self._save_report(report)
            self._last_run = datetime.now(timezone.utc)

            logger.info(
                "StrategyDoctor: prescription written → %s", self.state_path.name
            )
            return prescription

    def maybe_run(self, min_new_trades: int = 5) -> None:
        """Called by agent after each closed position.  Debounced."""
        try:
            trades = _load_journal()
            last_run_ts = (self._last_run or datetime.min.replace(tzinfo=timezone.utc))
            new_since = sum(
                1 for t in trades
                if t.get("close_ts") and
                   datetime.fromisoformat(t["close_ts"]) > last_run_ts
            )
            if new_since >= min_new_trades:
                self.run()
        except Exception as exc:
            logger.debug("StrategyDoctor.maybe_run skipped: %s", exc)

    def end_of_day(self) -> None:
        """Force a full diagnosis at market close."""
        try:
            self.run(force=True)
        except Exception as exc:
            logger.warning("StrategyDoctor EOD run failed: %s", exc)

    def load_adjustments(self) -> Dict[str, Any]:
        """Return current strategy adjustments (safe to call at any time)."""
        return self._load_state()

    # ── Prescription engine ───────────────────────────────────────────────────

    def _prescribe(self, trades: List[Dict]) -> Dict[str, Any]:
        """Generate parameter adjustments from closed trade statistics."""

        # Load current state as baseline so we don't over-correct on small moves.
        current = self._load_state()

        p: Dict[str, Any] = {
            "generated_at":         datetime.now(timezone.utc).isoformat(),
            "trade_count":          len(trades),
            "overall_win_rate":     round(_win_rate(trades), 3),
            "overall_expectancy_R": round(_expectancy(trades), 3),
            "adjustments":          {},
            "regime_overrides":     {},
            "symbol_blacklist":     list(current.get("symbol_blacklist", [])),
            "rationale":            [],
        }

        if not trades:
            p["rationale"].append("No closed trades yet — using defaults.")
            return p

        # ── 1. Regime × side win-rate analysis ──────────────────────────────
        by_regime_side: Dict[str, List[Dict]] = defaultdict(list)
        for t in trades:
            regime = t.get("regime_label") or t.get("features", {}).get("regime", "unknown")
            side   = t.get("side", "long")
            by_regime_side[f"{regime}|{side}"].append(t)

        regime_side_bias: Dict[str, str] = {}

        for key, group in by_regime_side.items():
            if len(group) < MIN_TRADES_FOR_SIDE_BIAS:
                continue
            regime, side = key.split("|")
            wr = _win_rate(group)
            er = _expectancy(group)

            if wr < SIDE_BAD_WIN_RATE and er < -0.1:
                # This side is losing money in this regime → block it.
                opposite_side = "short" if side == "long" else "long"
                new_bias = f"{opposite_side}_only"
                regime_side_bias[regime] = new_bias
                p["rationale"].append(
                    f"  {regime}/{side}: WR={wr:.0%} R={er:.2f} → "
                    f"switching to {new_bias}"
                )
            else:
                if regime not in regime_side_bias:
                    regime_side_bias[regime] = "both"

        if regime_side_bias:
            p["regime_overrides"]["side_bias"] = regime_side_bias

        # ── 2. Confidence threshold calibration ────────────────────────────
        if len(trades) >= MIN_TRADES_FOR_CONF_TUNE:
            # Split into low / high confidence bins
            sorted_conf = sorted(t.get("confidence", 0) for t in trades)
            median_conf = sorted_conf[len(sorted_conf) // 2]

            low_conf  = [t for t in trades if (t.get("confidence") or 0) <= median_conf]
            high_conf = [t for t in trades if (t.get("confidence") or 0) >  median_conf]

            wr_low  = _win_rate(low_conf)
            wr_high = _win_rate(high_conf)

            current_thr = float(
                current.get("adjustments", {}).get("confidence_threshold", 0.29)
            )

            if wr_low < CONF_BAD_WIN_RATE and len(low_conf) >= 5:
                # Low-confidence trades are losing → raise threshold
                new_thr = min(current_thr + 0.03, CONF_MAX)
                p["adjustments"]["confidence_threshold"] = round(new_thr, 3)
                p["rationale"].append(
                    f"  Low-conf WR={wr_low:.0%} < {CONF_BAD_WIN_RATE:.0%} → "
                    f"raising threshold {current_thr:.3f} → {new_thr:.3f}"
                )
            elif wr_high > 0.62 and wr_low > 0.55 and current_thr > 0.28:
                # Both buckets doing well → we can be less restrictive
                new_thr = max(current_thr - 0.02, CONF_MIN)
                p["adjustments"]["confidence_threshold"] = round(new_thr, 3)
                p["rationale"].append(
                    f"  Both confidence buckets WR>{55:.0%} → "
                    f"relaxing threshold {current_thr:.3f} → {new_thr:.3f}"
                )

        # ── 3. Stop-loss width calibration ─────────────────────────────────
        if len(trades) >= MIN_TRADES_FOR_STOP_TUNE:
            stop_hr = _stop_hit_rate(trades)
            cur_mult = float(
                current.get("adjustments", {}).get("stop_pct_multiplier", 1.0)
            )
            if stop_hr > STOP_HIT_RATE_HIGH:
                # Stops are firing too often → widen them
                new_mult = min(cur_mult + 0.15, STOP_MULT_MAX)
                p["adjustments"]["stop_pct_multiplier"] = round(new_mult, 2)
                p["rationale"].append(
                    f"  Stop hit rate={stop_hr:.0%} > {STOP_HIT_RATE_HIGH:.0%} → "
                    f"widening stops ×{cur_mult:.2f} → ×{new_mult:.2f}"
                )
            elif stop_hr < 0.20 and cur_mult > 1.0:
                # Stops rarely fire (exits happening elsewhere) → tighten
                new_mult = max(cur_mult - 0.10, 1.0)
                p["adjustments"]["stop_pct_multiplier"] = round(new_mult, 2)
                p["rationale"].append(
                    f"  Stop hit rate={stop_hr:.0%} (low) → "
                    f"tightening stops ×{cur_mult:.2f} → ×{new_mult:.2f}"
                )

        # ── 4. Take-profit calibration ─────────────────────────────────────
        if len(trades) >= MIN_TRADES_FOR_STOP_TUNE:
            scratch_r = _scratch_rate(trades)
            cur_tp = float(
                current.get("adjustments", {}).get("tp_pct_multiplier", 1.0)
            )
            if scratch_r > TP_SCRATCH_RATE_HIGH:
                # Too many scratches → TP is too far, let's tighten
                new_tp = max(cur_tp - 0.15, TP_MULT_MIN)
                p["adjustments"]["tp_pct_multiplier"] = round(new_tp, 2)
                p["rationale"].append(
                    f"  Scratch rate={scratch_r:.0%} > {TP_SCRATCH_RATE_HIGH:.0%} → "
                    f"tightening TP ×{cur_tp:.2f} → ×{new_tp:.2f}"
                )

        # ── 5. Symbol-level blacklist ───────────────────────────────────────
        by_sym: Dict[str, List[Dict]] = defaultdict(list)
        for t in trades[-60:]:   # last 60 trades for recency
            sym = t.get("symbol", "")
            if sym:
                by_sym[sym].append(t)

        blacklist = list(p["symbol_blacklist"])

        for sym, sym_trades in by_sym.items():
            wins   = sum(1 for t in sym_trades if t.get("outcome") == "win")
            losses = sum(1 for t in sym_trades if t.get("outcome") == "loss")
            cum_r  = sum(
                float(t["r_multiple"])
                for t in sym_trades if t.get("r_multiple") is not None
            )
            if losses >= MIN_TRADES_FOR_BAN and wins == 0 and cum_r < -1.5:
                if sym not in blacklist:
                    blacklist.append(sym)
                    p["rationale"].append(
                        f"  {sym}: {losses} losses / 0 wins, R={cum_r:.2f} → blacklisted"
                    )
            elif sym in blacklist and wins >= 2:
                # Rehabilitation — remove from blacklist after 2 recent wins
                blacklist.remove(sym)
                p["rationale"].append(
                    f"  {sym}: {wins} wins observed → removed from blacklist"
                )

        p["symbol_blacklist"] = blacklist

        # ── 6. Overall position factor (if broad drawdown) ─────────────────
        if len(trades) >= 15:
            wr_all = _win_rate(trades)
            er_all = _expectancy(trades)
            cur_pf = float(
                current.get("adjustments", {}).get("max_positions_factor", 1.0)
            )
            if wr_all < 0.35 and er_all < -0.2:
                # Systematic underperformance → cut max positions
                new_pf = max(cur_pf - 0.20, POS_FACTOR_MIN)
                p["adjustments"]["max_positions_factor"] = round(new_pf, 2)
                p["rationale"].append(
                    f"  Broad WR={wr_all:.0%} R={er_all:.2f} → "
                    f"cutting position factor {cur_pf:.2f} → {new_pf:.2f}"
                )
            elif wr_all > 0.60 and er_all > 0.30 and cur_pf < 1.0:
                # Performing well → restore position factor
                new_pf = min(cur_pf + 0.20, 1.0)
                p["adjustments"]["max_positions_factor"] = round(new_pf, 2)
                p["rationale"].append(
                    f"  Strong WR={wr_all:.0%} R={er_all:.2f} → "
                    f"restoring position factor {cur_pf:.2f} → {new_pf:.2f}"
                )

        if not p["rationale"]:
            p["rationale"].append("  All metrics within acceptable range — no changes.")

        return p

    # ── Report builder ────────────────────────────────────────────────────────

    def _build_report(
        self, trades: List[Dict], p: Dict[str, Any]
    ) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        n   = len(trades)
        wr  = _win_rate(trades)
        er  = _expectancy(trades)
        sr  = _stop_hit_rate(trades)
        scr = _scratch_rate(trades)

        # Per-regime breakdown
        by_regime: Dict[str, List] = defaultdict(list)
        for t in trades:
            r = t.get("regime_label") or t.get("features", {}).get("regime", "unknown")
            by_regime[r].append(t)

        # Per-side breakdown
        by_side: Dict[str, List] = defaultdict(list)
        for t in trades:
            by_side[t.get("side", "?")].append(t)

        lines = [
            f"# BRZRKR Strategy Doctor  —  {now}",
            "",
            "## Overall Performance",
            f"  Trades analysed : {n}",
            f"  Win rate        : {wr:.1%}",
            f"  Expectancy      : {er:+.2f} R",
            f"  Stop-hit rate   : {sr:.1%}",
            f"  Scratch rate    : {scr:.1%}",
            "",
            "## By Regime",
        ]
        for regime, group in sorted(by_regime.items()):
            lines.append(
                f"  {regime:<16s}  n={len(group):>3d}  "
                f"WR={_win_rate(group):.0%}  "
                f"R={_mean_r(group):+.2f}"
            )

        lines += ["", "## By Side"]
        for side, group in sorted(by_side.items()):
            lines.append(
                f"  {side:<8s}  n={len(group):>3d}  "
                f"WR={_win_rate(group):.0%}  "
                f"R={_mean_r(group):+.2f}"
            )

        lines += ["", "## Diagnosis & Actions"]
        for line in p["rationale"]:
            lines.append(line)

        if p.get("symbol_blacklist"):
            lines += ["", f"## Blacklisted Symbols  (temporary)"]
            for sym in p["symbol_blacklist"]:
                lines.append(f"  - {sym}")

        lines += ["", "---", f"*Doctor report auto-generated by BRZRKR strategy_doctor.py*"]
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text())
        except Exception:
            return {}

    def _save_state(self, prescription: Dict[str, Any]) -> None:
        try:
            self.state_path.write_text(
                json.dumps(prescription, indent=2, default=str)
            )
        except Exception as exc:
            logger.warning("StrategyDoctor: failed to save state: %s", exc)

    def _save_report(self, report: str) -> None:
        try:
            self.report_path.write_text(report)
        except Exception as exc:
            logger.warning("StrategyDoctor: failed to save report: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton + convenience
# ─────────────────────────────────────────────────────────────────────────────

_doctor: Optional[StrategyDoctor] = None


def get_doctor() -> StrategyDoctor:
    global _doctor
    if _doctor is None:
        _doctor = StrategyDoctor()
    return _doctor


def apply_doctor_adjustments(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay strategy_state.json adjustments on top of a config dict.

    Call this from the agent's regime-refresh path to pick up doctor changes
    without restarting.  Returns the merged dict.

    Example
    -------
    ::
        cfg_dict = apply_doctor_adjustments(cfg_dict)
        agent_cfg.confidence_threshold = cfg_dict.get(
            "confidence_threshold", agent_cfg.confidence_threshold
        )
    """
    try:
        state = get_doctor().load_adjustments()
        adj   = state.get("adjustments", {})
        for key, val in adj.items():
            cfg_dict[key] = val
        # Regime-specific side_bias
        if "regime_overrides" in state and "side_bias" in state["regime_overrides"]:
            cfg_dict["regime_side_bias"] = state["regime_overrides"]["side_bias"]
        if state.get("symbol_blacklist"):
            cfg_dict["symbol_blacklist"] = state["symbol_blacklist"]
    except Exception as exc:
        logger.debug("apply_doctor_adjustments skipped: %s", exc)
    return cfg_dict
