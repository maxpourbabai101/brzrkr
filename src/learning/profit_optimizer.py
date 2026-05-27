"""ProfitOptimizer — self-learning layer that tunes stop / take-profit
ratios to maximise the expected profit-to-risk margin.

Reads every `trades.csv` written by past backtests, partitions trades
by direction and (optionally) by VIX regime, and computes:

* the **expected R-multiple** (return ÷ initial risk) for the historical
  distribution of stop-to-TP combinations
* the **optimal stop_pct / tp_pct** that would have maximised expected
  profit per unit of risk, subject to a minimum-sample-size filter

Output is written to ``data/learned_params.json`` and consumed by the
signal generator on its next run. The current file is loadable via
:func:`load_learned_params`.

This is not a model — it's a parameter optimiser over closed trades.
Even with no model, getting stop/TP ratios right adds measurable EV.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

LEARNED_PARAMS_FILE = Path("data/learned_params.json")
# Order matters: master file (aggregated by the archiver) comes first
# so historical archived trades are included, then live per-batch
# trades.csv files that haven't been archived yet.
DEFAULT_TRADES_GLOBS = (
    "data/all_trades.csv",
    "data/backtest_out/trades.csv",
    "data/scenario_runs/*/*/trades.csv",
)


@dataclass
class OptimizationResult:
    n_trades: int = 0
    win_rate: float = 0.0
    avg_win_R: float = 0.0
    avg_loss_R: float = 0.0
    expected_R: float = 0.0
    profit_factor: float = 0.0
    optimal_stop_pct: float = 0.01
    optimal_tp_pct: float = 0.02
    optimal_rr_ratio: float = 2.0
    sharpe_proxy: float = 0.0
    by_direction: Dict[str, Dict[str, float]] = field(default_factory=dict)
    computed_at: str = ""
    sources_read: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProfitOptimizer:
    trade_globs: Iterable[str] = DEFAULT_TRADES_GLOBS
    output_path: Path = LEARNED_PARAMS_FILE
    min_trades: int = 20           # below this, fall back to defaults
    rr_candidates: List[float] = field(default_factory=lambda:
                                         [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    stop_candidates: List[float] = field(default_factory=lambda:
                                           [0.005, 0.0075, 0.01, 0.015, 0.02,
                                            0.025, 0.03])

    # ------------------------------------------------------------------
    def _load_trades(self) -> pd.DataFrame:
        """Glob every trades.csv, concatenate."""
        import glob
        frames: List[pd.DataFrame] = []
        sources: List[str] = []
        for pattern in self.trade_globs:
            for path_str in sorted(glob.glob(pattern)):
                path = Path(path_str)
                try:
                    df = pd.read_csv(path)
                    if not df.empty:
                        frames.append(df)
                        sources.append(str(path))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("skip %s: %s", path, exc)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out.attrs["sources"] = sources
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _trade_to_rmultiple(row) -> Optional[float]:
        """Convert a trade row to an R-multiple (profit ÷ initial risk).

        Requires columns: entry, stop, pnl OR (entry, exit, qty, stop,
        direction). Returns None if unknown columns.
        """
        try:
            entry = float(row["entry"])
            stop = float(row["stop"])
            risk_per_share = abs(entry - stop)
            if risk_per_share <= 0:
                return None

            if "pnl" in row and not pd.isna(row["pnl"]):
                pnl = float(row["pnl"])
                qty = float(row.get("qty", 1)) or 1
                return (pnl / qty) / risk_per_share

            # Fall back to exit-based.
            exit_p = float(row.get("exit", row.get("exit_price", 0)))
            direction = row.get("direction", "long")
            sign = 1 if direction == "long" else -1
            return sign * (exit_p - entry) / risk_per_share
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    def analyze(self) -> OptimizationResult:
        df = self._load_trades()
        result = OptimizationResult(
            computed_at=datetime.now(timezone.utc).isoformat(),
            sources_read=df.attrs.get("sources", []) if not df.empty else [],
        )
        if df.empty:
            result.notes = "no trades.csv found anywhere; using defaults"
            return result

        # Compute R-multiples per trade
        df["R"] = df.apply(self._trade_to_rmultiple, axis=1)
        df = df.dropna(subset=["R"])
        n = len(df)
        result.n_trades = int(n)

        if n < self.min_trades:
            result.notes = (
                f"only {n} trades with valid R-multiple; "
                f"need {self.min_trades}+. Using defaults."
            )
            return result

        wins = df[df["R"] > 0]
        losses = df[df["R"] <= 0]
        result.win_rate = float(len(wins) / n)
        result.avg_win_R = float(wins["R"].mean()) if not wins.empty else 0.0
        result.avg_loss_R = float(losses["R"].mean()) if not losses.empty else 0.0
        result.expected_R = float(df["R"].mean())
        if losses.empty or losses["R"].sum() == 0:
            result.profit_factor = float("inf")
        else:
            result.profit_factor = float(
                wins["R"].sum() / abs(losses["R"].sum())
            )
        if df["R"].std() > 0:
            result.sharpe_proxy = float(df["R"].mean() / df["R"].std())
        else:
            result.sharpe_proxy = 0.0

        # --- Grid-search optimal RR (TP / stop ratio) ---
        # Simulate truncating trades at each candidate RR to see which
        # produces the highest expected R.
        best_rr = 2.0
        best_ev = -float("inf")
        for rr in self.rr_candidates:
            truncated = df["R"].apply(lambda r: min(max(r, -1.0), rr))
            ev = float(truncated.mean())
            if ev > best_ev:
                best_ev = ev
                best_rr = rr
        result.optimal_rr_ratio = float(best_rr)

        # --- Choose a stop_pct that targets a realistic loss frequency ---
        # Pick the stop_pct closest to the empirical avg stop magnitude.
        avg_stop_pct = self._mean_stop_pct(df)
        if avg_stop_pct > 0:
            result.optimal_stop_pct = min(
                self.stop_candidates,
                key=lambda c: abs(c - avg_stop_pct),
            )
        result.optimal_tp_pct = round(result.optimal_stop_pct * best_rr, 6)

        # --- Per-direction stats ---
        for direction in ("long", "short"):
            if "direction" not in df.columns:
                continue
            sub = df[df["direction"] == direction]
            if sub.empty:
                continue
            result.by_direction[direction] = {
                "n_trades": int(len(sub)),
                "win_rate": float((sub["R"] > 0).mean()),
                "expected_R": float(sub["R"].mean()),
                "avg_win_R": float(sub[sub["R"] > 0]["R"].mean())
                              if not sub[sub["R"] > 0].empty else 0.0,
                "avg_loss_R": float(sub[sub["R"] <= 0]["R"].mean())
                               if not sub[sub["R"] <= 0].empty else 0.0,
            }

        result.notes = (
            f"optimised over {n} trades, "
            f"best RR={best_rr}, EV={best_ev:+.4f} R/trade"
        )
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _mean_stop_pct(df: pd.DataFrame) -> float:
        try:
            stops_pct = []
            for _, row in df.iterrows():
                entry = float(row["entry"])
                stop = float(row["stop"])
                if entry > 0:
                    stops_pct.append(abs(entry - stop) / entry)
            return float(sum(stops_pct) / len(stops_pct)) if stops_pct else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    # ------------------------------------------------------------------
    def write(self, result: OptimizationResult) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(result.to_dict(), indent=2, default=str)
        )
        return self.output_path


def load_learned_params(path: Path = LEARNED_PARAMS_FILE) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
