"""Vectorized walk-forward backtester.

Loads a historical OHLCV (and optional options) file, iterates through
each bar, asks the model ensemble for a prediction, runs the same risk
manager used in live trading, and aggregates per-trade outcomes.

Key realism improvements vs. naive backtests:
  • Multi-position support (up to max_positions simultaneous trades)
  • Gap-open fills: if bar opens beyond stop, fills at open (not stop price)
  • Stop-fill slippage: extra 0.05 % on stop-outs to model market impact
  • Realistic fees: 2 bp/side = 4 bp round-trip (Alpaca + slippage)
  • Sharpe adjusted for risk-free rate (10y Tbill proxy)
  • Fractional qty removed — whole shares only (matches live broker)

Outputs three artifacts:

* ``equity_curve.csv``  — bar-level equity series
* ``trades.csv``        — per-trade ledger
* a summary dict: win rate, avg P&L, Sharpe, max DD, MAE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.risk.risk_manager import (
    apply_stop_loss,
    apply_take_profit,
    calculate_position_size,
    monitor_drawdown,
)
from src.signals.signal_generator import CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# Risk-free rate proxy (annualised). Update periodically or pull from FRED.
RISK_FREE_ANNUAL = 0.04           # ~4 % (2025-2026 short-end rate)
RISK_FREE_DAILY  = RISK_FREE_ANNUAL / 252


@dataclass
class BacktestResult:
    summary: Dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.Series


@dataclass
class BacktestRunner:
    predict_fn: Callable[[pd.DataFrame], Dict[str, Any]]
    initial_equity: float = 100_000.0
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    fee_bps: float = 2.0           # 2 bp per side (entry + exit = 4 bp round-trip)
    slippage_bps: float = 1.0      # extra stop-fill slippage
    max_positions: int = 8         # max simultaneous open positions
    output_dir: Path = field(default_factory=lambda: Path("data/backtest_out"))
    live_writer: Optional[Any] = None
    live_update_every: int = 3

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    @staticmethod
    def load_history(path: str | Path) -> pd.DataFrame:
        p = Path(path)
        if p.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        df.columns = [c.lower() for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        df = df.sort_index()
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Backtest CSV missing required columns: {missing}")
        return df

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def run(self, history: pd.DataFrame, window: int = 256) -> BacktestResult:
        if len(history) <= window:
            raise ValueError(
                f"Need more than {window} bars to backtest; got {len(history)}"
            )

        equity = self.initial_equity
        equity_path: List[float] = [equity]
        trades: List[Dict[str, Any]] = []
        open_trades: List[Dict[str, Any]] = []  # ← multi-position

        bars  = history.iloc[window:]
        prior = history.iloc[:window]

        for ts, bar in bars.iterrows():
            feature_window = prior.tail(window)
            bar_open  = float(bar["open"])
            bar_high  = float(bar["high"])
            bar_low   = float(bar["low"])

            # ── 1. Manage all open trades (check stop / TP) ──────────
            still_open: List[Dict[str, Any]] = []
            for trade in open_trades:
                direction = trade["direction"]
                stop      = trade["stop"]
                tp        = trade["take_profit"]

                # Gap-open fill: if bar opens beyond stop, fill at open
                if direction == "long":
                    gap_stopped = bar_open <= stop
                    intra_stopped = bar_low <= stop
                    hit_stop = gap_stopped or intra_stopped
                    hit_tp   = bar_high >= tp
                    # For shorts the inverse
                else:
                    gap_stopped = bar_open >= stop
                    intra_stopped = bar_high >= stop
                    hit_stop = gap_stopped or intra_stopped
                    hit_tp   = bar_low <= tp

                exit_price: Optional[float] = None
                exit_reason = ""
                if hit_stop and hit_tp:
                    # Both triggered — assume stop (worse fill) came first
                    hit_tp = False

                if hit_stop:
                    if direction == "long":
                        # Gap fill: open already below stop
                        fill = bar_open if gap_stopped else stop
                        # Apply stop-fill slippage (fills slightly worse)
                        exit_price = fill * (1.0 - self.slippage_bps / 10_000)
                    else:
                        fill = bar_open if gap_stopped else stop
                        exit_price = fill * (1.0 + self.slippage_bps / 10_000)
                    exit_reason = "stop"
                elif hit_tp:
                    exit_price = tp
                    exit_reason = "tp"

                if exit_price is not None:
                    pnl = self._compute_pnl(trade, exit_price)
                    equity += pnl
                    trades.append({
                        **trade,
                        "exit": float(exit_price),
                        "exit_ts": ts,
                        "exit_reason": exit_reason,
                        "pnl": pnl,
                        # Max adverse excursion this bar
                        "mae_bar": float(
                            (bar_low - trade["entry"]) / trade["entry"]
                            if direction == "long"
                            else (trade["entry"] - bar_high) / trade["entry"]
                        ),
                    })
                else:
                    still_open.append(trade)

            open_trades = still_open

            # ── 2. Look for new entries (if below max_positions) ─────
            if len(open_trades) < self.max_positions:
                pred = self.predict_fn(feature_window)
                if pred and pred.get("confidence", 0) >= self.confidence_threshold:
                    direction = pred["direction"]
                    entry = bar_open          # fill at next-bar open
                    atr   = self._atr(prior.tail(14))
                    if atr > 0:
                        stop = apply_stop_loss(entry, direction, atr)
                        tp   = apply_take_profit(entry, stop, direction)
                        stop_pct = abs(entry - stop) / max(entry, 1e-6)
                        notional = calculate_position_size(
                            account_equity=equity,
                            confidence=pred["confidence"],
                            expected_return_pct=abs(pred.get("expected_return_pct", 0.01)),
                            max_loss_pct=stop_pct,
                        )
                        if notional > 0:
                            # Whole shares only — matches live broker
                            qty = int(notional // entry)
                            if qty >= 1:
                                actual_notional = qty * entry
                                open_trades.append({
                                    "entry_ts": ts,
                                    "direction": direction,
                                    "entry": entry,
                                    "stop": stop,
                                    "take_profit": tp,
                                    "qty": float(qty),
                                    "notional": actual_notional,
                                    "confidence": float(pred["confidence"]),
                                })

            equity_path.append(equity)
            prior = pd.concat([prior.iloc[1:], history.loc[[ts]]])

            # ── 3. Stream live status ─────────────────────────────────
            bars_done = len(equity_path) - 1
            if self.live_writer is not None and (
                bars_done % self.live_update_every == 0
                or bars_done == len(bars)
            ):
                try:
                    open_summary = None
                    if open_trades:
                        t = open_trades[-1]
                        open_summary = {
                            "direction": t["direction"],
                            "entry": float(t["entry"]),
                            "stop": float(t["stop"]),
                            "qty": float(t["qty"]),
                        }
                    self.live_writer.update(
                        bars_processed=bars_done,
                        current_equity=equity,
                        trades_so_far=len(trades),
                        open_trade=open_summary,
                    )
                except Exception:
                    pass

            if monitor_drawdown(equity_path, threshold=0.25):
                logger.warning("Drawdown breach at %s — halting backtest", ts)
                break

        # Close any position still open at end of series at last close price
        last_close = float(history["close"].iloc[-1])
        for trade in open_trades:
            pnl = self._compute_pnl(trade, last_close)
            equity += pnl
            trades.append({
                **trade,
                "exit": last_close,
                "exit_ts": bars.index[-1] if len(bars) else None,
                "exit_reason": "eod",
                "pnl": pnl,
                "mae_bar": 0.0,
            })

        equity_series = pd.Series(
            equity_path[1:], index=bars.index[: len(equity_path) - 1]
        )
        trades_df = pd.DataFrame(trades)
        summary   = self._summarize(trades_df, equity_series)

        if self.live_writer is not None:
            try:
                self.live_writer.update(
                    bars_processed=len(equity_path) - 1,
                    current_equity=equity,
                    trades_so_far=len(trades),
                    open_trade=None,
                )
            except Exception:
                pass

        self._write_outputs(trades_df, equity_series)
        return BacktestResult(summary=summary, trades=trades_df, equity_curve=equity_series)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_pnl(self, trade: Dict[str, Any], exit_price: float) -> float:
        sign  = 1.0 if trade["direction"] == "long" else -1.0
        gross = sign * (exit_price - trade["entry"]) * trade["qty"]
        # Round-trip fees: 2 bp/side × 2 sides = 4 bp of notional
        fees  = trade["notional"] * (self.fee_bps * 2) / 10_000.0
        return float(gross - fees)

    @staticmethod
    def _atr(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        high_low    = df["high"] - df["low"]
        high_close  = (df["high"] - df["close"].shift()).abs()
        low_close   = (df["low"]  - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return float(tr.mean())

    def _summarize(self, trades: pd.DataFrame, equity: pd.Series) -> Dict[str, float]:
        n = len(trades)
        if trades.empty or n == 0:
            return {
                "trades": 0, "win_rate": 0.0, "avg_pnl": 0.0,
                "sharpe": 0.0, "max_drawdown_pct": 0.0,
                "final_equity": float(
                    equity.iloc[-1] if not equity.empty else self.initial_equity
                ),
                "max_mae_pct": 0.0,
            }

        wins     = int((trades["pnl"] > 0).sum())
        win_rate = float(wins / n)
        avg_pnl  = float(trades["pnl"].mean())

        returns  = equity.pct_change().dropna()
        if not returns.empty and returns.std() > 1e-9:
            # Sharpe adjusted for risk-free rate
            sharpe = float(
                np.sqrt(252) * (returns.mean() - RISK_FREE_DAILY) / returns.std()
            )
        else:
            sharpe = 0.0

        running_max = equity.cummax()
        max_dd = float(((equity - running_max) / running_max).min())

        # Max adverse excursion across all trades
        max_mae = float(trades["mae_bar"].min()) if "mae_bar" in trades.columns else 0.0

        return {
            "trades": n,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd,
            "final_equity": float(equity.iloc[-1]),
            "max_mae_pct": max_mae * 100,
        }

    def _write_outputs(self, trades: pd.DataFrame, equity: pd.Series) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        equity.to_csv(self.output_dir / "equity_curve.csv", header=["equity"])
        trades.to_csv(self.output_dir / "trades.csv", index=False)
        logger.info("Backtest artifacts written to %s", self.output_dir)
