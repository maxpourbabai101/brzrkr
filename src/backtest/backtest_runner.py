"""Vectorized walk‑forward backtester.

Loads a historical OHLCV (and optional options) file, iterates through
each bar, asks the model ensemble for a prediction, runs the same risk
manager used in live trading, and aggregates per‑trade outcomes.

Outputs three artifacts:

* ``equity_curve.csv``  — bar‑level equity series
* ``trades.csv``        — per‑trade ledger
* a summary dict containing win rate, average profit, Sharpe, max DD.
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
    fee_bps: float = 1.0           # 1 bp per side
    slippage_bps: float = 1.0
    output_dir: Path = field(default_factory=lambda: Path("data/backtest_out"))
    live_writer: Optional[Any] = None  # LiveStatusWriter (typed loosely to avoid hard dep)
    live_update_every: int = 3         # write live status every N bars

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
        open_trade: Optional[Dict[str, Any]] = None

        bars = history.iloc[window:]
        prior = history.iloc[:window]

        for ts, bar in bars.iterrows():
            feature_window = prior.tail(window)

            # Manage an existing open trade first.
            if open_trade is not None:
                hit_stop = (
                    bar["low"] <= open_trade["stop"]
                    if open_trade["direction"] == "long"
                    else bar["high"] >= open_trade["stop"]
                )
                hit_tp = (
                    bar["high"] >= open_trade["take_profit"]
                    if open_trade["direction"] == "long"
                    else bar["low"] <= open_trade["take_profit"]
                )
                exit_price: Optional[float] = None
                if hit_stop:
                    exit_price = open_trade["stop"]
                elif hit_tp:
                    exit_price = open_trade["take_profit"]

                if exit_price is not None:
                    pnl = self._compute_pnl(open_trade, exit_price)
                    equity += pnl
                    trades.append({**open_trade, "exit": float(exit_price),
                                   "exit_ts": ts, "pnl": pnl})
                    open_trade = None

            # Look for a new entry if we're flat.
            if open_trade is None:
                pred = self.predict_fn(feature_window)
                if pred and pred.get("confidence", 0) >= self.confidence_threshold:
                    direction = pred["direction"]
                    entry = float(bar["open"])
                    atr = self._atr(prior.tail(14))
                    if atr > 0:
                        stop = apply_stop_loss(entry, direction, atr)
                        tp = apply_take_profit(entry, stop, direction)
                        notional = calculate_position_size(
                            account_equity=equity,
                            confidence=pred["confidence"],
                            expected_return_pct=abs(pred.get("expected_return_pct", 0.01)),
                            max_loss_pct=abs((entry - stop) / max(entry, 1e-6)),
                        )
                        if notional > 0:
                            qty = notional / entry
                            open_trade = {
                                "entry_ts": ts,
                                "direction": direction,
                                "entry": entry,
                                "stop": stop,
                                "take_profit": tp,
                                "qty": qty,
                                "notional": notional,
                            }

            equity_path.append(equity)
            prior = pd.concat([prior.iloc[1:], history.loc[[ts]]])

            # Stream live status every N bars so the dashboard can show
            # the equity curve while the run is still in progress.
            bars_done = len(equity_path) - 1
            if self.live_writer is not None and (
                bars_done % self.live_update_every == 0
                or bars_done == len(bars)
            ):
                try:
                    open_summary = None
                    if open_trade is not None:
                        open_summary = {
                            "direction": open_trade["direction"],
                            "entry": float(open_trade["entry"]),
                            "stop": float(open_trade["stop"]),
                            "qty": float(open_trade["qty"]),
                        }
                    self.live_writer.update(
                        bars_processed=bars_done,
                        current_equity=equity,
                        trades_so_far=len(trades),
                        open_trade=open_summary,
                    )
                except Exception:  # noqa: BLE001
                    pass

            if monitor_drawdown(equity_path, threshold=0.25):
                logger.warning("Drawdown breach at %s — halting backtest", ts)
                break

        equity_series = pd.Series(equity_path[1:], index=bars.index[: len(equity_path) - 1])
        trades_df = pd.DataFrame(trades)
        summary = self._summarize(trades_df, equity_series)

        # Final live status update before returning.
        if self.live_writer is not None:
            try:
                self.live_writer.update(
                    bars_processed=len(equity_path) - 1,
                    current_equity=equity,
                    trades_so_far=len(trades),
                    open_trade=None,
                )
            except Exception:  # noqa: BLE001
                pass

        self._write_outputs(trades_df, equity_series)
        return BacktestResult(summary=summary, trades=trades_df, equity_curve=equity_series)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_pnl(self, trade: Dict[str, Any], exit_price: float) -> float:
        sign = 1.0 if trade["direction"] == "long" else -1.0
        gross = sign * (exit_price - trade["entry"]) * trade["qty"]
        fees = trade["notional"] * (self.fee_bps + self.slippage_bps) / 10_000.0 * 2
        return float(gross - fees)

    @staticmethod
    def _atr(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return float(tr.mean())

    def _summarize(self, trades: pd.DataFrame, equity: pd.Series) -> Dict[str, float]:
        if trades.empty:
            return {"trades": 0, "win_rate": 0.0, "avg_pnl": 0.0,
                    "sharpe": 0.0, "max_drawdown_pct": 0.0,
                    "final_equity": float(equity.iloc[-1] if not equity.empty else self.initial_equity)}
        wins = (trades["pnl"] > 0).sum()
        win_rate = float(wins / len(trades))
        avg_pnl = float(trades["pnl"].mean())
        returns = equity.pct_change().dropna()
        sharpe = float(np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)) if not returns.empty else 0.0
        running_max = equity.cummax()
        max_dd = float(((equity - running_max) / running_max).min())
        return {
            "trades": int(len(trades)),
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd,
            "final_equity": float(equity.iloc[-1]),
        }

    def _write_outputs(self, trades: pd.DataFrame, equity: pd.Series) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        equity.to_csv(self.output_dir / "equity_curve.csv", header=["equity"])
        trades.to_csv(self.output_dir / "trades.csv", index=False)
        logger.info("Backtest artifacts written to %s", self.output_dir)
