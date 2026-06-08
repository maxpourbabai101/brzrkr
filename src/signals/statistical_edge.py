"""
StatisticalEdge — pure quantitative signals that detect trending/mean-reverting
regimes and generate entry signals accordingly.

Signals:
  1. Hurst Exponent (>0.55 = persistent trend = ride it)
  2. Return autocorrelation (positive lag-1 = momentum persists)
  3. Relative strength vs SPY (outperforming benchmark = institutional favour)
  4. Volume accumulation Z-score (unusual vol + up price = institutions buying)
  5. Seasonal / calendar alpha (FOMC week, earnings season, Jan effect)
  6. Z-score mean reversion (extreme deviation from mean = revert signal)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.signals.advanced_scanners import ScanResult

logger = logging.getLogger(__name__)


def _hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Estimate Hurst exponent via R/S analysis.
    H > 0.55: trending (persistent). H < 0.45: mean-reverting. ~0.5: random walk.
    """
    lags = range(2, min(max_lag, len(series) // 4))
    tau  = []
    for lag in lags:
        pp = np.subtract(series.values[lag:], series.values[:-lag])
        tau.append(np.sqrt(np.std(pp)))
    if len(tau) < 2:
        return 0.5
    try:
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(poly[0])
    except Exception:
        return 0.5


class StatisticalEdge:
    """Pure statistical signals: trend persistence, momentum, relative strength."""

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            close  = prices["close"].astype(float)
            volume = prices["volume"].astype(float)

            if len(close) < 60:
                return None

            price = float(close.iloc[-1])
            votes: list[tuple[str, float, str]] = []  # (direction, boost, reason)

            # ── 1. Hurst Exponent ─────────────────────────────────────────
            h = _hurst_exponent(close.tail(100))
            if h > 0.58:
                # Persistent trend — determine direction from recent momentum
                mom20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 21 else 0.0
                if mom20 > 0:
                    boost = round(min(0.06 + (h - 0.55) * 0.30, 0.14), 3)
                    votes.append(("long", boost, f"Hurst={h:.2f} persistent uptrend (mom={mom20:+.1f}%)"))
                elif mom20 < 0:
                    boost = round(min(0.06 + (h - 0.55) * 0.30, 0.14), 3)
                    votes.append(("short", boost, f"Hurst={h:.2f} persistent downtrend (mom={mom20:+.1f}%)"))
            elif h < 0.45:
                # Mean-reverting — trade the reversion
                zscore_20 = float((close.iloc[-1] - close.tail(20).mean()) / (close.tail(20).std() + 1e-9))
                if zscore_20 < -1.5:
                    votes.append(("long", 0.07, f"Hurst={h:.2f} mean-reverting, z={zscore_20:.2f} oversold"))
                elif zscore_20 > 1.5:
                    votes.append(("short", 0.07, f"Hurst={h:.2f} mean-reverting, z={zscore_20:.2f} overbought"))

            # ── 2. Return Autocorrelation (lag-1 momentum) ─────────────
            rets = close.pct_change().dropna()
            if len(rets) >= 20:
                lag1_ac = float(rets.tail(20).autocorr(lag=1))
                if lag1_ac > 0.15:
                    votes.append(("long", round(lag1_ac * 0.20, 3),
                        f"AC lag-1={lag1_ac:.2f} positive momentum persistence"))
                elif lag1_ac < -0.15:
                    votes.append(("short", round(abs(lag1_ac) * 0.20, 3),
                        f"AC lag-1={lag1_ac:.2f} negative momentum persistence"))

            # ── 3. Relative Strength vs SPY ────────────────────────────
            try:
                import yfinance as yf
                spy = yf.download("SPY", period="3mo", interval="1d",
                                  progress=False, auto_adjust=True)
                if not spy.empty and symbol != "SPY":
                    if isinstance(spy.columns[0], tuple):
                        spy.columns = [c[0].lower() for c in spy.columns]
                    else:
                        spy.columns = [c.lower() for c in spy.columns]
                    spy_close = spy["close"].dropna()
                    n = min(len(spy_close), len(close), 20)
                    sym_ret = float((close.iloc[-1] / close.iloc[-n] - 1))
                    spy_ret = float((spy_close.iloc[-1] / spy_close.iloc[-n] - 1))
                    rs = sym_ret - spy_ret
                    if rs > 0.03:   # outperforming by 3%+
                        boost = round(min(0.04 + rs * 0.50, 0.12), 3)
                        votes.append(("long", boost,
                            f"RS vs SPY={rs:+.1%} over {n}d — institutional favourite"))
                    elif rs < -0.03:
                        boost = round(min(0.04 + abs(rs) * 0.50, 0.12), 3)
                        votes.append(("short", boost,
                            f"RS vs SPY={rs:+.1%} over {n}d — laggard/distribution"))
            except Exception:
                pass

            # ── 4. Volume Accumulation Z-score ─────────────────────────
            if len(volume) >= 21:
                vol_mean = float(volume.tail(21).iloc[:-1].mean())
                vol_std  = float(volume.tail(21).iloc[:-1].std()) + 1e-9
                vol_z    = float((volume.iloc[-1] - vol_mean) / vol_std)
                price_up = float(close.iloc[-1]) > float(close.iloc[-2])
                if vol_z > 2.0 and price_up:
                    boost = round(min(0.04 + (vol_z - 2.0) * 0.02, 0.10), 3)
                    votes.append(("long", boost,
                        f"Volume Z={vol_z:.1f}σ above avg + price up = accumulation"))
                elif vol_z > 2.0 and not price_up:
                    boost = round(min(0.04 + (vol_z - 2.0) * 0.02, 0.10), 3)
                    votes.append(("short", boost,
                        f"Volume Z={vol_z:.1f}σ above avg + price down = distribution"))

            # ── 5. Seasonal / Calendar Alpha ───────────────────────────
            today = datetime.now(timezone.utc)
            month = today.month
            week  = today.isocalendar()[1]

            # "Best 6 months" (Nov–Apr): statistically better returns for equities
            if month in (11, 12, 1, 2, 3, 4) and symbol in ("SPY", "QQQ", "DIA", "IWM", "TQQQ"):
                votes.append(("long", 0.04,
                    f"Seasonal: 'best 6 months' window (Nov–Apr) for {symbol}"))

            # January small-cap effect
            if month == 1 and symbol in ("IWM",):
                votes.append(("long", 0.05, "January small-cap seasonal effect (IWM)"))

            # Earnings season (Jan, Apr, Jul, Oct weeks 1-4): market tends to drift up
            if month in (1, 4, 7, 10) and symbol in ("SPY", "QQQ"):
                votes.append(("long", 0.03, f"Earnings season drift (month={month})"))

            # ── 6. Z-score vs 50-day mean (overbought / oversold) ──────
            if len(close) >= 50:
                mean50 = float(close.tail(50).mean())
                std50  = float(close.tail(50).std()) + 1e-9
                z50    = float((price - mean50) / std50)
                if -2.5 < z50 < -1.5:
                    votes.append(("long", 0.06,
                        f"Z-50d={z50:.2f}: modestly oversold vs 50d distribution"))
                elif 1.5 < z50 < 2.5:
                    votes.append(("short", 0.05,
                        f"Z-50d={z50:.2f}: modestly overbought vs 50d distribution"))

            if not votes:
                return None

            long_boost  = sum(b for d, b, _ in votes if d == "long")
            short_boost = sum(b for d, b, _ in votes if d == "short")

            if long_boost > short_boost and long_boost > 0.03:
                reasons = [r for d, b, r in votes if d == "long"][:3]
                return ScanResult(
                    scanner="statistical_edge", symbol=symbol,
                    direction="long",
                    signal_boost=round(min(long_boost, 0.22), 3),
                    reason="; ".join(reasons),
                    metadata={"hurst": round(h, 3), "long_boost": round(long_boost, 3),
                              "short_boost": round(short_boost, 3)},
                )
            if short_boost > long_boost and short_boost > 0.03:
                reasons = [r for d, b, r in votes if d == "short"][:3]
                return ScanResult(
                    scanner="statistical_edge", symbol=symbol,
                    direction="short",
                    signal_boost=round(-min(short_boost, 0.22), 3),
                    reason="; ".join(reasons),
                    metadata={"hurst": round(h, 3), "long_boost": round(long_boost, 3),
                              "short_boost": round(short_boost, 3)},
                )

        except Exception as exc:
            logger.debug("StatisticalEdge scan failed for %s: %s", symbol, exc)

        return None
