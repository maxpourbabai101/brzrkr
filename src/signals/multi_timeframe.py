"""
MultiTimeframeConfluence — weekly + daily timeframe alignment.

A daily buy signal confirmed by a weekly uptrend has a significantly
higher win rate than a daily signal alone. Triple confirmation
(monthly + weekly + daily) is the highest-conviction setup.

Logic:
  - Weekly: fetch 1-week bars, check 12-week momentum + EMA alignment
  - Monthly: fetch 1-month bars, check 6-month momentum
  - Daily: already have from prices arg — check 20-day EMA slope

If all 3 timeframes agree → boost=0.15 (highest conviction)
If 2/3 agree → boost=0.08
If 1/3 agree or disagree → no signal (or slight penalty)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

from src.signals.advanced_scanners import ScanResult

logger = logging.getLogger(__name__)


class MultiTimeframeConfluence:

    CACHE_TTL = 3600   # 1 hour — weekly/monthly data doesn't change fast

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[float, Optional[ScanResult]]] = {}
        self._lock = threading.RLock()

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(symbol)
        if cached and (now - cached[0]) < self.CACHE_TTL:
            return cached[1]

        result = self._do_scan(symbol, prices)
        with self._lock:
            self._cache[symbol] = (now, result)
        return result

    def _do_scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            import yfinance as yf

            close = prices["close"].astype(float)
            if len(close) < 20:
                return None

            signals: Dict[str, str] = {}   # timeframe → "long" | "short" | "neutral"
            details: Dict[str, str] = {}

            # ── Daily (from provided prices) ──────────────────────────────────
            ema20_d   = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50_d   = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else ema20_d
            mom20_d   = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0.0
            price     = float(close.iloc[-1])

            if price > ema20_d and ema20_d > ema50_d and mom20_d > 0:
                signals["daily"] = "long"
                details["daily"] = f"Daily: price>{ema20_d:.0f} EMA, +{mom20_d:.1f}%"
            elif price < ema20_d and ema20_d < ema50_d and mom20_d < 0:
                signals["daily"] = "short"
                details["daily"] = f"Daily: price<{ema20_d:.0f} EMA, {mom20_d:.1f}%"
            else:
                signals["daily"] = "neutral"

            # ── Weekly ────────────────────────────────────────────────────────
            try:
                wk = yf.download(symbol, period="1y", interval="1wk",
                                  progress=False, auto_adjust=True)
                if not wk.empty:
                    if isinstance(wk.columns[0], tuple):
                        wk.columns = [c[0].lower() for c in wk.columns]
                    else:
                        wk.columns = [c.lower() for c in wk.columns]
                    wk_close = wk["close"].dropna()
                    if len(wk_close) >= 12:
                        ema12_w = float(wk_close.ewm(span=12, adjust=False).mean().iloc[-1])
                        mom12_w = float((wk_close.iloc[-1] / wk_close.iloc[-13] - 1) * 100)
                        wk_price = float(wk_close.iloc[-1])
                        if wk_price > ema12_w and mom12_w > 2.0:
                            signals["weekly"] = "long"
                            details["weekly"] = f"Weekly: above EMA12, +{mom12_w:.1f}% 12wk"
                        elif wk_price < ema12_w and mom12_w < -2.0:
                            signals["weekly"] = "short"
                            details["weekly"] = f"Weekly: below EMA12, {mom12_w:.1f}% 12wk"
                        else:
                            signals["weekly"] = "neutral"
            except Exception:
                pass

            # ── Monthly ───────────────────────────────────────────────────────
            try:
                mo = yf.download(symbol, period="3y", interval="1mo",
                                  progress=False, auto_adjust=True)
                if not mo.empty:
                    if isinstance(mo.columns[0], tuple):
                        mo.columns = [c[0].lower() for c in mo.columns]
                    else:
                        mo.columns = [c.lower() for c in mo.columns]
                    mo_close = mo["close"].dropna()
                    if len(mo_close) >= 6:
                        mom6_m = float((mo_close.iloc[-1] / mo_close.iloc[-7] - 1) * 100)
                        ema6_m = float(mo_close.ewm(span=6, adjust=False).mean().iloc[-1])
                        mo_price = float(mo_close.iloc[-1])
                        if mo_price > ema6_m and mom6_m > 5.0:
                            signals["monthly"] = "long"
                            details["monthly"] = f"Monthly: +{mom6_m:.1f}% 6mo, above EMA6"
                        elif mo_price < ema6_m and mom6_m < -5.0:
                            signals["monthly"] = "short"
                            details["monthly"] = f"Monthly: {mom6_m:.1f}% 6mo, below EMA6"
                        else:
                            signals["monthly"] = "neutral"
            except Exception:
                pass

            if not signals:
                return None

            long_tf  = sum(1 for v in signals.values() if v == "long")
            short_tf = sum(1 for v in signals.values() if v == "short")
            n_tf     = len(signals)

            if long_tf == n_tf:   # all timeframes agree long
                boost = 0.15
                direction = "long"
                reason = "Triple TF confluence LONG: " + " | ".join(details.values())
            elif long_tf == n_tf - 1 and short_tf == 0:   # 2/3 long, 1 neutral
                boost = 0.08
                direction = "long"
                reason = "2/3 TF long: " + " | ".join(v for k, v in details.items() if signals[k] == "long")
            elif short_tf == n_tf:   # all timeframes agree short
                boost = -0.15
                direction = "short"
                reason = "Triple TF confluence SHORT: " + " | ".join(details.values())
            elif short_tf == n_tf - 1 and long_tf == 0:
                boost = -0.08
                direction = "short"
                reason = "2/3 TF short: " + " | ".join(v for k, v in details.items() if signals[k] == "short")
            else:
                return None

            return ScanResult(
                scanner="multi_timeframe", symbol=symbol,
                direction=direction, signal_boost=round(boost, 3), reason=reason,
                metadata={"tf_signals": signals, "long_tf": long_tf, "short_tf": short_tf},
            )

        except Exception as exc:
            logger.debug("MultiTimeframeConfluence scan failed for %s: %s", symbol, exc)
            return None
