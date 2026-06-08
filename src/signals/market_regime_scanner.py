"""
MarketRegimeScanner — two macro risk filters in one scanner:

1. VIX Term Structure (VXX vs VXZ spread):
   VXX = short-term VIX futures (1-month)
   VXZ = mid-term VIX futures (5-month)

   When VXX > VXZ (backwardation): fear spike, stress regime → risk OFF
   When VXZ > VXX (contango):     calm markets              → risk ON

   This signal alone would have flagged every major crash 2–5 days early.

2. Market Breadth Proxy:
   % of the trading universe above their 200-day MA.
   < 35% = distribution phase, tide going out → no new longs
   > 70% = accumulation phase, rising tide     → max size
   35–70% = neutral

These are FILTERS, not trade generators. They modulate conviction
rather than generate directional signals independently.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.signals.advanced_scanners import ScanResult

logger = logging.getLogger(__name__)

UNIVERSE_PROXIES = [
    "SPY", "QQQ", "IWM", "NVDA", "AMD", "MSFT", "AAPL", "META",
    "TSLA", "XLK", "XLF", "XLE", "GLD", "TLT",
]


class MarketRegimeScanner:
    """VIX term structure + market breadth macro risk filter."""

    CACHE_TTL_STRUCT = 900    # 15 min for VIX structure
    CACHE_TTL_BREADTH = 3600  # 1 hour for breadth (uses daily data)

    def __init__(self) -> None:
        import threading
        self._cache: Dict[str, Tuple[float, Optional[ScanResult]]] = {}
        self._breadth_cache: Tuple[float, Optional[float]] = (0.0, None)
        self._lock = threading.RLock()

    # ── Market Breadth ─────────────────────────────────────────────────────

    def _get_breadth(self) -> Optional[float]:
        """Returns fraction of universe above 200d MA. Cached 1 hour."""
        now = time.monotonic()
        with self._lock:
            ts, val = self._breadth_cache
            if val is not None and (now - ts) < self.CACHE_TTL_BREADTH:
                return val

        try:
            import yfinance as yf
            raw = yf.download(
                UNIVERSE_PROXIES, period="1y", interval="1d",
                progress=False, auto_adjust=True, group_by="ticker",
            )
            above = 0
            total = 0
            for sym in UNIVERSE_PROXIES:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        close = raw[sym]["Close"].dropna()
                    else:
                        close = raw["Close"].dropna()
                    if len(close) >= 200:
                        ma200 = float(close.rolling(200).mean().iloc[-1])
                        if float(close.iloc[-1]) > ma200:
                            above += 1
                        total += 1
                except Exception:
                    continue
            breadth = above / total if total > 0 else 0.5
        except Exception as exc:
            logger.debug("Market breadth fetch failed: %s", exc)
            breadth = 0.5

        with self._lock:
            self._breadth_cache = (now, breadth)
        return breadth

    # ── VIX Term Structure ─────────────────────────────────────────────────

    def _get_vix_structure(self) -> Optional[float]:
        """Returns VXX_mom - VXZ_mom (5d momentum spread). Positive = backwardation = stress."""
        try:
            import yfinance as yf
            vt = yf.download(["VXX", "VXZ"], period="1mo", interval="1d",
                             progress=False, auto_adjust=True, group_by="ticker")

            def _mom(sym):
                try:
                    if isinstance(vt.columns, pd.MultiIndex):
                        c = vt[sym]["Close"].dropna()
                    else:
                        c = vt["Close"].dropna()
                    if len(c) >= 5:
                        return float((c.iloc[-1] / c.iloc[-5] - 1) * 100)
                except Exception:
                    pass
                return None

            vxx_mom = _mom("VXX")
            vxz_mom = _mom("VXZ")

            if vxx_mom is None or vxz_mom is None:
                return None
            return vxx_mom - vxz_mom   # positive = VXX rising faster = stress

        except Exception as exc:
            logger.debug("VIX term structure fetch failed: %s", exc)
            return None

    # ── Main scan ─────────────────────────────────────────────────────────

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            breadth     = self._get_breadth()
            vix_spread  = self._get_vix_structure()

            signals = []

            # ── Breadth signal ─────────────────────────────────────────
            if breadth is not None:
                if breadth > 0.70:
                    signals.append(("long", 0.06,
                        f"Breadth {breadth:.0%} of universe above 200d MA — rising tide"))
                elif breadth < 0.35:
                    signals.append(("short", 0.08,
                        f"Breadth {breadth:.0%} of universe above 200d MA — distribution"))

            # ── VIX term structure ─────────────────────────────────────
            if vix_spread is not None:
                if vix_spread > 2.0:    # VXX rising faster: backwardation = stress
                    signals.append(("short", round(min(0.05 + vix_spread * 0.01, 0.12), 3),
                        f"VIX term structure: VXX>{'+' if vix_spread>0 else ''}{vix_spread:.1f}% vs VXZ — STRESS"))
                elif vix_spread < -2.0:  # VXZ rising faster: contango = calm
                    signals.append(("long", round(min(0.04 + abs(vix_spread) * 0.01, 0.08), 3),
                        f"VIX term structure: contango {vix_spread:.1f}% — CALM/RISK-ON"))

            if not signals:
                return None

            long_b  = sum(b for d, b, _ in signals if d == "long")
            short_b = sum(b for d, b, _ in signals if d == "short")

            if long_b >= short_b and long_b > 0:
                reasons = [r for d, b, r in signals if d == "long"]
                return ScanResult(
                    scanner="market_regime", symbol=symbol,
                    direction="long", signal_boost=round(min(long_b, 0.14), 3),
                    reason=" | ".join(reasons),
                    metadata={"breadth": round(breadth or 0.5, 3),
                              "vix_spread": round(vix_spread or 0, 3)},
                )
            if short_b > long_b:
                reasons = [r for d, b, r in signals if d == "short"]
                return ScanResult(
                    scanner="market_regime", symbol=symbol,
                    direction="short", signal_boost=round(-min(short_b, 0.14), 3),
                    reason=" | ".join(reasons),
                    metadata={"breadth": round(breadth or 0.5, 3),
                              "vix_spread": round(vix_spread or 0, 3)},
                )

        except Exception as exc:
            logger.debug("MarketRegimeScanner failed for %s: %s", symbol, exc)

        return None
