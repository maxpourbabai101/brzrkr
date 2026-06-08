"""
SectorRotationTracker — follow institutional money as it rotates sectors.

Tracks 11 GICS sector ETFs relative to SPY.
When a sector is in the top-3 by 20-day relative strength AND
a symbol is in that sector, generate a long signal.
When a sector is in the bottom-3, generate a short/avoid signal.

Sector → Universe mapping tells which traded symbols belong to each sector.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.signals.advanced_scanners import ScanResult

logger = logging.getLogger(__name__)

SECTOR_ETFS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLU",   # Utilities
    "XLP",   # Consumer Staples
    "XLY",   # Consumer Discretionary
    "XLC",   # Communication Services
]

# Map universe symbols to their primary sector ETF
SYMBOL_TO_SECTOR: Dict[str, str] = {
    "NVDA": "XLK", "AMD":  "XLK", "MSFT": "XLK",
    "AAPL": "XLK", "META": "XLC", "TSLA": "XLY",
    "XLK":  "XLK", "XLF":  "XLF", "XLE":  "XLE",
    "XLV":  "XLV", "GLD":  "XLB", "SLV":  "XLB",
    "USO":  "XLE", "TLT":  "XLU", "TQQQ": "XLK",
    "SQQQ": "XLK", "SPY":  None,  "QQQ":  "XLK",
    "DIA":  None,  "IWM":  None,  "IBIT": "XLK",
    "GBTC": "XLK",
}


class SectorRotationTracker:
    """
    Downloads sector ETF returns, ranks them, and signals whether
    the symbol's sector is getting inflows or outflows.
    """

    CACHE_TTL   = 1800   # 30 min — sector rotation is slow

    def __init__(self) -> None:
        self._cache_ts: float = 0.0
        self._rankings: Optional[List[Tuple[str, float]]] = None   # [(etf, mom), ...]
        self._lock = threading.RLock()

    def _refresh_rankings(self) -> None:
        now = time.monotonic()
        with self._lock:
            if self._rankings is not None and (now - self._cache_ts) < self.CACHE_TTL:
                return
        try:
            import yfinance as yf
            raw = yf.download(
                SECTOR_ETFS, period="3mo", interval="1d",
                progress=False, auto_adjust=True, group_by="ticker",
            )
            rankings: List[Tuple[str, float]] = []
            for etf in SECTOR_ETFS:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        close = raw[etf]["Close"].dropna()
                    else:
                        close = raw["Close"].dropna()
                    if len(close) < 20:
                        continue
                    mom20 = float((close.iloc[-1] / close.iloc[-20] - 1) * 100)
                    rankings.append((etf, mom20))
                except Exception:
                    continue
            with self._lock:
                self._rankings = sorted(rankings, key=lambda x: x[1], reverse=True)
                self._cache_ts = now
        except Exception as exc:
            logger.debug("SectorRotation refresh failed: %s", exc)

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            sector_etf = SYMBOL_TO_SECTOR.get(symbol)
            if sector_etf is None:
                return None

            self._refresh_rankings()
            if not self._rankings:
                return None

            n = len(self._rankings)
            etf_names = [e for e, _ in self._rankings]

            if sector_etf not in etf_names:
                return None

            rank = etf_names.index(sector_etf)   # 0 = strongest
            mom  = self._rankings[rank][1]

            top3    = rank < 3
            bottom3 = rank >= n - 3

            if top3 and mom > 1.0:
                boost = round(min(0.04 + mom * 0.01 + (2 - rank) * 0.02, 0.12), 3)
                return ScanResult(
                    scanner="sector_rotation", symbol=symbol,
                    direction="long", signal_boost=boost,
                    reason=f"Sector {sector_etf} rank #{rank+1}/{n} (mom={mom:+.1f}%) — inflow",
                    metadata={"sector": sector_etf, "rank": rank + 1, "momentum_20d": round(mom, 2)},
                )
            if bottom3 and mom < -1.0:
                boost = round(max(-(0.04 + abs(mom) * 0.01 + (rank - (n - 3)) * 0.02), -0.12), 3)
                return ScanResult(
                    scanner="sector_rotation", symbol=symbol,
                    direction="short", signal_boost=boost,
                    reason=f"Sector {sector_etf} rank #{rank+1}/{n} (mom={mom:+.1f}%) — outflow",
                    metadata={"sector": sector_etf, "rank": rank + 1, "momentum_20d": round(mom, 2)},
                )

            return None

        except Exception as exc:
            logger.debug("SectorRotation scan failed for %s: %s", symbol, exc)
            return None
