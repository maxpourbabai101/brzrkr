"""UniverseScanner — scores every symbol in the master watchlist.

Fetches recent OHLCV from yfinance for each symbol, computes RSI,
MACD, volume ratio, and momentum, then ranks symbols by signal
strength.  Outputs a list of ScoredSymbol records consumed by the
Research tab and the agent's dynamic universe expansion.

Asset classes covered
─────────────────────
  INDEX_ETF     — SPY, QQQ, IWM, DIA, VTI
  SECTOR_ETF    — XLF XLK XLV XLE XLU XLRE XLI XLB XLP XLY
  COMMODITY_ETF — GLD, SLV, USO, TLT, PDBC
  LEVERAGED_ETF — TQQQ, SQQQ, UPRO, SPXS, UVXY, VXX
  STOCK         — NVDA TSLA AAPL MSFT META AMD COIN MSTR AMZN GOOGL
  CRYPTO_ETF    — IBIT, GBTC (Bitcoin ETFs on Alpaca equities)
  SMALL_CAP     — dynamic, injected from sweep scanner alerts

Futures note: Alpaca's stock feed doesn't serve CME futures (/ES /NQ).
The closest proxy is SPY/QQQ for equity, USO for crude, GLD for gold.
Crypto perpetuals (BTC/USD) are available 24/7 through Alpaca's crypto
endpoint and are listed in a separate CRYPTO section.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Master watchlist ────────────────────────────────────────────────
UNIVERSE: Dict[str, str] = {
    # Index ETFs
    "SPY":  "INDEX_ETF",
    "QQQ":  "INDEX_ETF",
    "IWM":  "INDEX_ETF",   # Russell 2000 (small caps proxy)
    "DIA":  "INDEX_ETF",   # Dow Jones
    "VTI":  "INDEX_ETF",   # Total market
    # Sector ETFs
    "XLF":  "SECTOR_ETF",  # Financials
    "XLK":  "SECTOR_ETF",  # Technology
    "XLV":  "SECTOR_ETF",  # Health Care
    "XLE":  "SECTOR_ETF",  # Energy
    "XLU":  "SECTOR_ETF",  # Utilities
    "XLI":  "SECTOR_ETF",  # Industrials
    "XLB":  "SECTOR_ETF",  # Materials
    "XLP":  "SECTOR_ETF",  # Consumer Staples
    "XLY":  "SECTOR_ETF",  # Consumer Discretionary
    "XLRE": "SECTOR_ETF",  # Real Estate
    # Commodity / macro ETFs (futures proxies)
    "GLD":  "COMMODITY_ETF",  # Gold futures proxy
    "SLV":  "COMMODITY_ETF",  # Silver
    "USO":  "COMMODITY_ETF",  # Crude oil (/CL proxy)
    "UNG":  "COMMODITY_ETF",  # Natural gas (/NG proxy)
    "TLT":  "BOND_ETF",       # 20y Treasury (/ZB proxy)
    "IEF":  "BOND_ETF",       # 7-10y Treasury
    "HYG":  "BOND_ETF",       # High-yield credit
    # Volatility / fear gauges
    "VXX":  "VOLATILITY",    # VIX short-term futures
    "UVXY": "VOLATILITY",    # 1.5x VIX
    # Leveraged ETFs (high-beta tactical plays)
    "TQQQ": "LEVERAGED_ETF", # 3x QQQ
    "SQQQ": "LEVERAGED_ETF", # 3x inverse QQQ
    "UPRO": "LEVERAGED_ETF", # 3x SPY
    "SPXS": "LEVERAGED_ETF", # 3x inverse SPY
    # High-beta individual stocks
    "NVDA": "STOCK",
    "TSLA": "STOCK",
    "AAPL": "STOCK",
    "MSFT": "STOCK",
    "META": "STOCK",
    "AMD":  "STOCK",
    "AMZN": "STOCK",
    "GOOGL":"STOCK",
    "COIN": "STOCK",  # Crypto-correlated
    "MSTR": "STOCK",  # Bitcoin-correlated
    # Bitcoin ETFs (equity-accessible crypto exposure)
    "IBIT": "CRYPTO_ETF",
    "GBTC": "CRYPTO_ETF",
}

# Crypto perpetuals on Alpaca's crypto endpoint (traded 24/7)
CRYPTO_SYMBOLS: List[str] = ["BTC/USD", "ETH/USD", "SOL/USD"]

# Sessions Alpaca supports
SESSIONS = {
    "regular":    {"label": "Regular Market",   "hours": "09:30 – 16:00 ET"},
    "pre_market": {"label": "Pre-Market",        "hours": "04:00 – 09:30 ET"},
    "after_hours":{"label": "After-Hours",       "hours": "16:00 – 20:00 ET"},
    "crypto":     {"label": "Crypto (24/7)",     "hours": "00:00 – 24:00 UTC"},
}


# ── Result dataclass ─────────────────────────────────────────────────

@dataclass
class ScoredSymbol:
    symbol:      str
    asset_type:  str
    score:       float       # 0–100, higher = stronger setup
    signal:      str         # "bullish" | "bearish" | "neutral"
    rsi:         float
    macd_signal: str         # "cross_up" | "cross_down" | "flat"
    momentum_5d: float       # % change over 5 days
    vol_ratio:   float       # today's vol / 20d avg vol
    trend:       str         # "up" | "down" | "flat"
    note:        str         = ""
    scanned_at:  str         = field(default_factory=lambda:
                                  datetime.now(timezone.utc).isoformat())
    current_price: float     = 0.0
    change_1d_pct: float     = 0.0


# ── Scanner ─────────────────────────────────────────────────────────

class UniverseScanner:
    """Score every symbol in UNIVERSE and return ranked ScoredSymbol list."""

    def __init__(self, fast_mode: bool = True) -> None:
        """
        fast_mode=True  — fetch only 60 days of daily data (faster).
        fast_mode=False — fetch 252 days for richer regime context.
        """
        self.period = "3mo" if fast_mode else "1y"

    def scan(self, extra_symbols: Optional[List[str]] = None) -> List[ScoredSymbol]:
        """Run the full scan.  Returns list sorted by |score| descending."""
        symbols = dict(UNIVERSE)
        for sym in (extra_symbols or []):
            if sym not in symbols:
                symbols[sym] = "SMALL_CAP"

        # Also add recent sweep scanner hits
        for sym in self._load_sweep_hits():
            if sym not in symbols:
                symbols[sym] = "SMALL_CAP"

        results: List[ScoredSymbol] = []
        for sym, atype in symbols.items():
            try:
                scored = self._score_symbol(sym, atype)
                if scored is not None:
                    results.append(scored)
            except Exception as exc:
                logger.debug("UniverseScanner: skipping %s — %s", sym, exc)

        results.sort(key=lambda x: abs(x.score), reverse=True)
        return results

    # ── Per-symbol scoring ───────────────────────────────────────────

    def _score_symbol(self, symbol: str, asset_type: str) -> Optional[ScoredSymbol]:
        import yfinance as yf

        raw = yf.download(symbol, period=self.period, interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or len(raw) < 20:
            return None

        raw.columns = [c.lower() for c in raw.columns]
        close  = raw["close"].squeeze().astype(float)
        volume = raw["volume"].squeeze().astype(float)

        # ── Indicators
        rsi_val    = self._rsi(close, 14)
        macd_sig   = self._macd_signal(close)
        mom5       = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
        vol_ratio  = float(volume.iloc[-1] / volume.iloc[-21:-1].mean()) \
                         if len(volume) > 21 else 1.0
        trend      = self._trend(close)
        price      = float(close.iloc[-1])
        chg1d      = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) \
                         if len(close) > 1 else 0.0

        # ── Composite score  (-100 to +100; positive = bullish setup)
        score = 0.0
        # RSI contribution
        if rsi_val < 30:
            score += 30    # oversold = bullish setup
        elif rsi_val < 40:
            score += 15
        elif rsi_val > 70:
            score -= 30    # overbought
        elif rsi_val > 60:
            score -= 10

        # MACD contribution
        if macd_sig == "cross_up":
            score += 20
        elif macd_sig == "cross_down":
            score -= 20

        # Momentum
        score += max(min(mom5 * 3, 25), -25)

        # Volume surge (unusual volume = institutional interest)
        if vol_ratio > 2.0:
            score += 10
        elif vol_ratio > 1.5:
            score += 5

        # Trend
        if trend == "up":
            score += 10
        elif trend == "down":
            score -= 10

        score = max(-100.0, min(100.0, score))

        if score >= 15:
            signal = "bullish"
        elif score <= -15:
            signal = "bearish"
        else:
            signal = "neutral"

        note = self._note(rsi_val, macd_sig, mom5, vol_ratio, trend, asset_type)

        return ScoredSymbol(
            symbol       = symbol,
            asset_type   = asset_type,
            score        = round(score, 1),
            signal       = signal,
            rsi          = round(rsi_val, 1),
            macd_signal  = macd_sig,
            momentum_5d  = round(mom5, 2),
            vol_ratio    = round(vol_ratio, 2),
            trend        = trend,
            note         = note,
            current_price= round(price, 2),
            change_1d_pct= round(chg1d, 2),
        )

    # ── Technical helpers ────────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> float:
        delta = close.diff().dropna()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, 1e-9)
        rsi   = (100 - 100 / (1 + rs)).dropna()
        return float(rsi.iloc[-1]) if len(rsi) > 0 else 50.0

    @staticmethod
    def _macd_signal(close: pd.Series) -> str:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        prev_diff = float(macd.iloc[-2] - sig.iloc[-2]) if len(macd) > 2 else 0
        curr_diff = float(macd.iloc[-1] - sig.iloc[-1])
        if prev_diff <= 0 < curr_diff:
            return "cross_up"
        if prev_diff >= 0 > curr_diff:
            return "cross_down"
        return "flat"

    @staticmethod
    def _trend(close: pd.Series) -> str:
        if len(close) < 50:
            return "flat"
        sma20 = float(close.iloc[-20:].mean())
        sma50 = float(close.iloc[-50:].mean())
        price = float(close.iloc[-1])
        if price > sma20 > sma50:
            return "up"
        if price < sma20 < sma50:
            return "down"
        return "flat"

    @staticmethod
    def _note(rsi: float, macd: str, mom5: float,
              vol_ratio: float, trend: str, atype: str) -> str:
        parts = []
        if rsi < 30:
            parts.append("oversold RSI")
        elif rsi > 70:
            parts.append("overbought RSI")
        if macd == "cross_up":
            parts.append("MACD bullish cross")
        elif macd == "cross_down":
            parts.append("MACD bearish cross")
        if vol_ratio > 2.0:
            parts.append(f"vol surge {vol_ratio:.1f}×")
        if abs(mom5) > 5:
            parts.append(f"{mom5:+.1f}% 5d move")
        if atype == "LEVERAGED_ETF":
            parts.append("leveraged — 3× exposure")
        if atype in ("VOLATILITY",):
            parts.append("volatility product")
        return "  ·  ".join(parts) if parts else "no standout signal"

    # ── Sweep scanner integration ─────────────────────────────────────

    @staticmethod
    def _load_sweep_hits() -> List[str]:
        """Pull recent small-cap sweep alerts for dynamic inclusion."""
        try:
            p = Path("data/sweeps.jsonl")
            if not p.exists():
                return []
            import json
            syms = []
            for line in p.read_text().splitlines()[-50:]:
                try:
                    d = json.loads(line)
                    sym = d.get("symbol", "")
                    if sym and len(sym) <= 5:
                        syms.append(sym)
                except Exception:
                    pass
            return list(dict.fromkeys(syms))[:10]  # dedupe, cap at 10
        except Exception:
            return []
