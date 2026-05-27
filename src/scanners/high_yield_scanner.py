"""High-Yield Scanner — surfaces the highest expected-return trade setups.

Covers ~66 symbols (roughly 3× the active agent universe) across every
liquid asset class accessible on Alpaca equities.  Each symbol is scored
on a *yield-focused* rubric that weights expected return magnitude and
risk/reward quality over simple directional confidence.

Yield score components (0–100):
  - Momentum quality     : rate-of-change over 5 / 10 / 20 bars
  - Breakout detection   : price vs recent ATH / consolidation range
  - R:R quality          : implied ATR-based stop vs price target
  - Volume confirmation  : abnormal volume as institutional signal
  - RSI / MACD momentum  : classic filter, but only adds when aligned
  - Volatility regime    : slight bonus for moderate vol (tradable swings)
  - Trend alignment      : SMA20 vs SMA50 vs SMA200

HunterSignal output schema::

    {
      "symbol":          "NVDA",
      "asset_type":      "STOCK",
      "direction":       "long" | "short",
      "entry_price":     891.50,
      "stop_price":      871.20,
      "target_price":    952.30,
      "rr_ratio":        3.02,
      "expected_yield_pct": 6.77,
      "yield_score":     84.2,
      "confidence":      0.81,
      "momentum_5d":     4.3,
      "momentum_20d":    11.2,
      "vol_ratio":       2.3,
      "atr":             18.6,
      "rsi":             58.4,
      "breakout":        "range_break" | "52wk_high" | "none",
      "note":            "...",
      "scanned_at":      "2026-05-26T...",
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expanded universe — ~66 symbols across all liquid asset classes
# ---------------------------------------------------------------------------
HIGH_YIELD_UNIVERSE: Dict[str, str] = {
    # ── Index ETFs (regime anchors + vol plays)
    "SPY":   "INDEX_ETF",
    "QQQ":   "INDEX_ETF",
    "IWM":   "INDEX_ETF",
    "DIA":   "INDEX_ETF",
    "VTI":   "INDEX_ETF",
    "MDY":   "INDEX_ETF",    # Mid-cap S&P 400

    # ── Sector ETFs (rotation plays)
    "XLK":   "SECTOR_ETF",   # Technology
    "XLF":   "SECTOR_ETF",   # Financials
    "XLV":   "SECTOR_ETF",   # Health Care
    "XLE":   "SECTOR_ETF",   # Energy
    "XLY":   "SECTOR_ETF",   # Consumer Discretionary
    "XLI":   "SECTOR_ETF",   # Industrials
    "XLB":   "SECTOR_ETF",   # Materials
    "XLP":   "SECTOR_ETF",   # Consumer Staples
    "XLU":   "SECTOR_ETF",   # Utilities
    "XLRE":  "SECTOR_ETF",   # Real Estate
    "XLC":   "SECTOR_ETF",   # Communication Services
    "ARKK":  "SECTOR_ETF",   # ARK Innovation (high-beta tech)
    "SOXX":  "SECTOR_ETF",   # Semiconductors
    "IBB":   "SECTOR_ETF",   # Biotech

    # ── Commodity / macro ETFs (futures proxies)
    "GLD":   "COMMODITY_ETF",
    "SLV":   "COMMODITY_ETF",
    "USO":   "COMMODITY_ETF",
    "UNG":   "COMMODITY_ETF",
    "PDBC":  "COMMODITY_ETF", # Diversified commodities
    "TLT":   "BOND_ETF",
    "IEF":   "BOND_ETF",
    "HYG":   "BOND_ETF",
    "LQD":   "BOND_ETF",      # Investment-grade corp bonds

    # ── Volatility products
    "VXX":   "VOLATILITY",
    "UVXY":  "VOLATILITY",

    # ── Leveraged ETFs (3× high-beta tactical)
    "TQQQ":  "LEVERAGED_ETF",
    "SQQQ":  "LEVERAGED_ETF",
    "UPRO":  "LEVERAGED_ETF",
    "SPXS":  "LEVERAGED_ETF",
    "LABU":  "LEVERAGED_ETF", # 3× biotech
    "SOXL":  "LEVERAGED_ETF", # 3× semiconductors
    "SOXS":  "LEVERAGED_ETF", # 3× inverse semiconductors
    "FNGU":  "LEVERAGED_ETF", # 3× FANG+
    "FNGD":  "LEVERAGED_ETF", # 3× inverse FANG+

    # ── Mega-cap / high-beta individual stocks
    "NVDA":  "STOCK",
    "TSLA":  "STOCK",
    "AAPL":  "STOCK",
    "MSFT":  "STOCK",
    "META":  "STOCK",
    "AMZN":  "STOCK",
    "GOOGL": "STOCK",
    "AMD":   "STOCK",
    "COIN":  "STOCK",
    "MSTR":  "STOCK",

    # ── High-momentum growth stocks
    "PLTR":  "STOCK",   # Palantir
    "CRWD":  "STOCK",   # CrowdStrike
    "ARM":   "STOCK",   # ARM Holdings
    "SMCI":  "STOCK",   # Super Micro Computer
    "HOOD":  "STOCK",   # Robinhood
    "SOFI":  "STOCK",   # SoFi Technologies
    "IONQ":  "STOCK",   # Quantum computing
    "RKLB":  "STOCK",   # Rocket Lab
    "ACHR":  "STOCK",   # Archer Aviation
    "JOBY":  "STOCK",   # Joby Aviation
    "PATH":  "STOCK",   # UiPath
    "SNOW":  "STOCK",   # Snowflake
    "NET":   "STOCK",   # Cloudflare
    "DDOG":  "STOCK",   # Datadog
    "GTLB":  "STOCK",   # GitLab
    "RXRX":  "STOCK",   # Recursion Pharma (AI bio)
    "DNA":   "STOCK",   # Ginkgo Bioworks

    # ── Bitcoin / crypto ETFs
    "IBIT":  "CRYPTO_ETF",
    "GBTC":  "CRYPTO_ETF",
    "FBTC":  "CRYPTO_ETF",  # Fidelity Bitcoin ETF
    "BITB":  "CRYPTO_ETF",  # Bitwise Bitcoin ETF
}

# R:R target for yield estimation (can be tuned per asset class)
_RR_TARGETS: Dict[str, float] = {
    "LEVERAGED_ETF": 3.5,
    "VOLATILITY":    3.0,
    "STOCK":         3.0,
    "CRYPTO_ETF":    3.5,
    "SECTOR_ETF":    2.5,
    "INDEX_ETF":     2.5,
    "COMMODITY_ETF": 2.5,
    "BOND_ETF":      2.0,
}
DEFAULT_RR_TARGET = 2.5


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class HunterSignal:
    symbol:           str
    asset_type:       str
    direction:        str          # "long" | "short"
    entry_price:      float
    stop_price:       float
    target_price:     float
    rr_ratio:         float
    expected_yield_pct: float      # (target - entry) / entry × 100
    yield_score:      float        # 0–100, primary ranking key
    confidence:       float        # 0–1, directional conviction
    momentum_5d:      float        # % change over 5 bars
    momentum_20d:     float        # % change over 20 bars
    vol_ratio:        float        # today vol / 20d avg
    atr:              float        # average true range (dollars)
    rsi:              float
    breakout:         str          # "52wk_high" | "range_break" | "none"
    note:             str          = ""
    scanned_at:       str          = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class HighYieldScanner:
    """Scan the expanded universe and return HunterSignals ranked by yield."""

    # How many trading days of history to pull
    _PERIOD = "6mo"

    def __init__(self, min_yield_score: float = 40.0,
                 top_n: Optional[int] = None) -> None:
        """
        min_yield_score : drop signals below this threshold (0–100).
        top_n           : if set, return only the top-N ranked signals.
        """
        self.min_yield_score = min_yield_score
        self.top_n = top_n

    def scan(self, extra_symbols: Optional[List[str]] = None) -> List[HunterSignal]:
        """Run the full universe scan.  Returns list sorted by yield_score desc."""
        universe = dict(HIGH_YIELD_UNIVERSE)
        for sym in (extra_symbols or []):
            if sym not in universe:
                universe[sym] = "STOCK"

        results: List[HunterSignal] = []
        for sym, atype in universe.items():
            try:
                sig = self._score_symbol(sym, atype)
                if sig is not None and sig.yield_score >= self.min_yield_score:
                    results.append(sig)
            except Exception as exc:
                logger.debug("HighYieldScanner: skipping %s — %s", sym, exc)

        results.sort(key=lambda s: s.yield_score, reverse=True)

        if self.top_n is not None:
            results = results[: self.top_n]

        logger.info(
            "HighYieldScanner: %d symbols scanned, %d signals above min_score=%.0f",
            len(universe), len(results), self.min_yield_score,
        )
        return results

    # ------------------------------------------------------------------ #
    #  Per-symbol scoring                                                  #
    # ------------------------------------------------------------------ #

    def _score_symbol(self, symbol: str, asset_type: str) -> Optional[HunterSignal]:
        import yfinance as yf

        raw = yf.download(symbol, period=self._PERIOD, interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or len(raw) < 30:
            return None

        raw.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                       for c in raw.columns]

        close  = raw["close"].squeeze().astype(float)
        high   = raw["high"].squeeze().astype(float)
        low    = raw["low"].squeeze().astype(float)
        volume = raw["volume"].squeeze().astype(float)

        # ── Indicators ────────────────────────────────────────────────
        atr_val     = self._atr(high, low, close, 14)
        rsi_val     = self._rsi(close, 14)
        macd_sig    = self._macd_signal(close)
        mom5        = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
        mom10       = float((close.iloc[-1] / close.iloc[-11] - 1) * 100) \
                          if len(close) > 11 else mom5
        mom20       = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) \
                          if len(close) > 21 else mom10
        vol_ratio   = float(volume.iloc[-1] / volume.iloc[-21:-1].mean()) \
                          if len(volume) > 21 else 1.0
        breakout    = self._detect_breakout(close, high)
        trend       = self._trend(close)
        price       = float(close.iloc[-1])

        # ── Directional bias ──────────────────────────────────────────
        bull_signals = sum([
            mom5 > 0, mom20 > 0,
            rsi_val > 50,
            macd_sig == "cross_up",
            trend == "up",
        ])
        bear_signals = sum([
            mom5 < 0, mom20 < 0,
            rsi_val < 50,
            macd_sig == "cross_down",
            trend == "down",
        ])
        direction = "long" if bull_signals >= bear_signals else "short"
        directional_agreement = max(bull_signals, bear_signals) / 5.0

        # ── R:R geometry ──────────────────────────────────────────────
        atr_mult_stop = 1.5 if asset_type in ("LEVERAGED_ETF", "VOLATILITY") else 2.0
        rr_target = _RR_TARGETS.get(asset_type, DEFAULT_RR_TARGET)

        if direction == "long":
            stop_price   = max(price - atr_mult_stop * atr_val, 0.01)
            target_price = price + rr_target * atr_mult_stop * atr_val
        else:
            stop_price   = price + atr_mult_stop * atr_val
            # target can't go below $0.01 for short plays on low-priced symbols
            target_price = max(price - rr_target * atr_mult_stop * atr_val, 0.01)

        risk_dist = abs(price - stop_price)
        rw_dist   = abs(target_price - price)
        rr_ratio  = rw_dist / max(risk_dist, 1e-6)

        if direction == "long":
            expected_yield_pct = (target_price - price) / max(price, 1e-6) * 100
        else:
            expected_yield_pct = (price - target_price) / max(price, 1e-6) * 100

        # ── Yield score (0–100) ───────────────────────────────────────
        score = 0.0

        # 1. Momentum quality (up to 30 pts)
        mom_score = min(abs(mom5) * 2.5, 15) + min(abs(mom20) * 1.0, 15)
        # Only credit if momentum aligns with direction
        if (direction == "long" and mom5 > 0) or (direction == "short" and mom5 < 0):
            score += mom_score
        else:
            score += mom_score * 0.2   # partial credit for counter-trend setups

        # 2. Directional agreement across indicators (up to 20 pts)
        score += directional_agreement * 20

        # 3. Volume confirmation (up to 15 pts)
        if vol_ratio >= 3.0:
            score += 15
        elif vol_ratio >= 2.0:
            score += 10
        elif vol_ratio >= 1.5:
            score += 5

        # 4. Breakout bonus (up to 15 pts)
        if breakout == "52wk_high" and direction == "long":
            score += 15
        elif breakout == "range_break":
            score += 10
        elif breakout == "52wk_low" and direction == "short":
            score += 15

        # 5. R:R quality (up to 10 pts)
        rr_score = min((rr_ratio - 1.5) / 2.0, 1.0) * 10  # best at rr >= 3.5
        score += max(rr_score, 0)

        # 6. Expected yield magnitude (up to 10 pts)
        score += min(expected_yield_pct * 1.5, 10)

        # 7. RSI zone bonus (moderate RSI = room to run; up to 0–5 pts)
        if direction == "long" and 40 <= rsi_val <= 60:
            score += 5   # not overbought
        elif direction == "short" and 40 <= rsi_val <= 60:
            score += 5   # not oversold
        elif direction == "long" and rsi_val < 30:
            score += 3   # deep oversold can snap
        elif direction == "short" and rsi_val > 70:
            score += 3

        score = round(max(0.0, min(100.0, score)), 1)

        # ── Confidence derived from directional agreement + breakout ──
        confidence = min(
            directional_agreement * 0.7
            + (0.15 if breakout != "none" else 0.0)
            + min(vol_ratio / 10.0, 0.15),
            1.0,
        )

        note = self._build_note(
            direction, rsi_val, macd_sig, mom5, mom20,
            vol_ratio, breakout, trend, asset_type, rr_ratio,
        )

        return HunterSignal(
            symbol            = symbol,
            asset_type        = asset_type,
            direction         = direction,
            entry_price       = round(price, 4),
            stop_price        = round(stop_price, 4),
            target_price      = round(target_price, 4),
            rr_ratio          = round(rr_ratio, 2),
            expected_yield_pct= round(expected_yield_pct, 2),
            yield_score       = score,
            confidence        = round(confidence, 3),
            momentum_5d       = round(mom5, 2),
            momentum_20d      = round(mom20, 2),
            vol_ratio         = round(vol_ratio, 2),
            atr               = round(atr_val, 4),
            rsi               = round(rsi_val, 1),
            breakout          = breakout,
            note              = note,
        )

    # ------------------------------------------------------------------ #
    #  Technical helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> float:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().dropna()
        return float(atr.iloc[-1]) if len(atr) > 0 else float((high - low).mean())

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
        prev  = float(macd.iloc[-2] - sig.iloc[-2]) if len(macd) > 2 else 0.0
        curr  = float(macd.iloc[-1] - sig.iloc[-1])
        if prev <= 0 < curr:
            return "cross_up"
        if prev >= 0 > curr:
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
    def _detect_breakout(close: pd.Series, high: pd.Series) -> str:
        """Classify whether price is at a 52wk high, breaking a range, or flat."""
        if len(close) < 20:
            return "none"
        price     = float(close.iloc[-1])
        high_52wk = float(high.max())
        # Within 0.5% of 52-week high
        if price >= high_52wk * 0.995:
            return "52wk_high"
        # Low of the 52-week cycle
        low_52wk  = float(close.min())
        if price <= low_52wk * 1.005:
            return "52wk_low"
        # Range break: price > 20-day high (excluding today)
        recent_high = float(high.iloc[-21:-1].max()) if len(high) > 21 else float(high.max())
        if price > recent_high * 1.002:
            return "range_break"
        return "none"

    @staticmethod
    def _build_note(direction: str, rsi: float, macd: str,
                    mom5: float, mom20: float, vol_ratio: float,
                    breakout: str, trend: str, atype: str,
                    rr_ratio: float) -> str:
        parts = []
        if breakout == "52wk_high":
            parts.append("52-wk high breakout")
        elif breakout == "52wk_low":
            parts.append("52-wk low — short setup")
        elif breakout == "range_break":
            parts.append("20d range break")
        if vol_ratio >= 2.0:
            parts.append(f"vol surge {vol_ratio:.1f}×")
        if abs(mom5) > 3:
            parts.append(f"{mom5:+.1f}% 5d")
        if abs(mom20) > 8:
            parts.append(f"{mom20:+.1f}% 20d")
        if macd in ("cross_up", "cross_down"):
            parts.append(f"MACD {macd.replace('_', ' ')}")
        if rsi < 30:
            parts.append("oversold RSI")
        elif rsi > 70:
            parts.append("overbought RSI")
        if atype == "LEVERAGED_ETF":
            parts.append("3× leveraged")
        parts.append(f"R:R {rr_ratio:.1f}×")
        return "  ·  ".join(parts) if parts else f"R:R {rr_ratio:.1f}×"
