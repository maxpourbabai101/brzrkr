"""
IVRankScanner — implied volatility rank signal.

IV Rank = (current_IV - IV_52w_low) / (IV_52w_high - IV_52w_low)

Interpretation:
  IV rank < 0.20 — options are cheap; market expects nothing; explosive
                   moves are underpriced. BUY directional setups.
  IV rank > 0.80 — options are expensive; fear is elevated; mean
                   reversion likely. FADE momentum, expect compression.
  0.20–0.80      — neutral zone; no additional IV signal.

IV is estimated from the nearest ATM option's implied vol using
yfinance options chains (free). Not perfect, but captures 80% of
the signal with 0% of the cost.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.signals.advanced_scanners import ScanResult

logger = logging.getLogger(__name__)


class IVRankScanner:
    """Implied volatility rank signal from yfinance options chains."""

    CACHE_TTL = 1800   # 30 min — IV changes slowly intraday

    def __init__(self) -> None:
        import threading
        self._cache: Dict[str, Tuple[float, Optional[ScanResult]]] = {}
        self._lock  = threading.RLock()

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
            ticker = yf.Ticker(symbol)
            exps   = ticker.options
            if not exps:
                return None

            current_price = float(prices["close"].iloc[-1])

            # Get ATM implied vol from nearest expiry
            chain   = ticker.option_chain(exps[0])
            calls   = chain.calls
            puts    = chain.puts

            # Find closest-to-ATM strike
            all_strikes = calls["strike"].values
            atm_idx     = int(np.argmin(np.abs(all_strikes - current_price)))
            atm_strike  = float(all_strikes[atm_idx])

            atm_call = calls[calls["strike"] == atm_strike]
            atm_put  = puts[puts["strike"]   == atm_strike]

            call_iv = float(atm_call["impliedVolatility"].iloc[0]) if not atm_call.empty else None
            put_iv  = float(atm_put["impliedVolatility"].iloc[0])  if not atm_put.empty  else None

            iv_vals = [v for v in [call_iv, put_iv] if v is not None and not np.isnan(v) and v > 0]
            if not iv_vals:
                return None

            current_iv = float(np.mean(iv_vals))

            # Build 52-week IV history proxy from historical vol
            close    = prices["close"].astype(float)
            log_rets = np.log(close / close.shift(1)).dropna()
            hv_series = log_rets.rolling(21).std() * np.sqrt(252)
            hv_series = hv_series.dropna()

            if len(hv_series) < 30:
                return None

            # Use realised vol distribution as IV proxy for rank calculation
            iv_52w_low  = float(hv_series.tail(252).quantile(0.05))
            iv_52w_high = float(hv_series.tail(252).quantile(0.95))

            # Map current IV onto the historical distribution
            iv_range = iv_52w_high - iv_52w_low
            if iv_range < 0.01:
                return None

            iv_rank = float(np.clip((current_iv - iv_52w_low) / iv_range, 0.0, 1.0))

            if iv_rank < 0.20:
                # Low IV: options cheap, market complacent — buy momentum setups
                boost = round(0.05 + (0.20 - iv_rank) * 0.30, 3)
                return ScanResult(
                    scanner="iv_rank", symbol=symbol,
                    direction="long", signal_boost=round(min(boost, 0.10), 3),
                    reason=f"IV Rank={iv_rank:.0%} — options cheap, expect breakout",
                    metadata={"iv_rank": round(iv_rank, 3), "current_iv": round(current_iv, 3)},
                )
            if iv_rank > 0.80:
                # High IV: fear is elevated, mean reversion coming
                boost = round(-(0.05 + (iv_rank - 0.80) * 0.25), 3)
                return ScanResult(
                    scanner="iv_rank", symbol=symbol,
                    direction="short", signal_boost=round(max(boost, -0.08), 3),
                    reason=f"IV Rank={iv_rank:.0%} — fear elevated, expect compression/fade",
                    metadata={"iv_rank": round(iv_rank, 3), "current_iv": round(current_iv, 3)},
                )

        except Exception as exc:
            logger.debug("IVRankScanner failed for %s: %s", symbol, exc)

        return None
