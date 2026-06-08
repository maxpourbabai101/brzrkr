"""
VolumeProfileEngine — VWAP bands + Point of Control analysis.

Professional traders use volume profile to find:
  - VWAP: fair value intraday; price above = bullish bias
  - VWAP ±1σ / ±2σ bands: institutional buy/sell zones
  - Point of Control (POC): price level with highest volume
    traded — acts as magnet and support/resistance
  - Value Area High/Low (VAH/VAL): 70% of volume traded here

Signal:
  - Price > VWAP + POC support = strong long
  - Price < VWAP + POC resistance = strong short
  - Price bouncing off VWAP from above = mean-reversion long
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.signals.advanced_scanners import ScanResult

logger = logging.getLogger(__name__)


class VolumeProfileEngine:

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            close  = prices["close"].astype(float)
            high   = prices["high"].astype(float)
            low    = prices["low"].astype(float)
            volume = prices["volume"].astype(float)

            if len(close) < 20:
                return None

            price = float(close.iloc[-1])

            # ── VWAP + standard deviation bands ──────────────────────────────
            tp     = (high + low + close) / 3
            cum_tv = (tp * volume).cumsum()
            cum_v  = volume.cumsum().replace(0, np.nan)
            vwap   = cum_tv / cum_v

            # VWAP standard deviation
            variance  = ((tp - vwap) ** 2 * volume).cumsum() / cum_v
            vwap_std  = variance.apply(lambda x: max(x, 0) ** 0.5)

            vwap_val  = float(vwap.iloc[-1])
            std_val   = float(vwap_std.iloc[-1])

            band1_up  = vwap_val + 1 * std_val
            band1_dn  = vwap_val - 1 * std_val
            band2_up  = vwap_val + 2 * std_val
            band2_dn  = vwap_val - 2 * std_val

            # ── Point of Control (POC) ────────────────────────────────────────
            # Bin prices into 50 buckets, find the bucket with most volume
            n_bins = 50
            price_min = float(low.min())
            price_max = float(high.max())
            if price_max <= price_min:
                return None

            bins  = np.linspace(price_min, price_max, n_bins + 1)
            vol_arr = volume.values
            close_arr = close.values

            vol_by_bin = np.zeros(n_bins)
            for i, (p, v) in enumerate(zip(close_arr, vol_arr)):
                idx = np.searchsorted(bins, p, side="right") - 1
                idx = int(np.clip(idx, 0, n_bins - 1))
                vol_by_bin[idx] += v

            poc_idx = int(np.argmax(vol_by_bin))
            poc     = float((bins[poc_idx] + bins[poc_idx + 1]) / 2)

            # Value Area: accumulate from POC outward until 70% of volume covered
            total_vol = vol_by_bin.sum()
            target    = total_vol * 0.70
            accum     = vol_by_bin[poc_idx]
            lo_idx, hi_idx = poc_idx, poc_idx
            while accum < target and (lo_idx > 0 or hi_idx < n_bins - 1):
                add_lo = vol_by_bin[lo_idx - 1] if lo_idx > 0 else 0
                add_hi = vol_by_bin[hi_idx + 1] if hi_idx < n_bins - 1 else 0
                if add_lo >= add_hi:
                    lo_idx -= 1
                    accum  += add_lo
                else:
                    hi_idx += 1
                    accum  += add_hi

            val = float((bins[lo_idx] + bins[lo_idx + 1]) / 2)
            vah = float((bins[hi_idx] + bins[hi_idx + 1]) / 2)

            # ── Signal logic ──────────────────────────────────────────────────
            above_vwap = price > vwap_val
            above_poc  = price > poc
            near_vwap  = abs(price - vwap_val) / vwap_val < 0.005  # within 0.5%

            if above_vwap and above_poc and price < band1_up:
                boost = round(0.06 + (price - vwap_val) / (band1_up - vwap_val) * 0.06, 3)
                return ScanResult(
                    scanner="volume_profile", symbol=symbol,
                    direction="long", signal_boost=round(min(boost, 0.10), 3),
                    reason=f"Price above VWAP={vwap_val:.2f} and POC={poc:.2f}, in value area",
                    metadata={"vwap": round(vwap_val, 2), "poc": round(poc, 2),
                              "vah": round(vah, 2), "val": round(val, 2)},
                )
            if near_vwap and float(close.iloc[-2]) > vwap_val:
                # Bouncing off VWAP from above — mean reversion long
                return ScanResult(
                    scanner="volume_profile", symbol=symbol,
                    direction="long", signal_boost=0.05,
                    reason=f"VWAP bounce from above (VWAP={vwap_val:.2f})",
                    metadata={"vwap": round(vwap_val, 2), "poc": round(poc, 2)},
                )
            if not above_vwap and not above_poc and price > band1_dn:
                boost = round(-(0.06 + (vwap_val - price) / (vwap_val - band1_dn) * 0.06), 3)
                return ScanResult(
                    scanner="volume_profile", symbol=symbol,
                    direction="short", signal_boost=round(max(boost, -0.10), 3),
                    reason=f"Price below VWAP={vwap_val:.2f} and POC={poc:.2f}",
                    metadata={"vwap": round(vwap_val, 2), "poc": round(poc, 2)},
                )

            return None

        except Exception as exc:
            logger.debug("VolumeProfileEngine scan failed for %s: %s", symbol, exc)
            return None
