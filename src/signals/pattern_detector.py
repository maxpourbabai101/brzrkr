"""
PricePatternDetector — candlestick + chart pattern recognition.

Candlestick patterns (single/multi-bar):
  - Bullish Engulfing   — strong reversal up
  - Hammer / Pin Bar    — rejection of lower prices
  - Morning Star        — 3-bar bullish reversal
  - Bearish Engulfing   — strong reversal down
  - Shooting Star       — rejection of higher prices
  - Evening Star        — 3-bar bearish reversal

Chart patterns:
  - Bull Flag           — consolidation after strong up-move
  - Higher Highs/Lows   — confirmed uptrend structure
  - Lower Highs/Lows    — confirmed downtrend structure
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.signals.advanced_scanners import ScanResult

logger = logging.getLogger(__name__)


class PricePatternDetector:

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            close  = prices["close"].astype(float)
            open_  = prices["open"].astype(float)
            high   = prices["high"].astype(float)
            low    = prices["low"].astype(float)
            volume = prices["volume"].astype(float)

            if len(close) < 20:
                return None

            bullish_patterns = []
            bearish_patterns = []

            # ── Bullish Engulfing ─────────────────────────────────────────────
            c1_o, c1_c = float(open_.iloc[-2]), float(close.iloc[-2])
            c0_o, c0_c = float(open_.iloc[-1]), float(close.iloc[-1])
            if (c1_c < c1_o                         # prior bar bearish
                    and c0_c > c0_o                 # current bar bullish
                    and c0_o < c1_c                 # opens below prior close
                    and c0_c > c1_o):               # closes above prior open
                bullish_patterns.append("Bullish Engulfing")

            # ── Bearish Engulfing ─────────────────────────────────────────────
            if (c1_c > c1_o
                    and c0_c < c0_o
                    and c0_o > c1_c
                    and c0_c < c1_o):
                bearish_patterns.append("Bearish Engulfing")

            # ── Hammer (last bar) ─────────────────────────────────────────────
            body    = abs(c0_c - c0_o)
            candle  = float(high.iloc[-1]) - float(low.iloc[-1])
            lower_s = min(c0_o, c0_c) - float(low.iloc[-1])
            upper_s = float(high.iloc[-1]) - max(c0_o, c0_c)
            if candle > 0 and lower_s >= 2 * body and upper_s <= 0.3 * body:
                bullish_patterns.append("Hammer/Pin Bar")

            # ── Shooting Star (last bar) ──────────────────────────────────────
            if candle > 0 and upper_s >= 2 * body and lower_s <= 0.3 * body:
                bearish_patterns.append("Shooting Star")

            # ── Morning Star (3-bar) ──────────────────────────────────────────
            if len(close) >= 3:
                b2_o, b2_c = float(open_.iloc[-3]), float(close.iloc[-3])
                b1_o, b1_c = float(open_.iloc[-2]), float(close.iloc[-2])
                b0_o, b0_c = float(open_.iloc[-1]), float(close.iloc[-1])
                if (b2_c < b2_o                        # day 1 bearish
                        and abs(b1_c - b1_o) < abs(b2_c - b2_o) * 0.5  # day 2 small body
                        and b0_c > b0_o                # day 3 bullish
                        and b0_c > (b2_o + b2_c) / 2): # closes above midpoint of day 1
                    bullish_patterns.append("Morning Star")

            # ── Evening Star (3-bar) ──────────────────────────────────────────
            if len(close) >= 3:
                b2_o, b2_c = float(open_.iloc[-3]), float(close.iloc[-3])
                b1_o, b1_c = float(open_.iloc[-2]), float(close.iloc[-2])
                b0_o, b0_c = float(open_.iloc[-1]), float(close.iloc[-1])
                if (b2_c > b2_o
                        and abs(b1_c - b1_o) < abs(b2_c - b2_o) * 0.5
                        and b0_c < b0_o
                        and b0_c < (b2_o + b2_c) / 2):
                    bearish_patterns.append("Evening Star")

            # ── Bull Flag ─────────────────────────────────────────────────────
            # Pole: >5% gain in 5 days; Flag: tight consolidation next 5 days
            if len(close) >= 15:
                pole_end   = float(close.iloc[-10])
                pole_start = float(close.iloc[-15])
                flag_slice = close.iloc[-10:]
                pole_gain  = (pole_end / pole_start - 1) * 100
                flag_range = float((flag_slice.max() - flag_slice.min()) / pole_end * 100)
                if pole_gain > 5.0 and flag_range < 4.0 and float(close.iloc[-1]) > float(close.iloc[-6]):
                    bullish_patterns.append(f"Bull Flag (pole +{pole_gain:.1f}%)")

            # ── Higher Highs / Higher Lows (uptrend structure) ────────────────
            if len(high) >= 20:
                pivot_highs = [float(high.iloc[i]) for i in range(-20, 0, 5)]
                pivot_lows  = [float(low.iloc[i])  for i in range(-20, 0, 5)]
                if all(pivot_highs[i] < pivot_highs[i+1] for i in range(len(pivot_highs)-1)):
                    bullish_patterns.append("Higher Highs structure")
                if all(pivot_lows[i] < pivot_lows[i+1] for i in range(len(pivot_lows)-1)):
                    bullish_patterns.append("Higher Lows structure")
                if all(pivot_highs[i] > pivot_highs[i+1] for i in range(len(pivot_highs)-1)):
                    bearish_patterns.append("Lower Highs structure")
                if all(pivot_lows[i] > pivot_lows[i+1] for i in range(len(pivot_lows)-1)):
                    bearish_patterns.append("Lower Lows structure")

            if not bullish_patterns and not bearish_patterns:
                return None

            if len(bullish_patterns) > len(bearish_patterns):
                n = len(bullish_patterns)
                boost = round(min(0.05 + n * 0.03, 0.14), 3)
                return ScanResult(
                    scanner="pattern_detector", symbol=symbol,
                    direction="long", signal_boost=boost,
                    reason=f"Patterns: {', '.join(bullish_patterns)}",
                    metadata={"bullish": bullish_patterns, "bearish": bearish_patterns},
                )
            elif len(bearish_patterns) > len(bullish_patterns):
                n = len(bearish_patterns)
                boost = round(max(-(0.05 + n * 0.03), -0.14), 3)
                return ScanResult(
                    scanner="pattern_detector", symbol=symbol,
                    direction="short", signal_boost=boost,
                    reason=f"Patterns: {', '.join(bearish_patterns)}",
                    metadata={"bullish": bullish_patterns, "bearish": bearish_patterns},
                )

            return None

        except Exception as exc:
            logger.debug("PricePatternDetector scan failed for %s: %s", symbol, exc)
            return None
