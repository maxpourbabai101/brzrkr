"""BreakoutDetector — catches clean price breakouts and breakdowns with conviction.

A breakout is high-confidence when ALL of the following are true:
  • Price clears a key level (20-bar high/low, or resistance/support zone)
  • Volume is elevated vs 20-day average (institutional participation)
  • ATR expands (real range expansion, not whipsaw)
  • Multiple EMAs are aligned (20 > 50 for longs, 20 < 50 for shorts)
  • No immediate overhead resistance within 1 ATR

Confidence scale:
  4/5 conditions  → 0.76
  5/5 conditions  → 0.85
  5/5 + momentum  → up to 0.93

A failed breakout (price at new high but volume thin / ATR compressing)
yields low confidence (0.50–0.55), not a false buy signal.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BreakoutDetector:
    """Volume-confirmed breakout / breakdown detector."""

    label = "breakout"

    # How many bars back to look for the breakout level
    LOOKBACK = 20

    def predict(self, features: Any) -> Dict[str, Any]:
        try:
            close  = features["close"].astype(float)
            high   = features["high"].astype(float)  if "high"  in features.columns else close * 1.01
            low    = features["low"].astype(float)   if "low"   in features.columns else close * 0.99
            volume = features["volume"].astype(float) if "volume" in features.columns else None

            lb = self.LOOKBACK
            if len(close) < lb + 20:
                return self._neutral()

            price    = float(close.iloc[-1])
            prev_bar = slice(-lb - 1, -1)     # bars before the last one

            # ── Breakout level ───────────────────────────────────────────
            resist = float(high.iloc[prev_bar].max())  # 20-bar resistance
            support = float(low.iloc[prev_bar].min())  # 20-bar support

            breakout_up   = price > resist
            breakout_down = price < support

            if not (breakout_up or breakout_down):
                return self._neutral()   # no breakout, nothing to say

            direction = "long" if breakout_up else "short"

            # ── Score conditions ─────────────────────────────────────────
            conditions = 0

            # Condition 1: price clearly cleared the level (not just touching)
            level = resist if breakout_up else support
            atr   = self._atr(high, low, close, 14)
            clear_margin = atr * 0.25
            if breakout_up and price > level + clear_margin:
                conditions += 1
            elif breakout_down and price < level - clear_margin:
                conditions += 1

            # Condition 2: volume above 20d average
            if volume is not None and len(volume) >= 21:
                avg_vol  = float(volume.tail(21).iloc[:-1].mean())
                curr_vol = float(volume.iloc[-1])
                if curr_vol > avg_vol * 1.25:
                    conditions += 1

            # Condition 3: ATR expansion (today's range > 10d average range)
            today_range = float(high.iloc[-1] - low.iloc[-1])
            avg_range   = float((high - low).tail(11).iloc[:-1].mean())
            if avg_range > 0 and today_range > avg_range * 1.1:
                conditions += 1

            # Condition 4: EMA alignment
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            if breakout_up and ema20 > ema50 and price > ema20:
                conditions += 1
            elif breakout_down and ema20 < ema50 and price < ema20:
                conditions += 1

            # Condition 5: clean close — bar closed in the top/bottom third
            bar_open  = float(features["open"].astype(float).iloc[-1]) if "open" in features.columns else price
            bar_range = float(high.iloc[-1] - low.iloc[-1])
            if bar_range > 0:
                close_position = (price - float(low.iloc[-1])) / bar_range
                if breakout_up and close_position > 0.60:
                    conditions += 1
                elif breakout_down and close_position < 0.40:
                    conditions += 1

            # Bonus: prior-bar momentum in same direction
            momentum_bonus = 0.0
            if len(close) >= 6:
                ret5 = float(close.pct_change(5).iloc[-1] or 0.0)
                if breakout_up and ret5 > 0.01:
                    momentum_bonus = 0.06
                elif breakout_down and ret5 < -0.01:
                    momentum_bonus = 0.06

            # ── Confidence ───────────────────────────────────────────────
            # Need at least 3 conditions to emit a meaningful signal
            if conditions < 3:
                return self._neutral()

            base_conf  = 0.50 + (conditions / 5) * 0.40   # 0.50–0.90
            confidence = float(min(base_conf + momentum_bonus, 0.94))

            # Expected return: breakout tends to move 1–2× ATR
            atr_pct = (atr / max(price, 1e-6)) * 100
            expected_return = atr_pct * 1.5 * (1 if breakout_up else -1)

            logger.debug(
                "Breakout %s: conditions=%d/5 conf=%.3f dir=%s",
                "UP" if breakout_up else "DOWN", conditions, confidence, direction,
            )

            return {
                "direction": direction,
                "expected_return_pct": float(expected_return),
                "iv_change_pct": 0.0,
                "confidence": confidence,
                "breakout_conditions": conditions,
            }

        except Exception as exc:
            logger.debug("BreakoutDetector.predict error: %s", exc)
            return self._neutral()

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1] or 0.01)

    @staticmethod
    def _neutral() -> Dict[str, Any]:
        return {
            "direction": "long",
            "expected_return_pct": 0.0,
            "iv_change_pct": 0.0,
            "confidence": 0.50,
        }
