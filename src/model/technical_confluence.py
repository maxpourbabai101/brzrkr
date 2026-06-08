"""TechnicalConfluenceAgent — high-confidence signal from multi-factor agreement.

Core idea: a single indicator is noise; when 6+ INDEPENDENT indicators all
point the same direction, that is genuine signal worth acting on.

Factors checked (each votes -1 / 0 / +1):
  1. RSI extremes          (< 35 = long, > 65 = short)
  2. MACD vs signal line   (cross above = long, below = short)
  3. Price vs EMA-20       (above = long, below = short)
  4. Price vs EMA-50       (above = long, below = short)
  5. Bollinger position    (near lower band = long, near upper = short)
  6. Volume surge          (up-volume > avg = long, down-volume = short)
  7. ATR momentum          (positive close-over-close acceleration = long)
  8. Stochastic %K         (< 25 = oversold long, > 75 = overbought short)

Confidence formula:
  score = Σ(votes) / 8                         # range [-1, 1]
  confidence = 0.50 + |score| * 0.48           # range [0.50, 0.98]

  6/8 agree → 0.50 + 0.75*0.48 = 0.86
  7/8 agree → 0.50 + 0.875*0.48 = 0.92
  All 8     → 0.50 + 1.0*0.48 = 0.98

This is the agent that gets the ensemble above 0.75.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalConfluenceAgent:
    """8-factor technical confluence model. High confidence only when indicators agree."""

    label = "confluence"

    def predict(self, features: Any) -> Dict[str, Any]:
        try:
            close  = features["close"].astype(float)
            high   = features["high"].astype(float)  if "high"  in features.columns else close * 1.01
            low    = features["low"].astype(float)   if "low"   in features.columns else close * 0.99
            volume = features["volume"].astype(float) if "volume" in features.columns else None

            if len(close) < 52:
                return self._neutral()

            price = float(close.iloc[-1])
            votes: list[int] = []

            # ── Factor 1: RSI ────────────────────────────────────────────
            rsi = self._rsi(close, 14)
            if rsi < 35:
                votes.append(1)
            elif rsi > 65:
                votes.append(-1)
            else:
                votes.append(0)

            # ── Factor 2: MACD ───────────────────────────────────────────
            ema12 = float(close.ewm(span=12, adjust=False).mean().iloc[-1])
            ema26 = float(close.ewm(span=26, adjust=False).mean().iloc[-1])
            macd_line = ema12 - ema26
            signal_line = float(
                pd.Series(
                    close.ewm(span=12, adjust=False).mean()
                    - close.ewm(span=26, adjust=False).mean()
                ).ewm(span=9, adjust=False).mean().iloc[-1]
            )
            if macd_line > signal_line and macd_line > 0:
                votes.append(1)
            elif macd_line < signal_line and macd_line < 0:
                votes.append(-1)
            elif macd_line > signal_line:
                votes.append(1)
            elif macd_line < signal_line:
                votes.append(-1)
            else:
                votes.append(0)

            # ── Factor 3: Price vs EMA-20 ────────────────────────────────
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            votes.append(1 if price > ema20 else -1)

            # ── Factor 4: Price vs EMA-50 ────────────────────────────────
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            votes.append(1 if price > ema50 else -1)

            # ── Factor 5: Bollinger Band position ────────────────────────
            sma20 = float(close.rolling(20).mean().iloc[-1])
            std20 = float(close.rolling(20).std().iloc[-1])
            if std20 > 0:
                upper = sma20 + 2 * std20
                lower = sma20 - 2 * std20
                bb_pos = (price - lower) / (upper - lower)   # 0=lower, 1=upper
                if bb_pos < 0.25:
                    votes.append(1)    # near lower band — mean reversion long
                elif bb_pos > 0.75:
                    votes.append(-1)   # near upper band — mean reversion short
                else:
                    votes.append(0)
            else:
                votes.append(0)

            # ── Factor 6: Volume direction ───────────────────────────────
            if volume is not None and len(volume) >= 21:
                avg_vol  = float(volume.tail(21).iloc[:-1].mean())
                curr_vol = float(volume.iloc[-1])
                last_ret = float(close.pct_change().iloc[-1] or 0.0)
                if curr_vol > avg_vol * 1.3 and last_ret > 0:
                    votes.append(1)
                elif curr_vol > avg_vol * 1.3 and last_ret < 0:
                    votes.append(-1)
                elif curr_vol < avg_vol * 0.6:
                    votes.append(0)   # low-volume move = less conviction
                else:
                    votes.append(1 if last_ret > 0 else -1)
            else:
                votes.append(0)

            # ── Factor 7: ATR momentum (close-over-close acceleration) ───
            rets = close.pct_change().dropna()
            atr  = self._atr(high, low, close, 14)
            if atr > 0 and len(rets) >= 5:
                mom_score = float(rets.tail(5).mean()) / (atr / float(close.iloc[-1]))
                if mom_score > 0.15:
                    votes.append(1)
                elif mom_score < -0.15:
                    votes.append(-1)
                else:
                    votes.append(0)
            else:
                votes.append(0)

            # ── Factor 8: Stochastic %K ──────────────────────────────────
            if len(high) >= 14 and len(low) >= 14:
                h14 = float(high.tail(14).max())
                l14 = float(low.tail(14).min())
                denom = h14 - l14
                stoch_k = ((price - l14) / denom * 100) if denom > 0 else 50.0
                if stoch_k < 25:
                    votes.append(1)
                elif stoch_k > 75:
                    votes.append(-1)
                else:
                    votes.append(0)
            else:
                votes.append(0)

            # ── Tally ────────────────────────────────────────────────────
            total = sum(votes)          # range [-8, 8]
            n     = len(votes)          # 8
            score = total / n           # range [-1, 1]

            direction  = "long" if score >= 0 else "short"
            confidence = float(0.50 + abs(score) * 0.48)

            # Expected return: ATR-based projection
            atr_pct = (atr / max(price, 1e-6)) * 100
            expected_return = float(score * atr_pct * 2.5)

            logger.debug(
                "Confluence: votes=%s total=%d score=%.3f conf=%.3f dir=%s",
                votes, total, score, confidence, direction,
            )

            return {
                "direction": direction,
                "expected_return_pct": expected_return,
                "iv_change_pct": 0.0,
                "confidence": confidence,
                "confluence_score": float(score),
                "votes": votes,
            }

        except Exception as exc:
            logger.debug("TechnicalConfluenceAgent.predict error: %s", exc)
            return self._neutral()

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> float:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - 100 / (1 + rs)
        return float(rsi.iloc[-1] if not rsi.empty else 50.0)

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
