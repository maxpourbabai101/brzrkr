"""
TechnicalConfluence — 10 weighted technical indicators voting on direction.

Indicators (with weights):
  1. EMA Stack 20/50/200      weight=2  (trend alignment)
  2. MACD crossover/position  weight=1.5
  3. RSI zone                 weight=1
  4. Stochastic K/D cross     weight=1
  5. ADX + DI±                weight=2  (trend strength)
  6. Williams %R              weight=1
  7. CCI                      weight=1
  8. VWAP deviation           weight=1
  9. 5-day price momentum     weight=0.5
  10. Bollinger Band position  weight=1

7+/10 weighted agreement → boost=0.15
5-6/10 → boost=0.08
<5/10 → no signal
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.signals.advanced_scanners import ScanResult

logger = logging.getLogger(__name__)


class TechnicalConfluence:

    def scan(self, symbol: str, prices: pd.DataFrame) -> Optional[ScanResult]:
        try:
            close  = prices["close"].astype(float)
            high   = prices["high"].astype(float)
            low    = prices["low"].astype(float)
            volume = prices["volume"].astype(float)

            if len(close) < 200:
                return None

            price = float(close.iloc[-1])
            votes: list[tuple[str, float, str]] = []   # (direction, weight, reason)

            # ── 1. EMA Stack ──────────────────────────────────────────────────
            ema20  = float(close.ewm(span=20,  adjust=False).mean().iloc[-1])
            ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
            ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
            if price > ema20 > ema50 > ema200:
                votes.append(("long",  2.0, "EMA 20>50>200 bull stack"))
            elif price < ema20 < ema50 < ema200:
                votes.append(("short", 2.0, "EMA 20<50<200 bear stack"))
            elif price > ema50 > ema200:
                votes.append(("long",  1.0, "price above EMA50/200"))
            elif price < ema50 < ema200:
                votes.append(("short", 1.0, "price below EMA50/200"))

            # ── 2. MACD ────────────────────────────────────────────────────────
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line   = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            m_now  = float(macd_line.iloc[-1])
            m_prev = float(macd_line.iloc[-2])
            s_now  = float(signal_line.iloc[-1])
            s_prev = float(signal_line.iloc[-2])
            if m_now > s_now and m_prev <= s_prev:
                votes.append(("long",  1.5, f"MACD bullish crossover"))
            elif m_now < s_now and m_prev >= s_prev:
                votes.append(("short", 1.5, f"MACD bearish crossover"))
            elif m_now > s_now:
                votes.append(("long",  0.5, "MACD above signal"))
            else:
                votes.append(("short", 0.5, "MACD below signal"))

            # ── 3. RSI ─────────────────────────────────────────────────────────
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi   = float((100 - 100 / (1 + rs)).iloc[-1])
            if 40 <= rsi <= 65:
                votes.append(("long",  1.0, f"RSI={rsi:.0f} healthy bull zone"))
            elif 30 <= rsi < 40:
                votes.append(("long",  0.8, f"RSI={rsi:.0f} recovering from oversold"))
            elif rsi > 70:
                votes.append(("short", 1.0, f"RSI={rsi:.0f} overbought"))
            elif rsi < 30:
                votes.append(("long",  0.5, f"RSI={rsi:.0f} oversold bounce"))

            # ── 4. Stochastic ──────────────────────────────────────────────────
            low14  = low.rolling(14).min()
            high14 = high.rolling(14).max()
            stoch_range = (high14 - low14).replace(0, np.nan)
            k = ((close - low14) / stoch_range * 100)
            d = k.rolling(3).mean()
            k_val = float(k.iloc[-1])
            d_val = float(d.iloc[-1])
            k_prev = float(k.iloc[-2])
            d_prev = float(d.iloc[-2])
            if k_val > d_val and k_prev <= d_prev and k_val < 80:
                votes.append(("long",  1.0, f"Stoch bullish cross K={k_val:.0f}"))
            elif k_val < d_val and k_prev >= d_prev and k_val > 20:
                votes.append(("short", 1.0, f"Stoch bearish cross K={k_val:.0f}"))
            elif k_val > d_val and k_val < 80:
                votes.append(("long",  0.5, f"Stoch K above D"))
            elif k_val < d_val and k_val > 20:
                votes.append(("short", 0.5, f"Stoch K below D"))

            # ── 5. ADX + DI ────────────────────────────────────────────────────
            tr_df = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low  - close.shift(1)).abs(),
            ], axis=1)
            tr     = tr_df.max(axis=1)
            atr14  = tr.rolling(14).mean().replace(0, np.nan)
            dm_p   = high.diff().clip(lower=0)
            dm_m   = (-low.diff()).clip(lower=0)
            di_p   = dm_p.rolling(14).mean() / atr14 * 100
            di_m   = dm_m.rolling(14).mean() / atr14 * 100
            dx     = ((di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan) * 100)
            adx_s  = dx.rolling(14).mean()
            adx_val = float(adx_s.iloc[-1])
            dip_val = float(di_p.iloc[-1])
            dim_val = float(di_m.iloc[-1])
            if adx_val > 25:
                if dip_val > dim_val:
                    votes.append(("long",  2.0, f"ADX={adx_val:.0f} strong uptrend (DI+>{dip_val:.0f})"))
                else:
                    votes.append(("short", 2.0, f"ADX={adx_val:.0f} strong downtrend (DI->{dim_val:.0f})"))
            elif adx_val > 20:
                if dip_val > dim_val:
                    votes.append(("long",  1.0, f"ADX={adx_val:.0f} emerging uptrend"))
                else:
                    votes.append(("short", 1.0, f"ADX={adx_val:.0f} emerging downtrend"))

            # ── 6. Williams %R ─────────────────────────────────────────────────
            hh = high.rolling(14).max()
            ll = low.rolling(14).min()
            wr = ((hh - close) / (hh - ll).replace(0, np.nan) * -100)
            wr_val = float(wr.iloc[-1])
            if -20 >= wr_val >= -50:
                votes.append(("long",  1.0, f"Williams%R={wr_val:.0f} bullish zone"))
            elif -50 > wr_val >= -80:
                votes.append(("short", 1.0, f"Williams%R={wr_val:.0f} bearish zone"))
            elif wr_val > -20:
                votes.append(("long",  0.5, f"Williams%R={wr_val:.0f} strong bull"))

            # ── 7. CCI ─────────────────────────────────────────────────────────
            tp   = (high + low + close) / 3
            sma_tp = tp.rolling(20).mean()
            mad    = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            cci_s  = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
            cci_val = float(cci_s.iloc[-1])
            if 0 < cci_val < 100:
                votes.append(("long",  1.0, f"CCI={cci_val:.0f} bullish zone"))
            elif -100 < cci_val < 0:
                votes.append(("short", 1.0, f"CCI={cci_val:.0f} bearish zone"))
            elif cci_val >= 100:
                votes.append(("long",  0.5, f"CCI={cci_val:.0f} strong momentum"))
            elif cci_val <= -100:
                votes.append(("short", 0.5, f"CCI={cci_val:.0f} strong neg momentum"))

            # ── 8. VWAP deviation ──────────────────────────────────────────────
            tp_v  = (high + low + close) / 3
            cum_tpv = (tp_v * volume).cumsum()
            cum_v   = volume.cumsum().replace(0, np.nan)
            vwap_val = float((cum_tpv / cum_v).iloc[-1])
            vwap_dev = (price - vwap_val) / vwap_val * 100
            if 0 < vwap_dev < 2.5:
                votes.append(("long",  1.0, f"Price +{vwap_dev:.1f}% above VWAP"))
            elif -2.5 < vwap_dev < 0:
                votes.append(("short", 1.0, f"Price {vwap_dev:.1f}% below VWAP"))

            # ── 9. 5-day momentum ──────────────────────────────────────────────
            if len(close) >= 6:
                mom5 = float((close.iloc[-1] / close.iloc[-6] - 1) * 100)
                if mom5 > 1.5:
                    votes.append(("long",  0.5, f"+{mom5:.1f}% 5d momentum"))
                elif mom5 < -1.5:
                    votes.append(("short", 0.5, f"{mom5:.1f}% 5d momentum"))

            # ── 10. Bollinger Band position ────────────────────────────────────
            sma20  = close.rolling(20).mean()
            std20  = close.rolling(20).std()
            bb_upper = float((sma20 + 2 * std20).iloc[-1])
            bb_lower = float((sma20 - 2 * std20).iloc[-1])
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                bb_pos = (price - bb_lower) / bb_range
                if 0.45 <= bb_pos <= 0.80:
                    votes.append(("long",  1.0, f"BB position {bb_pos:.0%} bullish zone"))
                elif 0.20 <= bb_pos < 0.45:
                    votes.append(("short", 1.0, f"BB position {bb_pos:.0%} bearish zone"))

            # ── Tally ──────────────────────────────────────────────────────────
            long_w  = sum(w for d, w, _ in votes if d == "long")
            short_w = sum(w for d, w, _ in votes if d == "short")
            total_w = long_w + short_w

            if total_w < 3.0:
                return None

            dominant = "long" if long_w >= short_w else "short"
            dom_w    = max(long_w, short_w)
            agreement = dom_w / total_w

            if agreement < 0.55:
                return None

            # boost scale: agreement 0.55→0.08, 0.75→0.15, 1.0→0.20
            boost = round(float(np.clip(0.04 + agreement * 0.20, 0.0, 0.20)), 3)
            if dominant == "short":
                boost = -boost

            reasons = [r for d, w, r in votes if d == dominant][:3]
            return ScanResult(
                scanner="technical_confluence", symbol=symbol,
                direction=dominant, signal_boost=boost,
                reason=f"TechConf {dom_w:.1f}/{total_w:.1f}w ({agreement:.0%}): {'; '.join(reasons)}",
                metadata={
                    "long_weight":  round(long_w, 1),
                    "short_weight": round(short_w, 1),
                    "agreement":    round(agreement, 3),
                    "rsi":          round(rsi, 1),
                    "adx":          round(adx_val, 1),
                    "vwap_dev":     round(vwap_dev, 2),
                },
            )

        except Exception as exc:
            logger.debug("TechnicalConfluence scan failed for %s: %s", symbol, exc)
            return None
