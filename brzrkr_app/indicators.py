"""Technical indicators for the Market tab.

Each function takes a price DataFrame (OHLCV) and returns an
:class:`IndicatorReading` — current numeric value + bullish/bearish/
neutral interpretation + a short note. Designed for display, not for
the model (the model's features live in ``src/features/feature_engineer.py``).

Indicators implemented:
    RSI(14)             — momentum oscillator
    MACD(12,26,9)       — trend-following momentum
    Bollinger %B(20,2)  — price position within volatility envelope
    SMA cross (50/200)  — long-term trend / golden cross
    ADX(14)             — trend strength
    ATR(14)             — volatility / stop-loss sizing
    OBV                 — volume-confirmed accumulation
    Volume ratio        — today vs 20-day average
    52-week percentile  — where price sits in its yearly range
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------
@dataclass
class IndicatorReading:
    name: str
    value: float
    display: str            # human-readable value, e.g. "54.2" or "+0.18"
    state: str              # "bullish" | "bearish" | "neutral"
    note: str               # one-liner interpretation


def _state_for_value(value: float, *, bullish_above: float | None = None,
                     bearish_below: float | None = None) -> str:
    if bullish_above is not None and value > bullish_above:
        return "bullish"
    if bearish_below is not None and value < bearish_below:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# 1. RSI
# ---------------------------------------------------------------------------
def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / (avg_down + 1e-9)
    return 100 - 100 / (1 + rs)


def reading_rsi(df: pd.DataFrame) -> IndicatorReading:
    v = float(rsi(df["close"]).iloc[-1])
    if v >= 70:
        state, note = "bearish", "overbought — reversion risk"
    elif v <= 30:
        state, note = "bullish", "oversold — bounce risk"
    else:
        state, note = "neutral", "no extreme"
    return IndicatorReading("RSI (14)", v, f"{v:.1f}", state, note)


# ---------------------------------------------------------------------------
# 2. MACD
# ---------------------------------------------------------------------------
def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def reading_macd(df: pd.DataFrame) -> IndicatorReading:
    m, s, h = macd(df["close"])
    hist = float(h.iloc[-1])
    hist_prev = float(h.iloc[-2]) if len(h) > 1 else 0.0
    cross_up = hist > 0 and hist_prev <= 0
    cross_dn = hist < 0 and hist_prev >= 0
    if cross_up:
        state, note = "bullish", "bullish crossover (signal flipped positive)"
    elif cross_dn:
        state, note = "bearish", "bearish crossover (signal flipped negative)"
    elif hist > 0:
        state, note = "bullish", "above signal line (momentum positive)"
    else:
        state, note = "bearish", "below signal line (momentum negative)"
    sign = "+" if hist >= 0 else ""
    return IndicatorReading("MACD", hist, f"hist {sign}{hist:.3f}", state, note)


# ---------------------------------------------------------------------------
# 3. Bollinger Bands %B
# ---------------------------------------------------------------------------
def bollinger(prices: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def reading_bbands(df: pd.DataFrame) -> IndicatorReading:
    u, m, l = bollinger(df["close"])
    price = float(df["close"].iloc[-1])
    upper = float(u.iloc[-1]); lower = float(l.iloc[-1])
    if upper == lower:
        pct_b = 0.5
    else:
        pct_b = (price - lower) / (upper - lower)
    if pct_b > 1:
        state, note = "bearish", "breakout above upper band — extreme"
    elif pct_b > 0.8:
        state, note = "neutral", "near upper band"
    elif pct_b < 0:
        state, note = "bullish", "breakdown below lower band — extreme"
    elif pct_b < 0.2:
        state, note = "neutral", "near lower band"
    else:
        state, note = "neutral", "mid-band, no edge"
    return IndicatorReading("Bollinger %B", pct_b, f"{pct_b:.2f}",
                            state, note)


# ---------------------------------------------------------------------------
# 4. SMA cross — long-term trend
# ---------------------------------------------------------------------------
def reading_sma_cross(df: pd.DataFrame, fast: int = 50,
                       slow: int = 200) -> IndicatorReading:
    if len(df) < slow + 1:
        return IndicatorReading(f"SMA {fast}/{slow}", 0,
                                "n/a", "neutral",
                                f"need {slow}+ bars (have {len(df)})")
    sma_fast = df["close"].rolling(fast).mean()
    sma_slow = df["close"].rolling(slow).mean()
    f = float(sma_fast.iloc[-1]); s = float(sma_slow.iloc[-1])
    if pd.isna(f) or pd.isna(s):
        return IndicatorReading(f"SMA {fast}/{slow}", 0,
                                "n/a", "neutral", "indeterminate")
    spread = (f - s) / s * 100
    # Golden cross / death cross detection
    if len(sma_fast) > 2:
        prev_f = float(sma_fast.iloc[-2]); prev_s = float(sma_slow.iloc[-2])
        if f > s and prev_f <= prev_s:
            return IndicatorReading(f"SMA {fast}/{slow}", spread,
                                    f"{spread:+.2f}%", "bullish",
                                    "GOLDEN CROSS today — strong buy signal")
        if f < s and prev_f >= prev_s:
            return IndicatorReading(f"SMA {fast}/{slow}", spread,
                                    f"{spread:+.2f}%", "bearish",
                                    "DEATH CROSS today — strong sell signal")
    if f > s:
        return IndicatorReading(f"SMA {fast}/{slow}", spread,
                                f"{spread:+.2f}%", "bullish",
                                f"{fast}d above {slow}d — long-term uptrend")
    return IndicatorReading(f"SMA {fast}/{slow}", spread,
                            f"{spread:+.2f}%", "bearish",
                            f"{fast}d below {slow}d — long-term downtrend")


# ---------------------------------------------------------------------------
# 5. ADX — trend strength
# ---------------------------------------------------------------------------
def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([(high - low),
                     (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr_ + 1e-9)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr_ + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def reading_adx(df: pd.DataFrame) -> IndicatorReading:
    v = float(adx(df).iloc[-1])
    if v >= 40:
        state, note = "bullish", "very strong trend — momentum strategies favored"
    elif v >= 25:
        state, note = "bullish", "trending market — trend-following works"
    elif v < 20:
        state, note = "neutral", "weak/no trend — mean-reversion favored"
    else:
        state, note = "neutral", "borderline trend"
    return IndicatorReading("ADX (14)", v, f"{v:.1f}", state, note)


# ---------------------------------------------------------------------------
# 6. ATR — volatility
# ---------------------------------------------------------------------------
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([(high - low),
                     (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def reading_atr(df: pd.DataFrame) -> IndicatorReading:
    v = float(atr(df).iloc[-1])
    price = float(df["close"].iloc[-1])
    pct = v / price * 100 if price > 0 else 0
    if pct >= 3:
        state, note = "neutral", f"high vol ({pct:.1f}%/day) — use wider stops"
    elif pct >= 1.5:
        state, note = "neutral", f"medium vol ({pct:.1f}%/day)"
    else:
        state, note = "neutral", f"low vol ({pct:.1f}%/day) — tight stops OK"
    return IndicatorReading("ATR (14)", v, f"${v:.2f}", state, note)


# ---------------------------------------------------------------------------
# 7. OBV
# ---------------------------------------------------------------------------
def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0))
    return (sign * df["volume"]).cumsum()


def reading_obv(df: pd.DataFrame, lookback: int = 20) -> IndicatorReading:
    v = obv(df)
    if len(v) < lookback + 2:
        return IndicatorReading("OBV", 0, "n/a", "neutral",
                                "not enough history")
    slope = float(v.iloc[-1] - v.iloc[-lookback]) / lookback
    price_slope = float(df["close"].iloc[-1] - df["close"].iloc[-lookback]) \
        / lookback
    # Divergence detection
    if slope > 0 and price_slope < 0:
        state, note = "bullish", "OBV rising while price falling — BULL divergence"
    elif slope < 0 and price_slope > 0:
        state, note = "bearish", "OBV falling while price rising — BEAR divergence"
    elif slope > 0:
        state, note = "bullish", "accumulation (OBV rising)"
    else:
        state, note = "bearish", "distribution (OBV falling)"
    direction = "↑" if slope > 0 else "↓"
    return IndicatorReading("OBV", slope, direction, state, note)


# ---------------------------------------------------------------------------
# 8. Volume vs 20-day average
# ---------------------------------------------------------------------------
def reading_volume_ratio(df: pd.DataFrame, window: int = 20) -> IndicatorReading:
    if len(df) < window + 1:
        return IndicatorReading("Vol vs 20d avg", 1.0, "n/a", "neutral",
                                "not enough history")
    today = float(df["volume"].iloc[-1])
    avg = float(df["volume"].iloc[-window - 1:-1].mean())
    ratio = today / avg if avg > 0 else 1.0
    if ratio >= 2:
        state, note = "bullish", f"unusual volume ({ratio:.1f}× avg) — institutional interest"
    elif ratio >= 1.3:
        state, note = "neutral", f"above-average volume ({ratio:.2f}× avg)"
    elif ratio <= 0.6:
        state, note = "neutral", f"low volume ({ratio:.2f}× avg) — apathy"
    else:
        state, note = "neutral", f"normal volume ({ratio:.2f}× avg)"
    return IndicatorReading("Vol vs 20d avg", ratio, f"{ratio:.2f}×",
                            state, note)


# ---------------------------------------------------------------------------
# 9. 52-week percentile
# ---------------------------------------------------------------------------
def reading_52w_percentile(df: pd.DataFrame) -> IndicatorReading:
    window = min(len(df), 252)
    hi = float(df["high"].iloc[-window:].max())
    lo = float(df["low"].iloc[-window:].min())
    price = float(df["close"].iloc[-1])
    if hi == lo:
        pct = 0.5
    else:
        pct = (price - lo) / (hi - lo)
    if pct >= 0.95:
        state, note = "bullish", "at 52-week high — breakout regime"
    elif pct >= 0.7:
        state, note = "bullish", "near 52-week high — uptrend territory"
    elif pct <= 0.05:
        state, note = "bearish", "at 52-week low — downtrend"
    elif pct <= 0.3:
        state, note = "bearish", "near 52-week low"
    else:
        state, note = "neutral", "mid-range"
    return IndicatorReading("52w percentile", pct * 100, f"{pct * 100:.0f}%",
                            state, note)


# ---------------------------------------------------------------------------
# Compute everything at once
# ---------------------------------------------------------------------------
def all_indicators(df: pd.DataFrame) -> list[IndicatorReading]:
    """Returns every indicator in a fixed order suitable for grid display."""
    fns = [reading_rsi, reading_macd, reading_bbands,
           reading_sma_cross, reading_adx, reading_atr,
           reading_obv, reading_volume_ratio, reading_52w_percentile]
    out: list[IndicatorReading] = []
    for fn in fns:
        try:
            out.append(fn(df))
        except Exception as exc:  # noqa: BLE001
            out.append(IndicatorReading(fn.__name__, 0, "err", "neutral",
                                        f"calc failed: {exc}"))
    return out


# ---------------------------------------------------------------------------
# Explanations — used by the "Indicator Explanations" panel
# ---------------------------------------------------------------------------
EXPLANATIONS: dict[str, dict[str, str]] = {
    "RSI (14)": {
        "what": "Relative Strength Index — momentum oscillator 0–100.",
        "formula": "Smoothed avg of up-moves / down-moves over 14 bars.",
        "signals": "> 70 = overbought (reversion risk). < 30 = oversold (bounce risk).",
        "pitfall": "RSI can stay > 70 (or < 30) for weeks in strong trends. Use ADX to confirm.",
    },
    "MACD": {
        "what": "Moving Average Convergence Divergence — trend + momentum.",
        "formula": "12-day EMA − 26-day EMA. Signal = 9-day EMA of MACD. Histogram = MACD − Signal.",
        "signals": "Histogram crosses 0 = momentum flip. Above 0 = bullish, below 0 = bearish.",
        "pitfall": "Lags price by design. Whipsaws in choppy markets.",
    },
    "Bollinger %B": {
        "what": "Where price sits within Bollinger Bands (20-period, ±2σ).",
        "formula": "(price − lower band) / (upper − lower band).",
        "signals": ">1 = breakout above upper band. <0 = breakdown below. Persistent bands-walking signals strong trend.",
        "pitfall": "Mean-reversion at extremes ONLY works in range-bound markets — fatal in strong trends.",
    },
    "SMA 50/200": {
        "what": "50-day vs 200-day Simple Moving Average — defines long-term trend.",
        "formula": "Average of last 50 (and 200) closes.",
        "signals": "50 crossing ABOVE 200 = GOLDEN CROSS (classic buy). 50 BELOW 200 = DEATH CROSS (classic sell).",
        "pitfall": "These crosses are very lagging — they confirm a trend already underway, not predict one.",
    },
    "ADX (14)": {
        "what": "Average Directional Index — measures trend STRENGTH (not direction) on 0–100 scale.",
        "formula": "Smoothed |+DI − −DI| / (+DI + −DI) — uses high/low/close.",
        "signals": "ADX > 25 = trending (use trend-following). ADX < 20 = ranging (use mean-reversion).",
        "pitfall": "Doesn't tell you UP or DOWN — only that a trend exists. Pair with SMA cross.",
    },
    "ATR (14)": {
        "what": "Average True Range — typical daily price movement in dollars.",
        "formula": "14-day smoothed max(high−low, |high−prevClose|, |low−prevClose|).",
        "signals": "Used for STOP-LOSS sizing: e.g. stop = entry − 2×ATR for long.",
        "pitfall": "ATR is NOT directional. Higher ATR ≠ more bullish. Adjust position size, not bias.",
    },
    "OBV": {
        "what": "On-Balance Volume — cumulative volume signed by daily price change.",
        "formula": "OBV today = OBV yesterday ± today's volume (+ if close↑, − if close↓).",
        "signals": "Trending UP = accumulation. DIVERGENCE (OBV up while price down) = bullish; reverse = bearish.",
        "pitfall": "Sensitive to volume reporting differences across venues.",
    },
    "Vol vs 20d avg": {
        "what": "Today's volume divided by trailing 20-day average.",
        "formula": "volume_today / mean(volume[-20:-1]).",
        "signals": "> 2× = unusual activity (often institutional). < 0.6× = apathy. Confirms or rejects price moves.",
        "pitfall": "Single-day spikes around earnings/news are noise, not signal. Look for sustained 1.5×+ over multiple days.",
    },
    "52w percentile": {
        "what": "Where today's close sits in the 52-week high/low range, as percent.",
        "formula": "(price − 52w_low) / (52w_high − 52w_low).",
        "signals": "≥ 95% = at/near 52w high (breakout regime). ≤ 5% = at/near 52w low.",
        "pitfall": "New highs in low-vol grinds are durable; in vol spikes they often mark exhaustion.",
    },
}


# ---------------------------------------------------------------------------
# Time-spread presets — pair interval with appropriate range
# ---------------------------------------------------------------------------
# Each key is what the user sees in the dropdown; value is (range, interval)
# both compatible with Yahoo's chart endpoint.
TIME_SPREADS: dict[str, tuple[str, str]] = {
    "1 min  — last 1 day":     ("1d",  "1m"),
    "5 min  — last 5 days":    ("5d",  "5m"),
    "15 min — last 1 month":   ("1mo", "15m"),
    "30 min — last 1 month":   ("1mo", "30m"),
    "1 hour — last 3 months":  ("3mo", "1h"),
    "1 day  — last 6 months":  ("6mo", "1d"),
    "1 day  — last 1 year":    ("1y",  "1d"),
    "1 day  — last 2 years":   ("2y",  "1d"),
    "1 week — last 5 years":   ("5y",  "1wk"),
}
