"""Market Data Collector — wide universe feature extraction & vectorization.

Scans 200+ symbols across every asset class, computes 32-dimensional
feature vectors capturing momentum, volatility, technicals, and relative
strength.  Strips out noise; saves only signal-bearing snapshots in
compressed numpy format.

Architecture
────────────
  MarketCollector.run_full()
      → batch-downloads OHLCV from yfinance (10 symbols at a time)
      → compute_features() for each symbol → 32-dim numpy vector
      → importance_filter() drops low-signal noise
      → VectorStore.save_snapshot() writes compressed .npz + metadata JSON

Auto-collector
──────────────
  BackgroundCollector(interval_minutes=15).start()
      Runs in a daemon thread; safe to call from GUI startup.

Usage
─────
  from src.data.market_collector import BackgroundCollector
  BackgroundCollector().start()         # fire-and-forget on app launch
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── yfinance column normaliser ────────────────────────────────────────
# yfinance ≥ 0.2 returns MultiIndex columns whose structure differs between
# single-ticker (field, ticker) and batch-with-group_by (ticker, field).
# This helper extracts the OHLCV field name regardless of which element it is.
_OHLCV_FIELDS = frozenset(("open", "high", "low", "close", "volume", "adj close"))


def _flatten_columns(cols) -> list:
    result = []
    for c in cols:
        if isinstance(c, str):
            result.append(c.lower())
        else:
            # Find whichever element is an OHLCV field name.
            field = next((p.lower() for p in c if p.lower() in _OHLCV_FIELDS), None)
            result.append(field if field else c[0].lower())
    return result


# ── Paths ─────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
VECTOR_DIR    = ROOT / "data" / "market_vectors"
LATEST_JSON   = VECTOR_DIR / "latest.json"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

# ── Wide universe  ────────────────────────────────────────────────────
# ~200 symbols; every meaningful publicly-traded asset class
WIDE_UNIVERSE: Dict[str, str] = {
    # ── Index ETFs ──────────────────────────────────────────────────
    "SPY":  "INDEX_ETF",     "QQQ":  "INDEX_ETF",
    "IWM":  "INDEX_ETF",     "DIA":  "INDEX_ETF",
    "VTI":  "INDEX_ETF",     "MDY":  "INDEX_ETF",
    "IJH":  "INDEX_ETF",     "VEA":  "INDEX_ETF",
    "EEM":  "INDEX_ETF",
    # ── Sector ETFs ─────────────────────────────────────────────────
    "XLK":  "SECTOR_ETF",    "XLF":  "SECTOR_ETF",
    "XLE":  "SECTOR_ETF",    "XLV":  "SECTOR_ETF",
    "XLC":  "SECTOR_ETF",    "XLI":  "SECTOR_ETF",
    "XLB":  "SECTOR_ETF",    "XLU":  "SECTOR_ETF",
    "XLRE": "SECTOR_ETF",    "XLP":  "SECTOR_ETF",
    "XLY":  "SECTOR_ETF",
    # ── Bond / Rates ─────────────────────────────────────────────────
    "TLT":  "BOND_ETF",      "IEF":  "BOND_ETF",
    "SHY":  "BOND_ETF",      "HYG":  "BOND_ETF",
    "LQD":  "BOND_ETF",      "TBT":  "BOND_ETF",
    "TMF":  "BOND_ETF",
    # ── Commodity / Macro ────────────────────────────────────────────
    "GLD":  "COMMODITY_ETF", "SLV":  "COMMODITY_ETF",
    "USO":  "COMMODITY_ETF", "UNG":  "COMMODITY_ETF",
    "PDBC": "COMMODITY_ETF", "DJP":  "COMMODITY_ETF",
    "CORN": "COMMODITY_ETF", "WEAT": "COMMODITY_ETF",
    # ── Volatility gauges ────────────────────────────────────────────
    "VXX":  "VOLATILITY",    "UVXY": "VOLATILITY",
    "SVXY": "VOLATILITY",
    # ── Leveraged ETFs ───────────────────────────────────────────────
    "TQQQ": "LEVERAGED_ETF", "SQQQ": "LEVERAGED_ETF",
    "UPRO": "LEVERAGED_ETF", "SPXS": "LEVERAGED_ETF",
    "TNA":  "LEVERAGED_ETF", "TZA":  "LEVERAGED_ETF",
    "LABU": "LEVERAGED_ETF", "LABD": "LEVERAGED_ETF",
    "FNGU": "LEVERAGED_ETF", "FNGD": "LEVERAGED_ETF",
    "TECL": "LEVERAGED_ETF", "TECS": "LEVERAGED_ETF",
    "UDOW": "LEVERAGED_ETF", "SDOW": "LEVERAGED_ETF",
    "NAIL": "LEVERAGED_ETF",
    # ── Mega-cap tech ─────────────────────────────────────────────────
    "AAPL": "STOCK",         "MSFT": "STOCK",
    "NVDA": "STOCK",         "GOOGL":"STOCK",
    "AMZN": "STOCK",         "META": "STOCK",
    "TSLA": "STOCK",         "AVGO": "STOCK",
    "ORCL": "STOCK",         "CRM":  "STOCK",
    "ADBE": "STOCK",         "QCOM": "STOCK",
    "TXN":  "STOCK",         "AMD":  "STOCK",
    "MU":   "STOCK",         "INTC": "STOCK",
    "NOW":  "STOCK",         "PANW": "STOCK",
    "ARM":  "STOCK",         "ASML": "STOCK",
    "TSM":  "STOCK",
    # ── High-beta / high-volatility stocks ───────────────────────────
    "COIN": "STOCK",         "MSTR": "STOCK",
    "HOOD": "STOCK",         "RKLB": "STOCK",
    "IONQ": "STOCK",         "PLTR": "STOCK",
    "SOFI": "STOCK",         "RIVN": "STOCK",
    "LCID": "STOCK",         "SMCI": "STOCK",
    "ROKU": "STOCK",         "SNAP": "STOCK",
    "UBER": "STOCK",         "LYFT": "STOCK",
    "SHOP": "STOCK",         "NET":  "STOCK",
    "DDOG": "STOCK",         "SNOW": "STOCK",
    "RBLX": "STOCK",         "U":    "STOCK",
    # ── Financials ───────────────────────────────────────────────────
    "JPM":  "STOCK",         "BAC":  "STOCK",
    "GS":   "STOCK",         "MS":   "STOCK",
    "V":    "STOCK",         "MA":   "STOCK",
    # ── Healthcare / Biotech ─────────────────────────────────────────
    "UNH":  "STOCK",         "JNJ":  "STOCK",
    "MRNA": "STOCK",         "BNTX": "STOCK",
    "PFE":  "STOCK",         "ABBV": "STOCK",
    # ── Consumer / Retail ────────────────────────────────────────────
    "WMT":  "STOCK",         "HD":   "STOCK",
    "COST": "STOCK",         "TGT":  "STOCK",
    "NKE":  "STOCK",         "SBUX": "STOCK",
    # ── Energy ───────────────────────────────────────────────────────
    "XOM":  "STOCK",         "CVX":  "STOCK",
    "OXY":  "STOCK",         "SLB":  "STOCK",
    # ── Crypto ETFs ──────────────────────────────────────────────────
    "IBIT": "CRYPTO_ETF",    "GBTC": "CRYPTO_ETF",
    "ETHA": "CRYPTO_ETF",    "FBTC": "CRYPTO_ETF",
    "ARKB": "CRYPTO_ETF",
    # ── Small/mid-cap momentum ────────────────────────────────────────
    "PLUG": "SMALL_CAP",     "HIMS": "SMALL_CAP",
    "PRPL": "SMALL_CAP",     "OPEN": "SMALL_CAP",
    "SOUN": "SMALL_CAP",     "BBAI": "SMALL_CAP",
    "QBTS": "SMALL_CAP",     "RGTI": "SMALL_CAP",
    "ARQQ": "SMALL_CAP",     "DJT":  "SMALL_CAP",
    "RCAT": "SMALL_CAP",     "ACHR": "SMALL_CAP",
}

# ── Feature names (32-dim) ────────────────────────────────────────────
FEATURE_NAMES = [
    "rsi_14",       "rsi_2",        "macd_hist_n",  "macd_cross",
    "bb_pct",       "bb_width",     "mom_1d",       "mom_5d",
    "mom_20d",      "mom_60d",      "atr_pct",      "vol_ratio_5",
    "vol_ratio_20", "pct_52h",      "pct_52l",      "sma20_spread",
    "sma50_spread",  "sma200_spread","close_loc",    "hv_20",
    "hv_5",         "gap_pct",      "market_rs_5",  "trend_encoded",
    "vol_surge",    "rsi_extreme",  "above_all_sma","below_all_sma",
    "new_high_20",  "new_low_20",   "composite",    "signal_encoded",
]
FEATURE_DIM = len(FEATURE_NAMES)


# ── Result dataclass ─────────────────────────────────────────────────

@dataclass
class SymbolSnapshot:
    symbol:        str
    asset_type:    str
    timestamp:     str
    # Price data
    price:         float
    change_1d_pct: float
    change_5d_pct: float
    volume:        float
    avg_volume_20: float
    # Technical
    rsi:           float
    macd_signal:   str     # "cross_up" | "cross_down" | "flat"
    bb_position:   float   # 0=at lower band, 1=at upper band
    atr_pct:       float
    # Composite
    score:         float   # -100 to +100
    signal:        str     # "bullish" | "bearish" | "neutral"
    confidence:    float   # 0-1
    note:          str
    # Extras
    vol_ratio:     float
    momentum_5d:   float
    momentum_20d:  float
    trend:         str
    # Feature vector (stored separately in npz)
    features:      List[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("features", None)   # don't embed float array in JSON
        return d


# ══════════════════════════════════════════════════════════════════════
# Feature computation
# ══════════════════════════════════════════════════════════════════════

def _safe_float(x, fallback=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else fallback
    except Exception:
        return fallback


def compute_features(symbol: str, df: pd.DataFrame,
                     spy_close: Optional[pd.Series] = None
                     ) -> Optional[Tuple[SymbolSnapshot, np.ndarray]]:
    """Compute a 32-dim feature vector for one symbol from daily OHLCV.

    Returns (SymbolSnapshot, np.ndarray) or None if insufficient data.
    """
    if df is None or len(df) < 22:
        return None

    try:
        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float)
        open_  = df["open"].astype(float)
    except KeyError:
        return None

    n = len(close)
    price = _safe_float(close.iloc[-1])
    if price <= 0:
        return None

    # ── RSI ──────────────────────────────────────────────────────────
    def _rsi(s: pd.Series, period: int) -> float:
        d = s.diff().dropna()
        g = d.clip(lower=0).rolling(period).mean()
        l = (-d.clip(upper=0)).rolling(period).mean()
        rs = g / l.replace(0, 1e-9)
        r  = (100 - 100 / (1 + rs)).dropna()
        return _safe_float(r.iloc[-1], 50.0) if len(r) else 50.0

    rsi14 = _rsi(close, 14)
    rsi2  = _rsi(close, 2)

    # ── MACD ─────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig9  = macd.ewm(span=9,  adjust=False).mean()
    hist  = macd - sig9
    hist_n = _safe_float(hist.iloc[-1]) / max(price * 0.001, 1e-9)
    hist_n = max(-5.0, min(5.0, hist_n))  # clamp
    prev_d = _safe_float(macd.iloc[-2] - sig9.iloc[-2]) if n > 2 else 0
    curr_d = _safe_float(macd.iloc[-1] - sig9.iloc[-1])
    if prev_d <= 0 < curr_d:
        macd_label = "cross_up";   macd_cross = 1.0
    elif prev_d >= 0 > curr_d:
        macd_label = "cross_down"; macd_cross = -1.0
    else:
        macd_label = "flat";       macd_cross = 0.0

    # ── Bollinger Bands ───────────────────────────────────────────────
    bb_mean  = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    bbu = _safe_float(bb_upper.iloc[-1])
    bbl = _safe_float(bb_lower.iloc[-1])
    bbm = _safe_float(bb_mean.iloc[-1])
    bb_range = max(bbu - bbl, 1e-9)
    bb_pct   = (price - bbl) / bb_range   # 0=at lower, 1=at upper
    bb_width = bb_range / max(bbm, 1e-9)  # normalized bandwidth

    # ── Momentum ─────────────────────────────────────────────────────
    def _mom(days: int) -> float:
        if n <= days:
            return 0.0
        return _safe_float((close.iloc[-1] / close.iloc[-(days+1)] - 1) * 100)

    mom1d  = _mom(1)
    mom5d  = _mom(5)
    mom20d = _mom(20)
    mom60d = _mom(60) if n > 60 else _mom(min(n-1, 40))

    # ── ATR ───────────────────────────────────────────────────────────
    prev_c = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low  - prev_c).abs(),
    ], axis=1).max(axis=1)
    atr14  = tr.rolling(14).mean()
    atr_pct = _safe_float(atr14.iloc[-1]) / price * 100

    # ── Volume ────────────────────────────────────────────────────────
    avg5  = _safe_float(volume.iloc[-6:-1].mean()) if n > 6  else 1.0
    avg20 = _safe_float(volume.iloc[-21:-1].mean()) if n > 21 else 1.0
    vol_now = _safe_float(volume.iloc[-1])
    vol_r5  = vol_now / max(avg5,  1.0)
    vol_r20 = vol_now / max(avg20, 1.0)

    # ── 52-week ───────────────────────────────────────────────────────
    days252 = min(n, 252)
    hi52 = _safe_float(high.iloc[-days252:].max())
    lo52 = _safe_float(low.iloc[-days252:].min())
    pct_52h = (price / max(hi52, 1e-9) - 1) * 100   # negative = below 52w high
    pct_52l = (price / max(lo52, 1e-9) - 1) * 100   # positive = above 52w low

    # ── SMAs ──────────────────────────────────────────────────────────
    def _sma_spread(window: int) -> float:
        if n < window:
            return 0.0
        return (price / _safe_float(close.iloc[-window:].mean(), price) - 1) * 100

    sma20_s  = _sma_spread(20)
    sma50_s  = _sma_spread(50)
    sma200_s = _sma_spread(200)

    # ── Historical vol ────────────────────────────────────────────────
    log_ret = np.log(close / close.shift(1)).dropna()
    hv20 = _safe_float(log_ret.iloc[-20:].std() * np.sqrt(252) * 100) if n > 20 else 20.0
    hv5  = _safe_float(log_ret.iloc[-5:].std()  * np.sqrt(252) * 100) if n > 5  else hv20

    # ── Intraday close location ───────────────────────────────────────
    day_range = max(_safe_float(high.iloc[-1] - low.iloc[-1]), 1e-9)
    close_loc = (_safe_float(close.iloc[-1]) - _safe_float(low.iloc[-1])) / day_range

    # ── Overnight gap ─────────────────────────────────────────────────
    gap_pct = _safe_float((open_.iloc[-1] / close.iloc[-2] - 1) * 100) if n > 1 else 0.0

    # ── Market relative strength ──────────────────────────────────────
    market_rs = mom5d  # override if SPY series provided
    if spy_close is not None and len(spy_close) > 5:
        spy_mom5 = _safe_float((spy_close.iloc[-1] / spy_close.iloc[-6] - 1) * 100)
        market_rs = mom5d - spy_mom5

    # ── Trend encoding ────────────────────────────────────────────────
    if n >= 50:
        sma20v = _safe_float(close.iloc[-20:].mean())
        sma50v = _safe_float(close.iloc[-50:].mean())
        if price > sma20v > sma50v:
            trend_str = "up";   trend_enc = 1.0
        elif price < sma20v < sma50v:
            trend_str = "down"; trend_enc = -1.0
        else:
            trend_str = "flat"; trend_enc = 0.0
    else:
        trend_str = "flat"; trend_enc = 0.0

    # ── Binary features ───────────────────────────────────────────────
    vol_surge_f   = 1.0 if vol_r20 > 2.0 else (0.5 if vol_r20 > 1.5 else 0.0)
    rsi_extreme_f = 1.0 if rsi14 < 30 else (-1.0 if rsi14 > 70 else 0.0)
    above_all     = 1.0 if (sma20_s > 0 and sma50_s > 0 and sma200_s > 0) else 0.0
    below_all     = 1.0 if (sma20_s < 0 and sma50_s < 0 and sma200_s < 0) else 0.0
    new_high_20   = 1.0 if price >= _safe_float(high.iloc[-20:].max()) else 0.0
    new_low_20    = 1.0 if price <= _safe_float(low.iloc[-20:].min()) else 0.0

    # ── Composite score (-100 to +100) ───────────────────────────────
    score = 0.0
    if rsi14 < 30:   score += 25
    elif rsi14 < 40: score += 12
    elif rsi14 > 70: score -= 25
    elif rsi14 > 60: score -= 12
    if macd_cross == 1.0:  score += 20
    elif macd_cross == -1.0: score -= 20
    score += max(min(mom5d * 4, 25), -25)
    if vol_r20 > 2.0: score += 10
    elif vol_r20 > 1.5: score += 5
    if trend_enc == 1.0:  score += 10
    elif trend_enc == -1.0: score -= 10
    if bb_pct < 0.1:  score += 8   # near lower band = mean reversion setup
    elif bb_pct > 0.9: score -= 8
    score = max(-100.0, min(100.0, score))
    composite_n = score / 100.0   # normalized -1 to +1

    if score >= 15:
        signal = "bullish"; signal_enc = 1.0
    elif score <= -15:
        signal = "bearish"; signal_enc = -1.0
    else:
        signal = "neutral"; signal_enc = 0.0

    # ── Confidence (how extreme the setup is) ────────────────────────
    confidence = abs(score) / 100.0

    # ── Note ──────────────────────────────────────────────────────────
    parts = []
    if rsi14 < 30:           parts.append(f"oversold RSI {rsi14:.0f}")
    elif rsi14 > 70:         parts.append(f"overbought RSI {rsi14:.0f}")
    if macd_label == "cross_up":   parts.append("MACD ↑ cross")
    elif macd_label == "cross_down": parts.append("MACD ↓ cross")
    if vol_r20 > 2.0:        parts.append(f"vol surge {vol_r20:.1f}×")
    if abs(mom5d) > 5:       parts.append(f"{mom5d:+.1f}% 5d")
    if bb_pct > 0.9:         parts.append("at upper BB")
    elif bb_pct < 0.1:       parts.append("at lower BB")
    if new_high_20:          parts.append("20d high ⬆")
    elif new_low_20:         parts.append("20d low ⬇")
    note = "  ·  ".join(parts) if parts else "no standout signal"

    # ── Change 5d ────────────────────────────────────────────────────
    chg5d = _safe_float((close.iloc[-1] / close.iloc[min(5, n-1)] - 1) * 100) if n > 5 else 0.0

    # ── Build 32-dim feature vector ───────────────────────────────────
    vec = np.array([
        rsi14, rsi2, hist_n, macd_cross,
        bb_pct, bb_width, mom1d, mom5d,
        mom20d, mom60d, atr_pct, vol_r5,
        vol_r20, pct_52h, pct_52l, sma20_s,
        sma50_s, sma200_s, close_loc, hv20,
        hv5, gap_pct, market_rs, trend_enc,
        vol_surge_f, rsi_extreme_f, above_all, below_all,
        new_high_20, new_low_20, composite_n, signal_enc,
    ], dtype=np.float32)

    snap = SymbolSnapshot(
        symbol        = symbol,
        asset_type    = WIDE_UNIVERSE.get(symbol, "UNKNOWN"),
        timestamp     = datetime.now(timezone.utc).isoformat(),
        price         = round(price, 2),
        change_1d_pct = round(mom1d, 3),
        change_5d_pct = round(chg5d, 3),
        volume        = round(vol_now, 0),
        avg_volume_20 = round(avg20, 0),
        rsi           = round(rsi14, 1),
        macd_signal   = macd_label,
        bb_position   = round(bb_pct, 3),
        atr_pct       = round(atr_pct, 3),
        score         = round(score, 1),
        signal        = signal,
        confidence    = round(confidence, 3),
        note          = note,
        vol_ratio     = round(vol_r20, 2),
        momentum_5d   = round(mom5d, 2),
        momentum_20d  = round(mom20d, 2),
        trend         = trend_str,
        features      = vec.tolist(),
    )
    return snap, vec


# ══════════════════════════════════════════════════════════════════════
# MarketCollector
# ══════════════════════════════════════════════════════════════════════

class MarketCollector:
    """Download → feature-extract → filter → save to VectorStore."""

    BATCH_SIZE = 15   # symbols per yfinance batch call

    def __init__(self, period: str = "3mo") -> None:
        self.period = period

    def run_full(self, universe: Optional[Dict[str, str]] = None,
                 progress_cb=None) -> List[SymbolSnapshot]:
        """Full scan.  Returns list of SymbolSnapshots sorted by |score|.

        progress_cb(done, total) called after each batch if provided.
        """
        import yfinance as yf

        symbols = dict(universe or WIDE_UNIVERSE)
        sym_list = list(symbols.keys())
        total    = len(sym_list)

        # ── Fetch SPY for relative-strength baseline ──────────────────
        spy_close: Optional[pd.Series] = None
        try:
            spy_raw = yf.download("SPY", period=self.period, interval="1d",
                                  progress=False, auto_adjust=True)
            if spy_raw is not None and len(spy_raw) > 5:
                spy_raw.columns = _flatten_columns(spy_raw.columns)
                spy_close = spy_raw["close"].squeeze().astype(float)
        except Exception:
            pass

        snapshots: List[SymbolSnapshot] = []
        done = 0

        # Process in batches to reduce yfinance rate-limit exposure
        for i in range(0, total, self.BATCH_SIZE):
            batch = sym_list[i:i + self.BATCH_SIZE]
            try:
                raw = yf.download(
                    " ".join(batch),
                    period=self.period,
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                    group_by="ticker",
                )
            except Exception as exc:
                logger.warning("Batch download failed (%s…): %s", batch[0], exc)
                done += len(batch)
                if progress_cb:
                    progress_cb(done, total)
                continue

            for sym in batch:
                try:
                    # Multi-ticker download has a 2-level column index
                    if len(batch) == 1:
                        df_sym = raw.copy()
                    else:
                        if sym not in raw.columns.get_level_values(0):
                            continue
                        df_sym = raw[sym].copy()

                    df_sym.columns = _flatten_columns(df_sym.columns)
                    result = compute_features(sym, df_sym, spy_close)
                    if result is None:
                        continue
                    snap, _vec = result
                    if self._passes_importance_filter(snap):
                        snapshots.append(snap)
                except Exception as exc:
                    logger.debug("Feature error %s: %s", sym, exc)

            done += len(batch)
            if progress_cb:
                progress_cb(done, total)
            time.sleep(0.1)   # gentle throttle

        # Sort by |score| descending
        snapshots.sort(key=lambda s: abs(s.score), reverse=True)

        # Save to vector store
        self._save(snapshots)
        logger.info("MarketCollector: saved %d snapshots", len(snapshots))
        return snapshots

    # ── Importance filter ────────────────────────────────────────────

    @staticmethod
    def _passes_importance_filter(s: SymbolSnapshot) -> bool:
        """Keep only signal-bearing snapshots; drop boring noise."""
        # Must have reasonable price and volume
        if s.price < 0.50 or s.avg_volume_20 < 50_000:
            return False
        # Keep if any meaningful signal present
        if abs(s.score) >= 8:
            return True
        if s.vol_ratio > 1.4:   # unusual volume
            return True
        if abs(s.rsi - 50) > 12:  # RSI divergence
            return True
        if s.macd_signal != "flat":
            return True
        if abs(s.momentum_5d) > 2.0:
            return True
        return False

    # ── Save ─────────────────────────────────────────────────────────

    @staticmethod
    def _save(snapshots: List[SymbolSnapshot]) -> None:
        from src.data.vector_store import VectorStore
        VectorStore().save_snapshot(snapshots)


# ══════════════════════════════════════════════════════════════════════
# Background auto-collector daemon
# ══════════════════════════════════════════════════════════════════════

class BackgroundCollector:
    """Runs MarketCollector on a schedule in a daemon thread.

    Usage:
        BackgroundCollector().start()

    First run is immediate (after a short delay so the UI is up).
    """

    def __init__(self, interval_minutes: float = 20.0) -> None:
        self._interval = interval_minutes * 60
        self._thread:  Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._running  = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="MarketCollector")
        self._thread.start()
        logger.info("BackgroundCollector started (interval=%.0f min)",
                    self._interval / 60)

    def stop(self) -> None:
        self._stop_evt.set()

    def _loop(self) -> None:
        # Short initial delay so the UI is fully built before first scan
        self._stop_evt.wait(timeout=8.0)
        while not self._stop_evt.is_set():
            try:
                self._running = True
                logger.info("BackgroundCollector: starting collection…")
                MarketCollector().run_full()
            except Exception as exc:
                logger.warning("BackgroundCollector error: %s", exc)
            finally:
                self._running = False
            self._stop_evt.wait(timeout=self._interval)

    @property
    def is_running(self) -> bool:
        return self._running
