"""SweepDetector — flag unusual activity on small / penny stocks.

A "sweep" in this codebase means: a single bar where price + volume
both move dramatically beyond a symbol's recent baseline. These are
the bars where institutional intent or coordinated retail activity
is most likely to be hiding.

Detection criteria (configurable):

* **Volume ratio**: today's volume ≥ N × the trailing 20-day average
  (default N=3.0)
* **Price ratio**: |today's return| ≥ M × the trailing 20-day realised
  volatility (default M=2.0), OR absolute return ≥ price_threshold
  (default 5%)
* **Liquidity floor**: trailing 20-day average dollar volume must be
  above a floor (default $100k) — avoids dead names
* **Optional price ceiling**: only consider stocks at or below $X
  (default no ceiling; set to e.g. 10 for true "penny stock" scope)

Output is a list of :class:`SweepAlert` records, written to
``data/sweeps.jsonl`` for the dashboard to consume.

Uses :class:`WebDataScraper` for OHLCV → no extra API keys needed.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/sweeps.jsonl")

# Small-cap / volatile names commonly involved in sweeps + meme runs.
# Curated mix of: low-float biotech, EV/AI thematic micro-caps, classic
# meme tickers, and SPAC remnants. Edit freely.
DEFAULT_SMALL_CAP_WATCHLIST: List[str] = [
    # Meme + retail-favorite
    "GME", "AMC", "BBBY", "KOSS", "RDDT", "MARA", "RIOT", "CLSK",
    "BBAI", "SOUN", "AI", "PLTR",
    # Low-float biotech (volatile)
    "INMB", "OCGN", "VTGN", "ATER", "RDHL", "CYTO",
    # Small-cap EVs / cleantech
    "WKHS", "GOEV", "NKLA", "QS", "ARRY", "LCID",
    # SPAC remnants / busted IPOs
    "OPEN", "CVNA", "WISH", "BARK",
    # True penny territory
    "TLRY", "SNDL", "ACB", "HEXO",
    # Crypto-adjacent small caps
    "COIN", "HOOD",
]


@dataclass
class SweepAlert:
    symbol: str
    detected_at: str
    bar_date: str
    direction: str               # "up" | "down"
    price_change_pct: float
    volume_ratio: float          # today_vol / avg_20d_vol
    realized_vol_sigmas: float   # |return| / 20d realized vol
    close: float
    volume: int
    dollar_volume_avg_20d: float
    score: float                 # composite score 0..∞ (higher = more notable)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SweepDetector:
    watchlist: Iterable[str] = field(
        default_factory=lambda: list(DEFAULT_SMALL_CAP_WATCHLIST))
    output_path: Path = DEFAULT_OUTPUT
    volume_multiplier: float = 3.0
    sigma_multiplier: float = 2.0
    price_threshold: float = 0.05         # 5% absolute move
    min_dollar_volume_avg: float = 100_000
    max_price: Optional[float] = None     # set to e.g. 10 for penny-only
    lookback_days: int = 90               # how much history to pull
    avg_window: int = 20                  # bars for the baseline averages

    # Injection for tests: callable(symbol) -> DataFrame
    fetch_fn: Optional[callable] = None

    # ------------------------------------------------------------------
    def _fetch(self, symbol: str) -> pd.DataFrame:
        if self.fetch_fn is not None:
            return self.fetch_fn(symbol)
        from src.data_scraper import WebDataScraper
        agent = WebDataScraper()
        rng = "3mo" if self.lookback_days <= 90 else "6mo"
        return agent.scrape_ohlcv(symbol, range_=rng)

    # ------------------------------------------------------------------
    def scan_symbol(self, symbol: str) -> Optional[SweepAlert]:
        """Inspect a single symbol's most recent bar. Returns an alert
        or None."""
        try:
            df = self._fetch(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.debug("fetch failed for %s: %s", symbol, exc)
            return None
        if df is None or len(df) < self.avg_window + 1:
            return None

        df = df.copy()
        _ohlcv = frozenset(("open", "high", "low", "close", "volume"))
        df.columns = [next((p.lower() for p in c if p.lower() in _ohlcv), c[0].lower())
                      if isinstance(c, tuple) else c.lower() for c in df.columns]
        if "close" not in df.columns or "volume" not in df.columns:
            return None
        df["ret"] = df["close"].pct_change()
        df["dol_vol"] = df["close"] * df["volume"]

        # Baseline = average of the trailing window EXCLUDING today.
        recent = df.iloc[-self.avg_window - 1: -1]
        if recent.empty:
            return None
        avg_vol = float(recent["volume"].mean())
        avg_dol_vol = float(recent["dol_vol"].mean())
        rv = float(recent["ret"].std())     # realized vol of returns

        if avg_dol_vol < self.min_dollar_volume_avg:
            return None
        if self.max_price is not None and float(df["close"].iloc[-1]) > self.max_price:
            return None

        last = df.iloc[-1]
        close = float(last["close"])
        vol = float(last["volume"])
        ret = float(last["ret"]) if not pd.isna(last["ret"]) else 0.0

        vol_ratio = vol / avg_vol if avg_vol > 0 else 0.0
        sigma_mult = abs(ret) / rv if rv > 0 else 0.0

        vol_hit = vol_ratio >= self.volume_multiplier
        sigma_hit = sigma_mult >= self.sigma_multiplier
        price_hit = abs(ret) >= self.price_threshold

        # Need volume confirmation AND either sigma or absolute price spike
        if not vol_hit or not (sigma_hit or price_hit):
            return None

        score = vol_ratio * max(sigma_mult, abs(ret) * 20)
        bar_date = str(df.index[-1])[:10] if hasattr(df.index[-1], "year") else "?"

        return SweepAlert(
            symbol=symbol,
            detected_at=datetime.now(timezone.utc).isoformat(),
            bar_date=bar_date,
            direction="up" if ret > 0 else "down",
            price_change_pct=round(ret * 100, 3),
            volume_ratio=round(vol_ratio, 2),
            realized_vol_sigmas=round(sigma_mult, 2),
            close=round(close, 4),
            volume=int(vol),
            dollar_volume_avg_20d=round(avg_dol_vol, 0),
            score=round(score, 2),
            note=("volume + price both hit" if vol_hit and price_hit
                   else "volume + sigma hit"),
        )

    # ------------------------------------------------------------------
    def scan(self, symbols: Optional[Iterable[str]] = None) -> List[SweepAlert]:
        """Scan every symbol in the watchlist (or override). Returns
        alerts sorted by score descending."""
        syms = list(symbols) if symbols is not None else list(self.watchlist)
        out: List[SweepAlert] = []
        for s in syms:
            alert = self.scan_symbol(s)
            if alert is not None:
                out.append(alert)
        out.sort(key=lambda a: a.score, reverse=True)
        return out

    # ------------------------------------------------------------------
    def write(self, alerts: List[SweepAlert],
              *, append: bool = True) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with self.output_path.open(mode) as f:
            for a in alerts:
                f.write(json.dumps(a.to_dict()) + "\n")
        return self.output_path

    @classmethod
    def load_recent(cls, path: Path = DEFAULT_OUTPUT,
                     limit: int = 50) -> List[Dict[str, Any]]:
        """Read the most recent N alerts back from disk."""
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        out: List[Dict[str, Any]] = []
        for ln in lines[-limit * 2:]:    # over-read in case of malformed lines
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        # Sort by detected_at descending, take latest `limit`
        out.sort(key=lambda d: d.get("detected_at", ""), reverse=True)
        return out[:limit]
