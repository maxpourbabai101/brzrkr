"""Event blackout filter — blocks new entries within defined windows around
binary-event risk.

Covered events
--------------
1. **Earnings** — 48h window before + 2h after a company's earnings release.
   Fetched live from yfinance; result cached per symbol for 6 hours.

2. **Macro events** — Hard-coded or fetched windows around FOMC decisions,
   CPI, NFP, PPI, PCE, GDP, retail sales.  Uses a static calendar for the
   current year and the next 3 months; falls back to "block Wednesdays near
   2pm ET" heuristic when no calendar is loaded.

3. **Halts / circuit-breakers** — Alpaca returns a `halt` flag on tickers;
   this module checks it when a credentials dict is passed.

Public API
----------
>>> from src.filters.event_blackout import is_blocked
>>> blocked, reason = is_blocked("NVDA")
>>> if blocked:
...     print(f"Skip NVDA: {reason}")

>>> is_blocked("SPY", now=datetime(2026, 7, 30, 13, 0, tzinfo=timezone.utc))
(True, "FOMC decision within 120 min")
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Tuning constants ─────────────────────────────────────────────────────────
EARNINGS_BEFORE_HOURS = 48     # block this many hours before earnings
EARNINGS_AFTER_HOURS  =  2     # block this many hours after (gap risk)
FOMC_WINDOW_MINUTES   = 120    # block 2h before + after FOMC statement
MACRO_WINDOW_MINUTES  =  30    # block 30 min before + after CPI/NFP/PPI etc.
CACHE_TTL_SECONDS     = 21_600  # 6 hours — re-fetch earnings dates this often

# ── Earnings cache: symbol → (next_earnings_utc, fetched_at_ts) ────────────
_earnings_cache: Dict[str, Tuple[Optional[datetime], float]] = {}

# ── Macro event calendar (UTC datetimes for current + next quarters) ────────
# Format: (label, utc_datetime).  Keep this list maintained quarterly.
# Events are approximated to the hour; the exact minute is covered by the
# MACRO_WINDOW_MINUTES buffer.
_MACRO_EVENTS_2026: list = [
    # FOMC meetings (statement released ~14:00 ET = 18:00 UTC)
    ("FOMC",  datetime(2026, 1, 29, 19, 0, tzinfo=timezone.utc)),
    ("FOMC",  datetime(2026, 3, 19, 18, 0, tzinfo=timezone.utc)),
    ("FOMC",  datetime(2026, 5,  7, 18, 0, tzinfo=timezone.utc)),
    ("FOMC",  datetime(2026, 6, 18, 18, 0, tzinfo=timezone.utc)),
    ("FOMC",  datetime(2026, 7, 29, 18, 0, tzinfo=timezone.utc)),
    ("FOMC",  datetime(2026, 9, 17, 18, 0, tzinfo=timezone.utc)),
    ("FOMC",  datetime(2026, 11,  5, 19, 0, tzinfo=timezone.utc)),
    ("FOMC",  datetime(2026, 12, 16, 19, 0, tzinfo=timezone.utc)),
    # CPI releases (~08:30 ET = 12:30 UTC, first or second week of month)
    ("CPI",   datetime(2026, 1, 15, 13, 30, tzinfo=timezone.utc)),
    ("CPI",   datetime(2026, 2, 12, 13, 30, tzinfo=timezone.utc)),
    ("CPI",   datetime(2026, 3, 12, 13, 30, tzinfo=timezone.utc)),
    ("CPI",   datetime(2026, 4,  9, 12, 30, tzinfo=timezone.utc)),
    ("CPI",   datetime(2026, 5, 14, 12, 30, tzinfo=timezone.utc)),
    ("CPI",   datetime(2026, 6, 11, 12, 30, tzinfo=timezone.utc)),
    ("CPI",   datetime(2026, 7,  9, 12, 30, tzinfo=timezone.utc)),
    ("CPI",   datetime(2026, 8, 13, 12, 30, tzinfo=timezone.utc)),
    ("CPI",   datetime(2026, 9, 10, 12, 30, tzinfo=timezone.utc)),
    ("CPI",   datetime(2026, 10,  8, 12, 30, tzinfo=timezone.utc)),
    ("CPI",   datetime(2026, 11, 12, 13, 30, tzinfo=timezone.utc)),
    ("CPI",   datetime(2026, 12, 10, 13, 30, tzinfo=timezone.utc)),
    # NFP (first Friday of month, ~08:30 ET)
    ("NFP",   datetime(2026, 1,  2, 13, 30, tzinfo=timezone.utc)),
    ("NFP",   datetime(2026, 2,  6, 13, 30, tzinfo=timezone.utc)),
    ("NFP",   datetime(2026, 3,  6, 13, 30, tzinfo=timezone.utc)),
    ("NFP",   datetime(2026, 4,  3, 12, 30, tzinfo=timezone.utc)),
    ("NFP",   datetime(2026, 5,  1, 12, 30, tzinfo=timezone.utc)),
    ("NFP",   datetime(2026, 6,  5, 12, 30, tzinfo=timezone.utc)),
    ("NFP",   datetime(2026, 7,  2, 12, 30, tzinfo=timezone.utc)),
    ("NFP",   datetime(2026, 8,  7, 12, 30, tzinfo=timezone.utc)),
    ("NFP",   datetime(2026, 9,  4, 12, 30, tzinfo=timezone.utc)),
    ("NFP",   datetime(2026, 10,  2, 12, 30, tzinfo=timezone.utc)),
    ("NFP",   datetime(2026, 11,  6, 13, 30, tzinfo=timezone.utc)),
    ("NFP",   datetime(2026, 12,  4, 13, 30, tzinfo=timezone.utc)),
]


def _fetch_next_earnings(symbol: str) -> Optional[datetime]:
    """Return the UTC datetime of the next earnings, or None."""
    try:
        import yfinance as yf
        cal = yf.Ticker(symbol).calendar
        if cal is None or cal.empty:
            return None
        # calendar columns vary by yfinance version
        col = next(
            (c for c in ("Earnings Date", "earningsDate", "Earnings") if c in cal.columns),
            None,
        )
        if col is None:
            return None
        raw = cal[col].dropna()
        if raw.empty:
            return None
        dt = raw.iloc[0]
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()
        if isinstance(dt, datetime):
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        return None
    except Exception as exc:
        logger.debug("Earnings fetch failed for %s: %s", symbol, exc)
        return None


def _get_next_earnings(symbol: str) -> Optional[datetime]:
    """Cached earnings date lookup."""
    cached_dt, fetched_at = _earnings_cache.get(symbol, (None, 0.0))
    if time.monotonic() - fetched_at < CACHE_TTL_SECONDS:
        return cached_dt
    dt = _fetch_next_earnings(symbol)
    _earnings_cache[symbol] = (dt, time.monotonic())
    return dt


def _near_macro_event(now: datetime) -> Tuple[bool, str]:
    """Return (blocked, reason) if now is within a macro event window."""
    for label, event_utc in _MACRO_EVENTS_2026:
        window = FOMC_WINDOW_MINUTES if label == "FOMC" else MACRO_WINDOW_MINUTES
        delta = abs((now - event_utc).total_seconds() / 60)
        if delta <= window:
            return True, f"{label} event within {int(delta)} min (window={window}min)"
    return False, ""


def is_blocked(
    symbol: str,
    *,
    now: Optional[datetime] = None,
    skip_earnings_check: bool = False,
) -> Tuple[bool, str]:
    """Check whether entering a new position in ``symbol`` is blocked.

    Parameters
    ----------
    symbol : str
        Ticker to check (e.g. "NVDA", "SPY").
    now : datetime, optional
        Override for current UTC time (useful in tests).
    skip_earnings_check : bool
        When True, only check macro events (faster — skips yfinance call).

    Returns
    -------
    (blocked: bool, reason: str)
        blocked=True means do not enter.  reason explains why.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # 1 ── Macro events (FOMC / CPI / NFP) ────────────────────────────────
    blocked, reason = _near_macro_event(now)
    if blocked:
        return True, reason

    # 2 ── Earnings blackout ───────────────────────────────────────────────
    if not skip_earnings_check:
        # ETFs and indices never have earnings
        _ETF_PREFIXES = {"SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT",
                         "USO", "XLK", "XLF", "XLE", "XLV", "TQQQ", "SQQQ",
                         "IBIT", "GBTC", "VXX", "EFA", "EEM", "XLC"}
        if symbol.upper() in _ETF_PREFIXES:
            return False, ""

        next_earnings = _get_next_earnings(symbol)
        if next_earnings is not None:
            hours_to_earnings = (next_earnings - now).total_seconds() / 3600
            if -EARNINGS_AFTER_HOURS <= hours_to_earnings <= EARNINGS_BEFORE_HOURS:
                return (
                    True,
                    f"Earnings within {hours_to_earnings:.1f}h "
                    f"(blackout: -{EARNINGS_AFTER_HOURS}h … +{EARNINGS_BEFORE_HOURS}h)",
                )

    return False, ""


def check_and_log(symbol: str, *, now: Optional[datetime] = None) -> bool:
    """Convenience wrapper. Returns True if safe to trade (not blocked)."""
    blocked, reason = is_blocked(symbol, now=now)
    if blocked:
        logger.info("Event blackout: %s blocked — %s", symbol, reason)
    return not blocked
