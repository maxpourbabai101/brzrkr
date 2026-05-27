"""Shared pytest fixtures for the trading_enhancer test suite."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path so `import src.*` works under pytest.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """300‑bar synthetic OHLCV frame with a tradable trend + noise."""
    rng = np.random.default_rng(seed=42)
    n = 300
    base = 100 + np.cumsum(rng.normal(0, 1.0, size=n))
    high = base + rng.uniform(0.1, 1.0, size=n)
    low = base - rng.uniform(0.1, 1.0, size=n)
    open_ = base + rng.normal(0, 0.2, size=n)
    close = base + rng.normal(0, 0.2, size=n)
    volume = rng.integers(10_000, 200_000, size=n)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="1D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def sample_options_chain() -> pd.DataFrame:
    """Small synthetic options chain for IV‑surface tests."""
    expiries = [
        datetime.now(timezone.utc) + timedelta(days=d) for d in (7, 30, 90)
    ]
    strikes = [90, 95, 100, 105, 110]
    rows = []
    for e in expiries:
        for k in strikes:
            rows.append({
                "strike": k,
                "type": "call",
                "expiry": e,
                "bid": 1.0,
                "ask": 1.2,
                "iv": 0.15 + 0.001 * abs(100 - k) + 0.0005 * (e - datetime.now(timezone.utc)).days,
                "delta": 0.5,
                "gamma": 0.02,
                "theta": -0.05,
                "vega": 0.1,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def mock_model_output() -> dict:
    return {
        "direction": "long",
        "expected_return_pct": 0.012,
        "iv_change_pct": 0.004,
        "confidence": 0.82,
    }


@pytest.fixture
def mock_risk_params() -> dict:
    return {
        "account_equity": 100_000.0,
        "entry_price": 450.25,
        "atr": 1.75,
        "vix": 18.0,
        "realized_vol": 0.01,
        "current_time": datetime(2026, 5, 21, 15, 0, tzinfo=timezone.utc),
        "existing_positions": [],
        "correlation_matrix": {},
    }
