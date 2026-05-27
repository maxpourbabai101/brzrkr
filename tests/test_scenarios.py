"""Tests for src.backtest.scenarios + scenario_runner."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.scenarios import (
    MarketScenario, all_scenarios, by_category, by_name,
    benchmark_summary, categories,
)
from src.backtest.scenario_runner import ScenarioResult, ScenarioRunner


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
def test_library_has_diverse_categories():
    cats = categories()
    expected = {"crash", "rally", "vol_spike", "regime_change",
                 "grinding", "crisis", "squeeze", "post_event"}
    assert expected.issubset(set(cats))


def test_all_scenarios_have_valid_dates():
    for s in all_scenarios():
        assert s.start < s.end
        assert s.days > 0
        assert s.symbols
        assert s.category
        assert s.description


def test_by_name_round_trip():
    s = by_name("covid_crash_2020")
    assert s is not None
    assert s.category == "crash"
    assert "covid" in s.description.lower()


def test_by_name_unknown_returns_none():
    assert by_name("not_a_real_scenario_xyz") is None


def test_by_category_returns_only_that_category():
    rallies = by_category("rally")
    assert all(s.category == "rally" for s in rallies)
    assert len(rallies) >= 2


def test_benchmark_summary_runs():
    text = benchmark_summary()
    assert "Scenario library" in text
    assert "CRASH" in text


# ---------------------------------------------------------------------------
# ScenarioRunner — stubbed to avoid network
# ---------------------------------------------------------------------------
def _synthetic_prices(n=400, trend=0.001, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, 0.012, n)
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    op = close * (1 + rng.normal(0, 0.001, n))
    vol = rng.integers(50_000, 500_000, n)
    idx = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="1D")
    df = pd.DataFrame({"open": op, "high": high, "low": low,
                       "close": close, "volume": vol}, index=idx)
    return df


def _stub_ensemble_factory():
    class _Stub:
        def predict(self, features):
            return {
                "direction": "long",
                "expected_return_pct": 0.01,
                "iv_change_pct": 0.0,
                "confidence": 0.55,   # below default 0.75 → no trades
            }
    return _Stub()


def _stub_engineer_factory():
    class _E:
        def build_features(self, bundle):
            df = bundle.get("prices")
            if df is None or df.empty:
                return pd.DataFrame()
            out = df.copy()
            out.attrs["context"] = {}
            return out
    return _E()


@pytest.fixture
def runner(tmp_path):
    return ScenarioRunner(
        ensemble_factory=_stub_ensemble_factory,
        feature_engineer_factory=_stub_engineer_factory,
        output_dir=tmp_path / "scenarios",
        seq_len=64,
    )


def test_run_one_succeeds_on_synthetic(runner):
    sc = MarketScenario(
        name="test_scenario", category="rally",
        symbols=["FAKE"],
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 30, tzinfo=timezone.utc),
        description="synthetic",
    )
    with patch("src.data_scraper.WebDataScraper.scrape_ohlcv",
                return_value=_synthetic_prices()):
        res = runner.run_one(sc, "FAKE")
    assert isinstance(res, ScenarioResult)
    assert not res.failed
    assert res.bars > 0


def test_run_one_handles_empty_data(runner):
    sc = MarketScenario(
        name="empty_test", category="rally", symbols=["NONE"],
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 30, tzinfo=timezone.utc),
        description="no data",
    )
    with patch("src.data_scraper.WebDataScraper.scrape_ohlcv",
                return_value=pd.DataFrame()):
        res = runner.run_one(sc, "NONE")
    assert res.failed
    assert "no data" in res.error


def test_run_one_handles_scraper_exception(runner):
    sc = MarketScenario(
        name="broken", category="crash", symbols=["BAD"],
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 2, 1, tzinfo=timezone.utc),
        description="should fail",
    )
    with patch("src.data_scraper.WebDataScraper.scrape_ohlcv",
                side_effect=RuntimeError("network down")):
        res = runner.run_one(sc, "BAD")
    assert res.failed
    assert "network down" in res.error


def test_run_all_persists_progress(runner, tmp_path):
    sc1 = MarketScenario(
        name="s1", category="rally", symbols=["A"],
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 30, tzinfo=timezone.utc),
        description="x",
    )
    sc2 = MarketScenario(
        name="s2", category="rally", symbols=["B"],
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 6, 30, tzinfo=timezone.utc),
        description="y",
    )
    with patch("src.data_scraper.WebDataScraper.scrape_ohlcv",
                return_value=_synthetic_prices()):
        results = runner.run_all([sc1, sc2])
    assert len(results) == 2
    partial = runner.output_dir / "_results_so_far.csv"
    assert partial.exists()


def test_write_report_creates_files(runner, tmp_path):
    results = [
        ScenarioResult(
            scenario="t1", category="rally", symbol="A",
            start="2024-01-01", end="2024-06-30", bars=100,
            trades=3, win_rate=0.66, avg_pnl=10.0, sharpe=1.2,
            max_drawdown_pct=-0.05, final_equity=101_500.0,
            benchmark_return_pct=5.0, relative_vs_benchmark_pct=-3.5,
            difficulty="easy",
        ),
    ]
    md = runner.write_report(results)
    assert md.exists()
    text = md.read_text()
    assert "Scenario Battery Report" in text
    assert "t1" in text


def test_write_report_with_failures(runner):
    results = [
        ScenarioResult(
            scenario="t1", category="crash", symbol="A",
            start="2020-03-01", end="2020-04-01", bars=0,
            trades=0, win_rate=0.0, avg_pnl=0.0, sharpe=0.0,
            max_drawdown_pct=0.0, final_equity=100_000.0,
            benchmark_return_pct=0.0, relative_vs_benchmark_pct=0.0,
            difficulty="brutal", failed=True, error="boom",
        ),
    ]
    md = runner.write_report(results)
    text = md.read_text()
    assert "Failures" in text
    assert "boom" in text
